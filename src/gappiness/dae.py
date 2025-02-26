import torch
import numpy as np
import torch.nn as nn
import scipy.stats as stats
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# TODO
# 1 train with noise, test without noise
# 2 try testing residues even without a "good" model
# 3 make deeper, possibly wider models
# 4 Flag "exploding" gradients
#
# If no progress for DAE just move forward to mapper


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, alpha=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return self.alpha * x + self.block(x)  # Residual connection (alpha * x + F(x))


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim, mlp_blocks=2):
        super().__init__()

        layers = [MLPBlock(input_dim, hidden_dim)]
        layers += [MLPBlock(hidden_dim, hidden_dim) for _ in range(mlp_blocks-1)]
        layers += [nn.Linear(hidden_dim, encoded_dim)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, encoded_dim, hidden_dim, output_dim, mlp_blocks=2):
        super().__init__()

        layers = [MLPBlock(encoded_dim, hidden_dim)]
        layers += [MLPBlock(hidden_dim, hidden_dim) for _ in range(mlp_blocks-1)]
        layers += [nn.Linear(hidden_dim, output_dim)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim, mlp_blocks=10):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, encoded_dim, mlp_blocks=mlp_blocks)
        self.decoder = Decoder(encoded_dim, hidden_dim, input_dim, mlp_blocks=mlp_blocks)


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


    def save(self, encoder_path, decoder_path):
        # Save the trained encoder and decoder separately
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)


def standard_normalize(data_):
    scaler = StandardScaler()
    scaler.fit(data_)
    return scaler.transform(data_)


def load_data(path_, batch_size, holdout_ratio=0.1, normalize=None):
    data = np.load(path_)
    if normalize is not None:
        data = normalize(data)
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Split data into training and holdout sets
    train_data, holdout_data = train_test_split(
        data_tensor,
        test_size=holdout_ratio,
        shuffle=True
    )

    # Create datasets
    train_dataset = TensorDataset(train_data)
    holdout_dataset = TensorDataset(holdout_data)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, holdout_dataset


def add_2d_gaussian_noise(tensor, mean=(0.0, 0.0), cov_matrix=((0.1, 0.0), (0.0, 0.1))):
    """
    Adds 2D Gaussian noise to each row of a tensor (n x 2).

    Args:
        tensor (torch.Tensor): Input tensor of shape (n, 2).
        mean (tuple): Mean of the 2D Gaussian noise.
        cov_matrix (tuple of tuples): 2x2 covariance matrix for Gaussian noise.

    Returns:
        torch.Tensor: Noisy tensor with same shape as input.
    """
    n = tensor.shape[0]  # Number of rows
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    cov_tensor = torch.tensor(cov_matrix, dtype=tensor.dtype, device=tensor.device)

    # Sample from a multivariate normal distribution (n rows of 2D noise)
    noise = torch.distributions.MultivariateNormal(mean_tensor, cov_tensor).sample((n,))

    return tensor + noise  # Add noise to each row


def train_dae(
        train_loader,
        input_dim,
        hidden_dim,
        encoded_dim,
        add_noise,
        epochs=10,
        criterion_factory=None,
        test_dataset=None,
        hidden_layers=10,
):
    if criterion_factory is None:
        criterion = nn.MSELoss()

    else:
        criterion = criterion_factory()

    autoencoder = Autoencoder(input_dim, hidden_dim, encoded_dim, mlp_blocks=hidden_layers)
    optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.
        for batch in train_loader:
            inputs = batch[0]  # Extract the input data
            noisy_inputs = add_noise(inputs)  # Corrupt data
            targets = inputs  # The target remains the clean data

            # Forward pass
            outputs = autoencoder(noisy_inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}")

    if test_dataset is not None:
        validate(autoencoder, test_dataset, add_noise)

    return autoencoder


def validate(model, test_data, add_noise, shapiro_alpha=0.05, significance_alpha=0.05):
    with torch.no_grad():
        inputs = test_data.tensors[0]
        noisy_inputs = add_noise(inputs)
        outputs = model(noisy_inputs)

        # Convert to numpy arrays for use with other libraries
        inputs = inputs.numpy()
        noisy_inputs = noisy_inputs.numpy()
        outputs = outputs.numpy()

        noise = np.sum((noisy_inputs - inputs) ** 2, axis=1)
        prediction_error = np.sum((outputs - inputs) ** 2, axis=1)
        diff_ = prediction_error - noise

        # Normality test
        is_normal = True
        _, shapiro_p = stats.shapiro(diff_)
        if shapiro_p < shapiro_alpha:
            is_normal = False

        if is_normal:
            _, sig_p = stats.ttest_rel(prediction_error, noise, alternative='greater')

        else:
            res = stats.wilcoxon(prediction_error, noise, alternative='greater')
            sig_p = res.pvalue

        if sig_p < significance_alpha:
            print("Model did not reduce noise significantly")

        else:
            print("Model did reduce noise significantly")
