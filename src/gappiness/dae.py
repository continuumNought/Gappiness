import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import sqrt


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoded_dim),
            nn.Tanh(),
        )


    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, encoded_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )


    def forward(self, y):
        return self.decoder(y)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, encoded_dim)
        self.decoder = Decoder(encoded_dim, hidden_dim, input_dim)


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
    holdout_loader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, holdout_loader


def add_gauss_noise(mean=0, s=1):
    def add_noise(tensor):
        noise = torch.randn_like(tensor) * sqrt(s) + mean
        return tensor + noise

    return add_noise


def train_dae(
        train_loader,
        input_dim,
        hidden_dim,
        encoded_dim,
        add_noise,
        epochs=10,
        criterion_factory=None,
        test_loader=None,
):
    if criterion_factory is None:
        criterion = nn.MSELoss()

    else:
        criterion = criterion_factory()

    autoencoder = Autoencoder(input_dim, hidden_dim, encoded_dim)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(epochs):
        for batch in train_loader:
            inputs = batch[0]  # Extract the input data
            noisy_inputs = add_noise(inputs)  # Corrupt data
            targets = inputs  # The target remains the clean data

            # Forward pass
            outputs = autoencoder(noisy_inputs)
            loss = criterion(outputs, targets)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss / len(train_loader):.4f}")

    if test_loader is not None:
        with torch.no_grad():
            val_loss = 0
            for batch in test_loader:
                inputs = batch[0]
                noisy_inputs = add_noise(inputs)
                targets = inputs

                outputs = autoencoder(noisy_inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            print(f"Test Loss: {val_loss / len(test_loader):.4f}")

    return autoencoder
