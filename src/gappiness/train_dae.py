import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .dae import Autoencoder


def load_data(path_, batch_size):
    data = np.load(path_)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def dae_training_factory(
    train_loader,
    input_dim,
    hidden_dim,
    encoded_dim,
    add_noise,
    criterion=nn.MSELoss(),
    epochs=10,
):
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

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return autoencoder