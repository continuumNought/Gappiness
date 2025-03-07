import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
