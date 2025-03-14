import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def whiten(data_):
    return PCA(whiten=True).fit_transform(data_)


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


def load_shuffle(path_):
    data = np.load(path_)
    np.random.shuffle(data)

    return data


def normalize_split_batch(data, batch_size, holdout_ratio=0.01, normalize=None):
    if normalize is not None:
        data = normalize(data)

    train_data, holdout_data = train_test_split(
        torch.tensor(data, dtype=torch.float32),
        test_size=holdout_ratio,
        shuffle=False,
    )

    # Create datasets
    train_dataset = TensorDataset(train_data)
    holdout_dataset = TensorDataset(holdout_data)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, holdout_dataset
