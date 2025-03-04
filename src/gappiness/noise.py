import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
