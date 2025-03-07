import torch


def add_2d_gaussian_noise(tensor, mean=(0.0, 0.0), cov_matrix=((0.1, 0.0), (0.0, 0.1))):
    n = tensor.shape[0]  # Number of rows
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    cov_tensor = torch.tensor(cov_matrix, dtype=tensor.dtype, device=tensor.device)

    # Sample from a multivariate normal distribution (n rows of 2D noise)
    noise = torch.distributions.MultivariateNormal(mean_tensor, cov_tensor).sample((n,))

    return tensor + noise  # Add noise to each row


def add_gaussian_noise(tensor, mean, cov_matrix):
    n = tensor.shape[0]
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    cov_tensor = torch.tensor(cov_matrix, dtype=tensor.dtype, device=tensor.device)
    noise = torch.distributions.MultivariateNormal(mean_tensor, cov_tensor).sample((n,))

    return tensor + noise  # Add noise to each row
