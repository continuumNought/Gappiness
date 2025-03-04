import torch.nn as nn
import torch.autograd.functional as F


class MSELossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, outputs, targets):
        return self.mse.forward(outputs, targets)


class JacobianRegularizedLoss(nn.Module):
    def __init__(self, model, sigma_sq=1.0):
        """
        Custom loss: MSE + λ * Expected sum of squares of Jacobian components.
        Args:
            sigma_sq (float): Scaling factor for Jacobian regularization.
        """
        super().__init__()
        self.sigma_sq = sigma_sq
        self.model = model
        self.mse = nn.MSELoss()

    def forward(self, inputs, outputs, targets):
        # Compute MSE Loss
        reconstruction_loss = self.mse(outputs, targets)

        # Compute Jacobian Regularization Term using torch.autograd.functional.jacobian
        jacobian = F.jacobian(self.model, inputs, create_graph=True)  # Shape: (batch_size, output_dim, input_dim)
        jacobian_penalty = jacobian.pow(2).sum(dim=(1, 2)).mean()  # Expected sum of squares

        # Final loss: MSE + λ * Jacobian regularization
        return reconstruction_loss + self.sigma_sq * jacobian_penalty


class FastJacobianRegularizedLoss(nn.Module):
    def __init__(self, model, sigma_sqr=1.0):
        super().__init__()
        self.model = model
        self.sigma_sqr = sigma_sqr
        self.mse = nn.MSELoss()

    def forward(self, inputs, outputs, targets):
        # Compute MSE Loss
        reconstruction_loss = self.mse(outputs, targets)

        # Compute Approximate Jacobian Regularization
        jacobian_penalty = self.approximate_jacobian_penalty(inputs)

        return reconstruction_loss + self.sigma_sqr * jacobian_penalty

    def approximate_jacobian_penalty(self, inputs):
        """
        Computes a fast approximation of the Jacobian norm using weight matrices and ReLU masks.
        """
        x = inputs
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                W = layer.weight  # Get weight matrix
                x = x @ W.T  # Apply weight transformation

            elif isinstance(layer, nn.ReLU):
                mask = (x > 0).float()  # ReLU derivative is 1 for positive values, 0 otherwise
                x = x * mask  # Apply ReLU mask

            elif isinstance(layer, nn.BatchNorm1d):
                pass  # Ignore BatchNorm (gradient ~1 for normalized activations)

        # Compute squared norm of the final transformation
        total_norm = (x ** 2).sum(dim=1).mean()

        return total_norm
