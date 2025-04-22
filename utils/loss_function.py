import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCategoricalCrossEntropy(nn.Module):
    def __init__(self, weights):
        """
        Initializes the weighted categorical cross-entropy loss.

        Args:
            weights (torch.Tensor): A 1D tensor of shape (num_classes,) containing the weight for each class.
        """
        super(WeightedCategoricalCrossEntropy, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        """
        Computes the weighted categorical cross-entropy loss.

        Args:
            y_pred (torch.Tensor): Predicted probabilities (after softmax) with shape (batch_size, num_classes).
            y_true (torch.Tensor): One-hot encoded true labels with shape (batch_size, num_classes).

        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure the predictions are probabilities
        y_pred = torch.clamp(y_pred, 1e-7, 1.0)  # To avoid log(0)

        # Compute the unweighted loss
        unweighted_loss = -torch.sum(y_true * torch.log(y_pred), dim=1)

        # Compute the weight mask
        weight_mask = torch.sum(self.weights * y_true, dim=1)

        # Apply the weights
        weighted_loss = unweighted_loss * weight_mask

        # Return the mean loss over the batch
        return torch.mean(weighted_loss)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()
