from torch.autograd import Variable
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


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is None:
            self.alpha = torch.tensor([1.0, 1.0, 1.0])  # default equal
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, C, H, W)
        """
        if self.alpha.device != logits.device:
            self.alpha = self.alpha.to(logits.device)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        probas = torch.sigmoid(logits)

        pt = targets * probas + (1 - targets) * (1 - probas)
        focal_term = (1 - pt) ** self.gamma

        # Apply per-class alpha
        alpha_factor = self.alpha.view(1, -1, 1, 1)  # reshape to (1, C, 1, 1)
        loss = alpha_factor * focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ComboLoss(nn.Module):
    def __init__(self, pos_weight, alpha=0.5, beta=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()
        return self.alpha * bce + self.beta * dice_loss
