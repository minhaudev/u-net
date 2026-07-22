import torch
import torch.nn as nn


def dice_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    dims = tuple(range(1, predictions.ndim))
    intersection = (predictions * targets).sum(dim=dims)
    denominator = predictions.sum(dim=dims) + targets.sum(dim=dims)

    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, probabilities.ndim))

    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)

    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * soft_dice_loss(logits, targets)
        )
