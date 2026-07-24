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



class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, probabilities.ndim))
        
        # True Positives, False Positives, False Negatives
        tp = (probabilities * targets).sum(dim=dims)
        fp = (probabilities * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probabilities) * targets).sum(dim=dims)
        
        # Tversky Index
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        
        # Focal Tversky Loss
        loss = (1.0 - tversky) ** self.gamma
        return loss.mean()


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_criterion: nn.Module, weights: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)):
        super().__init__()
        self.base_criterion = base_criterion
        self.weights = weights

    def forward(self, logits_list: list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        # logits_list is a list of tensors from different decoder stages
        # We assume targets is the ground truth mask for the highest resolution
        total_loss = 0.0
        
        for i, logits in enumerate(logits_list):
            weight = self.weights[i] if i < len(self.weights) else 0.0
            if weight == 0.0:
                continue
            
            # Downsample target to match logits shape if necessary
            target_scaled = targets
            if logits.shape[2:] != targets.shape[2:]:
                target_scaled = nn.functional.interpolate(
                    targets, size=logits.shape[2:], mode="nearest"
                )
                
            loss = self.base_criterion(logits, target_scaled)
            total_loss += weight * loss
            
        return total_loss
