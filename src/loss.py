import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with logits.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 pos_weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B] or [B,1]
        targets: [B] or [B,1] (0/1)
        """
        logits = logits.view(-1)
        targets = targets.view(-1)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        prob = torch.sigmoid(logits)
        p_t = targets * prob + (1 - targets) * (1 - prob)
        loss = self.alpha * (1 - p_t).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

