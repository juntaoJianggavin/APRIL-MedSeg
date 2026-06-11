"""Cross Entropy Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("ce")
class CELoss(nn.Module):
    """Standard Cross Entropy loss for segmentation."""
    def __init__(self, weight=None, ignore_index=-100, label_smoothing=0.0, **kwargs):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                       label_smoothing=label_smoothing)

    def forward(self, pred, target):
        """pred: B,C,H,W  target: B,H,W (long)"""
        return self.ce(pred, target)


@LOSS_REGISTRY.register("bce")
class BCELoss(nn.Module):
    """Binary cross-entropy for single-channel logits (common in LViT configs)."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred, target):
        if pred.ndim == 4 and pred.shape[1] == 1:
            return F.binary_cross_entropy_with_logits(
                pred.squeeze(1), target.float())
        return CELoss()(pred, target)
