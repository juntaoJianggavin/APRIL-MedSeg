"""BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for
Semantic Segmentation (Dai, He & Sun, ICCV 2015).

Official idea: iterate between generating region proposals inside boxes and
training the network.  The proposals serve as **pseudo-masks** for pixel-level
supervision.

This module provides a faithful, practical implementation of the BoxSup
training recipe:
    1. Generate pseudo-masks from bounding boxes (ellipse or rectangle).
    2. Supervise the network with standard CE/Dice on these pseudo-masks.
    3. Optional outside-box penalty to suppress predictions outside boxes.

Note: BoxInst (Tian et al., CVPR 2021) is a **different** method that uses
projection + pairwise colour-affinity losses.  See ``boxinst.py`` for that.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
from medseg.registry import LOSS_REGISTRY


# ---------------------------------------------------------------------------
# Pseudo-mask generation (faithful to BoxSup: region proposals inside boxes)
# ---------------------------------------------------------------------------
def generate_ellipse_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Generate an inscribed ellipse mask inside an (h, w) crop.

    The original BoxSup uses MCG region proposals; an inscribed ellipse is
    the simplest geometric approximation that avoids treating box corners
    as foreground (a key insight from the paper).

    Returns:
        (h, w) float tensor with values in [0, 1].
    """
    cy, cx = h / 2.0, w / 2.0
    ry, rx = max(h / 2.0, 1.0), max(w / 2.0, 1.0)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij',
    )
    # Ellipse equation: ((y-cy)/ry)^2 + ((x-cx)/rx)^2 <= 1
    dist = ((yy - cy + 0.5) / ry) ** 2 + ((xx - cx + 0.5) / rx) ** 2
    mask = (dist <= 1.0).float()
    return mask


def generate_box_pseudo_masks(
    predictions: torch.Tensor,
    boxes: list,
    box_classes: Optional[list] = None,
    num_classes: int = 2,
    mask_type: str = "ellipse",
) -> torch.Tensor:
    """Generate (B, H, W) integer pseudo-labels from bounding box annotations.

    Args:
        predictions: (B, C, H, W) — used only for shape and device.
        boxes: list of (N_b, 4) tensors [x1, y1, x2, y2] per image.
        box_classes: list of (N_b,) class-id tensors per image.
        num_classes: number of classes (for validation).
        mask_type: ``"ellipse"`` (default, BoxSup-style) or ``"rectangle"``.

    Returns:
        (B, H, W) long tensor with 0 = background, >0 = class id.
    """
    B, C, H, W = predictions.shape
    pseudo = torch.zeros(B, H, W, device=predictions.device, dtype=torch.long)

    for b in range(B):
        box_list = boxes[b]
        cls_list = box_classes[b] if box_classes is not None else None

        if not torch.is_tensor(box_list):
            box_list = torch.as_tensor(box_list, device=predictions.device)
        if box_list.numel() == 0:
            continue
        box_list = box_list.view(-1, 4)

        for i in range(box_list.shape[0]):
            x1, y1, x2, y2 = box_list[i].long().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = 1
            if cls_list is not None:
                cls_id = int(cls_list[i].item())
                cls_id = max(1, min(cls_id, num_classes - 1))

            crop_h, crop_w = y2 - y1, x2 - x1

            if mask_type == "ellipse" and crop_h > 2 and crop_w > 2:
                mask = generate_ellipse_mask(
                    crop_h, crop_w, predictions.device)
            else:
                mask = torch.ones(crop_h, crop_w, device=predictions.device)

            # Only overwrite background (0), don't overwrite existing classes
            region = pseudo[b, y1:y2, x1:x2]
            pseudo[b, y1:y2, x1:x2] = torch.where(
                region == 0,
                (mask > 0.5).long() * cls_id,
                region,
            )

    return pseudo


# ---------------------------------------------------------------------------
# Main loss module
# ---------------------------------------------------------------------------
@LOSS_REGISTRY.register("box_supervised")
class BoxSupervisedLoss(nn.Module):
    """BoxSup loss: pseudo-mask supervision from bounding box annotations.

    Faithful to Dai et al. (ICCV 2015) — uses region proposals inside boxes
    as pixel-level pseudo-labels to supervise the segmentation network.

    Args:
        mask_type: ``"ellipse"`` (default) uses inscribed ellipses (avoids
            treating box corners as foreground — key BoxSup insight).
            ``"rectangle"`` uses the full box rectangle.
        outside_penalty: weight for suppressing predictions outside boxes.
        base_loss_fn: loss function for pseudo-mask supervision.
            Defaults to standard cross-entropy.
    """

    def __init__(
        self,
        mask_type: str = "ellipse",
        outside_penalty: float = 0.1,
        base_loss_fn=None,
        **kwargs,
    ):
        super().__init__()
        assert mask_type in ("ellipse", "rectangle"), \
            f"mask_type must be 'ellipse' or 'rectangle', got '{mask_type}'"
        self.mask_type = mask_type
        self.outside_penalty = outside_penalty

        if base_loss_fn is None:
            from medseg.losses.compound_loss import CompoundLoss
            self.base_loss_fn = CompoundLoss()
        else:
            self.base_loss_fn = base_loss_fn

    def forward(
        self,
        predictions: torch.Tensor,
        boxes: Optional[list] = None,
        box_classes: Optional[list] = None,
        image_labels: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the BoxSup loss.

        Args:
            predictions: (B, C, H, W) class logits.
            boxes: list of (N_b, 4) tensors per image.
            box_classes: list of (N_b,) class-id tensors per image.
            image_labels: (B, C) multilabel targets (fallback).
            target: (B, H, W) ground-truth mask (bypasses BoxSup logic).
            image: unused (kept for interface compatibility).
        """
        B, C, H, W = predictions.shape

        # Full pixel-level supervision available → use it directly
        if target is not None:
            return self.base_loss_fn(predictions, target)

        # No boxes → fall back to image-level classification
        if boxes is None:
            if image_labels is not None:
                return self._image_level_loss(predictions, image_labels)
            raise ValueError(
                "Either boxes, image_labels, or target must be provided")

        # Generate pseudo-masks from boxes (BoxSup core idea)
        pseudo = generate_box_pseudo_masks(
            predictions, boxes, box_classes,
            num_classes=C, mask_type=self.mask_type)

        losses = {}

        # 1. Standard segmentation loss on pseudo-masks
        ce_loss = self.base_loss_fn(predictions, pseudo)
        losses["ce"] = ce_loss

        # 2. Outside-box penalty (suppress predictions outside boxes)
        if self.outside_penalty > 0:
            outside_loss = self._compute_outside_penalty(predictions, boxes)
            losses["outside"] = self.outside_penalty * outside_loss

        return sum(losses.values())

    # ------------------------------------------------------------------
    # Outside-box penalty
    # ------------------------------------------------------------------
    def _compute_outside_penalty(self, predictions, boxes):
        """Penalise foreground predictions outside any bounding box."""
        B, C, H, W = predictions.shape
        box_mask = torch.zeros(B, H, W, device=predictions.device)
        for b in range(B):
            box_list = boxes[b]
            if not torch.is_tensor(box_list):
                box_list = torch.as_tensor(box_list, device=predictions.device)
            if box_list.numel() == 0:
                continue
            box_list = box_list.view(-1, 4)
            for i in range(box_list.shape[0]):
                x1, y1, x2, y2 = box_list[i].long().tolist()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                box_mask[b, y1:y2, x1:x2] = 1.0

        outside = 1.0 - box_mask
        if C >= 2:
            prob_outside = (predictions[:, 1:, :, :].softmax(dim=1)
                            * outside.unsqueeze(1))
        else:
            prob_outside = predictions.sigmoid() * outside.unsqueeze(1)
        return prob_outside.mean()

    # ------------------------------------------------------------------
    # Image-level fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _image_level_loss(predictions, image_labels):
        gp = predictions.mean(dim=[2, 3])
        return F.binary_cross_entropy_with_logits(gp, image_labels.float())
