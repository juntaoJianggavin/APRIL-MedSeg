# Reference: Lin et al., "ScribbleSup: Scribble-Supervised Convolutional
# Networks for Semantic Segmentation", CVPR 2016.
# Paper: https://arxiv.org/abs/1604.05144
"""ScribbleSup — Scribble-Supervised CNNs for Semantic Segmentation.

Lin et al., "ScribbleSup: Scribble-Supervised Convolutional Networks for
Semantic Segmentation", CVPR 2016.
Paper: https://arxiv.org/abs/1604.05144

Faithful-in-spirit ``light`` variant suitable for in-tree training:

    L = CE(scribble pixels)
      + lambda_crf * PairwiseCRF(predictions, images)
      + lambda_ent * mean( H(softmax(predictions)) )

The original paper alternates training with a graph-cut step that
propagates scribble labels to the full image, then refits the CNN on the
propagated mask. Implementing a true GraphCut inside an autograd loop is
expensive and brings a non-differentiable dependency; instead we inline a
differentiable **pairwise colour-affinity** surrogate (a local-window,
lightweight variant of the Gated CRF smoothness prior from Obukhov et al.,
BMVC 2019), which plays the same role of "labels should be smooth where
colours are smooth". A predictor-entropy-minimisation term sharpens
predictions on unlabelled pixels, matching the EM behaviour of the
original alternating optimisation.

This module is self-contained: it does not import from the removed
``gated_crf.py`` (that file claimed to reproduce Obukhov et al.'s Gated
CRF but reduced to a 3x3 pairwise smoother without the gating network or
dense mean-field, so it was dropped in a code-fidelity cleanup).
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


# ---------------------------------------------------------------------------
# Inlined pairwise colour-affinity smoothness (self-contained)
# ---------------------------------------------------------------------------
def _pairwise_colour_affinity(
    predictions: torch.Tensor,
    images: torch.Tensor,
    kernel_size: int = 3,
    dilation: int = 1,
    sigma_rgb: float = 15.0,
    sigma_xy: float = 100.0,
) -> torch.Tensor:
    """Local pairwise colour-affinity smoothness loss.

    For each pixel pair inside a dilated window, penalise prediction
    differences weighted by ``exp(-rgb_diff / (2*sigma_rgb^2)) *
    exp(-xy_diff / (2*sigma_xy^2))``. This is a lightweight, differentiable
    surrogate for the Gated CRF smoothness term.

    Args:
        predictions: (B, C, H, W) class logits.
        images: (B, 3, H, W) input image.
        kernel_size: local window radius.
        dilation: dilation factor.
        sigma_rgb: colour Gaussian bandwidth.
        sigma_xy: spatial Gaussian bandwidth.

    Returns:
        Scalar smoothness loss.
    """
    B, C_img, H, W = images.shape
    prob = F.softmax(predictions, dim=1)
    half_k = max(kernel_size // 2, 1)
    d = max(dilation, 1)
    total = predictions.new_zeros(())
    count = 0
    for dy in range(-half_k, half_k + 1):
        for dx in range(-half_k, half_k + 1):
            if dx == 0 and dy == 0:
                continue
            sdy, sdx = dy * d, dx * d
            xy_gate = float(np.exp(-(sdx * sdx + sdy * sdy) / (2.0 * sigma_xy ** 2)))
            src_y = slice(max(0, -sdy), H - max(0, sdy))
            dst_y = slice(max(0, sdy), H - max(0, -sdy))
            src_x = slice(max(0, -sdx), W - max(0, sdx))
            dst_x = slice(max(0, sdx), W - max(0, -sdx))
            img_diff = (images[:, :, src_y, src_x] - images[:, :, dst_y, dst_x]).pow(2).sum(dim=1)
            rgb_gate = torch.exp(-img_diff / (2.0 * sigma_rgb ** 2))
            pred_diff = (prob[:, :, src_y, src_x] - prob[:, :, dst_y, dst_x]).pow(2).sum(dim=1)
            total = total + xy_gate * (rgb_gate * pred_diff).mean()
            count += 1
    return total / max(count, 1)


@LOSS_REGISTRY.register("scribble_sup")
class ScribbleSupLoss(nn.Module):
    """Scribble-supervised segmentation loss (light variant).

    Args:
        ignore_index: Pixel value in ``scribbles`` to treat as unlabelled
            (default -1, matches PyTorch's CE convention).
        crf_weight: Weight on the pairwise colour-affinity surrogate for
            the paper's graph-cut propagation step (default 0.1).
        entropy_weight: Weight on the predictor-entropy minimisation term
            applied to all pixels (default 0.05).
        kernel_size: CRF surrogate window size (default 3).
        dilation: CRF surrogate dilation (default 1).
        sigma_rgb: Colour-Gaussian sigma for the CRF surrogate (default 15.0).
        sigma_xy: Spatial-Gaussian sigma for the CRF surrogate (default 100.0).
    """

    def __init__(
        self,
        ignore_index: int = -1,
        crf_weight: float = 0.1,
        entropy_weight: float = 0.05,
        kernel_size: int = 3,
        dilation: int = 1,
        sigma_rgb: float = 15.0,
        sigma_xy: float = 100.0,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.crf_weight = crf_weight
        self.entropy_weight = entropy_weight
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy

    @staticmethod
    def _entropy(predictions: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(predictions, dim=1)
        return -(prob * torch.log(prob + 1e-8)).sum(dim=1).mean()

    def forward(
        self,
        predictions: torch.Tensor,
        scribbles: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) semantic logits.
            scribbles: (B, H, W) integer labels; ``ignore_index`` for
                unlabelled pixels. May be omitted in the rare case of pure
                CRF/entropy-only regularisation.
            images: (B, 3, H, W) for the colour-affinity kernel. Required
                whenever ``crf_weight > 0``; missing is a hard error.
            labeled_loss: Optional pre-computed supervised CE/dice term to
                add (mixed-supervision setting).
        """
        if predictions.dim() != 4:
            raise ValueError(
                f"predictions must be (B, C, H, W); got shape {tuple(predictions.shape)}"
            )

        total = predictions.new_zeros(())

        # ---- (1) Sparse CE on scribble pixels ---------------------------
        if scribbles is not None:
            if scribbles.dim() == 4 and scribbles.shape[1] == 1:
                scribbles = scribbles.squeeze(1)
            total = total + F.cross_entropy(
                predictions,
                scribbles.long(),
                ignore_index=self.ignore_index,
            )

        # ---- (2) Pairwise colour-affinity surrogate ---------------------
        if self.crf_weight > 0:
            if images is None:
                raise ValueError(
                    "ScribbleSupLoss with crf_weight>0 needs ``images`` "
                    "(B, 3, H, W) for the colour-affinity kernel."
                )
            crf_term = _pairwise_colour_affinity(
                predictions, images,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                sigma_rgb=self.sigma_rgb,
                sigma_xy=self.sigma_xy,
            )
            total = total + self.crf_weight * crf_term

        # ---- (3) Predictor-entropy minimisation -------------------------
        if self.entropy_weight > 0:
            total = total + self.entropy_weight * self._entropy(predictions)

        if labeled_loss is not None:
            total = total + labeled_loss
        return total
