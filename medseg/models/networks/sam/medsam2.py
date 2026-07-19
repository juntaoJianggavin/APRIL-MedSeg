"""MedSAM2 (2D variant) — Segment Anything in Medical Images.
    MedSAM2 ( 2D variant ) — Segment Anything in 医学 图像。

Reference:
    Jun Ma, Zongxin Yang, Sumin Kim, Bihui Chen, Mohammed Baharoon, et al.,
    "MedSAM2: Segment Anything in 3D Medical Images and Videos", 2025.
    arXiv: 2504.03600.  https://github.com/bowang-lab/MedSAM2

MedSAM2 extends the SAM2 architecture to 3D medical images and videos by
fine-tuning the Hiera image encoder on large-scale medical CT / MRI / PET /
ultrasound / endoscopy data.  The upstream model processes volumetric inputs
slice-by-slice with temporal memory across slices.

This module provides the **2D prompt-free** variant: we reuse the SAM2 Hiera
encoder (``hiera_tiny_224``, matching the ``sam2.1_hiera_tiny`` checkpoint used
in MedSAM2 training) and a multi-scale CNN decoder.  This keeps the model
self-contained (torch + timm only) and trainable on any 2D medical dataset
without box / point prompts or 3D memory banks.

Inputs are padded to a multiple of 32 (the Hiera total stride). Outputs are
cropped back to the original H × W.
"""
# Source: https://github.com/bowang-lab/MedSAM2

from __future__ import annotations

import os

# Keep huggingface_hub from hanging the constructor on flaky networks.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_base import SAMBase, load_with_ssl_fallback

# Shared Hiera tools (generic, not SAM2-specific).
from .sam2 import (
    _HieraEncoder,
    _LateralProj,
    _UpFuseBlock,
    _TOTAL_STRIDE,
    _round_up,
)

_MEDSAM2_BACKBONE = "hiera_tiny_224"
_MEDSAM2_PROJ_DIM = 256


# ---------------------------------------------------------------------------
# Mask decoder (MedSAM2 multi-scale feature fusion)
# ---------------------------------------------------------------------------
class _MedSAM2MaskDecoder(nn.Module):
    """4-stage CNN mask decoder that fuses multi-scale Hiera features.
        4-stage CNN mask 解码器。

    Stage layout (Hiera strides 4/8/16/32):
        stage1: 1/32 -> 1/16, add lateral(f3)
        stage2: 1/16 -> 1/8,  add lateral(f2)
        stage3: 1/8  -> 1/4,  add lateral(f1)
        stage4: 1/4  -> 1/1,  ConvTranspose ×4 to full resolution

    All intermediate features live at ``proj_dim`` channels.
    """

    def __init__(self, in_channels: Sequence[int], num_classes: int,
                 proj_dim: int = _MEDSAM2_PROJ_DIM):
        super().__init__()
        c1, c2, c3, c4 = in_channels  # strides 4, 8, 16, 32
        self.lateral4 = _LateralProj(c4, proj_dim)
        self.lateral3 = _LateralProj(c3, proj_dim)
        self.lateral2 = _LateralProj(c2, proj_dim)
        self.lateral1 = _LateralProj(c1, proj_dim)

        self.up1 = _UpFuseBlock(proj_dim, proj_dim)  # 1/32 -> 1/16
        self.up2 = _UpFuseBlock(proj_dim, proj_dim)  # 1/16 -> 1/8
        self.up3 = _UpFuseBlock(proj_dim, proj_dim)  # 1/8  -> 1/4

        # Final 4x upsample to full resolution, producing class logits.
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(proj_dim, proj_dim // 2,
                               kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=min(32, proj_dim // 2),
                         num_channels=proj_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(proj_dim // 2, proj_dim // 4,
                               kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=min(32, proj_dim // 4),
                         num_channels=proj_dim // 4),
            nn.GELU(),
            nn.Conv2d(proj_dim // 4, num_classes, kernel_size=1),
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = feats  # strides 4, 8, 16, 32
        x4 = self.lateral4(f4)
        x3 = self.lateral3(f3)
        x2 = self.lateral2(f2)
        x1 = self.lateral1(f1)

        y = self.up1(x4, x3)
        y = self.up2(y, x2)
        y = self.up3(y, x1)
        y = self.up4(y)
        return y


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class MedSAM2(SAMBase):
    """MedSAM2 2D medical-segmentation network.
        MedSAM2 2D 医学 segmentation network。

    MedSAM2 fine-tunes SAM2's Hiera encoder on medical data.  This 2D variant
    uses the same ``hiera_tiny_224`` backbone as the upstream MedSAM2
    (``sam2.1_hiera_tiny`` checkpoint) together with a multi-scale CNN decoder.

    Args:
        in_channels: number of input channels.
        num_classes: number of output segmentation classes.
        img_size: input spatial size. Padded internally to a multiple of 32.
        pretrained: whether to attempt loading pretrained Hiera weights via
            timm. Falls back to random init when the download is unreachable.
        pretrained_path: unused; accepted for API parity.
        freeze_image_encoder / freeze_prompt_encoder / freeze_mask_decoder:
            standard freezing knobs handled by ``SAMBase.apply_freeze``.
        unfreeze_last_n_blocks: when > 0 and ``freeze_image_encoder`` is True,
            re-enables training on the last N transformer blocks.
        inference_only: when True, the entire model is put into eval mode and
            all parameters are frozen.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
        unfreeze_last_n_blocks: int = 0,
        inference_only: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            freeze_image_encoder=freeze_image_encoder,
            freeze_prompt_encoder=freeze_prompt_encoder,
            freeze_mask_decoder=freeze_mask_decoder,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            inference_only=inference_only,
        )

        # Hiera positional structures are baked at construction; pick the
        # backbone img_size as the smallest multiple of 32 that is >= img_size.
        self._backbone_size = max(_TOTAL_STRIDE,
                                  _round_up(img_size, _TOTAL_STRIDE))

        self.image_encoder = _HieraEncoder(
            in_channels=in_channels,
            img_size=self._backbone_size,
            pretrained=self._pretrained,
            backbone=_MEDSAM2_BACKBONE,
        )
        # MedSAM2 here is prompt-free; expose None so apply_freeze is uniform.
        self.prompt_encoder = None

        self.mask_decoder = _MedSAM2MaskDecoder(
            in_channels=self.image_encoder.channels,
            num_classes=num_classes,
            proj_dim=_MEDSAM2_PROJ_DIM,
        )

        self.apply_freeze()

    # ------------------------------------------------------------------
    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, mult: int):
        H, W = x.shape[-2:]
        pad_h = (mult - H % mult) % mult
        pad_w = (mult - W % mult) % mult
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        return F.pad(x, (0, pad_w, 0, pad_h)), (pad_h, pad_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        # 1. Pad input to a multiple of 32.
        x_pad, (pad_h, pad_w) = self._pad_to_multiple(x, _TOTAL_STRIDE)
        Hp, Wp = x_pad.shape[-2:]

        # 2. If the padded input does not match the backbone's strict size,
        # bilinear-resize into the backbone, then resize logits back.
        if (Hp, Wp) != (self._backbone_size, self._backbone_size):
            x_bb = F.interpolate(
                x_pad,
                size=(self._backbone_size, self._backbone_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_bb = x_pad

        # 3. Multi-scale features (strides 4, 8, 16, 32).
        feats = self.image_encoder(x_bb)
        if len(feats) < 4:
            raise RuntimeError(
                f"MedSAM2: expected 4 backbone stages, got {len(feats)}."
            )
        feats = feats[-4:]

        # 4. Decode.
        logits = self.mask_decoder(feats)

        # 5. Snap back to the padded input size, then crop to (H, W).
        if logits.shape[-2:] != (Hp, Wp):
            logits = F.interpolate(
                logits, size=(Hp, Wp), mode="bilinear", align_corners=False,
            )
        if pad_h or pad_w:
            logits = logits[..., :H, :W]
        return logits
