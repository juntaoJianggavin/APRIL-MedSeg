"""MedSAM3 (2D variant) — Delving into Segment Anything with Medical Concepts.
    MedSAM3 ( 2D variant ) — Delving into Segment Anything with 医学 概念。

Reference:
    Anglin Liu, Rundong Xue, Xu R. Cao, Yifan Shen, Yi Lu, Xiang Li,
    Qianqian Chen, Jintai Chen.
    "MedSAM3: Delving into Segment Anything with Medical Concepts", 2025.
    arXiv: 2511.19046.

MedSAM3 fine-tunes the SAM3 architecture on medical images paired with
semantic conceptual labels, enabling *Medical Promptable Concept Segmentation*
(PCS).  Architecturally, MedSAM3 shares SAM3's **Perception Encoder (PE)**
backbone — a ViT-L/14 with RoPE (``vit_pe_spatial_large_patch14_448`` via
timm) — and uses LoRA fine-tuning for efficient adaptation.

This module provides the **2D prompt-free** variant: we reuse the PE backbone
(loaded via timm, weights from HuggingFace Hub), extract intermediate features
from 4 transformer blocks via a DPT-style projector to form a genuine
multi-scale pyramid (strides 4/8/16/32), and feed it to a 4-stage CNN mask
decoder.  This keeps the model self-contained and trainable on any 2D medical
dataset without text / box / point prompts.

Inputs are padded to a multiple of the patch size (14).  Outputs are cropped
back to the original H × W.
"""
# Source: https://arxiv.org/abs/2511.19046
#          https://github.com/Joey-S-Liu/MedSAM3  (based on facebookresearch/sam3)

from __future__ import annotations

import os

# Keep huggingface_hub from hanging the constructor on flaky networks.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_base import SAMBase

# Shared decoder utilities (generic, not SAM2-specific).
from .sam2 import _LateralProj, _UpFuseBlock, _SAM2MaskDecoder

# PE backbone wrapper + DPTHead (shared with SAM3, composition not inheritance).
from .sam3 import _PEEncoder, _PE_MODEL, _PE_PATCH_SIZE, _PE_PROJ_DIM


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class MedSAM3(SAMBase):
    """MedSAM3 2D medical-segmentation network with a Perception Encoder backbone.
        MedSAM3 2D 医学 segmentation network with a Perception Encoder backbone。

    MedSAM3 shares SAM3's PE backbone architecture (ViT-L/14 with RoPE).  This
    2D variant uses the same ``vit_pe_spatial_large_patch14_448`` backbone via
    timm together with a DPT-style multi-scale projector and a 4-stage CNN mask
    decoder.  All constructor arguments are identical to :class:`SAM3`.

    Args:
        in_channels: number of input channels.
        num_classes: number of output segmentation classes.
        img_size: input spatial size. Padded internally to a multiple of 14
            (the PE patch size).
        pretrained: whether to attempt loading pretrained PE weights via
            timm. Falls back to random init when the download is unreachable.
        pretrained_path: unused; accepted for API parity.
        freeze_image_encoder / freeze_prompt_encoder / freeze_mask_decoder:
            standard freezing knobs handled by ``SAMBase.apply_freeze``.
        unfreeze_last_n_blocks: when > 0 and ``freeze_image_encoder`` is True,
            re-enables training on the last N transformer blocks.
        inference_only: when True, the entire model is put into eval mode and
            all parameters are frozen.
    """

    PROJ_DIM = _PE_PROJ_DIM
    _BACKBONE = _PE_MODEL

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

        self.image_encoder = _PEEncoder(
            in_channels=in_channels,
            img_size=img_size,
            pretrained=self._pretrained,
            backbone=self._BACKBONE,
        )
        # MedSAM3 here is prompt-free; expose None so apply_freeze is uniform.
        self.prompt_encoder = None

        self.mask_decoder = _SAM2MaskDecoder(
            in_channels=self.image_encoder.channels,
            num_classes=num_classes,
            proj_dim=self.PROJ_DIM,
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

        # 1. Pad input to a multiple of the PE patch size (14).
        x_pad, (pad_h, pad_w) = self._pad_to_multiple(x, _PE_PATCH_SIZE)
        Hp, Wp = x_pad.shape[-2:]

        # 2. Multi-scale features (strides 4, 8, 16, 32) from PE + DPTHead.
        feats = self.image_encoder(x_pad)
        if len(feats) < 4:
            raise RuntimeError(
                f"MedSAM3: expected 4 backbone stages, got {len(feats)}."
            )
        feats = feats[-4:]

        # 3. Decode.
        logits = self.mask_decoder(feats)

        # 4. Snap back to the padded input size, then crop to (H, W).
        if logits.shape[-2:] != (Hp, Wp):
            logits = F.interpolate(
                logits, size=(Hp, Wp), mode="bilinear", align_corners=False,
            )
        if pad_h or pad_w:
            logits = logits[..., :H, :W]
        return logits
