"""SAM3 (Segment Anything Model 3, Meta 2025) - 2D medical-segmentation variant.
    SAM3 ( Segment Anything Model 3, Meta 2025 ) - 2D medical-segmentation variant。

Reference:
    Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, et al.,
    "SAM 3: Segment Anything with Concepts", Meta Superintelligence Labs, 2025.
    arXiv: 2511.16719.  https://github.com/facebookresearch/sam3

SAM3 is a unified foundation model for promptable segmentation in images and
videos (848M parameters).  Compared to SAM2, SAM3 replaces the Hiera vision
encoder with **Perception Encoder (PE)** — Meta's ViT variant trained via
contrastive vision-language learning (arXiv:2504.13181).  PE uses RoPE
(Rotary Position Embeddings), LayerScale, and a pre-transformer norm.

PE is available in timm as ``vit_pe_spatial_large_patch14_448`` (ViT-L/14,
``embed_dim=1024``, ``depth=24``, ``patch_size=14``, native 448px).  Pretrained
weights are auto-downloaded from HuggingFace Hub.

This module provides the *2D prompt-free* variant: we reuse the PE backbone
(via timm), extract intermediate features from 4 transformer blocks, and build
a genuine multi-scale pyramid (DPT-style, strides 4/8/16/32) that feeds a
4-stage CNN mask decoder — the same decoder topology used by SAM2.

Inputs are padded to a multiple of the patch size (14).  Outputs are cropped
back to the original H × W.
"""
# Source: https://github.com/facebookresearch/sam3
#          https://github.com/facebookresearch/perception_models

from __future__ import annotations

import os

# Keep huggingface_hub from hanging the constructor on flaky networks.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")

import warnings
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_base import SAMBase, load_with_ssl_fallback

# Shared decoder utilities (generic, not SAM2-specific).
from .sam2 import _LateralProj, _UpFuseBlock, _SAM2MaskDecoder

# DPT-style multi-scale projector (genuine multi-scale from ViT blocks).
from medseg.models.encoders.foundation._base import DPTHead


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PE_MODEL = "vit_pe_spatial_large_patch14_448"
_PE_PATCH_SIZE = 14
_PE_NATIVE_SIZE = 448
_PE_PROJ_DIM = 256


# ---------------------------------------------------------------------------
# Backbone wrapper: Perception Encoder (PE) via timm
# ---------------------------------------------------------------------------
class _PEEncoder(nn.Module):
    """Perception Encoder (PE) backbone wrapper via timm.
        Perception Encoder ( PE ) 骨干网络 wrapper via timm。

    PE is a ViT-L/14 with RoPE (``embed_dim=1024``, ``depth=24``,
    ``patch_size=14``).  Since PE is a *non-hierarchical* ViT, all transformer
    blocks operate at the same spatial resolution (patch grid).  To obtain a
    genuine multi-scale feature pyramid (strides 4/8/16/32) we extract
    intermediate token outputs from 4 evenly-spaced blocks and feed them
    through a DPT-style projector — the same approach used by the project's
    other ViT-based foundation encoders (KEEP, MedCLIP, …).

    Exposes ``.blocks`` (a flat ``nn.ModuleList`` of transformer blocks) so
    ``SAMBase.apply_freeze`` can implement the ``unfreeze_last_n_blocks``
    schedule.  Also exposes ``.channels`` and ``.strides`` for the decoder.
    """

    def __init__(self, in_channels: int, img_size: int, pretrained: bool,
                 backbone: str = _PE_MODEL):
        super().__init__()
        import timm

        def _create(pretrained: bool):
            return timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,          # remove classification head
                in_chans=in_channels,   # timm auto-resamples patch-embed weights
                dynamic_img_size=True,  # allow variable input resolutions
            )

        model = load_with_ssl_fallback(_create, pretrained=pretrained)

        self.model = model
        self.embed_dim = int(model.embed_dim)
        ps = model.patch_embed.patch_size
        if isinstance(ps, (tuple, list)):
            ps = ps[0]
        self.patch_size = int(ps)
        self.num_prefix_tokens = int(model.num_prefix_tokens)

        # Expose transformer blocks so SAMBase.apply_freeze can selectively
        # unfreeze the tail.
        self.blocks = model.blocks

        # DPT-style multi-scale projector: builds a 4-level pyramid from
        # tokens extracted at different depths.
        self._dpt = DPTHead(
            embed_dim=self.embed_dim,
            num_prefix_tokens=self.num_prefix_tokens,
        )
        self.channels = list(self._dpt.out_channels)  # [dim//8, dim//4, dim//2, dim]
        self.strides = [4, 8, 16, 32]
        self._block_indices = DPTHead.default_block_indices(len(model.blocks))
        self._backbone_name = backbone

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return a 4-level multi-scale feature pyramid (deepest LAST)."""
        B, C, H, W = x.shape
        p = self.patch_size

        # Pad to a multiple of patch_size so the patch grid is integer.
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2], x.shape[-1]
        h_patches = Hp // p
        w_patches = Wp // p

        # Extract intermediate token outputs from 4 evenly-spaced blocks.
        # output_fmt='NLC' → list of (B, N, C) tensors, prefix tokens stripped.
        intermediates = self.model.forward_intermediates(
            x,
            indices=self._block_indices,
            output_fmt="NLC",
            norm=True,
            intermediates_only=True,
        )

        # Build multi-scale pyramid (strides 4/8/16/32) via DPTHead.
        features = self._dpt(
            list(intermediates), h_patches, w_patches, H, W,
        )
        return features


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class SAM3(SAMBase):
    """SAM3 2D segmentation network with a Perception Encoder backbone.
        SAM3 2D segmentation network with a Perception Encoder backbone。

    Unlike SAM2 (which uses a Hiera hierarchical ViT), SAM3 uses **Perception
    Encoder (PE)** — a ViT-L/14 with RoPE trained via contrastive
    vision-language learning.  PE is loaded via timm
    (``vit_pe_spatial_large_patch14_448``); pretrained weights are
    auto-downloaded from HuggingFace Hub.

    Because PE is a non-hierarchical ViT, a DPT-style projector extracts
    features from 4 transformer blocks to form a genuine multi-scale pyramid
    (strides 4/8/16/32).  This pyramid feeds a 4-stage CNN mask decoder —
    the same decoder topology used by SAM2.

    Args:
        in_channels: number of input channels.
        num_classes: number of output segmentation classes.
        img_size: input spatial size. Padded internally to a multiple of 14
            (the PE patch size).
        pretrained: whether to attempt loading pretrained PE weights via
            timm. Falls back to random init when the download is unreachable.
        pretrained_path: unused; accepted for API parity with other SAM-family
            constructors.
        freeze_image_encoder / freeze_prompt_encoder / freeze_mask_decoder:
            standard freezing knobs handled by ``SAMBase.apply_freeze``.
        unfreeze_last_n_blocks: when > 0 and ``freeze_image_encoder`` is True,
            re-enables training on the last N transformer blocks of the
            backbone.
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
        # SAM3 here is prompt-free; expose None so apply_freeze is uniform.
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
                f"SAM3: expected 4 backbone stages, got {len(feats)}."
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


# Public alias for downstream registries.
Sam3 = SAM3
