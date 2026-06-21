"""SurgicalSAM foundation-model encoder (surgical ViT-H/14 SAM).
    SurgicalSAM foundation-model 编码器。

Reference:
    Yue et al., "SurgicalSAM: Efficient Class Promptable Surgical Instrument
    Segmentation", AAAI 2024.
    https://github.com/wenxi-yue/SurgicalSAM

SurgicalSAM uses SAM's ViT-Huge/14 image encoder (frozen) with learned class
prompts for surgical instrument segmentation.  The backbone is a
ViT-Huge/14 (``embed_dim=1280``, ``patch_size=14``, 32 layers).

``pretrained=True`` loads timm ViT-H/14 weights.  To use the actual SAM
checkpoint (``sam_vit_h_4b8939.pth``), download it and pass via
``pretrained_path``.

Registered as ``"surgical_sam"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://github.com/wenxi-yue/SurgicalSAM

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import (
    DPTHead, BaseFoundationEncoder, load_with_ssl_fallback,
)


# 架构 constants / Architecture constants.
_SURGICAL_SAM_EMBED_DIM = 1280
_SURGICAL_SAM_PATCH_SIZE = 14
_SURGICAL_SAM_TIMM_NAME = "vit_huge_patch14_224"

PRIMARY_BACKBONE_NAME = _SURGICAL_SAM_TIMM_NAME


@ENCODER_REGISTRY.register("surgical_sam")
class SurgicalSAMEncoder(BaseFoundationEncoder):
    """SurgicalSAM (SAM ViT-H/14) encoder with DPT-style multi-block output.
        SurgicalSAM (SAM ViT-H/14) 编码器。

    The backbone is a ViT-Huge/14 (``embed_dim=1280``, ``patch_size=14``,
    32 layers).  ``out_channels = [dim/8, dim/4, dim/2, dim]`` (deepest LAST).

    Parameters
    ----------
    in_channels : int
        Number of input image channels (default 3).
    img_size : int
        Reference spatial size (default 224).  SAM was pretrained at 1024×1024.
    pretrained : bool
        Load timm ViT-H/14 pretrained weights.
    pretrained_path : Optional[str]
        Path to a SAM ViT-H checkpoint (``sam_vit_h_4b8939.pth``).
    """

    native_img_size: int = 1024  # SAM ViT-H was pretrained at 1024×1024
    PATCH_SIZE = _SURGICAL_SAM_PATCH_SIZE
    EMBED_DIM = _SURGICAL_SAM_EMBED_DIM

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False, **kwargs):
        super().__init__(
            in_channels=in_channels, img_size=img_size,
            pretrained=pretrained, pretrained_path=pretrained_path,
            freeze=freeze, unfreeze_last_n=unfreeze_last_n,
            inference_only=inference_only, **kwargs,
        )

        # ---- 通道 adapter for non-RGB inputs ---- / ---- channel adapter ----
        if in_channels != 3:
            self.input_adapter: nn.Module = nn.Conv2d(
                in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # ---- Backbone — timm ViT-H/14 ---- / ---- backbone ----
        import timm
        self.backbone = load_with_ssl_fallback(
            timm.create_model, _SURGICAL_SAM_TIMM_NAME,
            pretrained=pretrained and pretrained_path is None,
            num_classes=0,
            img_size=img_size,
            in_chans=3,
            dynamic_img_size=True,
        )
        self._backbone_name = PRIMARY_BACKBONE_NAME

        # ---- 可选 local checkpoint ( SAM ViT-H format ) ---- / ---- optional local checkpoint ----
        if pretrained_path:
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                if isinstance(state, dict):
                    for key in ("state_dict", "model", "model_state_dict"):
                        if key in state and isinstance(state[key], dict):
                            state = state[key]
                            break
                # SAM checkpoints have keys like "image_encoder.blocks.{i}..."
                # Strip "image_encoder." prefix if present
                if isinstance(state, dict):
                    cleaned = {}
                    for k, v in state.items():
                        nk = k
                        for pref in ("module.", "image_encoder.", "backbone.",
                                     "encoder."):
                            if nk.startswith(pref):
                                nk = nk[len(pref):]
                                break
                        cleaned[nk] = v
                    state = cleaned
                msg = self.backbone.load_state_dict(state, strict=False)
                warnings.warn(
                    f"[surgical_sam] loaded SAM weights from "
                    f"'{pretrained_path}': {msg}"
                )
            except Exception as e:
                warnings.warn(
                    f"[surgical_sam] failed to load checkpoint "
                    f"'{pretrained_path}': {e}"
                )

        # ---- Backbone introspection ---- / ---- backbone introspection ----
        ps = self.backbone.patch_embed.patch_size
        if isinstance(ps, (tuple, list)):
            self.patch_size = int(ps[0])
        else:
            self.patch_size = int(ps)
        self.embed_dim = int(
            getattr(self.backbone, "embed_dim", _SURGICAL_SAM_EMBED_DIM))
        self.num_prefix_tokens = int(
            getattr(self.backbone, "num_prefix_tokens", 1))

        # ---- DPT head ----
        self.dpt = DPTHead(
            embed_dim=self.embed_dim,
            num_prefix_tokens=self.num_prefix_tokens,
        )
        self.out_channels = self.dpt.out_channels
        self._block_indices = DPTHead.default_block_indices(
            len(self.backbone.blocks))

        self._maybe_inject_adapters()
        self._apply_freeze_policy()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_adapter(x)
        B, _, H, W = x.shape
        p = self.patch_size

        # 填充到 patch_size * 2 的倍数 / Pad to multiple of patch_size * 2
        unit = p * 2
        pad_h = (unit - H % unit) % unit
        pad_w = (unit - W % unit) % unit
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2], x.shape[-1]

        # 从不同深度 block 提取 token（DPT 核心）
        # Extract tokens from different-depth blocks (DPT core)
        multi_tokens = self.backbone.get_intermediate_layers(
            x, n=self._block_indices,
        )

        h_patches = Hp // p
        w_patches = Wp // p

        return self.dpt(list(multi_tokens), h_patches, w_patches, H, W)
