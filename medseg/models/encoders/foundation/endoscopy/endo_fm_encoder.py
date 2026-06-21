"""Endo-FM foundation-model encoder (endoscopy ViT-B/16).
    Endo-FM foundation-model 编码器。

Reference:
    Wang et al., "Foundation Model for Endoscopy Video Analysis via
    Large-scale Self-supervised Pre-train", MICCAI 2023.
    https://github.com/med-air/Endo-FM

Endo-FM is a ViT-Base/16 (``embed_dim=768``, ``patch_size=16``, 12 layers)
pretrained on 33K+ endoscopic video clips via self-supervised spatial-temporal
learning.  The architecture follows DINO's ViT-B/16.

``pretrained=True`` loads timm ViT-B/16 weights.  To use the actual Endo-FM
checkpoint, download it from the GitHub repo and pass via ``pretrained_path``.

Registered as ``"endo_fm"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://github.com/med-air/Endo-FM

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import (
    DPTHead, BaseFoundationEncoder, load_with_ssl_fallback,
)


# 架构 constants / Architecture constants.
_ENDOFM_EMBED_DIM = 768
_ENDOFM_PATCH_SIZE = 16
_ENDOFM_TIMM_NAME = "vit_base_patch16_224"

PRIMARY_BACKBONE_NAME = _ENDOFM_TIMM_NAME


@ENCODER_REGISTRY.register("endo_fm")
class EndoFMEncoder(BaseFoundationEncoder):
    """Endo-FM (endoscopy ViT-B/16) encoder with DPT-style multi-block output.
        Endo-FM (内窥镜 ViT-B/16) 编码器。

    The backbone is a ViT-Base/16 (``embed_dim=768``, ``patch_size=16``).
    ``out_channels = [dim/8, dim/4, dim/2, dim]`` (deepest LAST).
    """

    native_img_size: int = 224
    PATCH_SIZE = _ENDOFM_PATCH_SIZE
    EMBED_DIM = _ENDOFM_EMBED_DIM

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

        # ---- Backbone — timm ViT-B/16 (DINO-style) ---- / ---- backbone ----
        import timm
        self.backbone = load_with_ssl_fallback(
            timm.create_model, _ENDOFM_TIMM_NAME,
            pretrained=pretrained and pretrained_path is None,
            num_classes=0,
            img_size=img_size,
            in_chans=3,
            dynamic_img_size=True,
        )
        self._backbone_name = PRIMARY_BACKBONE_NAME

        # ---- 可选 local checkpoint ( Endo-FM format ) ---- / ---- optional local checkpoint ----
        if pretrained_path:
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                if isinstance(state, dict):
                    for key in ("state_dict", "model", "model_state_dict",
                                "teacher", "student"):
                        if key in state and isinstance(state[key], dict):
                            state = state[key]
                            break
                # Strip common prefixes for Endo-FM / DINO checkpoints
                if isinstance(state, dict):
                    cleaned = {}
                    for k, v in state.items():
                        nk = k
                        for pref in ("module.", "backbone.", "encoder.",
                                     "model."):
                            if nk.startswith(pref):
                                nk = nk[len(pref):]
                                break
                        cleaned[nk] = v
                    state = cleaned
                msg = self.backbone.load_state_dict(state, strict=False)
                warnings.warn(
                    f"[endo_fm] loaded Endo-FM weights from "
                    f"'{pretrained_path}': {msg}"
                )
            except Exception as e:
                warnings.warn(
                    f"[endo_fm] failed to load checkpoint "
                    f"'{pretrained_path}': {e}"
                )

        # ---- Backbone introspection ---- / ---- backbone introspection ----
        ps = self.backbone.patch_embed.patch_size
        if isinstance(ps, (tuple, list)):
            self.patch_size = int(ps[0])
        else:
            self.patch_size = int(ps)
        self.embed_dim = int(getattr(self.backbone, "embed_dim", _ENDOFM_EMBED_DIM))
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

        # 填充到 patch_size 的倍数 / Pad to multiple of patch_size
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
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
