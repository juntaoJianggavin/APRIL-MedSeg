"""CheXzero foundation-model encoder (chest X-ray CLIP ViT-B/32).
    CheXzero foundation-model 编码器。

Reference:
    Tiu et al., "Expert-level detection of pathologies from unannotated
    chest X-ray images via self-supervised learning", Nat. Biomed. Eng. 2022.
    https://github.com/rajpurkarlab/CheXzero

CheXzero is a CLIP ViT-B/32 model fine-tuned on MIMIC-CXR chest X-rays via
self-supervised contrastive learning. The backbone is a ViT-Base/32
(``embed_dim=768``, ``patch_size=32``, 12 layers).

``pretrained=True`` loads timm CLIP ViT-B/32 weights (OpenAI CLIP).  To use
the actual CheXzero checkpoint, download it from the GitHub repo and pass
via ``pretrained_path``.  The checkpoint is in OpenAI CLIP format; a key
conversion helper is included.

Registered as ``"chexzero"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://github.com/rajpurkarlab/CheXzero

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
_CHEXZERO_EMBED_DIM = 768
_CHEXZERO_PATCH_SIZE = 32
_CHEXZERO_TIMM_NAME = "vit_base_patch32_clip_224"

PRIMARY_BACKBONE_NAME = _CHEXZERO_TIMM_NAME


def _convert_clip_visual_to_timm(state_dict: dict) -> dict:
    """Convert OpenAI CLIP visual-encoder keys to timm ViT format.
        转换 OpenAI CLIP 视觉编码器 keys to timm ViT format.

    CheXzero checkpoints store the full CLIP model (vision + text).  This
    helper extracts only the ``visual.*`` keys and remaps them to the timm
    ViT naming convention so they can be loaded via ``load_state_dict``.
    """
    converted = {}
    for k, v in state_dict.items():
        if not k.startswith("visual."):
            continue
        nk = k
        # Skip the CLIP projection (not needed for dense feature extraction)
        if "visual.proj" in k:
            continue
        nk = nk.replace("visual.conv1.", "patch_embed.proj.")
        nk = nk.replace("visual.class_embedding", "cls_token")
        nk = nk.replace("visual.positional_embedding", "pos_embed")
        nk = nk.replace("visual.ln_pre.", "norm_pre.")
        nk = nk.replace("visual.ln_post.", "norm.")
        nk = nk.replace("visual.transformer.resblocks.", "blocks.")
        nk = nk.replace(".attn.in_proj_weight", ".attn.qkv.weight")
        nk = nk.replace(".attn.in_proj_bias", ".attn.qkv.bias")
        nk = nk.replace(".attn.out_proj.", ".attn.proj.")
        nk = nk.replace(".ln_1.", ".norm1.")
        nk = nk.replace(".ln_2.", ".norm2.")
        nk = nk.replace(".mlp.c_fc.", ".mlp.fc1.")
        nk = nk.replace(".mlp.c_proj.", ".mlp.fc2.")
        converted[nk] = v
    return converted


@ENCODER_REGISTRY.register("chexzero")
class CheXzeroEncoder(BaseFoundationEncoder):
    """CheXzero (chest X-ray CLIP ViT-B/32) encoder with DPT-style output.
        CheXzero (胸部 X-ray CLIP ViT-B/32) 编码器。

    The backbone is a ViT-Base/32 (``embed_dim=768``, ``patch_size=32``).
    ``out_channels = [dim/8, dim/4, dim/2, dim]`` (deepest LAST).

    Parameters
    ----------
    in_channels : int
        Number of input image channels (default 3).
    img_size : int
        Reference spatial size (default 224, unused at runtime).
    pretrained : bool
        Load timm CLIP ViT-B/32 pretrained weights.
    pretrained_path : Optional[str]
        Path to a CheXzero checkpoint (OpenAI CLIP format).  Keys are
        automatically converted from ``visual.*`` to timm ViT naming.
    """

    native_img_size: int = 224
    PATCH_SIZE = _CHEXZERO_PATCH_SIZE
    EMBED_DIM = _CHEXZERO_EMBED_DIM

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

        # ---- Backbone — timm ViT-B/32 CLIP ---- / ---- backbone ----
        import timm
        self.backbone = load_with_ssl_fallback(
            timm.create_model, _CHEXZERO_TIMM_NAME,
            pretrained=pretrained and pretrained_path is None,
            num_classes=0,
            img_size=img_size,
            in_chans=3,
            dynamic_img_size=True,
        )
        self._backbone_name = PRIMARY_BACKBONE_NAME

        # ---- 可选 local checkpoint ( CheXzero format ) ---- / ---- optional local checkpoint ----
        if pretrained_path:
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                if isinstance(state, dict):
                    for key in ("state_dict", "model", "model_state_dict"):
                        if key in state and isinstance(state[key], dict):
                            state = state[key]
                            break
                # Convert CLIP visual keys to timm ViT format
                converted = _convert_clip_visual_to_timm(state)
                if converted:
                    msg = self.backbone.load_state_dict(converted, strict=False)
                    warnings.warn(
                        f"[chexzero] loaded CheXzero weights from "
                        f"'{pretrained_path}': {msg}"
                    )
                else:
                    # Maybe already in timm format
                    msg = self.backbone.load_state_dict(state, strict=False)
                    warnings.warn(
                        f"[chexzero] loaded weights from '{pretrained_path}' "
                        f"(no CLIP key conversion needed): {msg}"
                    )
            except Exception as e:
                warnings.warn(
                    f"[chexzero] failed to load checkpoint "
                    f"'{pretrained_path}': {e}"
                )

        # ---- Backbone introspection ---- / ---- backbone introspection ----
        ps = self.backbone.patch_embed.patch_size
        if isinstance(ps, (tuple, list)):
            self.patch_size = int(ps[0])
        else:
            self.patch_size = int(ps)
        self.embed_dim = int(getattr(self.backbone, "embed_dim", _CHEXZERO_EMBED_DIM))
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
