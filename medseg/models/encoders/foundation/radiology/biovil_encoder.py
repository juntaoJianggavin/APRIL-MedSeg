"""BioViL foundation-model encoder (chest X-ray ResNet-50).
    BioViL foundation-model 编码器。

Reference:
    Bannur et al., "Making the Most of Text Semantics to Improve Biomedical
    Vision-Language Processing", ECCV 2022.
    https://github.com/microsoft/hi-ml

BioViL uses a ResNet-50 image encoder pretrained on MIMIC-CXR chest X-rays.
Unlike ViT-based foundation encoders, this is a **CNN** backbone — multi-scale
features are extracted directly from ``layer1``–``layer4`` to form a 4-level
pyramid (deepest LAST), without a DPT head.

``out_channels = [256, 512, 1024, 2048]``.
``pretrained=True`` loads timm ImageNet-pretrained ResNet-50 weights.  To use
the actual BioViL checkpoint, pass via ``pretrained_path``.

Registered as ``"biovil"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://github.com/microsoft/hi-ml

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import (
    BaseFoundationEncoder, load_with_ssl_fallback,
)


# 架构 constants / Architecture constants.
_BIOVIL_TIMM_NAME = "resnet50"

# ResNet-50 stage output channels: layer1=256, layer2=512, layer3=1024, layer4=2048
_STAGE_CHANNELS = [256, 512, 1024, 2048]

PRIMARY_BACKBONE_NAME = _BIOVIL_TIMM_NAME


@ENCODER_REGISTRY.register("biovil")
class BioViLEncoder(BaseFoundationEncoder):
    """BioViL (chest X-ray ResNet-50) encoder with multi-scale CNN pyramid.
        BioViL (胸部 X-ray ResNet-50) 编码器。

    The backbone is a standard ResNet-50.  Features are extracted from
    ``layer1``–``layer4`` to form a 4-level pyramid at strides
    [H/4, H/8, H/16, H/32] (deepest LAST).

    .. note::
        This is a **CNN** backbone, so the DPT head (designed for ViT tokens)
        is NOT used.  Adapter injection and ``unfreeze_last_n_blocks`` are
        also not applicable (they target ViT ``blocks``/``layers``).

    Parameters
    ----------
    in_channels : int
        Number of input image channels (default 3).
    img_size : int
        Reference spatial size (unused, kept for API consistency).
    pretrained : bool
        Load timm ImageNet-pretrained ResNet-50 weights.
    pretrained_path : Optional[str]
        Path to a BioViL checkpoint.  Loaded with ``strict=False``.
    """

    native_img_size: int = 224

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

        # ---- Backbone — timm ResNet-50 ---- / ---- backbone ----
        import timm
        self.backbone = load_with_ssl_fallback(
            timm.create_model, _BIOVIL_TIMM_NAME,
            pretrained=pretrained and pretrained_path is None,
            num_classes=0,
            in_chans=3,
        )
        self._backbone_name = PRIMARY_BACKBONE_NAME

        # ---- 可选 local checkpoint ---- / ---- optional local checkpoint ----
        if pretrained_path:
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                if isinstance(state, dict):
                    for key in ("state_dict", "model", "model_state_dict"):
                        if key in state and isinstance(state[key], dict):
                            state = state[key]
                            break
                # Strip common prefixes for hi-ml / BioViL checkpoints
                if isinstance(state, dict):
                    cleaned = {}
                    for k, v in state.items():
                        nk = k
                        for pref in ("module.", "backbone.", "image_encoder.",
                                     "encoder."):
                            if nk.startswith(pref):
                                nk = nk[len(pref):]
                                break
                        cleaned[nk] = v
                    state = cleaned
                msg = self.backbone.load_state_dict(state, strict=False)
                warnings.warn(
                    f"[biovil] loaded weights from '{pretrained_path}': {msg}"
                )
            except Exception as e:
                warnings.warn(
                    f"[biovil] failed to load checkpoint "
                    f"'{pretrained_path}': {e}"
                )

        # ---- Output channels (deepest LAST) ---- / ---- output channels ----
        self.out_channels = list(_STAGE_CHANNELS)

        self._apply_freeze_policy()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_adapter(x)

        # 手动多尺度特征提取 / Manual multi-scale feature extraction
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        f1 = self.backbone.layer1(x)   # H/4,  256 ch
        f2 = self.backbone.layer2(f1)  # H/8,  512 ch
        f3 = self.backbone.layer3(f2)  # H/16, 1024 ch
        f4 = self.backbone.layer4(f3)  # H/32, 2048 ch

        return [f1, f2, f3, f4]  # deepest LAST
