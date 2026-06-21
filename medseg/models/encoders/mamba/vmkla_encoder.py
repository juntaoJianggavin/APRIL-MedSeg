"""VMKLA-UNet encoder.
    VMKLA-UNet 编码器。

Stride-4 patch-embed stem followed by 4 hierarchical stages of VSS blocks
(SS2D 2D selective scan). Stages 0/1/2 are each followed by a strided 3x3
downsample (stride-2), producing a 4-level feature pyramid at strides
4 / 8 / 16 / 32.

Reference:
    VMKLA-UNet: Vision Mamba with KAN Linear Attention U-Net for
    Medical Image Segmentation. Nature Scientific Reports 2025.
    DOI: 10.1038/s41598-025-97397-2
"""
# Implemented from paper formulas; no official code released.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.vmunet_encoder import SS2D


def _load_with_ssl_fallback(load_fn, *args, **kwargs):
    """Try a download/load, falling back to unverified SSL, then random init."""
    import ssl
    import warnings
    try:
        return load_fn(*args, **kwargs)
    except Exception as e1:
        prev = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            return load_fn(*args, **kwargs)
        except Exception as e2:
            warnings.warn(f"Pretrained download failed ({e2}); using random init.")
            return load_fn(*args, **{**kwargs, 'pretrained': False})
        finally:
            ssl._create_default_https_context = prev


class VSSBlock(nn.Module):
    """Visual State Space block with SS2D selective scan (Eq. 5).

    E=Linear(x) -> E1=SiLU(Conv3x3(E)) -> S1=LayerNorm(SS2D(E1))
    -> S2=SiLU(E) -> Y=S1*S2
    """

    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.ss2d = SS2D(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        x_bhwc = x.permute(0, 2, 3, 1)

        E = self.linear(x_bhwc)
        E_bchw = E.permute(0, 3, 1, 2)
        E1 = self.act(self.conv3x3(E_bchw))
        E1_bhwc = E1.permute(0, 2, 3, 1)

        S1 = self.norm(self.ss2d(E1_bhwc))
        S2 = self.act(E)
        Y = S1 * S2

        return Y.permute(0, 3, 1, 2).contiguous()


@ENCODER_REGISTRY.register("vmkla")
class VMKLAUNetEncoder(nn.Module):
    """VMKLA-UNet encoder.
        VMKLA-UNet 编码器。

    Architecture:
        Conv 4x4 stride-4 stem -> stage_0 (depth=2, C=embed_dim)
        -> Conv 3x3 stride-2 -> stage_1 (depth=2, C=2*embed_dim)
        -> Conv 3x3 stride-2 -> stage_2 (depth=6, C=4*embed_dim)
        -> Conv 3x3 stride-2 -> stage_3 (depth=2, C=8*embed_dim)

    Returns 4 multi-scale feature maps in (B, C, H, W) at strides 4 / 8 / 16 / 32.
    The deepest (stride-32) map is LAST.

    Args:
        in_channels: input image channels. If != 3 a 1x1 conv stem maps to 3.
        img_size: nominal input resolution (informational only; spatial state
            is derived from the runtime tensor shape).
        pretrained: no public checkpoint is shipped for VMKLA-UNet; the flag is
            accepted for interface symmetry and results in random init.
        embed_dim: channels of the first stage.
        depths: per-stage block counts (must have length 4).
        pretrained_path: optional local checkpoint path.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = False,
        embed_dim: int = 64,
        depths: Optional[List[int]] = None,
        pretrained_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        depths = list(depths) if depths is not None else [2, 2, 6, 2]
        assert len(depths) == 4, "VMKLAUNetEncoder expects 4 stages."

        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.in_channels = in_channels
        self.img_size = img_size
        self.depths = tuple(depths)
        self.dims = tuple(dims)
        self.out_channels: List[int] = list(dims)

        # 1x1 通道 适配器 when the user feeds non-RGB inputs / 1x1 channel adapter when the user feeds non-RGB inputs.
        if in_channels != 3:
            self.input_stem = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
            stem_in = 3
        else:
            self.input_stem = nn.Identity()
            stem_in = in_channels

        # 步长 - 4 patch-embed 主干 / Stride-4 patch-embed stem.
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, dims[0], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(dims[0]),
        )

        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(depths)):
            self.enc.append(
                nn.Sequential(*[VSSBlock(dims[i]) for _ in range(depths[i])])
            )
            if i < len(depths) - 1:
                self.downs.append(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1))

        if pretrained:
            _load_with_ssl_fallback(self._maybe_load_pretrained, pretrained_path)

    # - - - - 预训练 loading - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - / ---- Pretrained loading -------------------------------------------------

    def _maybe_load_pretrained(self, pretrained_path: Optional[str] = None, **_):
        import warnings
        if not pretrained_path:
            warnings.warn(
                "VMKLAUNetEncoder: no pretrained_path provided; using random init."
            )
            return
        state = torch.load(pretrained_path, map_location="cpu")
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            if "state_dict" in state:
                state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            nk = k
            for prefix in ("encoder.", "backbone.", "module."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        msg = self.load_state_dict(cleaned, strict=False)
        print(f"VMKLAUNetEncoder loaded pretrained from {pretrained_path}: {msg}")

    # ---- Forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Args:
            x: (B, in_channels, H, W).
        Returns:
            List of 4 feature maps in (B, C, H, W) at strides 4/8/16/32.
            Deepest (stride-32) feature is LAST.
        """
        x = self.input_stem(x)
        x = self.stem(x)
        features: List[torch.Tensor] = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            features.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return features
