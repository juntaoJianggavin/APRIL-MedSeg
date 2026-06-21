"""WA-UKAN Encoder.
    WA-UKAN 编码器。

Standalone encoder extracted from
``medseg.models.networks.kan_mlp.wa_ukan.WAUKAN``.

Pipeline (default ``dims=(64, 128, 256, 512)``):
    in -> [1x1 conv if in_channels != 3] ->
    enc1 (Conv-BN-ReLU,  in -> 64)  -> MaxPool/2 -> t1 (C=64,  H/2)
    enc2 (Conv-BN-ReLU,  64 -> 128) -> MaxPool/2 -> t2 (C=128, H/4)
    enc3 (Conv-BN-ReLU,  128 -> 256) -> MaxPool/2 -> t3 (C=256, H/8)
    enc4 (Conv-BN-ReLU,  256 -> 512) -> MaxPool/2 -> t4 (C=512, H/16)
    patch_embed (stride 2) + 3x WA-KAN Embedding + LN ->  t5 (C=512, H/32)

Returns 5 multi-scale features ordered shallow -> deep, deepest LAST.
Inputs should be divisible by 32.
"""
# Reference: Liu & Wang, "WA-UKAN: Wavelet-Enhanced Attention Kolmogorov-Arnold
# Networks for medical image segmentation", Signal Processing: Image
# Communication, 2026. DOI: 10.1016/j.image.2026.117534

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.networks.kan_mlp.wa_ukan import (
    _ConvBlock,
    _PatchEmbed,
    _WAKANEmbedding,
)


@ENCODER_REGISTRY.register("wa_ukan")
class WAUKANEncoder(nn.Module):
    """WA-UKAN encoder.
        WA-UKAN 编码器。

    Returns 5 multi-scale features (shallow -> deep, deepest LAST):
        [t1 @ H/2, t2 @ H/4, t3 @ H/8, t4 @ H/16, t5 @ H/32]
    with channel counts equal to ``dims`` (default ``[64, 128, 256, 512, 512]``).

    Args:
        in_channels: Number of input channels. If != 3, a 1x1 stem maps to 3.
        img_size: Reference spatial resolution (forward reads true H/W).
        pretrained: Unused (no public weights); kept for interface parity.
        dims: 4 encoder channel dimensions + bottleneck dim.
        wavelet: Wavelet kernel name (``morlet``, ``mexican_hat``, ``dog``,
            ``shannon``). Default ``morlet`` per the paper.
        drop_rate: Dropout in WA-KAN blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = False,
        dims=(64, 128, 256, 512),
        wavelet: str = "morlet",
        drop_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        dims = tuple(dims)
        assert len(dims) == 4, "dims must have 4 entries (4-level encoder)"
        self.dims = dims
        self.img_size = img_size
        c0, c1, c2, c3 = dims

        # Optional 1x1 stem for non-RGB inputs
        if in_channels != 3:
            self.in_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            stem_in = 3
        else:
            self.in_proj = nn.Identity()
            stem_in = in_channels

        # ---- Encoder: Conv-BN-ReLU + MaxPool (Sec 3.2.1) ----
        self.enc1 = _ConvBlock(stem_in, c0)
        self.enc2 = _ConvBlock(c0, c1)
        self.enc3 = _ConvBlock(c1, c2)
        self.enc4 = _ConvBlock(c2, c3)
        self.pool = nn.MaxPool2d(2)

        # ---- Bottleneck: WA-KAN Embedding (Sec 3.2.4) ----
        self.patch_embed = _PatchEmbed(c3, c3, patch_size=3, stride=2)
        self.wakan_embed = nn.ModuleList([
            _WAKANEmbedding(c3, wavelet, drop_rate) for _ in range(3)
        ])

        # out_channels: 4 encoder levels + 1 bottleneck level
        self.out_channels: List[int] = list(dims) + [c3]

        self._pretrained_requested = bool(pretrained)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.in_proj(x)
        B = x.shape[0]

        # ---- Conv encoder stages ----
        e1 = self.enc1(x)                           # (c0, H)
        e2 = self.enc2(self.pool(e1))               # (c1, H/2)
        e3 = self.enc3(self.pool(e2))               # (c2, H/4)
        e4 = self.enc4(self.pool(e3))               # (c3, H/8)

        # ---- WA-KAN Embedding bottleneck ----
        tokens, Hb, Wb = self.patch_embed(e4)       # (B, N, c3), H/16
        for blk in self.wakan_embed:
            tokens = blk(tokens, Hb, Wb)
        e5 = tokens.reshape(B, Hb, Wb, -1).permute(0, 3, 1, 2)  # (c3, H/16)

        return [e1, e2, e3, e4, e5]


__all__ = ["WAUKANEncoder"]
