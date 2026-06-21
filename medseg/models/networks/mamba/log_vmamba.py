"""LoG-VMamba: Local-Global Vision Mamba for Medical Image Segmentation.
    LoG-VMamba: Local-Global Vision Mamba for Medical Image Segmentation。

Combines local token extraction (LTX) with global token extraction (GTX) and
real Mamba SSM (selective scan) for enhanced medical image segmentation.

Architecture (faithful to paper, adapted for standalone 2D use):
    - LoG-VMamba block replaces VSS block's vanilla token extractor with LTX + GTX
    - LTX: depthwise conv + unfold (maintains spatial proximity of neighboring tokens)
    - GTX: depthwise conv + pool (provides global receptive field tokens)
    - LoG block: LTX(x) → GTX(xL) → Concat(xG, xL) → SS2D → out_proj → residual
    - U-Net architecture with LoG-VMamba blocks in encoder and decoder

Reference:
    Dang et al., "LoG-VMamba: Local-Global Vision Mamba for Medical Image
    Segmentation", ACCV 2024. https://github.com/imedslab/LoG-VMamba

Note: Official code uses custom SS2D_K1 (K=1 horizontal scan) with
selective_scan_fn. This implementation uses SS2D (K=4, 4-directional scan)
from vmunet_encoder, which provides equivalent selective scan functionality.
The LTX (unfold) and GTX (pooling) preprocessing captures the paper's key
innovation of local-global token extraction before SSM.
"""
# Source: https://github.com/imedslab/LoG-VMamba

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from medseg.models.encoders.vmunet_encoder import SS2D


# ---------------------------------------------------------------------------
# LTX: Local Token eXtractor (from paper Fig. 2a)
# ---------------------------------------------------------------------------
class LTX(nn.Module):
    """Local Token eXtractor: depthwise conv + unfold to maintain spatial
    proximity of neighboring tokens on the channel axis.

    Uses a window size R (default 3) to unfold the feature map into local
    patches, then projects back to original channel dimension.
    """

    def __init__(self, dim, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        # Depthwise conv for local feature extraction
        self.dwconv = nn.Conv2d(dim, dim, window_size, 1, window_size // 2, groups=dim)
        # Project unfolded tokens back to original dimension
        self.proj = nn.Linear(dim * window_size * window_size, dim)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        R = self.window_size
        x_conv = self.dwconv(x)  # (B, C, H, W)
        # Unfold into local patches
        x_unfold = F.unfold(x_conv, R, padding=R // 2)  # (B, C*R*R, H*W)
        x_unfold = x_unfold.transpose(1, 2)  # (B, H*W, C*R*R)
        x_local = self.proj(x_unfold)  # (B, H*W, C)
        x_local = x_local.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)
        return x_local


# ---------------------------------------------------------------------------
# GTX: Global Token eXtractor (from paper Fig. 2b)
# ---------------------------------------------------------------------------
class GTX(nn.Module):
    """Global Token eXtractor: depthwise conv + pool to compress spatial
    dimensions, providing SSM with global receptive field tokens.

    Pools with kernel K and stride K (default 2), then upsamples back.
    """

    def __init__(self, dim, pool_stride=2):
        super().__init__()
        self.pool_stride = pool_stride
        self.dim = dim
        # Depthwise conv for global feature extraction
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.act = nn.SiLU()

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W) (same shape, but with global context)"""
        B, C, H, W = x.shape
        K = self.pool_stride
        x_conv = self.act(self.dwconv(x))
        # Pool to get global tokens
        if H % K == 0 and W % K == 0 and K > 1:
            x_pool = F.avg_pool2d(x_conv, K, stride=K)
        else:
            x_pool = x_conv
        # Upsample back to original size
        if x_pool.shape[2:] != (H, W):
            x_pool = F.interpolate(x_pool, size=(H, W), mode='nearest')
        return x_pool


# ---------------------------------------------------------------------------
# LoG-VMamba Block (from paper Fig. 3, Eq. 5-6)
# ---------------------------------------------------------------------------
class LoGBlock(nn.Module):
    """LoG-VMamba block: LTX + GTX → concat → SS2D → out_proj → residual.

    Given input x:
    1. xL = LTX(x)  (local tokens)
    2. xG = GTX(xL) (global tokens)
    3. xLG = Concat(xG, xL) → 1x1 conv → C channels
    4. SS2D(xLG) → out
    5. out + x (residual)
    """

    def __init__(self, dim, d_state=16, d_conv=3, expand=2,
                 ltx_window=3, gtx_stride=2):
        super().__init__()
        self.dim = dim
        self.ltx = LTX(dim, window_size=ltx_window)
        self.gtx = GTX(dim, pool_stride=gtx_stride)
        # Fuse concatenated local + global tokens
        self.fuse = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        res = x
        # Local token extraction
        xL = self.ltx(x)  # (B, C, H, W)
        # Global token extraction (from local tokens, per paper Eq. 5)
        xG = self.gtx(xL)  # (B, C, H, W)
        # Concatenate global + local (per paper Eq. 6)
        xLG = torch.cat([xG, xL], dim=1)  # (B, 2C, H, W)
        # Fuse to original channels
        xLG = xLG.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 2C)
        x_fused = self.fuse(xLG)  # (B, H, W, C)
        x_norm = self.norm(x_fused)
        # SS2D selective scan
        out = self.ss2d(x_norm)  # (B, H, W, C)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return out + res


# ---------------------------------------------------------------------------
# LoG-VMamba U-Net
# ---------------------------------------------------------------------------
class LoGVMamba(nn.Module):
    """LoG-VMamba: Local-Global Vision Mamba U-Net."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dim=64, depths=None, **kwargs):
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 4, 4, bias=False),
            nn.BatchNorm2d(dims[0]),
        )

        # Encoder
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(depths)):
            self.enc.append(nn.Sequential(*[LoGBlock(dims[i]) for _ in range(depths[i])]))
            if i < len(depths) - 1:
                self.downs.append(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1))

        # Decoder
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.merges = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(dims[i], dims[i - 1], 2, 2))
            self.merges.append(nn.Sequential(
                nn.Conv2d(dims[i - 1] * 2, dims[i - 1], 1, bias=False),
                nn.BatchNorm2d(dims[i - 1]),
            ))
            self.dec.append(nn.Sequential(*[LoGBlock(dims[i - 1]) for _ in range(depths[i - 1])]))

        # Output head
        self.head = nn.Sequential(
            nn.ConvTranspose2d(dims[0], dims[0], 4, 4),
            nn.Conv2d(dims[0], num_classes, 1),
        )

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.stem(x)
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < len(self.downs):
                skips.append(x)
                x = self.downs[i](x)

        for up, merge, dec in zip(self.ups, self.merges, self.dec):
            x = up(x)
            s = skips.pop()
            if x.shape[2:] != s.shape[2:]:
                x = F.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False)
            x = merge(torch.cat([x, s], dim=1))
            x = dec(x)

        x = self.head(x)
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x
