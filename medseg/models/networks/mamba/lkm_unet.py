"""LKM-UNet: Large Kernel Vision Mamba UNet for Medical Image Segmentation.
    LKM-UNet: Large Kernel Vision Mamba UNet for Medical Image Segmentation。

Combines large kernel convolutions with real Mamba SSM (selective scan) for
capturing both local and global features in medical image segmentation.

Architecture (faithful to official code, adapted for standalone 2D use):
    - Encoder: stem conv + n_stages of [residual conv blocks (large kernel) +
      PiM (pixel-level SSM via SS2D) + PaM (patch-level SSM via SS2D with pooling)]
    - Decoder: standard UNet decoder with transposed convs + residual blocks
    - PiM: large-kernel depthwise conv → LN → SS2D (bidirectional via 4-direction scan)
    - PaM: AvgPool → LN → SS2D → Upsample (patch-level global modeling)

Reference:
    Wang et al., "LKM-UNet: Large Kernel Vision Mamba UNet for Medical Image
    Segmentation", MICCAI 2024. https://github.com/wjh892521292/LKM-UNet

Note: Official code uses mamba_ssm.Mamba (1D SSM) with bidirectional scanning.
This implementation uses SS2D (2D selective scan with 4 directions) from
vmunet_encoder, which provides equivalent selective scan functionality.
"""
# Source: https://github.com/wjh892521292/LKM-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple

from medseg.models.encoders.vmunet_encoder import SS2D


# ---------------------------------------------------------------------------
# PiM: Pixel-level SSM (BiPixelMambaLayer in official code)
# ---------------------------------------------------------------------------
class PiMBlock(nn.Module):
    """Pixel-level SSM block with SS2D selective scan.

    In the official code, this is BiPixelMambaLayer which uses bidirectional
    Mamba (forward + backward) on pixel-windowed tokens. Here we use SS2D
    which performs 4-directional selective scan (covering both forward and
    backward directions) on 2D feature maps directly.
    """

    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        # Convert to (B, H, W, C) for SS2D
        x_bhwc = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_bhwc)
        out = self.ss2d(x_norm)  # (B, H, W, C)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return out + x  # residual


# ---------------------------------------------------------------------------
# PaM: Patch-level SSM (BiWindowMambaLayer in official code)
# ---------------------------------------------------------------------------
class PaMBlock(nn.Module):
    """Patch-level SSM block with pooling and SS2D.

    In the official code, this is BiWindowMambaLayer which pools the feature
    map, applies bidirectional Mamba on pooled tokens, then upsamples back.
    """

    def __init__(self, dim, pool_size=2, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.dim = dim
        self.pool_size = pool_size
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        ps = self.pool_size
        # Pool to get patch-level tokens
        if H % ps == 0 and W % ps == 0 and ps > 1:
            x_pool = F.avg_pool2d(x, ps, stride=ps)
        else:
            x_pool = x
        Hp, Wp = x_pool.shape[2], x_pool.shape[3]
        # SS2D on pooled features
        x_bhwc = x_pool.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_bhwc)
        out = self.ss2d(x_norm)  # (B, Hp, Wp, C)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, C, Hp, Wp)
        # Upsample back
        if ps > 1 and (Hp != H or Wp != W):
            out = F.interpolate(out, size=(H, W), mode='nearest')
        return out + x  # residual


# ---------------------------------------------------------------------------
# Residual conv block (large kernel, from official nnU-Net framework)
# ---------------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    """Residual conv block with InstanceNorm + LeakyReLU (nnU-Net style)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True)
        self.norm1 = nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, 1, padding, bias=True)
        self.norm2 = nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=True) if in_ch != out_ch or stride != 1 else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


# ---------------------------------------------------------------------------
# LKM-UNet
# ---------------------------------------------------------------------------
class LKMUNet(nn.Module):
    """LKM-UNet: Large Kernel Mamba UNet.

    U-Net with large kernel convolutions and Mamba SSM (PiM + PaM) in encoder.
    Standard UNet decoder with transposed convs and residual conv blocks.
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dim=48, depths=None, kernel_sizes=None, **kwargs):
        super().__init__()
        depths = depths or [2, 2, 2, 2, 2]
        n_stages = len(depths)
        dims = [embed_dim * (2 ** i) for i in range(n_stages)]
        # Default large kernel sizes per stage (paper: [40,20,20,10,10,5,5] for 7 stages)
        if kernel_sizes is None:
            kernel_sizes = [7, 7, 5, 5, 3][:n_stages]
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages

        # Pool sizes for PaM (patch-level SSM): decreases with depth
        pool_sizes = [max(2 ** (n_stages - s - 1), 1) for s in range(n_stages)]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 3, 1, 1, bias=True),
            nn.InstanceNorm2d(dims[0], eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
        )

        # Encoder: stages of [residual conv blocks + PiM + PaM + downsample]
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for s in range(n_stages):
            enc_blocks = nn.Sequential(
                *[ResidualConvBlock(dims[s], dims[s], kernel_sizes[s]) for _ in range(depths[s])]
            )
            enc = nn.ModuleDict({
                'conv': enc_blocks,
                'pim': PiMBlock(dims[s]),
                'pam': PaMBlock(dims[s], pool_size=pool_sizes[s]),
            })
            self.encoders.append(enc)
            if s < n_stages - 1:
                self.downs.append(nn.Conv2d(dims[s], dims[s + 1], 3, 2, 1, bias=True))

        # Decoder: transposed conv + merge conv + residual conv blocks
        self.ups = nn.ModuleList()
        self.merges = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for s in range(n_stages - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(dims[s], dims[s - 1], 2, 2))
            # Merge concatenated skip (2*dims[s-1] -> dims[s-1])
            self.merges.append(nn.Sequential(
                nn.Conv2d(dims[s - 1] * 2, dims[s - 1], 1, bias=False),
                nn.InstanceNorm2d(dims[s - 1], eps=1e-5, affine=True),
                nn.LeakyReLU(inplace=True),
            ))
            dec_blocks = nn.Sequential(
                *[ResidualConvBlock(dims[s - 1], dims[s - 1], kernel_sizes[s - 1]) for _ in range(depths[s - 1])]
            )
            self.decoders.append(dec_blocks)

        # Output head
        self.head = nn.Conv2d(dims[0], num_classes, 1, bias=True)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.stem(x)
        skips = []
        for s, enc in enumerate(self.encoders):
            x = enc['conv'](x)
            x = enc['pim'](x)
            x = enc['pam'](x)
            if s < len(self.downs):
                skips.append(x)
                x = self.downs[s](x)

        for up, merge, dec in zip(self.ups, self.merges, self.decoders):
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
