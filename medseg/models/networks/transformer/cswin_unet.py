"""CSWin-UNet: Transformer UNet with Cross-Shaped Windows.

Uses CSWin self-attention (cross-shaped window) in a U-Net architecture
for efficient medical image segmentation.

Reference:
    Liu et al., "CSWin-UNet: Transformer UNet with Cross-Shaped Windows
    for Medical Image Segmentation", Information Fusion 2024.
    arXiv: 2407.18070. DOI: 10.1016/j.inffus.2024.102634.
    https://github.com/eatbeanss/CSWin-UNet

Key components:
    - Cross-shaped window attention (horizontal + vertical strips)
    - U-shaped encoder-decoder with skip connections
    - Merge blocks for multi-scale feature aggregation
"""
# Source: https://github.com/eatbeanss/CSWin-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class _CrossWindowAttention(nn.Module):
    """Cross-shaped window self-attention (CSWin).

    Splits heads into two equal groups:
    - First half: horizontal strips (``strip_size`` rows × full width)
    - Second half: vertical strips (full height × ``strip_size`` columns)

    Self-attention is computed independently within each strip, then the
    horizontal and vertical outputs are concatenated along the head axis.
    """

    def __init__(self, dim, num_heads=4, strip_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.strip_size = strip_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def _strip_attention(self, q, k, v, strip_size):
        """Compute attention within strips along the first spatial axis.

        Args:
            q, k, v: (B, nh, axis, other, head_dim)
            strip_size: rows/cols per strip

        Returns:
            (B, nh, axis, other, head_dim)
        """
        B, nh, axis, other, hd = q.shape
        # Pad axis to a multiple of strip_size
        pad = (strip_size - axis % strip_size) % strip_size
        if pad > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))

        axis_p = axis + pad
        ns = axis_p // strip_size  # number of strips

        # (B, nh, ns, strip_size, other, hd) → (B*nh*ns, strip_size*other, hd)
        q = q.reshape(B, nh, ns, strip_size, other, hd).reshape(
            B * nh * ns, strip_size * other, hd)
        k = k.reshape(B, nh, ns, strip_size, other, hd).reshape(
            B * nh * ns, strip_size * other, hd)
        v = v.reshape(B, nh, ns, strip_size, other, hd).reshape(
            B * nh * ns, strip_size * other, hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B*nh*ns, strip_size*other, hd)

        # Reshape back and remove padding
        out = out.reshape(B, nh, ns, strip_size, other, hd).reshape(
            B, nh, axis_p, other, hd)
        if pad > 0:
            out = out[:, :, :axis, :, :]
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        # (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()

        # QKV projection
        head_dim = C // self.num_heads
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, HW, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, nh, HW, hd)

        # Reshape to spatial: (B, nh, H, W, hd)
        q = q.reshape(B, self.num_heads, H, W, head_dim)
        k = k.reshape(B, self.num_heads, H, W, head_dim)
        v = v.reshape(B, self.num_heads, H, W, head_dim)

        # Split heads: first half for horizontal, second half for vertical
        nh = self.num_heads // 2
        q_h, q_v = q[:, :nh], q[:, nh:]
        k_h, k_v = k[:, :nh], k[:, nh:]
        v_h, v_v = v[:, :nh], v[:, nh:]

        # Horizontal strips: strip along H (strip_size rows × W cols)
        out_h = self._strip_attention(q_h, k_h, v_h, self.strip_size)

        # Vertical strips: strip along W (H rows × strip_size cols)
        # Swap H and W axes to reuse _strip_attention
        q_v = q_v.permute(0, 1, 3, 2, 4)  # (B, nh, W, H, hd)
        k_v = k_v.permute(0, 1, 3, 2, 4)
        v_v = v_v.permute(0, 1, 3, 2, 4)
        out_v = self._strip_attention(q_v, k_v, v_v, self.strip_size)
        out_v = out_v.permute(0, 1, 3, 2, 4)  # back to (B, nh, H, W, hd)

        # Concatenate along head dimension
        out = torch.cat([out_h, out_v], dim=1)  # (B, num_heads, H, W, hd)

        # Reshape to (B, H, W, C) and project
        out = out.reshape(B, H, W, C)
        out = self.proj(out)

        # Back to (B, C, H, W)
        return out.permute(0, 3, 1, 2).contiguous()


class _CSWinBlock(nn.Module):
    def __init__(self, dim, num_heads=4, strip_size=7, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _CrossWindowAttention(dim, num_heads, strip_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Pre-norm: LayerNorm before attention
        tokens = x.flatten(2).transpose(1, 2)            # (B, HW, C)
        normed = self.norm1(tokens).transpose(1, 2).view(B, C, H, W)
        tokens = tokens + self.attn(normed).flatten(2).transpose(1, 2)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W)


class _CSWinStage(nn.Module):
    def __init__(self, dim, num_heads, depth, strip_size=7, downsample=False):
        super().__init__()
        self.blocks = nn.Sequential(*[
            _CSWinBlock(dim, num_heads, strip_size) for _ in range(depth)
        ])
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(dim, dim * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(dim * 2),
            )

    def forward(self, x):
        x = self.blocks(x)
        out = x
        if self.downsample is not None:
            x = self.downsample(x)
            return out, x
        return out


class CSWinUNet(nn.Module):
    """CSWin-UNet: Cross-shaped Window Transformer UNet.
        CSWin-UNet: Cross-shaped 窗口 Transformer UNet。

    Args:
        in_channels: Input channels.
        num_classes: Segmentation classes.
        img_size: Input spatial size.
        embed_dim: Base embedding dimension.
        depths: Blocks per stage.
        num_heads: Attention heads per stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        embed_dim: int = 64,
        depths: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [2, 4, 8, 16]
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 4, 4, bias=False),
            nn.BatchNorm2d(dims[0]),
        )

        # 编码器 / Encoder
        self.enc_stages = nn.ModuleList()
        for i in range(len(depths)):
            ds = i < len(depths) - 1
            self.enc_stages.append(
                _CSWinStage(dims[i], num_heads[i], depths[i], downsample=ds)
            )

        # 解码 / Decoder
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.up_convs.append(nn.ConvTranspose2d(dims[i], dims[i - 1], 2, 2))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(dims[i - 1] * 2, dims[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(dims[i - 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(dims[i - 1], dims[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(dims[i - 1]),
                nn.ReLU(inplace=True),
            ))

        self.head = nn.Sequential(
            nn.Conv2d(dims[0], dims[0] // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0] // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]
        x = self.patch_embed(x)

        skips = []
        for i, stage in enumerate(self.enc_stages):
            if i < len(self.enc_stages) - 1:
                out, x = stage(x)
                skips.append(out)
            else:
                x = stage(x)

        for up, dec in zip(self.up_convs, self.dec_blocks):
            x = up(x)
            skip = skips.pop()
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(F.interpolate(x, size=(H_in, W_in), mode="bilinear", align_corners=False))
