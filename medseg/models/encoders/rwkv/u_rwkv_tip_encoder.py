"""U-RWKV (TIP) encoder: 4 ConvBlock+RWKV levels + bottleneck level.
    U-RWKV (TIP) 编码器。

Extracted from ``medseg.models.networks.rwkv.u_rwkv_tip`` (U-RWKV, IEEE TIP
2026 reimplementation). Wraps the standard U-Net encoder with post-conv RWKV
attention and exposes 5 multi-scale feature maps consumed by a decoder.

Each level:
  - ConvBlock (double Conv-BN-ReLU)
  - ``n`` RWKV attention blocks (GroupNorm → SpatialMix/OmniShift → ChannelMix)
  - MaxPool2d for downsampling (except at bottleneck)

WKV is dispatched through :func:`medseg.kernels.wkv.run_wkv` (CUDA op on GPU,
pure-PyTorch fallback on CPU).
"""
# Source: https://github.com/hbyecoding/U-RWKV (IEEE TIP 2026)

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.kernels.wkv import run_wkv as _run_wkv


# ---------------------------------------------------------------------------
# WKV entry point
# ---------------------------------------------------------------------------

def _wkv(B, T, C, w, u, k, v):
    return _run_wkv(B, T, C, w, u, k, v)


# ---------------------------------------------------------------------------
# OmniShift: 多尺度 depth-wise conv / OmniShift: multi-scale depth-wise conv
# ---------------------------------------------------------------------------

class _OmniShift(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.conv5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.alpha = nn.Parameter(torch.ones(3))

    def forward(self, x):
        a = F.softmax(self.alpha, dim=0)
        return a[0] * self.conv1(x) + a[1] * self.conv3(x) + a[2] * self.conv5(x)


# ---------------------------------------------------------------------------
# RWKV 空间的 Mix with OmniShift ( TIP variant ) / RWKV Spatial Mix with OmniShift (TIP variant)
# ---------------------------------------------------------------------------

class _SpatialMixTIP(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_pixel=1):
        super().__init__()
        self.n_embd = n_embd
        self.layer_id = layer_id

        ratio_0 = layer_id / max(n_layer - 1, 1)
        ratio_1 = 1.0 - layer_id / max(n_layer, 1)

        decay_speed = torch.ones(n_embd)
        for h in range(n_embd):
            decay_speed[h] = -5 + 8 * (h / max(n_embd - 1, 1)) ** (
                0.7 + 1.3 * ratio_0)
        self.spatial_decay = nn.Parameter(decay_speed)

        zigzag = torch.tensor(
            [(i + 1) % 3 - 1 for i in range(n_embd)], dtype=torch.float32) * 0.5
        self.spatial_first = nn.Parameter(
            torch.ones(n_embd) * math.log(0.3) + zigzag)

        x = torch.ones(1, 1, n_embd)
        for i in range(n_embd):
            x[0, 0, i] = i / n_embd
        self.mix_k = nn.Parameter(torch.pow(x, ratio_1))
        self.mix_v = nn.Parameter(torch.pow(x, ratio_1) + 0.3 * ratio_0)
        self.mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1))

        self.shift_pixel = shift_pixel
        self.omni = _OmniShift(n_embd)

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)

    def _shift(self, x, B, H, W):
        C = x.shape[1]
        g = C // 4
        out = torch.zeros_like(x)
        s = self.shift_pixel
        out[:, 0:g, :, s:W] = x[:, 0:g, :, 0:W - s]
        out[:, g:2 * g, :, 0:W - s] = x[:, g:2 * g, :, s:W]
        out[:, 2 * g:3 * g, s:H, :] = x[:, 2 * g:3 * g, 0:H - s, :]
        out[:, 3 * g:4 * g, 0:H - s, :] = x[:, 3 * g:4 * g, s:H, :]
        out[:, 4 * g:] = x[:, 4 * g:]
        return out

    def forward(self, x_2d):
        B, C, H, W = x_2d.shape
        x = self._shift(x_2d, B, H, W)

        x_seq = x_2d.flatten(2).transpose(1, 2)
        x_sh = x.flatten(2).transpose(1, 2)

        xk = x_seq * self.mix_k + x_sh * (1 - self.mix_k)
        xv = x_seq * self.mix_v + x_sh * (1 - self.mix_v)
        xr = x_seq * self.mix_r + x_sh * (1 - self.mix_r)

        k = self.key_norm(self.key(xk))
        v = self.value(xv)
        r = self.receptance(xr)

        rwkv_out = _wkv(B, H * W, C,
                        self.spatial_decay.float(),
                        self.spatial_first.float(),
                        k.float(), v.float()).to(x_2d.dtype)

        out = self.output(torch.sigmoid(r) * rwkv_out)
        return out.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# RWKV 通道 Mix with OmniShift ( TIP variant ) / RWKV Channel Mix with OmniShift (TIP variant)
# ---------------------------------------------------------------------------

class _ChannelMixTIP(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_ratio=4, shift_pixel=1):
        super().__init__()
        hidden = int(n_embd * hidden_ratio)
        ratio_1 = 1.0 - layer_id / max(n_layer, 1)

        x = torch.ones(1, 1, n_embd)
        for i in range(n_embd):
            x[0, 0, i] = i / n_embd
        self.mix_k = nn.Parameter(torch.pow(x, ratio_1))
        self.mix_r = nn.Parameter(torch.pow(x, ratio_1))

        self.shift_pixel = shift_pixel
        self.omni = _OmniShift(n_embd)

        self.key = nn.Linear(n_embd, hidden, bias=False)
        self.value = nn.Linear(hidden, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x_2d):
        B, C, H, W = x_2d.shape
        x_sh = self.omni(x_2d)

        x_seq = x_2d.flatten(2).transpose(1, 2)
        x_sh_seq = x_sh.flatten(2).transpose(1, 2)

        xk = x_seq * self.mix_k + x_sh_seq * (1 - self.mix_k)
        xr = x_seq * self.mix_r + x_sh_seq * (1 - self.mix_r)

        k = torch.relu(self.key(xk)) ** 2
        kv = self.value(k)
        out = torch.sigmoid(self.receptance(xr)) * kv
        return out.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# RWKV 注意力 块 ( TIP: GroupNorm → 空间的 → LN → 通道, on 2D ) / RWKV Attention Block (TIP: GroupNorm → Spatial → LN → Channel, on 2D)
# ---------------------------------------------------------------------------

class _RWKVAttentionBlock(nn.Module):
    def __init__(self, channels, n_layer, layer_id, shift_pixel=1):
        super().__init__()
        self.ln1 = nn.GroupNorm(min(32, channels), channels)
        self.ln2 = nn.GroupNorm(min(32, channels), channels)
        self.spatial = _SpatialMixTIP(channels, n_layer, layer_id,
                                       shift_pixel=shift_pixel)
        self.channel = _ChannelMixTIP(channels, n_layer, layer_id,
                                       shift_pixel=shift_pixel)

    def forward(self, x):
        x = x + self.spatial(self.ln1(x))
        x = x + self.channel(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Conv 编码器 / Conv Encoder Block (standard double conv)
# Conv 编码器 Block (standard double conv)
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# 编码器 Level: ConvBlock + RWKV 注意力 / Encoder Level: ConvBlock + RWKV attention
# 编码器 Level: ConvBlock + RWKV 注意力
# ---------------------------------------------------------------------------

class _EncoderLevel(nn.Module):
    def __init__(self, in_ch, out_ch, n_rwkv, total_rwkv_layers,
                 layer_offset, shift_pixel=1):
        super().__init__()
        self.conv = _ConvBlock(in_ch, out_ch)
        self.rwkv_blocks = nn.ModuleList()
        for i in range(n_rwkv):
            self.rwkv_blocks.append(
                _RWKVAttentionBlock(out_ch, total_rwkv_layers,
                                    layer_offset + i, shift_pixel))

    def forward(self, x):
        x = self.conv(x)
        for blk in self.rwkv_blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# U-RWKV (TIP) 编码器 / U-RWKV (TIP) Encoder
# U-RWKV (TIP) 编码器
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("u_rwkv_tip")
class URWKVTIPEncoder(nn.Module):
    """U-RWKV (TIP) encoder: 4 ConvBlock+RWKV levels + bottleneck level.
        U-RWKV (TIP) 编码器。

    MaxPool downsampling between levels gives output strides [1, 2, 4, 8, 16]
    relative to the input.  Default ``embed_dims = [32, 64, 128, 256, 512]``
    and ``rwkv_depths = [1, 1, 1, 1, 2]``.

    The 5 returned feature maps are ordered shallowest-first / deepest-last
    (framework convention).  The deepest map is the bottleneck.

    Args:
        in_channels: Input channels.
        img_size: Nominal input size (unused at runtime).
        pretrained: Accepted for interface uniformity; no published weights.
        embed_dims / rwkv_depths / shift_pixel: Architectural knobs.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = False,
                 embed_dims: List[int] = None,
                 rwkv_depths: List[int] = None,
                 shift_pixel: int = 1, **kwargs):
        super().__init__()
        if embed_dims is None:
            embed_dims = [32, 64, 128, 256, 512]
        if rwkv_depths is None:
            rwkv_depths = [1, 1, 1, 1, 2]
        assert len(embed_dims) == len(rwkv_depths), \
            "embed_dims and rwkv_depths must have equal length"

        self.in_channels = in_channels
        self.img_size = img_size
        self._embed_dims = list(embed_dims)

        total_rwkv = sum(rwkv_depths)
        pool = nn.MaxPool2d(2, 2)

        # 4 编码器 / 4 encoder levels + 1 bottleneck
        # 4 编码器 levels + 1 瓶颈层
        self.levels = nn.ModuleList()
        layer_offset = 0
        for i in range(len(embed_dims)):
            in_ch = in_channels if i == 0 else embed_dims[i - 1]
            self.levels.append(_EncoderLevel(
                in_ch, embed_dims[i], rwkv_depths[i], total_rwkv,
                layer_offset, shift_pixel))
            layer_offset += rwkv_depths[i]

        self.pools = nn.ModuleList([pool] * (len(embed_dims) - 1))

        # Multi-scale 通道 dims ( deepest LAST ) / Multi-scale channel dims (deepest LAST)
        # 多-scale channel dims (deepest LAST)
        self.out_channels = list(embed_dims)

        if pretrained:
            import warnings
            warnings.warn(
                "URWKVTIPEncoder has no published pretrained weights; "
                "using random init.")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        for i, level in enumerate(self.levels):
            if i > 0:
                x = self.pools[i - 1](x)
            x = level(x)
            feats.append(x)
        return feats
