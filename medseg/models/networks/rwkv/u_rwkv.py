"""U-RWKV: Medical Image Segmentation with RWKV Attention Mechanism.
    U-RWKV: 医学的 图像 分割 with RWKV 注意力 Mechanism。

Strict reimplementation of the official LoRA_4_5 architecture from:
  https://github.com/hbyecoding/U-RWKV  (MICCAI 2025)

Architecture (official LoRA_4_5):
  - 5-level Conv encoder (EncoderBlock: Conv+BN+GELU x2 + MaxPool)
  - BinaryOrientatedRWKV2D at stage 4 and 5 only (forward + reverse scan)
  - Dual decoder: segmentation decoder (s1-s5) + autoencoder decoder (a1-a5)
  - Attention maps (m1-m5) from autoencoder to gate segmentation features
  - Two outputs: segmentation + autoencoder reconstruction

Key components:
  - VRWKV_SpatialMix: q_shift + WKV with fancy initialization
  - VRWKV_ChannelMix: q_shift + FFN-like channel mixing
  - BinaryOrientatedRWKV2D: bidirectional RWKV (forward + transpose scan)
  - ResidualBlock with SE attention for decoder

WKV is computed by the unified dispatcher in :mod:`medseg.kernels.wkv` so this
architecture automatically uses the official Vision-RWKV CUDA op when running
on a GPU and falls back to a vectorised PyTorch implementation otherwise. Both
paths are autograd-differentiable.
"""
# Source: https://github.com/hbyecoding/U-RWKV

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from medseg.kernels.wkv import run_wkv as _run_wkv


# ---------------------------------------------------------------------------
# WKV computation (CUDA-accelerated when available, pure PyTorch otherwise)
# ---------------------------------------------------------------------------

def RUN_CUDA(B, T, C, w, u, k, v):
    """WKV entry-point dispatching to CUDA op or PyTorch fallback."""
    return _run_wkv(B, T, C, w, u, k, v)


# ---------------------------------------------------------------------------
# Q-Shift: spatial token shifting for multi-direction information flow
# ---------------------------------------------------------------------------

def q_shift(x, shift_pixel=1, gamma=0.25, patch_resolution=None):
    """Shift tokens in 4 directions for spatial mixing.

    x: (B, N, C) where N = H*W. Shifts C/4 channels in each direction.
    """
    B, N, C = x.shape
    if patch_resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            patch_resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError(f"Cannot infer patch_resolution from N={N}")

    H, W = patch_resolution
    x = x.transpose(1, 2).reshape(B, C, H, W)
    out = torch.zeros_like(x)
    g = int(C * gamma)

    out[:, 0:g, :, shift_pixel:W] = x[:, 0:g, :, 0:W-shift_pixel]
    out[:, g:2*g, :, 0:W-shift_pixel] = x[:, g:2*g, :, shift_pixel:W]
    out[:, 2*g:3*g, shift_pixel:H, :] = x[:, 2*g:3*g, 0:H-shift_pixel, :]
    out[:, 3*g:4*g, 0:H-shift_pixel, :] = x[:, 3*g:4*g, shift_pixel:H, :]
    out[:, 4*g:, ...] = x[:, 4*g:, ...]

    return out.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# VRWKV_SpatialMix: q_shift + WKV attention with fancy initialization
# ---------------------------------------------------------------------------

class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, init_mode='fancy', k_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode

        if shift_pixel > 0:
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self._init_weights(init_mode)

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(n_embd) if k_norm else None
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        # Scale init attributes (for compatibility)
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0
        self.value.scale_init = 1

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1) if self.n_layer > 1 else 0
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)

                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones(1, 1, self.n_embd))
            self.spatial_mix_v = nn.Parameter(torch.ones(1, 1, self.n_embd))
            self.spatial_mix_r = nn.Parameter(torch.ones(1, 1, self.n_embd))
        elif init_mode == 'global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones(1, 1, self.n_embd) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones(1, 1, self.n_embd) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones(1, 1, self.n_embd) * 0.5)
        else:
            raise NotImplementedError(f"Unknown init_mode: {init_mode}")

    def forward(self, x, patch_resolution=None):
        B, T, C = x.shape

        # Mix x with shifted version
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = xv = xr = x

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        rwkv = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv


# ---------------------------------------------------------------------------
# VRWKV_ChannelMix: q_shift + FFN-like channel mixing
# ---------------------------------------------------------------------------

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy', k_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.shift_pixel = shift_pixel

        if shift_pixel > 0:
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        self._init_weights(init_mode)

        hidden = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden, bias=False)
        self.key_norm = nn.LayerNorm(hidden) if k_norm else None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.value.scale_init = 1

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1) if self.n_layer > 1 else 0
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)

                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones(1, 1, self.n_embd))
            self.spatial_mix_r = nn.Parameter(torch.ones(1, 1, self.n_embd))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones(1, 1, self.n_embd) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones(1, 1, self.n_embd) * 0.5)
        else:
            raise NotImplementedError(f"Unknown init_mode: {init_mode}")

    def forward(self, x, patch_resolution=None):
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = xr = x

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(xr)) * kv
        return x


# ---------------------------------------------------------------------------
# VRWKV_Bottleneck: SpatialMix + ChannelMix with LayerNorm
# ---------------------------------------------------------------------------

class VRWKV_Bottleneck(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 drop_path=0., k_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd

        self.spatial_mix = VRWKV_SpatialMix(
            n_embd=n_embd, n_layer=n_layer, layer_id=layer_id,
            shift_mode=shift_mode, channel_gamma=channel_gamma,
            shift_pixel=shift_pixel, init_mode=init_mode, k_norm=k_norm
        )
        self.channel_mix = VRWKV_ChannelMix(
            n_embd=n_embd, n_layer=n_layer, layer_id=layer_id,
            shift_mode=shift_mode, channel_gamma=channel_gamma,
            shift_pixel=shift_pixel, hidden_rate=hidden_rate,
            init_mode=init_mode, k_norm=k_norm
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, patch_resolution=None):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            patch_resolution = (H, W)

        x = x + self.spatial_mix(self.ln1(x), patch_resolution)
        x = x + self.channel_mix(self.ln2(x), patch_resolution)
        return x


# ---------------------------------------------------------------------------
# BinaryOrientatedRWKV2D: bidirectional RWKV (forward + transpose scan)
# ---------------------------------------------------------------------------

class BinaryOrientatedRWKV2D(nn.Module):
    def __init__(self, n_embd, n_layer, shift_mode='q_shift', channel_gamma=1/4,
                 shift_pixel=1, hidden_rate=4, init_mode='fancy', drop_path=0., k_norm=True):
        super().__init__()
        self.rwkv_forward = VRWKV_Bottleneck(
            n_embd=n_embd, n_layer=n_layer, layer_id=0,
            shift_mode=shift_mode, channel_gamma=channel_gamma,
            shift_pixel=shift_pixel, hidden_rate=hidden_rate,
            init_mode=init_mode, drop_path=drop_path, k_norm=k_norm
        )
        self.rwkv_reverse = VRWKV_Bottleneck(
            n_embd=n_embd, n_layer=n_layer, layer_id=1,
            shift_mode=shift_mode, channel_gamma=channel_gamma,
            shift_pixel=shift_pixel, hidden_rate=hidden_rate,
            init_mode=init_mode, drop_path=drop_path, k_norm=k_norm
        )

    def forward(self, z):
        B, C, H, W = z.shape
        z_f = z.flatten(2).transpose(1, 2)  # Forward scan: (B, H*W, C)
        z_r = z.transpose(2, 3).flatten(2).transpose(1, 2)  # Reverse scan: (B, W*H, C)

        rwkv_f = self.rwkv_forward(z_f, (H, W))
        rwkv_r = self.rwkv_reverse(z_r, (W, H))

        fused = rwkv_f + rwkv_r
        if len(fused.shape) == 3:
            fused = fused.transpose(1, 2).view(B, C, H, W)
        return fused


# ---------------------------------------------------------------------------
# SE Layer
# ---------------------------------------------------------------------------

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ---------------------------------------------------------------------------
# EncoderBlock: Conv+BN+GELU x2 + MaxPool
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


# ---------------------------------------------------------------------------
# ResidualBlock with SE for decoder
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SELayer(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        x3 = self.bn3(self.conv3(x))
        out = self.relu(x2 + x3)
        out = self.se(out)
        return out


# ---------------------------------------------------------------------------
# DecoderBlock: ConvTranspose + ResidualBlocks
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(in_c + out_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, x, s):
        x = self.upsample(x)
        x = torch.cat([x, s], dim=1)
        x = self.r1(x)
        x = self.r2(x)
        return x


# ---------------------------------------------------------------------------
# URWKV: Official LoRA_4_5 architecture
# ---------------------------------------------------------------------------

class URWKV(nn.Module):
    """U-RWKV: Official LoRA_4_5 architecture from GitHub.

    Architecture:
      - 5-level Conv encoder (e1-e5)
      - BinaryOrientatedRWKV2D at stage 4 and 5 only
      - Dual decoder: segmentation (s1-s5) + autoencoder (a1-a5)
      - Attention maps (m1-m5) from autoencoder to gate segmentation
      - Two outputs: segmentation + autoencoder reconstruction

    Args:
        in_channels: Input channels (default 3).
        num_classes: Output segmentation classes (default 2).
        img_size: Input spatial size (default 224, must be divisible by 16).
        dims: Channel dimensions for each level [e1, e2, e3, e4, e5].
        n_layer: Number of RWKV layers in BinaryOrientatedRWKV2D.
        deep_supervision: Not used in this architecture (kept for API compatibility).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 dims=None, n_layer=12, deep_supervision=False, **kwargs):
        super().__init__()
        if dims is None:
            dims = [12, 48, 96, 192, 384]  # Official default

        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # Shared Encoder (5 levels)
        self.e1 = EncoderBlock(in_channels, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])

        # BinaryOrientatedRWKV2D at stage 4 and 5 only
        self.Brwkv_4 = BinaryOrientatedRWKV2D(
            n_embd=dims[3], n_layer=n_layer,
            shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1,
            hidden_rate=4, init_mode='fancy', drop_path=0, k_norm=True
        )
        self.Brwkv_5 = BinaryOrientatedRWKV2D(
            n_embd=dims[4], n_layer=n_layer,
            shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1,
            hidden_rate=4, init_mode='fancy', drop_path=0, k_norm=True
        )

        # Decoder: Segmentation
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        # Decoder: Autoencoder
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)

        # Attention maps from autoencoder
        self.m1 = nn.Sequential(nn.Conv2d(dims[3], 1, kernel_size=1), nn.Sigmoid())
        self.m2 = nn.Sequential(nn.Conv2d(dims[2], 1, kernel_size=1), nn.Sigmoid())
        self.m3 = nn.Sequential(nn.Conv2d(dims[1], 1, kernel_size=1), nn.Sigmoid())
        self.m4 = nn.Sequential(nn.Conv2d(dims[0], 1, kernel_size=1), nn.Sigmoid())
        self.m5 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid())

        # Output heads
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1)  # Segmentation
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1)  # Autoencoder

    def forward(self, x):
        H, W = x.shape[-2:]

        # Encoder
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x4 = self.Brwkv_4(x4)
        p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        x5 = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)

        # Decoder 1
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        # Decoder 2
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        # Decoder 3
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        # Decoder 4
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        # Decoder 5
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        # Output
        out1 = self.output1(x10)  # Segmentation
        out2 = self.output2(a5)   # Autoencoder

        # Upsample to input size if needed
        if out1.shape[-2:] != (H, W):
            out1 = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=False)
            out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)

        if self.training:
            return [out1, out2]  # Deep supervision style: seg + autoencoder
        return out1
