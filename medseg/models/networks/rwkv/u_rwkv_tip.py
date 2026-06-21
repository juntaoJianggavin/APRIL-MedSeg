"""U-RWKV (TIP): Accurate and Efficient Volumetric Medical Image Segmentation via RWKV.
    U-RWKV ( TIP ): Accurate and 高效的 Volumetric 医学的 图像 分割 via RWKV。

中文: U-RWKV (TIP)：基于 RWKV 的精确高效体素医学图像分割。

2D adaptation of the volumetric segmentation network in:
  Cai et al., "U-RWKV: Accurate and Efficient Volumetric Medical Image
  Segmentation via RWKV", IEEE TIP 2026.
  Official code: https://github.com/hbyecoding/U-RWKV
  RWKV building blocks (OmniShift, spatial/channel mixing) adapted from
  Vision-RWKV / Restore-RWKV style 2D RWKV implementations.

Architecture: Standard U-Net encoder-decoder with RWKV attention blocks
applied after each convolutional encoder stage and at the bottleneck.
Unlike the MICCAI 2025 U-RWKV (which integrates RWKV *within* conv stages
and uses DARM/SASE), this variant applies RWKV as a post-hoc global
attention refinement on top of conventional conv feature extractors.

WKV is computed by the unified dispatcher in :mod:`medseg.kernels.wkv`.
"""
# Source: https://github.com/hbyecoding/U-RWKV (IEEE TIP 2026)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.kernels.wkv import run_wkv as _run_wkv


# ---------------------------------------------------------------------------
# WKV dispatcher
# ---------------------------------------------------------------------------

def _wkv(B, T, C, w, u, k, v):
    return _run_wkv(B, T, C, w, u, k, v)


# ---------------------------------------------------------------------------
# OmniShift: 多尺度 depth-wise conv ( reparameterizable ) / OmniShift: multi-scale depth-wise conv (reparameterizable)
# ---------------------------------------------------------------------------

class OmniShift(nn.Module):
    """Multi-scale depth-wise conv: 1×1 + 3×3 + 5×5 with learnable weights.
        Multi-scale depth-wise conv: 1 × 1 + 3 × 3 + 5 × 5 with learnable 权重。

    Can be reparameterized to a single 5×5 conv at inference time.
    """
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

class SpatialMixTIP(nn.Module):
    """RWKV 空间的 mixing with OmniShift for 多尺度 标记 awareness。
        RWKV spatial mixing with OmniShift for multi-scale token awareness."""
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
        self.omni = OmniShift(n_embd)

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)

    def _shift(self, x, B, H, W):
        """简单 4-direction 空间的 shift。
            Simple 4-direction spatial shift."""
        C = x.shape[1]
        g = C // 4
        out = torch.zeros_like(x)
        s = self.shift_pixel
        # right
        out[:, 0:g, :, s:W] = x[:, 0:g, :, 0:W - s]
        # left
        out[:, g:2*g, :, 0:W - s] = x[:, g:2*g, :, s:W]
        # down
        out[:, 2*g:3*g, s:H, :] = x[:, 2*g:3*g, 0:H - s, :]
        # up
        out[:, 3*g:4*g, 0:H - s, :] = x[:, 3*g:4*g, s:H, :]
        # no shift
        out[:, 4*g:] = x[:, 4*g:]
        return out

    def forward(self, x_2d):
        B, C, H, W = x_2d.shape
        x = self._shift(x_2d, B, H, W)  # spatial shift in 2D

        x_seq = x_2d.flatten(2).transpose(1, 2)   # (B, HW, C)
        x_sh  = x.flatten(2).transpose(1, 2)       # (B, HW, C)

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

class ChannelMixTIP(nn.Module):
    """RWKV 通道 mixing with OmniShift。
        RWKV channel mixing with OmniShift."""
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
        self.omni = OmniShift(n_embd)

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
# RWKV 注意力 块 ( TIP: LayerNorm → 空间的 → LN → 通道, on 2D ) / RWKV Attention Block (TIP: LayerNorm → Spatial → LN → Channel, on 2D)
# ---------------------------------------------------------------------------

class RWKVAttentionBlock(nn.Module):
    """Single RWKV 注意力 块 operating on 2D 特征图。
        Single RWKV attention block operating on 2D feature maps."""
    def __init__(self, channels, n_layer, layer_id, shift_pixel=1):
        super().__init__()
        self.ln1 = nn.GroupNorm(min(32, channels), channels)
        self.ln2 = nn.GroupNorm(min(32, channels), channels)
        self.spatial = SpatialMixTIP(channels, n_layer, layer_id,
                                     shift_pixel=shift_pixel)
        self.channel = ChannelMixTIP(channels, n_layer, layer_id,
                                     shift_pixel=shift_pixel)

    def forward(self, x):
        x = x + self.spatial(self.ln1(x))
        x = x + self.channel(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Conv 编码器 / Conv Encoder Block (standard double conv)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Double conv 块: Conv-BN-ReLU × 2。
        Double conv block: Conv-BN-ReLU × 2."""
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
# ---------------------------------------------------------------------------

class EncoderLevel(nn.Module):
    """Conv 特征 extraction followed by RWKV 注意力 refinement。
        Conv feature extraction followed by RWKV attention refinement."""
    def __init__(self, in_ch, out_ch, n_rwkv, total_rwkv_layers,
                 layer_offset, shift_pixel=1):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.rwkv_blocks = nn.ModuleList()
        for i in range(n_rwkv):
            self.rwkv_blocks.append(
                RWKVAttentionBlock(out_ch, total_rwkv_layers,
                                   layer_offset + i, shift_pixel))

    def forward(self, x):
        x = self.conv(x)
        for blk in self.rwkv_blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Decoder Block: ConvTranspose upsample + 跳跃连接 / Decoder Block: ConvTranspose upsample + skip concat + double conv
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Standard U-Net 解码器。
        Standard U-Net decoder: upsample + concat skip + double conv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2, bias=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-RWKV (TIP)
# ---------------------------------------------------------------------------

class URWKVTIP(nn.Module):
    """U-RWKV (TIP 2026): U-Net with post-conv RWKV attention blocks.
        U-RWKV ( TIP 2026 ): U-Net with post-conv RWKV 注意力 blocks。

中文: U-RWKV (TIP 2026)：标准 U-Net + 卷积后 RWKV 注意力模块。

    Architecture (2D adaptation of IEEE TIP 2026 volumetric network):
      - 4 encoder levels: ConvBlock + RWKV attention + MaxPool
      - Bottleneck: ConvBlock + RWKV attention
      - 4 decoder levels: ConvTranspose upsample + skip + ConvBlock
      - 1×1 conv segmentation head

    Key difference from MICCAI 2025 U-RWKV:
      - MICCAI: RWKV integrated within conv stages (ConvRWKV), DARM + SASE
      - TIP: Standard conv blocks with RWKV as post-hoc attention refinement

    Args:
        in_channels: Input channels (default 3).
        num_classes: Output segmentation classes (default 2).
        img_size: Input spatial size (default 224, must be divisible by 16).
        embed_dims: Channel dims for each level [enc1, enc2, enc3, enc4, bottleneck].
        rwkv_depths: Number of RWKV blocks per level.
        shift_pixel: Pixel shift for spatial mixing.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dims=None, rwkv_depths=None, shift_pixel=1,
                 deep_supervision=False, **kwargs):
        super().__init__()
        if embed_dims is None:
            embed_dims = [32, 64, 128, 256, 512]
        if rwkv_depths is None:
            rwkv_depths = [1, 1, 1, 1, 2]
        self.deep_supervision = deep_supervision
        self._embed_dims = embed_dims

        total_rwkv = sum(rwkv_depths)
        pool = nn.MaxPool2d(2, 2)

        # Encoder (4 levels + 瓶颈层 / Encoder (4 levels + bottleneck)
        self.enc1 = EncoderLevel(in_channels, embed_dims[0], rwkv_depths[0],
                                 total_rwkv, 0, shift_pixel)
        self.enc2 = EncoderLevel(embed_dims[0], embed_dims[1], rwkv_depths[1],
                                 total_rwkv, rwkv_depths[0], shift_pixel)
        self.enc3 = EncoderLevel(embed_dims[1], embed_dims[2], rwkv_depths[2],
                                 total_rwkv, sum(rwkv_depths[:2]), shift_pixel)
        self.enc4 = EncoderLevel(embed_dims[2], embed_dims[3], rwkv_depths[3],
                                 total_rwkv, sum(rwkv_depths[:3]), shift_pixel)

        # 瓶颈层 / Bottleneck
        self.bottleneck = EncoderLevel(embed_dims[3], embed_dims[4],
                                       rwkv_depths[4], total_rwkv,
                                       sum(rwkv_depths[:4]), shift_pixel)

        self.pools = nn.ModuleList([pool, pool, pool, pool])

        # 解码 ( 4 levels ) / Decoder (4 levels)
        self.dec4 = DecoderBlock(embed_dims[4], embed_dims[3], embed_dims[3])
        self.dec3 = DecoderBlock(embed_dims[3], embed_dims[2], embed_dims[2])
        self.dec2 = DecoderBlock(embed_dims[2], embed_dims[1], embed_dims[1])
        self.dec1 = DecoderBlock(embed_dims[1], embed_dims[0], embed_dims[0])

        # 分割 头部 / Segmentation head
        self.head = nn.Conv2d(embed_dims[0], num_classes, 1)

        # 深度 supervision heads ( from dec2, dec3 ) / Deep supervision heads (from dec2, dec3)
        if deep_supervision:
            self.ds_head2 = nn.Conv2d(embed_dims[1], num_classes, 1)
            self.ds_head3 = nn.Conv2d(embed_dims[2], num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]

        # 编码器 / Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pools[0](e1))
        e3 = self.enc3(self.pools[1](e2))
        e4 = self.enc4(self.pools[2](e3))

        # 瓶颈层 / Bottleneck
        b = self.bottleneck(self.pools[3](e4))

        # 解码 / Decoder
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # 分割 头部 / Segmentation head
        out = self.head(d1)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear',
                                align_corners=False)

        if self.training and self.deep_supervision:
            aux2 = self.ds_head2(d2)
            aux3 = self.ds_head3(d3)
            for aux in [aux2, aux3]:
                if aux.shape[2:] != input_size:
                    aux = F.interpolate(aux, size=input_size, mode='bilinear',
                                        align_corners=False)
            return [out,
                    F.interpolate(aux2, size=input_size, mode='bilinear',
                                  align_corners=False),
                    F.interpolate(aux3, size=input_size, mode='bilinear',
                                  align_corners=False)]
        return out


__all__ = ['URWKVTIP']
