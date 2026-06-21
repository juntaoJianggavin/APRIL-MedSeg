"""DermoMamba: Cross-Scale Mamba for Skin Lesion Segmentation.

Reference:
    Hoang et al., "DermoMamba: A cross-scale Mamba-based model with Guide
    Fusion Loss for skin lesion segmentation in dermoscopy images",
    Pattern Analysis and Applications, 2025.
    https://github.com/hnkhai25/DermoMamba

Rewritten to match the official source module-by-module:
    * CBAM (module/CBAM.py): BasicConv + ChannelGate(avg+max) + SpatialGate(BN).
    * Cross_Scale_Mamba_Block (module/CSMB.py): 4 channel groups, dilated
      axial DW + VSSBlock on first 3 groups, 4th identity, cat -> BN+ReLU.
    * VSSBlock (module/VSSBlock.py): LayerNorm -> SS2D -> DropPath -> residual.
    * PCA (module/PACM.py): dw9 -> channel attention (reduce/einsum).
    * Sweep_Mamba (module/SMB.py): 3-directional SS2D bottleneck.
    * ResMambaBlock / EncoderBlock (model/encoder.py).
    * DecoderBlock (model/decoder.py).
    * DermoMamba (model/proposed_net.py): bottleneck = Sweep_Mamba(PCA(x) + x).

Note: the official Sweep_Mamba hardcodes d_model=6 and d_model=8 for the 2nd
and 3rd SS2D, which only works for a fixed bottleneck size of H=6, W=8. This
rewrite generalises those to dim//ratio (keeping channels last) so the model
runs on arbitrary input sizes while preserving the 3-directional structure.

Constructor:
    DermoMamba(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/hnkhai25/DermoMamba

from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from timm.models.layers import DropPath

from medseg.models.encoders.vmunet_encoder import SS2D


# ---------------------------------------------------------------------------
# VSS Block (module/VSSBlock.py)
# ---------------------------------------------------------------------------

class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0, d_state: int = 16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate,
                                   d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


# ---------------------------------------------------------------------------
# CBAM (module/CBAM.py)
# ---------------------------------------------------------------------------

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.GELU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                        stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)),
                                        stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)),
                                      stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1,
                                 padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# ---------------------------------------------------------------------------
# Cross-Scale Mamba Block (module/CSMB.py)
# ---------------------------------------------------------------------------

class Axial_Spatial_DW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.mixer_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same',
                                 groups=dim, dilation=dilation)
        self.mixer_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same',
                                 groups=dim, dilation=dilation)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding='same',
                              groups=dim, dilation=dilation)

    def forward(self, x):
        skip = x
        x = self.mixer_w(x)
        x = self.mixer_h(x)
        x = self.conv(x)
        return x + skip


class Cross_Scale_Mamba_Block(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dw1 = Axial_Spatial_DW(dim // 4, (7, 7), dilation=1)
        self.dw2 = Axial_Spatial_DW(dim // 4, (7, 7), dilation=2)
        self.dw3 = Axial_Spatial_DW(dim // 4, (7, 7), dilation=3)
        self.vss = VSSBlock(dim // 4)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1 = self.vss(self.dw1(x1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x2 = self.vss(self.dw2(x2).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x3 = self.vss(self.dw3(x3).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.act(self.bn(x))
        return x


# ---------------------------------------------------------------------------
# PCA (module/PACM.py)
# ---------------------------------------------------------------------------

class PCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=9, groups=dim, padding="same")
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        c = reduce(x, 'b c h w -> b c', 'mean')
        x = self.dw(x)
        c_ = reduce(x, 'b c h w -> b c', 'mean')
        raise_ch = self.prob(c_ - c)
        att_score = torch.sigmoid(c_ * (1 + raise_ch))
        return torch.einsum('bchw, bc -> bchw', x, att_score)


# ---------------------------------------------------------------------------
# Sweep_Mamba (module/SMB.py)
# ---------------------------------------------------------------------------

class Sweep_Mamba(nn.Module):
    def __init__(self, dim, ratio=8):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, dim // ratio, 1)
        # Official hardcodes d_model=6 and d_model=8 for mamba2/mamba3, which
        # only works for a fixed bottleneck size (H=6, W=8). Generalised to
        # dim//ratio so the model runs on arbitrary input sizes.
        self.mamba1 = SS2D(d_model=dim // ratio, dropout=0, d_state=16)
        self.mamba2 = SS2D(d_model=dim // ratio, dropout=0, d_state=16)
        self.mamba3 = SS2D(d_model=dim // ratio, dropout=0, d_state=16)
        self.act = nn.SiLU()
        self.relu = nn.ReLU()
        self.proj_out = nn.Linear(dim // ratio, dim, 1)
        self.scale = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # x: (B, C, H, W) -> operate in (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        skip = x
        x = self.proj_in(self.ln(x))                 # (B, H, W, dim//ratio)
        x1 = self.mamba1(x)
        x2 = self.mamba2(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x3 = self.mamba3(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        w = self.act(x)
        out = w * x1 + w * x2 + w * x3
        out = self.proj_out(out) + skip * self.scale
        out = out.permute(0, 3, 1, 2)                 # back to (B, C, H, W)
        out = self.bn(out)
        out = self.relu(out)
        return out


# ---------------------------------------------------------------------------
# ResMambaBlock / EncoderBlock (model/encoder.py)
# ---------------------------------------------------------------------------

class ResMambaBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.block = Cross_Scale_Mamba_Block(in_c)
        self.conv = nn.Conv2d(in_c, in_c, kernel_size=3, padding='same')
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self.block(x)
        x = self.act(self.ins_norm(self.conv(x))) + x * self.scale
        return x


class EncoderBlock(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.resmamba = ResMambaBlock(in_c)
        self.down = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.resmamba(x)
        skip = self.act(self.bn(self.pw(x)))
        x = self.down(skip)
        return x, skip


# ---------------------------------------------------------------------------
# DecoderBlock (model/decoder.py)
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c * 2, in_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.pw(x)
        x = self.bn(self.act(self.pw2(x)))
        return x


# ---------------------------------------------------------------------------
# DermoMamba (model/proposed_net.py)
# ---------------------------------------------------------------------------

class DermoMamba(nn.Module):
    """Cross-scale Mamba with CBAM skip + PCA/Sweep_Mamba bottleneck.

    Channel progression: 16 -> 32 -> 64 -> 128 -> 256 -> 512 (5-stage UNet).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 base_channels=16, **kwargs):
        super().__init__()
        c = base_channels  # 16

        self.pw_in = nn.Conv2d(in_channels, c, kernel_size=1)

        """Encoder"""
        self.e1 = EncoderBlock(c, c * 2)
        self.e2 = EncoderBlock(c * 2, c * 4)
        self.e3 = EncoderBlock(c * 4, c * 8)
        self.e4 = EncoderBlock(c * 8, c * 16)
        self.e5 = EncoderBlock(c * 16, c * 32)

        """Skip connection"""
        self.s1 = CBAM(c * 2)
        self.s2 = CBAM(c * 4)
        self.s3 = CBAM(c * 8)
        self.s4 = CBAM(c * 16)
        self.s5 = CBAM(c * 32)

        """Bottle Neck"""
        self.b1 = Sweep_Mamba(c * 32)
        self.b2 = PCA(c * 32)

        """Decoder"""
        self.d5 = DecoderBlock(c * 32, c * 16)
        self.d4 = DecoderBlock(c * 16, c * 8)
        self.d3 = DecoderBlock(c * 8, c * 4)
        self.d2 = DecoderBlock(c * 4, c * 2)
        self.d1 = DecoderBlock(c * 2, c)
        # Final layer
        self.conv_out = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2:]
        # 5 MaxPool stages => /32; pad to a multiple of 32.
        pH = (32 - H % 32) % 32
        pW = (32 - W % 32) % 32
        if pH > 0 or pW > 0:
            x = F.pad(x, [0, pW, 0, pH], mode='reflect')

        """Encoder"""
        x = self.pw_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        """Skip connection"""
        skip1 = self.s1(skip1)
        skip2 = self.s2(skip2)
        skip3 = self.s3(skip3)
        skip4 = self.s4(skip4)
        skip5 = self.s5(skip5)

        """BottleNeck"""
        x = self.b1(self.b2(x) + x)

        """Decoder"""
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.conv_out(x)

        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x
