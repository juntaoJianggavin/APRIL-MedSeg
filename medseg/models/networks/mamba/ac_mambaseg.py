"""AC-MambaSeg: Adaptive Convolution and Mamba-based Skin Lesion Segmentation.

Reference:
    Nguyen et al., "AC-MAMBASEG: An Adaptive Convolution and Mamba-based
    Architecture for Enhanced Skin Lesion Segmentation", 2024.
    https://github.com/vietthanh2710/AC-MambaSeg

Rewritten to match the official source (models/AC_MambaSeg.py) line-by-line:
    * SKConv_7 / SKConv (Selective Kernel) + SKUnit bottleneck.
    * VSSBlock (LayerNorm -> SS2D -> DropPath -> residual).
    * ResMambaBlock (depthwise conv -> VSSBlock -> InstanceNorm -> LeakyReLU).
    * EncoderBlock (ResMambaBlock -> Conv3x3-BN-ReLU -> MaxPool).
    * CBAM skip (BasicConv + ChannelGate(avg+max) + SpatialGate with BN).
    * DecoderBlock (Upsample -> Attention_block -> concat -> Conv -> ResMambaBlock).
    * 5-stage UNet 16->32->64->128->256->512.

Constructor:
    ACMambaSeg(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/vietthanh2710/AC-MambaSeg

from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from medseg.models.encoders.vmunet_encoder import SS2D


# ---------------------------------------------------------------------------
# Selective Kernels
# ---------------------------------------------------------------------------

class SKConv_7(nn.Module):
    def __init__(self, features, M=3, G=16, r=16, stride=1, L=32):
        super(SKConv_7, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=7, stride=stride,
                          padding='same', dilation=i + 1, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv2d(d, features, kernel_size=1, stride=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features,
                           feats.shape[2], feats.shape[3])
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(
            batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(feats * attention_vectors, dim=1)
        return feats_V


class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride,
                          padding='same', dilation=i + 1, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv2d(d, features, kernel_size=1, stride=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features,
                           feats.shape[2], feats.shape[3])
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(
            batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(feats * attention_vectors, dim=1)
        return feats_V


class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, M=2, G=32,
                 r=16, stride=1, L=64):
        super(SKUnit, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
        )
        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)
        return self.relu(out + self.shortcut(residual))


# ---------------------------------------------------------------------------
# VSS Block
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
# Encoder Block
# ---------------------------------------------------------------------------

class ResMambaBlock(nn.Module):
    def __init__(self, in_c, k_size=3):
        super().__init__()
        self.in_c = in_c
        self.conv = nn.Conv2d(in_c, in_c, k_size, stride=1, padding='same',
                              dilation=1, groups=in_c, bias=True,
                              padding_mode='zeros')
        self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.block = VSSBlock(hidden_dim=in_c)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        skip = x
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.block(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.ins_norm(x))
        return x + skip * self.scale


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
# Skip Connection (CBAM)
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
# Decoder Block
# ---------------------------------------------------------------------------

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.att = Attention_block(F_g=in_c, F_l=skip_c, F_int=skip_c // 2)
        self.conv2 = nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_c)
        self.resmamba = ResMambaBlock(out_c)
        self.act = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        skip = self.att(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.resmamba(x)
        return x


# ---------------------------------------------------------------------------
# AC-MambaSeg
# ---------------------------------------------------------------------------

class ACMambaSeg(nn.Module):
    """AC-MambaSeg for skin lesion segmentation.

    Channel progression: 16 -> 32 -> 64 -> 128 -> 256 -> 512 (5-stage UNet).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        self.pw_in = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.sk_in = SKConv_7(16, M=2, G=16, r=4, stride=1, L=32)
        """Encoder"""
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)
        self.e5 = EncoderBlock(256, 512)

        """Skip connection"""
        self.s1 = CBAM(gate_channels=32)
        self.s2 = CBAM(gate_channels=64)
        self.s3 = CBAM(gate_channels=128)
        self.s4 = CBAM(gate_channels=256)
        self.s5 = CBAM(gate_channels=512)

        """Bottle Neck"""
        self.b5 = SKUnit(512, 512, 512, M=2, G=16, r=2, stride=1, L=32)

        """Decoder"""
        self.d5 = DecoderBlock(512, 512, 256)
        self.d4 = DecoderBlock(256, 256, 128)
        self.d3 = DecoderBlock(128, 128, 64)
        self.d2 = DecoderBlock(64, 64, 32)
        self.d1 = DecoderBlock(32, 32, 16)
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2:]
        # 5 MaxPool stages => /32; pad to a multiple of 32.
        pH = (32 - H % 32) % 32
        pW = (32 - W % 32) % 32
        if pH > 0 or pW > 0:
            x = F.pad(x, [0, pW, 0, pH], mode='reflect')

        """Encoder"""
        x = self.pw_in(x)
        x = self.sk_in(x)
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
        x = self.b5(x)

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
