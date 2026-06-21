"""SkinMamba: CNN-Mamba Hybrid for Skin Lesion Segmentation.
    SkinMamba: CNN-Mamba Hybrid for Skin Lesion Segmentation.

Faithful reimplementation from:
  https://github.com/zs1314/skinmamba  (ACCV Workshop 2025)
  Source files: models/SkinMamba.py (614 lines) + models/freqency.py (95 lines)

Architecture (faithful to official source):
  - pw_in (1x1 conv) + SKConv_7 (Selective Kernel 7x7)
  - 5 EncoderBlocks: SRSSB (SS2D + SMFFL) → Conv+BN+ReLU → MaxPool
  - FFMB (FFT frequency modulation) at bottleneck
  - 5 DecoderBlocks: Upsample → concat skip → Conv+BN+ReLU → SRSSB
  - conv_out (1x1 conv) → sigmoid

Constructor:
    SkinMamba(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/zs1314/skinmamba

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable

from medseg.models.encoders.vmunet_encoder import SS2D


# ---------------------------------------------------------------------------
# Selective Kernel Convolution (faithful to official source)
# ---------------------------------------------------------------------------

class SKConv_7(nn.Module):
    """Selective Kernel with 7x7 dilated convolutions (faithful to official).
        Selective Kernel with 7x7 dilated convolutions.
    """
    def __init__(self, features, M=3, G=16, r=16, stride=1, L=32):
        super().__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=7, stride=stride,
                          padding=3 * (i + 1), dilation=i + 1, groups=G, bias=False),
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
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(feats * attention_vectors, dim=1)
        return feats_V


class SKConv(nn.Module):
    """Selective Kernel with 3x3 dilated convolutions (faithful to official).
        Selective Kernel with 3x3 dilated convolutions.
    """
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super().__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride,
                          padding=1 * (i + 1), dilation=i + 1, groups=G, bias=False),
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
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(feats * attention_vectors, dim=1)
        return feats_V


class SKUnit(nn.Module):
    """Selective Kernel Unit (faithful to official source)."""
    def __init__(self, in_features, mid_features, out_features, M=2, G=32, r=16, stride=1, L=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True))
        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features))
        if in_features == out_features:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)
        return self.relu(out + self.shortcut(residual))


# ---------------------------------------------------------------------------
# SMFFL: Multi-Scale Feature Fusion Layer (faithful to official source)
# ---------------------------------------------------------------------------

class SMFFL(nn.Module):
    """Multi-scale feature fusion with 3x3 + 5x5 depthwise conv (faithful to official).
        Multi-scale feature fusion with 3x3 + 5x5 depthwise conv.
    """
    def __init__(self, in_channels, reduction_ratio=2, hidden_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.hidden_dim = hidden_dim if hidden_dim else in_channels // reduction_ratio
        self.linear1 = nn.Linear(in_channels, self.hidden_dim)
        self.linear2 = nn.Linear(in_channels, self.hidden_dim)
        self.dw_conv3x3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=self.hidden_dim)
        self.dw_conv5x5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2, groups=self.hidden_dim)
        self.gelu = nn.GELU()
        self.linear3 = nn.Linear(self.hidden_dim * 2, in_channels)
        self.ln = nn.LayerNorm(in_channels)

    def forward(self, x):
        x_ln = self.ln(x)
        x1 = self.linear1(x_ln)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.dw_conv3x3(x1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = self.linear2(x_ln)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = self.dw_conv5x5(x2)
        x2 = x2.permute(0, 2, 3, 1)
        x_concat = torch.cat([x1, x2], dim=-1)
        x_out = self.gelu(x_concat)
        x_out = self.linear3(x_out)
        out = x + x_out
        return out


# ---------------------------------------------------------------------------
# SRSSB: SS2D + SMFFL Block (faithful to official source)
# ---------------------------------------------------------------------------

class SRSSB(nn.Module):
    """Selective Rectangular State Space Block: SS2D + SMFFL (faithful to official).
        SRSSB: SS2D attention + SMFFL feedforward with residual connections.
    """
    def __init__(self, hidden_dim=0, drop_path=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate=0.0, d_state=16, **kwargs):
        super().__init__()
        self.ln = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate,
                                   d_state=d_state, **kwargs)
        self.drop_path = nn.Identity()  # DropPath not used for simplicity
        self.ffn = SMFFL(in_channels=hidden_dim)

    def forward(self, input):
        x = input + self.drop_path(self.self_attention(self.ln(input)))
        residual = x
        x = residual + self.drop_path(self.ffn(self.ln(x)))
        return x


# ---------------------------------------------------------------------------
# Frequency Boundary Guidance (faithful to official freqency.py)
# ---------------------------------------------------------------------------

class FFML(nn.Module):
    """FFT Frequency Modulation Layer (faithful to official freqency.py).
        FFT-based frequency modulation with conv + sigmoid gating.
    """
    def __init__(self, out_channel, norm='backward'):
        super().__init__()
        self.main_fft = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel * 2, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel * 2, out_channel * 2, kernel_size=1, stride=1)
        )
        self.norm = norm
        self.act = nn.Sigmoid()

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = self.act(y)
        y = y * x
        return y


class _LayerNorm(nn.Module):
    """LayerNorm with channels_first support (faithful to official freqency.py)."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class _FC(nn.Module):
    """Feedforward conv layer (faithful to official freqency.py)."""
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.fc(x)


class FFMB(nn.Module):
    """Frequency Feature Modulation Block (faithful to official freqency.py).
        FFT modulation + feedforward with residual connections.
    """
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.norm1 = _LayerNorm(dim)
        self.norm2 = _LayerNorm(dim)
        self.fft = FFML(dim)
        self.fc = _FC(dim, ffn_scale)

    def forward(self, x):
        y = self.norm1(x)
        y = self.fft(y)
        y = self.fc(self.norm2(y)) + y
        return y


# ---------------------------------------------------------------------------
# Encoder/Decoder Blocks (faithful to official source)
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Encoder block: SRSSB → Conv+BN+ReLU → MaxPool (faithful to official).
        Returns (downsampled, skip).
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.SRSSB = SRSSB(hidden_dim=in_c)
        self.down = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
        x = self.SRSSB(x)
        x = x.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W
        skip = self.act(self.bn(self.conv(x)))
        x = self.down(skip)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block: Upsample → concat skip → Conv+BN+ReLU → SRSSB (faithful to official)."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.SRSSB = SRSSB(hidden_dim=out_c)
        self.act = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
        x = self.SRSSB(x)
        x = x.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W
        return x


# ---------------------------------------------------------------------------
# SkinMamba model (faithful to official source)
# ---------------------------------------------------------------------------

class SkinMamba(nn.Module):
    """SkinMamba for skin lesion segmentation (faithful to official source).
        SkinMamba for skin lesion segmentation.

    Architecture:
      pw_in(1x1) → SKConv_7 → 5 EncoderBlocks(16→32→64→128→256→512)
      → FFMB(512) → 5 DecoderBlocks(512→256→128→64→32→16) → conv_out → sigmoid
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.pw_in = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.sk_in = SKConv_7(16, M=2, G=16, r=4, stride=1, L=32)

        # Encoder
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)
        self.e5 = EncoderBlock(256, 512)

        # Bottleneck: FFT frequency modulation
        self.ffmb = FFMB(dim=512)

        # Decoder
        self.d5 = DecoderBlock(512, 512, 256)
        self.d4 = DecoderBlock(256, 256, 128)
        self.d3 = DecoderBlock(128, 128, 64)
        self.d2 = DecoderBlock(64, 64, 32)
        self.d1 = DecoderBlock(32, 32, 16)

        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        H, W = x.shape[2:]
        # Pad to multiple of 32 (5 downsampling stages)
        pH = ((H + 31) // 32) * 32
        pW = ((W + 31) // 32) * 32
        if pH != H or pW != W:
            x = F.pad(x, [0, pW - W, 0, pH - H], mode='reflect')

        # Encoder
        x = self.pw_in(x)
        x = self.sk_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        # Bottleneck
        x = self.ffmb(x)

        # Decoder
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)

        x = self.conv_out(x)
        if self.num_classes == 1:
            x = self.sigmoid(x)

        # Crop to original size
        if pH != H or pW != W:
            x = x[:, :, :H, :W]
        return x
