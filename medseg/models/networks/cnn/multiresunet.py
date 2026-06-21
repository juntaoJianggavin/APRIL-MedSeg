"""MultiResUNet – faithful port from nibtehaz/MultiResUNet (PyTorch version).

MultiResUNet: Rethinking the U-Net architecture for multimodal biomedical
image segmentation (Ibtehaz & Rahman, Neural Networks 2020).

Faithful re-implementation based on the **official PyTorch source**:
    https://github.com/nibtehaz/MultiResUNet/blob/master/pytorch/MultiResUNet.py

Key components (matching original source exactly):
    - Multiresblock: cascaded 3x3 convolutions (factorising 5x5/7x7)
    - Respath: residual skip connections with double BN
    - Full U-Net encoder-decoder with raw concatenated channels
"""
# Source: https://github.com/nibtehaz/MultiResUNet

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (faithful to original Conv2d_batchnorm)
# ---------------------------------------------------------------------------
class Conv2d_batchnorm(nn.Module):
    """Conv → BN → [optional ReLU].  Matches original ``Conv2d_batchnorm``."""

    def __init__(self, num_in_filters, num_out_filters, kernel_size,
                 stride=(1, 1), activation='relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(
            in_channels=num_in_filters, out_channels=num_out_filters,
            kernel_size=kernel_size, stride=stride, padding='same')
        self.batchnorm = nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        if self.activation == 'relu':
            return F.relu(x)
        else:
            return x


# ---------------------------------------------------------------------------
# Multiresblock (faithful to original: cascaded 3x3 → 3x3 → 3x3)
# ---------------------------------------------------------------------------
class Multiresblock(nn.Module):
    """MultiRes Block — cascaded 3x3 convolutions (factorising 5x5/7x7).

    Faithful to the original ``Multiresblock``:
        conv_3x3(x) → a
        conv_5x5(a) → b     (3x3 on top of 3x3 = effective 5x5)
        conv_7x7(b) → c     (3x3 on top of 5x5 = effective 7x7)
        out = BN2(cat([a,b,c]) + shortcut)  → ReLU

    Note: output channels = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
          which may differ from ``num_filters`` due to int rounding.
    """

    def __init__(self, num_in_channels, num_filters, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha

        filt_cnt_3x3 = int(self.W * 0.167)
        filt_cnt_5x5 = int(self.W * 0.333)
        filt_cnt_7x7 = int(self.W * 0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

        # Shortcut (1x1 conv, no activation)
        self.shortcut = Conv2d_batchnorm(
            num_in_channels, num_out_filters,
            kernel_size=(1, 1), activation='None')

        # Cascaded 3x3 convolutions (all take 3x3 kernels)
        self.conv_3x3 = Conv2d_batchnorm(
            num_in_channels, filt_cnt_3x3,
            kernel_size=(3, 3), activation='relu')
        self.conv_5x5 = Conv2d_batchnorm(
            filt_cnt_3x3, filt_cnt_5x5,
            kernel_size=(3, 3), activation='relu')
        self.conv_7x7 = Conv2d_batchnorm(
            filt_cnt_5x5, filt_cnt_7x7,
            kernel_size=(3, 3), activation='relu')

        self.batch_norm1 = nn.BatchNorm2d(num_out_filters)
        self.batch_norm2 = nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        shrtct = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)       # cascaded: takes a, not x
        c = self.conv_7x7(b)       # cascaded: takes b, not a

        x = torch.cat([a, b, c], axis=1)
        x = self.batch_norm1(x)

        x = x + shrtct
        x = self.batch_norm2(x)
        x = F.relu(x)

        return x


# ---------------------------------------------------------------------------
# Respath (faithful to original: double BN per stage)
# ---------------------------------------------------------------------------
class Respath(nn.Module):
    """Residual path for skip connections (faithful to original).

    Each stage: conv3x3 → BN → ReLU → +shortcut → BN → ReLU
    """

    def __init__(self, num_in_filters, num_out_filters, respath_length):
        super().__init__()
        self.respath_length = respath_length
        self.shortcuts = nn.ModuleList([])
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        for i in range(self.respath_length):
            if i == 0:
                self.shortcuts.append(
                    Conv2d_batchnorm(
                        num_in_filters, num_out_filters,
                        kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    Conv2d_batchnorm(
                        num_in_filters, num_out_filters,
                        kernel_size=(3, 3), activation='relu'))
            else:
                self.shortcuts.append(
                    Conv2d_batchnorm(
                        num_out_filters, num_out_filters,
                        kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    Conv2d_batchnorm(
                        num_out_filters, num_out_filters,
                        kernel_size=(3, 3), activation='relu'))

            self.bns.append(nn.BatchNorm2d(num_out_filters))

    def forward(self, x):
        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)

            x = self.convs[i](x)       # Conv → BN → ReLU (built-in)
            x = self.bns[i](x)         # extra BN
            x = F.relu(x)              # extra ReLU

            x = x + shortcut           # add shortcut
            x = self.bns[i](x)         # reuses same BN (matches original)
            x = F.relu(x)

        return x


# ---------------------------------------------------------------------------
# MultiResUnet (faithful to original pytorch/MultiResUNet.py)
# ---------------------------------------------------------------------------
class MultiResUNet(nn.Module):
    """MultiResUNet (faithful to official PyTorch source).

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224, unused).
        alpha: MultiResBlock width scaling factor (default 1.67).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 alpha=1.67, **kwargs):
        super().__init__()
        self.alpha = alpha

        # Helper to compute raw MultiResBlock output channels
        def _raw_ch(base):
            W = base * alpha
            return int(W * 0.167) + int(W * 0.333) + int(W * 0.5)

        # ── Encoder path ─────────────────────────────────────────────
        self.multiresblock1 = Multiresblock(in_channels, 32)
        self.in_filters1 = _raw_ch(32)       # 53
        self.pool1 = nn.MaxPool2d(2)
        self.respath1 = Respath(self.in_filters1, 32, respath_length=4)

        self.multiresblock2 = Multiresblock(self.in_filters1, 32 * 2)
        self.in_filters2 = _raw_ch(32 * 2)   # 107
        self.pool2 = nn.MaxPool2d(2)
        self.respath2 = Respath(self.in_filters2, 32 * 2, respath_length=3)

        self.multiresblock3 = Multiresblock(self.in_filters2, 32 * 4)
        self.in_filters3 = _raw_ch(32 * 4)   # 214
        self.pool3 = nn.MaxPool2d(2)
        self.respath3 = Respath(self.in_filters3, 32 * 4, respath_length=2)

        self.multiresblock4 = Multiresblock(self.in_filters3, 32 * 8)
        self.in_filters4 = _raw_ch(32 * 8)   # 428
        self.pool4 = nn.MaxPool2d(2)
        self.respath4 = Respath(self.in_filters4, 32 * 8, respath_length=1)

        # ── Bottleneck ───────────────────────────────────────────────
        self.multiresblock5 = Multiresblock(self.in_filters4, 32 * 16)
        self.in_filters5 = _raw_ch(32 * 16)  # 856

        # ── Decoder path ─────────────────────────────────────────────
        self.upsample6 = nn.ConvTranspose2d(
            self.in_filters5, 32 * 8, kernel_size=(2, 2), stride=(2, 2))
        self.multiresblock6 = Multiresblock(32 * 8 * 2, 32 * 8)
        self.in_filters6 = _raw_ch(32 * 8)

        self.upsample7 = nn.ConvTranspose2d(
            self.in_filters6, 32 * 4, kernel_size=(2, 2), stride=(2, 2))
        self.multiresblock7 = Multiresblock(32 * 4 * 2, 32 * 4)
        self.in_filters7 = _raw_ch(32 * 4)

        self.upsample8 = nn.ConvTranspose2d(
            self.in_filters7, 32 * 2, kernel_size=(2, 2), stride=(2, 2))
        self.multiresblock8 = Multiresblock(32 * 2 * 2, 32 * 2)
        self.in_filters8 = _raw_ch(32 * 2)

        self.upsample9 = nn.ConvTranspose2d(
            self.in_filters8, 32, kernel_size=(2, 2), stride=(2, 2))
        self.multiresblock9 = Multiresblock(32 * 2, 32)
        self.in_filters9 = _raw_ch(32)

        # ── Final output (original: num_classes + 1) ─────────────────
        self.conv_final = Conv2d_batchnorm(
            self.in_filters9, num_classes + 1,
            kernel_size=(1, 1), activation='None')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x_multires1 = self.multiresblock1(x)
        x_pool1 = self.pool1(x_multires1)
        x_multires1 = self.respath1(x_multires1)

        x_multires2 = self.multiresblock2(x_pool1)
        x_pool2 = self.pool2(x_multires2)
        x_multires2 = self.respath2(x_multires2)

        x_multires3 = self.multiresblock3(x_pool2)
        x_pool3 = self.pool3(x_multires3)
        x_multires3 = self.respath3(x_multires3)

        x_multires4 = self.multiresblock4(x_pool3)
        x_pool4 = self.pool4(x_multires4)
        x_multires4 = self.respath4(x_multires4)

        # Bottleneck
        x_multires5 = self.multiresblock5(x_pool4)

        # Decoder
        up6 = torch.cat(
            [self.upsample6(x_multires5), x_multires4], axis=1)
        x_multires6 = self.multiresblock6(up6)

        up7 = torch.cat(
            [self.upsample7(x_multires6), x_multires3], axis=1)
        x_multires7 = self.multiresblock7(up7)

        up8 = torch.cat(
            [self.upsample8(x_multires7), x_multires2], axis=1)
        x_multires8 = self.multiresblock8(up8)

        up9 = torch.cat(
            [self.upsample9(x_multires8), x_multires1], axis=1)
        x_multires9 = self.multiresblock9(up9)

        out = self.conv_final(x_multires9)

        # Interpolate if output size differs from input
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(
                out, size=x.shape[-2:], mode='bilinear',
                align_corners=False)
        return out
