"""UNet – the original U-Net (Ronneberger et al., MICCAI 2015).
    UNet – 原始 U-Net (Ronneberger et al., MICCAI 2015)。

Reference: https://github.com/milesial/Pytorch-UNet
Paper: https://arxiv.org/abs/1505.04597

Architecture: Classic 4-level encoder-decoder with skip connections
(concatenation). Each level uses two consecutive 3×3 conv-BN-ReLU blocks
(DoubleConv). Upsampling is done via transposed convolution (ConvTranspose2d)
followed by skip concatenation and DoubleConv fusion.

This is the most fundamental segmentation network in the framework.
For a modular version (encoder + UNet decoder), use ``decoder.name: unet``
in YAML configs. This standalone version bundles encoder, bottleneck,
and decoder into a single self-contained module.
"""
# Source: https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _DoubleConv(nn.Module):
    """两个连续 Conv-BN-ReLU blocks。
        Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet (Ronneberger et al., 2015)
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """Classic U-Net with 4 encoder/decoder levels.
        经典 U-Net，4 层编码/解码。

    Reference:
        Ronneberger, O., Fischer, P., Brox, T. (2015).
        U-Net: Convolutional Networks for Biomedical Image Segmentation.
        MICCAI 2015. https://arxiv.org/abs/1505.04597

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224, unused).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        # 编码器 / Encoder
        self.enc1 = _DoubleConv(in_channels, 64)
        self.enc2 = _DoubleConv(64, 128)
        self.enc3 = _DoubleConv(128, 256)
        self.enc4 = _DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层 / Bottleneck
        self.bottleneck = _DoubleConv(512, 1024)

        # 解码器 / Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = _DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = _DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = _DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = _DoubleConv(128, 64)

        # 输出 / Output
        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # 编码器 / Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 瓶颈层 / Bottleneck
        b = self.bottleneck(self.pool(e4))

        # 解码器（跳跃连接 = 拼接）/ Decoder (skip = concatenation)
        d4 = self.up4(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = _pad_to_match(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = _pad_to_match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = _pad_to_match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = _pad_to_match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear',
                                align_corners=False)
        return out


def _pad_to_match(x, target):
    """零填充 x 使其空间尺寸与 target 一致（如原始 UNet 论文）。
        Zero-pad x to match target spatial size (as in original UNet paper)."""
    diff_h = target.shape[2] - x.shape[2]
    diff_w = target.shape[3] - x.shape[3]
    return F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                     diff_h // 2, diff_h - diff_h // 2])
