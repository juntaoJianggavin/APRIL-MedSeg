"""DeepLabV3 Decoder — ASPP-based segmentation decoder.
    DeepLabV3 Decoder — ASPP-based segmentation 解码器。

中文: DeepLabV3 解码器：基于ASPP的分割解码器。

Reference:
    Chen et al., "Rethinking Atrous Convolution for Semantic Image
    Segmentation." arXiv 2017 (DeepLabV3).
    Official TF code: github.com/tensorflow/models/tree/master/research/deeplab

Architecture:
    The DeepLabV3 decoder applies ASPP (Atrous Spatial Pyramid Pooling) on
    the deepest encoder/bottleneck feature, then upsamples to full input
    resolution with a 1×1 classification head. Unlike U-Net decoders, it
    does NOT use skip connections — all multi-scale context is captured by
    ASPP's parallel atrous convolutions at different dilation rates.

    This decoder can be paired with ANY encoder (ResNet, ConvNeXt, HRNet,
    etc.) via the modular pipeline. When used with bottleneck=aspp, the
    ASPP is applied in the bottleneck stage and this decoder simply
    upsamples + classifies. When bottleneck=none, the decoder applies its
    own ASPP internally.

Integration:
    Set ``decoder: { name: deeplabv3 }`` in YAML.
    External skip_connection is IGNORED (has_internal_skip = True).
"""
# Source: https://github.com/tensorflow/models/tree/master/research/deeplab

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from medseg.registry import DECODER_REGISTRY


class _ASPPConv(nn.Module):
    """Single ASPP 分支: atrous conv + BN + ReLU。
        Single ASPP branch: atrous conv + BN + ReLU."""

    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        if dilation == 1:
            conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            conv = nn.Conv2d(in_ch, out_ch, 3, padding=dilation,
                             dilation=dilation, bias=False)
        self.block = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _ASPPPooling(nn.Module):
    """全局的 average 池化 分支 for ASPP。
        Global average pooling branch for ASPP."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # No BN here: GAP produces 1 × 1 空间的 → BN fails with 批次 _ 大小 = 1 / No BN here: GAP produces 1×1 spatial → BN fails with batch_size=1.
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        x = self.block(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class _ASPPModule(nn.Module):
    """Atrous 空间的 金字塔 池化 with 4 并行的 branches + 全局的 池化。
        Atrous Spatial Pyramid Pooling with 4 parallel branches + global pooling."""

    def __init__(self, in_ch, out_ch=256, atrous_rates=(6, 12, 18)):
        super().__init__()
        branches = [_ASPPConv(in_ch, out_ch, 1)]  # 1×1 conv
        for rate in atrous_rates:
            branches.append(_ASPPConv(in_ch, out_ch, rate))
        branches.append(_ASPPPooling(in_ch, out_ch))
        self.convs = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(atrous_rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.convs], dim=1)
        return self.project(out)


@DECODER_REGISTRY.register("deeplabv3")
class DeepLabV3Decoder(nn.Module):
    """DeepLabV3 decoder: ASPP + bilinear upsample.
        DeepLabV3 解码器。

    中文: DeepLabV3 解码器：ASPP + 双线性上采样。

    This decoder applies ASPP on the deepest feature (or bottleneck output),
    then upsamples to the full input resolution. The framework's
    SegmentationHead is responsible for the final 1×1 classification.
    External skip connections are ignored.

    Args:
        encoder_channels: list of encoder output channel counts.
        bottleneck_channels: bottleneck output channels (if bottleneck is used).
        aspp_channels: ASPP output channels (default 256).
        atrous_rates: dilation rates for ASPP (default [6, 12, 18]).
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 aspp_channels: int = 256, atrous_rates=(6, 12, 18),
                 **kwargs):
        super().__init__()
        # Use bottleneck output if available, else deepest 编码器 / Use bottleneck output if available, else deepest encoder feature
        in_ch = bottleneck_channels if bottleneck_channels else encoder_channels[-1]
        self.aspp = _ASPPModule(in_ch, aspp_channels, atrous_rates)
        self.conv_out = nn.Sequential(
            nn.Conv2d(aspp_channels, aspp_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
        )
        self._out_channels = aspp_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor], **kwargs):
        """Forward pass.
            前向传播 pass。

        The framework calls ``decoder(bottleneck_feat, skip_features)``.
        DeepLabV3 ignores skip features (``has_internal_skip = True``) and
        applies ASPP solely on the bottleneck output.  The framework's
        ``SegmentationModel`` handles the final bilinear upsample.

        Args:
            bottleneck_feat: deepest / bottleneck feature (B, C, H', W').
            skip_features: ignored (kept for API compatibility).
        """
        x = self.aspp(bottleneck_feat)
        x = self.conv_out(x)
        return x
