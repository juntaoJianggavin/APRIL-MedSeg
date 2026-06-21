"""nnMamba 2D: nnMambaSeg with ResNet-Mamba blocks for medical segmentation.

Reference:
    Gong et al., "nnMamba: 3D Mamba for Medical Image Segmentation",
    https://github.com/lhaof/nnMamba

Rewritten to match the official source (nnunet/network_architecture/nnMamba.py):
    * nnMambaSeg: ResNet-style encoder with MambaLayer inside BasicBlock +
      channel-attention skip connections + deep supervision.
    * BasicBlock (ResNet): conv3x3(stride) -> BN -> ReLU -> conv3x3 -> BN ->
      (+ mamba_layer(x) if mamba) -> (+ identity) -> ReLU.
    * make_res_layer: downsample(conv1x1+BN) + BasicBlock(stride) +
      (blocks-1) * BasicBlock(with mamba_layer).
    * MambaLayer: nin(conv1x1) -> BN -> ReLU -> flatten -> Mamba -> reshape.
    * Attentionlayer: channel attention (Linear -> ReLU -> Linear -> Sigmoid)
      applied as scale factors on skip connections.
    * Decoder: Upsample -> concat(attended_skip) -> DoubleConv, 4 stages.
    * Deep supervision: ds1-3 auxiliary outputs at lower resolutions.

This is a 2D adaptation of the official 3D model — Conv3d->Conv2d,
BatchNorm3d->BatchNorm2d, Upsample trilinear->bilinear, etc.

Constructor:
    NnMamba2D(in_channels=3, num_classes=2, img_size=224,
              channels=32, blocks=3, deep_supervision=False, **kwargs)
"""
# Source: https://github.com/lhaof/nnMamba (nnunet/network_architecture/nnMamba.py)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mamba SSM wrapper (hard dependency on mamba_ssm, matching official source)
# ---------------------------------------------------------------------------

class _MambaSSM(nn.Module):
    """Wrapper around ``mamba_ssm.Mamba`` (1D SSM, no bimamba).

    Interface: (B, L, D) -> (B, L, D)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        try:
            from mamba_ssm import Mamba  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "nnMamba requires the `mamba_ssm` CUDA package. "
                "Install from https://github.com/state-spaces/mamba. "
                "The official nnMamba source hard-depends on mamba_ssm."
            ) from e
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mamba(x)


# ---------------------------------------------------------------------------
# Conv helpers (matching official nnMamba.py)
# ---------------------------------------------------------------------------

def _conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def _conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


# ---------------------------------------------------------------------------
# MambaLayer (matching official nnMamba.py)
# ---------------------------------------------------------------------------

class _MambaLayer(nn.Module):
    """nin(conv1x1) -> BN -> ReLU -> flatten -> Mamba -> reshape.

    (B, C, H, W) -> (B, C, H, W)
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.nin = _conv1x1(dim, dim)
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.mamba = _MambaSSM(dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        B, C = x.shape[:2]
        x = self.nin(x)
        x = self.norm(x)
        x = self.relu(x)
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  # (B, N, C)
        x_mamba = self.mamba(x_flat)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


# ---------------------------------------------------------------------------
# ResNet BasicBlock (matching official nnMamba.py)
# ---------------------------------------------------------------------------

class _BasicBlock(nn.Module):
    """ResNet BasicBlock with optional MambaLayer.

    conv3x3(stride) -> BN -> ReLU -> conv3x3 -> BN ->
    (+ mamba_layer(x) if mamba) -> (+ identity/downsample) -> ReLU
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 mamba_layer=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.mamba_layer = mamba_layer
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.mamba_layer is not None:
            global_att = self.mamba_layer(x)
            out += global_att
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def _make_res_layer(inplanes, planes, blocks, stride=1, mamba_layer=None):
    """Create a ResNet layer: downsample + BasicBlock(stride) + (blocks-1)*BasicBlock(mamba)."""
    downsample = nn.Sequential(
        _conv1x1(inplanes, planes, stride),
        nn.BatchNorm2d(planes),
    )
    layers = [_BasicBlock(inplanes, planes, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(_BasicBlock(planes, planes, mamba_layer=mamba_layer))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# DoubleConv (matching official nnMamba.py)
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """2x Conv-BN-ReLU (first conv carries the stride)."""

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Attentionlayer (matching official nnMamba.py)
# ---------------------------------------------------------------------------

class _AttentionLayer(nn.Module):
    """Channel attention: Linear -> ReLU -> Linear -> Sigmoid."""

    def __init__(self, dim, r=16):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim // r)
        self.layer2 = nn.Linear(dim // r, dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        # inp: (B, C) -> (B, C)
        return self.sigmoid(self.layer2(self.relu(self.layer1(inp))))


# ---------------------------------------------------------------------------
# nnMambaSeg (2D adaptation)
# ---------------------------------------------------------------------------

class NnMamba2D(nn.Module):
    """nnMambaSeg 2D — ResNet-Mamba encoder with attention skips + deep supervision.

    Matches the official ``nnMambaSeg`` architecture:
        * in_conv: DoubleConv(in -> channels, stride=2)
        * 3 ResNet layers (channels -> channels*2 -> channels*4 -> channels*8),
          each with MambaLayer inside BasicBlocks
        * Channel-attention scale factors on skip connections
        * 4-stage decoder: Upsample -> concat(attended_skip) -> DoubleConv
        * Deep supervision: 3 auxiliary 1x1 conv outputs

    Args:
        in_channels: Input channel count.
        num_classes: Output channel count.
        img_size: Spatial size (for documentation; accepts any size divisible
            by 2^4 = 16).
        channels: Base channel count (doubled at each encoder stage).
        blocks: Number of BasicBlocks per ResNet layer.
        deep_supervision: If True, returns a list of 4 outputs [main, ds1, ds2, ds3].
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 channels=32, blocks=3, deep_supervision=False, **kwargs):
        super().__init__()
        self.do_ds = deep_supervision
        self.in_conv = _DoubleConv(in_channels, channels, stride=2, kernel_size=3)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Encoder: 3 ResNet layers with MambaLayer
        self.att1 = _AttentionLayer(channels)
        self.layer1 = _make_res_layer(
            channels, channels * 2, blocks, stride=2,
            mamba_layer=_MambaLayer(channels * 2))

        self.att2 = _AttentionLayer(channels * 2)
        self.layer2 = _make_res_layer(
            channels * 2, channels * 4, blocks, stride=2,
            mamba_layer=_MambaLayer(channels * 4))

        self.att3 = _AttentionLayer(channels * 4)
        self.layer3 = _make_res_layer(
            channels * 4, channels * 8, blocks, stride=2,
            mamba_layer=_MambaLayer(channels * 8))

        # Decoder: 4 stages of Upsample -> concat(attended_skip) -> DoubleConv
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv5 = _DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv6 = _DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv7 = _DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv8 = _DoubleConv(channels, num_classes)

        # Deep supervision
        self.ds1_cls_conv = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.ds2_cls_conv = nn.Conv2d(channels * 2, num_classes, kernel_size=1)
        self.ds3_cls_conv = nn.Conv2d(channels * 4, num_classes, kernel_size=1)

    def forward(self, x):
        original_h, original_w = x.shape[-2], x.shape[-1]

        # pad to multiple of 16 (4 stride-2 stages)
        pH = ((x.shape[-2] + 15) // 16) * 16
        pW = ((x.shape[-1] + 15) // 16) * 16
        if pH != x.shape[-2] or pW != x.shape[-1]:
            x = F.pad(x, [0, pW - x.shape[-1], 0, pH - x.shape[-2]],
                      mode='reflect')

        # Encoder
        c1 = self.in_conv(x)
        scale_f1 = self.att1(
            self.pooling(c1).reshape(c1.shape[0], c1.shape[1])
        ).reshape(c1.shape[0], c1.shape[1], 1, 1)

        c2 = self.layer1(c1)
        scale_f2 = self.att2(
            self.pooling(c2).reshape(c2.shape[0], c2.shape[1])
        ).reshape(c2.shape[0], c2.shape[1], 1, 1)

        c3 = self.layer2(c2)
        scale_f3 = self.att3(
            self.pooling(c3).reshape(c3.shape[0], c3.shape[1])
        ).reshape(c3.shape[0], c3.shape[1], 1, 1)

        c4 = self.layer3(c3)

        # Decoder
        up_5 = self.up5(c4)
        if up_5.shape[-2:] != c3.shape[-2:]:
            up_5 = F.interpolate(up_5, size=c3.shape[-2:],
                                 mode='bilinear', align_corners=False)
        merge5 = torch.cat([up_5, c3 * scale_f3], dim=1)
        c5 = self.conv5(merge5)

        up_6 = self.up6(c5)
        if up_6.shape[-2:] != c2.shape[-2:]:
            up_6 = F.interpolate(up_6, size=c2.shape[-2:],
                                 mode='bilinear', align_corners=False)
        merge6 = torch.cat([up_6, c2 * scale_f2], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        if up_7.shape[-2:] != c1.shape[-2:]:
            up_7 = F.interpolate(up_7, size=c1.shape[-2:],
                                 mode='bilinear', align_corners=False)
        merge7 = torch.cat([up_7, c1 * scale_f1], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        c8 = self.conv8(up_8)

        # interpolate main output to original resolution
        if c8.shape[-2:] != (original_h, original_w):
            c8 = F.interpolate(c8, size=(original_h, original_w),
                               mode='bilinear', align_corners=False)

        if self.do_ds:
            logits = [c8,
                      self.ds1_cls_conv(c7),
                      self.ds2_cls_conv(c6),
                      self.ds3_cls_conv(c5)]
            return logits
        else:
            return c8


__all__ = ['NnMamba2D']
