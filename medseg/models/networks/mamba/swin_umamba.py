"""Swin-UMamba: VMamba Encoder + UNETR-style Conv Decoder for Medical Image Segmentation.
    Swin-UMamba: VMamba 编码器。

Faithful reimplementation from:
  https://github.com/JiarunLiu/Swin-UMamba  (2024, 384+ stars)

Architecture:
  - Encoder: VMamba (PatchEmbed2D → VSSLayer stages with PatchMerging2D)
  - Decoder: UNETR-style conv blocks with upsampling (self-contained, no MONAI)
  - Skip connections via concatenation + conv fusion

Reuses SS2D/VSSBlock/PatchEmbed2D/PatchMerging2D from vmunet_encoder.
"""
# Source: https://github.com/JiarunLiu/Swin-UMamba

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from medseg.models.encoders.vmunet_encoder import (
    SS2D, VSSBlock, VSSLayer, PatchEmbed2D, PatchMerging2D
)


# ---------------------------------------------------------------------------
# UNETR-style 解码器 / UNETR-style decoder blocks (self-contained, no MONAI dependency)
# ---------------------------------------------------------------------------

class UnetrBasicBlock(nn.Module):
    """Basic residual conv block for UNETR 解码器。
        Basic residual conv block for UNETR decoder."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class UnetrUpBlock(nn.Module):
    """UNETR-style upsample + concat 跳跃连接。
        UNETR-style upsample + concat skip + conv block."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=2, stride=2)
        self.conv_block = UnetrBasicBlock(out_channels * 2, out_channels,
                                           kernel_size=kernel_size)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


# ---------------------------------------------------------------------------
# VMamba 编码器 / VMamba Encoder (faithful to official Swin-UMamba)
# ---------------------------------------------------------------------------

class VSSMEncoder(nn.Module):
    """VMamba encoder faithful to official Swin-UMamba.

    Key differences from a generic VMamba encoder:
      - VSSLayer has NO downsample inside; PatchMerging2D is separate
      - No LayerNorm on stage outputs
      - forward prepends the stem output to the returned feature list
      - patch_size=2 when used after stem (total stride = 2*2 = 4)
    """
    def __init__(self, patch_size=2, in_chans=48, depths=(2, 2, 9, 2),
                 dims=(96, 192, 384, 768), d_state=16, d_conv=3, expand=2,
                 drop_path_rate=0.2):
        super().__init__()
        num_stages = len(depths)
        self.dims = list(dims)
        embed_dim = self.dims[0]

        self.patch_embed = PatchEmbed2D(patch_size, in_chans, embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(num_stages):
            layer = VSSLayer(
                dim=self.dims[i], depth=depths[i], d_state=d_state,
                d_conv=d_conv, expand=expand,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=None,  # downsample is separate, faithful to official
            )
            self.layers.append(layer)
            if i < num_stages - 1:
                self.downsamples.append(PatchMerging2D(self.dims[i]))

    def forward(self, x):
        """Returns [stem_out, stage0, stage1, stage2, stage3] in (B, C, H, W)."""
        x_ret = []
        x_ret.append(x)  # stem output (prepended, faithful to official)

        x = self.patch_embed(x)  # (B, H/ps, W/ps, C)
        for s, layer in enumerate(self.layers):
            x_out, _ = layer(x)  # VSSBlocks only (no downsample in layer)
            x_ret.append(x_out.permute(0, 3, 1, 2).contiguous())  # (B, C, H, W)
            if s < len(self.downsamples):
                x = self.downsamples[s](x_out)
        return x_ret


# ---------------------------------------------------------------------------
# SwinUMamba: main model (faithful to official Swin-UMamba)
# ---------------------------------------------------------------------------

class SwinUMamba(nn.Module):
    """Swin-UMamba: VMamba encoder + UNETR-style decoder.

    Faithful to https://github.com/JiarunLiu/Swin-UMamba

    Architecture (official):
      1. stem:       Conv2d(k=7, s=2) + InstanceNorm  →  H/2 @ 48
      2. VSSMEncoder: patch_size=2, dims=[96,192,384,768]
                      returns [stem_out, s0, s1, s2, s3]
      3. enc1–enc5:  UnetrBasicBlock at each scale
      4. dec6–dec2:  UnetrUpBlock (5 upsampling stages)
      5. dec1:        UnetrBasicBlock (final)
      6. out_layers:  4 deep-supervision heads (Conv1x1)

    Args:
        in_channels: Input image channels.
        num_classes: Number of output classes.
        img_size: Input image size.
        feat_size: Feature dimensions per scale [48, 96, 192, 384, 768].
        depths: VSSBlock counts per encoder stage.
        d_state: SS2D state dimension.
        drop_path_rate: Stochastic depth rate.
        deep_supervision: Enable deep supervision outputs.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 feat_size=None, depths=None, d_state=16,
                 drop_path_rate=0.2, deep_supervision=False, **kwargs):
        super().__init__()
        if feat_size is None:
            feat_size = [48, 96, 192, 384, 768]
        if depths is None:
            depths = [2, 2, 9, 2]
        self.deep_supervision = deep_supervision
        self.feat_size = feat_size
        hidden_size = feat_size[-1]  # 768

        # Stem: Conv2d(k=7, s=2) + InstanceNorm → H/2 @ feat_size[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, feat_size[0], kernel_size=7, stride=2,
                      padding=3),
            nn.InstanceNorm2d(feat_size[0], eps=1e-5, affine=True),
        )

        # VSSM encoder: patch_size=2 (after stem, total stride=4)
        self.vssm_encoder = VSSMEncoder(
            patch_size=2, in_chans=feat_size[0],
            depths=depths, dims=feat_size[1:],
            d_state=d_state, drop_path_rate=drop_path_rate)

        # 5 encoder blocks (process raw input + VSSM outputs)
        self.encoder1 = UnetrBasicBlock(in_channels, feat_size[0])   # raw → H @ 48
        self.encoder2 = UnetrBasicBlock(feat_size[0], feat_size[1])  # stem → H/2 @ 96
        self.encoder3 = UnetrBasicBlock(feat_size[1], feat_size[2])  # s0  → H/4 @ 192
        self.encoder4 = UnetrBasicBlock(feat_size[2], feat_size[3])  # s1  → H/8 @ 384
        self.encoder5 = UnetrBasicBlock(feat_size[3], feat_size[4])  # s2  → H/16 @ 768

        # 5 decoder upsampling stages
        self.decoder6 = UnetrUpBlock(hidden_size, feat_size[4])  # enc_hidden + enc5 → H/16
        self.decoder5 = UnetrUpBlock(hidden_size, feat_size[3])  # dec4 + enc4 → H/8
        self.decoder4 = UnetrUpBlock(feat_size[3], feat_size[2]) # dec3 + enc3 → H/4
        self.decoder3 = UnetrUpBlock(feat_size[2], feat_size[1]) # dec2 + enc2 → H/2
        self.decoder2 = UnetrUpBlock(feat_size[1], feat_size[0]) # dec1 + enc1 → H

        # Final basic block
        self.decoder1 = UnetrBasicBlock(feat_size[0], feat_size[0])

        # Output heads (deep supervision: 4 heads at different scales)
        self.out_layers = nn.ModuleList([
            nn.Conv2d(feat_size[i], num_classes, 1) for i in range(4)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        input_size = x.shape[2:]

        # Stem
        x1 = self.stem(x)  # H/2 @ 48

        # VSSM encoder: [stem_out, s0, s1, s2, s3]
        vss_outs = self.vssm_encoder(x1)

        # Encoder blocks
        enc1 = self.encoder1(x)              # H @ 48
        enc2 = self.encoder2(vss_outs[0])    # H/2 @ 96
        enc3 = self.encoder3(vss_outs[1])    # H/4 @ 192
        enc4 = self.encoder4(vss_outs[2])    # H/8 @ 384
        enc5 = self.encoder5(vss_outs[3])    # H/16 @ 768
        enc_hidden = vss_outs[4]             # H/32 @ 768

        # Decoder (5 upsampling stages)
        dec4 = self.decoder6(enc_hidden, enc5)  # H/16 @ 768
        dec3 = self.decoder5(dec4, enc4)        # H/8 @ 384
        dec2 = self.decoder4(dec3, enc3)        # H/4 @ 192
        dec1 = self.decoder3(dec2, enc2)        # H/2 @ 96
        dec0 = self.decoder2(dec1, enc1)        # H @ 48
        dec_out = self.decoder1(dec0)           # H @ 48

        if self.deep_supervision:
            feat_out = [dec_out, dec1, dec2, dec3]
            out = []
            for i in range(4):
                pred = self.out_layers[i](feat_out[i])
                if pred.shape[2:] != input_size:
                    pred = F.interpolate(pred, size=input_size,
                                         mode='bilinear', align_corners=False)
                out.append(pred)
            return out
        else:
            out = self.out_layers[0](dec_out)
            if out.shape[2:] != input_size:
                out = F.interpolate(out, size=input_size,
                                    mode='bilinear', align_corners=False)
            return out
