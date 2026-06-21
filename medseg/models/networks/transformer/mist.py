"""MIST: Medical Image Segmentation Transformer with CAM Decoder.
    MIST: Medical Image Segmentation Transformer with CAM Decoder。

Reference:
    Rahman et al., "MIST: Medical Image Segmentation Transformer With
    Convolutional Attention Mixing (CAM) Decoder", WACV 2024.
    arXiv: 2310.19898. DOI: 10.1109/WACV57701.2024.00047.
    Official code: https://github.com/Rahman-Motiur/MIST

Architecture (faithful port from official source):
    - Encoder: MaxViT (multi-axis vision transformer) via timm features_only.
      'small' scale: channels [96, 192, 384, 768] at strides [4, 8, 16, 32].
      'tiny' scale: channels [64, 128, 256, 512].
      Grayscale input (1 channel) is projected to 3 channels.
    - Decoder: CAM (Convolutional Attention Mixing) decoder, ported from
      lib/MIST.py. Each decoder block uses:
        * Convolutional projected multi-head self-attention (depthwise conv
          projections for Q/K/V + nn.MultiheadAttention)
        * Wide-Focus module (Dilated_Conv): parallel dilated convolutions
          (d=1,2,3) with GELU + dropout
        * Residual connections
      Bottleneck block processes the deepest feature before decoding.
      4 decoder stages fuse encoder skips via concatenation + conv.
    - Deep supervision: 4 prediction heads, each upsampled to input resolution.
      Returns a list [p1, p2, p3, p4] during training (compatible with
      deep_supervision loss). During eval, returns the sum of all heads.

Self-contained: only torch and timm are required.
"""
# Source: https://github.com/Rahman-Motiur/MIST

import os

# Limit huggingface_hub retry/timeout budgets so a network outage does not
# stall model construction for minutes. Must be set before importing timm.
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '3')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '5')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


# ---------------------------------------------------------------------------
# CAM Decoder components (ported from lib/MIST.py)
# ---------------------------------------------------------------------------
class _ConvAttn(nn.Module):
    """Convolutional projected multi-head self-attention.

    Uses depthwise convolutions (groups=channels) to project Q, K, V,
    followed by LayerNorm and nn.MultiheadAttention.
    """

    def __init__(self, channels, num_heads, proj_drop=0.0, kernel_size=3,
                 stride_kv=1, stride_q=1, attention_bias=True):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        padding_kv = kernel_size // 2
        padding_q = kernel_size // 2

        self.conv_q = nn.Conv2d(channels, channels, kernel_size, stride_q,
                                padding_q, bias=attention_bias, groups=channels)
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size, stride_kv,
                                padding_kv, bias=attention_bias, groups=channels)
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size, stride_kv,
                                padding_kv, bias=attention_bias, groups=channels)
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels, bias=attention_bias, batch_first=True,
            num_heads=num_heads,
        )

    def _build_projection(self, x, qkv):
        x1 = F.relu(getattr(self, f'conv_{qkv}')(x))
        x1 = x1.permute(0, 2, 3, 1)
        x1 = getattr(self, f'layernorm_{qkv}')(x1)
        return x1.permute(0, 3, 1, 2)

    def forward(self, x):
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")

        B, C, H, W = x.shape
        q = q.view(B, C, H * W).permute(0, 2, 1)
        k = k.view(B, C, H * W).permute(0, 2, 1)
        v = v.view(B, C, H * W).permute(0, 2, 1)

        x1 = self.attention(query=q, value=v, key=k, need_weights=False)
        x1 = x1[0].permute(0, 2, 1)
        s = int(np.sqrt(x1.shape[2]))
        x1 = x1.view(B, C, s, s)
        x1 = F.dropout(x1, self.proj_drop)
        return x1


class _DilatedConv(nn.Module):
    """Wide-Focus module: parallel dilated convolutions (d=1,2,3)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)

    def forward(self, x):
        x1 = F.dropout(F.gelu(self.conv1(x)), 0.1)
        x2 = F.dropout(F.gelu(self.conv2(x)), 0.1)
        x3 = F.dropout(F.gelu(self.conv3(x)), 0.1)
        added = x1 + x2 + x3
        out = F.dropout(F.gelu(self.conv4(added)), 0.1)
        return out


class _TransformerBlock(nn.Module):
    """Transformer block: conv-attention + conv + Wide-Focus + residuals."""

    def __init__(self, out_channels, num_heads, dpr):
        super().__init__()
        self.attention_output = _ConvAttn(
            channels=out_channels, num_heads=num_heads,
        )
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.layernorm = nn.LayerNorm(out_channels, eps=1e-5)
        self.wide_focus = _DilatedConv(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x1 = self.conv1(x1)
        x2 = x1 + x
        x3 = x2.permute(0, 2, 3, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 3, 1, 2)
        x3 = self.wide_focus(x3)
        x3 = x2 + x3
        return x3


class _BottleneckBlock(nn.Module):
    """Bottleneck block: LayerNorm + convs + maxpool + Transformer."""

    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.trans = _TransformerBlock(out_channels, att_heads, dpr)

    def forward(self, x):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.dropout(x1, 0.3)
        x1 = F.max_pool2d(x1, (2, 2))
        out = self.trans(x1)
        return out


class _DecoderBlock(nn.Module):
    """Decoder block: LayerNorm + upsample + conv + concat skip + conv + Transformer."""

    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, 3, 1, padding=1)
        self.trans = _TransformerBlock(out_channels, att_heads, dpr)

    def forward(self, x, skip):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.upsample(x1)
        x1 = F.relu(self.conv1(x1))
        x1 = torch.cat((skip, x1), dim=1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.dropout(x1, 0.3)
        out = self.trans(x1)
        return out


class _CAMDecoder(nn.Module):
    """CAM (Convolutional Attention Mixing) decoder.

    Takes 4 encoder skip features (shallow to deep) and produces 4
    decoded feature maps for deep supervision.
    """

    def __init__(self):
        super().__init__()
        att_heads = [2, 4, 8, 12, 16, 12, 8, 4, 2]
        filters = [96, 192, 384, 768, 768 * 2, 768, 384, 192, 96]
        blocks = len(filters)
        stochastic_depth_rate = 1.0
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

        # Bottleneck on deepest feature
        self.block_5 = _BottleneckBlock(filters[3], filters[4], att_heads[4], dpr[4])
        # Decoder blocks (deep to shallow)
        self.block_6 = _DecoderBlock(filters[4], filters[5], att_heads[5], dpr[5])
        self.block_7 = _DecoderBlock(filters[5], filters[6], att_heads[6], dpr[6])
        self.block_8 = _DecoderBlock(filters[6], filters[7], att_heads[7], dpr[7])
        self.block_9 = _DecoderBlock(filters[7], filters[8], att_heads[8], dpr[8])

    def forward(self, skip1, skip2, skip3, skip4):
        """Args: skip1 (shallowest, 96ch), skip2 (192ch), skip3 (384ch), skip4 (768ch, deepest)."""
        x = self.block_5(skip4)       # bottleneck
        x = self.block_6(x, skip4)    # decode + fuse skip4
        out4 = x
        x = self.block_7(x, skip3)    # decode + fuse skip3
        out3 = x
        x = self.block_8(x, skip2)    # decode + fuse skip2
        out2 = x
        x = self.block_9(x, skip1)    # decode + fuse skip1
        out1 = x
        return out4, out3, out2, out1


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
def _build_backbone(model_scale='small', in_channels=3, pretrained=True):
    """Create MaxViT backbone in features_only mode.

    Tries the official 'maxxvit_rmlp_small_rw_256' first, then falls back
    to 'maxvit_tiny_tf_224' if not available in the installed timm.
    """
    candidates = {
        'small': ['maxxvit_rmlp_small_rw_256', 'maxvit_rmlp_small_rw_224',
                   'maxvit_small_tf_224'],
        'tiny': ['maxvit_rmlp_tiny_rw_256', 'maxvit_tiny_rw_224',
                  'maxvit_tiny_tf_224'],
    }
    for name in candidates.get(model_scale, candidates['small']):
        try:
            model = load_with_ssl_fallback(
                timm.create_model,
                name,
                features_only=True,
                out_indices=(1, 2, 3, 4),  # skip stem (index 0), keep 4 stages
                pretrained=pretrained,
                in_chans=3,
            )
            # Ensure we got exactly 4 feature maps
            if len(model.feature_info.channels()) != 4:
                # Fallback: try without out_indices and slice later
                model = load_with_ssl_fallback(
                    timm.create_model,
                    name,
                    features_only=True,
                    pretrained=pretrained,
                    in_chans=3,
                )
            return model, name
        except Exception:
            continue
    raise RuntimeError(
        f"No MaxViT model available for scale='{model_scale}' in installed timm. "
        "Please install a timm version that includes MaxViT models."
    )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class MIST(nn.Module):
    """MIST: Medical Image Segmentation Transformer with CAM Decoder.

    Args:
        in_channels: number of input image channels (1=grayscale, 3=RGB).
        num_classes: number of segmentation classes.
        img_size: nominal input spatial size.
        model_scale: 'small' or 'tiny' MaxViT backbone.
        pretrained: whether to load pretrained MaxViT weights.
        deep_supervision: if True, return list of 4 outputs during training.
    """

    _PATCH_MULT = 32

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 model_scale='small', pretrained=True,
                 deep_supervision=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_scale = model_scale
        self.deep_supervision = deep_supervision

        # Grayscale -> 3 channels
        if in_channels == 1:
            self.input_proj = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=1),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_proj = nn.Identity()

        # MaxViT backbone
        self.backbone, self._backbone_name = _build_backbone(
            model_scale, in_channels=3, pretrained=pretrained,
        )
        # MaxViT requires input divisible by window size; the official MIST
        # code resizes to the backbone's native resolution (256 for rw_256).
        self._native_size = 256 if '256' in self._backbone_name else 224
        enc_channels = list(self.backbone.feature_info.channels())
        # If backbone returns 5 features (stem + 4 stages), drop the stem
        if len(enc_channels) == 5:
            enc_channels = enc_channels[1:]
            self._drop_stem = True
        else:
            self._drop_stem = False
        if len(enc_channels) != 4:
            raise RuntimeError(
                f'Expected 4 encoder features, got {len(enc_channels)}: {enc_channels}'
            )
        # timm returns shallow -> deep: [96, 192, 384, 768] for 'small'
        c1, c2, c3, c4 = enc_channels  # strides 4, 8, 16, 32

        # CAM decoder
        self.decoder = _CAMDecoder()

        # Deep supervision prediction heads (match decoder output channels)
        self.out_head1 = nn.Conv2d(c4, num_classes, 1)   # 768 -> n_class (out4)
        self.out_head2 = nn.Conv2d(c3, num_classes, 1)   # 384 -> n_class (out3)
        self.out_head3 = nn.Conv2d(c2, num_classes, 1)   # 192 -> n_class (out2)
        self.out_head4 = nn.Conv2d(c1, num_classes, 1)   # 96  -> n_class (out1)

    def _pad_to_multiple(self, x, mult):
        H, W = x.shape[-2:]
        pad_h = (mult - H % mult) % mult
        pad_w = (mult - W % mult) % mult
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w)

    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.input_proj(x)

        # Resize to backbone native resolution (official MIST does this)
        native = self._native_size
        if x.shape[-2] != native or x.shape[-1] != native:
            x_enc = F.interpolate(x, size=(native, native), mode='bilinear',
                                  align_corners=False)
        else:
            x_enc = x

        # Encoder: timm returns shallow -> deep
        feats = self.backbone(x_enc)
        if self._drop_stem:
            feats = feats[1:]  # drop stem feature
        f1, f2, f3, f4 = feats  # strides 4, 8, 16, 32

        # CAM decoder: skip1=f1(shallow), skip2=f2, skip3=f3, skip4=f4(deep)
        out4, out3, out2, out1 = self.decoder(f1, f2, f3, f4)

        # Deep supervision heads
        p1 = self.out_head1(out4)  # deepest decoder output
        p2 = self.out_head2(out3)
        p3 = self.out_head3(out2)
        p4 = self.out_head4(out1)  # shallowest decoder output

        # Upsample all predictions to original input resolution
        p1 = F.interpolate(p1, size=(H, W), mode='bilinear', align_corners=False)
        p2 = F.interpolate(p2, size=(H, W), mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=(H, W), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(H, W), mode='bilinear', align_corners=False)

        if self.training and self.deep_supervision:
            return [p1, p2, p3, p4]
        else:
            return p1 + p2 + p3 + p4
