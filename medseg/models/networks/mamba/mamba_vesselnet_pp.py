"""MambaVesselNet: Hybrid CNN-Mamba for Medical Image Segmentation (2D).

Reference:
    "MambaVesselNet: A Hybrid CNN-Mamba Architecture for Medical Image
    Segmentation", https://github.com/CC0117/MambaVesselNet

Rewritten to match the official source (model/mvn/mvn.py):
    * MambaLayer: flatten spatial -> LayerNorm -> Mamba(bimamba_type="v2")
      -> reshape.  The bimamba_type="v2" bidirectional processing (forward +
      backward flip, then sum) is implemented manually since the standard
      ``mamba_ssm.Mamba`` does not support the ``bimamba_type`` argument.
    * MambaBlock: (optional downsample) -> depth MambaLayers -> LayerNorm ->
      MlpChannel (1x1 Conv -> GELU -> 1x1 Conv).
    * mvnNet: 5-level UNet with UnetrBasicBlock encoder, strided-conv
      downsampling, 4 encoder MambaBlocks + 4 decoder MambaBlocks at the
      bottleneck resolution, UnetrUpBlock decoder, 1x1 output.

This is a 2D adaptation of the official 3D model — Conv3d->Conv2d,
InstanceNorm3d->InstanceNorm2d, ConvTranspose3d->ConvTranspose2d, etc.

Constructor:
    MambaVesselNetPP(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/CC0117/MambaVesselNet (model/mvn/mvn.py)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LayerNorm (channels_first / channels_last, matching official mvn.py)
# ---------------------------------------------------------------------------

class _LayerNorm(nn.Module):
    """LayerNorm supporting channels_first (B,C,H,W) or channels_last (B,H,W,C)."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        else:  # channels_first: (B, C, H, W)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# ---------------------------------------------------------------------------
# Mamba SSM wrapper (hard dependency on mamba_ssm)
# ---------------------------------------------------------------------------

class _MambaSSM(nn.Module):
    """Mamba SSM with bimamba_type='v2' (bidirectional: forward + backward, sum).

    Interface: (B, L, D) -> (B, L, D)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        try:
            from mamba_ssm import Mamba  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "MambaVesselNet requires the `mamba_ssm` CUDA package. "
                "Install from https://github.com/state-spaces/mamba. "
                "The official source uses Mamba(bimamba_type='v2')."
            ) from e
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand)

    def forward(self, x):
        # bimamba_type="v2": process forward and backward, then sum
        x_f = self.mamba(x)
        x_b = self.mamba(x.flip([1])).flip([1])
        return x_f + x_b


# ---------------------------------------------------------------------------
# MambaLayer (matching official mvn.py)
# ---------------------------------------------------------------------------

class _MambaLayer(nn.Module):
    """Flatten spatial -> LayerNorm -> Mamba(bimamba v2) -> reshape.

    (B, C, H, W) -> (B, C, H, W)
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = _MambaSSM(dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  # (B, N, C)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


# ---------------------------------------------------------------------------
# MlpChannel (1x1 Conv -> GELU -> 1x1 Conv, matching official mvn.py)
# ---------------------------------------------------------------------------

class _MlpChannel(nn.Module):
    """Channel MLP using 1x1 convolutions."""

    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# MambaBlock (matching official mvn.py)
# ---------------------------------------------------------------------------

class _MambaBlock(nn.Module):
    """MambaBlock: (optional downsample) -> depth MambaLayers -> LayerNorm -> MLP."""

    def __init__(self, in_channels, out_channels, depth=2, downsample=True):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
                _LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            )
        self.mamba_layers = nn.Sequential(
            *[_MambaLayer(dim=out_channels) for _ in range(depth)]
        )
        self.norm = _LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.mlp = _MlpChannel(out_channels, out_channels * 4)

    def forward(self, x):
        if self.downsample:
            x = self.downsample_layer(x)
        x = self.mamba_layers(x)
        x = self.norm(x)
        x = self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# 2D equivalents of MONAI UnetrBasicBlock / UnetrUpBlock
# ---------------------------------------------------------------------------

class _UnetrBasicBlock2D(nn.Module):
    """2D equivalent of MONAI UnetrBasicBlock.

    Structure (matching MONAI):
        layer = Conv-Norm-Act(in -> out)
        conv  = Conv-Norm-Act(out -> out)
        if res_block: residual = Conv1x1(in -> out)
        forward: out = layer(x); if res: out += residual(x); out = conv(out)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 res_block=True):
        super().__init__()
        pad = kernel_size // 2
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=pad),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size,
                      stride=1, padding=pad),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        if res_block:
            self.residual = nn.Conv2d(in_channels, out_channels, 1,
                                      stride=stride)

    def forward(self, x):
        out = self.layer(x)
        if hasattr(self, "residual"):
            out = out + self.residual(x)
        out = self.conv(out)
        return out


class _UnetrUpBlock2D(nn.Module):
    """2D equivalent of MONAI UnetrUpBlock.

    Structure (matching MONAI):
        transp_conv = ConvTranspose2d(in -> out, upsample_kernel, stride)
        conv_block  = UnetrBasicBlock2D(out*2 -> out)
        forward: out = transp_conv(x); out = cat(out, skip); out = conv_block(out)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 upsample_kernel_size=2, res_block=True):
        super().__init__()
        self.transp_conv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=upsample_kernel_size, stride=upsample_kernel_size)
        self.conv_block = _UnetrBasicBlock2D(
            out_channels * 2, out_channels, kernel_size, 1, res_block)

    def forward(self, x, skip):
        out = self.transp_conv(x)
        if out.shape[-2:] != skip.shape[-2:]:
            out = F.interpolate(out, size=skip.shape[-2:],
                                mode='bilinear', align_corners=False)
        out = torch.cat([out, skip], dim=1)
        out = self.conv_block(out)
        return out


def _downsample_layer(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=kernel_size, stride=stride),
        _LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
    )


# ---------------------------------------------------------------------------
# MambaVesselNet (2D adaptation of mvnNet)
# ---------------------------------------------------------------------------

class MambaVesselNetPP(nn.Module):
    """MambaVesselNet 2D — hybrid CNN-Mamba UNet.

    Matches the official ``mvnNet`` architecture (model/mvn/mvn.py):
        * 5-level encoder with UnetrBasicBlock + strided-conv downsample
        * 4 encoder MambaBlocks + 4 decoder MambaBlocks at bottleneck res
        * 4-level decoder with UnetrUpBlock + final UnetrBasicBlock
        * 1x1 output conv

    Args:
        in_channels: Input channel count.
        num_classes: Output channel count.
        img_size: Spatial size (used only for documentation; the network
            accepts any size divisible by 2^4 = 16).
        feature_dims: Channel dimensions for the 5 encoder levels.
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 feature_dims=None, **kwargs):
        super().__init__()
        if feature_dims is None:
            feature_dims = [48, 96, 192, 384, 768]
        D = feature_dims
        self.num_classes = num_classes

        # ---- Encoder ----
        self.enc_conv1 = _UnetrBasicBlock2D(in_channels, D[0])
        self.downsample1 = _downsample_layer(D[0], D[0])

        self.enc_conv2 = _UnetrBasicBlock2D(D[0], D[1])
        self.downsample2 = _downsample_layer(D[1], D[1])

        self.enc_conv3 = _UnetrBasicBlock2D(D[1], D[2])
        self.downsample3 = _downsample_layer(D[2], D[2])

        self.enc_conv4 = _UnetrBasicBlock2D(D[2], D[3])
        self.downsample4 = _downsample_layer(D[3], D[3])

        self.enc_conv5 = _UnetrBasicBlock2D(D[3], D[4])

        # ---- Encoder Mamba Blocks (at bottleneck resolution) ----
        self.enc_mamba1 = _MambaBlock(D[4], D[4], depth=2, downsample=False)
        self.enc_mamba2 = _MambaBlock(D[4], D[4], depth=2, downsample=False)
        self.enc_mamba3 = _MambaBlock(D[4], D[4], depth=2, downsample=False)
        self.enc_mamba4 = _MambaBlock(D[4], D[4], depth=2, downsample=False)

        # ---- Decoder Mamba Blocks (at bottleneck resolution) ----
        self.dec_mamba1 = _MambaBlock(D[4], D[4], depth=2, downsample=False)
        self.dec_mamba2 = _MambaBlock(D[4], D[4], depth=2, downsample=False)
        self.dec_mamba3 = _MambaBlock(D[4], D[4], depth=2, downsample=False)
        self.dec_mamba4 = _MambaBlock(D[4], D[4], depth=2, downsample=False)

        # ---- Decoder ----
        self.dec_conv5 = _UnetrUpBlock2D(D[4], D[3])
        self.dec_conv4 = _UnetrUpBlock2D(D[3], D[2])
        self.dec_conv3 = _UnetrUpBlock2D(D[2], D[1])
        self.dec_conv2 = _UnetrUpBlock2D(D[1], D[0])
        self.dec_conv1 = _UnetrBasicBlock2D(D[0], D[0])

        # ---- Output ----
        self.out = nn.Conv2d(D[0], num_classes, 1)

    def forward(self, x):
        original_h, original_w = x.shape[-2], x.shape[-1]

        # pad to multiple of 16
        pH = ((x.shape[-2] + 15) // 16) * 16
        pW = ((x.shape[-1] + 15) // 16) * 16
        if pH != x.shape[-2] or pW != x.shape[-1]:
            x = F.pad(x, [0, pW - x.shape[-1], 0, pH - x.shape[-2]],
                      mode='reflect')

        # Encoder
        enc1 = self.enc_conv1(x)
        enc1_down = self.downsample1(enc1)

        enc2 = self.enc_conv2(enc1_down)
        enc2_down = self.downsample2(enc2)

        enc3 = self.enc_conv3(enc2_down)
        enc3_down = self.downsample3(enc3)

        enc4 = self.enc_conv4(enc3_down)
        enc4_down = self.downsample4(enc4)

        enc5 = self.enc_conv5(enc4_down)

        # Encoder Mamba Blocks
        x = self.enc_mamba1(enc5)
        x = self.enc_mamba2(x)
        x = self.enc_mamba3(x)
        x = self.enc_mamba4(x)

        # Decoder Mamba Blocks
        x = self.dec_mamba1(x)
        x = self.dec_mamba2(x)
        x = self.dec_mamba3(x)
        x = self.dec_mamba4(x)

        # Decoder
        dec5 = self.dec_conv5(x, enc4)
        dec4 = self.dec_conv4(dec5, enc3)
        dec3 = self.dec_conv3(dec4, enc2)
        dec2 = self.dec_conv2(dec3, enc1)

        x = self.dec_conv1(dec2)
        out = self.out(x)

        # interpolate to original resolution
        if out.shape[-2:] != (original_h, original_w):
            out = F.interpolate(out, size=(original_h, original_w),
                                mode='bilinear', align_corners=False)
        return out


__all__ = ['MambaVesselNetPP']
