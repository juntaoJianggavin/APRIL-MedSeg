"""ViM-UNet: Vision Mamba for Biomedical Segmentation.

Reference:
    Archit et al., "ViM-UNet: Vision Mamba for Biomedical Segmentation",
    MIDL 2024.
    https://github.com/constantinpape/torch-em

Rewritten to match the official source (torch_em/model/vim.py + unetr.py):
    * ViM encoder: Vision Mamba with bidirectional Mamba layers
      (bimamba_type="v2"), patch_size=16, embed_dim=192 (vim_t), RMSNorm,
      middle cls token, absolute position embedding, depth=24 (12 bi-pairs).
      The encoder outputs single-scale features at H/patch_size resolution.
    * UNETR decoder (use_skip_connection=False): progressive upsampling with
      a Deconv2DBlock side-path + Decoder with concat, ConvBlock2d
      (InstanceNorm), final 1x1 output conv.

The official ViM-UNet depends on the ``vim`` package (hustvl/Vim fork) and
``torch_em``. This reimplementation depends only on ``mamba_ssm.Mamba`` for
the SSM core, matching the framework's convention for all Mamba models.

Constructor:
    ViMUNet(in_channels=3, num_classes=2, img_size=224, model_type="vim_t", **kwargs)
"""
# Source: https://github.com/constantinpape/torch-em (torch_em/model/vim.py, unetr.py)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mamba SSM wrapper (hard dependency on mamba_ssm, matching official source)
# ---------------------------------------------------------------------------

class _MambaSSM(nn.Module):
    """Wrapper around ``mamba_ssm.Mamba`` (1D SSM).

    Interface: (B, L, D) -> (B, L, D)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        try:
            from mamba_ssm import Mamba  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ViM-UNet requires the `mamba_ssm` CUDA package. "
                "Install from https://github.com/state-spaces/mamba. "
                "The official ViM-UNet (torch_em) hard-depends on the Vim "
                "package which itself requires mamba_ssm."
            ) from e
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mamba(x)


# ---------------------------------------------------------------------------
# RMSNorm (matching Vim's rms_norm_fn / RMSNorm)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """RMSNorm as used in Vision Mamba (Vim)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True)
        norm = norm.add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# ViM (Vision Mamba) encoder
# ---------------------------------------------------------------------------

class _VimLayer(nn.Module):
    """Single Vim Mamba layer with residual management.

    Matches the ``T2TLayer`` / block in hustvl/Vim:
        if residual is None:
            residual = hidden_states
            hidden_states = norm(hidden_states)
        else:
            hidden_states = residual + drop_path(hidden_states)
            residual = hidden_states
            hidden_states = norm(hidden_states)
        hidden_states = mamba(hidden_states)
        return hidden_states, residual
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, drop_path_rate=0.0):
        super().__init__()
        self.norm = _RMSNorm(dim)
        self.mixer = _MambaSSM(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, hidden_states, residual=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = residual + self.drop_path(hidden_states)
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


class _VimEncoder(nn.Module):
    """Vision Mamba (ViM) encoder with bidirectional Mamba (bimamba_type="v2").

    Matches ``get_vim_encoder`` / ``ViM`` from torch_em/model/vim.py:
        * patch_embed: Conv2d patchify (patch_size=16)
        * cls_token inserted at middle position (use_middle_cls_token=True)
        * absolute position embedding (if_abs_pos_embed=True)
        * 24 Mamba layers processed as 12 bidirectional pairs (bimamba_type="v2"):
            forward = layer[i*2](hidden, residual)
            backward = layer[i*2+1](hidden.flip([1]), residual.flip([1]))
            hidden = forward + backward.flip([1])
        * final RMSNorm (norm_f)
        * remove cls token, reshape to (B, C, H/p, W/p)
    """

    def __init__(self, in_channels=3, embed_dim=192, depth=24,
                 patch_size=16, img_size=224,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.img_size = img_size

        # patch embedding (Conv2d patchify)
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size)

        # cls token (middle position)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # absolute position embedding (sized for default img_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.0)

        # drop path schedule (defaults to 0 for ViM-UNet)
        drop_path_rates = [0.0] * depth

        # 24 Mamba layers (12 bidirectional pairs)
        self.layers = nn.ModuleList([
            _VimLayer(embed_dim, d_state=d_state, d_conv=d_conv,
                      expand=expand, drop_path_rate=drop_path_rates[i])
            for i in range(depth)
        ])

        # final norm
        self.norm_f = _RMSNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        B = x.shape[0]
        # patch embed: (B, C, H, W) -> (B, embed_dim, H/p, W/p)
        x = self.patch_embed(x)
        Hp, Wp = x.shape[2], x.shape[3]
        # flatten spatial: (B, embed_dim, H/p*W/p) -> (B, H/p*W/p, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        M = x.shape[1]  # number of patch tokens

        # cls token at middle position (use_middle_cls_token=True)
        token_position = M // 2
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat(
            (x[:, :token_position, :], cls_token, x[:, token_position:, :]),
            dim=1)

        # absolute position embedding (interpolate if size mismatch)
        if x.shape[1] != self.pos_embed.shape[1]:
            pe = self.pos_embed
            cls_pe = pe[:, :1, :]
            patch_pe = pe[:, 1:, :]  # (1, num_patches, embed_dim)
            patch_pe = patch_pe.transpose(1, 2).reshape(1, self.embed_dim, -1)
            # infer original grid size
            orig_n = patch_pe.shape[2]
            orig_size = int(math.sqrt(orig_n))
            patch_pe = patch_pe.reshape(1, self.embed_dim, orig_size, orig_size)
            patch_pe = F.interpolate(
                patch_pe, size=(Hp, Wp), mode='bicubic', align_corners=False)
            patch_pe = patch_pe.reshape(1, self.embed_dim, -1).transpose(1, 2)
            pe = torch.cat([cls_pe, patch_pe], dim=1)
            x = x + pe
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # bidirectional Mamba (12 pairs)
        residual = None
        hidden_states = x
        for i in range(self.depth // 2):
            # forward
            hidden_states_f, residual_f = self.layers[i * 2](
                hidden_states, residual)
            # backward (flip sequence dim)
            flip_residual = None if residual is None else residual.flip([1])
            hidden_states_b, residual_b = self.layers[i * 2 + 1](
                hidden_states.flip([1]), flip_residual)
            # combine
            hidden_states = hidden_states_f + hidden_states_b.flip([1])
            residual = residual_f + residual_b.flip([1])

        # final norm
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
        hidden_states = self.norm_f(residual)

        # remove cls token (at middle position)
        token_position = M // 2
        x = torch.cat(
            (hidden_states[:, :token_position, :],
             hidden_states[:, token_position + 1:, :]),
            dim=1)

        # reshape: (B, M, C) -> (B, C, H/p, W/p)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)
        return x


# ---------------------------------------------------------------------------
# UNETR decoder building blocks (from torch_em/model/unet.py)
# ---------------------------------------------------------------------------

class _ConvBlock2d(nn.Module):
    """ConvBlock2d: InstanceNorm -> Conv3x3 -> ReLU -> InstanceNorm -> Conv3x3 -> ReLU.

    Matches torch_em ConvBlock (norm="InstanceNorm", default).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _Upsampler2d(nn.Module):
    """Upsampler2d: bilinear interpolate -> Conv1x1.

    Matches torch_em Upsampler2d (mode="bilinear").
    """

    def __init__(self, scale_factor, in_channels, out_channels):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=self.scale_factor,
            mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class _Deconv2DBlock(nn.Module):
    """Deconv2DBlock: Upsampler2d -> Conv3x3 -> BatchNorm2d -> ReLU.

    Matches torch_em Deconv2DBlock (use_conv_transpose=False).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            _Upsampler2d(scale_factor=2,
                         in_channels=in_channels, out_channels=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class _Decoder(nn.Module):
    """UNet-style decoder with concat skip connections.

    Matches torch_em Decoder: at each step, upsample -> concat(skip) -> conv block.
    """

    def __init__(self, features, scale_factors):
        super().__init__()
        self.blocks = nn.ModuleList([
            _ConvBlock2d(inc, outc)
            for inc, outc in zip(features[:-1], features[1:])
        ])
        self.samplers = nn.ModuleList([
            _Upsampler2d(factor, inc, outc)
            for factor, inc, outc in zip(scale_factors, features[:-1], features[1:])
        ])
        self.out_channels = features[-1]

    @staticmethod
    def _crop(x, shape):
        shape_diff = [(xsh - sh) // 2 for xsh, sh in zip(x.shape, shape)]
        crop = tuple(slice(sd, xsh - sd) for sd, xsh in zip(shape_diff, x.shape))
        return x[crop]

    def _concat(self, x1, x2):
        return torch.cat([x1, self._crop(x2, x1.shape)], dim=1)

    def forward(self, x, encoder_inputs):
        for block, sampler, from_encoder in zip(
                self.blocks, self.samplers, encoder_inputs):
            x = sampler(x)
            x = block(self._concat(x, from_encoder))
        return x


# ---------------------------------------------------------------------------
# ViM-UNet
# ---------------------------------------------------------------------------

class ViMUNet(nn.Module):
    """ViM-UNet: Vision Mamba encoder + UNETR decoder (no skip connections).

    Matches ``get_vimunet_model`` from torch_em/model/vim.py which builds:
        UNETR(encoder=ViM(...), use_skip_connection=False, final_activation="Sigmoid")

    The UNETR decoder (use_skip_connection=False) has:
        * A Deconv2DBlock side-path creating multi-scale features (z9, z6, z3, z0)
        * A Decoder that concatenates upsampled features with the side-path outputs
        * deconv_out + decoder_head + out_conv at full resolution

    Args:
        in_channels: Input channel count.
        num_classes: Output channel count.
        img_size: Reference image size (for pos_embed sizing; the model accepts
            any size divisible by patch_size).
        model_type: ViM backbone type — "vim_t" (embed_dim=192, depth=24),
            "vim_s" (embed_dim=384, depth=24), or "vim_b" (embed_dim=768, depth=24).
    """

    # ViM backbone configs (from get_vim_encoder)
    _VIM_CONFIGS = {
        "vim_t": {"embed_dim": 192, "depth": 24},
        "vim_s": {"embed_dim": 384, "depth": 24},
        "vim_b": {"embed_dim": 768, "depth": 24},
    }

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 model_type="vim_t", **kwargs):
        super().__init__()
        cfg = self._VIM_CONFIGS.get(model_type, self._VIM_CONFIGS["vim_t"])
        embed_dim = cfg["embed_dim"]
        depth = cfg["depth"]
        patch_size = 16
        self.patch_size = patch_size
        self.num_classes = num_classes

        # ViM encoder
        self.encoder = _VimEncoder(
            in_channels=in_channels, embed_dim=embed_dim, depth=depth,
            patch_size=patch_size, img_size=img_size)

        # UNETR decoder parameters (from UNETR.__init__)
        dec_depth = 3
        initial_features = 64
        gain = 2
        features_decoder = [
            initial_features * gain ** i for i in range(dec_depth + 1)
        ][::-1]  # [512, 256, 128, 64]
        scale_factors = dec_depth * [2]  # [2, 2, 2]
        self.embed_dim = embed_dim

        # Deconv side-path (use_skip_connection=False)
        # Channels sized so Decoder concat works: skip_ch = feat[i] - feat[i+1]
        # deconv1: embed_dim -> feat[0]-feat[1] (e.g. 192 -> 256)
        # deconv2: feat[0]-feat[1] -> feat[1]-feat[2] (e.g. 256 -> 128)
        # deconv3: feat[1]-feat[2] -> feat[2]-feat[3] (e.g. 128 -> 64)
        # deconv4: feat[2]-feat[3] -> feat[3] (e.g. 64 -> 64)
        d1_out = features_decoder[0] - features_decoder[1]  # 256
        d2_out = features_decoder[1] - features_decoder[2]  # 128
        d3_out = features_decoder[2] - features_decoder[3]  # 64
        d4_out = features_decoder[3]                         # 64
        self.deconv1 = _Deconv2DBlock(embed_dim, d1_out)
        self.deconv2 = _Deconv2DBlock(d1_out, d2_out)
        self.deconv3 = _Deconv2DBlock(d2_out, d3_out)
        self.deconv4 = _Deconv2DBlock(d3_out, d4_out)

        # base: ConvBlock2d(embed_dim -> features_decoder[0])
        self.base = _ConvBlock2d(embed_dim, features_decoder[0])

        # Decoder with concat (features and scale_factors as in UNETR)
        self.decoder = _Decoder(
            features=features_decoder,
            scale_factors=scale_factors[::-1])

        # deconv_out + decoder_head + out_conv
        self.deconv_out = _Upsampler2d(
            scale_factor=2,
            in_channels=features_decoder[-1],
            out_channels=features_decoder[-1])
        self.decoder_head = _ConvBlock2d(
            2 * features_decoder[-1], features_decoder[-1])
        self.out_conv = nn.Conv2d(features_decoder[-1], num_classes, 1)

    def forward(self, x):
        original_h, original_w = x.shape[-2], x.shape[-1]

        # pad to multiple of patch_size
        ps = self.patch_size
        pH = ((x.shape[-2] + ps - 1) // ps) * ps
        pW = ((x.shape[-1] + ps - 1) // ps) * ps
        if pH != x.shape[-2] or pW != x.shape[-1]:
            x = F.pad(x, [0, pW - x.shape[-1], 0, pH - x.shape[-2]],
                      mode='reflect')

        # ViM encoder: (B, C, H, W) -> (B, embed_dim, H/ps, W/ps)
        z12 = self.encoder(x)

        # Deconv side-path (progressive upsampling, no skip from encoder)
        z9 = self.deconv1(z12)
        z6 = self.deconv2(z9)
        z3 = self.deconv3(z6)
        z0 = self.deconv4(z3)

        # Decoder: base -> 3-stage upsampling with concat(z9, z6, z3)
        x = self.base(z12)
        x = self.decoder(x, encoder_inputs=[z9, z6, z3])

        # final upsample + concat(z0) + head
        x = self.deconv_out(x)
        x = torch.cat([x, z0], dim=1)
        x = self.decoder_head(x)
        x = self.out_conv(x)

        # interpolate to original resolution
        if x.shape[-2:] != (original_h, original_w):
            x = F.interpolate(
                x, size=(original_h, original_w),
                mode='bilinear', align_corners=False)
        return x


__all__ = ['ViMUNet']
