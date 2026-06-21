"""MUCM-Net: Mamba-powered UCM-Net for Skin Lesion Segmentation.

Reference:
    Yuan et al., "MUCM-Net: A Mamba Powered UCM-Net for Skin Lesion
    Segmentation", Exploration of Engineering Materials 2024.
    https://github.com/chunyuyuan/MUCM-Net

Rewritten to match the official source (archs/mucm_dev.py):
    * MambaLayer wraps ``mamba_ssm.Mamba`` (1D SSM, d_state=4, d_conv=4, expand=2).
    * UCMBlock = norm2 -> Mamba(residual) -> fc1 -> DWConv -> GELU -> drop ->
      fc2 -> DWConv -> drop -> +mamba_residual -> +drop_path.
    * DWConv: 3x3 depthwise conv with F.layer_norm over [H, W].
    * OverlapPatchEmbed: 1x1 strided conv + LayerNorm (kernel_size=1, NOT 3).
    * MUCM_Net: encoder1 -> maxpool -> 5 patch-embed stages (block+norm) ->
      bottleneck -> 6-stage decoder with deep supervision, mlp_ratio=1.

Constructor:
    MUCMNet(in_channels=3, num_classes=2, img_size=256, deep_supervision=False, **kwargs)
"""
# Source: https://github.com/chunyuyuan/MUCM-Net

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ---------------------------------------------------------------------------
# MambaLayer (wraps mamba_ssm.Mamba; official hard-depends on mamba_ssm)
# ---------------------------------------------------------------------------

class _MambaLayer(nn.Module):
    def __init__(self, d_model, d_state=4, d_conv=4, expand=2):
        super(_MambaLayer, self).__init__()
        try:
            from mamba_ssm import Mamba  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "MUCM-Net requires the `mamba_ssm` CUDA package. "
                "Install from https://github.com/state-spaces/mamba. "
                "The official MUCM-Net source hard-depends on mamba_ssm."
            ) from e
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.mamba(x)
        return x


# ---------------------------------------------------------------------------
# DWConv
# ---------------------------------------------------------------------------

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.layer_norm(x, [H, W])
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# OverlapPatchEmbed (1x1 strided conv)
# ---------------------------------------------------------------------------

class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# UCMBlock
# ---------------------------------------------------------------------------

class UCMBlock(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1, shift_size=5):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.dim = dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.dwconv = DWConv(mlp_hidden_dim)
        self.dwconv1 = DWConv(mlp_hidden_dim)
        self.act = nn.GELU()
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.drop = nn.Dropout(drop)
        self.fc21 = _MambaLayer(
            d_model=mlp_hidden_dim,
            d_state=4,
            d_conv=4,
            expand=2,
        )
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.norm2(x)
        B, N, C = x.shape
        x1 = x.clone().detach()
        x2 = self.fc21(x1)
        x1 = x2 + x1
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = self.act1(xn)
        x = self.drop(xn)
        x_s = x.reshape(B, C, H * W).contiguous()
        x = x_s.transpose(1, 2)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.dwconv1(x, H, W)
        x = self.drop(x)
        x += x1
        x = x + self.drop_path(x)
        return x


# ---------------------------------------------------------------------------
# MUCM-Net
# ---------------------------------------------------------------------------

class MUCMNet(nn.Module):
    """MUCM-Net for skin lesion segmentation.

    Default embed_dims=[8,16,24,32,48,64,3] matches the original paper.
    mlp_ratio=1 for all UCMBlocks.
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=256,
                 embed_dims=None, num_heads=None, mlp_ratios=None,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=None, sr_ratios=None, deep_supervision=False, **kwargs):
        super().__init__()
        if embed_dims is None:
            embed_dims = [8, 16, 24, 32, 48, 64, 3]
        if depths is None:
            depths = [1, 1, 1]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        if num_heads is None:
            num_heads = [1]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]

        self.num_classes = num_classes
        self.img_size = img_size
        self.deep_supervision = deep_supervision

        self.encoder1 = nn.Conv2d(in_channels, embed_dims[0], 3, stride=1, padding=1)
        self.ebn1 = nn.GroupNorm(4, embed_dims[0])

        self.norm1 = norm_layer(embed_dims[1])
        self.norm2 = norm_layer(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.norm5 = norm_layer(embed_dims[5])

        self.dnorm2 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])
        self.dnorm5 = norm_layer(embed_dims[1])
        self.dnorm6 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block_0_1 = nn.ModuleList([UCMBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.block0 = nn.ModuleList([UCMBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.block1 = nn.ModuleList([UCMBlock(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([UCMBlock(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.block3 = nn.ModuleList([UCMBlock(
            dim=embed_dims[5], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.dblock0 = nn.ModuleList([UCMBlock(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock1 = nn.ModuleList([UCMBlock(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock2 = nn.ModuleList([UCMBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock3 = nn.ModuleList([UCMBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock4 = nn.ModuleList([UCMBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=3, stride=2,
                                              in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 2, patch_size=3, stride=2,
                                              in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                              in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                              in_chans=embed_dims[3], embed_dim=embed_dims[4])
        self.patch_embed5 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                              in_chans=embed_dims[4], embed_dim=embed_dims[5])

        self.decoder0 = nn.Conv2d(embed_dims[5], embed_dims[4], 1, stride=1, padding=0)
        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], 1, stride=1, padding=0)
        self.decoder2 = nn.Conv2d(embed_dims[3], embed_dims[2], 1, stride=1, padding=0)
        self.decoder3 = nn.Conv2d(embed_dims[2], embed_dims[1], 1, stride=1, padding=0)
        self.decoder4 = nn.Conv2d(embed_dims[1], embed_dims[0], 1, stride=1, padding=0)
        self.decoder5 = nn.Conv2d(embed_dims[0], embed_dims[-1], 1, stride=1, padding=0)

        self.dbn0 = nn.GroupNorm(4, embed_dims[4])
        self.dbn1 = nn.GroupNorm(4, embed_dims[3])
        self.dbn2 = nn.GroupNorm(4, embed_dims[2])
        self.dbn3 = nn.GroupNorm(4, embed_dims[1])
        self.dbn4 = nn.GroupNorm(4, embed_dims[0])

        self.finalpre0 = nn.Conv2d(embed_dims[4], num_classes, kernel_size=1)
        self.finalpre1 = nn.Conv2d(embed_dims[3], num_classes, kernel_size=1)
        self.finalpre2 = nn.Conv2d(embed_dims[2], num_classes, kernel_size=1)
        self.finalpre3 = nn.Conv2d(embed_dims[1], num_classes, kernel_size=1)
        self.finalpre4 = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)

        self.final = nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1)

    def forward(self, x, inference_mode=False):
        B = x.shape[0]
        H0, W0 = x.shape[2:]
        # maxpool (/2) + 5 patch-embed strides (/32) => /64; pad to a multiple of 64.
        pH = (64 - H0 % 64) % 64
        pW = (64 - W0 % 64) % 64
        if pH > 0 or pW > 0:
            x = F.pad(x, [0, pW, 0, pH], mode='reflect')

        ### Encoder
        ### Conv Stage
        out = self.encoder1(x)
        out = F.relu(F.max_pool2d(self.ebn1(out), 2, 2))
        t1 = out

        ### Stage 2
        out, H, W = self.patch_embed1(out)
        for i, blk in enumerate(self.block_0_1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2 = out

        ### Stage 3
        out, H, W = self.patch_embed2(out)
        for i, blk in enumerate(self.block0):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t3 = out

        ### Stage 4
        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck
        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = out

        out, H, W = self.patch_embed5(out)
        for i, blk in enumerate(self.block3):
            out = blk(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Decoder
        out = F.relu(F.interpolate(self.dbn0(self.decoder0(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t5)
        if self.deep_supervision and not inference_mode:
            outtpre0 = self.finalpre0(F.interpolate(out, scale_factor=32, mode='bilinear', align_corners=True))

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock0):
            out = blk(out, H, W)
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t4)
        if self.deep_supervision and not inference_mode:
            outtpre1 = self.finalpre1(F.interpolate(out, scale_factor=16, mode='bilinear', align_corners=True))

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t3)
        if self.deep_supervision and not inference_mode:
            outtpre2 = self.finalpre2(F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True))

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t2)
        if self.deep_supervision and not inference_mode:
            outtpre3 = self.finalpre3(F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True))

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock3):
            out = blk(out, H, W)
        out = self.dnorm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t1)
        if self.deep_supervision and not inference_mode:
            outtpre4 = self.finalpre4(F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True))

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock4):
            out = blk(out, H, W)
        out = self.dnorm6(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear'))
        out = self.final(out)

        if out.shape[2:] != (H0, W0):
            out = F.interpolate(out, size=(H0, W0), mode='bilinear', align_corners=False)

        if self.deep_supervision and not inference_mode:
            return (outtpre0, outtpre1, outtpre2, outtpre3, outtpre4), out
        return out
