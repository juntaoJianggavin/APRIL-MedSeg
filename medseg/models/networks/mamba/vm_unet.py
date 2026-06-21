"""VM-UNet: Vision Mamba UNet for Medical Image Segmentation.
    VM-UNet: Vision Mamba UNet for 医学的 图像 分割。

Faithful reimplementation based on the official source:
    https://github.com/JCruan519/VM-UNet/blob/main/models/vmunet/vmamba.py

Uses Visual State Space (VSS) blocks in a U-Net architecture for
efficient 2D medical image segmentation with linear complexity.

Key components (matching original source):
    - SS2D: 4-direction selective scan via ``selective_scan_fn`` from mamba_ssm
    - VSSBlock: LayerNorm → SS2D → residual (+ drop_path)
    - VSSLayer/VSSLayer_up: stacked VSSBlocks with down/upsample

Requires ``mamba_ssm`` (and ``causal-conv1d``).
"""
# Source: https://github.com/JCruan519/VM-UNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from functools import partial
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# SS2D: 2D Selective Scan (faithful to vmamba.py)
# ---------------------------------------------------------------------------
class SS2D(nn.Module):
    """2D Selective Scan: 4-direction scan via ``selective_scan_fn``.

    Faithful port of the SS2D class from the official VM-UNet source
    (``models/vmunet/vmamba.py``).  Uses ``selective_scan_fn`` from
    ``mamba_ssm.ops.selective_scan_interface`` for the CUDA-accelerated
    selective scan, with 4-direction scanning (HW forward, HW backward,
    WH forward, WH backward).

    Hard-depends on ``mamba_ssm`` and ``causal-conv1d``.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = (
            math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        )

        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except ImportError as e:
            raise RuntimeError(
                "VM-UNet SS2D requires `mamba_ssm` with `causal-conv1d`. "
                "Install from https://github.com/state-spaces/mamba."
            ) from e
        self.selective_scan = selective_scan_fn

        # Input projection (x, z branches)
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias)

        # Depthwise conv on x branch
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            groups=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # Per-direction projection layers (4 directions)
        self.x_proj = tuple(
            nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
            )
            for _ in range(4)
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = tuple(
            self._dt_init(
                self.dt_rank, self.d_inner, dt_scale, dt_init,
                dt_min, dt_max, dt_init_floor)
            for _ in range(4)
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # SSM A and D parameters (4 directions)
        self.A_logs = self._A_log_init(self.d_state, self.d_inner,
                                       copies=4, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=4, merge=True)

        # Output
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    # --- Initializers (from original vmamba.py) ---
    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                 **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
        if merge:
            A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
        if merge:
            D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # --- 4-direction forward (faithful to original forward_corev0) ---
    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        # Stack HW and WH orderings, then their flips:
        # [HW, WH, HW_flip, WH_flip]
        x_hw = x.view(B, -1, L)
        x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        x_hw_wh = torch.stack([x_hw, x_wh], dim=1).view(B, 2, -1, L)
        xs = torch.cat(
            [x_hw_wh, torch.flip(x_hw_wh, dims=[-1])], dim=1
        )  # (B, K, d_inner, L)

        # Compute dt, B, C via einsum with stacked projection weights
        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            xs.view(B, K, -1, L), self.x_proj_weight,
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.view(B, K, -1, L), self.dt_projs_weight,
        )

        xs = xs.float().view(B, -1, L)          # (B, K*d_inner, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)       # (B, K, d_state, L)
        Cs = Cs.float().view(B, K, -1, L)       # (B, K, d_state, L)
        Ds = self.Ds.float().view(-1)            # (K*d_inner,)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # Un-flip the reversed directions
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)

        # Transpose WH results back to HW layout
        wh_y = (torch.transpose(
            out_y[:, 1].view(B, -1, W, H),
            dim0=2, dim1=3,
        ).contiguous().view(B, -1, L))

        invwh_y = (torch.transpose(
            inv_y[:, 1].view(B, -1, W, H),
            dim0=2, dim1=3,
        ).contiguous().view(B, -1, L))

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # --- Forward (faithful to original SS2D.forward) ---
    def forward(self, x: torch.Tensor, **kwargs):
        # Input: (B, H, W, C) in channels-last layout (matching original)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, H, W, d_inner) each

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (B, d_inner, H, W)

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32

        # Sum 4 directions (matching original: y = y1 + y2 + y3 + y4)
        y = y1 + y2 + y3 + y4

        # Transpose back to (B, H, W, d_inner) and apply z-gate
        y = y.transpose(1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# VSSBlock (faithful to vmamba.py)
# ---------------------------------------------------------------------------
class VSSBlock(nn.Module):
    """VSS block: LayerNorm → SS2D → residual (faithful to source)."""

    def __init__(self, hidden_dim, drop_path=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate=0.0, d_state=16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim, dropout=attn_drop_rate,
            d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


# DropPath from timm (or simple fallback)
try:
    from timm.layers import DropPath
except ImportError:
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            if not self.training or self.drop_prob == 0.0:
                return x
            keep = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = torch.bernoulli(
                torch.full(shape, keep, device=x.device))
            return x * mask / keep


# ---------------------------------------------------------------------------
# PatchEmbed / PatchMerging / PatchExpand (faithful to vmamba.py)
# ---------------------------------------------------------------------------
class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96,
                 norm_layer=None):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)  # (B, H, W, C)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(
            x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
            p1=self.dim_scale, p2=self.dim_scale,
            c=C // self.dim_scale)
        x = self.norm(x)
        return x


class FinalPatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(
            x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
            p1=self.dim_scale, p2=self.dim_scale,
            c=C // self.dim_scale)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# VSSLayer / VSSLayer_up (faithful to vmamba.py)
# ---------------------------------------------------------------------------
class VSSLayer(nn.Module):
    def __init__(self, dim, depth, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list)
                else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)
        ])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSLayer_up(nn.Module):
    def __init__(self, dim, depth, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, upsample=None,
                 use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list)
                else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)
        ])
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# VMUNet (faithful to vmamba.py VSSM architecture)
# ---------------------------------------------------------------------------
class VMUNet(nn.Module):
    """VM-UNet: Vision Mamba UNet (faithful to official source).

    Architecture matches the original VSSM from ``vmamba.py``:
    PatchEmbed → VSSLayers (encoder) → VSSLayers_up (decoder with skips)
    → FinalPatchExpand → Conv output.

    Requires ``mamba_ssm`` and ``causal-conv1d``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        embed_dim: int = 96,
        depths: Optional[List[int]] = None,
        depths_decoder: Optional[List[int]] = None,
        dims: Optional[List[int]] = None,
        dims_decoder: Optional[List[int]] = None,
        d_state: int = 16,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        depths = depths or [2, 2, 9, 2]
        depths_decoder = depths_decoder or [2, 9, 2, 2]
        num_layers = len(depths)

        if dims is None:
            dims = [int(embed_dim * (2 ** i)) for i in range(num_layers)]
        if dims_decoder is None:
            dims_decoder = list(reversed(dims))

        self.num_layers = num_layers
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(
            patch_size=4, in_chans=in_channels, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # Stochastic depth
        dpr = [
            x.item() for x in torch.linspace(
                0, drop_path_rate, sum(depths))]
        dpr_decoder = [
            x.item() for x in torch.linspace(
                0, drop_path_rate, sum(depths_decoder))][::-1]

        # Encoder layers
        self.layers = nn.ModuleList()
        for i_layer in range(num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None
                else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):
                              sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D
                if (i_layer < num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # Decoder layers
        self.layers_up = nn.ModuleList()
        for i_layer in range(num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None
                else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[
                    sum(depths_decoder[:i_layer]):
                    sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = FinalPatchExpand2D(
            dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)

    def forward_features(self, x):
        x = self.patch_embed(x)  # (B, H, W, C)
        skip_list = []
        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def forward_features_up(self, x, skip_list):
        for idx, layer_up in enumerate(self.layers_up):
            if idx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-idx])
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.final_up(x)            # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)       # (B, C, H, W)
        x = self.final_conv(x)
        if x.shape[2:] != (H_in, W_in):
            x = F.interpolate(x, size=(H_in, W_in),
                              mode="bilinear", align_corners=False)
        return x
