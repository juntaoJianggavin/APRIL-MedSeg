"""WA-UKAN -- Wavelet-Enhanced Attention Kolmogorov-Arnold Networks for medical image segmentation.
    WA-UKAN -- 用于医学图像分割的小波增强注意力 Kolmogorov-Arnold 网络。

Reference:
    Liu & Wang, "WA-UKAN: Wavelet-Enhanced Attention Kolmogorov-Arnold Networks
    for medical image segmentation", Signal Processing: Image Communication, 2026.
    DOI: 10.1016/j.image.2026.117534

Architecture (faithful port from the paper):
    1. **WA-KAN block** (Sec 3.1, Fig 2): wavelet-basis KAN (Morlet) + ECA channel
       attention, fused via element-wise addition.
    2. **LoFA** (Sec 3.2.2, Fig 3): Low-order Feature Aggregation — depthwise
       separable conv + channel attention + residual, refines skip features.
    3. **WDMAF** (Sec 3.2.3, Fig 4): Wavelet Dual-branch Multi-scale Attention
       Fusion — Haar DWT splits low/high frequency, multi-scale strip convs
       generate Q/K/V, cross-branch attention fuses inter-frequency features.
    4. **Encoder** (Sec 3.2.1): 3 Conv-BN-ReLU + 2 WA-KAN encoder blocks +
       MaxPool downsampling (5 layers total).
    5. **Bottleneck** (Sec 3.2.4): 3 x WA-KAN Embedding layers (WA-KAN + DWConv +
       BN + ReLU + residual + LayerNorm).
    6. **Decoder** (Sec 3.2.5): 5 WA-KAN Expanding layers (WA-KAN + bilinear
       upsample + BN + ReLU + residual + LayerNorm), with LoFA skip connections
       and WDMAF feature fusion at the two deepest levels.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Wavelet kernels (psi) — Morlet is the default per the paper (Sec 3.1).
# 小波核函数 —— 论文默认使用 Morlet 小波（Sec 3.1）。
# ---------------------------------------------------------------------------

def _mexican_hat(x: torch.Tensor) -> torch.Tensor:
    return (1.0 - x * x) * torch.exp(-0.5 * x * x)


def _morlet(x: torch.Tensor, omega0: float = 5.0) -> torch.Tensor:
    return torch.cos(omega0 * x) * torch.exp(-0.5 * x * x)


def _dog(x: torch.Tensor) -> torch.Tensor:
    return -x * torch.exp(-0.5 * x * x)


def _shannon(x: torch.Tensor) -> torch.Tensor:
    return torch.sinc(x) * torch.cos(2.0 * math.pi * x)


_WAVELETS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "morlet": _morlet,
    "mexican_hat": _mexican_hat,
    "dog": _dog,
    "shannon": _shannon,
}


# ---------------------------------------------------------------------------
# Wavelet-basis KAN linear layer (Sec 3.1, Eq 1-3).
# 小波基 KAN 线性层（Sec 3.1, Eq 1-3）。
# ---------------------------------------------------------------------------

class _WavKANLinear(nn.Module):
    """Wavelet-basis KAN linear layer.
        小波基 KAN 线性层。

    Maps ``(..., in_features)`` to ``(..., out_features)`` using learnable
    wavelet basis functions:

        out_j = sum_i W_ij * psi((x_i - b_ij) / s_ij) + base_j(x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        wavelet: str = "morlet",
        with_base: bool = True,
        out_chunk: int = 32,
    ) -> None:
        super().__init__()
        if wavelet not in _WAVELETS:
            raise ValueError(
                f"unknown wavelet '{wavelet}', expected one of {list(_WAVELETS)}")
        self.in_features = in_features
        self.out_features = out_features
        self._psi = _WAVELETS[wavelet]
        self.out_chunk = max(1, int(out_chunk))

        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.with_base = with_base
        if with_base:
            self.base_weight = nn.Parameter(
                torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            self.base_activation = nn.SiLU()

        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        out_chunks = []
        for j_start in range(0, self.out_features, self.out_chunk):
            j_end = min(j_start + self.out_chunk, self.out_features)
            b_c = self.translation[j_start:j_end]
            s_c = self.scale[j_start:j_end]
            w_c = self.weight[j_start:j_end]
            scaled = (x_flat.unsqueeze(1) - b_c) / (s_c + 1e-4)
            psi = self._psi(scaled)
            out_chunks.append((w_c * psi).sum(dim=-1))
        out = torch.cat(out_chunks, dim=1) if len(out_chunks) > 1 \
            else out_chunks[0]

        if self.with_base:
            out = out + F.linear(self.base_activation(x_flat), self.base_weight)

        out = self.norm(out)
        return out.reshape(*orig_shape[:-1], self.out_features)


# ---------------------------------------------------------------------------
# ECA — Efficient Channel Attention (Sec 3.1, Eq 5).
# ECA —— 高效通道注意力（Sec 3.1, Eq 5）。
# ---------------------------------------------------------------------------

class _ECA(nn.Module):
    """Efficient Channel Attention via adaptive 1-D convolution.
        通过自适应一维卷积实现的高效通道注意力。

    GAP -> adaptive kernel 1-D conv -> Sigmoid -> channel re-weighting.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        t = int(abs(math.log2(max(channels, 2)) / gamma + b / gamma))
        k = t if t % 2 else t + 1  # ensure odd
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


# ---------------------------------------------------------------------------
# Helper building blocks / 辅助构建块
# ---------------------------------------------------------------------------

class _DWConv(nn.Module):
    """Depth-wise 3x3 conv on tokenized (B, N, C) features.
        对 tokenized (B, N, C) 特征做深度 3x3 卷积。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class _PatchEmbed(nn.Module):
    """Overlapping patch embedding: stride-2 conv downsample then norm.
        重叠图块嵌入：步长2卷积下采样后归一化。"""

    def __init__(self, in_chans: int, embed_dim: int,
                 patch_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class _ConvBlock(nn.Module):
    """Conv3x3 - BN - ReLU (single, per Sec 3.2.1).
        Conv3x3 - BN - ReLU（单个，Sec 3.2.1）。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _EncoderWAKAN(nn.Module):
    """Encoder stage with Conv + WA-KAN (paper Sec 3, layers 4-5).
        带 WA-KAN 的编码器阶段（论文 Sec 3，第 4-5 层）。

    Conv3x3-BN-ReLU -> LayerNorm -> WA-KAN spatial block + residual.
    """

    def __init__(self, in_ch: int, out_ch: int, wavelet: str = "morlet",
                 drop: float = 0.0) -> None:
        super().__init__()
        self.conv = _ConvBlock(in_ch, out_ch)
        self.norm = nn.LayerNorm(out_ch)
        self.wakan = _WAKANBlock(out_ch, wavelet, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                          # (B, out_ch, H, W)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)      # (B, N, C)
        tokens = tokens + self.wakan(self.norm(tokens), H, W)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# WA-KAN block (Sec 3.1, Fig 2): wavelet KAN + ECA + element-wise fusion.
# WA-KAN 块（Sec 3.1, Fig 2）：小波 KAN + ECA + 逐元素融合。
# ---------------------------------------------------------------------------

class _WAKANBlock(nn.Module):
    """Wavelet-Enhanced Attention KAN block.
        小波增强注意力 KAN 块。

    Pipeline (Fig 2):
        1. LayerNorm -> wavelet KAN linear (Morlet) -> BN -> ReLU  [wavelet path]
        2. ECA channel attention on the original spatial feature   [attention path]
        3. Element-wise addition of the two paths.
    Operates on tokenized (B, N, C); needs H, W for spatial reshape.
    """

    def __init__(self, dim: int, wavelet: str = "morlet",
                 drop: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.kan = _WavKANLinear(dim, dim, wavelet=wavelet)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)
        self.eca = _ECA(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        # --- Wavelet path: LN -> KAN -> reshape -> BN -> ReLU ---
        wv = self.kan(self.norm(x))                       # (B, N, C)
        wv = wv.transpose(1, 2).reshape(B, C, H, W)       # (B, C, H, W)
        wv = self.act(self.bn(wv))                         # BN + ReLU (Eq 4)
        wv = wv.flatten(2).transpose(1, 2)                 # back to (B, N, C)

        # --- ECA path: channel attention on original spatial feature (Eq 5) ---
        x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
        eca_out = self.eca(x_spatial)                      # (B, C, H, W)
        eca_out = eca_out.flatten(2).transpose(1, 2)       # (B, N, C)

        # --- Fusion: element-wise addition ---
        out = wv + eca_out
        out = self.drop(out)
        return out


# ---------------------------------------------------------------------------
# LoFA — Low-order Feature Aggregation (Sec 3.2.2, Fig 3).
# LoFA —— 低阶特征聚合（Sec 3.2.2, Fig 3）。
# ---------------------------------------------------------------------------

class _ChannelAttention(nn.Module):
    """Simple channel attention (GAP -> FC -> ReLU -> FC -> Sigmoid).
        简单通道注意力（GAP -> FC -> ReLU -> FC -> Sigmoid）。"""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class _LoFA(nn.Module):
    """Low-order Feature Aggregation module.
        低阶特征聚合模块。

    DW conv (3x3) -> PW conv (1x1) -> channel attention -> residual.
    Refines encoder skip features (Sec 3.2.2).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.ca = _ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw(x)
        out = self.act(self.bn(self.pw(out)))
        out = self.ca(out)
        return out + x


# ---------------------------------------------------------------------------
# Haar DWT — one-level wavelet decomposition (Sec 3.2.3).
# Haar 小波变换 —— 单级小波分解（Sec 3.2.3）。
# ---------------------------------------------------------------------------

def _haar_dwt(x: torch.Tensor):
    """One-level Haar DWT. Returns (LL, LH, HL, HH) at half resolution.
        单级 Haar 小波分解，返回 (LL, LH, HL, HH)，分辨率为一半。"""
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5   # horizontal detail
    hl = (a - b + c - d) * 0.5   # vertical detail
    hh = (a - b - c + d) * 0.5   # diagonal detail
    return ll, lh, hl, hh


# ---------------------------------------------------------------------------
# WDMAF — Wavelet Dual-branch Multi-scale Attention Fusion (Sec 3.2.3, Fig 4).
# WDMAF —— 小波双分支多尺度注意力融合（Sec 3.2.3, Fig 4）。
# ---------------------------------------------------------------------------

class _WDMAF(nn.Module):
    """Wavelet Dual-branch Multi-scale Attention Fusion.
        小波双分支多尺度注意力融合。

    Pipeline (Fig 4):
        1. Conv -> Haar DWT -> LL (low-freq) + concat(LH,HL,HH) (high-freq)
        2. 1x1 conv + LayerNorm on both branches -> unify to C channels
        3. Multi-scale vertical strip convs -> Q, K, V per branch
        4. Cross-branch attention: Q_HF attends K/V_LF, Q_LF attends K/V_HF
        5. 3x3 conv refine -> concat -> DwConv -> concat original -> norm
    Only applied at deep decoder levels to keep attention memory feasible.
    """

    def __init__(self, channels: int, num_heads: int = 4,
                 strip_sizes=(7, 11, 21)) -> None:
        super().__init__()
        C = channels
        self.channels = C
        self.num_heads = num_heads
        self.scale = (C // num_heads) ** -0.5

        self.conv = nn.Conv2d(C, C, 3, padding=1)

        # Channel adjustment: LL (C) and HF (3C) both -> C
        self.ll_adj = nn.Conv2d(C, C, 1)
        self.hf_adj = nn.Conv2d(C * 3, C, 1)
        self.ll_ln = nn.LayerNorm(C)
        self.hf_ln = nn.LayerNorm(C)

        # Multi-scale vertical strip convolutions (k x 1, depthwise)
        n_strips = len(strip_sizes)
        self.ll_strips = nn.ModuleList([
            nn.Conv2d(C, C, (k, 1), padding=(k // 2, 0), groups=C)
            for k in strip_sizes
        ])
        self.hf_strips = nn.ModuleList([
            nn.Conv2d(C, C, (k, 1), padding=(k // 2, 0), groups=C)
            for k in strip_sizes
        ])

        # QKV projections: concat(n_strips * C) -> 3C
        self.ll_qkv = nn.Conv2d(C * n_strips, C * 3, 1)
        self.hf_qkv = nn.Conv2d(C * n_strips, C * 3, 1)

        # 3x3 conv refinement after cross-attention
        self.ll_refine = nn.Conv2d(C, C, 3, padding=1)
        self.hf_refine = nn.Conv2d(C, C, 3, padding=1)

        # DwConv on concatenated output (2C)
        self.dwconv = nn.Conv2d(C * 2, C * 2, 3, padding=1, groups=C * 2)

        # Final: concat(2C dwconv + C original) -> norm -> 1x1 conv -> C
        self.final_norm = nn.GroupNorm(8, C * 3)
        self.final_conv = nn.Conv2d(C * 3, C, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        identity = x

        x = self.conv(x)
        ll, lh, hl, hh = _haar_dwt(x)          # each (B, C, H/2, W/2)
        hf = torch.cat([lh, hl, hh], dim=1)     # (B, 3C, H/2, W/2)

        # --- Channel adjust + LayerNorm ---
        ll = self.ll_adj(ll)                     # (B, C, h, w)
        hf = self.hf_adj(hf)
        h, w = ll.shape[-2:]
        ll = self.ll_ln(ll.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        hf = self.hf_ln(hf.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # --- Multi-scale strip convs ---
        ll_s = torch.cat([s(ll) for s in self.ll_strips], dim=1)  # (B, nC, h, w)
        hf_s = torch.cat([s(hf) for s in self.hf_strips], dim=1)

        # --- QKV ---
        ll_qkv = self.ll_qkv(ll_s)               # (B, 3C, h, w)
        hf_qkv = self.hf_qkv(hf_s)

        N = h * w
        d = C // self.num_heads

        def split_qkv(qkv):
            qkv = qkv.reshape(B, 3, self.num_heads, d, N)
            qkv = qkv.permute(0, 1, 2, 4, 3)     # (B, 3, nh, N, d)
            return qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, nh, N, d)

        ll_q, ll_k, ll_v = split_qkv(ll_qkv)
        hf_q, hf_k, hf_v = split_qkv(hf_qkv)

        # --- Cross-branch attention (Eq 11) ---
        # Q_HF x K_LF, Q_LF x K_HF
        attn_hf = torch.softmax(
            (hf_q @ ll_k.transpose(-2, -1)) * self.scale, dim=-1)
        out_hf = attn_hf @ ll_v                  # (B, nh, N, d)

        attn_ll = torch.softmax(
            (ll_q @ hf_k.transpose(-2, -1)) * self.scale, dim=-1)
        out_ll = attn_ll @ hf_v

        # Reshape to spatial
        out_hf = out_hf.transpose(-2, -1).reshape(B, C, h, w)
        out_ll = out_ll.transpose(-2, -1).reshape(B, C, h, w)

        # --- 3x3 refine (Eq 12) ---
        out_hf = self.hf_refine(out_hf)
        out_ll = self.ll_refine(out_ll)

        # --- Concat (Eq 13) ---
        fused = torch.cat([out_hf, out_ll], dim=1)   # (B, 2C, h, w)

        # --- DwConv + concat original (Eq 14) ---
        fused = self.dwconv(fused)
        fused = F.interpolate(fused, size=(H, W), mode='bilinear',
                              align_corners=False)
        fused = torch.cat([fused, identity], dim=1)    # (B, 3C, H, W)
        fused = self.final_conv(self.final_norm(fused))  # (B, C, H, W)
        return fused


# ---------------------------------------------------------------------------
# WA-KAN Embedding block (Sec 3.2.4, Eq 16) and Expanding block (Sec 3.2.5, Eq 17).
# WA-KAN 嵌入块（Sec 3.2.4, Eq 16）与扩展块（Sec 3.2.5, Eq 17）。
# ---------------------------------------------------------------------------

class _WAKANEmbedding(nn.Module):
    """WA-KAN Embedding block (bottleneck, Eq 16).
        WA-KAN 嵌入块（瓶颈层，Eq 16）。

    WA-KAN -> DWConv -> BN -> ReLU -> residual -> LayerNorm.
    Operates on tokenized (B, N, C).
    """

    def __init__(self, dim: int, wavelet: str = "morlet",
                 drop: float = 0.0) -> None:
        super().__init__()
        self.wakan = _WAKANBlock(dim, wavelet, drop)
        self.dwconv = _DWConv(dim)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        shortcut = x
        x = self.wakan(x, H, W)                  # (B, N, C)
        x = self.dwconv(x, H, W)                 # (B, N, C)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.act(self.bn(x))
        x = x.flatten(2).transpose(1, 2)         # (B, N, C)
        x = x + shortcut                         # residual
        x = self.norm(x)
        return x


class _WAKANExpanding(nn.Module):
    """WA-KAN Expanding block (decoder, Eq 17).
        WA-KAN 扩展块（解码器，Eq 17）。

    WA-KAN -> bilinear upsample 2x + 1x1 conv -> BN -> ReLU -> residual -> LN.
    Input: tokenized (B, N, C_in) at resolution (H, W).
    Output: spatial (B, C_out, 2H, 2W).
    """

    def __init__(self, in_ch: int, out_ch: int, wavelet: str = "morlet",
                 drop: float = 0.0) -> None:
        super().__init__()
        self.wakan = _WAKANBlock(in_ch, wavelet, drop)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(out_ch)
        # Residual path: upsample input + 1x1 conv to match output
        self.residual_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.wakan(x, H, W)                  # (B, N, C_in)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Residual path (upsample input to match output resolution)
        res = self.residual_up(x)                # (B, out_ch, 2H, 2W)

        # Transposed conv upsample
        x = self.act(self.bn(self.up(x)))        # (B, out_ch, 2H, 2W)

        x = x + res                              # residual
        # LayerNorm on channel dim
        x = x.permute(0, 2, 3, 1)                # (B, 2H, 2W, out_ch)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)                # (B, out_ch, 2H, 2W)
        return x


# ---------------------------------------------------------------------------
# WA-UKAN network / WA-UKAN 网络
# ---------------------------------------------------------------------------

class WAUKAN(nn.Module):
    """WA-UKAN: Wavelet-Enhanced Attention Kolmogorov-Arnold Networks.
        WA-UKAN：小波增强注意力 Kolmogorov-Arnold 网络。

    Overall architecture (Sec 3.2):
        Encoder: 5 layers — 3 Conv blocks (Conv-BN-ReLU + MaxPool) +
                 2 WA-KAN encoder blocks (Conv + WA-KAN + MaxPool)
        Bottleneck: PatchEmbed -> 3 x WA-KAN Embedding
        Decoder: 5 WA-KAN Expanding (bilinear upsample), with LoFA skip
                 connections and WDMAF at the two deepest levels.

    Args:
        in_channels: Input image channels (default 3).
        num_classes: Output segmentation classes (default 2).
        img_size: Nominal input spatial size (forward accepts arbitrary H, W
                  divisible by 32).
        dims: 5 encoder channel dimensions. Default (32, 64, 128, 256, 512)
              per paper Table 7 optimal config.
        wavelet: Wavelet kernel (``morlet``, ``mexican_hat``, ``dog``,
                 ``shannon``). Default ``morlet`` per the paper.
        drop_rate: Dropout in WA-KAN blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        dims=(32, 64, 128, 256, 512),
        wavelet: str = "morlet",
        drop_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        dims = tuple(dims)
        assert len(dims) == 5, "dims must have 5 entries (5-level encoder)"
        self.dims = dims
        self.img_size = img_size
        self.num_classes = num_classes

        c0, c1, c2, c3, c4 = dims

        # ---- Encoder: 3 Conv + 2 WA-KAN + MaxPool (Sec 3.2.1) ----
        self.enc1 = _ConvBlock(in_channels, c0)          # H
        self.enc2 = _ConvBlock(c0, c1)                    # H/2
        self.enc3 = _ConvBlock(c1, c2)                    # H/4
        self.enc4 = _EncoderWAKAN(c2, c3, wavelet, drop_rate)  # H/8
        self.enc5 = _EncoderWAKAN(c3, c4, wavelet, drop_rate)  # H/16
        self.pool = nn.MaxPool2d(2)

        # ---- Bottleneck: WA-KAN Embedding (Sec 3.2.4) ----
        self.patch_embed = _PatchEmbed(c4, c4, patch_size=3, stride=2)  # H/16 -> H/32
        self.wakan_embed = nn.ModuleList([
            _WAKANEmbedding(c4, wavelet, drop_rate) for _ in range(3)
        ])

        # ---- Decoder: WA-KAN Expanding (Sec 3.2.5) ----
        self.dec5 = _WAKANExpanding(c4, c4, wavelet, drop_rate)  # H/32 -> H/16
        self.dec4 = _WAKANExpanding(c4, c3, wavelet, drop_rate)  # H/16 -> H/8
        self.dec3 = _WAKANExpanding(c3, c2, wavelet, drop_rate)  # H/8 -> H/4
        self.dec2 = _WAKANExpanding(c2, c1, wavelet, drop_rate)  # H/4 -> H/2
        self.dec1 = _WAKANExpanding(c1, c0, wavelet, drop_rate)  # H/2 -> H

        # ---- LoFA skip connections (Sec 3.2.2) ----
        self.lofa5 = _LoFA(c4)
        self.lofa4 = _LoFA(c3)
        self.lofa3 = _LoFA(c2)
        self.lofa2 = _LoFA(c1)
        self.lofa1 = _LoFA(c0)

        # ---- Fusion convs after skip concat ----
        self.fuse5 = nn.Conv2d(c4 * 2, c4, 1, bias=False)
        self.fuse4 = nn.Conv2d(c3 * 2, c3, 1, bias=False)
        self.fuse3 = nn.Conv2d(c2 * 2, c2, 1, bias=False)
        self.fuse2 = nn.Conv2d(c1 * 2, c1, 1, bias=False)
        self.fuse1 = nn.Conv2d(c0 * 2, c0, 1, bias=False)

        # ---- WDMAF at two deepest decoder levels (Sec 3.2.3) ----
        self.wdmaf5 = _WDMAF(c4)
        self.wdmaf4 = _WDMAF(c3)

        # ---- Output head ----
        self.head = nn.Conv2d(c0, num_classes, 1)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= max(1, m.groups)
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H_in, W_in = x.shape

        # Pad to multiple of 32 (4 MaxPools + 1 PatchEmbed = /32)
        ph = (32 - H_in % 32) % 32
        pw = (32 - W_in % 32) % 32
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))

        # ---- Encoder (5 layers, 4 MaxPools) ----
        e1 = self.enc1(x)                        # (c0, H)
        e2 = self.enc2(self.pool(e1))            # (c1, H/2)
        e3 = self.enc3(self.pool(e2))            # (c2, H/4)
        e4 = self.enc4(self.pool(e3))            # (c3, H/8)
        e5 = self.enc5(self.pool(e4))            # (c4, H/16)

        # ---- Bottleneck: WA-KAN Embedding ----
        tokens, Hb, Wb = self.patch_embed(e5)    # tokens (B, N, c4), Hb=H/32
        for blk in self.wakan_embed:
            tokens = blk(tokens, Hb, Wb)

        # ---- Decoder stage 5 (H/32 -> H/16) ----
        d5 = self.dec5(tokens, Hb, Wb)           # (c4, H/16)
        d5 = torch.cat([d5, self.lofa5(e5)], dim=1)
        d5 = self.wdmaf5(self.fuse5(d5))          # (c4, H/16)

        # ---- Decoder stage 4 (H/16 -> H/8) ----
        B5, C5, H5, W5 = d5.shape
        d5_tok = d5.flatten(2).transpose(1, 2)    # (B, N, c4)
        d4 = self.dec4(d5_tok, H5, W5)            # (c3, H/8)
        d4 = torch.cat([d4, self.lofa4(e4)], dim=1)
        d4 = self.wdmaf4(self.fuse4(d4))          # (c3, H/8)

        # ---- Decoder stage 3 (H/8 -> H/4) ----
        B4, C4, H4, W4 = d4.shape
        d4_tok = d4.flatten(2).transpose(1, 2)
        d3 = self.dec3(d4_tok, H4, W4)            # (c2, H/4)
        d3 = self.fuse3(torch.cat([d3, self.lofa3(e3)], dim=1))

        # ---- Decoder stage 2 (H/4 -> H/2) ----
        B3, C3, H3, W3 = d3.shape
        d3_tok = d3.flatten(2).transpose(1, 2)
        d2 = self.dec2(d3_tok, H3, W3)            # (c1, H/2)
        d2 = self.fuse2(torch.cat([d2, self.lofa2(e2)], dim=1))

        # ---- Decoder stage 1 (H/2 -> H) ----
        B2, C2, H2, W2 = d2.shape
        d2_tok = d2.flatten(2).transpose(1, 2)
        d1 = self.dec1(d2_tok, H2, W2)            # (c0, H)
        d1 = self.fuse1(torch.cat([d1, self.lofa1(e1)], dim=1))

        out = self.head(d1)                       # (num_classes, H)

        # Crop to original size
        if out.shape[-2:] != (H_in, W_in):
            out = out[..., :H_in, :W_in]
        if out.shape[-2:] != (H_in, W_in):
            out = F.interpolate(out, size=(H_in, W_in), mode='bilinear',
                                align_corners=False)
        return out


__all__ = ["WAUKAN"]
