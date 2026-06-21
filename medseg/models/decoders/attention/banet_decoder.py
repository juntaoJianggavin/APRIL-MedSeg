"""BANet Decoder — Boundary-Aware Network for Segmentation.
    BANet 解码器。

中文: BANet 解码器：边界感知网络分割解码器。

Reference:
    Fan et al., "Boundary-Aware Network for Fast and High-Accuracy
    Portrait Segmentation", arXiv 2019 / ICCV Workshop.
    Source: https://github.com/suruoxi/BANet (unofficial)

Architecture:
    1. **Region branch**: Apply standard conv layers on the bottleneck
       feature to produce a region feature map.
    2. **Boundary branch**: Learn a Sobel-like edge-detection filter
       bank on the bottleneck feature, producing boundary feature maps.
       A side boundary prediction head provides auxiliary boundary loss.
    3. **Boundary-guided Refinement**: Use boundary features as spatial
       attention to selectively refine region features at object borders.
    4. **Fusion**: Concatenate refined region + boundary features, then
       project to the output channel count.

    ``has_internal_skip = True``: the BANet decoder operates purely on
    the deepest feature; external skip connections are IGNORED.

Integration:
    Set ``decoder: { name: banet }`` in YAML.
"""
# Source: https://github.com/suruoxi/BANet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from medseg.registry import DECODER_REGISTRY


class _BoundaryBranch(nn.Module):
    """Learned boundary-detection branch.
        Learned boundary-detection 分支。

    Uses depth-wise Sobel-like filters (learned) followed by 1×1 conv to
    produce boundary features.  The side head predicts a boundary probability
    map (used as auxiliary loss during training).
    """

    def __init__(self, in_ch: int, mid_ch: int = 64):
        super().__init__()
        # Learned 边缘 filters ( depth-wise, initialised near Sobel ) / Learned edge filters (depth-wise, initialised near Sobel)
        self.edge_conv = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1,
            groups=in_ch, bias=False,
        )
        self._init_sobel()
        # 通道 projection after 边缘 filter / Channel projection after edge filter
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        # Side 边界 预测 头部 ( 1-channel ) / Side boundary prediction head (1-channel)
        self.side_head = nn.Sequential(
            nn.Conv2d(mid_ch, 1, 1),
            nn.Sigmoid(),
        )

    def _init_sobel(self):
        """Initialise depth-wise conv near Sobel operator."""
        with torch.no_grad():
            w = self.edge_conv.weight  # (in_ch, 1, 3, 3)
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32)
            for i in range(w.shape[0]):
                if i % 2 == 0:
                    w[i, 0] = sobel_x
                else:
                    w[i, 0] = sobel_y

    def forward(self, x: torch.Tensor):
        edge = self.edge_conv(x)
        edge_feat = self.proj(edge)
        boundary_pred = self.side_head(edge_feat)  # (B, 1, H, W)
        return edge_feat, boundary_pred


class _BoundaryRefinement(nn.Module):
    """Boundary-guided spatial attention refinement.
        Boundary-guided 空间的 注意力 refinement。

    Boundary features produce a spatial attention mask that selectively
    enhances region features near object borders.
    """

    def __init__(self, region_ch: int, boundary_ch: int, out_ch: int):
        super().__init__()
        # Gate: 边界 特征 → 空间的 注意力 over 区域 特征 / Gate: boundary features → spatial attention over region features
        self.gate = nn.Sequential(
            nn.Conv2d(boundary_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid(),
        )
        # Transform 区域 特征 / Transform region features
        self.region_transform = nn.Sequential(
            nn.Conv2d(region_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, region_feat: torch.Tensor,
                boundary_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            region_feat: (B, region_ch, H, W) region features.
            boundary_feat: (B, boundary_ch, H, W) boundary features.
        Returns:
            (B, out_ch, H, W) boundary-refined features.
        """
        gate = self.gate(boundary_feat)             # (B, out_ch, H, W)
        region = self.region_transform(region_feat) # (B, out_ch, H, W)
        refined = region * gate + region            # residual refinement
        return refined


@DECODER_REGISTRY.register("banet")
class BANetDecoder(nn.Module):
    """BANet decoder: boundary-aware region refinement.
        BANet 解码器。

    中文: BANet 解码器：边界感知区域细化。

    Operates on the deepest (bottleneck) feature only; external skip
    connections are IGNORED (``has_internal_skip = True``).

    Args:
        encoder_channels: list of encoder output channels (unused, kept for API).
        bottleneck_channels: bottleneck output channel count.
        skip_connection: unused (has_internal_skip = True).
        region_channels: region branch output channels (default 256).
        boundary_channels: boundary branch output channels (default 64).
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, region_channels: int = 256,
                 boundary_channels: int = 64, **kwargs):
        super().__init__()
        in_ch = bottleneck_channels

        # 区域 分支 / Region branch
        self.region_branch = nn.Sequential(
            nn.Conv2d(in_ch, region_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(region_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(region_channels, region_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(region_channels),
            nn.ReLU(inplace=True),
        )

        # 边界 分支 / Boundary branch
        self.boundary_branch = _BoundaryBranch(in_ch, mid_ch=boundary_channels)

        # Boundary-guided refinement
        self.refinement = _BoundaryRefinement(
            region_ch=region_channels,
            boundary_ch=boundary_channels,
            out_ch=region_channels,
        )

        # 融合: concat refined 区域 + 边界 特征 → project / Fusion: concat refined region + boundary features → project
        self.fusion = nn.Sequential(
            nn.Conv2d(region_channels + boundary_channels, region_channels,
                      1, bias=False),
            nn.BatchNorm2d(region_channels),
            nn.ReLU(inplace=True),
        )
        self._out_channels = region_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]):
        """Forward pass.
            前向传播 pass。

        Args:
            bottleneck_feat: deepest feature after bottleneck (B, C, H, W).
            skip_features: list of skip features (IGNORED by BANet decoder).
        Returns:
            Boundary-refined feature map (B, out_ch, H, W).
        """
        # 区域 分支 / Region branch
        region_feat = self.region_branch(bottleneck_feat)   # (B, region_ch, H, W)

        # 边界 分支 / Boundary branch
        boundary_feat, _boundary_pred = self.boundary_branch(bottleneck_feat)
        # _boundary_pred can be used for auxiliary boundary 损失 / _boundary_pred can be used for auxiliary boundary loss

        # Boundary-guided refinement
        refined = self.refinement(region_feat, boundary_feat)

        # 融合 / Fusion
        out = self.fusion(torch.cat([refined, boundary_feat], dim=1))
        return out


__all__ = ['BANetDecoder']
