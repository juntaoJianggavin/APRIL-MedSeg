"""Serp-Mamba: Selective State-Space Model for Retinal Vessel Segmentation.
    Serp-Mamba: Selective State-Space Model for Retinal Vessel Segmentation.

Faithful reimplementation from:
  https://github.com/whq-xxh/Serp-Mamba  (IEEE TMI 2025)

Architecture (faithful to official source SerpMamba.py, 1184 lines):
  - nnU-Net-style encoder: stem + n_stages (StackedResidualBlocks) + MambaLayer per stage
  - MambaLayer_Serpentine_Scan: mamba_ssm.Mamba (1D) + SerpScan (deformable strip conv)
    with 4-direction processing (x_mamba, y_mamba, serpentine_x, serpentine_y) + EncoderConv fusion
  - MambaLayer_ambiguous_scan: mamba_ssm.Mamba + Continuity_Perception + Pixel_Extractor
    + Dual_Driving + UncertaintyCheck (at bottleneck stage)
  - nnU-Net-style decoder: ConvTranspose2d + StackedResidualBlocks + deep supervision
  - Norm: InstanceNorm, Nonlin: LeakyReLU(0.01, inplace=True)

Constructor:
    SerpMamba(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/whq-xxh/Serp-Mamba

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Union, List, Tuple, Type

from .umamba import MambaSSM, BasicResBlock, BasicBlockD


# ---------------------------------------------------------------------------
# Helper functions for ambiguous pixel extraction (faithful to official source)
# ---------------------------------------------------------------------------

def point_sample(input, point_coords):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0,
                           align_corners=True, mode='bilinear',
                           padding_mode='zeros')
    if add_dim:
        output = output.squeeze(3)
    return output


def get_thresholded_points(feature_map, threshold1, threshold2):
    B = feature_map.shape[0]
    feature_map = feature_map.mean(dim=1, keepdim=True)
    all_coords = []
    all_coords1 = []
    all_coords2 = []
    for b in range(B):
        feature_map_b = feature_map[b, 0]
        coords = torch.stack(((feature_map_b > threshold1) & (feature_map_b < threshold2)).nonzero(as_tuple=True))
        coords1 = torch.stack((feature_map_b > threshold1).nonzero(as_tuple=True))
        coords2 = torch.stack((feature_map_b < threshold2).nonzero(as_tuple=True))
        coords = coords.permute(1, 0).unsqueeze(0)
        coords1 = coords1.permute(1, 0).unsqueeze(0)
        coords2 = coords2.permute(1, 0).unsqueeze(0)
        all_coords.append(coords)
        all_coords1.append(coords1)
        all_coords2.append(coords2)
    return all_coords1, all_coords2, all_coords


def extract_features_with_thresholds(input, feature_map, threshold1, threshold2):
    B = input.shape[0]
    coords1_list, coords2_list, coords_list = get_thresholded_points(feature_map, threshold1, threshold2)
    output1_list = []
    output2_list = []
    output_list = []
    for b in range(B):
        output1 = point_sample(input[b:b+1], coords1_list[b])
        output2 = point_sample(input[b:b+1], coords2_list[b])
        output = point_sample(input[b:b+1], coords_list[b])
        output1_list.append(output1)
        output2_list.append(output2)
        output_list.append(output)
    return output1_list, output2_list, output_list, coords1_list, coords2_list, coords_list


def Dual_Driving(AmbiguousPixel, DrivenPixel):
    Q = AmbiguousPixel.permute(1, 0)
    K = DrivenPixel
    V = DrivenPixel.permute(1, 0)
    vascular_scores = torch.matmul(Q, K)
    vascular_weights = F.softmax(vascular_scores, dim=-1)
    output = torch.matmul(vascular_weights, V)
    output = output.permute(1, 0)
    return output


# ---------------------------------------------------------------------------
# Ambiguous pixel detection modules (faithful to official source)
# ---------------------------------------------------------------------------

class Continuity_Perception(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

    def forward(self, x_out, threshold1, threshold2):
        B, C, H, W = x_out.shape
        x_min = x_out.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max = x_out.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        x_normalized = (x_out - x_min) / (x_max - x_min)
        denorm_value = x_min
        for b in range(B):
            for channel in range(C):
                patches = F.unfold(x_normalized[b:b+1, channel:channel + 1], kernel_size=3, padding=1)
                patches = patches.permute(0, 2, 1).view(-1, H, W, 9)
                center_pixel_condition = (patches[..., 4] > threshold1) & (patches[..., 4] < threshold2)
                surrounding_condition = patches[..., [0, 1, 2, 3, 5, 6, 7, 8]] < threshold1
                surrounding_condition = surrounding_condition.all(dim=-1)
                condition = (center_pixel_condition & surrounding_condition).unsqueeze(1)
                denorm_value_channel = denorm_value[b:b+1, channel:channel + 1, :, :]
                x_out[b:b+1, channel:channel+1][condition] = denorm_value_channel
        return x_out


class UncertaintyCheck(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv_reduce = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.conv_expand = nn.ConvTranspose2d(1, output_channels, kernel_size=1)

    def forward(self, uncertainty_vessel, uncertainty_background, x_out, uncertainty_coords, threshold2):
        B = x_out.shape[0]
        uncertainty_vessel_normalized = (uncertainty_vessel - uncertainty_vessel.min()) / (uncertainty_vessel.max() - uncertainty_vessel.min())
        uncertainty_background_normalized = (uncertainty_background - uncertainty_background.min()) / (uncertainty_background.max() - uncertainty_background.min())
        uncertainty_vessel_reduced = torch.mean(uncertainty_vessel_normalized, dim=1, keepdim=True)
        uncertainty_background_reduced = torch.mean(uncertainty_background_normalized, dim=1, keepdim=True)
        x_out_reduced = self.conv_reduce(x_out)
        x_min = x_out_reduced.min()
        x_max = x_out_reduced.max()
        for b in range(B):
            if len(uncertainty_coords[b]) > 0 and uncertainty_coords[b].numel() > 0:
                coords_indexing = uncertainty_coords[b].squeeze(0).long()
                condition_mask = (uncertainty_vessel_reduced[b, 0] > threshold2) | (
                    uncertainty_background_reduced[b, 0] > threshold2)
                if coords_indexing.numel() > 0:
                    x_out_reduced[b, 0, coords_indexing[:, 0], coords_indexing[:, 1]][condition_mask] = x_max
        x_out_restored = self.conv_expand(x_out_reduced)
        return x_out_restored


class Pixel_Extractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_map, threshold1, threshold2):
        B = feature_map.shape[0]
        feature_map_min = feature_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        feature_map_max = feature_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        feature_map_norm = (feature_map - feature_map_min) / (feature_map_max - feature_map_min)
        (feature_map1_list, feature_map2_list, feature_map3_list,
         coords1_list, coords2_list, coords_list) = extract_features_with_thresholds(
            feature_map, feature_map_norm, threshold1, threshold2)
        if all(fm.numel() > 0 for fm in feature_map1_list):
            feature_map1 = torch.cat(feature_map1_list, dim=0)
        else:
            feature_map1 = torch.empty(B, feature_map.shape[1], 0, device=feature_map.device)
        if all(fm.numel() > 0 for fm in feature_map2_list):
            feature_map2 = torch.cat(feature_map2_list, dim=0)
        else:
            feature_map2 = torch.empty(B, feature_map.shape[1], 0, device=feature_map.device)
        if all(fm.numel() > 0 for fm in feature_map3_list):
            feature_map3 = torch.cat(feature_map3_list, dim=0)
        else:
            feature_map3 = torch.empty(B, feature_map.shape[1], 0, device=feature_map.device)
        return feature_map1, feature_map2, feature_map3, coords1_list, coords2_list, coords_list


# ---------------------------------------------------------------------------
# MambaLayer_ambiguous_scan (faithful to official source)
# ---------------------------------------------------------------------------

class MambaLayer_ambiguous_scan(nn.Module):
    """Ambiguous pixel driving Mamba layer (faithful to official source).
        Uses mamba_ssm.Mamba (1D SSM) + Continuity_Perception + Dual_Driving.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vessel_norm = nn.LayerNorm(dim)
        self.mamba = MambaSSM(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.vessel_mamba = MambaSSM(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.threshold1 = nn.Parameter(torch.tensor(0.45))
        self.threshold2 = nn.Parameter(torch.tensor(0.55))
        self.threshold1.data.clamp_(0.4, 0.5)
        self.threshold2.data.clamp_(0.5, 0.6)
        # Moved from forward to __init__ so parameters are registered/trainable
        self.check_scan = Continuity_Perception(dim, dim)
        self.extractor = Pixel_Extractor()
        self.check_uncertainty = UncertaintyCheck(dim, dim)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_mamba = self.mamba(x_flat)
        x_norm = self.norm(x_mamba)
        x_out = x_norm.transpose(-1, -2).reshape(B, C, *img_dims)
        x_mamba = x_out

        # Detect ambiguous pixels surrounded by vessels
        x_out = self.check_scan(x_out, self.threshold1, self.threshold2)

        # ADDR scan: extract vessel/background/uncertainty pixels
        (out_mamba_vessel, out_mamba_background, out_mamba_uncertainty,
         vessel_coords, background_coords, uncertainty_coords) = self.extractor(
            x_out, self.threshold1, self.threshold2)

        has_uncertainty = any(coords.numel() > 0 for coords in uncertainty_coords if len(coords) > 0)
        if not has_uncertainty:
            return x_out

        # Vessel/Background Driven
        uncertainty_vessel_list = []
        uncertainty_background_list = []
        for b in range(B):
            if len(out_mamba_uncertainty) > b and out_mamba_uncertainty[b].numel() > 0:
                out_mamba_vessel_b = out_mamba_vessel[b] if len(out_mamba_vessel) > b else torch.empty_like(out_mamba_uncertainty[b])
                out_mamba_background_b = out_mamba_background[b] if len(out_mamba_background) > b else torch.empty_like(out_mamba_uncertainty[b])
                uncertainty_vessel_b = Dual_Driving(out_mamba_uncertainty[b], out_mamba_vessel_b)
                uncertainty_background_b = Dual_Driving(out_mamba_uncertainty[b], out_mamba_background_b)
                uncertainty_vessel_list.append(uncertainty_vessel_b.unsqueeze(0))
                uncertainty_background_list.append(uncertainty_background_b.unsqueeze(0))
            else:
                empty_tensor = torch.empty(1, C, 0, device=x.device)
                uncertainty_vessel_list.append(empty_tensor)
                uncertainty_background_list.append(empty_tensor)

        if uncertainty_vessel_list:
            uncertainty_vessel = torch.cat(uncertainty_vessel_list, dim=0)
            uncertainty_background = torch.cat(uncertainty_background_list, dim=0)
        else:
            return x_out

        x_out = self.check_uncertainty(uncertainty_vessel, uncertainty_background, x_out,
                                        uncertainty_coords, self.threshold2)
        x_out = (x_out - x_out.min()) / (x_out.max() - x_out.min())
        x_out = x_out * x_mamba

        # Final vessel mamba block
        x_out = x_out.reshape(B, C, n_tokens).transpose(-1, -2)
        x_out = self.vessel_mamba(x_out)
        x_out = self.vessel_norm(x_out)
        x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims) + x_mamba
        return x_out


# ---------------------------------------------------------------------------
# SerpScan: Deformable strip convolution (faithful to official source)
# ---------------------------------------------------------------------------

class SerpScan(nn.Module):
    """Deformable strip convolution for serpentine scanning (faithful to DSCNet).
        Deformable strip convolution for serpentine scanning.
    """
    def __init__(self, in_channels=1, out_channels=1, kernel_size=9,
                 extend_scope=1.0, morph=0, if_offset=True):
        super().__init__()
        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        padding_size = kernel_size // 2
        self.bn = nn.BatchNorm2d(kernel_size)
        self.tanh = nn.Tanh()
        self.offset_conv = nn.Conv2d(in_channels, kernel_size, kernel_size=kernel_size,
                                     stride=(kernel_size, 1), padding=(0, padding_size))

    def forward(self, input):
        offset_y = self.offset_conv(input)
        offset_y = self.bn(offset_y)
        offset_y = self.tanh(offset_y)
        offset_x = self.offset_conv(input.permute(0, 1, 3, 2))
        offset_x = self.bn(offset_x)
        offset_x = self.tanh(offset_x)
        offset = torch.cat([offset_y, offset_x], dim=1)
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset, morph=self.morph, extend_scope=self.extend_scope)
        deformed_feature, grid = get_interpolated_feature(input, y_coordinate_map, x_coordinate_map)
        return deformed_feature, grid


def _coordinate_map_scaling(coordinate_map, origin, target=(-1, 1)):
    min_val, max_val = origin
    a, b = target
    coordinate_map_scaled = torch.clamp(coordinate_map, min_val, max_val)
    scale_factor = (b - a) / (max_val - min_val)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min_val)
    return coordinate_map_scaled


def pad_to_match_dimensions(tensor1, tensor2):
    B1, C1, H1, W1 = tensor1.shape
    B2, C2, H2, W2 = tensor2.shape
    assert B1 == B2 and C1 == C2
    max_height = max(H1, H2)
    max_width = max(W1, W2)
    if H1 < max_height:
        tensor1 = F.pad(tensor1, (0, 0, 0, max_height - H1), "constant", 0)
    if H2 < max_height:
        tensor2 = F.pad(tensor2, (0, 0, 0, max_height - H2), "constant", 0)
    if W1 < max_width:
        tensor1 = F.pad(tensor1, (0, max_width - W1, 0, 0), "constant", 0)
    if W2 < max_width:
        tensor2 = F.pad(tensor2, (0, max_width - W2, 0, 0), "constant", 0)
    return tensor1, tensor2


def get_coordinate_map_2D(offset, morph, extend_scope=1.0):
    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")
    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = offset.device
    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    if morph == 0:
        y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
        y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)
        x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
        x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)
        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_
        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)
        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()
        y_offset_new_[center] = 0
        for index in range(1, center + 1):
            y_offset_new_[center + index] = y_offset_new_[center + index - 1] + y_offset_[center + index]
            y_offset_new_[center - index] = y_offset_new_[center - index + 1] + y_offset_[center - index]
        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")
        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))
        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")
    else:
        batch_size, _, height, width = offset.shape
        y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
        y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)
        x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
        x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)
        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)
        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_
        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)
        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()
        x_offset_new_[center] = 0
        for index in range(1, center + 1):
            x_offset_new_[center + index] = x_offset_new_[center + index - 1] + x_offset_[center + index]
            x_offset_new_[center - index] = x_offset_new_[center - index + 1] + x_offset_[center - index]
        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")
        x_offset_new_ = x_offset_new_.permute(0, 1, 3, 2)
        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))
        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")
    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(input_feature, y_coordinate_map, x_coordinate_map,
                             interpolate_mode="bilinear"):
    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1
    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])
    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)
    interpolated_feature = F.grid_sample(
        input=input_feature, grid=grid, mode=interpolate_mode,
        padding_mode="zeros", align_corners=True)
    return interpolated_feature, grid


def map_deformed_to_input(deformed_feature, input_feature, grid,
                          interpolate_mode="bilinear"):
    remapped_feature = F.grid_sample(
        input=deformed_feature, grid=grid, mode=interpolate_mode,
        padding_mode="zeros", align_corners=True)
    mask = F.grid_sample(
        input=torch.ones_like(deformed_feature), grid=grid,
        mode=interpolate_mode, padding_mode="zeros", align_corners=True)
    mask = (mask > 0).float()
    remapped_feature, input_feature = pad_to_match_dimensions(remapped_feature, input_feature)
    mask, input_feature = pad_to_match_dimensions(mask, input_feature)
    combined_feature = mask * remapped_feature + (1 - mask) * input_feature
    return combined_feature


# ---------------------------------------------------------------------------
# EncoderConv: fusion conv (faithful to official source)
# ---------------------------------------------------------------------------

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.leaky_relu(x)
        return x


# ---------------------------------------------------------------------------
# MambaLayer_Serpentine_Scan (faithful to official source)
# ---------------------------------------------------------------------------

class MambaLayer_Serpentine_Scan(nn.Module):
    """Serpentine scan Mamba layer (faithful to official source).
        Uses mamba_ssm.Mamba (1D) + SerpScan (deformable strip conv) with
        4-direction processing + EncoderConv fusion.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = MambaSSM(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        # Moved from forward to __init__ so parameters are registered/trainable
        self.serp_scan_x = SerpScan(dim, dim, kernel_size=9, morph=0)
        self.serp_scan_y = SerpScan(dim, dim, kernel_size=9, morph=1)
        self.fusion = EncoderConv(2 * dim, dim)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        assert C == self.dim
        kernel_size = 9

        # process_x_mamba: flatten spatial -> Mamba -> norm -> reshape
        x_out = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_out = self.mamba(x_out)
        x_out = self.norm(x_out)
        x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)

        # process_y_mamba: permute H/W -> flatten -> Mamba -> norm -> reshape -> permute back
        y_out = x.permute(0, 1, 3, 2)
        y_out = y_out.reshape(B, C, n_tokens).transpose(-1, -2)
        y_out = self.mamba(y_out)
        y_out = self.norm(y_out)
        y_out = y_out.transpose(-1, -2).reshape(B, C, *img_dims)
        y_out = y_out.permute(0, 1, 3, 2)

        # process_x_direction: SerpScan (morph=0) -> reshape chunks -> Mamba -> norm -> map back
        dscx, grid_x = self.serp_scan_x(x)
        B2, C2, H, W = dscx.shape
        chunks = H // kernel_size
        new_H = chunks * kernel_size
        dscx_reshaped = dscx[:, :, :new_H].view(B, C, chunks, kernel_size, W)
        dscx_reshaped = dscx_reshaped.view(B, C, chunks, -1)
        dscx_reshaped = dscx_reshaped.view(B, C, -1).transpose(-1, -2)
        dscx_reshaped = self.mamba(dscx_reshaped)
        dscx_reshaped = self.norm(dscx_reshaped)
        dscx_reshaped = dscx_reshaped.transpose(-1, -2).view(B, C, kernel_size, H * W // kernel_size)
        dscx_reshaped = (dscx_reshaped.view(B, C, kernel_size, W, chunks).permute(0, 1, 2, 4, 3)).reshape(B, C, -1, W)
        dscx_reshaped = map_deformed_to_input(dscx_reshaped, x, grid_x, "bilinear")

        # process_y_direction: SerpScan (morph=1) -> flatten -> Mamba -> norm -> map back
        dscy, grid_y = self.serp_scan_y(x)
        B3, C3, Hy, Wy = dscy.shape
        n_tokens_y = dscy.shape[2:].numel()
        img_dims_y = dscy.shape[2:]
        dscy_flat = dscy.reshape(B, C, n_tokens_y).transpose(-1, -2)
        dscy_flat = self.mamba(dscy_flat)
        dscy_flat = self.norm(dscy_flat)
        dscy_flat = dscy_flat.transpose(-1, -2).reshape(B, C, *img_dims_y)
        dscy_flat = map_deformed_to_input(dscy_flat, x, grid_y, "bilinear")

        out = self.fusion(torch.cat([dscx_reshaped, dscy_flat], 1)) + x_out + y_out
        return out


# ---------------------------------------------------------------------------
# ResidualMambaEncoder (faithful to official source)
# ---------------------------------------------------------------------------

class ResidualMambaEncoder(nn.Module):
    """Encoder with StackedResidualBlocks + MambaLayer per stage (faithful to source).
        Encoder with StackedResidualBlocks + MambaLayer per stage.
    """
    def __init__(self, input_channels, n_stages, features_per_stage,
                 kernel_sizes=None, strides=None, n_blocks_per_stage=None,
                 conv_bias=True):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if kernel_sizes is None:
            kernel_sizes = [3] * n_stages
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if strides is None:
            strides = [1] + [2] * (n_stages - 1)
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages

        self.output_channels = list(features_per_stage)
        self.strides = [strides] if isinstance(strides[0], int) else [list(s) for s in strides]
        self.kernel_sizes = list(kernel_sizes)
        self.conv_bias = conv_bias

        # Stem (stride-1, StackedConvBlocks equivalent)
        stem_ch = features_per_stage[0]
        ks0 = kernel_sizes[0]
        pad0 = ks0 // 2
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, stem_ch, ks0, stride=1, padding=pad0, bias=conv_bias),
            nn.InstanceNorm2d(stem_ch, eps=1e-5, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # Stages with StackedResidualBlocks + MambaLayer
        stages = []
        mamba_layers = []
        ch = stem_ch
        for s in range(n_stages):
            ks = kernel_sizes[s]
            pad = ks // 2
            stage = nn.Sequential(
                BasicResBlock(ch, features_per_stage[s], ks, pad,
                              stride=strides[s], use_1x1conv=True),
                *[BasicBlockD(features_per_stage[s], ks, conv_bias)
                  for _ in range(n_blocks_per_stage[s] - 1)]
            )
            stages.append(stage)
            ch = features_per_stage[s]
            # Stage 5 (bottleneck) uses ambiguous_scan, others use serpentine_scan
            if s == n_stages - 1 and n_stages >= 6:
                mamba_layers.append(MambaLayer_ambiguous_scan(ch))
            else:
                mamba_layers.append(MambaLayer_Serpentine_Scan(ch))

        self.stages = nn.ModuleList(stages)
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.return_skips = True

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            ret.append(x)
        if self.return_skips:
            return ret
        return ret[-1]


# ---------------------------------------------------------------------------
# UNetResDecoder (faithful to official source: ConvTranspose2d + StackedResidualBlocks)
# ---------------------------------------------------------------------------

class UNetResDecoder(nn.Module):
    """Decoder with ConvTranspose2d + StackedResidualBlocks + deep supervision (faithful to source).
        Decoder with ConvTranspose2d + StackedResidualBlocks + deep supervision.
    """
    def __init__(self, encoder, num_classes, n_conv_per_stage=None,
                 deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        n_stages_encoder = len(encoder.output_channels)
        if n_conv_per_stage is None:
            n_conv_per_stage = [2] * (n_stages_encoder - 1)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            if isinstance(stride_for_transpconv, (list, tuple)):
                stride_for_transpconv = stride_for_transpconv[0]
            transpconvs.append(nn.ConvTranspose2d(
                input_features_below, input_features_skip,
                stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias))
            ks = encoder.kernel_sizes[-(s + 1)]
            pad = ks // 2
            stages.append(nn.Sequential(
                BasicResBlock(2 * input_features_skip, input_features_skip, ks, pad,
                              stride=1, use_1x1conv=True),
                *[BasicBlockD(input_features_skip, ks, encoder.conv_bias)
                  for _ in range(n_conv_per_stage[s - 1] - 1)]
            ))
            seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            y = skips[-(s + 2)]
            if x.shape[2:] != y.shape[2:]:
                x = F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat((x, y), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x
        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision or not self.training:
            return seg_outputs[0]
        return seg_outputs


# ---------------------------------------------------------------------------
# SerpMamba model (faithful to official source)
# ---------------------------------------------------------------------------

class SerpMamba(nn.Module):
    """Serp-Mamba for retinal vessel segmentation (faithful to official source).
        Serp-Mamba for retinal vessel segmentation.

    Architecture:
      ResidualMambaEncoder (stem + n_stages with MambaLayer_Serpentine_Scan
      and MambaLayer_ambiguous_scan at bottleneck) + UNetResDecoder with
      ConvTranspose2d and deep supervision.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 n_stages=5, base_features=32, n_conv_per_stage=2,
                 n_conv_per_stage_decoder=2, deep_supervision=False, **kwargs):
        super().__init__()
        features_per_stage = [min(base_features * (2 ** i), 320) for i in range(n_stages)]
        kernel_sizes = [3] * n_stages
        strides = [1] + [2] * (n_stages - 1)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        self.encoder = ResidualMambaEncoder(
            input_channels=in_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_conv_per_stage,
            conv_bias=True)
        self.decoder = UNetResDecoder(
            self.encoder, num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision)

    def forward(self, x):
        input_size = x.shape[2:]
        skips = self.encoder(x)
        out = self.decoder(skips)
        if isinstance(out, list):
            out = [F.interpolate(o, size=input_size, mode='bilinear', align_corners=False) for o in out]
        else:
            if out.shape[2:] != input_size:
                out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out
