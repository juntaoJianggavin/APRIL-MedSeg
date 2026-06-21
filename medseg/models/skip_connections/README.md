# Skip Connections (`medseg/models/skip_connections/`)

[中文文档](README_CN.md)

Modules that process features at each encoder-to-decoder skip level before concatenation.

## Registered Skip Connections (25 keys)

### basic/ — Basic

| Registry Key | Source File | Description |
|---|---|---|
| `add` | `basic_skip.py` | Element-wise addition (ResNet-style) |
| `concat` | `basic_skip.py` | Channel-wise concatenation (U-Net default) |
| `dense` | `dense_skip.py` | Dense connection (UNet++ style) |

### attention/ — Attention

| Registry Key | Source File | Description |
|---|---|---|
| `attention_gate` | `attention_gate_skip.py` | Attention Gate (Attention U-Net, Oktay 2018) |
| `cab` | `cab_skip.py` | Channel Attention Block |
| `sab` | `sab_skip.py` | Spatial Attention Block |
| `scse` | `scse_skip.py` | Concurrent Spatial + Channel SE (scSE) |
| `cbam` | `cbam_skip.py` | CBAM Channel + Spatial Attention |
| `gating` | `gating_skip.py` | Gating mechanism (attention gate) |
| `gru_gate` | `gru_gate_skip.py` | GRU-style gated skip |
| `gab` | `gab_skip.py` | Group Aggregation Bridge (EGE-UNet, MICCAI 2023 W) |
| `sc_att_bridge` | `sc_att_bridge_skip.py` | SC-Att-Bridge spatial + channel attention (MALUNet, BIBM 2022) |
| `ta_mosc` | `ta_mosc_skip.py` | Task-Adaptive Mixture of Skip Connections (UTANet, AAAI 2025) |

### transformer/ — Transformer

| Registry Key | Source File | Description |
|---|---|---|
| `cross_attn` | `cross_attn_skip.py` | Cross-attention between encoder and decoder features |
| `transformer_fusion` | `transformer_fusion_skip.py` | Transformer cross-fusion |
| `aggregation_attention` | `aggregation_attention_skip.py` | CASCADE AAM-style aggregation attention |
| `missformer_bridge` | `missformer_bridge_skip.py` | MISSFormer Bridge (MICCAI 2022) |
| `uctrans` | `uctrans_skip.py` | UCTransNet channel-wise cross transformer (AAAI 2022) |

### mamba/ — Mamba

| Registry Key | Source File | Description |
|---|---|---|
| `skvmpp` | `skvmpp_skip.py` | SK-VM++ Mamba-assisted skip (BSPC 2025) |

### fusion/ — CNN Fusion

| Registry Key | Source File | Description |
|---|---|---|
| `bifusion` | `bifusion_skip.py` | BiFusion parallel-branch (TransFuse-style) |
| `deformable` | `deformable_skip.py` | Deformable convolution skip |
| `multiscale` | `multiscale_skip.py` | Multi-scale feature fusion |
| `feature_refine` | `feature_refine_skip.py` | Learned feature refinement |
| `ccm` | `ccm_skip.py` | Cross-Channel Matching |
| `sdi` | `sdi_skip.py` | Scale-Diverse Integration (U-Net V2, ISBI 2025) |

## Usage in YAML Config

```yaml
model:
  skip:
    name: scse               # any registered key
    params: {}               # extra kwargs
```

## Choosing a Skip Connection

| Scenario | Recommended |
|---|---|
| Baseline / speed | `concat` or `add` |
| Better boundary detail | `scse`, `cab`, `sab` |
| Transformer / attention models | `cross_attn`, `feature_refine` |
| Dense skip (UNet++) | `dense` |
| Multi-scale fusion | `multiscale` |
