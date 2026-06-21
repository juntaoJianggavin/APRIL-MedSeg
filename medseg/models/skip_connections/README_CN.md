# 跳跃连接 (`medseg/models/skip_connections/`)

[English](README.md)

在编码器到解码器各层跳跃连接处处理特征的模块。

## 已注册跳跃连接（25 个注册键）

### basic/ — 基础

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `add` | `basic_skip.py` | 逐元素相加（ResNet 风格） |
| `concat` | `basic_skip.py` | 通道拼接（U-Net 默认） |
| `dense` | `dense_skip.py` | 密集连接（UNet++ 风格） |

### attention/ — 注意力

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `attention_gate` | `attention_gate_skip.py` | 注意力门控（Attention U-Net, Oktay 2018） |
| `cab` | `cab_skip.py` | 通道注意力块 |
| `sab` | `sab_skip.py` | 空间注意力块 |
| `scse` | `scse_skip.py` | 空间-通道并行 SE (scSE) |
| `cbam` | `cbam_skip.py` | CBAM 通道 + 空间注意力 |
| `gating` | `gating_skip.py` | 门控机制（注意力门控） |
| `gru_gate` | `gru_gate_skip.py` | GRU 风格门控跳跃 |
| `gab` | `gab_skip.py` | 组聚合桥（EGE-UNet, MICCAI 2023 W） |
| `sc_att_bridge` | `sc_att_bridge_skip.py` | SC-Att-Bridge 空间+通道注意力（MALUNet, BIBM 2022） |
| `ta_mosc` | `ta_mosc_skip.py` | 任务自适应混合跳跃连接（UTANet, AAAI 2025） |

### transformer/ — Transformer

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `cross_attn` | `cross_attn_skip.py` | 编码器-解码器特征交叉注意力 |
| `transformer_fusion` | `transformer_fusion_skip.py` | Transformer 交叉融合 |
| `aggregation_attention` | `aggregation_attention_skip.py` | CASCADE AAM 风格聚合注意力 |
| `missformer_bridge` | `missformer_bridge_skip.py` | MISSFormer 桥接（MICCAI 2022） |
| `uctrans` | `uctrans_skip.py` | UCTransNet 通道交叉 Transformer（AAAI 2022） |

### mamba/ — Mamba

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `skvmpp` | `skvmpp_skip.py` | SK-VM++ Mamba 辅助跳跃（BSPC 2025） |

### fusion/ — CNN 融合

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `bifusion` | `bifusion_skip.py` | BiFusion 并行分支（TransFuse 风格） |
| `deformable` | `deformable_skip.py` | 可变形卷积跳跃 |
| `multiscale` | `multiscale_skip.py` | 多尺度特征融合 |
| `feature_refine` | `feature_refine_skip.py` | 学习型特征精炼 |
| `ccm` | `ccm_skip.py` | 交叉通道匹配 |
| `sdi` | `sdi_skip.py` | 尺度多样集成（U-Net V2, ISBI 2025） |

## YAML 配置用法

```yaml
model:
  skip:
    name: scse               # 任意已注册键
    params: {}               # 额外参数
```

## 选择跳跃连接

| 场景 | 推荐 |
|---|---|
| 基线 / 速度 | `concat` 或 `add` |
| 更好的边界细节 | `scse`, `cab`, `sab` |
| Transformer / 注意力模型 | `cross_attn`, `feature_refine` |
| 密集跳跃 (UNet++) | `dense` |
| 多尺度融合 | `multiscale` |
