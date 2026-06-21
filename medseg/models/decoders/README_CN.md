# 解码器 (`medseg/models/decoders/`)

[English](README.md)

U 形上采样解码器，接收多尺度编码器特征并生成分割图。所有解码器遵循统一接口：`forward(bottleneck_feat, skip_features)`。

## 已注册解码器（47 个注册键）

### basic/ — 基础上采样

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `unet` / `deconv_upcat` | `unet_decoder.py` | UNet 解码器：ConvTranspose（通道减半）→ 拼接skip → DoubleConv。别名 `deconv_upcat` = 先**上采样**后**拼接** |
| `bilinear` | `bilinear_decoder.py` | 双线性上采样 + 1×1 卷积（最快基线） |
| `deconv` / `deconv_catup` | `deconv_decoder.py` | 转置卷积解码器：拼接skip → ConvTranspose（合并通道）→ 卷积。别名 `deconv_catup` = 先**拼接**后**上采样** |
| `dw_sep` | `dw_sep_decoder.py` | 深度可分离卷积解码器（轻量级） |

### dense/ — 密集连接

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `unetpp` | `unetpp_decoder.py` | UNet++ 密集跳跃解码器 |
| `unet3plus` | `unet3plus_decoder.py` | UNet3+ 全尺度跳跃解码器 |

### cascade/ — 级联细化

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `cascade` | `cascade_decoder.py` | 级联细化解码器 |
| `cascade_full` | `cascade_full_decoder.py` | 完全级联解码器（所有层级联） |
| `cascade_emcad` | `cascade_emcad_decoder.py` | CASCADE + EMCAD 聚合 |
| `cfm` | `cfm_decoder.py` | CFM（级联融合模块）解码器 |
| `emcad` | `emcad_decoder.py` | EMCAD 多尺度聚合解码器 |
| `edldnet` | `edldnet_decoder.py` | EDLDNet 解码器 |
| `gcascade` | `gcascade_decoder.py` | G-CASCADE 解码器（门控级联） |
| `gcascade_cat` | `gcascade_decoder.py` | G-CASCADE 解码器（拼接变体） |
| `merit_add` | `merit_decoder.py` | MERIT 解码器（加法融合） |
| `merit_cat` | `merit_decoder.py` | MERIT 解码器（拼接融合） |

### pyramid/ — 金字塔聚合

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `upernet` | `upernet_decoder.py` | UPerNet FPN + PPM 解码器 |
| `deeplabv3` | `deeplabv3_decoder.py` | DeepLabV3 ASPP 解码器 |

### mlp/ — MLP 解码器

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `mlp` | `mlp_decoder.py` | MLP 解码器（SegFormer 风格） |
| `segformer` | `segformer_decoder.py` | SegFormer MLP 解码器 |

### transformer/ — Transformer（Q/K/V 自注意力或交叉注意力块）

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `daeformer` | `daeformer_decoder.py` | DAEFormer 交叉注意力解码器（CrossAttentionBlock） |
| `missformer` | `missformer_decoder.py` | MISSFormer Transformer 解码器（PatchExpand + TransformerBlock） |
| `mtunet` | `mtunet_decoder.py` | MT-UNet 解码器（窗口 + 轴向 + 记忆高效注意力） |
| `nnformer` | `nnformer_decoder.py` | nnFormer 解码器（PatchExpand + Swin 窗口注意力） |
| `swinunet` | `swinunet_decoder.py` | Swin-UNet 分块扩展解码器（SwinTransformerBlock） |

### attention/ — 注意力（门控、空间/通道注意力、上下文聚合）

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `attention` | `attention_decoder.py` | 每层跳跃连接处注意力门控（Attention U-Net） |
| `banet` | `banet_decoder.py` | BANet 边界感知空间注意力解码器 |
| `ccnet` | `ccnet_decoder.py` | CCNet 十字交叉注意力解码器（Q/K/V + softmax） |
| `lawin` | `lawin_decoder.py` | Lawin 大窗口注意力解码器 + 空间金字塔 |
| `ocrnet` | `ocrnet_decoder.py` | OCRNet 目标上下文表示（Q/K/V 注意力做上下文聚合） |
| `uctransnet` | `uctransnet_decoder.py` | UCTransNet 通道交叉注意力（CCA, SE-Net 风格）+ 双线性上采样 |

### mamba/ — Mamba

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `vmunet` | `vmunet_decoder.py` | VM-UNet Mamba-SSM 解码器 |

### specific/ — 特定网络专属（独特机制）

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `cfanet` | `cfanet_decoder.py` | CFA-Net 跨层级特征聚合解码器 |
| `dcsaunet` | `dcsaunet_decoder.py` | DCSAU-Net 双路径解码器 |
| `ege_unet` | `ege_unet_decoder.py` | EGNet 边缘引导解码器 |
| `fatnet` | `fatnet_decoder.py` | FAT-Net 频率感知 Transformer 解码器 |
| `ffparser` | `ffparser_decoder.py` | FF-Parser 频域特征滤波解码器 |
| `h2former` | `h2former_decoder.py` | H2Former 双线性上采样 + 拼接解码器 |
| `ham` | `ham_decoder.py` | HAM（Hamburger）NMF 矩阵分解全局上下文建模解码器 |
| `hiformer` | `hiformer_decoder.py` | HiFormer ConvUpsample 解码器 |
| `kiunet` | `kiunet_decoder.py` | KI-UNet 解码器 |
| `malunet` | `malunet_decoder.py` | MALUNet 混合跳跃解码器 |
| `rwkv_unet` | `rwkv_unet_decoder.py` | RWKV 序列模型解码器 |
| `scaleformer` | `scaleformer_decoder.py` | ScaleFormer 4 阶段 UNet 解码器 |
| `transunet` | `transunet_decoder.py` | TransUNet 级联上卷积解码器 |

## YAML 配置用法

```yaml
model:
  decoder:
    name: attention         # 任意已注册键
    params:
      encoder_channels: [64, 256, 512, 1024, 2048]
      decoder_channels: [256, 128, 64, 32, 16]
```
