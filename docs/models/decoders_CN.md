# 解码器

[English](decoders.md)

本项目提供 40 个解码器模块，按类别分组如下。

## 基础 (4)

基础上采样解码器。

| 名称 | 说明 | YAML |
|---|---|---|
| `unet` | 标准 UNet 反卷积解码器 | [unet_basic](../../configs/architectures/combinations/general/unet_basic.yaml) |
| `bilinear` | 双线性插值上采样 | [bilinear](../../configs/architectures/decoder_study/general/basic_bilinear.yaml) |
| `deconv` | 转置卷积上采样 | [deconv](../../configs/architectures/combinations/general/deconv_resnet34.yaml) |
| `dw_sep` | 深度可分离卷积解码器 | [dw_sep](../../configs/architectures/combinations/general/dwsep_resnet34.yaml) |

## 密集连接 (2)

密集连接解码器。

| 名称 | 说明 | YAML |
|---|---|---|
| `unetpp` | UNet++ 密集嵌套解码器 | [unetpp](../../configs/architectures/decoder_study/general/basic_unetpp.yaml) |
| `unet3plus` | UNet 3+ 全尺度跳跃连接解码器 | [unet3plus](../../configs/architectures/decoder_study/general/basic_unet3plus.yaml) |

## 级联 (10)

级联解码器，逐步细化分割结果。

| 名称 | 说明 | YAML |
|---|---|---|
| `cascade` | CASCADE 级联解码器 | [cascade_resnet34](../../configs/architectures/combinations/general/cascade_resnet34.yaml) |
| `cascade_full` | CASCADE 完整版解码器 | [transunet_cascade_full](../../configs/architectures/combinations/general/transunet_cascade_full.yaml) |
| `cascade_emcad` | CASCADE + EMCAD 混合 | [mednext_cascade_emcad](../../configs/architectures/combinations/general/mednext_cascade_emcad.yaml) |
| `cfm` | CFM 级联特征融合 | [mednext_cfm](../../configs/architectures/combinations/general/mednext_cfm.yaml) |
| `emcad` | EMCAD 高效多尺度级联注意力解码器 | [mednext_emcad](../../configs/architectures/combinations/general/mednext_emcad.yaml) |
| `edldnet` | EDLDNet 解码器 | [pvtv2_edldnet](../../configs/architectures/combinations/general/pvtv2_edldnet.yaml) |
| `gcascade` | G-CASCADE（add 融合） | [pvtv2_gcascade](../../configs/architectures/combinations/general/pvtv2_gcascade.yaml) |
| `gcascade_cat` | G-CASCADE（concat 融合） | [gcascade_cat](../../configs/architectures/decoder_study/general/basic_gcascade_cat.yaml) |
| `merit_add` | MERIT 解码器（add 融合） | [merit_add](../../configs/architectures/decoder_study/general/basic_merit_add.yaml) |
| `merit_cat` | MERIT 解码器（concat 融合） | [merit_cat](../../configs/architectures/decoder_study/general/basic_merit_cat.yaml) |

## 金字塔 (1)

金字塔聚合解码器。

| 名称 | 说明 | YAML |
|---|---|---|
| `upernet` | UPerNet 统一感知金字塔 | [upernet](../../configs/architectures/decoder_study/general/basic_upernet.yaml) |

## MLP (2)

MLP 解码器。

| 名称 | 说明 | YAML |
|---|---|---|
| `mlp` | 通用 MLP 解码器 | [mlp_resnet34](../../configs/architectures/combinations/general/mlp_resnet34.yaml) |
| `segformer` | SegFormer 风格 MLP 解码器 | [swinunet_segformer](../../configs/architectures/combinations/general/swinunet_segformer.yaml) |

## 特定网络专属 (12)

特定网络专属解码器。

| 名称 | 对应网络 | YAML |
|---|---|---|
| `cfanet` | CFA-Net | [cfanet](../../configs/architectures/decoder_study/general/basic_cfanet.yaml) |
| `dcsaunet` | DCSAU-Net | [dcsaunet](../../configs/architectures/decoder_study/general/basic_dcsaunet.yaml) |
| `rwkv_unet` | RWKV-UNet | [rwkv_unet](../../configs/architectures/combinations/general/rwkv_unet.yaml) |
| `kiunet` | KiU-Net | [kiunet](../../configs/architectures/decoder_study/general/basic_kiunet.yaml) |
| `transunet` | TransUNet (CUP) | [transunet](../../configs/architectures/combinations/general/transunet_cascade_full.yaml) |
| `fatnet` | FAT-Net | [fatnet](../../configs/architectures/decoder_study/general/basic_fatnet.yaml) |
| `h2former` | H2Former | [h2former](../../configs/architectures/decoder_study/general/basic_h2former.yaml) |
| `hiformer` | HiFormer | [hiformer](../../configs/architectures/combinations/general/hiformer_cascade.yaml) |
| `missformer` | MISSFormer | [missformer](../../configs/architectures/decoder_study/general/basic_missformer.yaml) |
| `scaleformer` | ScaleFormer | [scaleformer](../../configs/architectures/combinations/general/scaleformer_cascade_full.yaml) |
| `malunet` | MALUNet | [malunet](../../configs/architectures/decoder_study/general/basic_malunet.yaml) |
| `ege_unet` | EGE-UNet | [ege_unet](../../configs/architectures/decoder_study/general/basic_ege_unet.yaml) |

## Transformer (5)

Transformer 解码器。

| 名称 | 说明 | YAML |
|---|---|---|
| `daeformer` | DAEFormer 解码器 | [daeformer](../../configs/architectures/combinations/general/daeformer_emcad.yaml) |
| `mtunet` | MT-UNet 解码器 | [mtunet](../../configs/architectures/decoder_study/general/basic_mtunet.yaml) |
| `nnformer` | nnFormer 解码器 | [mednext_nnformer](../../configs/architectures/combinations/general/mednext_nnformer.yaml) |
| `swinunet` | Swin-UNet 解码器 | [swinunet](../../configs/architectures/combinations/general/swinunet_segformer.yaml) |
| `uctransnet` | UCTransNet 解码器 | [uctransnet](../../configs/architectures/combinations/general/uctransnet.yaml) |

## 注意力 (3)

注意力机制解码器。

| 名称 | 说明 | YAML |
|---|---|---|
| `attention` | 注意力门控解码器 | [attention_unet](../../configs/architectures/combinations/general/attention_unet_basic.yaml) |
| `ham` | HAM 混合注意力 | [ham_resnet34](../../configs/architectures/combinations/general/ham_resnet34.yaml) |
| `lawin` | Lawin 大窗口注意力 | [lawin_resnet50](../../configs/architectures/combinations/general/lawin_resnet50.yaml) |

## Mamba (1)

| 名称 | 说明 | YAML |
|---|---|---|
| `vmunet` | VM-UNet Mamba 解码器 | [vmunet](../../configs/architectures/networks/general/vm_unet.yaml) |

---

## YAML 使用示例

```yaml
model:
  num_classes: 9
  img_size: 224
  encoder:
    name: timm_resnet50
    pretrained: true
    in_channels: 3
  decoder:
    name: emcad          # 选择任意解码器
    params: {}
  skip_connection:
    name: concat
  bottleneck:
    name: none

data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  test_dir: ./data/Synapse/test_vol_h5
  train_list: ./data/Synapse/lists/lists_Synapse/train.txt
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt

training:
  epochs: 200
  batch_size: 24
  num_workers: 4
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4
        - name: dice
          weight: 0.6
  optimizer:
    name: adamw
    lr: 0.01
    weight_decay: 0.0001
  scheduler:
    name: cosine
    min_lr: 0.000001
```
