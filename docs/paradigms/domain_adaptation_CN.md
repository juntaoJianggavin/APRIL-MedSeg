# 域适应

[English](domain_adaptation.md)

本框架内置 **18** 种无监督域适应 (UDA) 方法，位于 `medseg/training/domain_adaptation/`。

## 方法列表

| 方法 | 论文 | 发表 | GitHub | 说明 | YAML |
|------|------|------|--------|------|------|
| `source_only` | Baseline | - | [valeoai/ADVENT](https://github.com/valeoai/ADVENT) | 下界基线：仅源域 CE+Dice | [source_only.yaml](../../configs/training_paradigms/domain_adaptation/source_only.yaml) |
| `advent` | Vu et al. | CVPR 2019 | [valeoai/ADVENT](https://github.com/valeoai/ADVENT) | 对抗熵最小化 | [advent.yaml](../../configs/training_paradigms/domain_adaptation/advent.yaml) |
| `dann` | Ganin et al. | JMLR 2016 | [fungtion/DANN](https://github.com/fungtion/DANN) | 域对抗网络 | [dann.yaml](../../configs/training_paradigms/domain_adaptation/dann.yaml) |
| `tent` | Wang et al. | ICLR 2021 | [DequanWang/tent](https://github.com/DequanWang/tent) | 测试时自适应 | [tent.yaml](../../configs/training_paradigms/domain_adaptation/tent.yaml) |
| `dpl` | Chen et al. | MICCAI 2021 | [cchen-cc/SFDA-DPL](https://github.com/cchen-cc/SFDA-DPL) | 双伪标签 | [dpl.yaml](../../configs/training_paradigms/domain_adaptation/dpl.yaml) |
| `cbmt` | ADA4MIA benchmark | - | [whq-xxh/ADA4MIA](https://github.com/whq-xxh/ADA4MIA) | 类均衡均值教师 | [cbmt.yaml](../../configs/training_paradigms/domain_adaptation/cbmt.yaml) |
| `fda` | Yang & Soatto | CVPR 2020 | [YanchaoYang/FDA](https://github.com/YanchaoYang/FDA) | 傅里叶域适应 | [fda.yaml](../../configs/training_paradigms/domain_adaptation/fda.yaml) |
| `crst` | Zou et al. | ICCV 2019 | [yzou2/CRST](https://github.com/yzou2/CRST) | 类均衡自训练 | [crst.yaml](../../configs/training_paradigms/domain_adaptation/crst.yaml) |
| `pixmatch` | Melas-Kyriazi & Manrai | CVPR 2021 | [lukemelas/pixmatch](https://github.com/lukemelas/pixmatch) | 像素级对比匹配 | [pixmatch.yaml](../../configs/training_paradigms/domain_adaptation/pixmatch.yaml) |
| `mic` | Hoyer et al. | CVPR 2023 | [lhoyer/MIC](https://github.com/lhoyer/MIC) | 掩码图像一致性 | [mic.yaml](../../configs/training_paradigms/domain_adaptation/mic.yaml) |
| `daformer_fd` | Hoyer et al. | CVPR 2022 | [lhoyer/DAFormer](https://github.com/lhoyer/DAFormer) | 特征距离 + 稀有类采样 | [daformer_fd.yaml](../../configs/training_paradigms/domain_adaptation/daformer_fd.yaml) |
| `hrda` | Hoyer et al. | ECCV 2022 | [lhoyer/HRDA](https://github.com/lhoyer/HRDA) | 多分辨率尺度注意力 | [hrda.yaml](../../configs/training_paradigms/domain_adaptation/hrda.yaml) |
| `pipa` | Chen et al. | ACM MM 2023 | [chen742/PiPa](https://github.com/chen742/PiPa) | 像素 + patch InfoNCE | [pipa.yaml](../../configs/training_paradigms/domain_adaptation/pipa.yaml) |
| `ddb` | Du et al. | NeurIPS 2022 | [xiaoachen98/DDB](https://github.com/xiaoachen98/DDB) | 审慎域桥接 | [ddb.yaml](../../configs/training_paradigms/domain_adaptation/ddb.yaml) |
| `sepico` | Xie et al. | TPAMI 2023 | [BIT-DA/SePiCo](https://github.com/BIT-DA/SePiCo) | 语义像素对比 + KL | [sepico.yaml](../../configs/training_paradigms/domain_adaptation/sepico.yaml) |
| `diga` | Shen et al. | CVPR 2023 | [BIT-DA/DiGA](https://github.com/BIT-DA/DiGA) | 蒸馏引导适应 | [diga.yaml](../../configs/training_paradigms/domain_adaptation/diga.yaml) |
| `micdrop` | Hoyer et al. | ECCV 2024 | [lhoyer/HRDA](https://github.com/lhoyer/HRDA) | MIC + 互补特征丢弃 | [micdrop.yaml](../../configs/training_paradigms/domain_adaptation/micdrop.yaml) |
| `semivl_da` | Karazija et al. | ECCV 2024 | [google-research/semivl](https://github.com/google-research/semivl) | 视觉-语言引导自训练 | [semivl.yaml](../../configs/training_paradigms/domain_adaptation/semivl.yaml) |

## 配置示例

```yaml
model:
  num_classes: 4
  img_size: 224
  encoder:
    name: timm_resnet34
    pretrained: true
    in_channels: 3
  decoder:
    name: bilinear
  bottleneck:
    name: none

data:
  img_size: 224
  source:                          # 源域（有标签）
    image_dir: ./data/source/images
    mask_dir: ./data/source/masks
  target:                          # 目标域（无标签）
    image_dir: ./data/target/images
  val:                             # 目标域验证
    image_dir: ./data/target_val/images
    mask_dir: ./data/target_val/masks
  test:
    image_dir: ./data/target_test/images
    mask_dir: ./data/target_test/masks

domain_adaptation:
  method: advent                   # 表中任意方法名
  params:
    entropy_weight: 0.1
    adversarial_weight: 0.1
    num_classes: 4

training:
  epochs: 100
  batch_size: 8
  num_workers: 4
  val_interval: 10
  save_interval: 50
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
    lr: 1e-4
    weight_decay: 1e-4
  scheduler:
    name: cosine
    min_lr: 1e-6
```

## 用法

```bash
# 训练
python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/advent.yaml

# 多卡训练
torchrun --nproc_per_node=4 train_domain_adaptation.py \
    --config configs/training_paradigms/domain_adaptation/mic.yaml

# 目标域测试
python test.py --config configs/training_paradigms/domain_adaptation/advent.yaml \
    --checkpoint output/best_model.pth
```

## 流程

```
源域（有标签）──┐
                ├──► 域适应训练器 ──► 适应模型 ──► 目标域推理
目标域（无标签）─┘
```

每个方法的配置位于 `configs/training_paradigms/domain_adaptation/`。
