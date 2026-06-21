# 知识蒸馏

[English](distillation.md)

本框架内置 **27** 种知识蒸馏方法，位于 `medseg/training/distillation/`。

## 方法列表

### 经典蒸馏 (23)

| 方法 | 论文 | 发表 | GitHub | 说明 | YAML |
|------|------|------|--------|------|------|
| `vanilla_kd` | Hinton et al. | NeurIPS-W 2014 | [peterliht/knowledge-distillation-pytorch](https://github.com/peterliht/knowledge-distillation-pytorch) | 软标签蒸馏 | [vanilla_kd.yaml](../../configs/training_paradigms/distillation/vanilla_kd.yaml) |
| `unet_distillation` | Hinton et al. | - | - | Logit/特征/注意力多尺度蒸馏 | - |
| `hint_distillation` | Romero et al. (FitNets) | ICLR 2015 | [adri-romsor/FitNets](https://github.com/adri-romsor/FitNets) | 中间层提示 | - |
| `attention_mimicry` | 简化注意力 | - | - | 注意力图模仿基线 | - |
| `at` | Zagoruyko & Komodakis | ICLR 2017 | [szagoruyko/attention-transfer](https://github.com/szagoruyko/attention-transfer) | 注意力迁移 | [at.yaml](../../configs/training_paradigms/distillation/at.yaml) |
| `fsp` | Yim et al. (A Gift from KD) | CVPR 2017 | [yoshitomo-matsubara/torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) | 解题流程 | [fsp.yaml](../../configs/training_paradigms/distillation/fsp.yaml) |
| `nst` | Huang & Wang | 2017 | [HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller) | 神经元选择性迁移 | [nst.yaml](../../configs/training_paradigms/distillation/nst.yaml) |
| `rkd` | Park et al. | CVPR 2019 | [lenscloth/RKD](https://github.com/lenscloth/RKD) | 关系蒸馏 | [rkd.yaml](../../configs/training_paradigms/distillation/rkd.yaml) |
| `vid` | Ahn et al. | CVPR 2019 | [HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller) | 变分信息蒸馏 | [vid.yaml](../../configs/training_paradigms/distillation/vid.yaml) |
| `dkd` | Zhao et al. | CVPR 2022 | [megvii-research/mdistiller](https://github.com/megvii-research/mdistiller) | 解耦蒸馏 | - |
| `mgd` | Yang et al. | ECCV 2022 | [yzd-v/MGD](https://github.com/yzd-v/MGD) | 掩码生成蒸馏 | - |
| `dist` | Huang et al. | NeurIPS 2022 | [hunto/DIST_KD](https://github.com/hunto/DIST_KD) | DIST 蒸馏 | - |
| `cirkd_minibatch` | Yang et al. | CVPR 2022 | [winycg/CIRKD](https://github.com/winycg/CIRKD) | 跨图像关系蒸馏 | - |
| `cwd` | Shu et al. | ICCV 2021 | [irfanICMLL/TorchDistiller](https://github.com/irfanICMLL/TorchDistiller) | 通道级蒸馏 | [cwd.yaml](../../configs/training_paradigms/distillation/cwd.yaml) |
| `review_kd` | Chen et al. | CVPR 2021 | [dvlab-research/ReviewKD](https://github.com/dvlab-research/ReviewKD) | 知识回顾 | [review_kd.yaml](../../configs/training_paradigms/distillation/review_kd.yaml) |
| `simkd` | Chen et al. | CVPR 2022 | [DefangChen/SimKD](https://github.com/DefangChen/SimKD) | 简单投影蒸馏 | - |
| `norm_kd` | Liu et al. (NormKD) | ICLR 2023 | [xyliu7/NORM](https://github.com/xyliu7/NORM) *(已失效)* | 归一化 logits 蒸馏 | [norm_kd.yaml](../../configs/training_paradigms/distillation/norm_kd.yaml) |
| `sdd` | Wei et al. | CVPR 2024 | [shicaiwei123/SDD-CVPR2024](https://github.com/shicaiwei123/SDD-CVPR2024) | 尺度解耦蒸馏 | [sdd.yaml](../../configs/training_paradigms/distillation/sdd.yaml) |
| `aicsd` | Mansurian et al. | TNNLS 2024 | [AmirMansurian/AICSD](https://github.com/AmirMansurian/AICSD) | 自适应类间相似度 | [aicsd.yaml](../../configs/training_paradigms/distillation/aicsd.yaml) |
| `logit_std_kd` | Sun et al. | CVPR 2024 | [sunshangquan/logit-standardization-KD](https://github.com/sunshangquan/logit-standardization-KD) | Logit 标准化 | [logit_std_kd.yaml](../../configs/training_paradigms/distillation/logit_std_kd.yaml) |
| `ttm_kd` | Zheng & Yang | ICLR 2024 | [zkxufo/TTM](https://github.com/zkxufo/TTM) | 变换教师匹配 | [ttm_kd.yaml](../../configs/training_paradigms/distillation/ttm_kd.yaml) |
| `ctkd` | Li et al. | AAAI 2023 | [zhengli97/CTKD](https://github.com/zhengli97/CTKD) | 课程温度 | [ctkd.yaml](../../configs/training_paradigms/distillation/ctkd.yaml) |
| `mlkd` | Jin et al. | CVPR 2023 | [Jin-Ying/Multi-Level-Logit-Distillation](https://github.com/Jin-Ying/Multi-Level-Logit-Distillation) | 多级 logit 蒸馏 | [mlkd.yaml](../../configs/training_paradigms/distillation/mlkd.yaml) |

### 医学专用蒸馏 (4)

| 方法 | 说明 | YAML |
|------|------|------|
| `anatomy_kd` | 器官拓扑蒸馏 | [anatomy_kd.yaml](../../configs/training_paradigms/distillation/anatomy_kd.yaml) |
| `boundary_kd` | 边界感知蒸馏 | [boundary_kd.yaml](../../configs/training_paradigms/distillation/boundary_kd.yaml) |
| `multi_organ_kd` | 多器官类均衡蒸馏 | [multi_organ_kd.yaml](../../configs/training_paradigms/distillation/multi_organ_kd.yaml) |
| `cross_modality_kd` | 跨模态蒸馏（CT/MRI） | [cross_modality_kd.yaml](../../configs/training_paradigms/distillation/cross_modality_kd.yaml) |

## 配置示例

**教师配置** (`configs/training_paradigms/distillation/teacher_large.yaml`)：

```yaml
model:
  num_classes: 4
  img_size: 224
  encoder:
    name: timm_resnet50
    pretrained: true
    in_channels: 3
  decoder:
    name: unet
```

**学生配置**（如 `aicsd.yaml`）：

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
  source:
    image_dir: ./data/source/images
    mask_dir: ./data/source/masks
  target:
    image_dir: ./data/target/images
  val:
    image_dir: ./data/target_val/images
    mask_dir: ./data/target_val/masks

distillation:
  method: aicsd            # 表中任意方法名
  weight: 1.0
  params:
    temperature: 1.0
    lambda_ics: 1.0
    lambda_icc: 1.0

training:
  epochs: 100
  batch_size: 8
  num_workers: 4
  val_interval: 10
  save_interval: 50
```

## 用法

```bash
# 蒸馏训练
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --student_config configs/training_paradigms/distillation/aicsd.yaml

# 指定教师权重
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --teacher_checkpoint output/teacher/best_model.pth \
    --student_config configs/training_paradigms/distillation/vanilla_kd.yaml

# 测试学生模型
python test.py --config configs/training_paradigms/distillation/aicsd.yaml \
    --checkpoint output/student/best_model.pth
```

## 流程

```
预训练教师 ─────┐
                ├──► 蒸馏训练器 ──► 紧凑学生模型
训练数据 ───────┘
```

每个方法的配置位于 `configs/training_paradigms/distillation/`。
