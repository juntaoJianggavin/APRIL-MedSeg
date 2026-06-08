# 第 03 讲：数据与预处理

[上一讲：U-Net](02_unet_CN.md) | [English](03_data.md) | [下一讲：训练与评估](04_training_CN.md)

---

## 1. 背景与动机

数据质量和预处理直接决定分割性能。在医学影像中，挑战更为突出：

- **格式多样**：NIfTI (`.nii.gz`)、DICOM、PNG/JPG、NumPy (`.npz`)、HDF5 (`.h5`)
- **标注不一致**：不同医院/机构的标注方案各异
- **小数据集**：许多医学数据集仅有数百个样本
- **类别不平衡**：目标结构（肿瘤、病灶）相比背景往往极小

UltimateMedSeg 提供统一的数据管线，支持 25 个内置数据集、4 种切分策略和 24 种增强方法，全部通过 YAML 配置。

---

## 2. 核心概念

### 2.1 医学图像数据格式

| 格式 | 扩展名 | 典型用途 | Python 库 |
|------|--------|----------|-----------|
| NIfTI | `.nii.gz` | CT / MRI 体积数据 | `nibabel`, `SimpleITK` |
| DICOM | `.dcm` | 原始临床数据 | `pydicom`, `SimpleITK` |
| PNG/JPG | `.png/.jpg` | 2D 图像（皮肤镜、眼底、病理） | `PIL`, `cv2` |
| NumPy | `.npz` | 预处理数组（Synapse, ACDC） | `numpy` |
| HDF5 | `.h5` | 预处理测试体积 | `h5py` |

### 2.2 目录约定

UltimateMedSeg 支持两种目录布局：

**布局 A：显式 train/val/test 目录**

```
data/YourDataset/
├── train/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

每张图像必须有对应的 mask，**文件名相同**（或可配置后缀，如 `image_001_mask.png`）。

**布局 B：扁平目录 + 自动切分**

```
data/BUSI/
├── images/
│   ├── benign (1).png
│   ├── malignant (1).png
│   └── ...
└── masks/
    ├── benign (1)_mask.png
    ├── malignant (1)_mask.png
    └── ...
```

图像和 mask 会按照可配置的比例或 K 折自动切分为 train/val/test。

### 2.3 四种切分策略

| 策略 | 配置字段 | 适用场景 |
|------|----------|----------|
| **显式路径** | `train_dir`, `val_dir`, `test_dir` | 预切分数据集（Synapse, ACDC） |
| **按比例** | `train_ratio: 0.7`, `val_ratio: 0.15` | 从扁平目录自动切分 |
| **K 折** | `n_splits: 5`, `fold_idx: 0` | 交叉验证（推荐用于小数据集） |
| **预定义列表** | `train_list: train.txt`, `test_list: test.txt` | 遵循论文的官方划分 |

---

## 3. 方法详解

### 3.1 基础变换（内置）

框架在 `medseg/datasets/transforms.py` 中包含核心变换：

| 变换 | 说明 | 参数 |
|------|------|------|
| `Resize` | 将图像和 mask 缩放到目标尺寸 | `size` |
| `Normalize` | Min-max 或 mean/std 归一化 | `mean`, `std`（可选） |
| `RandomFlip` | 水平和垂直翻转 | `p=0.5` |
| `RandomRotate90` | 90/180/270 度旋转 | `p=0.5` |
| `RandomRotation` | 任意角度旋转 | `degrees=15, p=0.5` |
| `RandomScale` | 缩放后 resize 回原尺寸 | `scale_range=(0.8,1.2), p=0.5` |
| `RandomCrop` | 随机裁剪到目标尺寸 | `size` |
| `RandomElasticDeform` | 弹性形变 | `alpha=50, sigma=5, p=0.3` |
| `GaussianNoise` | 对图像添加高斯噪声 | `mean=0, std=0.05, p=0.3` |
| `GaussianBlur` | 图像高斯模糊 | `kernel_size=3, p=0.2` |
| `BrightnessContrastJitter` | 亮度和对比度抖动 | `brightness=0.2, contrast=0.2, p=0.3` |
| `GammaCorrection` | 随机 gamma 校正 | `gamma_range=(0.7,1.5), p=0.3` |
| `CutOut` | 随机矩形遮挡 | `num_holes=1, max_h_size=32, p=0.3` |

**重要**：所有变换对图像和 mask 施加**相同的几何操作**（翻转、旋转、缩放），但**仅影响图像**进行强度操作（噪声、模糊、亮度）。

### 3.2 三种增强级别

内置的 `get_train_transforms()` 提供三个预设级别：

```python
from medseg.datasets.transforms import get_train_transforms

# 轻度：仅翻转
transforms_light = get_train_transforms(img_size=224, augment_level='light')

# 标准：翻转 + 旋转 + 缩放 + 噪声 + 亮度（默认）
transforms_std = get_train_transforms(img_size=224, augment_level='standard')

# 重度：全部 10 种增强，含弹性形变
transforms_heavy = get_train_transforms(img_size=224, augment_level='heavy')
```

| 级别 | 包含的增强 |
|------|-----------|
| `light` | RandomFlip |
| `standard` | Flip + Rotate90 + Rotation + Scale + Noise + Brightness |
| `heavy` | 以上全部 + ElasticDeform + Blur + GammaCorrection + CutOut |

### 3.3 YAML 可配置管线（推荐）

对于精细控制，使用 YAML 中的 pipeline 模式：

```yaml
training:
  augmentation: pipeline
  aug_pipeline:
    - name: horizontal_flip
      params: {p: 0.5}
    - name: vertical_flip
      params: {p: 0.3}
    - name: random_rotate90
      params: {p: 0.3}
    - name: copy_paste
      params: {p: 0.3}
    - name: mosaic
      params: {p: 0.2}
    - name: photometric_distortion
      params: {p: 0.3}
    - name: grid_mask
      params: {p: 0.2}
    - name: random_erasing
      params: {p: 0.3}
```

管线支持 24 种增强方法，包括 `copy_paste`、`mosaic`、`photometric_distortion`、`grid_mask` 等高级方法（见 `medseg/datasets/advanced_aug.py`）。

### 3.4 Albumentations 集成

也可使用 Albumentations 库：

```yaml
training:
  augmentation: albumentations
  aug_params:
    p_flip: 0.5
    p_rotate: 0.3
    p_color: 0.3
    p_elastic: 0.2
    p_gridmask: 0.1
```

---

## 4. UltimateMedSeg 实操

### 4.1 搭建自定义数据集

**第一步**：组织数据

```
data/MyData/
├── images/
│   ├── patient_001.png
│   ├── patient_002.png
│   └── ...
└── masks/
    ├── patient_001_mask.png    # 二值: 0 (背景) / 255 (目标)
    ├── patient_002_mask.png
    └── ...
```

**第二步**：创建 YAML 配置

```yaml
model:
  num_classes: 2
  img_size: 256
  encoder:
    name: basic
    in_channels: 3
  decoder:
    name: bilinear
  bottleneck:
    name: none
  skip_connection:
    name: concat

data:
  type: generic                # 图像/mask 对使用 GenericDataset
  img_size: 256
  root_dir: ./data/MyData
  img_suffix: .png             # 图像文件扩展名
  mask_suffix: _mask.png       # Mask 文件后缀（追加到图像名后）
  train_ratio: 0.7             # 70% 用于训练
  val_ratio: 0.15              # 15% 用于验证
  random_state: 42             # 可复现切分

training:
  epochs: 200
  batch_size: 16
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
    lr: 0.0001
    weight_decay: 0.0001
  scheduler:
    name: cosine
    min_lr: 0.000001
```

**第三步**：训练

```bash
python train.py --config my_config.yaml
```

### 4.2 K 折交叉验证

对于小数据集，强烈推荐 K 折：

```yaml
data:
  type: generic
  root_dir: ./data/BUSI
  img_suffix: .png
  mask_suffix: _mask.png
  n_splits: 5              # 5 折交叉验证
  fold_idx: 0              # 当前折（0 到 4）
  random_state: 42         # 固定种子保证可复现分折
```

运行所有折：

```bash
for fold in 0 1 2 3 4; do
    python train.py --config my_config.yaml \
        --override data.fold_idx=$fold \
        --output_dir ./output/fold_$fold
done
```

### 4.3 使用内置数据集

示例：Synapse 多器官（8 个器官 + 背景）：

```yaml
data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  test_dir: ./data/Synapse/test_vol_h5
  train_list: ./data/Synapse/lists/lists_Synapse/train.txt
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt
```

预置数据集配置在 `configs/intro_to_datasets/`：

```bash
# BUSI（乳腺超声）
python train.py --config configs/intro_to_datasets/busi.yaml

# Kvasir-SEG（息肉）
python train.py --config configs/intro_to_datasets/kvasir_seg.yaml

# ISIC 2018（皮肤病灶）
python train.py --config configs/intro_to_datasets/isic2018.yaml
```

### 4.4 添加增强

```yaml
training:
  augmentation: pipeline
  aug_pipeline:
    - name: horizontal_flip
      params: {p: 0.5}
    - name: vertical_flip
      params: {p: 0.3}
    - name: random_rotate90
      params: {p: 0.3}
    - name: copy_paste
      params: {p: 0.3}
    - name: mosaic
      params: {p: 0.2}
```

### 4.5 验证变换

验证集始终只使用 `Resize` + `Normalize`（无随机增强）：

```python
# 框架自动应用
val_transform = Compose([
    Resize(img_size),
    Normalize(),
])
```

---

## 5. 推荐实验

### 实验 1：增强效果对比

在相同数据集上对比不同增强级别：

| 配置 | 增强 | 预期 Dice 变化 |
|------|------|---------------|
| A | `augmentation: none` | 基线（可能过拟合） |
| B | 内置 `standard` | +2-5% Dice |
| C | `pipeline` + copy-paste + mosaic | +3-8% Dice |

### 实验 2：切分策略对比

在 BUSI（647 张图像）上对比：

| 策略 | 配置 | 优点 |
|------|------|------|
| 比例 7:1.5:1.5 | `train_ratio: 0.7` | 简单 |
| 5 折交叉验证 | `n_splits: 5` | 更可靠的估计 |

### 实验 3：图像尺寸敏感性

测试输入分辨率的影响：

| `img_size` | 速度 | 预期质量 |
|-----------|------|---------|
| 128 | 快 | 较低（丢失精细细节） |
| 256 | 中等 | 良好平衡 |
| 512 | 慢 | 较高（更多细节） |

---

## 6. 延伸阅读

### 关键概念

- **医学影像中的类别不平衡**：Dice 损失直接解决此问题（见[第 04 讲](04_training_CN.md)）
- **弹性形变**：模拟自然组织形变，对器官分割尤为重要
- **Copy-paste 增强**：将目标区域从一张图像粘贴到另一张，有效翻倍训练数据

### 相关文档

- [数据指南](../data/README.md) -- 25 个内置数据集及下载链接
- [数据集配置](../../configs/intro_to_datasets/) -- 每个数据集的现成 YAML
- [增强源码](../../medseg/utils/augmentation.py) -- Pipeline 实现
- [变换源码](../../medseg/datasets/transforms.py) -- 内置变换

---

[上一讲：U-Net](02_unet_CN.md) | [下一讲：训练与评估](04_training_CN.md)
