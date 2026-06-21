# 弱监督分割

[English](weakly_supervised.md)

本框架内置 **20** 种弱监督方法，位于 `medseg/training/weakly_supervised/`。

## 方法列表

### 核心方法 (10)

| 方法 | 论文 | 发表 | 说明 | YAML |
|------|------|------|------|------|
| `box_supervised` | BoxSup family | - | 框监督：框生成掩码 + 前景/背景 CE | [box_supervised.yaml](../../configs/training_paradigms/weak_supervision/box_supervised.yaml) |
| `cam` | Zhou et al. / Selvaraju et al. | CVPR 2016 / ICCV 2017 | 类激活映射 (Grad-CAM) | [cam.yaml](../../configs/training_paradigms/weak_supervision/cam.yaml) |
| `mil` | 多实例学习 | - | 图像级多实例学习 | - |
| `point` | Bearman et al. | ECCV 2016 | 点监督 | [point.yaml](../../configs/training_paradigms/weak_supervision/point.yaml) |
| `tree_energy` | 树能量 | - | 树结构能量最小化 | [tree_energy.yaml](../../configs/training_paradigms/weak_supervision/tree_energy.yaml) |
| `seam` | Wang et al. | CVPR 2020 | 自监督等变注意力 | - |
| `puzzle_cam` | Jo & Yu | ICIP 2021 | 拼图匹配 CAM | - |
| `advcam` | Lee et al. | CVPR 2021 | 对抗互补擦除 | - |
| `mctformer` | Xu et al. | CVPR 2022 | 多类 token transformer | - |
| `scribble_sup` | Lin et al. | CVPR 2016 | 涂鸦标注监督（轻量变体，内联可微 pairwise CRF 替代） | [scribble_sup.yaml](../../configs/training_paradigms/weak_supervision/scribble_sup.yaml) |

### 扩展方法 (10)

| 方法 | 论文 | 发表 | GitHub | 说明 | YAML |
|------|------|------|--------|------|------|
| `eps` | EPS | - | - | 显式伪标签监督 | [eps.yaml](../../configs/training_paradigms/weak_supervision/eps.yaml) |
| `boxinst` | BoxInst | - | - | 框级实例分割 | [boxinst.yaml](../../configs/training_paradigms/weak_supervision/boxinst.yaml) |
| `recam` | ReCAM | - | - | 重加权 CAM | [recam.yaml](../../configs/training_paradigms/weak_supervision/recam.yaml) |
| `toco` | ToCo | - | - | Token 对比 | [toco.yaml](../../configs/training_paradigms/weak_supervision/toco.yaml) |
| `lpcam` | LPCAM | - | - | 低通滤波 CAM | [lpcam.yaml](../../configs/training_paradigms/weak_supervision/lpcam.yaml) |
| `mars` | MARS | - | - | 掩码感知精炼 | [mars.yaml](../../configs/training_paradigms/weak_supervision/mars.yaml) |
| `dupl` | DuPL | - | - | 双伪标签 | [dupl.yaml](../../configs/training_paradigms/weak_supervision/dupl.yaml) |
| `more` | MoRe | - | - | 动量精炼 | [more.yaml](../../configs/training_paradigms/weak_supervision/more.yaml) |
| `psdpm` | PSDPM | - | - | 先验伪标签去噪 | [psdpm.yaml](../../configs/training_paradigms/weak_supervision/psdpm.yaml) |
| `semples` | SemPLeS | - | - | 语义伪标签选择 | [semples.yaml](../../configs/training_paradigms/weak_supervision/semples.yaml) |

## 标注格式

弱监督方法使用不同类型的标注，均通过 JSON 文件加载。数据集类 (`WeaklySupervisedDataset`) 支持四种监督类型：

### 图像级标签 (`image_label`)

仅需图像级类别存在标签，无需空间标注。

```json
[
  {"image": "img_0001.png", "image_labels": [0, 2]},
  {"image": "img_0002.png", "image_labels": {"0": true, "1": false, "2": true}}
]
```

**使用方法：** `cam`、`mil`、`seam`、`puzzle_cam`、`advcam`、`mctformer`、`recam`、`lpcam`、`toco`、`mars`、`dupl`、`more`、`psdpm`、`semples`、`eps`

### 边界框 (`box`)

每张图像可有多个边界框。支持**可变长度**（无填充或截断）。支持两种标注格式：

**格式 1：简单格式（所有框共享同一类别）**

```json
[
  {
    "image": "img_0001.png",
    "boxes": [[0.1, 0.2, 0.8, 0.9], [0.3, 0.4, 0.6, 0.7]]
  }
]
```

**格式 2：逐框类别（实例级标注）**

```json
[
  {
    "image": "img_0001.png",
    "boxes": [
      {"box": [0.1, 0.2, 0.8, 0.9], "class": 1},
      {"box": [0.3, 0.4, 0.6, 0.7], "class": 3}
    ]
  }
]
```

所有坐标归一化到 `[0, 1]` 范围，格式为 `[x1, y1, x2, y2]`。数据集自动缩放到图像尺寸。

**实例到语义转换：** 逐框类别自动转换为图像级多标签。例如，图像有框类别 `[1, 3, 1]`，则 `image_labels` 为 `[0, 1, 0, 1, 0, 0, 0, 0, 0]`（类别 1 和 3 存在）。

**使用方法：** `box_supervised`、`boxinst`

### 点 (`point`)

每张图像可有多个点击点。支持**可变长度**。支持两种标注格式：

**格式 1：简单元组含类别 ID**

```json
[
  {
    "image": "img_0001.png",
    "points": [[0.5, 0.3, 1], [0.2, 0.7, 0], [0.8, 0.6, 2]]
  }
]
```

**格式 2：逐点类别（实例级标注）**

```json
[
  {
    "image": "img_0001.png",
    "points": [
      {"point": [0.5, 0.3], "class": 1},
      {"point": [0.2, 0.7], "class": 0},
      {"point": [0.8, 0.6], "class": 2}
    ]
  }
]
```

坐标为归一化 `[x, y]`，范围 `[0, 1]`。数据集自动缩放到图像尺寸。

**实例到语义转换：** 逐点类别自动转换为图像级多标签，与框转换相同。

**使用方法：** `point`

### 涂鸦 (`scribble`)

每张图像可有多个涂鸦。支持**可变长度**。支持两种标注格式：

**格式 1：简单格式（所有涂鸦共享同一类别）**

```json
[
  {
    "image": "img_0001.png",
    "scribbles": [[0.1, 0.2], [0.15, 0.25], [0.2, 0.3], [0.5, 0.5]]
  }
]
```

**格式 2：逐涂鸦类别（实例级标注）**

```json
[
  {
    "image": "img_0001.png",
    "scribbles": [
      {"scribble": [[0.1, 0.2], [0.15, 0.25]], "class": 1},
      {"scribble": [[0.5, 0.5], [0.55, 0.6]], "class": 3}
    ]
  }
]
```

坐标为归一化 `[x, y]`，范围 `[0, 1]`。每个涂鸦是形成一笔划的坐标对列表。

**实例到语义转换：** 逐涂鸦类别自动转换为图像级多标签，与框/点转换相同。

**使用方法：** `scribble_sup`

### 可变长度支持

所有空间标注类型（框、点、涂鸦）支持**每张图像可变数量的实例**。`train_weakly_supervised.py` 中的自定义 collate 函数 `weak_supervision_collate()` 对这些字段返回张量列表而非堆叠张量，避免填充或截断。

```python
# 在 train_weakly_supervised.py 中
def weak_supervision_collate(batch):
    # boxes, box_classes, points, point_classes, scribbles 保持为列表
    # 其他字段正常堆叠
```

### 预计算 CAM（可选）

对于基于 CAM 的方法，预计算的类激活映射可存储为 `.npy` 文件，放在单独目录中。每个 `.npy` 文件形状为 `[num_classes, H, W]`，文件名应与图像匹配（如 `img_0001.npy` 对应 `img_0001.png`）。

```yaml
data:
  cam_dir: ./data/cams   # 存放 .npy CAM 文件的目录
```

### 监督类型汇总

| 类型 | 标注方式 | 数据集类 | 方法 |
|------|----------|----------|------|
| 图像级 | 类别标签 | `ImageLabelDataset` | cam、mil、seam、puzzle_cam、advcam、mctformer + 12 个扩展方法 |
| 框 | 边界框（可变长度，逐框类别） | `BoxSupervisedDataset` | box_supervised、boxinst |
| 点 | 点击点（可变长度，逐点类别） | `WeaklySupervisedDataset(point)` | point |
| 涂鸦 | 涂鸦线（可变长度，逐涂鸦类别） | `WeaklySupervisedDataset(scribble)` | scribble_sup |
| 混合 | 多种类型 | — | eps |

### 需要标注文件的方法

以下方法需要在 YAML 中配置特定的标注文件：

| 方法 | 标注文件 | YAML 键 |
|------|----------|---------|
| `box_supervised` | boxes.json | `data.annotation_file` |
| `boxinst` | boxes.json | `data.annotation_file` |
| `point` | points.json | `data.annotation_file` |
| `scribble_sup` | scribbles.json | `data.annotation_file` |
| `tree_energy` | sparse_labels.json | `data.label_file` |

## 配置示例

```yaml
model:
  num_classes: 9
  img_size: 224
  encoder:
    name: timm_resnet50
    pretrained: true
    in_channels: 3
  decoder:
    name: unet
  bottleneck:
    name: none

data:
  img_size: 224
  image_dir: ./data/images
  label_file: ./data/annotations/image_labels.json   # 图像级标签
  cam_dir: ./data/cams                                # 预计算 CAM（可选）
  val:
    image_dir: ./data/val/images
    mask_dir: ./data/val/masks
  test:
    image_dir: ./data/test/images
    mask_dir: ./data/test/masks

weak_supervision:
  method: seam           # 上表任意方法名
  params:
    scale_factor: 0.3
    ecr_top_k_ratio: 0.2

training:
  epochs: 150
  batch_size: 16
  num_workers: 4
  val_interval: 10
  save_interval: 30
  loss:
    name: seam_loss
    params:
      scale_factor: 0.3
      ecr_top_k_ratio: 0.2
  optimizer:
    name: adamw
    lr: 2e-4
    weight_decay: 1e-4
  scheduler:
    name: cosine
    min_lr: 1e-6
```

## 用法

```bash
# 训练
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/cam.yaml

# 框监督
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml

# 点监督
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/point.yaml

# 涂鸦监督
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/scribble_sup.yaml

# 测试
python test.py --config configs/training_paradigms/weak_supervision/cam.yaml \
    --checkpoint output/best_model.pth
```

每个方法的配置位于 `configs/training_paradigms/weak_supervision/`。
