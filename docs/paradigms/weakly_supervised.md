# Weakly Supervised Segmentation

[中文文档](weakly_supervised_CN.md)

20 built-in weakly supervised methods in `medseg/training/weakly_supervised/`.

## Methods

### Core Methods (10)

| Method | Paper | Published | Description | YAML |
|--------|-------|-------|-------------|------|
| `box_supervised` | BoxSup family | - | Box-only mask + FG/BG CE | [box_supervised.yaml](../../configs/training_paradigms/weak_supervision/box_supervised.yaml) |
| `cam` | Zhou et al. / Selvaraju et al. | CVPR 2016 / ICCV 2017 | Class Activation Mapping (Grad-CAM) | [cam.yaml](../../configs/training_paradigms/weak_supervision/cam.yaml) |
| `mil` | Multi-instance learning | - | Image-level label MIL | - |
| `point` | Bearman et al. | ECCV 2016 | Point supervision | [point.yaml](../../configs/training_paradigms/weak_supervision/point.yaml) |
| `tree_energy` | Tree energy | - | Tree-structured energy minimization | [tree_energy.yaml](../../configs/training_paradigms/weak_supervision/tree_energy.yaml) |
| `seam` | Wang et al. | CVPR 2020 | Self-supervised equivariant attention | - |
| `puzzle_cam` | Jo & Yu | ICIP 2021 | Puzzle piece matching CAM | - |
| `advcam` | Lee et al. | CVPR 2021 | Adversarial complementary erasing | - |
| `mctformer` | Xu et al. | CVPR 2022 | Multi-class token transformer | - |
| `scribble_sup` | Lin et al. | CVPR 2016 | Scribble annotation supervision (light variant with inlined pairwise CRF surrogate) | [scribble_sup.yaml](../../configs/training_paradigms/weak_supervision/scribble_sup.yaml) |

### Extended Methods (10)

| Method | Paper | Published | GitHub | Description | YAML |
|--------|-------|-------|--------|-------------|------|
| `eps` | EPS | - | - | Explicit pseudo-label supervision | [eps.yaml](../../configs/training_paradigms/weak_supervision/eps.yaml) |
| `boxinst` | BoxInst | - | - | Box-level instance segmentation | [boxinst.yaml](../../configs/training_paradigms/weak_supervision/boxinst.yaml) |
| `recam` | ReCAM | - | - | Re-weighted CAM | [recam.yaml](../../configs/training_paradigms/weak_supervision/recam.yaml) |
| `toco` | ToCo | - | - | Token contrast | [toco.yaml](../../configs/training_paradigms/weak_supervision/toco.yaml) |
| `lpcam` | LPCAM | - | - | Low-pass filtered CAM | [lpcam.yaml](../../configs/training_paradigms/weak_supervision/lpcam.yaml) |
| `mars` | MARS | - | - | Mask-aware refinement | [mars.yaml](../../configs/training_paradigms/weak_supervision/mars.yaml) |
| `dupl` | DuPL | - | - | Dual pseudo label | [dupl.yaml](../../configs/training_paradigms/weak_supervision/dupl.yaml) |
| `more` | MoRe | - | - | Momentum refinement | [more.yaml](../../configs/training_paradigms/weak_supervision/more.yaml) |
| `psdpm` | PSDPM | - | - | Pseudo-label denoising with prior | [psdpm.yaml](../../configs/training_paradigms/weak_supervision/psdpm.yaml) |
| `semples` | SemPLeS | - | - | Semantic pseudo label selection | [semples.yaml](../../configs/training_paradigms/weak_supervision/semples.yaml) |

## Annotation Formats

Weakly supervised methods use different types of annotations, all loaded via a JSON file. The dataset class (`WeaklySupervisedDataset`) supports four supervision types:

### Image-Level Labels (`image_label`)

Only the image-level class presence is required — no spatial annotation.

```json
[
  {"image": "img_0001.png", "image_labels": [0, 2]},
  {"image": "img_0002.png", "image_labels": {"0": true, "1": false, "2": true}}
]
```

**Used by:** `cam`, `mil`, `seam`, `puzzle_cam`, `advcam`, `mctformer`, `recam`, `lpcam`, `toco`, `mars`, `dupl`, `more`, `psdpm`, `semples`, `eps`

### Bounding Box (`box`)

Each image can have multiple boxes. Boxes support **variable-length** per image (no padding or truncation). Two annotation formats are supported:

**Format 1: Simple (all boxes share same class)**

```json
[
  {
    "image": "img_0001.png",
    "boxes": [[0.1, 0.2, 0.8, 0.9], [0.3, 0.4, 0.6, 0.7]]
  }
]
```

**Format 2: Per-box class (instance-level annotations)**

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

All coordinates are normalised to `[0, 1]` range as `[x1, y1, x2, y2]`. The dataset automatically scales them to image size.

**Instance-to-semantic conversion:** Per-box classes are automatically converted to image-level multi-labels. For example, if an image has boxes with classes `[1, 3, 1]`, the resulting `image_labels` will be `[0, 1, 0, 1, 0, 0, 0, 0, 0]` (classes 1 and 3 present).

**Used by:** `box_supervised`, `boxinst`

### Point (`point`)

Each image can have multiple click points. Points support **variable-length** per image. Two annotation formats are supported:

**Format 1: Simple tuple with class ID**

```json
[
  {
    "image": "img_0001.png",
    "points": [[0.5, 0.3, 1], [0.2, 0.7, 0], [0.8, 0.6, 2]]
  }
]
```

**Format 2: Per-point class (instance-level annotations)**

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

Coordinates are normalised `[x, y]` in `[0, 1]` range. The dataset automatically scales them to image size.

**Instance-to-semantic conversion:** Per-point classes are automatically converted to image-level multi-labels, same as box conversion.

**Used by:** `point`

### Scribble (`scribble`)

Each image can have multiple scribbles. Scribbles support **variable-length** per image. Two annotation formats are supported:

**Format 1: Simple (all scribbles share same class)**

```json
[
  {
    "image": "img_0001.png",
    "scribbles": [[0.1, 0.2], [0.15, 0.25], [0.2, 0.3], [0.5, 0.5]]
  }
]
```

**Format 2: Per-scribble class (instance-level annotations)**

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

Coordinates are normalised `[x, y]` in `[0, 1]` range. Each scribble is a list of coordinate pairs forming a stroke.

**Instance-to-semantic conversion:** Per-scribble classes are automatically converted to image-level multi-labels, same as box/point conversion.

**Used by:** `scribble_sup`

### Variable-Length Support

All spatial annotation types (boxes, points, scribbles) support **variable number of instances per image**. The custom collate function `weak_supervision_collate()` in `train_weakly_supervised.py` returns lists of tensors instead of stacked tensors for these fields, avoiding padding or truncation.

```python
# In train_weakly_supervised.py
def weak_supervision_collate(batch):
    # boxes, box_classes, points, point_classes, scribbles are kept as lists
    # Everything else is stacked normally
```

### Pre-computed CAMs (optional)

For CAM-based methods, pre-computed Class Activation Maps can be stored as `.npy` files in a separate directory. Each `.npy` file has shape `[num_classes, H, W]` and should match the image filename (e.g. `img_0001.npy` for `img_0001.png`).

```yaml
data:
  cam_dir: ./data/cams   # directory with .npy CAM files
```

### Supervision Type Summary

| Type | Annotation | Dataset Class | Methods |
|------|-----------|---------------|---------|
| Image-level | Class label | `ImageLabelDataset` | cam, mil, seam, puzzle_cam, advcam, mctformer, + 12 extended |
| Box | Bounding box (variable-length, per-box class) | `BoxSupervisedDataset` | box_supervised, boxinst |
| Point | Click points (variable-length, per-point class) | `WeaklySupervisedDataset(point)` | point |
| Scribble | Scribble lines (variable-length, per-scribble class) | `WeaklySupervisedDataset(scribble)` | scribble_sup |
| Mixed | Multiple types | — | eps |

### Methods Requiring Annotation Files

The following methods require specific annotation files to be configured in the YAML:

| Method | Annotation File | YAML Key |
|--------|----------------|----------|
| `box_supervised` | boxes.json | `data.annotation_file` |
| `boxinst` | boxes.json | `data.annotation_file` |
| `point` | points.json | `data.annotation_file` |
| `scribble_sup` | scribbles.json | `data.annotation_file` |
| `tree_energy` | sparse_labels.json | `data.label_file` |

## YAML Config

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
  label_file: ./data/annotations/image_labels.json   # image-level labels
  cam_dir: ./data/cams                                # pre-computed CAMs (optional)
  val:
    image_dir: ./data/val/images
    mask_dir: ./data/val/masks
  test:
    image_dir: ./data/test/images
    mask_dir: ./data/test/masks

weak_supervision:
  method: seam           # any method name from tables above
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

## Usage

```bash
# Train
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/cam.yaml

# Box supervision
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml

# Point supervision
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/point.yaml

# Scribble supervision
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/scribble_sup.yaml

# Test
python test.py --config configs/training_paradigms/weak_supervision/cam.yaml \
    --checkpoint output/best_model.pth
```

Each method config is in `configs/training_paradigms/weak_supervision/`.
