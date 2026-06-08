# Chapter 03: Data and Preprocessing

[Previous: U-Net](02_unet.md) | [中文文档](03_data_CN.md) | [Next: Training and Evaluation](04_training.md)

---

## 1. Background and Motivation

Data quality and preprocessing directly determine segmentation performance. In medical imaging, the challenges are amplified:

- **Diverse formats**: NIfTI (`.nii.gz`), DICOM, PNG/JPG, NumPy (`.npz`), HDF5 (`.h5`)
- **Inconsistent labeling**: Different annotation protocols across hospitals/institutions
- **Small datasets**: Many medical datasets have only hundreds of samples
- **Class imbalance**: Target structures (tumors, lesions) are often tiny compared to background

UltimateMedSeg provides a unified data pipeline that handles 25 built-in datasets with 4 split strategies and 24 augmentation methods, all configurable through YAML.

---

## 2. Core Concepts

### 2.1 Medical Image Data Formats

| Format | Extension | Typical Use | Python Library |
|--------|-----------|-------------|----------------|
| NIfTI | `.nii.gz` | CT / MRI volumes | `nibabel`, `SimpleITK` |
| DICOM | `.dcm` | Raw clinical data | `pydicom`, `SimpleITK` |
| PNG/JPG | `.png/.jpg` | 2D images (dermoscopy, fundus, pathology) | `PIL`, `cv2` |
| NumPy | `.npz` | Preprocessed arrays (Synapse, ACDC) | `numpy` |
| HDF5 | `.h5` | Preprocessed test volumes | `h5py` |

### 2.2 Directory Conventions

UltimateMedSeg supports two directory layouts:

**Layout A: Explicit train/val/test directories**

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

Each image must have a corresponding mask with the **same filename** (or configurable suffix, e.g., `image_001_mask.png`).

**Layout B: Flat directory with auto-split**

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

Images and masks are automatically split into train/val/test according to configurable ratios or K-fold.

### 2.3 Four Split Strategies

| Strategy | Config Field | When to Use |
|----------|-------------|-------------|
| **Explicit Paths** | `train_dir`, `val_dir`, `test_dir` | Pre-split datasets (Synapse, ACDC) |
| **Ratio-Based** | `train_ratio: 0.7`, `val_ratio: 0.15` | Auto-split from flat directory |
| **K-Fold** | `n_splits: 5`, `fold_idx: 0` | Cross-validation (recommended for small datasets) |
| **Predefined List** | `train_list: train.txt`, `test_list: test.txt` | Follow paper's official split |

---

## 3. Method Details

### 3.1 Basic Transforms (Built-in)

The framework includes essential transforms in `medseg/datasets/transforms.py`:

| Transform | Description | Parameters |
|-----------|-------------|------------|
| `Resize` | Resize image and mask to target size | `size` |
| `Normalize` | Min-max or mean/std normalization | `mean`, `std` (optional) |
| `RandomFlip` | Horizontal and vertical flip | `p=0.5` |
| `RandomRotate90` | 90/180/270 degree rotation | `p=0.5` |
| `RandomRotation` | Arbitrary angle rotation | `degrees=15, p=0.5` |
| `RandomScale` | Zoom in/out then resize | `scale_range=(0.8,1.2), p=0.5` |
| `RandomCrop` | Random crop to target size | `size` |
| `RandomElasticDeform` | Elastic deformation | `alpha=50, sigma=5, p=0.3` |
| `GaussianNoise` | Add Gaussian noise to image | `mean=0, std=0.05, p=0.3` |
| `GaussianBlur` | Gaussian blur on image | `kernel_size=3, p=0.2` |
| `BrightnessContrastJitter` | Brightness and contrast jitter | `brightness=0.2, contrast=0.2, p=0.3` |
| `GammaCorrection` | Random gamma correction | `gamma_range=(0.7,1.5), p=0.3` |
| `CutOut` | Random rectangular masking | `num_holes=1, max_h_size=32, p=0.3` |

**Important**: All transforms apply the **same geometric operation** to both image and mask (flip, rotate, scale), but **only affect the image** for intensity operations (noise, blur, brightness).

### 3.2 Three Augmentation Levels

The built-in `get_train_transforms()` provides three preset levels:

```python
from medseg.datasets.transforms import get_train_transforms

# Light: only flip
transforms_light = get_train_transforms(img_size=224, augment_level='light')

# Standard: flip + rotate + scale + noise + brightness (default)
transforms_std = get_train_transforms(img_size=224, augment_level='standard')

# Heavy: all 10 augmentations including elastic deformation
transforms_heavy = get_train_transforms(img_size=224, augment_level='heavy')
```

| Level | Augmentations Included |
|-------|----------------------|
| `light` | RandomFlip |
| `standard` | Flip + Rotate90 + Rotation + Scale + Noise + Brightness |
| `heavy` | All above + ElasticDeform + Blur + GammaCorrection + CutOut |

### 3.3 YAML-Configurable Pipeline (Recommended)

For fine-grained control, use the pipeline mode in YAML:

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

The pipeline supports 24 augmentation methods including advanced ones like `copy_paste`, `mosaic`, `photometric_distortion`, and `grid_mask` (see `medseg/datasets/advanced_aug.py`).

### 3.4 Albumentations Integration

Alternatively, use the Albumentations library:

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

## 4. Hands-On with UltimateMedSeg

### 4.1 Setting Up a Custom Dataset

**Step 1**: Organize your data

```
data/MyData/
├── images/
│   ├── patient_001.png
│   ├── patient_002.png
│   └── ...
└── masks/
    ├── patient_001_mask.png    # Binary: 0 (background) / 255 (target)
    ├── patient_002_mask.png
    └── ...
```

**Step 2**: Create a YAML config

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
  type: generic                # Use GenericDataset for image/mask pairs
  img_size: 256
  root_dir: ./data/MyData
  img_suffix: .png             # Image file extension
  mask_suffix: _mask.png       # Mask file suffix (appended to image stem)
  train_ratio: 0.7             # 70% for training
  val_ratio: 0.15              # 15% for validation
  random_state: 42             # Reproducible split

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

**Step 3**: Train

```bash
python train.py --config my_config.yaml
```

### 4.2 K-Fold Cross-Validation

For small datasets, K-fold is strongly recommended:

```yaml
data:
  type: generic
  root_dir: ./data/BUSI
  img_suffix: .png
  mask_suffix: _mask.png
  n_splits: 5              # 5-fold CV
  fold_idx: 0              # Current fold (0 to 4)
  random_state: 42         # Fixed seed for reproducible folds
```

Run all folds:

```bash
for fold in 0 1 2 3 4; do
    python train.py --config my_config.yaml \
        --override data.fold_idx=$fold \
        --output_dir ./output/fold_$fold
done
```

### 4.3 Using a Built-in Dataset

Example: Synapse multi-organ (8 organs + background):

```yaml
data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  test_dir: ./data/Synapse/test_vol_h5
  train_list: ./data/Synapse/lists/lists_Synapse/train.txt
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt
```

Pre-built dataset configs are in `configs/intro_to_datasets/`:

```bash
# BUSI (breast ultrasound)
python train.py --config configs/intro_to_datasets/busi.yaml

# Kvasir-SEG (polyp)
python train.py --config configs/intro_to_datasets/kvasir_seg.yaml

# ISIC 2018 (skin lesion)
python train.py --config configs/intro_to_datasets/isic2018.yaml
```

### 4.4 Adding Augmentation

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

### 4.5 Validation Transform

Validation always uses only `Resize` + `Normalize` (no random augmentations):

```python
# Automatically applied by the framework
val_transform = Compose([
    Resize(img_size),
    Normalize(),
])
```

---

## 5. Recommended Experiments

### Experiment 1: Augmentation Impact

Compare training with different augmentation levels on the same dataset:

| Config | Augmentation | Expected Dice Change |
|--------|-------------|---------------------|
| A | `augmentation: none` | Baseline (may overfit) |
| B | Built-in `standard` | +2-5% Dice |
| C | `pipeline` with copy-paste + mosaic | +3-8% Dice |

### Experiment 2: Split Strategy Comparison

On BUSI (647 images), compare:

| Strategy | Config | Pros |
|----------|--------|------|
| Ratio 7:1.5:1.5 | `train_ratio: 0.7` | Simple |
| 5-fold CV | `n_splits: 5` | More reliable estimates |

### Experiment 3: Image Size Sensitivity

Test the effect of input resolution:

| `img_size` | Speed | Expected Quality |
|-----------|-------|-----------------|
| 128 | Fast | Lower (loses fine detail) |
| 256 | Medium | Good balance |
| 512 | Slow | Higher (more detail) |

---

## 6. Further Reading

### Key Concepts

- **Class imbalance in medical imaging**: Dice loss directly addresses this (see [Chapter 04](04_training.md))
- **Elastic deformation**: Simulates natural tissue deformation, especially important for organ segmentation
- **Copy-paste augmentation**: Paste target regions from one image onto another, effectively doubling training data

### Related Documentation

- [Data Guide](../data/README.md) -- All 25 built-in datasets with download links
- [Dataset Configs](../../configs/intro_to_datasets/) -- Ready-to-use YAML for each dataset
- [Augmentation Source](../../medseg/utils/augmentation.py) -- Pipeline implementation
- [Transform Source](../../medseg/datasets/transforms.py) -- Built-in transforms

---

[Previous: U-Net](02_unet.md) | [Next: Training and Evaluation](04_training.md)
