# UltimateMedSeg Tutorial

[中文文档](README_CN.md)

A hands-on tutorial series for deep learning medical image segmentation, built around the **UltimateMedSeg** framework. Designed for lab-internal use, balancing theoretical depth with engineering practice.

---

## Learning Roadmap

We recommend reading the tutorials in order. Each chapter builds on the previous one.

```
01 Introduction ──> 02 U-Net ──> 03 Data ──> 04 Training
   (What & Why)    (Architecture)  (Pipeline)   (Optimization)
```

---

## Tutorial Index

| Chapter | Title | Key Topics |
|---------|-------|------------|
| [01](01_introduction.md) | **Introduction to Medical Image Segmentation** | Segmentation concepts, clinical significance, evaluation metrics, method evolution, framework overview |
| [02](02_unet.md) | **U-Net in Detail** | Encoder-decoder architecture, skip connections, U-Net variants, YAML configuration, training commands |
| [03](03_data.md) | **Data and Preprocessing** | Data formats, directory conventions, split strategies, augmentation pipeline, custom datasets |
| [04](04_training.md) | **Training and Evaluation** | Loss functions, optimizers, LR scheduling, AMP/DDP, evaluation workflow, logging |

### Coming Soon

> **Work in Progress** -- The following chapters are planned and will be added in future updates.

| Chapter | Title | Topics | Status |
|---------|-------|--------|--------|
| 05 | Encoder Deep Dive | CNN / Transformer / Mamba / RWKV encoder comparison | Planned |
| 06 | Decoders and Skip Connections | CASCADE / EMCAD / Attention Gate | Planned |
| 07 | Foundation Models | Pre-trained ViT encoders, DPT head, fine-tuning strategies | Planned |
| 08 | Advanced Training Paradigms | Semi-supervised / Domain adaptation / Distillation / Weakly supervised | Planned |
| 09 | Deployment and Inference | ONNX export / TTA / Ensemble inference | Planned |

---

## Installation

```bash
git clone https://github.com/juntaoJianggavin/UltimateMedSeg.git
cd UltimateMedSeg

pip install -r requirements.txt
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.0.0 | Deep learning framework |
| `timm` | >= 0.9.0 | Encoder backbone library |
| `monai` | >= 1.2.0 | Medical image utilities |
| `albumentations` | >= 1.3.0 | Data augmentation |
| `einops` | >= 0.6.0 | Tensor operations |
| `tensorboard` | >= 2.13.0 | Training visualization |

---

## Quick Start

After completing any tutorial chapter, you can run a training with one command:

```bash
python train.py --config configs/architectures/combinations/general/unet_basic.yaml
```

Override any config value from the command line:

```bash
python train.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --override training.epochs=100 training.batch_size=8 model.num_classes=9
```

---

## Related Documentation

| Document | Content |
|----------|---------|
| [Models](../models/README.md) | 169 encoders, 40 decoders, 128 networks |
| [Paradigms](../paradigms/README.md) | 6 training paradigms |
| [Data](../data/README.md) | 25 datasets, augmentation pipeline |
| [Deployment](../deployment/README.md) | ONNX export, TTA, ensemble |
| [Research Guide](../research_guide.md) | Ablation studies, benchmarking |
