# Chapter 07: Foundation Models

[Previous: Decoders](06_decoders.md) | [中文文档](07_foundation_CN.md) | [Next: Advanced Paradigms](08_paradigms.md)

---

## 1. Background and Motivation

Training a deep segmentation model from scratch requires large annotated datasets — pixel-level masks that are expensive and time-consuming to create, especially in medical imaging where expert radiologists must label each image.

**Foundation models** offer a radical alternative: pre-train a large Vision Transformer on millions of images *without human annotations*, then transfer the learned features to downstream medical tasks with minimal labeled data.

Three theoretical foundations explain why this works:

- **Transfer learning theory**: Features learned on a broad data distribution generalize to narrow downstream tasks.
- **Scaling laws**: Model quality improves predictably with data volume, model size, and compute.
- **Self-supervised learning**: Cleverly designed pretext tasks force the model to learn semantically meaningful representations without labels.

This chapter explains the theoretical mechanisms behind foundation models, how they are adapted for medical segmentation, and how fine-tuning strategies control the transfer learning trade-off.

---

## 2. Core Concepts

### 2.1 Self-Supervised Learning Paradigms

Self-supervised learning (SSL) creates supervision signals from the data itself — no human labels required. The model must predict a *hidden part* of its input, forcing it to understand the underlying structure.

**Contrastive learning** (DINO, CLIP): Given two augmented views of the same image, the model learns to pull their representations together while pushing apart representations of different images:

$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k) / \tau)}$$

where $\tau$ is the temperature parameter controlling the sharpness of the similarity distribution, and $\text{sim}(\cdot,\cdot)$ is typically cosine similarity.

| Method | Views | Key Innovation |
|--------|-------|----------------|
| DINO | Multi-crop of same image | Student-teacher self-distillation, no negative pairs |
| CLIP | Image + text | Cross-modal alignment (vision ↔ language) |
| DINOv2 | Multi-crop of same image | Scaling to 142M images, improved features |

**Masked image modeling** (MAE, DINOv2): Randomly mask 50-75% of image patches, then reconstruct the masked content:

$$\mathcal{L}_{\text{recon}} = \frac{1}{|M|} \sum_{i \in M} \| x_i - f_\theta(x_{\setminus M})_i \|^2$$

where $M$ is the set of masked positions and $f_\theta$ is the encoder-decoder that processes the visible patches $x_{\setminus M}$ to predict the masked ones.

**Why masking works**: To predict a missing patch, the model must understand *what* surrounds it (context) and *what* typically fills that position (semantics). This forces learning of high-level representations far beyond simple texture matching.

### 2.2 Why Transfer Learning Works

The fundamental insight is that **low-level and mid-level visual features are shared across domains**. Edges, textures, shapes, and spatial relationships appear in both natural images and medical scans.

**Feature hierarchy transfer**:

```
Pre-trained on natural images (ImageNet / DINOv2)
  Layer 1-4:   Edges, gradients, textures     ← Directly transferable
  Layer 5-8:   Patterns, local shapes          ← Partially transferable
  Layer 9-12:  Object parts, semantics          ← Domain-specific, needs fine-tuning

Fine-tuned on medical images
  Layer 1-4:   Frozen (reuse generic features)
  Layer 5-8:   Partially adapted
  Layer 9-12:  Fully adapted to medical domain
```

**The domain gap problem**: Natural images and medical images differ in texture distribution, color palette, and object structure. A foundation model pre-trained only on ImageNet may miss medical-specific features (e.g., tissue micro-structure, lesion morphology). This motivates **domain-specific pre-training** — training foundation models on large medical image collections.

### 2.3 DPT Architecture — Multi-Scale from Monocular ViT

A standard Vision Transformer processes all patches through identical layers, producing a single-scale feature map. This is problematic for segmentation, which requires multi-scale features.

The **Dense Prediction Transformer (DPT)** solves this by extracting features from *intermediate blocks* of the ViT and fusing them through specialized layers:

```
Pre-trained ViT (12 or 24 blocks)
  │
  ├─ block_3  ──→ Reassemble ──→ Fusion ──→ Stage 1 (1/4 res)
  ├─ block_7  ──→ Reassemble ──→ Fusion ──→ Stage 2 (1/8 res)
  ├─ block_11 ──→ Reassemble ──→ Fusion ──→ Stage 3 (1/16 res)
  └─ block_15 ──→ Reassemble ──→ Fusion ──→ Stage 4 (1/32 res)
```

**Reassemble layers** reshape 1D token sequences into 2D feature maps at the target resolution. They handle the dimension mismatch between ViT's constant-width tokens and the varying spatial sizes needed for multi-scale features.

**Fusion layers** aggregate information across scales using residual convolutions, similar to FPN but operating on features from a single network depth rather than a hierarchical backbone.

**Why not FPN?** Standard FPN assumes a hierarchical encoder (ResNet stages with different spatial sizes). ViT produces uniform-resolution tokens. DPT's reassemble+fusion design specifically addresses this structural mismatch.

### 2.4 Domain-Specific Pre-Training

General foundation models (DINOv2, CLIP) are pre-trained on natural images. Medical-specific models close the domain gap by pre-training on large medical image collections:

| Domain | Models | Pre-training Data | Why It Helps |
|--------|--------|-------------------|--------------|
| General | DINOv2, DINO | ImageNet, web images | Broad visual features |
| Pathology | Phikon, UNI | Histology slides (100K+ WSIs) | Tissue micro-structure |
| Retinal | RETFound | 1.6M retinal images | Fundus/OCT patterns |
| Radiology | Rad-DINO | Chest X-rays, CT scans | Anatomical structure |
| Dermatology | PanDerm | 100K+ skin images | Lesion morphology |
| Ultrasound | UltraFedFM | Ultrasound images | Speckle patterns, echogenicity |

**The transfer learning hierarchy**:

1. **Best match**: Domain-specific model on same modality (e.g., Phikon for pathology)
2. **Good match**: Domain-specific model on similar modality (e.g., Rad-DINO for CT)
3. **General**: DINOv2 or CLIP (works surprisingly well despite domain gap)

### 2.5 Fine-Tuning Theory

After selecting a foundation model, the key decision is **how much of the model to train** on the downstream task. This involves a fundamental trade-off between preserving pre-trained features and adapting to the target domain.

**Full fine-tuning** ($\theta_{\text{all}}$ trainable):

$$\theta^* = \arg\min_\theta \mathcal{L}_{\text{task}}(f_\theta(x), y)$$

Best accuracy when domain gap is large, but risks **catastrophic forgetting** — the model may overwrite useful pre-trained features with task-specific noise, especially with small datasets.

**Frozen encoder** (only decoder $\theta_{\text{dec}}$ trainable):

$$\theta_{\text{dec}}^* = \arg\min_{\theta_{\text{dec}}} \mathcal{L}_{\text{task}}(g_{\theta_{\text{dec}}}(f_{\theta_{\text{frozen}}}(x)), y)$$

Preserves all pre-trained knowledge, but limits adaptation. Works best when the domain gap is small.

**Partial fine-tuning** (last $N$ blocks trainable):

A compromise — early layers (generic features) stay frozen, while later layers (semantic features) adapt:

$$\theta^* = \{\theta_{\text{block}_{L-N+1}}, ..., \theta_{\text{block}_L}, \theta_{\text{dec}}\}$$

**Layer-wise learning rate decay**: Instead of binary freeze/train, apply decreasing learning rates from top to bottom:

$$\eta_l = \eta_{\text{base}} \cdot \gamma^{L-l}$$

where $\gamma \in (0, 1)$ is the decay factor and $l$ is the layer index. Deeper layers (closer to input) receive smaller updates, preserving generic features.

**LoRA** (Low-Rank Adaptation): Instead of updating the full weight matrix $W$, learn a low-rank update:

$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d,k)$$

This reduces trainable parameters by 90%+ while allowing full-model adaptation. Originally developed for LLMs, LoRA is increasingly used for ViT fine-tuning.

---

## 3. Method Details

### 3.1 Medical Modalities — 9 Categories

APRIL-MedSeg covers 38 foundation models across 9 medical modalities:

| Modality | Models | Key Application |
|----------|--------|----------------|
| General | DINOv2, DINO, CLIP, SAM, DINOv3 | Cross-domain transfer |
| Pathology | Phikon, UNI, PLIP, MUSK, Phikon-v2, KEEP | Histology, WSI analysis |
| Radiology | Rad-DINO, OmniRad, BioViL, CheXZero | X-ray, CT, MRI |
| Ophthalmology | RETFound, RETFound-DINOv2, FLAIR, OphMAE | Retinal disease detection |
| Dermatology | DermCLIP, MoNet, PanDerm | Skin lesion segmentation |
| General Medical | BiomedCLIP, MedCLIP, MedSigLIP | General biomedical |
| MLLM Vision | Qwen-VL, MedGemma, LLaVA-Med | Vision-language reasoning |
| Ultrasound | UltraFedFM, US-FMAE | Ultrasound analysis |
| Endoscopy | EndoViT, Endo-FM, Surgical-SAM | GI tract imaging |

### 3.2 Foundation Encoder Output

All foundation encoders produce multi-scale features through the DPT head, compatible with any decoder:

```python
# Foundation encoder output (example: DINOv2-Base, 12 blocks)
features = encoder(x)
# features[0]: (B, 256, H/4, W/4)    ← from block 3
# features[1]: (B, 256, H/8, W/8)    ← from block 7
# features[2]: (B, 256, H/16, W/16)  ← from block 11
# features[3]: (B, 256, H/32, W/32)  ← from block 15 (bottleneck)
```

The DPT head ensures consistent channel dimensions across stages, so the decoder doesn't need to know which foundation model is used.

---

## 4. Hands-On with APRIL-MedSeg

```yaml
# General foundation model (DINOv2)
model:
  encoder: { name: dinov2_base, pretrained: true, freeze: true }
  decoder: { name: unet }

# Pathology-specific (Phikon v2)
model:
  encoder: { name: phikon_v2, pretrained: true, freeze: false }
  decoder: { name: cascade, params: { num_stages: 4 } }

# Partial fine-tuning (last 4 blocks)
model:
  encoder:
    name: dinov2_base
    pretrained: true
    freeze: true
    params: { unfreeze_last_n: 4 }
  decoder: { name: unet }

# Retinal-specific (RETFound)
model:
  encoder: { name: retfound_dinov2, pretrained: true }
  decoder: { name: unet }
```

---

## 5. Recommended Experiments

### Experiment 1: Foundation vs. Standard Encoder

Same decoder (UNet), same dataset, compare encoder types:

| Encoder | Type | Trainable Params | Expected Dice |
|---------|------|-----------------|---------------|
| `timm_resnet50` | Standard CNN | ~25M | Baseline |
| `dinov2_base` (frozen) | Foundation | ~2M (decoder only) | +3-6% |
| `dinov2_base` (full) | Foundation | ~88M | +4-8% |

### Experiment 2: Frozen vs. Fine-Tuned

Same foundation encoder, different fine-tuning strategies:

| Strategy | Trainable Params | Training Speed | Best When |
|----------|-----------------|---------------|-----------|
| `freeze: true` | ~2M | Fast | Domain gap is small |
| `unfreeze_last_n: 4` | ~20M | Medium | Moderate domain gap |
| `freeze: false` | ~88M | Slow | Large domain gap, sufficient data |

### Experiment 3: Domain-Specific vs. General

On a pathology dataset, compare:

| Encoder | Pre-training Domain | Expected Performance |
|---------|-------------------|---------------------|
| `dinov2_base` | General (ImageNet) | Good baseline |
| `phikon_v2` | Pathology (histology) | Best (domain match) |
| `retfound_dinov2` | Retinal | Lower (domain mismatch) |

---

## 6. Further Reading

### Key Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| [DINO](https://arxiv.org/abs/2104.14294) | 2021 | ICCV | Self-distillation without negative pairs |
| [DINOv2](https://arxiv.org/abs/2304.07193) | 2023 | - | Scaling SSL to 142M images |
| [MAE](https://arxiv.org/abs/2111.06377) | 2022 | CVPR | Masked autoencoders are scalable visual learners |
| [CLIP](https://arxiv.org/abs/2103.00020) | 2021 | ICML | Vision-language contrastive pre-training |
| [DPT](https://arxiv.org/abs/2103.13413) | 2021 | ICCV | Dense Prediction Transformer architecture |
| [SAM](https://arxiv.org/abs/2304.02643) | 2023 | - | Segment Anything — promptable segmentation |
| [Phikon](https://arxiv.org/abs/2307.10873) | 2023 | - | Histopathology foundation model |
| [UNI](https://arxiv.org/abs/2308.15474) | 2024 | Nature Medicine | Universal pathology foundation model |
| [RETFound](https://arxiv.org/abs/2301.07786) | 2023 | - | Retinal foundation model from 1.6M images |
| [LoRA](https://arxiv.org/abs/2106.09685) | 2022 | ICLR | Low-rank adaptation of large models |

### Related Documentation

- [Foundation Encoders](../models/encoders.md#foundation-models) -- All 38 foundation encoders across 9 modalities
- [Weight Management](../models/encoders.md#weight-management) -- Auto-download and cache system
- [DPT Head](../models/encoders.md#dpt-head) -- Dense Prediction Transformer architecture details

---

[Previous: Decoders](06_decoders.md) | [Next: Advanced Paradigms](08_paradigms.md)
