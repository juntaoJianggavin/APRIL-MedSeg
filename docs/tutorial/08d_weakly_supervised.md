# Chapter 08d: Weakly Supervised Learning

[Back to Paradigms Overview](08_paradigms.md) | [中文文档](08d_weakly_supervised_CN.md) | [Previous: Knowledge Distillation](08c_distillation.md) | [Next: Text-Guided](08e_text_guided.md)

---

## 1. When Should You Use Weakly Supervised Learning?

Pixel-level annotation is the gold standard for segmentation, but it's not always available. In many clinical settings, you have:

- **Image-level labels**: A radiologist flagged the scan as "contains tumor" but didn't draw the boundary.
- **Bounding boxes**: An annotator drew a rectangle around the organ, but didn't trace the exact contour.
- **Points**: A few clicks marking "this is liver, this is background" — far cheaper than tracing.
- **Scribbles**: Quick freehand strokes over parts of the organ.

**Weakly supervised segmentation** trains from these coarse annotations and produces dense pixel-level predictions. The cost savings are dramatic: a bounding box takes ~5 seconds to draw; a pixel-level mask takes 10–30 minutes.

### Annotation Cost Comparison

| Annotation Type | Time per Image | Cost per 1000 Images | Dice (vs. Full Supervision) |
|----------------|---------------|---------------------|----------------------------|
| Pixel-level mask | 10–30 min | $5,000–$15,000 | 100% (baseline) |
| Bounding box | 5–10 sec | $50–$100 | 75–90% |
| Scribble | 15–30 sec | $150–$300 | 70–85% |
| Points (3–5 per class) | 3–5 sec | $30–$60 | 60–75% |
| Image-level label | 1–2 sec | $10–$30 | 50–70% |

---

## 2. Core Concepts

### 2.1 The Fundamental Challenge

Given only coarse supervision, how does the model learn dense predictions? The key challenge is the **information gap**: a bounding box tells you the organ is *somewhere inside* the box, but not *which exact pixels* belong to it.

The model must bridge this gap using **prior knowledge** (learned from data structure) and **regularization** (smoothness constraints, spatial consistency).

### 2.2 CAM-Based Methods (Class Activation Maps)

The simplest approach: use a **classification network** and extract spatial activation patterns from its last convolutional layer.

**How CAM works:**

$$M_c(x, y) = \sum_k w_k^c \cdot f_k(x, y)$$

where $f_k$ is the $k$-th feature map and $w_k^c$ is the classifier weight for class $c$. The resulting $M_c$ is a heatmap highlighting where the model "looks" when predicting class $c$.

```
Input Image ──▶ CNN Backbone ──▶ Feature Maps ──▶ Global Avg Pool ──▶ Classifier ──▶ "Liver"
                     │                                     │
                     └── weighted sum by classifier weights ──▶ CAM heatmap
```

**The CAM problem**: CAMs only highlight the **most discriminative** region, not the full object. For a liver, the CAM might activate strongly on the liver edge (most distinctive feature) but miss the smooth interior.

**Solutions:**

| Method | How It Fixes CAM | Key Idea |
|--------|-----------------|----------|
| SEAM | Self-supervised refinement | Use spatial consistency to expand CAMs |
| AdvCAM | Adversarial training | Discover non-discriminative regions |
| PuzzleCAM | Patch shuffling | Force model to recognize all parts |
| LPCAM | Learnable perturbations | Optimize perturbations to find full extent |

### 2.3 Box-Supervised Segmentation

Given bounding boxes, the model knows the organ is inside the box but not the exact boundary. The training strategy:

1. **Restricted loss**: Only compute the segmentation loss for pixels *inside* the box.
2. **Background prior**: Pixels far from any box are likely background.
3. **Boundary refinement**: Use edge-aware post-processing (CRF, watershed) to snap predictions to actual organ boundaries.

$$\mathcal{L}_{\text{box}} = \sum_{(i,j) \in \text{box}} \mathcal{L}_{\text{pixel}}(y_{ij}, \hat{y}_{ij}) \cdot \text{mask}_{ij}$$

**BoxInst** extends this by adding a pairwise loss that encourages nearby pixels with similar color to have the same label — even outside the box.

### 2.4 Point-Supervised Segmentation

Even sparser — only a few labeled points per class. The model must propagate these point labels to full regions using **learned feature similarity**.

```
       ·  (liver point)
      / \
     /   \    feature similarity
    /     \   propagation
   ·───────· (unlabeled pixels with similar features → also liver)
```

### 2.5 Scribble-Supervised Segmentation

Scribbles provide thin strips of labeled pixels along the organ. The model fills in the gaps using:
- **Random walks**: Propagate labels along pixels with similar features.
- **CRF regularization**: Encourage smooth predictions that respect image edges.

### 2.6 Method Comparison

| Method | Annotation Type | Approach | Typical Dice |
|--------|---------------|----------|-------------|
| CAM | Image-level | Classification → activation maps | 50–65% |
| SEAM/AdvCAM | Image-level | Refined CAMs | 60–75% |
| BoxSupervised | Bounding box | Restricted loss + refinement | 75–90% |
| BoxInst | Bounding box | Pairwise loss + box supervision | 78–88% |
| Point | Points | Feature similarity propagation | 60–75% |
| Scribble | Scribbles | Random walk + CRF | 70–85% |

---

## 3. How It Works in APRIL-MedSeg

### 3.1 Training Script

All weakly supervised methods use `train_weakly_supervised.py`:

```bash
# Box-supervised
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml \
    --supervision_type box

# CAM-based
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/cam.yaml \
    --supervision_type cam

# Multi-instance learning
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/mil.yaml \
    --supervision_type mil
```

### 3.2 YAML Configuration — Box-Supervised

```yaml
model:
  num_classes: 9            # Synapse dataset: 8 organs + background
  img_size: 224
  encoder:
    name: timm_resnet50     # Stronger encoder helps compensate for weak supervision
    pretrained: true
    in_channels: 3
  decoder:
    name: unet              # Full UNet decoder for dense predictions
    params: {}
  bottleneck:
    name: none
    params: {}

data:
  img_size: 224
  image_dir: ./data/images
  annotation_file: ./data/annotations/boxes.json   # Box annotations (JSON format)
  val:
    image_dir: ./data/val/images
    mask_dir: ./data/val/masks                      # Full masks for validation
  test:
    image_dir: ./data/test/images
    mask_dir: ./data/test/masks

weak_supervision:
  method: box_supervised
  params:
    box_penalty: 0.1              # Penalty for predictions outside the box
    refine_iterations: 3          # Number of CRF refinement iterations

training:
  epochs: 150
  batch_size: 16
  loss:
    name: box_supervised
    params:
      box_penalty: 0.1
      refine_iterations: 3
  optimizer:
    name: adamw
    lr: 0.0002                    # Slightly higher LR for weak supervision
  scheduler:
    name: cosine
    min_lr: 1e-6
```

### 3.3 YAML Configuration — CAM-Based

```yaml
data:
  image_dir: ./data/images
  label_file: ./data/annotations/image_labels.json   # Image-level labels only
  cam_dir: ./data/cams                                # Pre-computed CAMs (optional)
  val:
    image_dir: ./data/val/images
    mask_dir: ./data/val/masks

weak_supervision:
  method: cam
  params:
    cam_threshold: 0.5           # Threshold for binarizing CAMs into pseudo-masks
    refine: true                 # Apply CRF refinement to CAMs
```

### 3.4 Available Methods

| Method | Config File | Annotation Type | Complexity |
|--------|------------|----------------|-----------|
| Box Supervised | `box_supervised.yaml` | Bounding boxes | Low |
| CAM | `cam.yaml` | Image-level labels | Medium |
| MIL | `mil.yaml` | Image-level (bags) | Medium |
| SEAM | `semples.yaml` | Image-level + refinement | High |
| Point | `point.yaml` | Sparse points | Low |
| Scribble | `scribble_sup.yaml` | Freehand scribbles | Low |
| BoxInst | `boxinst.yaml` | Bounding boxes | Medium |

### 3.5 Data Preparation

#### Box Annotations Format

```json
{
  "image_001.npy": [
    {"class": "liver", "bbox": [50, 80, 200, 180]},
    {"class": "spleen", "bbox": [30, 40, 120, 100]}
  ],
  "image_002.npy": [
    {"class": "liver", "bbox": [45, 75, 210, 190]}
  ]
}
```

Each bounding box is `[x_min, y_min, x_max, y_max]` in pixel coordinates.

#### Image-Level Labels Format

```json
{
  "image_001.npy": {"labels": ["liver", "spleen", "right_kidney"]},
  "image_002.npy": {"labels": ["liver", "stomach"]}
}
```

---

## 4. Step-by-Step: Your First Weakly Supervised Run

### Step 1: Prepare weak annotations

For **box-supervised** (recommended starting point):
- Use a labeling tool (Label Studio, CVAT) to draw bounding boxes around organs.
- Export as JSON in the format above.

For **CAM-based**:
- Prepare image-level labels (which classes are present in each image).
- Optionally pre-compute CAMs using a classification model.

### Step 2: Choose a method

For your first run, **box-supervised** offers the best trade-off between annotation effort and accuracy.

### Step 3: Train

```bash
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml \
    --supervision_type box \
    --output_dir output/weak_box
```

### Step 4: Evaluate against full supervision

```bash
python test.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml \
    --checkpoint output/weak_box/best_model.pth
```

Compare with a fully supervised baseline to measure the gap.

---

## 5. Parameter Tuning Guide

### Box-Supervised Parameters

| Parameter | Effect | Tuning Advice |
|-----------|--------|--------------|
| `box_penalty` | How much to penalize predictions outside the box | 0.1 is a good start. Increase if predictions leak outside boxes. |
| `refine_iterations` | Number of CRF post-processing passes | 3 is typical. More iterations = sharper boundaries but slower. |

### CAM Parameters

| Parameter | Effect | Tuning Advice |
|-----------|--------|--------------|
| `cam_threshold` | Activation threshold for CAM → pseudo-mask | 0.3–0.5. Lower = more coverage but more noise. Higher = less coverage but cleaner. |
| `refine` | Whether to apply CRF post-processing | Always `true` for better results. |

### General Advice

- **Use a stronger encoder** than you would for full supervision — the model needs more capacity to compensate for weak labels.
- **Train longer** — weakly supervised models converge slower because the training signal is noisier.
- **Validate with full masks** — always evaluate on a held-out set with pixel-level ground truth.

---

## 6. Common Pitfalls

### Pitfall 1: CAM only highlights object edges

**Symptom**: The predicted mask only covers the boundary of the organ, missing the interior.

**Fix**:
- Lower `cam_threshold` to include more of the activation map.
- Enable CRF refinement (`refine: true`).
- Use SEAM or AdvCAM instead of raw CAM — they explicitly address this issue.

### Pitfall 2: Box-supervised model predicts rectangular masks

**Symptom**: Predictions look like boxes rather than organ shapes.

**Fix**:
- Increase `refine_iterations` to encourage boundary snapping.
- Add edge-aware regularization.
- Ensure the training data has diverse box sizes and positions.

### Pitfall 3: Image-level labels produce very coarse masks

**Symptom**: Dice is below 60%, masks are blob-like.

**Fix**:
- CAM-based methods have an inherent accuracy ceiling. Consider collecting at least bounding box annotations.
- Use a two-stage approach: first generate CAMs, then refine with a segmentation model trained on the CAMs.
- Add self-training: use the initial CAMs to train a segmenter, then iterate.

### Pitfall 4: Validation Dice is much higher than test Dice

**Symptom**: Overfitting to the validation set's annotation style.

**Fix**:
- Ensure validation and test sets are from the same distribution.
- Use early stopping based on validation Dice.
- Add stronger data augmentation.

---

## 7. Recommended Experiments

### Experiment 1: Annotation Type Comparison

Same dataset, different annotation types:

```bash
# Box-supervised
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml \
    --supervision_type box

# CAM-based
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/cam.yaml \
    --supervision_type cam

# Point-supervised
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/point.yaml \
    --supervision_type point
```

**Expected results:**

| Annotation Type | Annotation Time | Dice | % of Full Supervision |
|----------------|----------------|------|----------------------|
| Full mask (baseline) | 20 min/img | 88% | 100% |
| Bounding box | 8 sec/img | 82% | 93% |
| Scribble | 20 sec/img | 77% | 88% |
| Points (5/class) | 5 sec/img | 68% | 77% |
| Image-level label | 2 sec/img | 62% | 70% |

### Experiment 2: Box-Supervised Refinement

Vary the CRF refinement iterations:

| Refinement Iterations | Dice | Boundary Accuracy |
|----------------------|------|-------------------|
| 0 (no refinement) | 75% | Poor (box-like) |
| 1 | 79% | Fair |
| 3 (default) | 82% | Good |
| 5 | 83% | Good (diminishing returns) |

---

## 8. Further Reading

### Key Papers

| Paper | Year | Venue | Key Idea |
|-------|------|-------|----------|
| [CAM](https://arxiv.org/abs/1512.04150) | 2016 | CVPR | Class activation maps from classification networks |
| [SEAM](https://arxiv.org/abs/2003.13053) | 2020 | CVPR | Self-supervised equivariant CAM refinement |
| [BoxInst](https://arxiv.org/abs/2012.02646) | 2021 | CVPR | Box-supervised instance segmentation |
| [AdvCAM](https://arxiv.org/abs/2104.00200) | 2021 | CVPR | Adversarial CAM for discovering non-discriminative regions |

### Related Documentation

- [All Weakly Supervised Methods](../paradigms/weakly_supervised.md) — Complete method catalog (20 methods)

---

[Back to Paradigms Overview](08_paradigms.md) | [Previous: Knowledge Distillation](08c_distillation.md) | [Next: Text-Guided](08e_text_guided.md)
