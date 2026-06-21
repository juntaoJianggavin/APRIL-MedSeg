# Chapter 08e: Text-Guided Segmentation

[Back to Paradigms Overview](08_paradigms.md) | [中文文档](08e_text_guided_CN.md) | [Previous: Weakly Supervised](08d_weakly_supervised.md) | [Next: Deployment](09_deployment.md)

---

## 1. When Should You Use Text-Guided Segmentation?

Traditional segmentation requires pixel-level masks — someone must trace every organ boundary. But what if you could just **describe** what to segment in plain English?

> "Segment the liver tumor in this CT scan."
> "Find all regions of pulmonary consolidation in this chest X-ray."

**Text-guided segmentation** uses natural language descriptions as supervision, leveraging vision-language models (VLMs) like CLIP that have learned to align visual and textual representations. This opens up entirely new possibilities:

| Scenario | Traditional Approach | Text-Guided Approach |
|----------|---------------------|---------------------|
| New organ, no training data | Collect 500+ labeled scans | Describe it: "spleen organ in CT" |
| Zero-shot generalization | Retrain for each new task | Change the text prompt |
| Rare finding | Nearly impossible to label enough | Describe: "ground-glass opacity" |
| Cross-modal (CT → MRI) | Retrain from scratch | Same text prompt, different image |

---

## 2. Core Concepts

### 2.1 Vision-Language Models (CLIP)

The foundation of text-guided segmentation is **CLIP** (Contrastive Language-Image Pre-training), trained on 400 million image-text pairs.

CLIP learns a **shared embedding space** where images and their text descriptions are close together:

```
Image: [CT scan with liver] ──▶ Image Encoder ──▶ ┐
                                                    ├── cosine similarity ≈ 0.85
Text:  "liver organ in CT"  ──▶ Text Encoder  ──▶ ┘

Image: [CT scan with liver] ──▶ Image Encoder ──▶ ┐
                                                    ├── cosine similarity ≈ 0.20
Text:  "brain MRI scan"     ──▶ Text Encoder  ──▶ ┘
```

**Contrastive training objective:**

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_j \exp(\text{sim}(I_i, T_j)/\tau)} + \log \frac{\exp(\text{sim}(T_i, I_i)/\tau)}{\sum_j \exp(\text{sim}(T_i, I_j)/\tau)} \right]$$

This creates aligned representations: "liver tumor in CT scan" is geometrically close to actual liver tumor images, and far from brain MRIs.

### 2.2 CLIP-Based Segmentation (TextPromptUNet)

The TextPromptUNet approach uses CLIP's text encoder to generate class embeddings, then integrates them into a UNet via cross-attention and feature modulation:

```
┌─────────────────────────────────────────────────────┐
│                    TextPromptUNet                    │
│                                                      │
│  "spleen organ"  ──▶ CLIP Text ──▶ text_emb ──┐     │
│  "liver organ"   ──▶ Encoder   ─▶ text_emb ──┤     │
│  "kidney organ"  ──▶           ─▶ text_emb ──┘     │
│                                                   │  │
│  CT Image ──▶ Encoder ──▶ Features ──┐            │  │
│                                       │            │  │
│                    Cross-Attention ◀──┼────────────┘  │
│                    (image queries,    │               │
│                     text keys/values) │               │
│                                       ▼               │
│                    FiLM Modulation                    │
│                    (scale + shift                     │
│                     features by text)                 │
│                                       ▼               │
│                    Decoder ──▶ Segmentation Mask      │
└─────────────────────────────────────────────────────┘
```

**Cross-attention**: Image features attend to text embeddings — the model learns "when I see this visual pattern, this text description is relevant."

**FiLM modulation** (Feature-wise Linear Modulation): Text embeddings generate scale (γ) and shift (β) parameters that modulate the image features:

$$\text{FiLM}(F, \text{text}) = \gamma(\text{text}) \odot F + \beta(\text{text})$$

This lets the text description "turn up" the features relevant to the described organ and "turn down" irrelevant ones.

### 2.3 MLLM Pipeline (Detect-then-Segment)

A more powerful approach uses a **Multi-Modal Large Language Model** (MLLM) as a grounding detector, followed by a specialized segmenter:

```
                    Step 1: Grounding              Step 2: Segmentation
                    ┌──────────────┐              ┌──────────────┐
                    │   MLLM       │              │   SAM2 /     │
"segment the  ──▶  │  (Qwen2-VL,  │  ──bbox──▶  │   MedSAM     │  ──▶ mask
 liver"            │   InternVL,  │              │              │
                    │  Grounding   │              │              │
                    │  DINO)       │              │              │
                    └──────────────┘              └──────────────┘
```

**Why two steps?**
1. MLLMs are great at understanding text and locating objects, but produce coarse bounding boxes — not pixel-accurate masks.
2. SAM2 / MedSAM are excellent at segmenting given a prompt (box or point), but can't understand text.
3. Combining them gives you text understanding + precise segmentation.

**Supported MLLM grounders:**

| Grounder | Model | Speed | Accuracy | Dependencies |
|----------|-------|-------|----------|-------------|
| Grounding DINO | `tiny` / `large` | Fast | Good | `groundingdino-py` |
| Qwen2-VL | 7B / 72B | Medium | Excellent | `transformers`, `qwen-vl-utils` |
| Qwen3-VL | 2B / 8B | Medium | Excellent | `transformers` |
| InternVL | 2.5 / 3 | Medium | Excellent | `transformers` |

**Supported mask generators:**

| Generator | Speed | Quality | Dependencies |
|-----------|-------|---------|-------------|
| SAM2 (Hiera-Large) | Medium | Excellent | `sam2` |
| MedSAM | Medium | Excellent (medical) | Custom |
| SAMMed2D | Fast | Good | Custom |

### 2.4 SemanticGuidedUNet

An alternative approach that uses **class-level semantic embeddings** (not necessarily CLIP) to guide segmentation through multi-scale attention:

```
Class embeddings ──▶ Multi-scale attention ──▶ Guided features ──▶ Decoder ──▶ Mask
                        (at each skip level)
```

This is simpler than full CLIP integration but effective when you have well-defined class names.

---

## 3. How It Works in APRIL-MedSeg

### 3.1 Two Approaches

| Approach | Script | When to Use |
|----------|--------|-------------|
| **Training-based** (TextPromptUNet) | `train_text_guided.py` | You have training data + want to fine-tune |
| **Pipeline-based** (MLLM + SAM2) | `test.py` / Python API | Zero-shot inference, no training needed |

### 3.2 Training-Based: TextPromptUNet

```bash
python train_text_guided.py \
    --config configs/training_paradigms/text_guided/synapse_clip.yaml \
    --output_dir output/text_guided
```

**YAML configuration walkthrough:**

```yaml
model:
  text_guided:
    model_type: TextPromptUNet          # Which text-guided architecture
    prompt_mode: clip                   # Use CLIP text encoder
    embed_dim: 512                      # CLIP text hidden size
    use_external_encoder: true
    # Natural language class descriptions
    # CLIP works better with complete phrases than single words
    class_names:
      - background region
      - spleen organ
      - right kidney organ
      - left kidney organ
      - gallbladder organ
      - esophagus organ
      - liver organ
      - stomach organ
      - aorta vessel

  encoder:
    name: timm_vit_clip_base_p32_256   # CLIP ViT-B/32 weights (aligned with text)
    pretrained: true
    in_channels: 3
    img_size: 256                      # 256 = 32×8, patch32 → 8×8 feature map
    params:
      out_channels: [128, 256, 512]
      pyramid_scales: 3

data:
  type: synapse
  img_size: 256
  train_dir: ./data/Synapse/train_npz
  val_dir: ./data/Synapse/test_vol_h5
  test_dir: ./data/Synapse/test_vol_h5
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt

training:
  epochs: 200
  batch_size: 8
  optimizer:
    name: adamw
    lr: 1e-4
  scheduler:
    name: cosine
    min_lr: 1e-6
  loss:
    name: compound
    params:
      ce_weight: 1.0
      dice_weight: 1.0
```

**Key considerations for text prompts:**
- Use complete phrases: `"liver organ"` works better than `"liver"`.
- Add context: `"liver organ in CT scan"` can improve modality-specific alignment.
- Be consistent: All class names should follow the same pattern.

### 3.3 Pipeline-Based: MLLM Detect-then-Segment

No training needed — just run inference:

```yaml
# synapse_grounding_dino_sam2.yaml
mllm:
  class_names:
    - spleen
    - right kidney
    - left kidney
    - gallbladder
    - esophagus
    - liver
    - stomach
    - aorta

  grounder:
    type: grounding_dino
    model_id: tiny
    device: cuda
    dtype: float32
    box_threshold: 0.35
    text_threshold: 0.25
    prompt_template: "a medical CT image of {class_name}"

  mask_generator:
    type: sam2
    model_id: facebook/sam2-hiera-large
    device: cuda
    multimask: false

  refinement:
    enabled: false                     # Enable UNet refinement on SAM2 masks

data:
  type: synapse
  img_size: 1024
  test_dir: ./data/Synapse/test_vol_h5
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt
```

**Python API usage:**

```python
import yaml
from medseg.inference.mllm import build_pipeline_from_config

# Load config
cfg = yaml.safe_load(open('configs/training_paradigms/text_guided/synapse_grounding_dino_sam2.yaml'))

# Build pipeline
pipe = build_pipeline_from_config(cfg)

# Run inference
result = pipe(image_rgb_uint8)    # Returns PipelineOutput with label_map, per_class_masks

# Access results
mask = result.label_map           # (H, W) integer mask
per_class = result.per_class_masks  # (num_classes, H, W) binary masks
```

### 3.4 Available Configurations

| Config | Approach | Models | Use Case |
|--------|---------|--------|----------|
| `synapse_clip.yaml` | Training | CLIP ViT + TextPromptUNet | Fine-tune with training data |
| `synapse_grounding_dino_sam2.yaml` | Pipeline | Grounding DINO + SAM2 | Zero-shot inference |
| `synapse_grounding_dino_medsam.yaml` | Pipeline | Grounding DINO + MedSAM | Medical-optimized zero-shot |
| `synapse_qwen2vl_sam2.yaml` | Pipeline | Qwen2-VL + SAM2 | Best accuracy (7B model) |
| `synapse_qwen3vl_medsam.yaml` | Pipeline | Qwen3-VL + MedSAM | Latest VLM + medical SAM |
| `synapse_internvl_sam2.yaml` | Pipeline | InternVL + SAM2 | Alternative VLM |

---

## 4. Step-by-Step: Your First Text-Guided Run

### Option A: Zero-Shot Inference (Easiest)

If you just want to segment organs without training:

```bash
# Install dependencies
pip install groundingdino-py
pip install git+https://github.com/facebookresearch/sam2.git

# Run inference
python test.py --config configs/training_paradigms/text_guided/synapse_grounding_dino_sam2.yaml
```

### Option B: Training-Based (Better Accuracy)

If you have training data and want a fine-tuned model:

**Step 1**: Prepare text descriptions for each class.

```yaml
class_names:
  - background region
  - spleen organ in CT scan
  - right kidney organ in CT scan
  - liver organ in CT scan
```

**Step 2**: Train.

```bash
python train_text_guided.py \
    --config configs/training_paradigms/text_guided/synapse_clip.yaml \
    --output_dir output/text_guided
```

**Step 3**: Evaluate.

```bash
python test.py \
    --config configs/training_paradigms/text_guided/synapse_clip.yaml \
    --checkpoint output/text_guided/best_model.pth
```

---

## 5. Parameter Tuning Guide

### CLIP-Based Training

| Parameter | Effect | Tuning Advice |
|-----------|--------|--------------|
| `embed_dim` | Text embedding dimension | Match CLIP model: 512 for ViT-B, 768 for ViT-L |
| `prompt_mode` | How text is encoded | `clip` for CLIP encoder; `learnable` for trained embeddings |
| `class_names` | Text descriptions | Use descriptive phrases, not single words |
| `img_size` | Input resolution | Must match CLIP encoder (256 for ViT-B/32) |

### MLLM Pipeline

| Parameter | Effect | Tuning Advice |
|-----------|--------|--------------|
| `box_threshold` | Detection confidence threshold | 0.3–0.4. Lower = more detections (more false positives). Higher = fewer but more confident. |
| `text_threshold` | Text matching threshold | 0.2–0.3. Lower = more permissive text matching. |
| `prompt_template` | Grounding prompt format | `"a medical CT image of {class_name}"` works well for CT. |
| `multimask` | Whether SAM returns multiple masks | `false` for single best mask; `true` to choose from 3 candidates. |

### Choosing the Right MLLM

| Need | Recommended Grounder | Recommended Mask Generator |
|------|---------------------|---------------------------|
| Fastest inference | Grounding DINO (tiny) | SAMMed2D |
| Best accuracy | Qwen2-VL (7B) | SAM2 (Hiera-Large) |
| Medical-optimized | Grounding DINO | MedSAM |
| Smallest VRAM | Grounding DINO (tiny) | SAMMed2D |

---

## 6. Common Pitfalls

### Pitfall 1: CLIP text descriptions don't match well

**Symptom**: The model confuses similar organs (e.g., left vs. right kidney).

**Fix**:
- Use more distinctive descriptions: `"left kidney organ on the patient's left side"` instead of just `"left kidney"`.
- Add spatial context: `"spleen organ in upper left abdomen"`.
- Increase training epochs for the cross-attention to learn better alignment.

### Pitfall 2: MLLM fails to detect small organs

**Symptom**: Grounding DINO misses small structures like the gallbladder or esophagus.

**Fix**:
- Lower `box_threshold` to 0.2–0.25 for small structures.
- Use a larger Grounding DINO model (`large` instead of `tiny`).
- Try a different MLLM (Qwen2-VL or InternVL handle small objects better).

### Pitfall 3: SAM2 produces poor masks on medical images

**Symptom**: SAM2 segments the wrong region or produces fragmented masks.

**Fix**:
- Use MedSAM instead — it's trained specifically on medical images.
- Adjust the box prompt: add padding around the grounding box.
- Enable UNet refinement (`refinement.enabled: true`) to clean up SAM2 output.

### Pitfall 4: High VRAM usage with large MLLMs

**Symptom**: CUDA OOM when using Qwen2-VL 7B or InternVL.

**Fix**:
- Use 4-bit quantization: set `dtype: float16` or `dtype: bfloat16`.
- Use Grounding DINO (tiny) as the grounder — it's much smaller.
- Reduce `img_size` to 512 or 256.

### Pitfall 5: TextPromptUNet ignores text entirely

**Symptom**: Model predicts the same mask regardless of text input.

**Fix**:
- Ensure the CLIP encoder weights are actually loaded (`use_external_encoder: true`).
- Check that `embed_dim` matches the text encoder output dimension.
- Increase the learning rate for the cross-attention layers.
- Verify that different text inputs produce different embeddings.

---

## 7. Recommended Experiments

### Experiment 1: Zero-Shot Comparison

Compare different MLLM pipelines on the same test set:

```bash
# Grounding DINO + SAM2
python test.py --config configs/training_paradigms/text_guided/synapse_grounding_dino_sam2.yaml

# Grounding DINO + MedSAM
python test.py --config configs/training_paradigms/text_guided/synapse_grounding_dino_medsam.yaml

# Qwen2-VL + SAM2
python test.py --config configs/training_paradigms/text_guided/synapse_qwen2vl_sam2.yaml
```

**Expected results:**

| Pipeline | Dice | Inference Time | VRAM |
|----------|------|---------------|------|
| GDINO + SAM2 | 55–70% | 2–4 sec/img | ~4GB |
| GDINO + MedSAM | 60–75% | 2–4 sec/img | ~4GB |
| Qwen2-VL + SAM2 | 65–78% | 5–10 sec/img | ~12GB |
| Trained TextPromptUNet | 70–82% | 0.1 sec/img | ~2GB |

### Experiment 2: Text Prompt Engineering

Vary the text descriptions and measure impact:

| Prompt Style | Example | Expected Dice Change |
|-------------|---------|---------------------|
| Single word | "liver" | Baseline |
| Descriptive | "liver organ" | +2–5% |
| Contextual | "liver organ in CT scan" | +3–7% |
| Spatial | "liver organ in right upper abdomen" | +2–4% vs contextual |

### Experiment 3: Training vs. Zero-Shot

```bash
# Zero-shot (no training)
python test.py --config configs/training_paradigms/text_guided/synapse_grounding_dino_medsam.yaml

# Training-based
python train_text_guided.py --config configs/training_paradigms/text_guided/synapse_clip.yaml
python test.py --config configs/training_paradigms/text_guided/synapse_clip.yaml \
    --checkpoint output/text_guided/best_model.pth
```

The trained model should outperform zero-shot by 10–20% Dice, but requires training data.

---

## 8. Further Reading

### Key Papers

| Paper | Year | Key Idea |
|-------|------|----------|
| [CLIP](https://arxiv.org/abs/2103.00020) | 2021 | Vision-language contrastive pre-training |
| [CRIS](https://arxiv.org/abs/2211.10961) | 2023 | Text-guided medical segmentation via CLIP |
| [Grounding DINO](https://arxiv.org/abs/2303.05499) | 2023 | Open-set object detection with text |
| [SAM2](https://arxiv.org/abs/2408.00714) | 2024 | Segment anything model v2 |
| [MedSAM](https://arxiv.org/abs/2304.12306) | 2023 | Medical-adapted SAM |
| [BiomedParse](https://arxiv.org/abs/2305.09860) | 2024 | Unified biomedical parsing |

### Related Documentation

- [All Text-Guided Models](../paradigms/text_guided.md) — Complete model catalog (12 models + MLLM pipeline)
- [MLLM Inference Guide](../deployment/mllm_inference.md) — Detailed MLLM pipeline documentation

---

[Back to Paradigms Overview](08_paradigms.md) | [Previous: Weakly Supervised](08d_weakly_supervised.md) | [Next: Deployment](09_deployment.md)
