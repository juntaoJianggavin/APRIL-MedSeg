# Text-Guided Segmentation

[中文文档](text_guided_CN.md)

Two text-guided paradigms: trainable models (end-to-end) and inference pipelines (detector + segmenter).

---

## Trainable Models (12)

All 2D end-to-end text-vision segmentation models in `medseg/models/text_unet/`.

| Key | Model | Paper | Published | GitHub | YAML |
|-----|-------|-------|-------|--------|------|
| `tganet` | TGANet | Tomar et al. | MICCAI 2022 | [nikhilroxtomar/TGANet](https://github.com/nikhilroxtomar/TGANet) | - |
| `lvit` | LViT | Li et al. | TMI 2023 | [HUANGLIZI/LViT](https://github.com/HUANGLIZI/LViT) | [mosmed_plus_lvit.yaml](../../configs/training_paradigms/text_guided/mosmed_plus_lvit.yaml), [qata_covid19_lvit.yaml](../../configs/training_paradigms/text_guided/qata_covid19_lvit.yaml) |
| `languide` | LanGuideMedSeg | Zhong et al. | MICCAI 2023 | [Junelin2333/LanGuideMedSeg-MICCAI2023](https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023) | [mosmed_plus_languide.yaml](../../configs/training_paradigms/text_guided/mosmed_plus_languide.yaml), [qata_covid19_languide.yaml](../../configs/training_paradigms/text_guided/qata_covid19_languide.yaml) |
| `clip_universal` | CLIP-Driven Universal Model | Liu et al. | ICCV 2023 | [ljwztc/CLIP-Driven-Universal-Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model) | - |
| `cris` | CRIS | Wang et al. | CVPR 2022 | [DerrickWang005/CRIS.pytorch](https://github.com/DerrickWang005/CRIS.pytorch) | - |
| `biomedparse` | BiomedParse | Zhao et al. | Nature Methods 2025 | [microsoft/BiomedParse](https://github.com/microsoft/BiomedParse) | - |
| `tpro` | TPRO | Zhang et al. | MICCAI 2023 | [zhangst431/TPRO](https://github.com/zhangst431/TPRO) | - |
| `salip` | SaLIP | Aleem et al. | CVPRW 2024 | [aleemsidra/SaLIP](https://github.com/aleemsidra/SaLIP) | - |
| `causal_clipseg` | CausalCLIPSeg | Chen et al. | MICCAI 2024 | [WUTCM-Lab/CausalCLIPSeg](https://github.com/WUTCM-Lab/CausalCLIPSeg) | - |
| `medclip_sam` | MedCLIP-SAM | Koleilat et al. | MICCAI 2024 | [HealthX-Lab/MedCLIP-SAM](https://github.com/HealthX-Lab/MedCLIP-SAM) | - |
| `tp_drseg` | TP-DRSeg | Li et al. | MICCAI 2024 | [wxliii/TP-DRSeg](https://github.com/wxliii/TP-DRSeg) *(no longer available)* | - |
| `cxrclipseg` | CXR-CLIPSeg | You et al. | MICCAI 2023 | [Soombit-ai/cxr-clip](https://github.com/Soombit-ai/cxr-clip) | - |

### Text Input Format

Text input is provided via `class_names` in the config:

```yaml
model:
  text_guided:
    model_type: TextPromptUNet
    prompt_mode: clip              # clip | learnable
    embed_dim: 512
    use_external_encoder: true
    class_names:                   # natural language descriptions
      - background region
      - spleen organ
      - right kidney organ
      - liver organ
```

### Trainable Model YAML

**CLIP-aligned TextPromptUNet:**

```yaml
model:
  text_guided:
    model_type: TextPromptUNet
    prompt_mode: clip
    embed_dim: 512
    use_external_encoder: true
    class_names:
      - background region
      - spleen organ
      - right kidney organ
  encoder:
    name: timm_vit_clip_base_p32_256   # original CLIP weights, aligned with text encoder
    pretrained: true
    in_channels: 3
    img_size: 256
    params:
      out_channels: [128, 256, 512]
      pyramid_scales: 3

data:
  type: synapse
  img_size: 256
  train_dir: ./data/Synapse/train_npz
  val_dir: ./data/Synapse/test_vol_h5

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

**LViT (with per-image text):**

```yaml
model:
  num_classes: 1
  img_size: 224
  architecture: lvit
  arch_params:
    base_channel: 64
    text_len: 10
    text_embed_dim: 768

data:
  type: mosmed_plus
  img_size: 224
  data_root: ./data/MosMedDataPlus
  tokenizer_name: bert-base-uncased
  text_max_length: 10
  text_source: dataset       # per-image text from dataset Excel
```

### Train

```bash
python train_text_guided.py --config configs/training_paradigms/text_guided/synapse_clip.yaml
```

### Test

The `test_text_guided.py` script auto-detects the paradigm from the config:

```bash
# Trainable model (split-architecture: encoder + decoder)
python test_text_guided.py \
    --config configs/training_paradigms/text_guided/synapse_clip.yaml \
    --checkpoint output_text_guided/best_model.pth

# Inference-only pipeline (detect-then-segment, no --checkpoint needed)
python test_text_guided.py \
    --config configs/training_paradigms/text_guided/synapse_grounding_dino_sam2.yaml

# Save prediction masks
python test_text_guided.py --config ... --checkpoint ... --save_pred
```

---

## Inference Pipeline

Detect-then-Segment: detector grounding + segmenter mask generation.

### Detectors (9)

| Detector | Type | Source |
|----------|------|--------|
| Grounding DINO | Open-vocab detector | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |
| Qwen2-VL | MLLM grounding | [QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) |
| Qwen2.5-VL | MLLM grounding | [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) |
| Qwen3-VL | MLLM grounding | [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) |
| InternVL | MLLM grounding | [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) |
| LLaVA | MLLM grounding | [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) |
| MiniCPM-V | MLLM grounding | [OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) |
| Phi3-V | MLLM grounding | [microsoft/Phi-3-vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) |
| CogVLM | MLLM grounding | [THUDM/CogVLM](https://github.com/THUDM/CogVLM) |

### Segmenters (4)

| Segmenter | Paper | Source |
|-----------|-------|--------|
| SAM2 | Meta, 2024 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| MedSAM | Ma et al., NatComm 2024 | [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) |
| SAM-Med2D | - | - |
| LiteMedSAM | - | - |

### Pipeline YAML

```yaml
mllm:
  class_names:
    - spleen
    - right kidney
    - left kidney
    - liver
    - stomach
  grounder:
    type: grounding_dino     # grounding_dino | qwen2_vl | qwen3_vl | internvl
    model_id: tiny
    device: cuda
    box_threshold: 0.35
    text_threshold: 0.25
    prompt_template: "a medical CT image of {class_name}"
  mask_generator:
    type: sam2               # sam2 | medsam | sammed2d | litemedsam
    model_id: facebook/sam2-hiera-large
    device: cuda
    multimask: false
  refinement:
    enabled: false

data:
  type: synapse
  img_size: 1024
  test_dir: ./data/Synapse/test_vol_h5
```

### Pipeline Usage

```python
import yaml
from medseg.inference.mllm import build_pipeline_from_config

cfg = yaml.safe_load(open('configs/training_paradigms/text_guided/synapse_grounding_dino_sam2.yaml', encoding='utf-8'))
pipe = build_pipeline_from_config(cfg)

out = pipe(image_rgb_uint8)
label_map = out.label_map            # (H, W) int
per_class = out.per_class_masks      # {'spleen': mask, ...}
```

Configs are in `configs/training_paradigms/text_guided/`.
