# Chapter 08: Advanced Training Paradigms — Overview

[Previous: Foundation Models](07_foundation.md) | [中文文档](08_paradigms_CN.md) | [Next: Deployment](09_deployment.md)

---

## Why Beyond Supervised Learning?

Standard supervised training — pairing every input image with a pixel-level ground truth mask — is the simplest learning paradigm, but also the most demanding in terms of data. In medical imaging, this creates several practical bottlenecks:

- **Label scarcity**: Expert annotations are expensive. A single CT scan may take a radiologist 30+ minutes to label.
- **Domain shift**: A model trained on Scanner A's images may fail on Scanner B's due to differences in acquisition protocol.
- **Model size**: State-of-the-art models are large — deploying them on edge devices requires compression without losing accuracy.
- **Coarse annotations**: Often, only bounding boxes, points, or image-level labels are available, not pixel-level masks.

Five advanced paradigms address these challenges, each targeting a specific bottleneck.

---

## The Five Paradigms

Each paradigm has its own detailed tutorial. Click through to learn the theory, configuration, and hands-on practice.

### [08a: Semi-Supervised Learning](08a_semi_supervised.md)

**Problem**: You have lots of images but few labels.

**Solution**: Use unlabeled data alongside labeled data via consistency regularization and pseudo-labeling.

| Key Methods | Core Idea | Expected Benefit |
|-------------|-----------|-----------------|
| Mean Teacher | EMA teacher provides stable targets | 85–95% of full supervision with 10% labels |
| CPS | Two networks cross-supervise each other | 85–93% of full supervision |
| UniMatch | Weak-to-strong augmentation consistency | Best single-model performance |

**Script**: `semi_train.py` · **Configs**: `configs/training_paradigms/semi_supervision/`

---

### [08b: Domain Adaptation](08b_domain_adaptation.md)

**Problem**: Your model works on Scanner A but fails on Scanner B.

**Solution**: Align source and target feature distributions, or adapt at test time.

| Key Methods | Core Idea | Expected Benefit |
|-------------|-----------|-----------------|
| AdvEnt | Adversarial entropy minimization | +8–15% on target domain |
| DANN | Gradient reversal for domain-invariant features | +5–12% on target domain |
| TENT | Test-time BatchNorm adaptation (no source data needed) | +5–10% on target domain |

**Script**: `train_domain_adaptation.py` · **Configs**: `configs/training_paradigms/domain_adaptation/`

---

### [08c: Knowledge Distillation](08c_distillation.md)

**Problem**: Your best model is too large for deployment.

**Solution**: Transfer knowledge from a large teacher to a small student via soft labels.

| Key Methods | Core Idea | Expected Benefit |
|-------------|-----------|-----------------|
| Hinton KD | Match softened output distributions | 90–95% of teacher accuracy |
| CWD | Per-channel feature distillation | 93–97% of teacher accuracy |
| DKD | Decoupled target/non-target distillation | 94–98% of teacher accuracy |

**Script**: `train_distillation.py` · **Configs**: `configs/training_paradigms/distillation/`

---

### [08d: Weakly Supervised Learning](08d_weakly_supervised.md)

**Problem**: You only have bounding boxes, points, or image-level labels — not pixel masks.

**Solution**: Train from coarse annotations using CAM, box supervision, or feature propagation.

| Key Methods | Annotation Type | Expected Benefit |
|-------------|----------------|-----------------|
| Box Supervised | Bounding boxes | 75–90% of full supervision |
| CAM / SEAM | Image-level labels | 60–75% of full supervision |
| Point / Scribble | Sparse points or strokes | 60–85% of full supervision |

**Script**: `train_weakly_supervised.py` · **Configs**: `configs/training_paradigms/weak_supervision/`

---

### [08e: Text-Guided Segmentation](08e_text_guided.md)

**Problem**: You want to segment new structures without collecting labeled data.

**Solution**: Use natural language descriptions as supervision via vision-language models (CLIP, MLLMs).

| Key Methods | Core Idea | Expected Benefit |
|-------------|-----------|-----------------|
| TextPromptUNet | CLIP text embeddings guide UNet | 70–82% Dice with training |
| MLLM + SAM2 | Detect-then-segment pipeline | 55–78% Dice zero-shot |
| SemanticGuidedUNet | Class embeddings + multi-scale attention | 65–78% Dice with training |

**Scripts**: `train_text_guided.py` / `test.py` · **Configs**: `configs/training_paradigms/text_guided/`

---

## Quick Comparison

| Paradigm | Data Required | Annotation Cost | Typical Improvement | Script |
|----------|-------------|-----------------|-------------------|--------|
| Supervised (baseline) | 100% labeled | Highest | Baseline | `train.py` |
| [Semi-supervised](08a_semi_supervised.md) | 10% labeled + 90% unlabeled | Low | 80–95% of full supervision | `semi_train.py` |
| [Domain adaptation](08b_domain_adaptation.md) | Source labeled + target unlabeled | Medium | +5–15% on target domain | `train_domain_adaptation.py` |
| [Distillation](08c_distillation.md) | Teacher model + labeled data | Same as supervised | 90–98% of teacher accuracy | `train_distillation.py` |
| [Weakly supervised](08d_weakly_supervised.md) | Box/point/image labels | Low | 75–90% of full supervision | `train_weakly_supervised.py` |
| [Text-guided](08e_text_guided.md) | Text prompts | Lowest | Varies (zero-shot: 40–70%) | `train_text_guided.py` |

---

## How to Choose?

```
Do you have pixel-level masks for all training data?
├── YES → Is the model too large for deployment?
│         ├── YES → Knowledge Distillation (08c)
│         └── NO  → Standard supervised training (train.py)
│
└── NO → What annotations do you have?
          ├── Some pixel masks + many unlabeled images → Semi-Supervised (08a)
          ├── Labeled source + unlabeled target domain → Domain Adaptation (08b)
          ├── Bounding boxes, points, or image labels → Weakly Supervised (08d)
          └── Only text descriptions → Text-Guided (08e)
```

---

## Key Papers (All Paradigms)

| Paper | Year | Venue | Paradigm | Key Contribution |
|-------|------|-------|----------|-----------------|
| [Mean Teacher](https://arxiv.org/abs/1703.01780) | 2017 | NeurIPS | Semi | EMA teacher for consistency regularization |
| [CPS](https://arxiv.org/abs/2106.01226) | 2021 | CVPR | Semi | Cross pseudo supervision with two networks |
| [DANN](https://arxiv.org/abs/1505.07818) | 2016 | JMLR | DA | Domain-adversarial training |
| [AdvEnt](https://arxiv.org/abs/1811.12833) | 2019 | CVPR | DA | Adversarial entropy minimization |
| [TENT](https://arxiv.org/abs/2006.10726) | 2021 | ICLR | DA | Test-time entropy minimization |
| [Hinton KD](https://arxiv.org/abs/1503.02531) | 2015 | NeurIPS WS | KD | Temperature-based knowledge distillation |
| [DKD](https://arxiv.org/abs/2203.08679) | 2022 | CVPR | KD | Decoupled knowledge distillation |
| [SEAM](https://arxiv.org/abs/2003.13053) | 2020 | CVPR | Weak | Self-supervised equivariant attention for CAM |
| [CLIP](https://arxiv.org/abs/2103.00020) | 2021 | ICML | Text | Vision-language contrastive pre-training |
| [CRIS](https://arxiv.org/abs/2211.10961) | 2023 | — | Text | Text-guided medical segmentation via CLIP |

---

## Related Documentation

- [Semi-Supervised Methods](../paradigms/semi_supervised.md) — All 21 semi-supervised methods
- [Domain Adaptation](../paradigms/domain_adaptation.md) — All 18 domain adaptation methods
- [Distillation](../paradigms/distillation.md) — All 27 distillation methods
- [Weakly Supervised](../paradigms/weakly_supervised.md) — All 20 weakly supervised methods
- [Text-Guided](../paradigms/text_guided.md) — All 12 text-guided models + MLLM pipeline

---

[Previous: Foundation Models](07_foundation.md) | [Next: Deployment](09_deployment.md)
