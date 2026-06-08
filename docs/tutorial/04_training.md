# Chapter 04: Training and Evaluation

[Previous: Data](03_data.md) | [中文文档](04_training_CN.md)

---

## 1. Background and Motivation

Training a medical segmentation model involves more than just calling `model.fit()`. Key decisions include:

- **Loss function**: directly determines what the model optimizes
- **Optimizer and learning rate**: controls convergence speed and stability
- **Training tricks**: AMP, DDP, gradient clipping for efficiency and robustness
- **Evaluation protocol**: consistent metrics for fair comparison

This chapter covers all these aspects with practical YAML configurations and commands.

---

## 2. Core Concepts

### 2.1 Loss Functions

The framework provides 15 loss functions in `medseg/losses/`. The most commonly used:

| Loss | Name | Formula | When to Use |
|------|------|---------|-------------|
| **Cross-Entropy** | `ce` | $-\sum p \log q$ | General purpose, good initial convergence |
| **Dice Loss** | `dice` | $1 - \frac{2|P \cap G|}{|P|+|G|}$ | Directly optimizes Dice score, handles imbalance |
| **Focal Loss** | `focal` | $-\alpha(1-p)^\gamma \log p$ | Extreme class imbalance |
| **Tversky Loss** | `tversky` | $1 - \frac{|P \cap G|}{|P \cap G| + \beta|P \setminus G| + (1-\beta)|G \setminus P|}$ | Tunable FP/FN trade-off |
| **Lovasz-Softmax** | `lovasz` | Submodular extension of Jaccard | Optimizes IoU directly |
| **Boundary Loss** | `boundary` | Distance-weighted boundary loss | Sharpens segmentation boundaries |
| **Hausdorff Loss** | `hausdorff` | Differentiable HD approximation | Optimizes boundary distance |
| **Wasserstein Dice** | `wasserstein_dice` | Wasserstein-distance-based Dice | Handles disconnected components |
| **Edge Loss** | `edge` | Edge-aware boundary loss | Preserves edges in segmentation |
| **EL Loss** | `el` | Enhanced loss for boundary refinement | Combined boundary + region |
| **Contrastive Loss** | `contrastive` | Contrastive learning loss | Representation learning |

### 2.2 Compound Loss

The most flexible approach -- weighted sum of multiple losses:

```yaml
loss:
  name: compound
  params:
    losses:
      - name: ce
        weight: 0.4        # CE helps early convergence
      - name: dice
        weight: 0.6        # Dice optimizes the final metric
```

**Why compound?** No single loss is optimal for all scenarios:
- CE converges fast but doesn't handle class imbalance
- Dice directly optimizes the metric but can be unstable early in training
- Combining them gets the best of both worlds

### 2.3 Optimizers

| Optimizer | Name | Description | When to Use |
|-----------|------|-------------|-------------|
| **AdamW** | `adamw` | Adam with decoupled weight decay | Default, works well universally |
| **SGD** | `sgd` | Stochastic gradient descent with momentum | When fine-tuning foundation models |

```yaml
optimizer:
  name: adamw
  lr: 0.001              # Initial learning rate
  weight_decay: 0.0001   # L2 regularization strength
```

**Learning rate guidelines**:
- Foundation encoders (frozen): `lr=1e-4`
- Foundation encoders (fine-tuned): `lr=1e-5` to `lr=5e-5`
- From-scratch training: `lr=1e-3`
- Small datasets: lower LR (`1e-4`) to avoid overfitting

### 2.4 Learning Rate Schedulers

| Scheduler | Name | Behavior | When to Use |
|-----------|------|----------|-------------|
| **Cosine Annealing** | `cosine` | Smooth decay to `min_lr` | Default, smooth training curves |
| **Step** | `step` | Drop LR by `gamma` every `step_size` epochs | When you want sharp LR drops |
| **Polynomial** | `poly` | Polynomial decay | Alternative to cosine |

```yaml
scheduler:
  name: cosine
  min_lr: 1.0e-06        # Minimum learning rate at end of training
```

---

## 3. Method Details

### 3.1 Mixed Precision Training (AMP)

AMP uses FP16 for forward/backward passes and FP32 for weight updates, providing ~1.5x speedup with negligible accuracy loss.

```bash
python train.py --config my_config.yaml --amp
```

Or enable via YAML:

```yaml
training:
  amp: true
```

The framework uses `torch.cuda.amp.GradScaler` internally (see `medseg/utils/amp_ddp.py`):

```python
scaler = AMPScaler(enabled=use_amp)

with scaler.autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale_and_step(loss, optimizer, max_norm=1.0, model=model)
scaler.update()
```

### 3.2 Distributed Training (DDP)

For multi-GPU training, launch with `torchrun`:

```bash
# 4 GPUs
torchrun --nproc_per_node=4 train.py --config my_config.yaml

# 2 GPUs on 2 nodes
torchrun --nproc_per_node=2 --nnodes=2 \
    --node_rank=0 --master_addr=192.168.1.1 \
    train.py --config my_config.yaml
```

DDP features in the framework:
- Automatic `DistributedDataParallel` wrapping
- Synchronized BatchNorm (`sync_bn: true`)
- Distributed sampler (no manual shuffle needed)
- Main-process-only logging

```yaml
training:
  parallel: auto          # auto / ddp / dp / single
  sync_bn: true           # Synchronized BatchNorm for DDP
  find_unused: false      # Set true if some parameters are unused
```

### 3.3 Gradient Clipping

The framework applies gradient clipping by default (`max_norm=1.0`) to prevent exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3.4 Evaluation Protocol

The framework computes three metrics per class:

```python
# From medseg/utils/metrics.py
metrics = compute_metrics(pred, target, num_classes)
# Returns:
# {
#   "dice": {1: 0.85, 2: 0.72, ...},
#   "iou": {1: 0.74, 2: 0.56, ...},
#   "hd95": {1: 3.2, 2: 5.1, ...}
# }
```

Validation runs every `val_interval` epochs:

```yaml
training:
  val_interval: 10        # Validate every 10 epochs
  save_interval: 50       # Save checkpoint every 50 epochs
```

Best model is saved automatically when validation Dice improves.

### 3.5 Checkpointing

```bash
# Resume from checkpoint
python train.py --config my_config.yaml --resume output/checkpoint_epoch100.pth
```

Checkpoint format:

```python
{
    'epoch': 100,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'best_dice': 0.8234,
}
```

### 3.6 Logging

The framework supports TensorBoard logging:

```bash
# Start TensorBoard
tensorboard --logdir output/
```

Logged metrics:
- `train/loss`: Training loss per epoch
- Learning rate per epoch

---

## 4. Hands-On with UltimateMedSeg

### 4.1 Complete Training Workflow

```bash
# Step 1: Train
python train.py \
    --config configs/architectures/combinations/general/unet_basic.yaml \
    --output_dir ./experiments/unet_baseline \
    --amp

# Step 2: Evaluate
python test.py \
    --config configs/architectures/combinations/general/unet_basic.yaml \
    --checkpoint ./experiments/unet_baseline/best_model.pth

# Step 3: Evaluate with predictions saved
python test.py \
    --config configs/architectures/combinations/general/unet_basic.yaml \
    --checkpoint ./experiments/unet_baseline/best_model.pth \
    --save_pred --output_dir ./experiments/unet_baseline/test_output
```

### 4.2 Loss Function Comparison

Create configs with different losses:

**Dice-only:**
```yaml
training:
  loss:
    name: dice
```

**Focal loss (for extreme imbalance):**
```yaml
training:
  loss:
    name: focal
    params:
      gamma: 2.0
      alpha: 0.25
```

**Compound (recommended):**
```yaml
training:
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4
        - name: dice
          weight: 0.6
```

**Advanced compound (CE + Dice + Boundary):**
```yaml
training:
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.3
        - name: dice
          weight: 0.5
        - name: boundary
          weight: 0.2
```

Override from CLI without editing YAML:

```bash
python train.py --config my_config.yaml \
    --override training.loss.name=focal training.loss.params.gamma=2.0
```

### 4.3 Optimizer and Scheduler Tuning

```yaml
training:
  optimizer:
    name: adamw
    lr: 0.0005             # Try different LRs
    weight_decay: 0.01     # Stronger regularization
  scheduler:
    name: cosine
    min_lr: 0.000001
```

**Warmup** (gradual LR increase at the start):

```yaml
training:
  optimizer:
    name: adamw
    lr: 0.001
    warmup:
      type: linear         # linear or constant
      steps: 500           # Warmup for 500 steps
      start_lr: 0.00001    # Start from very low LR
```

### 4.4 Test-Time Augmentation (TTA)

TTA applies multiple augmentations at inference and merges predictions:

```bash
python test.py \
    --config my_config.yaml \
    --checkpoint best_model.pth \
    --tta \
    --tta-augs identity rot90 rot180 rot270 hflip vflip \
    --tta-merge mean
```

Expected: +1-3% Dice improvement at the cost of ~6x inference time.

### 4.5 Multi-Checkpoint Ensemble

Average predictions from multiple checkpoints:

```bash
python test.py \
    --config my_config.yaml \
    --checkpoint best_fold0.pth best_fold1.pth best_fold2.pth \
    --ensemble-weights 0.4 0.3 0.3 \
    --ensemble-average logit
```

### 4.6 Reproducibility

```yaml
training:
  random_state: 42         # Seed for all randomness
  deterministic: true      # cuDNN deterministic mode
```

```bash
python train.py --config my_config.yaml --seed 42
```

---

## 5. Recommended Experiments

### Experiment 1: Loss Function Ablation

On the same dataset and model, compare:

| Loss | Config | Expected Behavior |
|------|--------|-------------------|
| CE only | `loss: {name: ce}` | Fast convergence, lower Dice on small targets |
| Dice only | `loss: {name: dice}` | Better Dice, slower convergence |
| Compound | `loss: {name: compound, ...}` | Best balance |
| Focal | `loss: {name: focal, ...}` | Better on extreme imbalance |

### Experiment 2: Learning Rate Sweep

Test different initial learning rates:

| LR | Expected |
|----|---------|
| 1e-2 | May diverge or oscillate |
| 1e-3 | Fast convergence (default) |
| 5e-4 | Slower but more stable |
| 1e-4 | Very stable, may underfit |

### Experiment 3: AMP Impact

Compare training with and without AMP:

```bash
# Without AMP
python train.py --config my_config.yaml --output_dir ./exp/fp32

# With AMP
python train.py --config my_config.yaml --amp --output_dir ./exp/amp

# Compare: training time, GPU memory, final Dice
```

---

## 6. Further Reading

### Key Papers

| Paper | Year | Topic |
|-------|------|-------|
| [Dice Loss](https://arxiv.org/abs/1606.04797) | 2016 | V-Net: direct Dice optimization |
| [Focal Loss](https://arxiv.org/abs/1708.02002) | 2017 | RetinaNet: handling class imbalance |
| [Lovasz Loss](https://arxiv.org/abs/1705.08790) | 2018 | Submodular Jaccard optimization |
| [Tversky Loss](https://arxiv.org/abs/1706.05721) | 2017 | Tunable FP/FN trade-off |
| [AMP (Micikevicius)](https://arxiv.org/abs/1710.03740) | 2017 | Mixed precision training |

### Related Documentation

- [Loss Functions](../../medseg/losses/README.md) -- All 15 loss implementations
- [Metrics Source](../../medseg/utils/metrics.py) -- Dice, IoU, HD95 computation
- [AMP/DDP Source](../../medseg/utils/amp_ddp.py) -- Parallel training utilities
- [Warmup Source](../../medseg/utils/warmup.py) -- Learning rate warmup
- [TTA Source](../../medseg/inference/tta.py) -- Test-time augmentation
- [Ensemble Source](../../medseg/inference/ensemble.py) -- Multi-model ensemble
- [Research Guide](../research_guide.md) -- Fair benchmarking protocols

---

[Previous: Data](03_data.md) | [Tutorial Index](README.md)

> **To be continued** -- More chapters on encoders, decoders, foundation models, advanced training paradigms, and deployment are planned. Stay tuned!
