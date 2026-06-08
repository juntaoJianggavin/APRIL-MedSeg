# 第 04 讲：训练与评估

[上一讲：数据](03_data_CN.md) | [English](04_training.md)

---

## 1. 背景与动机

训练医学分割模型不仅仅是调用 `model.fit()`。关键决策包括：

- **损失函数**：直接决定模型优化的目标
- **优化器与学习率**：控制收敛速度和稳定性
- **训练技巧**：AMP、DDP、梯度裁剪，提升效率和鲁棒性
- **评估协议**：一致的指标以实现公平比较

本章涵盖所有这些方面，配合实用的 YAML 配置和命令。

---

## 2. 核心概念

### 2.1 损失函数

框架在 `medseg/losses/` 中提供 15 个损失函数。最常用的：

| 损失 | 名称 | 公式 | 适用场景 |
|------|------|------|----------|
| **交叉熵** | `ce` | $-\sum p \log q$ | 通用，初始收敛好 |
| **Dice 损失** | `dice` | $1 - \frac{2|P \cap G|}{|P|+|G|}$ | 直接优化 Dice 分数，处理不平衡 |
| **Focal 损失** | `focal` | $-\alpha(1-p)^\gamma \log p$ | 极端类别不平衡 |
| **Tversky 损失** | `tversky` | $1 - \frac{|P \cap G|}{|P \cap G| + \beta|P \setminus G| + (1-\beta)|G \setminus P|}$ | 可调 FP/FN 权衡 |
| **Lovasz-Softmax** | `lovasz` | Jaccard 的子模扩展 | 直接优化 IoU |
| **边界损失** | `boundary` | 距离加权边界损失 | 锐化分割边界 |
| **豪斯多夫损失** | `hausdorff` | 可微 HD 近似 | 优化边界距离 |
| **Wasserstein Dice** | `wasserstein_dice` | 基于 Wasserstein 距离的 Dice | 处理不连通组件 |
| **边缘损失** | `edge` | 边缘感知边界损失 | 保持分割边缘 |
| **EL 损失** | `el` | 增强边界精炼损失 | 组合边界 + 区域 |
| **对比损失** | `contrastive` | 对比学习损失 | 表征学习 |

### 2.2 复合损失

最灵活的方式——多个损失的加权和：

```yaml
loss:
  name: compound
  params:
    losses:
      - name: ce
        weight: 0.4        # CE 帮助早期收敛
      - name: dice
        weight: 0.6        # Dice 优化最终指标
```

**为什么用复合损失？** 没有单一损失对所有场景都最优：
- CE 收敛快但不处理类别不平衡
- Dice 直接优化指标但训练初期可能不稳定
- 组合两者兼得两者的优点

### 2.3 优化器

| 优化器 | 名称 | 说明 | 适用场景 |
|--------|------|------|----------|
| **AdamW** | `adamw` | 解耦权重衰减的 Adam | 默认，通用性好 |
| **SGD** | `sgd` | 带动量的随机梯度下降 | 微调 Foundation 模型时 |

```yaml
optimizer:
  name: adamw
  lr: 0.001              # 初始学习率
  weight_decay: 0.0001   # L2 正则化强度
```

**学习率指南**：
- Foundation 编码器（冻结）：`lr=1e-4`
- Foundation 编码器（微调）：`lr=1e-5` 到 `lr=5e-5`
- 从头训练：`lr=1e-3`
- 小数据集：更低 LR（`1e-4`）以避免过拟合

### 2.4 学习率调度器

| 调度器 | 名称 | 行为 | 适用场景 |
|--------|------|------|----------|
| **余弦退火** | `cosine` | 平滑衰减到 `min_lr` | 默认，训练曲线平滑 |
| **阶梯** | `step` | 每 `step_size` 个 epoch 降 `gamma` 倍 | 需要陡峭 LR 下降时 |
| **多项式** | `poly` | 多项式衰减 | 余弦的替代方案 |

```yaml
scheduler:
  name: cosine
  min_lr: 1.0e-06        # 训练结束时的最小学习率
```

---

## 3. 方法详解

### 3.1 混合精度训练 (AMP)

AMP 在前向/反向传播中使用 FP16，在权重更新中使用 FP32，提供约 1.5 倍加速，精度损失可忽略。

```bash
python train.py --config my_config.yaml --amp
```

或通过 YAML 启用：

```yaml
training:
  amp: true
```

框架内部使用 `torch.cuda.amp.GradScaler`（见 `medseg/utils/amp_ddp.py`）：

```python
scaler = AMPScaler(enabled=use_amp)

with scaler.autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale_and_step(loss, optimizer, max_norm=1.0, model=model)
scaler.update()
```

### 3.2 分布式训练 (DDP)

多 GPU 训练，使用 `torchrun` 启动：

```bash
# 4 张 GPU
torchrun --nproc_per_node=4 train.py --config my_config.yaml

# 2 个节点各 2 张 GPU
torchrun --nproc_per_node=2 --nnodes=2 \
    --node_rank=0 --master_addr=192.168.1.1 \
    train.py --config my_config.yaml
```

框架的 DDP 功能：
- 自动 `DistributedDataParallel` 包装
- 同步 BatchNorm（`sync_bn: true`）
- 分布式采样器（无需手动 shuffle）
- 仅主进程记录日志

```yaml
training:
  parallel: auto          # auto / ddp / dp / single
  sync_bn: true           # DDP 时使用同步 BatchNorm
  find_unused: false      # 若有未使用参数设为 true
```

### 3.3 梯度裁剪

框架默认应用梯度裁剪（`max_norm=1.0`）以防止梯度爆炸：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3.4 评估协议

框架按类别计算三个指标：

```python
# 来自 medseg/utils/metrics.py
metrics = compute_metrics(pred, target, num_classes)
# 返回:
# {
#   "dice": {1: 0.85, 2: 0.72, ...},
#   "iou": {1: 0.74, 2: 0.56, ...},
#   "hd95": {1: 3.2, 2: 5.1, ...}
# }
```

验证每 `val_interval` 个 epoch 运行一次：

```yaml
training:
  val_interval: 10        # 每 10 个 epoch 验证一次
  save_interval: 50       # 每 50 个 epoch 保存检查点
```

当验证 Dice 提升时，自动保存最佳模型。

### 3.5 检查点

```bash
# 从检查点恢复
python train.py --config my_config.yaml --resume output/checkpoint_epoch100.pth
```

检查点格式：

```python
{
    'epoch': 100,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'best_dice': 0.8234,
}
```

### 3.6 日志

框架支持 TensorBoard 日志：

```bash
# 启动 TensorBoard
tensorboard --logdir output/
```

记录的指标：
- `train/loss`：每个 epoch 的训练损失
- 每个 epoch 的学习率

---

## 4. UltimateMedSeg 实操

### 4.1 完整训练工作流

```bash
# 第一步：训练
python train.py \
    --config configs/architectures/combinations/general/unet_basic.yaml \
    --output_dir ./experiments/unet_baseline \
    --amp

# 第二步：评估
python test.py \
    --config configs/architectures/combinations/general/unet_basic.yaml \
    --checkpoint ./experiments/unet_baseline/best_model.pth

# 第三步：评估并保存预测
python test.py \
    --config configs/architectures/combinations/general/unet_basic.yaml \
    --checkpoint ./experiments/unet_baseline/best_model.pth \
    --save_pred --output_dir ./experiments/unet_baseline/test_output
```

### 4.2 损失函数对比

创建不同损失的配置：

**纯 Dice：**
```yaml
training:
  loss:
    name: dice
```

**Focal 损失（极端不平衡）：**
```yaml
training:
  loss:
    name: focal
    params:
      gamma: 2.0
      alpha: 0.25
```

**复合损失（推荐）：**
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

**高级复合（CE + Dice + Boundary）：**
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

从 CLI 覆盖，无需修改 YAML：

```bash
python train.py --config my_config.yaml \
    --override training.loss.name=focal training.loss.params.gamma=2.0
```

### 4.3 优化器和调度器调参

```yaml
training:
  optimizer:
    name: adamw
    lr: 0.0005             # 尝试不同 LR
    weight_decay: 0.01     # 更强正则化
  scheduler:
    name: cosine
    min_lr: 0.000001
```

**Warmup**（训练初期逐步增加 LR）：

```yaml
training:
  optimizer:
    name: adamw
    lr: 0.001
    warmup:
      type: linear         # linear 或 constant
      steps: 500           # 500 步 warmup
      start_lr: 0.00001    # 从非常低的 LR 开始
```

### 4.4 测试时增强 (TTA)

TTA 在推理时应用多种增强并合并预测：

```bash
python test.py \
    --config my_config.yaml \
    --checkpoint best_model.pth \
    --tta \
    --tta-augs identity rot90 rot180 rot270 hflip vflip \
    --tta-merge mean
```

预期：+1-3% Dice 提升，推理时间增加约 6 倍。

### 4.5 多检查点集成

对多个检查点的预测取平均：

```bash
python test.py \
    --config my_config.yaml \
    --checkpoint best_fold0.pth best_fold1.pth best_fold2.pth \
    --ensemble-weights 0.4 0.3 0.3 \
    --ensemble-average logit
```

### 4.6 可复现性

```yaml
training:
  random_state: 42         # 所有随机性的种子
  deterministic: true      # cuDNN 确定性模式
```

```bash
python train.py --config my_config.yaml --seed 42
```

---

## 5. 推荐实验

### 实验 1：损失函数消融

在相同数据集和模型上对比：

| 损失 | 配置 | 预期行为 |
|------|------|----------|
| 仅 CE | `loss: {name: ce}` | 快速收敛，小目标 Dice 较低 |
| 仅 Dice | `loss: {name: dice}` | Dice 更好，收敛更慢 |
| 复合 | `loss: {name: compound, ...}` | 最佳平衡 |
| Focal | `loss: {name: focal, ...}` | 极端不平衡时更好 |

### 实验 2：学习率扫描

测试不同初始学习率：

| LR | 预期 |
|----|------|
| 1e-2 | 可能发散或震荡 |
| 1e-3 | 快速收敛（默认） |
| 5e-4 | 更慢但更稳定 |
| 1e-4 | 非常稳定，可能欠拟合 |

### 实验 3：AMP 影响

对比有无 AMP 的训练：

```bash
# 无 AMP
python train.py --config my_config.yaml --output_dir ./exp/fp32

# 有 AMP
python train.py --config my_config.yaml --amp --output_dir ./exp/amp

# 对比：训练时间、GPU 显存、最终 Dice
```

---

## 6. 延伸阅读

### 关键论文

| 论文 | 年份 | 主题 |
|------|------|------|
| [Dice Loss](https://arxiv.org/abs/1606.04797) | 2016 | V-Net：直接 Dice 优化 |
| [Focal Loss](https://arxiv.org/abs/1708.02002) | 2017 | RetinaNet：处理类别不平衡 |
| [Lovasz Loss](https://arxiv.org/abs/1705.08790) | 2018 | 子模 Jaccard 优化 |
| [Tversky Loss](https://arxiv.org/abs/1706.05721) | 2017 | 可调 FP/FN 权衡 |
| [AMP (Micikevicius)](https://arxiv.org/abs/1710.03740) | 2017 | 混合精度训练 |

### 相关文档

- [损失函数](../../medseg/losses/README.md) -- 15 个损失函数实现
- [指标源码](../../medseg/utils/metrics.py) -- Dice, IoU, HD95 计算
- [AMP/DDP 源码](../../medseg/utils/amp_ddp.py) -- 并行训练工具
- [Warmup 源码](../../medseg/utils/warmup.py) -- 学习率预热
- [TTA 源码](../../medseg/inference/tta.py) -- 测试时增强
- [集成源码](../../medseg/inference/ensemble.py) -- 多模型集成
- [研究指南](../research_guide_CN.md) -- 公平基准测试协议

---

[上一讲：数据](03_data_CN.md) | [教程索引](README_CN.md)

> **未完待续** -- 后续将陆续更新编码器进阶、解码器、Foundation 模型、高级训练范式和部署推理等章节，敬请期待！
