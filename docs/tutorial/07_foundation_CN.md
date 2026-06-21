# 第 07 章：Foundation 模型

[上一章：解码器](06_decoders_CN.md) | [English](07_foundation.md) | [下一章：高级训练范式](08_paradigms_CN.md)

---

## 1. 背景与动机

从头训练深度分割模型需要大规模标注数据集——像素级 mask 创建成本高昂且耗时，尤其在医学影像领域需要专家放射科医师逐张标注。

**Foundation 模型**提供了一种激进的替代方案：在数百万张图像上*无需人工标注*地预训练大型视觉 Transformer，然后将学到的特征迁移到下游医学任务，仅需极少标注数据。

三个理论基础解释了为何这可行：

- **迁移学习理论**：在广泛数据分布上学到的特征可泛化到狭窄的下游任务。
- **缩放定律**：模型质量随数据量、模型规模和计算量的增长可预测地提升。
- **自监督学习**：巧妙设计的 pretext 任务迫使模型在无需标签的情况下学习语义上有意义的表示。

本章解释 Foundation 模型背后的理论机制、如何将其适配到医学分割，以及微调策略如何控制迁移学习的权衡。

---

## 2. 核心概念

### 2.1 自监督学习范式

自监督学习（SSL）从数据本身创建监督信号——无需人工标签。模型必须预测输入的*隐藏部分*，迫使其理解底层结构。

**对比学习**（DINO, CLIP）：给定同一图像的两个增强视图，模型学习拉近它们的表示，同时推远不同图像的表示：

$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k) / \tau)}$$

其中 $\tau$ 是温度参数，控制相似度分布的锐度，$\text{sim}(\cdot,\cdot)$ 通常为余弦相似度。

| 方法 | 视图 | 关键创新 |
|------|------|----------|
| DINO | 同一图像的多裁剪 | 学生-教师自蒸馏，无需负样本对 |
| CLIP | 图像 + 文本 | 跨模态对齐（视觉 ↔ 语言） |
| DINOv2 | 同一图像的多裁剪 | 扩展到 142M 图像，改进特征 |

**掩码图像建模**（MAE, DINOv2）：随机遮挡 50-75% 的图像 patch，然后重建被遮挡内容：

$$\mathcal{L}_{\text{recon}} = \frac{1}{|M|} \sum_{i \in M} \| x_i - f_\theta(x_{\setminus M})_i \|^2$$

其中 $M$ 是被遮挡位置集合，$f_\theta$ 是处理可见 patch $x_{\setminus M}$ 以预测被遮挡 patch 的编码器-解码器。

**掩码为何有效**：要预测缺失 patch，模型必须理解其*周围内容*（上下文）和*通常填充该位置的内容*（语义）。这迫使学习远超简单纹理匹配的高层表示。

### 2.2 迁移学习为何有效

核心洞察是**低层和中层视觉特征跨域共享**。边缘、纹理、形状和空间关系同时出现在自然图像和医学扫描中。

**特征层次迁移**：

```
在自然图像上预训练（ImageNet / DINOv2）
  第 1-4 层:   边缘、梯度、纹理          ← 可直接迁移
  第 5-8 层:   模式、局部形状             ← 部分可迁移
  第 9-12 层:  物体部件、语义              ← 领域特定，需微调

在医学图像上微调
  第 1-4 层:   冻结（复用通用特征）
  第 5-8 层:   部分适配
  第 9-12 层:  完全适配到医学领域
```

**域差距问题**：自然图像和医学图像在纹理分布、色彩方案和物体结构上存在差异。仅在 ImageNet 上预训练的 Foundation 模型可能遗漏医学特定特征（如组织微结构、病变形态）。这驱动了**领域特定预训练**——在大规模医学图像集合上训练 Foundation 模型。

### 2.3 DPT 架构——从单一 ViT 获取多尺度

标准视觉 Transformer 通过相同层处理所有 patch，产生单尺度特征图。这对分割来说是有问题的，因为分割需要多尺度特征。

**Dense Prediction Transformer（DPT）**通过从 ViT 的*中间 block* 提取特征并通过专用层融合来解决这一问题：

```
预训练 ViT（12 或 24 个 block）
  │
  ├─ block_3  ──→ 重组 ──→ 融合 ──→ 阶段 1（1/4 分辨率）
  ├─ block_7  ──→ 重组 ──→ 融合 ──→ 阶段 2（1/8 分辨率）
  ├─ block_11 ──→ 重组 ──→ 融合 ──→ 阶段 3（1/16 分辨率）
  └─ block_15 ──→ 重组 ──→ 融合 ──→ 阶段 4（1/32 分辨率）
```

**重组层**将 1D token 序列重塑为目标分辨率的 2D 特征图。它们处理 ViT 恒定宽度 token 与多尺度特征所需不同空间尺寸之间的维度不匹配。

**融合层**使用残差卷积跨尺度聚合信息，类似 FPN 但操作来自单一网络深度的特征，而非层次化骨干。

**为何不用 FPN？** 标准 FPN 假设层次化编码器（ResNet 各阶段具有不同空间尺寸）。ViT 产生统一分辨率的 token。DPT 的重组+融合设计专门针对这种结构不匹配。

### 2.4 领域特定预训练

通用 Foundation 模型（DINOv2, CLIP）在自然图像上预训练。医学特定模型通过在大规模医学图像集合上预训练来缩小域差距：

| 领域 | 模型 | 预训练数据 | 为何有帮助 |
|------|------|-----------|-----------|
| 通用 | DINOv2, DINO | ImageNet, 网络图像 | 广泛的视觉特征 |
| 病理 | Phikon, UNI | 组织学切片（100K+ WSI） | 组织微结构 |
| 视网膜 | RETFound | 160 万张视网膜图像 | 眼底/OCT 模式 |
| 放射 | Rad-DINO | 胸部 X 光、CT 扫描 | 解剖结构 |
| 皮肤 | PanDerm | 100K+ 张皮肤图像 | 病变形态 |
| 超声 | UltraFedFM | 超声图像 | 斑点模式、回声特性 |

**迁移学习层次**：

1. **最佳匹配**：同一模态的领域特定模型（如病理用 Phikon）
2. **良好匹配**：相似模态的领域特定模型（如 CT 用 Rad-DINO）
3. **通用**：DINOv2 或 CLIP（尽管域差距，效果出奇地好）

### 2.5 微调理论

选择 Foundation 模型后，关键决策是**在下游任务上训练多少模型**。这涉及保留预训练特征与适配目标域之间的根本权衡。

**全量微调**（$\theta_{\text{all}}$ 可训练）：

$$\theta^* = \arg\min_\theta \mathcal{L}_{\text{task}}(f_\theta(x), y)$$

域差距大时精度最佳，但存在**灾难性遗忘**风险——模型可能用任务特定噪声覆盖有用的预训练特征，尤其在小数据集上。

**冻结编码器**（仅解码器 $\theta_{\text{dec}}$ 可训练）：

$$\theta_{\text{dec}}^* = \arg\min_{\theta_{\text{dec}}} \mathcal{L}_{\text{task}}(g_{\theta_{\text{dec}}}(f_{\theta_{\text{frozen}}}(x)), y)$$

保留所有预训练知识，但限制适配。域差距小时效果最佳。

**部分微调**（最后 $N$ 个 block 可训练）：

折中方案——早期层（通用特征）保持冻结，后期层（语义特征）适配：

$$\theta^* = \{\theta_{\text{block}_{L-N+1}}, ..., \theta_{\text{block}_L}, \theta_{\text{dec}}\}$$

**逐层学习率衰减**：不是二元的冻结/训练，而是从上到下施加递减学习率：

$$\eta_l = \eta_{\text{base}} \cdot \gamma^{L-l}$$

其中 $\gamma \in (0, 1)$ 是衰减因子，$l$ 是层索引。更深层（靠近输入）接收更小更新，保留通用特征。

**LoRA**（低秩适配）：不更新完整权重矩阵 $W$，而是学习低秩更新：

$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d,k)$$

这将可训练参数减少 90% 以上，同时允许全模型适配。最初为 LLM 开发，LoRA 越来越多地用于 ViT 微调。

---

## 3. 方法细节

### 3.1 医学模态——9 大类别

APRIL-MedSeg 涵盖 9 个医学模态的 38 个 Foundation 模型：

| 模态 | 模型 | 关键应用 |
|------|------|----------|
| 通用 | DINOv2, DINO, CLIP, SAM, DINOv3 | 跨域迁移 |
| 病理 | Phikon, UNI, PLIP, MUSK, Phikon-v2, KEEP | 组织学、WSI 分析 |
| 放射 | Rad-DINO, OmniRad, BioViL, CheXZero | X 光、CT、MRI |
| 眼科 | RETFound, RETFound-DINOv2, FLAIR, OphMAE | 视网膜疾病检测 |
| 皮肤 | DermCLIP, MoNet, PanDerm | 皮肤病变分割 |
| 通用医学 | BiomedCLIP, MedCLIP, MedSigLIP | 通用生物医学 |
| MLLM 视觉 | Qwen-VL, MedGemma, LLaVA-Med | 视觉语言推理 |
| 超声 | UltraFedFM, US-FMAE | 超声分析 |
| 内窥镜 | EndoViT, Endo-FM, Surgical-SAM | 消化道成像 |

### 3.2 Foundation 编码器输出

所有 Foundation 编码器通过 DPT head 产生多尺度特征，与任何解码器兼容：

```python
# Foundation 编码器输出（示例：DINOv2-Base, 12 个 block）
features = encoder(x)
# features[0]: (B, 256, H/4, W/4)    ← 来自 block 3
# features[1]: (B, 256, H/8, W/8)    ← 来自 block 7
# features[2]: (B, 256, H/16, W/16)  ← 来自 block 11
# features[3]: (B, 256, H/32, W/32)  ← 来自 block 15（瓶颈）
```

DPT head 确保跨阶段通道维度一致，解码器无需知道使用的是哪个 Foundation 模型。

---

## 4. 在 APRIL-MedSeg 中实践

```yaml
# 通用 Foundation 模型（DINOv2）
model:
  encoder: { name: dinov2_base, pretrained: true, freeze: true }
  decoder: { name: unet }

# 病理特定（Phikon v2）
model:
  encoder: { name: phikon_v2, pretrained: true, freeze: false }
  decoder: { name: cascade, params: { num_stages: 4 } }

# 部分微调（最后 4 个 block）
model:
  encoder:
    name: dinov2_base
    pretrained: true
    freeze: true
    params: { unfreeze_last_n: 4 }
  decoder: { name: unet }

# 视网膜特定（RETFound）
model:
  encoder: { name: retfound_dinov2, pretrained: true }
  decoder: { name: unet }
```

---

## 5. 推荐实验

### 实验 1：Foundation vs 标准编码器

相同解码器（UNet），相同数据集，比较编码器类型：

| 编码器 | 类型 | 可训练参数 | 预期 Dice |
|--------|------|-----------|-----------|
| `timm_resnet50` | 标准 CNN | ~25M | 基线 |
| `dinov2_base`（冻结） | Foundation | ~2M（仅解码器） | +3-6% |
| `dinov2_base`（全量） | Foundation | ~88M | +4-8% |

### 实验 2：冻结 vs 微调

相同 Foundation 编码器，不同微调策略：

| 策略 | 可训练参数 | 训练速度 | 最佳时机 |
|------|-----------|----------|----------|
| `freeze: true` | ~2M | 快 | 域差距小 |
| `unfreeze_last_n: 4` | ~20M | 中等 | 中等域差距 |
| `freeze: false` | ~88M | 慢 | 域差距大，数据充足 |

### 实验 3：领域特定 vs 通用

在病理数据集上比较：

| 编码器 | 预训练领域 | 预期性能 |
|--------|-----------|----------|
| `dinov2_base` | 通用（ImageNet） | 良好基线 |
| `phikon_v2` | 病理（组织学） | 最佳（域匹配） |
| `retfound_dinov2` | 视网膜 | 较低（域不匹配） |

---

## 6. 延伸阅读

### 关键论文

| 论文 | 年份 | 会议 | 关键贡献 |
|------|------|------|----------|
| [DINO](https://arxiv.org/abs/2104.14294) | 2021 | ICCV | 无负样本对的自蒸馏 |
| [DINOv2](https://arxiv.org/abs/2304.07193) | 2023 | - | 将自监督扩展到 142M 图像 |
| [MAE](https://arxiv.org/abs/2111.06377) | 2022 | CVPR | 掩码自编码器是可扩展的视觉学习器 |
| [CLIP](https://arxiv.org/abs/2103.00020) | 2021 | ICML | 视觉语言对比预训练 |
| [DPT](https://arxiv.org/abs/2103.13413) | 2021 | ICCV | Dense Prediction Transformer 架构 |
| [SAM](https://arxiv.org/abs/2304.02643) | 2023 | - | Segment Anything——可提示分割 |
| [Phikon](https://arxiv.org/abs/2307.10873) | 2023 | - | 组织病理学 Foundation 模型 |
| [UNI](https://arxiv.org/abs/2308.15474) | 2024 | Nature Medicine | 通用病理 Foundation 模型 |
| [RETFound](https://arxiv.org/abs/2301.07786) | 2023 | - | 160 万张图像的视网膜 Foundation 模型 |
| [LoRA](https://arxiv.org/abs/2106.09685) | 2022 | ICLR | 大模型的低秩适配 |

### 相关文档

- [Foundation 编码器](../models/encoders.md#foundation-models) — 9 个模态的所有 38 个 Foundation 编码器
- [权重管理](../models/encoders.md#weight-management) — 自动下载和缓存系统
- [DPT Head](../models/encoders.md#dpt-head) — Dense Prediction Transformer 架构细节

---

[上一章：解码器](06_decoders_CN.md) | [下一章：高级训练范式](08_paradigms_CN.md)
