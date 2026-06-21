# 第 05 章：编码器进阶

[上一章：训练](04_training_CN.md) | [English](05_encoders.md) | [下一章：解码器](06_decoders_CN.md)

---

## 1. 背景与动机

编码器是分割模型的核心骨干——它将原始输入图像转换为多尺度特征的层次化表示。这些特征的质量直接决定了解码器能够重建出什么。

编码器设计围绕三个根本问题：

- **应提取哪些特征？** 边缘、纹理、形状、语义模式——不同任务需要不同特征。
- **感受野有多大？** 小病灶需要精细的局部特征；大器官需要全局上下文。
- **计算预算是多少？** 高分辨率医学图像（512×512+）要求高效架构。

编码器选择是分割中最具影响力的架构决策。本章解释各主要编码器家族的理论基础及其各自优势场景。

---

## 2. 核心概念

### 2.1 层次化特征提取

所有现代编码器都产生**特征金字塔**——一系列空间分辨率递减、通道深度递增的特征图：

```
输入: (B, 3, H, W)
  → 阶段 1: (B, C1, H/2, W/2)     # 低级：边缘、纹理
  → 阶段 2: (B, C2, H/4, W/4)     # 中级：模式、局部形状
  → 阶段 3: (B, C3, H/8, W/8)     # 高级：物体部件
  → 阶段 4: (B, C4, H/16, W/16)   # 语义：物体身份
  → 阶段 5: (B, C5, H/32, W/32)   # 抽象：类别级上下文
```

这种层次结构自然产生：早期层检测简单模式（边缘、梯度），更深层将其组合为复杂语义概念。这不是设计选择，而是堆叠非线性变换的数学必然。

### 2.2 感受野理论

**感受野（Receptive Field, RF）**是输入图像中影响某个神经元激活的区域。它是理解编码器行为最重要的概念。

对于堆叠的卷积层，有效感受野按以下公式增长：

$$RF = 1 + \sum_{i=1}^{L} (k_i - 1) \cdot \prod_{j=1}^{i-1} s_j$$

其中 $k_i$ 是核大小，$s_j$ 是第 $j$ 层的步幅。

| 架构 | 机制 | 感受野增长 |
|------|------|-----------|
| 标准 CNN（3×3，步幅 1） | 池化使每阶段感受野翻倍 | 随深度指数增长 |
| 空洞 CNN（膨胀率 $r$） | 有效核 $k + (k-1)(r-1)$ | 随膨胀率线性增长 |
| Transformer（自注意力） | 从第 1 层起全局 | 立即覆盖全图 |
| SSM（Mamba） | 状态携带完整历史 | 全局，线性代价 |

**关键洞察**：小感受野捕获精细细节（边界、小病灶），大感受野捕获全局上下文（器官位置、空间关系）。最佳编码器通过多尺度特征层次同时提供两者。

### 2.3 CNN 编码器——视觉的归纳偏置

卷积神经网络编码了两个强大的归纳偏置：

1. **平移等变性**：卷积无论位置如何都检测到相同模式——左上角的肿瘤和右下角的肿瘤激活同一滤波器。
2. **局部性**：每次卷积只看一个局部邻域，非常适合捕获空间层次（边缘 → 纹理 → 部件 → 物体）。

**标准 CNN 骨干**：

| 架构 | 关键创新 | 为何有效 |
|------|----------|----------|
| ResNet (2015) | 残差连接 $y = F(x) + x$ | 解决梯度消失，使深层网络可行 |
| ConvNeXt (2022) | 现代化 CNN（大核、LayerNorm） | 以 CNN 效率匹敌 Transformer 精度 |
| EfficientNet (2019) | 复合缩放（深度 × 宽度 × 分辨率） | 最优资源分配 |
| MedNeXt (2023) | 大核 3D CNN 用于医学图像 | 大感受野且无注意力代价 |

ResNet 的残差连接值得特别关注。没有它，更深的网络反而表现*更差*（退化问题）。跳跃连接 $y = F(x) + x$ 确保每层只需学习*残差* $\Delta y = F(x)$，使优化变得极为容易：

```
输入 x → [Conv → BN → ReLU → Conv → BN] → F(x)
         ↓ 恒等映射                         ↓
         + → F(x) + x → ReLU → 输出
```

### 2.4 Transformer 编码器——通过注意力获取全局上下文

视觉 Transformer（ViT）用**自注意力**取代卷积，同时计算所有空间位置对之间的关系：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$ 是查询、键、值矩阵，$N = H \times W$ 是空间 token 数量。

**注意力为何适用于分割**：
- **从第 1 层起全局感受野**：每个 token 关注所有其他 token——无需堆叠层来增长感受野。
- **动态权重**：注意力权重依赖输入，不同于固定卷积滤波器。模型可以自适应地聚焦于相关区域。
- **长距离依赖**：对上下文重要的医学图像至关重要（如器官 A 的位置约束器官 B）。

**复杂度问题**：自注意力在内存和计算上都是 $O(N^2)$。对于 224×224 图像、patch 大小 16，$N = 196$（可管理）。对于 512×512、patch 大小 4，$N = 16384$（不可承受）。

| 设计策略 | 模型 | 复杂度 | 机制 |
|----------|------|--------|------|
| 全局注意力 | ViT, TransUNet 编码器 | $O(N^2)$ | 完整注意力 |
| 窗口注意力 | Swin Transformer, MaxViT | $O(N \cdot w^2)$ | 局部窗口内注意力 |
| 金字塔结构 | PVTv2, SegFormer | $O(N \cdot k)$ | 深层阶段空间缩减 |
| 混合 CNN+Transformer | TransUNet, MISSFormer | 仅瓶颈处 $O(N^2)$ | CNN 处理局部，Transformer 处理全局 |

### 2.5 状态空间模型（Mamba）——线性全局上下文

状态空间模型（SSM）提供了一个极具吸引力的替代方案：全局感受野，**线性** $O(N)$ 复杂度。

连续 SSM 通过隐藏状态 $h(t)$ 将输入 $x(t)$ 映射到输出 $y(t)$：

$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t)$$

其中 $A$ 是状态转移矩阵，$B$ 将输入投射到状态，$C$ 将状态投射到输出。

**离散化**：连续 SSM 使用零阶保持（ZOH）方法离散化：

$$\bar{A} = e^{\Delta A}, \quad \bar{B} = (\Delta A)^{-1}(e^{\Delta A} - I) \cdot \Delta B$$

递推形式为：$h_t = \bar{A} h_{t-1} + \bar{B} x_t$，$y_t = C h_t$。

**Mamba 的选择性扫描**：标准 SSM 与输入无关（所有输入使用相同的 $A, B, C$）。Mamba 使 $B, C, \Delta$ 依赖输入，允许模型根据输入内容选择性地记住或遗忘信息——类似于 LSTM 的门控机制，但支持并行训练。

| 属性 | Transformer | Mamba (SSM) |
|------|-------------|-------------|
| 感受野 | 全局（第 1 层） | 全局（通过状态） |
| 复杂度 | $O(N^2)$ | $O(N)$ |
| 训练 | 并行 | 并行（通过并行扫描） |
| 推理 | 每 token $O(N)$（KV 缓存） | 每 token $O(1)$（状态） |
| 最佳场景 | 中等分辨率 | 高分辨率（512+） |

### 2.6 RWKV——两全其美

RWKV（Receptance Weighted Key Value）通过两个交替模块结合了 Transformer 的并行训练优势与 RNN 的高效推理：

- **Time-Mix**：插值当前和前一 token 的特征——像 RNN 一样捕获时序依赖。
- **Channel-Mix**：施加逐点非线性——类似 Transformer 的 FFN。

$$\text{Time-Mix: } \quad r_t = \sigma(W_r \cdot (\mu_r \cdot x_t + (1-\mu_r) \cdot x_{t-1}))$$
$$\text{Channel-Mix: } \quad o_t = W_o \cdot \text{ReLU}(W_k \cdot (\mu_k \cdot x_t + (1-\mu_k) \cdot x_{t-1}))^2$$

训练时，RWKV 并行处理整个序列（类似 Transformer）。推理时，它维护固定大小的状态（类似 RNN），每 token 代价为 $O(1)$，与序列长度无关。

### 2.7 Foundation 模型编码器——迁移学习

Foundation 模型是在大规模无标注数据集上预训练的大型视觉 Transformer。其力量来自**迁移学习**：在多样数据上学到的特征可以泛化到下游医学任务。

**三种自监督学习范式**：

| 范式 | 方法 | 预训练任务 | 示例 |
|------|------|-----------|------|
| 对比学习 | DINO, CLIP | 拉近相似对，推远不相似对 | "这张视网膜图像与那张相似" |
| 掩码建模 | MAE, DINOv2 | 重建被遮挡的 patch | "预测这张图像被隐藏的 75%" |
| 生成式 | SAM | 从提示预测分割 | "给定点或框，分割任何目标" |

**Foundation 模型优于从头训练的原因**：

1. **规模**：在数百万图像上预训练（DINOv2：142M，CLIP：400M 文本-图像对）。
2. **多样性**：涵盖远超任何单一医学数据集的纹理、形状和模式。
3. **特征质量**：自监督目标迫使模型学习语义上有意义的表示。

---

## 3. 方法细节

### 3.1 编码器家族对比

| 家族 | 复杂度 | 感受野 | 归纳偏置 | 最佳场景 |
|------|--------|--------|----------|----------|
| CNN | $O(N \cdot k^2)$ | 随深度增长 | 平移、局部性 | 通用，小数据集 |
| Transformer | $O(N^2)$ | 全局，第 1 层 | 极少（数据驱动） | 中等分辨率，需全局上下文 |
| Mamba/SSM | $O(N)$ | 全局，通过状态 | 序列排序 | 高分辨率（512+），显存受限 |
| RWKV | $O(N)$ 训练，$O(1)$ 推理 | 全局，通过状态 | 时序混合 | 高效部署，长序列 |
| Foundation | $O(N^2)$ | 全局，第 1 层 | 预训练特征 | 标注数据有限，迁移学习 |

### 3.2 特征图兼容性

框架中所有编码器输出多尺度特征图列表，具有一致接口：

```python
# 编码器输出：从精细到粗糙的张量列表
features = encoder(x)
# features[0]: (B, C1, H/2, W/2)   ← 跳跃到解码器阶段 4
# features[1]: (B, C2, H/4, W/4)   ← 跳跃到解码器阶段 3
# features[2]: (B, C3, H/8, W/8)   ← 跳跃到解码器阶段 2
# features[3]: (B, C4, H/16, W/16) ← 跳跃到解码器阶段 1
# features[4]: (B, C5, H/32, W/32) ← 瓶颈输入
```

解码器从编码器读取 `encoder_channels = [C1, C2, C3, C4, C5]` 以匹配自身架构。这使得任何编码器可以与任何解码器自由组合。

### 3.3 何时使用哪种

| 场景 | 推荐 | 原因 |
|------|------|------|
| 快速基线，小数据 | CNN（ResNet50） | 训练快，强归纳偏置 |
| SOTA 精度，中等分辨率 | Transformer（Swin, PVTv2） | 全局上下文改善边界 |
| 高分辨率（512+） | Mamba（VMUNet, LKM） | 线性复杂度可扩展 |
| 标注数据有限 | Foundation（DINOv2, Phikon） | 来自预训练的迁移学习 |
| 边缘部署 | RWKV 或轻量 CNN | 低内存，快速推理 |
| 任意架构 | `timm_*` 封装器 | 1000+ 模型，零注册 |

### 3.4 动态 timm 编码器

`timm_` 前缀激活动态封装器，将 `timm` 库中任何模型变为分割编码器：

```
timm_ + 模型名称 → timm.create_model() → 钩取中间层 → 自动检测通道数
```

这之所以可行，是因为所有 `timm` 模型暴露 `feature_info` 元数据描述其中间特征图。无需手动注册。

---

## 4. 在 APRIL-MedSeg 中实践

```yaml
# CNN 编码器
model:
  encoder: { name: timm_resnet50, pretrained: true }
  decoder: { name: unet }

# Transformer 编码器
model:
  encoder: { name: timm_swin_tiny_patch4_window7_224, pretrained: true }
  decoder: { name: unet }

# Mamba 编码器
model:
  encoder: { name: vmunet_tiny, pretrained: false }
  decoder: { name: vm_unet }

# Foundation 编码器（自动下载权重）
model:
  encoder: { name: dinov2_base, pretrained: true, freeze: true }
  decoder: { name: unet }

# 任意 timm 模型
model:
  encoder: { name: timm_convnextv2_tiny, pretrained: true }
  decoder: { name: cascade }
```

---

## 5. 推荐实验

### 实验 1：编码器家族对比

使用相同解码器（UNet）和数据集，仅更换编码器：

| 编码器 | 家族 | 预期参数量 | 预期 Dice |
|--------|------|-----------|-----------|
| `timm_resnet50` | CNN | ~25M | 基线 |
| `timm_swin_tiny_patch4_window7_224` | Transformer | ~28M | +2-4% |
| `vmunet_tiny` | Mamba | ~22M | +1-3% |
| `dinov2_base`（冻结） | Foundation | ~86M（冻结） | +3-6% |

### 实验 2：分辨率缩放

在不同输入分辨率下测试同一编码器：

| 分辨率 | 速度 | 预期质量 |
|--------|------|----------|
| 128×128 | 4 倍快 | 较低（丢失细节） |
| 224×224 | 基线 | 良好平衡 |
| 512×512 | 4 倍慢 | 较高（Mamba/Transformer 受益最大） |

### 实验 3：冻结 vs 微调 Foundation

| 设置 | 可训练参数 | 预期行为 |
|------|-----------|----------|
| `freeze: true` | 仅解码器（~2M） | 快速，域相似时效果好 |
| `freeze: false` | 全部（~88M） | 慢，域差异大时最佳 |
| `unfreeze_last_n: 4` | 最后 4 个 block（~20M） | 速度与精度平衡 |

---

## 6. 延伸阅读

### 关键论文

| 论文 | 年份 | 会议 | 关键贡献 |
|------|------|------|----------|
| [ResNet](https://arxiv.org/abs/1512.03385) | 2015 | CVPR | 残差连接解决深层网络训练 |
| [ViT](https://arxiv.org/abs/2010.11929) | 2020 | ICLR | 纯 Transformer 用于视觉 |
| [Swin Transformer](https://arxiv.org/abs/2103.14030) | 2021 | ICCV | 窗口注意力，窗口内线性复杂度 |
| [PVTv2](https://arxiv.org/abs/2106.13797) | 2022 | IJCV | 金字塔视觉 Transformer，原生多尺度 |
| [Mamba](https://arxiv.org/abs/2312.00752) | 2023 | - | 选择性状态空间模型，线性复杂度 |
| [VM-UNet](https://arxiv.org/abs/2402.02991) | 2024 | - | 视觉 Mamba 用于医学分割 |
| [RWKV](https://arxiv.org/abs/2305.13048) | 2023 | - | 并行训练 + RNN 推理 |
| [DINOv2](https://arxiv.org/abs/2304.07193) | 2024 | - | 无需标签的自监督特征 |
| [ConvNeXt](https://arxiv.org/abs/2201.03545) | 2022 | CVPR | 现代化 CNN 匹敌 Transformer |

### 相关文档

- [编码器指南](../models/encoders.md) — 所有 176 个编码器及模型路径
- [timm 封装器](../models/encoders.md#timm-wrapper) — 动态编码器封装器用法
- [Foundation 模型](../models/encoders.md#foundation-models) — 9 个模态的 38 个 Foundation 编码器

---

[上一章：训练](04_training_CN.md) | [下一章：解码器](06_decoders_CN.md)
