# 第 08d 章：弱监督学习

[返回训练范式总览](08_paradigms_CN.md) | [English](08d_weakly_supervised.md) | [上一章：知识蒸馏](08c_distillation_CN.md) | [下一章：文本引导](08e_text_guided_CN.md)

---

## 1. 什么时候该用弱监督学习？

像素级标注是分割的金标准，但并非总能获得。在许多临床场景中，你有的是：

- **图像级标签**：放射科医师标记扫描为"含有肿瘤"但没有画边界。
- **边界框**：标注者在器官周围画了一个矩形，但没有描绘精确轮廓。
- **点**：几次点击标记"这是肝脏，这是背景"——比描绘便宜得多。
- **涂鸦**：在器官部分区域上的快速手绘线条。

**弱监督分割**从这些粗糙标注训练出密集的像素级预测。成本节省是巨大的：一个边界框只需 ~5 秒画完；一个像素级 mask 需要 10–30 分钟。

### 标注成本对比

| 标注类型 | 每张图像耗时 | 每 1000 张成本 | Dice（vs. 全监督） |
|---------|------------|--------------|-------------------|
| 像素级 mask | 10–30 分钟 | $5,000–$15,000 | 100%（基线） |
| 边界框 | 5–10 秒 | $50–$100 | 75–90% |
| 涂鸦 | 15–30 秒 | $150–$300 | 70–85% |
| 点（每类 3–5 个） | 3–5 秒 | $30–$60 | 60–75% |
| 图像级标签 | 1–2 秒 | $10–$30 | 50–70% |

---

## 2. 核心概念

### 2.1 根本挑战

仅有粗糙监督时，模型如何学习密集预测？关键挑战是**信息缺口**：边界框告诉你器官在框*里面的某个地方*，但不是*哪些精确像素*属于它。

模型必须使用**先验知识**（从数据结构中学到的）和**正则化**（平滑约束、空间一致性）来弥合这个缺口。

### 2.2 基于 CAM 的方法（类别激活图）

最简单的方法：使用**分类网络**并从其最后一个卷积层提取空间激活模式。

**CAM 工作原理：**

$$M_c(x, y) = \sum_k w_k^c \cdot f_k(x, y)$$

其中 $f_k$ 是第 $k$ 个特征图，$w_k^c$ 是类别 $c$ 的分类器权重。产生的 $M_c$ 是一个热力图，突出显示模型预测类别 $c$ 时"看"的位置。

```
输入图像 ──▶ CNN 骨干 ──▶ 特征图 ──▶ 全局平均池化 ──▶ 分类器 ──▶ "肝脏"
                  │                          │
                  └── 用分类器权重加权求和 ──▶ CAM 热力图
```

**CAM 的问题**：CAM 仅突出**最具判别性**的区域，而非完整物体。对于肝脏，CAM 可能在肝脏边缘（最具区别性的特征）强烈激活，但遗漏平滑的内部。

**解决方案：**

| 方法 | 如何修复 CAM | 核心思想 |
|------|-------------|---------|
| SEAM | 自监督精炼 | 使用空间一致性扩展 CAM |
| AdvCAM | 对抗训练 | 发现非判别区域 |
| PuzzleCAM | Patch 打乱 | 迫使模型识别所有部分 |
| LPCAM | 可学习扰动 | 优化扰动以找到完整范围 |

### 2.3 框监督分割

给定边界框，模型知道器官在框内但不知道精确边界。训练策略：

1. **受限损失**：仅对框*内*的像素计算分割损失。
2. **背景先验**：远离任何框的像素很可能是背景。
3. **边界精炼**：使用边缘感知后处理（CRF、分水岭）让预测贴合实际器官边界。

$$\mathcal{L}_{\text{box}} = \sum_{(i,j) \in \text{box}} \mathcal{L}_{\text{pixel}}(y_{ij}, \hat{y}_{ij}) \cdot \text{mask}_{ij}$$

**BoxInst** 通过添加成对损失来扩展此方法，鼓励颜色相似的邻近像素具有相同标签——即使在框外。

### 2.4 点监督分割

更稀疏——每类仅有几个标注点。模型必须使用**学到的特征相似度**将这些点标签传播到完整区域。

```
       ·  (肝脏点)
      / \
     /   \    特征相似度
    /     \   传播
   ·───────· (具有相似特征的未标注像素 → 也是肝脏)
```

### 2.5 涂鸦监督分割

涂鸦提供了沿器官的窄条标注像素。模型使用以下方式填补间隙：
- **随机游走**：沿具有相似特征的像素传播标签。
- **CRF 正则化**：鼓励尊重图像边缘的平滑预测。

### 2.6 方法对比

| 方法 | 标注类型 | 方法类型 | 典型 Dice |
|------|---------|---------|----------|
| CAM | 图像级 | 分类 → 激活图 | 50–65% |
| SEAM/AdvCAM | 图像级 | 精炼 CAM | 60–75% |
| BoxSupervised | 边界框 | 受限损失 + 精炼 | 75–90% |
| BoxInst | 边界框 | 成对损失 + 框监督 | 78–88% |
| Point | 点 | 特征相似度传播 | 60–75% |
| Scribble | 涂鸦 | 随机游走 + CRF | 70–85% |

---

## 3. 在 APRIL-MedSeg 中使用

### 3.1 训练脚本

所有弱监督方法都使用 `train_weakly_supervised.py`：

```bash
# 框监督
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml \
    --supervision_type box

# 基于 CAM
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/cam.yaml \
    --supervision_type cam

# 多实例学习
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/mil.yaml \
    --supervision_type mil
```

### 3.2 YAML 配置 — 框监督

```yaml
model:
  num_classes: 9            # Synapse 数据集：8 个器官 + 背景
  img_size: 224
  encoder:
    name: timm_resnet50     # 更强的编码器补偿弱监督
    pretrained: true
    in_channels: 3
  decoder:
    name: unet              # 完整 UNet 解码器做密集预测
    params: {}
  bottleneck:
    name: none
    params: {}

data:
  img_size: 224
  image_dir: ./data/images
  annotation_file: ./data/annotations/boxes.json   # 框标注（JSON 格式）
  val:
    image_dir: ./data/val/images
    mask_dir: ./data/val/masks                      # 验证用完整 mask
  test:
    image_dir: ./data/test/images
    mask_dir: ./data/test/masks

weak_supervision:
  method: box_supervised
  params:
    box_penalty: 0.1              # 框外预测的惩罚
    refine_iterations: 3          # CRF 精炼迭代次数

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
    lr: 0.0002                    # 弱监督用稍高的学习率
  scheduler:
    name: cosine
    min_lr: 1e-6
```

### 3.3 YAML 配置 — 基于 CAM

```yaml
data:
  image_dir: ./data/images
  label_file: ./data/annotations/image_labels.json   # 仅图像级标签
  cam_dir: ./data/cams                                # 预计算的 CAM（可选）
  val:
    image_dir: ./data/val/images
    mask_dir: ./data/val/masks

weak_supervision:
  method: cam
  params:
    cam_threshold: 0.5           # CAM → 伪 mask 的二值化阈值
    refine: true                 # 对 CAM 施加 CRF 精炼
```

### 3.4 可用方法一览

| 方法 | 配置文件 | 标注类型 | 复杂度 |
|------|---------|---------|--------|
| Box Supervised | `box_supervised.yaml` | 边界框 | 低 |
| CAM | `cam.yaml` | 图像级标签 | 中 |
| MIL | `mil.yaml` | 图像级（bag） | 中 |
| SEAM | `semples.yaml` | 图像级 + 精炼 | 高 |
| Point | `point.yaml` | 稀疏点 | 低 |
| Scribble | `scribble_sup.yaml` | 手绘涂鸦 | 低 |
| BoxInst | `boxinst.yaml` | 边界框 | 中 |

### 3.5 数据准备

#### 框标注格式

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

每个边界框为 `[x_min, y_min, x_max, y_max]`，像素坐标。

#### 图像级标签格式

```json
{
  "image_001.npy": {"labels": ["liver", "spleen", "right_kidney"]},
  "image_002.npy": {"labels": ["liver", "stomach"]}
}
```

---

## 4. 手把手：你的第一次弱监督训练

### 第 1 步：准备弱标注

对于**框监督**（推荐起点）：
- 使用标注工具（Label Studio、CVAT）在器官周围画边界框。
- 导出为上述 JSON 格式。

对于**基于 CAM**：
- 准备图像级标签（每张图像中存在哪些类别）。
- 可选：用分类模型预计算 CAM。

### 第 2 步：选择方法

首次运行推荐**框监督**——标注成本与精度的最佳权衡。

### 第 3 步：训练

```bash
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml \
    --supervision_type box \
    --output_dir output/weak_box
```

### 第 4 步：与全监督对比评估

```bash
python test.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml \
    --checkpoint output/weak_box/best_model.pth
```

与全监督基线对比以衡量差距。

---

## 5. 参数调优指南

### 框监督参数

| 参数 | 效果 | 调优建议 |
|------|------|---------|
| `box_penalty` | 框外预测的惩罚力度 | 0.1 是好起点。如果预测泄漏到框外，增大。 |
| `refine_iterations` | CRF 后处理次数 | 3 是典型值。更多迭代 = 更锐利的边界但更慢。 |

### CAM 参数

| 参数 | 效果 | 调优建议 |
|------|------|---------|
| `cam_threshold` | CAM → 伪 mask 的激活阈值 | 0.3–0.5。更低 = 更多覆盖但更多噪声。更高 = 更少覆盖但更干净。 |
| `refine` | 是否施加 CRF 后处理 | 始终 `true` 以获得更好结果。 |

### 通用建议

- **使用比全监督更强的编码器**——模型需要更多容量来补偿弱标签。
- **训练更久**——弱监督模型收敛更慢，因为训练信号噪声更大。
- **用完整 mask 验证**——始终在有像素级 ground truth 的保留集上评估。

---

## 6. 常见坑

### 坑 1：CAM 仅突出物体边缘

**症状**：预测 mask 仅覆盖器官边界，遗漏内部。

**修复**：
- 降低 `cam_threshold` 以包含更多激活图。
- 启用 CRF 精炼（`refine: true`）。
- 使用 SEAM 或 AdvCAM 代替原始 CAM——它们明确解决此问题。

### 坑 2：框监督模型预测矩形 mask

**症状**：预测看起来像框而非器官形状。

**修复**：
- 增大 `refine_iterations` 以鼓励边界贴合。
- 添加边缘感知正则化。
- 确保训练数据有多样化的框大小和位置。

### 坑 3：图像级标签产生非常粗糙的 mask

**症状**：Dice 低于 60%，mask 呈团块状。

**修复**：
- 基于 CAM 的方法有固有的精度上限。考虑至少收集边界框标注。
- 使用两阶段方法：先生成 CAM，然后用 CAM 训练的分割器精炼。
- 添加自训练：用初始 CAM 训练分割器，然后迭代。

### 坑 4：验证 Dice 远高于测试 Dice

**症状**：过拟合到验证集的标注风格。

**修复**：
- 确保验证集和测试集来自同一分布。
- 基于验证 Dice 使用早停。
- 添加更强的数据增强。

---

## 7. 推荐实验

### 实验 1：标注类型对比

相同数据集，不同标注类型：

```bash
# 框监督
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml \
    --supervision_type box

# 基于 CAM
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/cam.yaml \
    --supervision_type cam

# 点监督
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/point.yaml \
    --supervision_type point
```

**预期结果：**

| 标注类型 | 标注时间 | Dice | 占全监督百分比 |
|---------|---------|------|--------------|
| 完整 mask（基线） | 20 分钟/张 | 88% | 100% |
| 边界框 | 8 秒/张 | 82% | 93% |
| 涂鸦 | 20 秒/张 | 77% | 88% |
| 点（5 个/类） | 5 秒/张 | 68% | 77% |
| 图像级标签 | 2 秒/张 | 62% | 70% |

### 实验 2：框监督精炼效果

变化 CRF 精炼迭代次数：

| 精炼迭代 | Dice | 边界精度 |
|---------|------|---------|
| 0（无精炼） | 75% | 差（框状） |
| 1 | 79% | 一般 |
| 3（默认） | 82% | 好 |
| 5 | 83% | 好（收益递减） |

---

## 8. 延伸阅读

### 关键论文

| 论文 | 年份 | 会议 | 核心思想 |
|------|------|------|---------|
| [CAM](https://arxiv.org/abs/1512.04150) | 2016 | CVPR | 从分类网络生成类别激活图 |
| [SEAM](https://arxiv.org/abs/2003.13053) | 2020 | CVPR | 自监督等变 CAM 精炼 |
| [BoxInst](https://arxiv.org/abs/2012.02646) | 2021 | CVPR | 框监督实例分割 |
| [AdvCAM](https://arxiv.org/abs/2104.00200) | 2021 | CVPR | 对抗 CAM 发现非判别区域 |

### 相关文档

- [所有弱监督方法](../paradigms/weakly_supervised.md) — 完整方法目录（20 个方法）

---

[返回训练范式总览](08_paradigms_CN.md) | [上一章：知识蒸馏](08c_distillation_CN.md) | [下一章：文本引导](08e_text_guided_CN.md)
