# 第 08e 章：文本引导分割

[返回训练范式总览](08_paradigms_CN.md) | [English](08e_text_guided.md) | [上一章：弱监督](08d_weakly_supervised_CN.md) | [下一章：部署](09_deployment_CN.md)

---

## 1. 什么时候该用文本引导分割？

传统分割需要像素级 mask——必须有人描绘每个器官的边界。但如果只需用**自然语言描述**要分割什么呢？

> "分割这张 CT 扫描中的肝脏肿瘤。"
> "找出这张胸部 X 光片中所有肺实变区域。"

**文本引导分割**使用自然语言描述作为监督，利用 CLIP 等视觉语言模型（VLM）已学到的视觉和文本表示对齐。这开辟了全新的可能：

| 场景 | 传统方法 | 文本引导方法 |
|------|---------|-------------|
| 新器官，无训练数据 | 收集 500+ 标注扫描 | 描述它："CT 中的脾脏" |
| 零样本泛化 | 为每个新任务重新训练 | 改变文本提示 |
| 罕见发现 | 几乎不可能标注足够数据 | 描述："磨玻璃影" |
| 跨模态（CT → MRI） | 从头重新训练 | 相同文本提示，不同图像 |

---

## 2. 核心概念

### 2.1 视觉语言模型（CLIP）

文本引导分割的基础是 **CLIP**（对比语言-图像预训练），在 4 亿图像-文本对上训练。

CLIP 学习一个**共享嵌入空间**，其中图像及其文本描述彼此接近：

```
图像: [含肝脏的 CT]  ──▶ 图像编码器 ──▶ ┐
                                          ├── 余弦相似度 ≈ 0.85
文本: "CT 中的肝脏"   ──▶ 文本编码器 ──▶ ┘

图像: [含肝脏的 CT]  ──▶ 图像编码器 ──▶ ┐
                                          ├── 余弦相似度 ≈ 0.20
文本: "脑部 MRI 扫描"  ──▶ 文本编码器 ──▶ ┘
```

**对比训练目标：**

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_j \exp(\text{sim}(I_i, T_j)/\tau)} + \log \frac{\exp(\text{sim}(T_i, I_i)/\tau)}{\sum_j \exp(\text{sim}(T_i, I_j)/\tau)} \right]$$

这创建了对齐的表示："CT 扫描中的肝脏肿瘤"在几何上与实际肝脏肿瘤图像接近，与脑部 MRI 远离。

### 2.2 基于 CLIP 的分割（TextPromptUNet）

TextPromptUNet 方法使用 CLIP 的文本编码器生成类别嵌入，然后通过交叉注意力和特征调制将其集成到 UNet 中：

```
┌─────────────────────────────────────────────────────┐
│                   TextPromptUNet                     │
│                                                      │
│  "脾脏器官"    ──▶ CLIP 文本 ──▶ text_emb ──┐      │
│  "肝脏器官"    ──▶ 编码器   ─▶ text_emb ──┤      │
│  "肾脏器官"    ──▶          ─▶ text_emb ──┘      │
│                                                   │  │
│  CT 图像 ──▶ 编码器 ──▶ 特征 ──┐                │  │
│                                  │                │  │
│               交叉注意力 ◀──────┼────────────────┘  │
│               (图像作 query,    │                   │
│                文本作 key/value) │                   │
│                                  ▼                   │
│               FiLM 调制                             │
│               (用文本缩放 + 平移                     │
│                图像特征)                             │
│                                  ▼                   │
│               解码器 ──▶ 分割 Mask                   │
└─────────────────────────────────────────────────────┘
```

**交叉注意力**：图像特征关注文本嵌入——模型学习"当我看到这个视觉模式时，这个文本描述是相关的。"

**FiLM 调制**（特征级线性调制）：文本嵌入生成缩放（γ）和平移（β）参数来调制图像特征：

$$\text{FiLM}(F, \text{text}) = \gamma(\text{text}) \odot F + \beta(\text{text})$$

这让文本描述能够"调高"与所描述器官相关的特征，"调低"无关特征。

### 2.3 MLLM 管线（检测再分割）

更强大的方法使用**多模态大语言模型**（MLLM）作为定位检测器，后接专用分割器：

```
               第 1 步：定位                第 2 步：分割
               ┌──────────────┐           ┌──────────────┐
               │    MLLM      │           │   SAM2 /     │
"分割肝脏" ──▶│  (Qwen2-VL,  │ ──bbox──▶│   MedSAM     │ ──▶ mask
               │   InternVL,  │           │              │
               │  Grounding   │           │              │
               │  DINO)       │           │              │
               └──────────────┘           └──────────────┘
```

**为什么分两步？**
1. MLLM 擅长理解文本和定位物体，但产生粗糙的边界框——不是像素精确的 mask。
2. SAM2 / MedSAM 在给定提示（框或点）时擅长分割，但不理解文本。
3. 结合两者可以得到文本理解 + 精确分割。

**支持的 MLLM 定位器：**

| 定位器 | 模型 | 速度 | 精度 | 依赖 |
|--------|------|------|------|------|
| Grounding DINO | `tiny` / `large` | 快 | 好 | `groundingdino-py` |
| Qwen2-VL | 7B / 72B | 中 | 极好 | `transformers`, `qwen-vl-utils` |
| Qwen3-VL | 2B / 8B | 中 | 极好 | `transformers` |
| InternVL | 2.5 / 3 | 中 | 极好 | `transformers` |

**支持的 mask 生成器：**

| 生成器 | 速度 | 质量 | 依赖 |
|--------|------|------|------|
| SAM2 (Hiera-Large) | 中 | 极好 | `sam2` |
| MedSAM | 中 | 极好（医学） | 自定义 |
| SAMMed2D | 快 | 好 | 自定义 |

### 2.4 SemanticGuidedUNet

一种替代方法，使用**类别级语义嵌入**（不一定是 CLIP）通过多尺度注意力引导分割：

```
类别嵌入 ──▶ 多尺度注意力 ──▶ 引导特征 ──▶ 解码器 ──▶ Mask
                (在每个跳跃层级)
```

这比完整的 CLIP 集成更简单，但在有明确定义类别名时很有效。

---

## 3. 在 APRIL-MedSeg 中使用

### 3.1 两种方法

| 方法 | 脚本 | 适用场景 |
|------|------|---------|
| **基于训练**（TextPromptUNet） | `train_text_guided.py` | 有训练数据 + 想微调 |
| **基于管线**（MLLM + SAM2） | `test.py` / Python API | 零样本推理，无需训练 |

### 3.2 基于训练：TextPromptUNet

```bash
python train_text_guided.py \
    --config configs/training_paradigms/text_guided/synapse_clip.yaml \
    --output_dir output/text_guided
```

**YAML 配置逐行解读：**

```yaml
model:
  text_guided:
    model_type: TextPromptUNet          # 文本引导架构类型
    prompt_mode: clip                   # 使用 CLIP 文本编码器
    embed_dim: 512                      # CLIP 文本隐藏维度
    use_external_encoder: true
    # 自然语言类别描述
    # CLIP 对完整短语效果更好
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
    name: timm_vit_clip_base_p32_256   # CLIP ViT-B/32 权重（与文本对齐）
    pretrained: true
    in_channels: 3
    img_size: 256                      # 256 = 32×8，patch32 → 8×8 特征图
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

**文本提示的关键注意事项：**
- 使用完整短语：`"liver organ"` 比 `"liver"` 效果更好。
- 添加上下文：`"liver organ in CT scan"` 可以改善模态特定对齐。
- 保持一致：所有类别名应遵循相同模式。

### 3.3 基于管线：MLLM 检测再分割

无需训练——直接推理：

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
    enabled: false                     # 启用 UNet 精炼 SAM2 mask

data:
  type: synapse
  img_size: 1024
  test_dir: ./data/Synapse/test_vol_h5
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt
```

**Python API 用法：**

```python
import yaml
from medseg.inference.mllm import build_pipeline_from_config

# 加载配置
cfg = yaml.safe_load(open('configs/training_paradigms/text_guided/synapse_grounding_dino_sam2.yaml'))

# 构建管线
pipe = build_pipeline_from_config(cfg)

# 运行推理
result = pipe(image_rgb_uint8)    # 返回 PipelineOutput，含 label_map, per_class_masks

# 访问结果
mask = result.label_map           # (H, W) 整数 mask
per_class = result.per_class_masks  # (num_classes, H, W) 二值 mask
```

### 3.4 可用配置

| 配置 | 方法 | 模型 | 用途 |
|------|------|------|------|
| `synapse_clip.yaml` | 训练 | CLIP ViT + TextPromptUNet | 有训练数据时微调 |
| `synapse_grounding_dino_sam2.yaml` | 管线 | Grounding DINO + SAM2 | 零样本推理 |
| `synapse_grounding_dino_medsam.yaml` | 管线 | Grounding DINO + MedSAM | 医学优化零样本 |
| `synapse_qwen2vl_sam2.yaml` | 管线 | Qwen2-VL + SAM2 | 最佳精度（7B 模型） |
| `synapse_qwen3vl_medsam.yaml` | 管线 | Qwen3-VL + MedSAM | 最新 VLM + 医学 SAM |
| `synapse_internvl_sam2.yaml` | 管线 | InternVL + SAM2 | 替代 VLM |

---

## 4. 手把手：你的第一次文本引导训练

### 方案 A：零样本推理（最简单）

如果只想分割器官而不训练：

```bash
# 安装依赖
pip install groundingdino-py
pip install git+https://github.com/facebookresearch/sam2.git

# 运行推理
python test.py --config configs/training_paradigms/text_guided/synapse_grounding_dino_sam2.yaml
```

### 方案 B：基于训练（更好精度）

如果有训练数据并想微调：

**第 1 步**：为每个类别准备文本描述。

```yaml
class_names:
  - background region
  - spleen organ in CT scan
  - right kidney organ in CT scan
  - liver organ in CT scan
```

**第 2 步**：训练。

```bash
python train_text_guided.py \
    --config configs/training_paradigms/text_guided/synapse_clip.yaml \
    --output_dir output/text_guided
```

**第 3 步**：评估。

```bash
python test.py \
    --config configs/training_paradigms/text_guided/synapse_clip.yaml \
    --checkpoint output/text_guided/best_model.pth
```

---

## 5. 参数调优指南

### 基于 CLIP 的训练

| 参数 | 效果 | 调优建议 |
|------|------|---------|
| `embed_dim` | 文本嵌入维度 | 匹配 CLIP 模型：ViT-B 用 512，ViT-L 用 768 |
| `prompt_mode` | 文本编码方式 | `clip` 用 CLIP 编码器；`learnable` 用可学习嵌入 |
| `class_names` | 文本描述 | 使用描述性短语，不要用单个词 |
| `img_size` | 输入分辨率 | 须匹配 CLIP 编码器（ViT-B/32 用 256） |

### MLLM 管线

| 参数 | 效果 | 调优建议 |
|------|------|---------|
| `box_threshold` | 检测置信度阈值 | 0.3–0.4。更低 = 更多检测（更多误检）。更高 = 更少但更自信。 |
| `text_threshold` | 文本匹配阈值 | 0.2–0.3。更低 = 更宽松的文本匹配。 |
| `prompt_template` | 定位提示格式 | `"a medical CT image of {class_name}"` 对 CT 效果好。 |
| `multimask` | SAM 是否返回多个 mask | `false` 返回单个最佳 mask；`true` 从 3 个候选中选。 |

### 选择正确的 MLLM

| 需求 | 推荐定位器 | 推荐 Mask 生成器 |
|------|-----------|-----------------|
| 最快推理 | Grounding DINO (tiny) | SAMMed2D |
| 最佳精度 | Qwen2-VL (7B) | SAM2 (Hiera-Large) |
| 医学优化 | Grounding DINO | MedSAM |
| 最小显存 | Grounding DINO (tiny) | SAMMed2D |

---

## 6. 常见坑

### 坑 1：CLIP 文本描述匹配不好

**症状**：模型混淆相似器官（如左肾 vs. 右肾）。

**修复**：
- 使用更有区分度的描述：`"left kidney organ on the patient's left side"` 而非仅 `"left kidney"`。
- 添加空间上下文：`"spleen organ in upper left abdomen"`。
- 增加训练 epoch 让交叉注意力学习更好的对齐。

### 坑 2：MLLM 漏检小器官

**症状**：Grounding DINO 遗漏小结构如胆囊或食管。

**修复**：
- 对小结构降低 `box_threshold` 到 0.2–0.25。
- 使用更大的 Grounding DINO 模型（`large` 而非 `tiny`）。
- 尝试不同的 MLLM（Qwen2-VL 或 InternVL 处理小目标更好）。

### 坑 3：SAM2 在医学图像上产生差的 mask

**症状**：SAM2 分割了错误区域或产生碎片化 mask。

**修复**：
- 改用 MedSAM——它专门在医学图像上训练。
- 调整框提示：在定位框周围添加填充。
- 启用 UNet 精炼（`refinement.enabled: true`）来清理 SAM2 输出。

### 坑 4：大型 MLLM 显存占用高

**症状**：使用 Qwen2-VL 7B 或 InternVL 时 CUDA OOM。

**修复**：
- 使用 4-bit 量化：设置 `dtype: float16` 或 `dtype: bfloat16`。
- 使用 Grounding DINO (tiny) 作为定位器——它小得多。
- 降低 `img_size` 到 512 或 256。

### 坑 5：TextPromptUNet 忽略文本

**症状**：模型无论输入什么文本都预测相同的 mask。

**修复**：
- 确保 CLIP 编码器权重实际被加载（`use_external_encoder: true`）。
- 检查 `embed_dim` 是否与文本编码器输出维度匹配。
- 增大交叉注意力层的学习率。
- 验证不同文本输入是否产生不同的嵌入。

---

## 7. 推荐实验

### 实验 1：零样本对比

在相同测试集上比较不同 MLLM 管线：

```bash
# Grounding DINO + SAM2
python test.py --config configs/training_paradigms/text_guided/synapse_grounding_dino_sam2.yaml

# Grounding DINO + MedSAM
python test.py --config configs/training_paradigms/text_guided/synapse_grounding_dino_medsam.yaml

# Qwen2-VL + SAM2
python test.py --config configs/training_paradigms/text_guided/synapse_qwen2vl_sam2.yaml
```

**预期结果：**

| 管线 | Dice | 推理时间 | 显存 |
|------|------|---------|------|
| GDINO + SAM2 | 55–70% | 2–4 秒/张 | ~4GB |
| GDINO + MedSAM | 60–75% | 2–4 秒/张 | ~4GB |
| Qwen2-VL + SAM2 | 65–78% | 5–10 秒/张 | ~12GB |
| 训练的 TextPromptUNet | 70–82% | 0.1 秒/张 | ~2GB |

### 实验 2：文本提示工程

变化文本描述并衡量影响：

| 提示风格 | 示例 | 预期 Dice 变化 |
|---------|------|---------------|
| 单个词 | "liver" | 基线 |
| 描述性 | "liver organ" | +2–5% |
| 上下文 | "liver organ in CT scan" | +3–7% |
| 空间 | "liver organ in right upper abdomen" | +2–4% vs 上下文 |

### 实验 3：训练 vs. 零样本

```bash
# 零样本（无训练）
python test.py --config configs/training_paradigms/text_guided/synapse_grounding_dino_medsam.yaml

# 基于训练
python train_text_guided.py --config configs/training_paradigms/text_guided/synapse_clip.yaml
python test.py --config configs/training_paradigms/text_guided/synapse_clip.yaml \
    --checkpoint output/text_guided/best_model.pth
```

训练后的模型应比零样本高 10–20% Dice，但需要训练数据。

---

## 8. 延伸阅读

### 关键论文

| 论文 | 年份 | 核心思想 |
|------|------|---------|
| [CLIP](https://arxiv.org/abs/2103.00020) | 2021 | 视觉语言对比预训练 |
| [CRIS](https://arxiv.org/abs/2211.10961) | 2023 | 通过 CLIP 的文本引导医学分割 |
| [Grounding DINO](https://arxiv.org/abs/2303.05499) | 2023 | 开放集文本定位检测 |
| [SAM2](https://arxiv.org/abs/2408.00714) | 2024 | Segment Anything 模型 v2 |
| [MedSAM](https://arxiv.org/abs/2304.12306) | 2023 | 医学适配的 SAM |
| [BiomedParse](https://arxiv.org/abs/2305.09860) | 2024 | 统一生物医学解析 |

### 相关文档

- [所有文本引导模型](../paradigms/text_guided.md) — 完整模型目录（12 个模型 + MLLM 管线）
- [MLLM 推理指南](../deployment/mllm_inference.md) — 详细 MLLM 管线文档

---

[返回训练范式总览](08_paradigms_CN.md) | [上一章：弱监督](08d_weakly_supervised_CN.md) | [下一章：部署](09_deployment_CN.md)
