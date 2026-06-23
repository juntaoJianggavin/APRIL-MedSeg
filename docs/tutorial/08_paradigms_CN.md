# 第 08 章：高级训练范式 — 总览

[上一章：Foundation 模型](07_foundation_CN.md) | [English](08_paradigms.md) | [下一章：部署](09_deployment_CN.md)

---

## 为什么需要超越监督学习？

标准监督训练——为每张输入图像配对像素级真值 mask——是最简单的学习范式，但对数据的要求也最为苛刻。在医学影像中，这造成了几个实际瓶颈：

- **标注稀缺**：专家标注成本高昂。单张 CT 扫描可能需要放射科医师 30 分钟以上才能标注完成。
- **域偏移**：在扫描仪 A 的图像上训练的模型可能因采集协议差异而在扫描仪 B 上失败。
- **模型规模**：最先进的模型体积庞大——在边缘设备上部署需要压缩且不损失精度。
- **粗糙标注**：通常只有边界框、点或图像级标签可用，而非像素级 mask。

五种高级范式针对这些挑战，各有其特定目标。

---

## 五大范式

每种范式都有独立的详细教程。点击进入学习理论、配置和实操练习。

### [08a：半监督学习](08a_semi_supervised_CN.md)

**问题**：图像很多但标注很少。

**方案**：通过一致性正则化和伪标签，将无标注数据与有标注数据一起使用。

| 关键方法 | 核心思想 | 预期收益 |
|---------|---------|---------|
| Mean Teacher | EMA 教师提供稳定目标 | 10% 标注时达到全监督的 85–95% |
| CPS | 双网络交叉伪监督 | 全监督的 85–93% |
| UniMatch | 弱→强增强一致性 | 最佳单模型性能 |

**脚本**：`semi_train.py` · **配置**：`configs/training_paradigms/semi_supervision/`

---

### [08b：域适应](08b_domain_adaptation_CN.md)

**问题**：模型在扫描仪 A 上效果好但在扫描仪 B 上失败。

**方案**：对齐源域和目标域特征分布，或在测试时适应。

| 关键方法 | 核心思想 | 预期收益 |
|---------|---------|---------|
| AdvEnt | 对抗熵最小化 | 目标域 +8–15% |
| DANN | 梯度反转获取域不变特征 | 目标域 +5–12% |
| TENT | 测试时 BatchNorm 适应（无需源数据） | 目标域 +5–10% |

**脚本**：`train_domain_adaptation.py` · **配置**：`configs/training_paradigms/domain_adaptation/`

---

### [08c：知识蒸馏](08c_distillation_CN.md)

**问题**：最好的模型太大，无法部署。

**方案**：通过软标签将大教师的知识迁移到小学生。

| 关键方法 | 核心思想 | 预期收益 |
|---------|---------|---------|
| Hinton KD | 匹配软化输出分布 | 教师精度的 90–95% |
| CWD | 逐通道特征蒸馏 | 教师精度的 93–97% |
| DKD | 解耦目标/非目标蒸馏 | 教师精度的 94–98% |

**脚本**：`train_distillation.py` · **配置**：`configs/training_paradigms/distillation/`

---

### [08d：弱监督学习](08d_weakly_supervised_CN.md)

**问题**：只有边界框、点或图像级标签——没有像素级 mask。

**方案**：通过 CAM、框监督或特征传播从粗糙标注训练。

| 关键方法 | 标注类型 | 预期收益 |
|---------|---------|---------|
| Box Supervised | 边界框 | 全监督的 75–90% |
| CAM / SEAM | 图像级标签 | 全监督的 60–75% |
| Point / Scribble | 稀疏点或线条 | 全监督的 60–85% |

**脚本**：`train_weakly_supervised.py` · **配置**：`configs/training_paradigms/weak_supervision/`

---

### [08e：文本引导分割](08e_text_guided_CN.md)

**问题**：想分割新结构但不想收集标注数据。

**方案**：通过视觉语言模型（CLIP、MLLM）使用自然语言描述作为监督。

| 关键方法 | 核心思想 | 预期收益 |
|---------|---------|---------|
| TextPromptUNet | CLIP 文本嵌入引导 UNet | 有训练数据时 70–82% Dice |
| MLLM + SAM2 | 检测再分割管线 | 零样本 55–78% Dice |
| SemanticGuidedUNet | 类别嵌入 + 多尺度注意力 | 有训练数据时 65–78% Dice |

**脚本**：`train_text_guided.py` / `test.py` · **配置**：`configs/training_paradigms/text_guided/`

---

## 快速对比

| 范式 | 所需数据 | 标注成本 | 典型提升 | 脚本 |
|------|---------|---------|---------|------|
| 监督（基线） | 100% 标注 | 最高 | 基线 | `train.py` |
| [半监督](08a_semi_supervised_CN.md) | 10% 标注 + 90% 无标注 | 低 | 全监督的 80–95% | `semi_train.py` |
| [域适应](08b_domain_adaptation_CN.md) | 源域标注 + 目标域无标注 | 中 | 目标域 +5–15% | `train_domain_adaptation.py` |
| [蒸馏](08c_distillation_CN.md) | 教师模型 + 标注数据 | 同监督 | 教师精度的 90–98% | `train_distillation.py` |
| [弱监督](08d_weakly_supervised_CN.md) | 框/点/图像标签 | 低 | 全监督的 75–90% | `train_weakly_supervised.py` |
| [文本引导](08e_text_guided_CN.md) | 文本提示 | 最低 | 变化大（零样本：40–70%） | `train_text_guided.py` |

---

## 如何选择？

```
你有所有训练数据的像素级 mask 吗？
├── 是 → 模型对部署来说太大了吗？
│        ├── 是 → 知识蒸馏 (08c)
│        └── 否 → 标准监督训练 (train.py)
│
└── 否 → 你有什么标注？
         ├── 部分像素 mask + 大量无标注图像 → 半监督 (08a)
         ├── 有标注源域 + 无标注目标域 → 域适应 (08b)
         ├── 边界框、点或图像级标签 → 弱监督 (08d)
         └── 只有文本描述 → 文本引导 (08e)
```

---

## 关键论文（所有范式）

| 论文 | 年份 | 会议 | 范式 | 关键贡献 |
|------|------|------|------|---------|
| [Mean Teacher](https://arxiv.org/abs/1703.01780) | 2017 | NeurIPS | 半监督 | EMA 教师用于一致性正则化 |
| [CPS](https://arxiv.org/abs/2106.01226) | 2021 | CVPR | 半监督 | 双网络交叉伪监督 |
| [DANN](https://arxiv.org/abs/1505.07818) | 2016 | JMLR | 域适应 | 域对抗训练 |
| [AdvEnt](https://arxiv.org/abs/1811.12833) | 2019 | CVPR | 域适应 | 对抗熵最小化 |
| [TENT](https://arxiv.org/abs/2006.10726) | 2021 | ICLR | 域适应 | 测试时熵最小化 |
| [Hinton KD](https://arxiv.org/abs/1503.02531) | 2015 | NeurIPS WS | 蒸馏 | 基于温度的知识蒸馏 |
| [DKD](https://arxiv.org/abs/2203.08679) | 2022 | CVPR | 蒸馏 | 解耦知识蒸馏 |
| [SEAM](https://arxiv.org/abs/2003.13053) | 2020 | CVPR | 弱监督 | CAM 的自监督等变注意力 |
| [CLIP](https://arxiv.org/abs/2103.00020) | 2021 | ICML | 文本 | 视觉语言对比预训练 |
| [CRIS](https://arxiv.org/abs/2211.10961) | 2023 | — | 文本 | 通过 CLIP 的文本引导医学分割 |

---

## 相关文档

- [半监督方法](../paradigms/semi_supervised.md) — 所有 20 个半监督方法
- [域适应](../paradigms/domain_adaptation.md) — 所有 18 个域适应方法
- [蒸馏](../paradigms/distillation.md) — 所有 27 个蒸馏方法
- [弱监督](../paradigms/weakly_supervised.md) — 所有 20 个弱监督方法
- [文本引导](../paradigms/text_guided.md) — 所有 12 个文本引导模型 + MLLM 管线

---

[上一章：Foundation 模型](07_foundation_CN.md) | [下一章：部署](09_deployment_CN.md)
