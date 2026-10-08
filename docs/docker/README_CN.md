# Docker

[English](README.md)

Docker 镜像提供 APRIL-MedSeg 的基础运行环境，包含 `requirements.txt` 中的核心依赖，
以及组件注册阶段所需的 Transformers 4.x。Mamba/SSM、Foundation/MLLM 模型专属依赖
和 ONNX 等可选依赖需要按所用模型安装，因为它们对 CUDA、PyTorch 和相关库的要求并
不完全相同。镜像使用 `opencv-python-headless` 替代 `opencv-python`，保留相同的图像
处理 API，同时避免安装桌面 GUI 依赖。

## 宿主机要求

- Linux x86_64
- NVIDIA GPU，驱动需支持 CUDA 12.4
- Docker Engine，并已配置 NVIDIA Container Toolkit
- 系统内存至少 16 GB，训练建议 32 GB 或以上
- 显存取决于模型、输入尺寸和 batch size：轻量推理可从 8 GB 起步，常规训练建议
  16 GB 或以上，大型 Foundation/MLLM 模型通常需要 24 GB 或以上
- 镜像、依赖和模型缓存建议预留至少 30 GB；数据集和 checkpoint 需要额外空间

宿主机只需要兼容的 NVIDIA 驱动，CUDA Runtime 已包含在镜像中。Apple Silicon 和
纯 CPU 主机无法运行这个 CUDA 镜像。

构建前先验证宿主机的 GPU 容器环境：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 构建与验证

在仓库根目录执行：

```bash
docker build -t april-medseg:base .

docker run --rm --gpus all april-medseg:base \
  python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name()); assert torch.cuda.is_available()"
```

默认基础镜像为 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`。如需替换，应选择
满足 `requirements.txt` 版本要求的 PyTorch 镜像：

```bash
docker build \
  --build-arg PYTORCH_IMAGE=pytorch/pytorch:<tag> \
  -t april-medseg:base .
```

## 训练

数据集、输出和下载的权重应保存在镜像外。目录挂载使用宿主机绝对路径，模型缓存可
使用 Docker volume 复用：

```bash
mkdir -p /absolute/path/to/output

docker run --rm --gpus all --ipc=host \
  -v /absolute/path/to/data:/workspace/APRIL-MedSeg/data:ro \
  -v /absolute/path/to/output:/workspace/APRIL-MedSeg/output \
  -v april-medseg-cache:/root/.cache \
  april-medseg:base \
  python train.py \
    --config configs/default.yaml \
    --output_dir output/default \
    --amp
```

使用自定义 YAML 时，将配置文件以只读方式挂载，并把容器内路径传给 `--config`。

## 测试

```bash
mkdir -p /absolute/path/to/test-output

docker run --rm --gpus all --ipc=host \
  -v /absolute/path/to/data:/workspace/APRIL-MedSeg/data:ro \
  -v /absolute/path/to/checkpoints:/workspace/APRIL-MedSeg/checkpoints:ro \
  -v /absolute/path/to/test-output:/workspace/APRIL-MedSeg/test_output \
  -v april-medseg-cache:/root/.cache \
  april-medseg:base \
  python test.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output_dir test_output/default \
    --save_pred
```

## 可选模型依赖

基础镜像不会安装全部可选依赖：

| 模型类别 | 附加依赖 | 说明 |
|---|---|---|
| Foundation 编码器 | `open_clip_torch`、`sentencepiece` 等模型专属依赖 | 基础镜像已包含 Transformers 4.x 和 `safetensors`，其余依赖按模型安装。 |
| MLLM Pipeline | `accelerate`、`qwen-vl-utils` 等模型专属依赖 | Qwen-VL、InternVL、LLaVA 等要求不同。 |
| Mamba / SSM | `causal-conv1d`、`mamba-ssm` | 需要匹配 PyTorch/CUDA 工具链，通常应使用 `devel` 基础镜像。 |
| ONNX | `onnx`、`onnxruntime` 或 `onnxruntime-gpu` | 根据目标部署环境选择。 |

建议针对经过验证的模型类别构建派生镜像，不要在每次容器启动时临时安装依赖。数据集、
checkpoint 和访问令牌不应写入镜像。
