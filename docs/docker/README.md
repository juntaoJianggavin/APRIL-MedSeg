# Docker

[中文文档](README_CN.md)

The Docker image provides the core APRIL-MedSeg environment. It includes the
dependencies in `requirements.txt` plus Transformers 4.x, which the component
registry needs during package import. Optional Mamba/SSM, model-specific
foundation/MLLM, and ONNX dependencies remain opt-in because their CUDA and
library requirements vary by model. The image substitutes
`opencv-python-headless` for `opencv-python`, which provides the same
image-processing API without desktop GUI libraries.

## Host requirements

- Linux x86_64 host
- NVIDIA GPU and a driver compatible with CUDA 12.4
- Docker Engine with NVIDIA Container Toolkit configured
- At least 16 GB system RAM; 32 GB or more is recommended for training
- GPU memory depends on the model, input size, and batch size. Start with 8 GB
  for lightweight inference, 16 GB or more for training, and 24 GB or more for
  large foundation/MLLM models.
- Allow at least 30 GB of free disk space for the image and dependency/model
  caches. Datasets and checkpoints require additional space.

The host only needs a compatible NVIDIA driver; the CUDA runtime is included in
the image. Apple Silicon and CPU-only hosts cannot run this CUDA image.

Verify the host GPU runtime before building:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Build and verify

From the repository root:

```bash
docker build -t april-medseg:base .

docker run --rm --gpus all april-medseg:base \
  python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name()); assert torch.cuda.is_available()"
```

The default base is `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`. Override it
only with a PyTorch image that satisfies the versions in `requirements.txt`:

```bash
docker build \
  --build-arg PYTORCH_IMAGE=pytorch/pytorch:<tag> \
  -t april-medseg:base .
```

## Train

Keep datasets, outputs, and downloaded weights outside the image. Use absolute
host paths for bind mounts and a named volume for reusable model caches:

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

For a custom YAML, mount it read-only and pass its container path to
`--config`.

## Test

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

## Optional model families

The base image intentionally does not install every optional dependency:

| Model family | Additional dependencies | Notes |
|---|---|---|
| Foundation encoders | Model-specific packages such as `open_clip_torch` or `sentencepiece` | The base image already includes Transformers 4.x and `safetensors`; install other packages required by the selected model. |
| MLLM pipeline | `accelerate`, `qwen-vl-utils`, and other model-specific packages | Qwen-VL, InternVL, LLaVA, and related models have different requirements. |
| Mamba / SSM | `causal-conv1d`, `mamba-ssm` | Requires a matching PyTorch/CUDA toolchain and normally a `devel` base image. |
| ONNX | `onnx`, `onnxruntime` or `onnxruntime-gpu` | Choose the runtime for the target deployment. |

Create a derived image for a tested model family instead of installing optional
packages every time the container starts. Do not bake datasets, checkpoints, or
access tokens into an image.
