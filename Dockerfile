ARG PYTORCH_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

LABEL org.opencontainers.image.title="APRIL-MedSeg" \
      org.opencontainers.image.description="Modular medical image segmentation framework" \
      org.opencontainers.image.source="https://github.com/juntaoJianggavin/APRIL-MedSeg"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace/APRIL-MedSeg

# Install third-party dependencies before copying the source to keep this layer
# reusable when only project code changes. Containers do not need OpenCV's GUI
# bindings, so use the headless wheel and avoid pulling X11/GL runtime packages.
# The base image already provides a compatible torch/torchvision pair, so the
# lower bounds do not replace it.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && sed 's/^opencv-python/opencv-python-headless/' requirements.txt \
        > /tmp/requirements-docker.txt \
    && python -m pip install -r /tmp/requirements-docker.txt \
    && rm /tmp/requirements-docker.txt

# The component registry imports text-guided model modules during package
# initialization, so transformers is required even when a basic model is
# selected. Stay on the supported 4.x API; model-specific extras remain opt-in.
RUN python -m pip install 'transformers>=4.50,<5'

COPY setup.py ./
COPY medseg ./medseg
RUN python -m pip install --no-deps .

COPY . .

CMD ["python", "train.py", "--help"]
