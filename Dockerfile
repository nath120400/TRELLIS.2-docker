# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    CUDA_HOME=/usr/local/cuda \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/root/.cache/huggingface

SHELL ["/bin/bash", "-o", "pipefail", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        libegl1 \
        libegl1-mesa-dev \
        libgl1 \
        libgl1-mesa-dev \
        libglib2.0-0 \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libwebp-dev \
        libxext6 \
        libxrender1 \
        ninja-build \
        pkg-config \
        wget \
    && rm -rf /var/lib/apt/lists/*

ARG MINIFORGE_VERSION=26.1.1-3
ARG MINIFORGE_SH=Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh
ARG MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_SH}
ARG MINIFORGE_SHA256=b25b828b702df4dd2a6d24d4eb56cfa912471dd8e3342cde2c3d86fe3dc2d870

RUN wget -qO /tmp/miniforge.sh "${MINIFORGE_URL}" \
    && echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum -c - \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniforge.sh \
    && "${CONDA_DIR}/bin/conda" config --system --set auto_activate_base false \
    && "${CONDA_DIR}/bin/conda" clean -afy

ENV PATH=${CONDA_DIR}/bin:$PATH

RUN conda create -y -n trellis2 python=3.10 pip \
    && conda clean -afy

ENV PATH=${CONDA_DIR}/envs/trellis2/bin:${CONDA_DIR}/bin:$PATH

WORKDIR /opt/TRELLIS.2
COPY . .

WORKDIR /opt/TRELLIS.2

RUN python -m pip install --upgrade pip setuptools wheel packaging \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0 \
    && python -m pip install \
        imageio \
        imageio-ffmpeg \
        tqdm \
        easydict \
        opencv-python-headless \
        ninja \
        trimesh \
        "transformers>=4.45,<5" \
        gradio==6.0.1 \
        tensorboard \
        aiohttp \
        pandas \
        lpips \
        zstandard \
        kornia \
        timm \
        Pillow \
    && python -m pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 \
    && MAX_JOBS=8 python -m pip install flash-attn==2.7.3 --no-build-isolation

RUN mkdir -p /tmp/extensions \
    && git clone --branch v0.4.0 --depth 1 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast \
    && git clone --branch renderutils --depth 1 https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec \
    && git clone --depth 1 --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh \
    && git clone --depth 1 --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM \
    && cp -r /opt/TRELLIS.2/o-voxel /tmp/extensions/o-voxel \
    && python -m pip install /tmp/extensions/nvdiffrast --no-build-isolation \
    && python -m pip install /tmp/extensions/nvdiffrec --no-build-isolation \
    && python -m pip install /tmp/extensions/CuMesh --no-build-isolation \
    && python -m pip install /tmp/extensions/FlexGEMM --no-build-isolation \
    && python -m pip install /tmp/extensions/o-voxel --no-build-isolation \
    && rm -rf /tmp/extensions

ENV PYTHONPATH=/opt/TRELLIS.2:${PYTHONPATH}

EXPOSE 7861/tcp

CMD ["python", "main.py"]