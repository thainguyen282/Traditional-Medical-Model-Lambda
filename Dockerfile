# syntax=docker/dockerfile:1.6
#
# Pre-baked training image for Lambda Cloud runs.
# Contains CUDA 12.8 + cuDNN + Python 3.10 + all requirements.txt deps.
# Repo source code is NOT baked in — it is bind-mounted at /workspace at run time
# so we always use the latest main2 without rebuilding the image.
#
# Build & push (done automatically by .github/workflows/docker.yml):
#   docker buildx build --platform linux/amd64 \
#       -t ghcr.io/venera-ai/traditional-medical-model:main2 \
#       --push .
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python3.10-venv \
        git git-lfs curl ca-certificates build-essential jq tmux \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN pip install --upgrade pip wheel setuptools

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && pip install "huggingface_hub[cli]" python-dotenv \
    && rm -f /tmp/requirements.txt

WORKDIR /workspace
CMD ["bash"]
