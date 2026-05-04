# syntax=docker/dockerfile:1.6
#
# Phonebot images:
#
#   target=base  -> CPU-only image. Slim Python base, no torch / Voxtral /
#                   Whisper. Runs the Scribe + OpenAI path; pipeline ceiling
#                   is 26/30. Builds in ~30s, image ~250 MB.
#
#   target=gpu   -> Full-pipeline image. Builds on python:3.11-slim and
#                   installs torch CUDA wheels from PyTorch's index, then
#                   transformers / accelerate / mistral-common / librosa
#                   on top. Voxtral runs through HF transformers (the
#                   29/30 path). Image ~7 GB.
#
# Build:
#     docker build --target base -t phonebot:cpu .
#     docker build --target gpu  -t phonebot:gpu .
#
# Run with GPU:
#     docker run --rm --gpus all --env-file .env \
#         -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
#         phonebot:gpu run

# ============================================================================
# CPU stage: slim Python base; only the API-driven pipeline path
# ============================================================================
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Core deps in their own layer so they cache across source-tree edits.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt

# Source tree + CLI install
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY study /app/study
COPY main.py /app/main.py
COPY README.md /app/README.md

# Editable install so the `phonebot` console script is on PATH and the
# in-tree edits are picked up by the same layer.
RUN pip install -e .

ENTRYPOINT ["phonebot"]
CMD ["doctor"]


# ============================================================================
# GPU stage: clean python:3.11-slim base + torch CUDA wheels installed
# from PyTorch's index. Avoids the conda environment in pytorch/pytorch
# images (which has been segfaulting pip's install hooks for native
# packages like pillow). Voxtral runs through HF transformers with
# `attn_implementation="eager"` (set in src/config.py via VOXTRAL_ATTN_IMPL),
# which bypasses the SDPA path responsible for the rare generation-time
# SIGSEGV. AUTO_RESUME recovers from the odd remaining crash.
#
# vLLM is no longer in the default image. The voxtral-vllm: backend is
# kept in src/transcription.py as opt-in code for users on torch 2.4 +
# CUDA 12 stacks where vLLM's Voxtral support is stable.
FROM python:3.11-slim AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false

# System libs needed at runtime by audio + image deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 ca-certificates curl git \
        libjpeg62-turbo libpng16-16 libwebp7 libtiff6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use whatever pip the python:3.11-slim image ships with (currently a
# stable 24.x). Earlier we ran `pip install --upgrade pip setuptools
# wheel` here, which pulled in pip 26.x and triggered SIGSEGVs in pip's
# wheel installer on subsequent torch/nvidia-cu12 installs. Pinning to
# a known-stable pip avoids that.
RUN pip install --upgrade 'pip<26' 'setuptools<80' wheel

# Install torch + torchaudio CUDA wheels from PyTorch's index. The wheels
# bundle the matching CUDA runtime libs (nvidia-cuda-runtime-cu12 etc.)
# so the host doesn't need a CUDA install - only the NVIDIA driver +
# nvidia-container-toolkit. Pinning to 2.5.1+cu124 because that's the
# combo Voxtral has been most heavily tested on.
#
# torch and torchaudio in separate RUN steps so each pip call has a
# fresh memory footprint and Docker layers them as separate cache units.
RUN pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torch==2.5.1
RUN pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torchaudio==2.5.1

# CPU-light deps in their own layer.
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# GPU-side deps in a separate layer. transformers + accelerate +
# mistral-common together pull a few hundred MB; keeping it isolated
# from torch's gigabytes lowers peak memory.
COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN pip install -r /app/requirements-gpu.txt

# Source tree + CLI install
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY study /app/study
COPY main.py /app/main.py
COPY README.md /app/README.md

RUN pip install -e .

ENTRYPOINT ["phonebot"]
CMD ["doctor"]
