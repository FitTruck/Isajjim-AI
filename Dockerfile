# =============================================================================
# Isajjim AI Dockerfile
#
# 코드 변경 시: cloudbuild.yaml (자동 빌드, ~5분)
# CUDA deps 변경 시: Dockerfile.base 수정 후 cloudbuild-base.yaml 실행
# =============================================================================

# =============================================================================
# Stage 1: Builder — 사전 빌드된 베이스 이미지 사용 (CUDA 컴파일 생략)
# =============================================================================
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS builder

# =============================================================================
# Stage 2: Runtime - Final image
# Using devel image for full CUDA/cuBLAS compatibility with kaolin
# =============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglew2.2 \
    ffmpeg \
    curl \
    git \
    python3.11 \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install remaining Python packages (non-CUDA)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    Pillow>=10.0.0 \
    numpy>=1.24.0 \
    opencv-python-headless>=4.9.0 \
    transformers>=4.43.0 \
    tokenizers>=0.15.0 \
    scipy>=1.10.0 \
    trimesh \
    pygltflib \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.2 \
    ultralytics>=8.3.0 \
    aiohttp \
    requests \
    loguru \
    roma \
    einops \
    timm \
    open3d \
    imageio \
    imageio-ffmpeg \
    seaborn

# Set working directory
WORKDIR /app

# Copy application code
COPY api.py .
COPY api/ ./api/
COPY ai/ ./ai/
COPY simulation/ ./simulation/

# =============================================================================
# Environment Variables
# =============================================================================

# spconv configuration
ENV SPCONV_TUNE_DEVICE=0
ENV SPCONV_ALGO_TIME_LIMIT=100
ENV TORCH_CUDA_ARCH_LIST=all
ENV LIDRA_SKIP_INIT=true

# Thread limits
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV VECLIB_MAXIMUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# HuggingFace cache paths
ENV HF_HOME=/data/sam3d/huggingface
ENV TRANSFORMERS_CACHE=/data/sam3d/huggingface/transformers
ENV TORCH_HOME=/data/sam3d/torch

# PyTorch configuration
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# PyTorch library path (required for kaolin and other extensions)
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}"

# =============================================================================
# Container Configuration
# =============================================================================

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

COPY docker-entrypoint.sh /app/
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
