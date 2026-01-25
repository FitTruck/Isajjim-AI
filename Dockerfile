# =============================================================================
# SAM3D API Dockerfile
# Multi-stage build for FastAPI + SAM-3D + YOLOE
#
# Build: docker build -t sam3d-api .
# Run:   docker run --gpus all -p 8000:8000 sam3d-api
# =============================================================================

# =============================================================================
# Stage 1: Builder - Install system deps and compile CUDA extensions
# =============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglew-dev \
    libglm-dev \
    ffmpeg \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set CUDA environment for compilation
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV MAX_JOBS=4

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir --timeout 300 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install spconv for CUDA 12.1
RUN pip install --no-cache-dir spconv-cu121==2.3.8

# Install xformers
RUN pip install --no-cache-dir xformers==0.0.28.post3

# Install flash-attn (takes a while to compile)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Install triton
RUN pip install --no-cache-dir triton

# Install mip-splatting (diff-gaussian-rasterization)
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization

# Install nvdiffrast
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/NVlabs/nvdiffrast.git

# Install pytorch3d
RUN pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git"

# Install MoGe
RUN pip install --no-cache-dir \
    git+https://github.com/microsoft/MoGe.git

# Install seaborn (required by sam-3d-objects inference)
RUN pip install --no-cache-dir seaborn

# Install kaolin from NVIDIA wheels
RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html

# Install gsplat (requires CUDA compilation)
RUN pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

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
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
