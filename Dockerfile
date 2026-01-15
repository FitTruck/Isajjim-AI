# =============================================================================
# SAM3D API Dockerfile
# Multi-stage build for FastAPI + SAM2 + SAM-3D + YOLO + CLIP
#
# Build: docker build -t sam3d-api .
# Run:   docker run --gpus all -p 8000:8000 -v /data/sam3d:/data/sam3d sam3d-api
# =============================================================================

# =============================================================================
# Stage 1: Builder - Install system deps and compile CUDA extensions
# =============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
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
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# Create conda environment with Python 3.11
RUN conda create -n sam3d python=3.11 -y

# Set shell to use conda environment
SHELL ["conda", "run", "-n", "sam3d", "/bin/bash", "-c"]

# Set CUDA environment for compilation
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install packages that require CUDA compilation
WORKDIR /tmp/build

# Install mip-splatting (diff-gaussian-rasterization)
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization

# Install nvdiffrast
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/NVlabs/nvdiffrast.git

# =============================================================================
# Stage 2: Dependencies - Install Python packages
# =============================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS deps

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
    && rm -rf /var/lib/apt/lists/*

# Copy conda from builder
COPY --from=builder /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:$PATH"

SHELL ["conda", "run", "-n", "sam3d", "/bin/bash", "-c"]

# Install remaining Python packages (non-CUDA)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    Pillow>=10.0.0 \
    numpy>=1.24.0 \
    opencv-python-headless>=4.9.0 \
    transformers==4.57.3 \
    tokenizers>=0.15.0 \
    scipy>=1.10.0 \
    trimesh \
    pygltflib \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.2 \
    ultralytics>=8.1.0 \
    sahi \
    aiohttp \
    requests \
    ftfy \
    regex \
    imageio \
    imageio-ffmpeg

# Install CLIP from OpenAI
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# =============================================================================
# Stage 3: Runtime - Final slim image
# =============================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglew2.2 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment from deps stage
COPY --from=deps /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY api.py .
COPY ai/ ./ai/

# =============================================================================
# Environment Variables
# CRITICAL: These must be set before torch/spconv import
# =============================================================================

# Conda environment
ENV CONDA_PREFIX=/opt/conda/envs/sam3d
ENV CUDA_HOME=/opt/conda/envs/sam3d

# spconv configuration (prevents infinite tuning)
ENV SPCONV_TUNE_DEVICE=0
ENV SPCONV_ALGO_TIME_LIMIT=100
ENV TORCH_CUDA_ARCH_LIST=all
ENV LIDRA_SKIP_INIT=true

# Thread limits (prevents thread explosion)
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV VECLIB_MAXIMUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# HuggingFace cache paths (volume-mounted)
ENV HF_HOME=/data/sam3d/huggingface
ENV TRANSFORMERS_CACHE=/data/sam3d/huggingface/transformers
ENV TORCH_HOME=/data/sam3d/torch

# PyTorch configuration
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# =============================================================================
# Container Configuration
# =============================================================================

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Copy and set entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
