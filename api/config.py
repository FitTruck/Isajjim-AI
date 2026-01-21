"""
API Configuration

환경변수 설정 및 디바이스 구성
CRITICAL: 이 모듈은 torch/spconv import 전에 로드되어야 함
"""

import os

# ============================================================================
# CRITICAL: Set environment variables BEFORE importing torch/spconv
# ============================================================================
os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME") or os.environ.get("CONDA_PREFIX") or "/usr/local/cuda"
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "100"
os.environ["TORCH_CUDA_ARCH_LIST"] = "all"

# Thread limits
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# macOS MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

# ============================================================================
# CRITICAL: Set PyTorch default dtype to float32
# ============================================================================
torch.set_default_dtype(torch.float32)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def configure_cuda():
    """Configure CUDA settings for optimal performance"""
    device = get_device()
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


# Assets directory
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Device
device = get_device()
configure_cuda()
print(f"Using device: {device}")
