"""
Seed Variance Experiment Configuration

실험 계획서 (EXPERIMENT_SEED_VARIANCE.md) 기반 설정.
"""

import os
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
PIX3D_DIR = DATA_DIR / "pix3d"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PLY_OUTPUT_DIR = RESULTS_DIR / "ply"

SAM3D_CONFIG = PROJECT_ROOT / "sam-3d-objects" / "checkpoints" / "hf" / "pipeline.yaml"
SAM3D_NOTEBOOK_DIR = PROJECT_ROOT / "sam-3d-objects" / "notebook"

# ============================================================================
# Pix3D Sampling
# ============================================================================
# 카테고리별 샘플 수 (총 50개)
CATEGORY_SAMPLES = {
    "bed": 8,
    "sofa": 8,
    "chair": 8,
    "table": 8,
    "desk": 6,
    "bookcase": 6,
    "wardrobe": 6,
}
TOTAL_SAMPLES = sum(CATEGORY_SAMPLES.values())  # 50

# 샘플링 기준
MIN_IMAGE_SIZE = 512  # 최소 해상도
MAX_TRUNCATION_RATIO = 0.2  # 가려짐 비율 상한
MIN_MASK_PIXELS = 1000  # 마스크 최소 픽셀 수

# ============================================================================
# Phase 1: Original SAM-3D (Seed Variance 측정)
# ============================================================================
SEEDS = [1, 2, 3, 4, 5]  # K=5 seeds

ORIGINAL_CONFIG = {
    "stage1_inference_steps": 25,  # 기본값
    "stage2_inference_steps": 25,  # 기본값
    "decode_formats": ["gaussian"],
    "with_mesh_postprocess": True,  # 기본값
    "with_texture_baking": True,  # 기본값
    "with_layout_postprocess": False,  # 기본값
    "use_synthetic_pointmap": True,  # MoGe 미설치로 synthetic 사용
    "compile": False,
    "enable_ss_caching": False,
    "enable_slat_caching": False,
}

# ============================================================================
# Phase 2: Optimized SAM-3D (Optimization Deviation 측정)
# ============================================================================
OPTIMIZED_SEED = 42

OPTIMIZED_CONFIG = {
    "stage1_inference_steps": 14,  # 최적화: 25→14
    "stage2_inference_steps": 4,  # 최적화: 25→4
    "decode_formats": ["gaussian"],
    "with_mesh_postprocess": False,  # 최적화: 비활성화
    "with_texture_baking": False,  # 최적화: 비활성화
    "with_layout_postprocess": False,
    "use_synthetic_pointmap": True,  # 최적화: MoGe→synthetic
    "compile": False,
    "enable_ss_caching": True,  # SS Step Caching 활성화
    "ss_cache_stride": 3,
    "ss_cache_warmup_steps": 2,
    "enable_slat_caching": False,
}

# ============================================================================
# GPU Settings
# ============================================================================
GPU_IDS = [0, 1]  # Available L4 GPUs

# ============================================================================
# Output Files
# ============================================================================
RAW_CSV = RESULTS_DIR / "phase1_raw.csv"
OPTIMIZED_CSV = RESULTS_DIR / "phase2_optimized.csv"
SAMPLE_LIST = DATA_DIR / "selected_samples.json"
ANALYSIS_REPORT = RESULTS_DIR / "analysis_report.md"
