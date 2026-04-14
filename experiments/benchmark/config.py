"""
Benchmark Configuration — 6개 실행 config 정의

R1: baseline (Original default) = 3종 Original + Ablation Baseline
R3: o1 (+ Gaussian-Only)
R4: o2 (+ VRAM Unload)
R5: o4 (+ Steps Reduction)
R6: o5 (Ours) = 3종 Ours + Ablation O5
R2: fastsam3d (별도 워커)
"""

from dataclasses import dataclass, field
from pathlib import Path

# ================================================================
# Paths
# ================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIX3D_DIR = PROJECT_ROOT / "experiments" / "seed_variance" / "data" / "pix3d"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "benchmark" / "results"
SAM3D_NOTEBOOK_DIR = PROJECT_ROOT / "sam-3d-objects" / "notebook"
SAM3D_CONFIG = PROJECT_ROOT / "sam-3d-objects" / "checkpoints" / "hf" / "pipeline.yaml"

SAMPLES_JSON = PROJECT_ROOT / "experiments" / "benchmark" / "data" / "benchmark_samples.json"
GT_DIMENSIONS_JSON = PROJECT_ROOT / "experiments" / "benchmark" / "data" / "gt_dimensions.json"

# ================================================================
# Data
# ================================================================
MIN_MASK_PIXELS = 1000
SEED = 42
FLUSH_INTERVAL = 10  # CSV 중간 저장 간격

# 500개 층화 추출 배분
STRATIFIED_SAMPLES = {
    "chair": 218,
    "sofa": 96,
    "table": 80,
    "bed": 38,
    "desk": 30,
    "bookcase": 16,
    "wardrobe": 13,
    "misc": 5,
    "tool": 4,
}
TOTAL_SAMPLES = sum(STRATIFIED_SAMPLES.values())  # 500


# ================================================================
# Ablation Configs
# ================================================================
@dataclass
class BenchmarkConfig:
    name: str
    decode_formats: list[str]
    stage1_steps: int
    stage2_steps: int
    mesh_postprocess: bool
    texture_baking: bool
    use_vertex_color: bool
    vram_unload: bool  # slat_decoder_mesh, slat_decoder_gs_4, depth_model 제거
    ss_caching: bool
    compile: bool = False  # torch.compile (reduce-overhead)
    ss_cache_stride: int = 3
    ss_cache_warmup: int = 2
    slat_carving: bool = False  # SLAT token carving (from Fast-SAM3D)
    slat_carving_ratio: float = 0.1
    slat_thresh: float = 1.5
    slat_warmup: int = 3


CONFIGS = {
    # R1: Original default = Ablation Baseline
    "baseline": BenchmarkConfig(
        name="Original (Baseline)",
        decode_formats=["gaussian", "mesh"],
        stage1_steps=25,
        stage2_steps=25,
        mesh_postprocess=True,
        texture_baking=True,
        use_vertex_color=False,
        vram_unload=False,
        ss_caching=False,
    ),
    # R3: + O1 Gaussian-Only
    "o1": BenchmarkConfig(
        name="+O1: Gaussian-Only",
        decode_formats=["gaussian"],
        stage1_steps=25,
        stage2_steps=25,
        mesh_postprocess=False,
        texture_baking=False,
        use_vertex_color=True,
        vram_unload=False,
        ss_caching=False,
    ),
    # R4: + O2 VRAM Unload
    "o2": BenchmarkConfig(
        name="+O2: VRAM Unload",
        decode_formats=["gaussian"],
        stage1_steps=25,
        stage2_steps=25,
        mesh_postprocess=False,
        texture_baking=False,
        use_vertex_color=True,
        vram_unload=True,
        ss_caching=False,
    ),
    # R5: + O4 Steps Reduction
    "o4": BenchmarkConfig(
        name="+O4: Steps 14/4",
        decode_formats=["gaussian"],
        stage1_steps=14,
        stage2_steps=4,
        mesh_postprocess=False,
        texture_baking=False,
        use_vertex_color=True,
        vram_unload=True,
        ss_caching=False,
    ),
    # R6: Ours = + O5 SS Step Caching
    "o5": BenchmarkConfig(
        name="Ours (+O5: SS Caching)",
        decode_formats=["gaussian"],
        stage1_steps=14,
        stage2_steps=4,
        mesh_postprocess=False,
        texture_baking=False,
        use_vertex_color=True,
        vram_unload=True,
        ss_caching=True,
        compile=False,
        ss_cache_stride=3,
        ss_cache_warmup=2,
    ),
    # R6b: Ours + torch.compile (논문 메인 결과)
    "o5c": BenchmarkConfig(
        name="Ours (+O5 + compile)",
        decode_formats=["gaussian"],
        stage1_steps=14,
        stage2_steps=4,
        mesh_postprocess=False,
        texture_baking=False,
        use_vertex_color=True,
        vram_unload=True,
        ss_caching=True,
        compile=True,
        ss_cache_stride=3,
        ss_cache_warmup=2,
    ),
    # R8: Ours + Fast-SAM3D SS 최적화 (SS steps=2, ShortCut 모델 특성 활용)
    # SS=2에서는 warmup=2이므로 Taylor 캐싱은 no-op → 순수 step 감소 효과 측정
    "o5_ss2": BenchmarkConfig(
        name="Ours (+O5 + SS=2)",
        decode_formats=["gaussian"],
        stage1_steps=2,
        stage2_steps=4,
        mesh_postprocess=False,
        texture_baking=False,
        use_vertex_color=True,
        vram_unload=True,
        ss_caching=False,  # 2 steps에서 캐싱 무의미
        compile=False,
        ss_cache_stride=3,
        ss_cache_warmup=2,
    ),
    # R9: Ours + Fast-SAM3D SS + SLaT=14 (SS=2로 부정확해진 구조를 SLaT이 보정할 수 있는지)
    "o5_ss2_slat14": BenchmarkConfig(
        name="Ours (+SS=2 +SLaT=14)",
        decode_formats=["gaussian"],
        stage1_steps=2,
        stage2_steps=14,
        mesh_postprocess=False,
        texture_baking=False,
        use_vertex_color=True,
        vram_unload=True,
        ss_caching=False,
        compile=False,
        ss_cache_stride=3,
        ss_cache_warmup=2,
    ),
    # R7: Ours + SLAT Carving (Fast-SAM3D stage2 token pruning)
    "o5_slat": BenchmarkConfig(
        name="Ours (+O5 + SLAT Carving)",
        decode_formats=["gaussian"],
        stage1_steps=14,
        stage2_steps=4,
        mesh_postprocess=False,
        texture_baking=False,
        use_vertex_color=True,
        vram_unload=True,
        ss_caching=True,
        compile=False,
        ss_cache_stride=3,
        ss_cache_warmup=2,
        slat_carving=True,
        slat_carving_ratio=0.1,
        slat_thresh=1.5,
        slat_warmup=3,
    ),
}
