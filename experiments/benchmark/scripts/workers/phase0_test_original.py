"""
Phase 0: Original SAM-3D default 설정 L4 GPU 동작 테스트

Original SAM-3D의 실제 default 설정:
- decode_formats=["gaussian", "mesh"]
- stage1=25, stage2=25
- mesh_postprocess=True, texture_baking=True
- use_vertex_color=False
- compile=True (pipeline.yaml default)

추가로 gaussian-only도 테스트하여 비교.

Usage:
    python experiments/benchmark/phase0_test_original.py
"""

import sys
import os

# CRITICAL: Set environment variables BEFORE importing torch/spconv
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME") or os.environ.get("CONDA_PREFIX") or "/usr/local/cuda"
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = os.environ.get("SPCONV_ALGO_TIME_LIMIT", "100")
os.environ["TORCH_CUDA_ARCH_LIST"] = "all"
os.environ["WARP_QUIET"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import gc
import time
import tempfile
import logging

import numpy as np
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.float32)

from PIL import Image
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Phase0] %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PIX3D_DIR = PROJECT_ROOT / "experiments" / "seed_variance" / "data" / "pix3d"
SAM3D_NOTEBOOK_DIR = PROJECT_ROOT / "sam-3d-objects" / "notebook"
SAM3D_CONFIG = PROJECT_ROOT / "sam-3d-objects" / "checkpoints" / "hf" / "pipeline.yaml"

# Test samples
TEST_SAMPLES = [
    {"category": "bed", "img": "img/bed/0004.png", "mask": "mask/bed/0004.png"},
    {"category": "chair", "img": "img/chair/0022.png", "mask": "mask/chair/0022.png"},
    {"category": "table", "img": "img/table/0006.png", "mask": "mask/table/0006.png"},
]


def make_synthetic_pointmap(image: np.ndarray, z: float = 1.0) -> torch.Tensor:
    """Create a simple pinhole-camera pointmap."""
    H, W = image.shape[:2]
    f = 0.9 * max(H, W)
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    Z = np.full((H, W), z, dtype=np.float32)
    X = (uu - cx) / f * Z
    Y = (vv - cy) / f * Z
    pm = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    return torch.from_numpy(pm)


def calculate_obb_dimensions(ply_path: str) -> dict:
    """PLY -> PCA OBB -> relative dimensions (w, d, h)."""
    import trimesh

    pc = trimesh.load(ply_path)
    if isinstance(pc, trimesh.PointCloud):
        points = pc.vertices
    elif isinstance(pc, trimesh.Trimesh):
        points = pc.vertices
    elif isinstance(pc, trimesh.Scene):
        all_pts = []
        for geom in pc.geometry.values():
            if hasattr(geom, 'vertices'):
                all_pts.append(geom.vertices)
        points = np.vstack(all_pts) if all_pts else np.zeros((0, 3))
    else:
        points = np.array(pc.vertices) if hasattr(pc, 'vertices') else np.zeros((0, 3))

    if len(points) < 4:
        return {"width": 0.0, "depth": 0.0, "height": 0.0, "n_points": len(points)}

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    rotated = centered @ eigenvectors
    obb_min = rotated.min(axis=0)
    obb_max = rotated.max(axis=0)
    obb_dims = obb_max - obb_min

    similarity = np.abs(eigenvectors.T)
    pairs = []
    for i in range(3):
        for j in range(3):
            pairs.append((similarity[i, j], i, j))
    pairs.sort(reverse=True)

    obb_to_coord = {}
    used_coords = set()
    for _, obb_idx, coord_idx in pairs:
        if obb_idx not in obb_to_coord and coord_idx not in used_coords:
            obb_to_coord[obb_idx] = coord_idx
            used_coords.add(coord_idx)

    dims = [0.0, 0.0, 0.0]
    for obb_idx, coord_idx in obb_to_coord.items():
        dims[coord_idx] = float(obb_dims[obb_idx])

    return {"width": dims[0], "depth": dims[2], "height": dims[1], "n_points": len(points)}


def measure_model_vram(pipe) -> dict:
    """각 서브모델의 VRAM 사용량을 개별 측정."""
    model_vram = {}

    if not hasattr(pipe, "models"):
        return model_vram

    for name, model in pipe.models.items():
        if model is None:
            model_vram[name] = {"params_millions": 0, "size_mb": 0, "device": "None"}
            continue

        total_params = 0
        total_bytes = 0
        device_str = "unknown"

        if hasattr(model, "parameters"):
            total_params = sum(p.numel() for p in model.parameters())
            total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            for p in model.parameters():
                device_str = str(p.device)
                break

        if hasattr(model, "buffers"):
            total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())

        model_vram[name] = {
            "params_millions": total_params / 1e6,
            "size_mb": total_bytes / (1024**2),
            "device": device_str,
        }

    return model_vram


def run_inference_test(
    pipe,
    make_scene_fn,
    ready_gs_fn,
    sample: dict,
    decode_formats: list[str],
    test_name: str,
    mesh_postprocess: bool = True,
    texture_baking: bool = True,
    use_vertex_color: bool = False,
) -> dict:
    """단일 샘플에 대해 추론을 실행하고 결과를 반환."""
    img_path = PIX3D_DIR / sample["img"]
    mask_path = PIX3D_DIR / sample["mask"]

    fail_result = {
        "test": test_name,
        "category": sample["category"],
        "sample": sample["img"],
        "decode_formats": str(decode_formats),
        "latency_s": 0.0,
        "vram_peak_gb": 0.0,
        "vram_allocated_before_gb": 0.0,
        "width": 0.0,
        "depth": 0.0,
        "height": 0.0,
        "n_points": 0,
        "success": False,
        "error": "",
    }

    if not img_path.exists() or not mask_path.exists():
        fail_result["error"] = f"File not found: {img_path} or {mask_path}"
        return fail_result

    image_pil = Image.open(img_path).convert("RGB")
    image = np.array(image_pil)

    mask_pil = Image.open(mask_path).convert("L")
    mask = np.array(mask_pil)
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    mask_pixels = int(np.sum(mask_u8 > 0))
    logger.info(f"  Image: {image.shape}, Mask pixels: {mask_pixels}")

    if mask_pixels == 0:
        fail_result["error"] = "Empty mask"
        return fail_result

    pointmap = make_synthetic_pointmap(image, z=1.0)

    torch.manual_seed(42)
    np.random.seed(42)

    # Measure VRAM before inference
    gc.collect()
    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated() / (1024**3)
    torch.cuda.reset_peak_memory_stats()

    logger.info(
        f"  Running {test_name} (stage1=25, stage2=25, "
        f"decode={decode_formats}, mesh_pp={mesh_postprocess}, "
        f"tex_bake={texture_baking}, vertex_color={use_vertex_color})..."
    )
    torch.cuda.synchronize()
    t_start = time.time()

    try:
        with torch.no_grad():
            output = pipe.run(
                image=image,
                mask=mask_u8,
                seed=42,
                pointmap=pointmap,
                decode_formats=decode_formats,
                stage1_inference_steps=25,
                stage2_inference_steps=25,
                with_mesh_postprocess=mesh_postprocess,
                with_texture_baking=texture_baking,
                with_layout_postprocess=False,
                use_vertex_color=use_vertex_color,
            )

        torch.cuda.synchronize()
        t_end = time.time()
        latency = t_end - t_start

        vram_peak = torch.cuda.max_memory_allocated() / (1024**3)

        # Save PLY and calculate dimensions
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            ply_path = tmp.name

        try:
            scene_gs = make_scene_fn(output, in_place=True)
            scene_gs = ready_gs_fn(scene_gs, in_place=True, fix_alignment=False)
            scene_gs.save_ply(ply_path)
            dims = calculate_obb_dimensions(ply_path)
        finally:
            if os.path.exists(ply_path):
                os.unlink(ply_path)

        result = {
            "test": test_name,
            "category": sample["category"],
            "sample": sample["img"],
            "decode_formats": str(decode_formats),
            "latency_s": latency,
            "vram_peak_gb": vram_peak,
            "vram_allocated_before_gb": vram_before,
            "width": dims["width"],
            "depth": dims["depth"],
            "height": dims["height"],
            "n_points": dims["n_points"],
            "success": True,
            "error": "",
        }

        logger.info(f"  Latency: {latency:.2f}s")
        logger.info(f"  VRAM Before: {vram_before:.2f}GB")
        logger.info(f"  VRAM Peak: {vram_peak:.2f}GB")
        logger.info(f"  VRAM Delta (inference): {vram_peak - vram_before:.2f}GB")
        logger.info(f"  Dims: W={dims['width']:.4f}, D={dims['depth']:.4f}, H={dims['height']:.4f}")
        logger.info(f"  Points: {dims['n_points']}")

    except torch.cuda.OutOfMemoryError as e:
        t_end = time.time()
        vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
        result = {**fail_result}
        result.update({
            "latency_s": t_end - t_start,
            "vram_peak_gb": vram_peak,
            "vram_allocated_before_gb": vram_before,
            "error": f"OOM: {e}",
        })
        logger.error(f"  OOM ERROR: {e}")
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        t_end = time.time()
        vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
        result = {**fail_result}
        result.update({
            "latency_s": t_end - t_start,
            "vram_peak_gb": vram_peak,
            "vram_allocated_before_gb": vram_before,
            "error": str(e),
        })
        logger.error(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    return result


def print_results_table(label: str, results: list[dict], gpu_total_mem: float):
    """결과 테이블 출력."""
    logger.info(f"\n--- {label} ---")
    logger.info(
        f"{'Category':<12s} {'Latency':>8s} {'VRAM Peak':>10s} "
        f"{'VRAM Model':>11s} {'VRAM Δ':>8s} {'W':>8s} {'D':>8s} {'H':>8s} {'Status':>8s}"
    )
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        vram_delta = r["vram_peak_gb"] - r["vram_allocated_before_gb"]
        logger.info(
            f"{r['category']:<12s} {r['latency_s']:>7.1f}s {r['vram_peak_gb']:>9.2f}G "
            f"{r['vram_allocated_before_gb']:>10.2f}G {vram_delta:>7.2f}G "
            f"{r['width']:>8.4f} {r['depth']:>8.4f} {r['height']:>8.4f} {status:>8s}"
        )
        if not r["success"]:
            logger.info(f"  Error: {r['error'][:80]}")

    success = [r for r in results if r["success"]]
    if success:
        avg_lat = np.mean([r["latency_s"] for r in success])
        max_vram = max(r["vram_peak_gb"] for r in success)
        logger.info(f"  성공: {len(success)}/{len(results)}")
        logger.info(f"  평균 Latency: {avg_lat:.1f}s")
        logger.info(f"  최대 VRAM Peak: {max_vram:.2f}GB / {gpu_total_mem:.0f}GB ({max_vram/gpu_total_mem*100:.0f}%)")
    else:
        logger.warning(f"  모든 샘플 실패")


def main():
    logger.info("=" * 70)
    logger.info("Phase 0: Original SAM-3D Default 설정 L4 동작 테스트")
    logger.info("  - Test A: Original Default (gaussian+mesh, mesh_pp=True, tex_bake=True)")
    logger.info("  - Test B: gaussian-only (비교용)")
    logger.info("=" * 70)

    # ================================================================
    # GPU Info
    # ================================================================
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU: {gpu_name} ({gpu_total_mem:.1f}GB)")

    vram_initial = torch.cuda.memory_allocated() / (1024**3)
    logger.info(f"VRAM initial (before model load): {vram_initial:.4f}GB")

    # ================================================================
    # Load SAM-3D (Original default: compile=True from pipeline.yaml)
    # ================================================================
    logger.info("\nLoading SAM-3D model (Original default, compile=False — _warmup bug workaround)...")
    sys.path.insert(0, str(SAM3D_NOTEBOOK_DIR))

    real_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    from inference import Inference, make_scene, ready_gaussian_for_video_rendering

    torch.cuda.reset_peak_memory_stats()
    load_start = time.time()
    # pipeline.yaml default는 compile=True이나 _warmup 버그로 crash
    # → compile=False로 사용 (Original SAM-3D의 실질적 default)
    sam3d = Inference(str(SAM3D_CONFIG), compile=False)
    pipe = sam3d._pipeline

    # Move ALL models to GPU (Original keeps all decoders)
    if hasattr(pipe, "models"):
        for name, model in pipe.models.items():
            if model is not None and hasattr(model, "cuda"):
                model.cuda()
            if model is not None and hasattr(model, "eval"):
                model.eval()

    sys.stdout = real_stdout
    load_time = time.time() - load_start

    torch.set_grad_enabled(False)

    # ================================================================
    # Model VRAM 상세 측정
    # ================================================================
    vram_after_load_allocated = torch.cuda.memory_allocated() / (1024**3)
    vram_after_load_reserved = torch.cuda.memory_reserved() / (1024**3)
    vram_load_peak = torch.cuda.max_memory_allocated() / (1024**3)

    logger.info(f"\n{'='*70}")
    logger.info("모델 로딩 VRAM 분석")
    logger.info(f"{'='*70}")
    logger.info(f"Model load time: {load_time:.1f}s")
    logger.info(f"compile=False (_warmup bug workaround)")
    logger.info(f"VRAM allocated (model weights):  {vram_after_load_allocated:.3f}GB")
    logger.info(f"VRAM reserved  (CUDA allocator): {vram_after_load_reserved:.3f}GB")
    logger.info(f"VRAM peak during loading:        {vram_load_peak:.3f}GB")

    # 개별 모델 VRAM
    if hasattr(pipe, "models"):
        logger.info(f"\nLoaded models: {list(pipe.models.keys())}")
        model_vram = measure_model_vram(pipe)
        logger.info(f"\n{'Model':<30s} {'Params(M)':>10s} {'Size(MB)':>10s} {'Device':>10s}")
        logger.info("-" * 65)
        total_model_mb = 0.0
        for name, info in sorted(model_vram.items()):
            if info["device"] == "None":
                logger.info(f"{name:<30s} {'(None)':>10s}")
            else:
                logger.info(
                    f"{name:<30s} {info['params_millions']:>10.1f} "
                    f"{info['size_mb']:>10.1f} {info['device']:>10s}"
                )
                total_model_mb += info["size_mb"]
        logger.info("-" * 65)
        logger.info(f"{'TOTAL (params only)':<30s} {'':>10s} {total_model_mb:>10.1f}")
        logger.info(
            f"\nCUDA allocator overhead = "
            f"{vram_after_load_allocated * 1024 - total_model_mb:.1f}MB "
            f"(activations, buffers, fragmentation)"
        )

    # ================================================================
    # Test A: Original Default (gaussian+mesh, mesh_pp=True, tex_bake=True)
    # ================================================================
    logger.info(f"\n{'='*70}")
    logger.info("Test A: Original Default")
    logger.info("  decode_formats=['gaussian', 'mesh']")
    logger.info("  mesh_postprocess=True, texture_baking=True, use_vertex_color=False")
    logger.info(f"{'='*70}")

    results_a = []
    for i, sample in enumerate(TEST_SAMPLES):
        logger.info(f"\n[A-{i+1}/{len(TEST_SAMPLES)}] {sample['category']}: {sample['img']}")
        result = run_inference_test(
            pipe=pipe,
            make_scene_fn=make_scene,
            ready_gs_fn=ready_gaussian_for_video_rendering,
            sample=sample,
            decode_formats=["gaussian", "mesh"],
            test_name="original-default",
            mesh_postprocess=True,
            texture_baking=True,
            use_vertex_color=False,
        )
        results_a.append(result)

        # If OOM, skip remaining
        if not result["success"] and "OOM" in result.get("error", ""):
            logger.warning("OOM detected — skipping remaining default tests")
            break

    # ================================================================
    # Test B: gaussian-only (비교용)
    # ================================================================
    logger.info(f"\n{'='*70}")
    logger.info("Test B: gaussian-only (비교용)")
    logger.info("  decode_formats=['gaussian']")
    logger.info("  mesh_postprocess=True, texture_baking=True, use_vertex_color=False")
    logger.info(f"{'='*70}")

    gc.collect()
    torch.cuda.empty_cache()

    results_b = []
    for i, sample in enumerate(TEST_SAMPLES):
        logger.info(f"\n[B-{i+1}/{len(TEST_SAMPLES)}] {sample['category']}: {sample['img']}")
        result = run_inference_test(
            pipe=pipe,
            make_scene_fn=make_scene,
            ready_gs_fn=ready_gaussian_for_video_rendering,
            sample=sample,
            decode_formats=["gaussian"],
            test_name="gaussian-only",
            mesh_postprocess=True,
            texture_baking=True,
            use_vertex_color=False,
        )
        results_b.append(result)

    # ================================================================
    # 종합 결과 요약
    # ================================================================
    logger.info(f"\n{'='*70}")
    logger.info("Phase 0 종합 결과 요약")
    logger.info(f"{'='*70}")

    logger.info(f"\nGPU: {gpu_name} ({gpu_total_mem:.1f}GB)")
    logger.info(f"Model load time: {load_time:.1f}s (compile=False)")
    logger.info(f"Model load VRAM (allocated): {vram_after_load_allocated:.3f}GB")
    logger.info(f"Model load VRAM (reserved):  {vram_after_load_reserved:.3f}GB")

    print_results_table("Test A: Original Default (gaussian+mesh, mesh_pp=True)", results_a, gpu_total_mem)
    print_results_table("Test B: gaussian-only (비교)", results_b, gpu_total_mem)

    # ================================================================
    # 결론
    # ================================================================
    logger.info(f"\n{'='*70}")
    logger.info("Phase 0 결론")
    logger.info(f"{'='*70}")

    success_a = [r for r in results_a if r["success"]]
    success_b = [r for r in results_b if r["success"]]
    oom_in_a = any(not r["success"] and "OOM" in r.get("error", "") for r in results_a)

    if oom_in_a:
        logger.info("[결론] Original default (gaussian+mesh, mesh_pp=True) → L4 24GB에서 OOM 발생")
        logger.info("[권장] 3종 벤치마크에서 decode_formats=['gaussian']으로 통일")
    elif success_a:
        max_a = max(r["vram_peak_gb"] for r in success_a)
        avg_lat_a = np.mean([r["latency_s"] for r in success_a])
        logger.info(f"[결론] Original default 정상 동작")
        logger.info(f"  VRAM peak: {max_a:.2f}GB ({max_a/gpu_total_mem*100:.0f}%)")
        logger.info(f"  평균 latency: {avg_lat_a:.1f}s")

    if success_b:
        max_b = max(r["vram_peak_gb"] for r in success_b)
        avg_lat_b = np.mean([r["latency_s"] for r in success_b])
        logger.info(f"\n[비교] gaussian-only")
        logger.info(f"  VRAM peak: {max_b:.2f}GB ({max_b/gpu_total_mem*100:.0f}%)")
        logger.info(f"  평균 latency: {avg_lat_b:.1f}s")

    if success_a and success_b:
        max_a = max(r["vram_peak_gb"] for r in success_a)
        max_b = max(r["vram_peak_gb"] for r in success_b)
        avg_a = np.mean([r["latency_s"] for r in success_a])
        avg_b = np.mean([r["latency_s"] for r in success_b])
        logger.info(f"\n[mesh 추가 비용]")
        logger.info(f"  VRAM: +{max_a - max_b:.2f}GB")
        logger.info(f"  Latency: +{avg_a - avg_b:.1f}s ({(avg_a/avg_b - 1)*100:.0f}% 증가)")

    logger.info("\nPhase 0 완료.")


if __name__ == "__main__":
    main()
