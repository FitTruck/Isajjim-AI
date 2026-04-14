"""
Fast-SAM3D Benchmark Worker — R2 전용

Usage:
    conda run -n sam3d-objects python worker_fastsam3d.py \
        --gpu 0 --samples benchmark_samples.json --output results/fastsam3d.csv
"""

import sys
import os
import argparse

# Parse args early
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--samples", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--save-ply-dir", type=str, default=None, help="Save PLY files to this dir (for CD computation)")
args = parser.parse_args()

# CRITICAL: Set environment variables BEFORE importing torch/spconv
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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

import csv
import gc
import json
import logging
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.float32)

from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import PIX3D_DIR, SEED, FLUSH_INTERVAL

# ================================================================
# Paths
# ================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
FASTSAM3D_DIR = PROJECT_ROOT / "fast-sam3d"
FASTSAM3D_NOTEBOOK = FASTSAM3D_DIR / "notebook"
FASTSAM3D_CONFIG = FASTSAM3D_DIR / "checkpoints" / "hf" / "pipeline.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Worker:fastsam3d] %(message)s")
logger = logging.getLogger(__name__)

# Fast-SAM3D default acceleration params (from README/infer.sh)
FASTSAM3D_DEFAULTS = SimpleNamespace(
    ss_cache_stride=3,
    ss_warmup=2,
    ss_order=1,
    ss_momentum_beta=0.5,
    slat_thresh=1.5,
    slat_warmup=3,
    slat_carving_ratio=0.1,
    mesh_spectral_threshold_low=0.5,
    mesh_spectral_threshold_high=0.7,
    enable_ss_cache=True,
    enable_slat_carving=True,
    enable_mesh_aggregation=True,
    enable_acceleration=True,
)


class FastSAM3DWorker:
    def __init__(self):
        self.inference = None

    def load_model(self):
        """Fast-SAM3D 모델 로드."""
        logger.info("Loading Fast-SAM3D...")

        # fast-sam3d의 sam3d_objects 모듈을 사용하도록 경로 설정
        # 중요: 기존 sam-3d-objects보다 먼저 삽입
        sys.path.insert(0, str(FASTSAM3D_DIR))
        sys.path.insert(0, str(FASTSAM3D_NOTEBOOK))

        from omegaconf import OmegaConf

        real_stdout = sys.stdout
        devnull = open(os.devnull, "w")
        try:
            sys.stdout = devnull

            from inference import Inference

            # Config 로드 + acceleration 설정
            config = OmegaConf.load(str(FASTSAM3D_CONFIG))
            config.rendering_engine = "pytorch3d"
            config.compile_model = False
            config.workspace_dir = os.path.dirname(str(FASTSAM3D_CONFIG))

            # Enable acceleration configs
            config["ss_generator_config_path"] = "ss_generator_faster.yaml"
            config["slat_generator_config_path"] = "slat_generator_faster.yaml"

            self.inference = Inference(config, compile=False, args=FASTSAM3D_DEFAULTS)

            # Set acceleration params
            self.inference.get_params(FASTSAM3D_DEFAULTS)

        finally:
            sys.stdout = real_stdout
            devnull.close()

        torch.set_grad_enabled(False)

        vram_loaded = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"Model VRAM after load: {vram_loaded:.2f}GB")

    def run_single(self, sample: dict) -> dict:
        """단일 샘플 추론 + 치수 계산."""
        img_path = PIX3D_DIR / sample["img_path"]
        mask_path = PIX3D_DIR / sample["mask_path"]

        result = {
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "img_path": sample["img_path"],
            "mask_path": sample["mask_path"],
            "model_path": sample["model_path"],
            "dim_small": 0.0,
            "dim_mid": 0.0,
            "dim_large": 0.0,
            "latency_seconds": 0.0,
            "vram_peak_mb": 0.0,
            "success": False,
            "error": "",
        }

        if not img_path.exists() or not mask_path.exists():
            result["error"] = "File not found"
            return result

        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_arr = np.array(Image.open(mask_path).convert("L"))
        mask_bool = mask_arr > 0

        mask_pixels = int(np.sum(mask_bool))
        if mask_pixels == 0:
            result["error"] = "Empty mask"
            return result

        # HFER calculation (Fast-SAM3D spectral analysis)
        from fft.fft2d import calculate_hfer_robust
        try:
            hfer = calculate_hfer_robust(str(mask_path))
        except Exception:
            hfer = 0.0
        self.inference.get_hfer(hfer)

        # Synthetic pointmap
        pointmap = self._make_synthetic_pointmap(image)

        # Set seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Reset VRAM stats
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Run inference
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        try:
            with torch.no_grad():
                output = self.inference(
                    image,
                    mask_bool,
                    seed=SEED,
                    pointmap=pointmap,
                )

            torch.cuda.synchronize()
            latency = time.perf_counter() - t_start
            vram_peak = torch.cuda.max_memory_allocated() / (1024**2)

            # Extract PLY from gaussian and calculate dimensions
            gs_model = output["gaussian"][0]
            dims = self._calculate_obb_from_gaussian(gs_model)

            # Save PLY if requested (for CD computation)
            if args.save_ply_dir:
                save_dir = Path(args.save_ply_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                self._save_gaussian_ply(gs_model, str(save_dir / f"{sample['sample_id']}.ply"))

            result.update({
                "dim_small": dims["dim_small"],
                "dim_mid": dims["dim_mid"],
                "dim_large": dims["dim_large"],
                "latency_seconds": round(latency, 3),
                "vram_peak_mb": round(vram_peak, 1),
                "success": True,
            })

        except torch.cuda.OutOfMemoryError as e:
            result["error"] = f"OOM: {e}"
            result["latency_seconds"] = round(time.perf_counter() - t_start, 3)
            result["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            result["error"] = str(e)[:200]
            result["latency_seconds"] = round(time.perf_counter() - t_start, 3)
            result["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)

        return result

    @staticmethod
    def _make_synthetic_pointmap(image: np.ndarray, z: float = 1.0) -> torch.Tensor:
        H, W = image.shape[:2]
        f = 0.9 * max(H, W)
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
        Z = np.full((H, W), z, dtype=np.float32)
        X = (uu - cx) / f * Z
        Y = (vv - cy) / f * Z
        return torch.from_numpy(np.stack([X, Y, Z], axis=-1))

    @staticmethod
    def _calculate_obb_from_gaussian(gs_model) -> dict:
        """GaussianModel → PCA OBB → 정규화 → 크기순 정렬."""
        points = gs_model._xyz.detach().cpu().numpy()

        if len(points) < 4:
            return {"dim_small": 0.0, "dim_mid": 0.0, "dim_large": 0.0}

        # PCA OBB
        centered = points - points.mean(axis=0)
        cov = np.cov(centered.T)
        _, eigenvectors = np.linalg.eigh(cov)
        rotated = centered @ eigenvectors
        obb_dims = rotated.max(axis=0) - rotated.min(axis=0)

        # 정규화 + 크기순 정렬
        max_dim = obb_dims.max()
        if max_dim < 1e-8:
            return {"dim_small": 0.0, "dim_mid": 0.0, "dim_large": 0.0}

        normalized = sorted((obb_dims / max_dim).tolist())
        return {
            "dim_small": normalized[0],
            "dim_mid": normalized[1],
            "dim_large": normalized[2],
        }

    @staticmethod
    def _save_gaussian_ply(gs_model, path: str):
        """GaussianModel → PLY (xyz + SH → rgb)."""
        import struct
        xyz = gs_model._xyz.detach().cpu().numpy()
        n = len(xyz)
        if hasattr(gs_model, "_features_dc"):
            sh_dc = gs_model._features_dc.detach().cpu().numpy().squeeze()
            SH_C0 = 0.28209479177387814
            rgb = np.clip(SH_C0 * sh_dc + 0.5, 0, 1)
            rgb_u8 = (rgb * 255).astype(np.uint8)
        else:
            rgb_u8 = np.full((n, 3), 128, dtype=np.uint8)
        header = (
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        )
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            for i in range(n):
                f.write(struct.pack("<fff", xyz[i, 0], xyz[i, 1], xyz[i, 2]))
                f.write(struct.pack("<BBB", rgb_u8[i, 0], rgb_u8[i, 1], rgb_u8[i, 2]))


def load_completed_ids(output_path: str) -> set:
    """이미 처리된 sample_id 목록 (중단 복구용)."""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(int(row["sample_id"]))
    return completed


def main():
    logger.info("=" * 60)
    logger.info("Fast-SAM3D Benchmark Worker (R2)")
    logger.info(f"  GPU: {args.gpu}")
    logger.info(f"  Samples: {args.samples}")
    logger.info(f"  Output: {args.output}")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")

    # Load samples
    with open(args.samples) as f:
        samples = json.load(f)
    logger.info(f"Total samples: {len(samples)}")

    # 중단 복구
    completed = load_completed_ids(args.output)
    if completed:
        logger.info(f"Resuming: {len(completed)} already completed, skipping")
    remaining = [s for s in samples if s["sample_id"] not in completed]
    logger.info(f"Remaining: {len(remaining)}")

    if not remaining:
        logger.info("All samples already completed!")
        return

    # Load model
    worker = FastSAM3DWorker()
    worker.load_model()

    # Warmup
    logger.info("Warmup run (not recorded)...")
    warmup_dummy = {
        "sample_id": -1,
        "category": "warmup",
        "img_path": remaining[0]["img_path"],
        "mask_path": remaining[0]["mask_path"],
        "model_path": remaining[0]["model_path"],
    }
    _ = worker.run_single(warmup_dummy)
    logger.info("Warmup done")

    # CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id", "category", "img_path", "mask_path", "model_path",
        "dim_small", "dim_mid", "dim_large",
        "latency_seconds", "vram_peak_mb", "success", "error",
    ]

    file_exists = output_path.exists() and output_path.stat().st_size > 0

    success_count = 0
    fail_count = 0
    latencies = []
    t_total_start = time.perf_counter()

    with open(output_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, sample in enumerate(remaining):
            sid = sample["sample_id"]
            cat = sample["category"]

            result = worker.run_single(sample)
            writer.writerow(result)

            if result["success"]:
                success_count += 1
                latencies.append(result["latency_seconds"])
                status = f"{result['latency_seconds']:.1f}s"
            else:
                fail_count += 1
                status = f"FAIL: {result['error'][:40]}"

            done = i + 1
            total = len(remaining)
            avg_lat = np.mean(latencies) if latencies else 0
            eta = avg_lat * (total - done) / 3600 if avg_lat > 0 else 0

            if done % 10 == 0 or done == total:
                logger.info(
                    f"[{done}/{total}] {cat}/{sid} → {status} | "
                    f"avg={avg_lat:.1f}s | ETA={eta:.1f}h | "
                    f"ok={success_count} fail={fail_count}"
                )

            if done % FLUSH_INTERVAL == 0:
                csv_file.flush()
                os.fsync(csv_file.fileno())

    elapsed_total = time.perf_counter() - t_total_start
    logger.info(f"\n{'='*60}")
    logger.info("완료: Fast-SAM3D (R2)")
    logger.info(f"  성공: {success_count}/{len(remaining)}")
    logger.info(f"  실패: {fail_count}")
    if latencies:
        logger.info(f"  평균 Latency: {np.mean(latencies):.1f}s")
        logger.info(f"  중간값 Latency: {np.median(latencies):.1f}s")
    logger.info(f"  총 소요: {elapsed_total/3600:.1f}h")
    logger.info(f"  결과: {args.output}")


if __name__ == "__main__":
    main()
