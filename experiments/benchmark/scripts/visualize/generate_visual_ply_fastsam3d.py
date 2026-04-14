"""
시각 비교용 PLY 생성 — Fast-SAM3D 전용

Usage (sam3d-objects env, fast-sam3d 디렉토리 필요):
    python experiments/benchmark/generate_visual_ply_fastsam3d.py --gpu 0

출력: results/visual_compare/{category}/{sid}_{model}/fastsam3d.ply
"""

import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

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

import gc
import json
import logging
import time
from types import SimpleNamespace

import numpy as np
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.float32)

from pathlib import Path
from PIL import Image

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BENCH_DIR = Path(__file__).resolve().parent.parent.parent
PIX3D_DIR = PROJECT_ROOT / "experiments" / "seed_variance" / "data" / "pix3d"
FASTSAM3D_DIR = PROJECT_ROOT / "fast-sam3d"
FASTSAM3D_NOTEBOOK = FASTSAM3D_DIR / "notebook"
FASTSAM3D_CONFIG = FASTSAM3D_DIR / "checkpoints" / "hf" / "pipeline.yaml"

VISUAL_DIR = BENCH_DIR / "results" / "visual_compare"
SAMPLES_PATH = BENCH_DIR / "data" / "visual_samples.json"
BENCHMARK_SAMPLES_PATH = BENCH_DIR / "data" / "benchmark_samples.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Visual:fastsam3d] %(message)s")
logger = logging.getLogger(__name__)

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


def load_visual_samples() -> list[dict]:
    visual = json.load(open(SAMPLES_PATH))
    benchmark = json.load(open(BENCHMARK_SAMPLES_PATH))
    bench_map = {s["sample_id"]: s for s in benchmark}
    return [bench_map[vs["sample_id"]] for vs in visual]


def make_output_dir(sample: dict) -> Path:
    cat = sample["category"]
    model_name = Path(sample["model_path"]).parent.name
    dir_name = f"{sample['sample_id']}_{model_name}"
    out_dir = VISUAL_DIR / cat / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def make_synthetic_pointmap(image: np.ndarray, z: float = 1.0) -> torch.Tensor:
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


class FastSAM3DVisualGenerator:
    def __init__(self):
        self.inference = None

    def load_model(self):
        logger.info("Loading Fast-SAM3D...")

        sys.path.insert(0, str(FASTSAM3D_DIR))
        sys.path.insert(0, str(FASTSAM3D_NOTEBOOK))

        from omegaconf import OmegaConf

        real_stdout = sys.stdout
        devnull = open(os.devnull, "w")
        try:
            sys.stdout = devnull
            from inference import Inference

            config = OmegaConf.load(str(FASTSAM3D_CONFIG))
            config.rendering_engine = "pytorch3d"
            config.compile_model = False
            config.workspace_dir = os.path.dirname(str(FASTSAM3D_CONFIG))
            config["ss_generator_config_path"] = "ss_generator_faster.yaml"
            config["slat_generator_config_path"] = "slat_generator_faster.yaml"

            self.inference = Inference(config, compile=False, args=FASTSAM3D_DEFAULTS)
            self.inference.get_params(FASTSAM3D_DEFAULTS)
        finally:
            sys.stdout = real_stdout
            devnull.close()

        torch.set_grad_enabled(False)
        vram = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"Model VRAM: {vram:.2f}GB")

    def warmup(self):
        logger.info("Warmup run...")
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy_mask = np.ones((256, 256), dtype=bool)
        pointmap = make_synthetic_pointmap(dummy_img)
        try:
            from fft.fft2d import calculate_hfer_robust
        except ImportError:
            pass
        try:
            self.inference.get_hfer(0.0)
            with torch.no_grad():
                self.inference(dummy_img, dummy_mask, seed=SEED, pointmap=pointmap)
        except Exception as e:
            logger.warning(f"Warmup error (ignored): {e}")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Warmup done")

    def generate_ply(self, sample: dict, out_dir: Path) -> bool:
        ply_path = out_dir / "fastsam3d.ply"
        if ply_path.exists():
            logger.info(f"SKIP (exists): {ply_path}")
            return True

        img_path = PIX3D_DIR / sample["img_path"]
        mask_path = PIX3D_DIR / sample["mask_path"]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask_arr = np.array(Image.open(mask_path).convert("L"))
        mask_bool = mask_arr > 0

        # HFER
        from fft.fft2d import calculate_hfer_robust
        try:
            hfer = calculate_hfer_robust(str(mask_path))
        except Exception:
            hfer = 0.0
        self.inference.get_hfer(hfer)

        pointmap = make_synthetic_pointmap(image)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        gc.collect()
        torch.cuda.empty_cache()

        try:
            with torch.no_grad():
                output = self.inference(
                    image, mask_bool, seed=SEED, pointmap=pointmap,
                )

            # Save gaussian as PLY
            gs_model = output["gaussian"][0]
            self._save_gaussian_ply(gs_model, str(ply_path))
            logger.info(f"Saved: {ply_path}")
            return True

        except Exception as e:
            logger.error(f"FAIL sample {sample['sample_id']}: {e}")
            return False

    @staticmethod
    def _save_gaussian_ply(gs_model, path: str):
        """GaussianModel → PLY (xyz + SH → rgb)."""
        import struct

        xyz = gs_model._xyz.detach().cpu().numpy()
        n = len(xyz)

        # Extract RGB from SH coefficients (DC component)
        if hasattr(gs_model, "_features_dc"):
            sh_dc = gs_model._features_dc.detach().cpu().numpy().squeeze()
            # SH DC to RGB: color = SH_C0 * sh + 0.5
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


def main():
    logger.info("=" * 60)
    logger.info("Visual PLY Generator — Fast-SAM3D")
    logger.info(f"  GPU: {args.gpu}")
    logger.info("=" * 60)

    samples = load_visual_samples()
    logger.info(f"Samples to process: {len(samples)}")

    gen = FastSAM3DVisualGenerator()
    gen.load_model()
    gen.warmup()

    success = 0
    fail = 0
    t_total = time.perf_counter()

    for i, s in enumerate(samples):
        out_dir = make_output_dir(s)
        logger.info(f"[{i+1}/{len(samples)}] {s['category']} / sample {s['sample_id']}")
        if gen.generate_ply(s, out_dir):
            success += 1
        else:
            fail += 1

    elapsed = time.perf_counter() - t_total
    logger.info("=" * 60)
    logger.info(f"Done: {success} success, {fail} fail in {elapsed:.1f}s")
    logger.info(f"Output: {VISUAL_DIR}")


if __name__ == "__main__":
    main()
