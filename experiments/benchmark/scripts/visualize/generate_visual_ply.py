"""
시각 비교용 PLY 생성 — 카테고리별 2개 × (Original + Ours) + GT OBJ 복사

Usage (sam3d-objects env):
    python experiments/benchmark/generate_visual_ply.py --gpu 0 --config baseline
    python experiments/benchmark/generate_visual_ply.py --gpu 0 --config o5

Fast-SAM3D는 별도:
    python experiments/benchmark/generate_visual_ply_fastsam3d.py --gpu 0

출력 구조:
    results/visual_compare/
    ├── bed/
    │   ├── 413_IKEA_MALM_3/
    │   │   ├── gt.obj
    │   │   ├── original.ply
    │   │   ├── fastsam3d.ply
    │   │   ├── ours.ply
    │   │   └── input.jpg
    │   └── 429_IKEA_HEMNES_3/
    │       └── ...
    └── ...
"""

import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--config", type=str, required=True, choices=["baseline", "o5"])
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
import shutil
import time

import numpy as np
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.float32)

from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import CONFIGS, PIX3D_DIR, SAM3D_CONFIG, SAM3D_NOTEBOOK_DIR, SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Visual:%(config)s] %(message)s")

CONFIG_TO_FILENAME = {
    "baseline": "original",
    "o5": "ours",
}

BENCH_DIR = Path(__file__).resolve().parent.parent.parent
VISUAL_DIR = BENCH_DIR / "results" / "visual_compare"
SAMPLES_PATH = BENCH_DIR / "data" / "visual_samples.json"
BENCHMARK_SAMPLES_PATH = BENCH_DIR / "data" / "benchmark_samples.json"


def load_visual_samples() -> list[dict]:
    """visual_samples.json에서 선정된 18개 샘플 + benchmark_samples.json에서 full info."""
    visual = json.load(open(SAMPLES_PATH))
    benchmark = json.load(open(BENCHMARK_SAMPLES_PATH))
    bench_map = {s["sample_id"]: s for s in benchmark}

    enriched = []
    for vs in visual:
        sid = vs["sample_id"]
        full = bench_map[sid]
        enriched.append(full)
    return enriched


def make_output_dir(sample: dict) -> Path:
    """카테고리/sampleid_modelname/ 디렉토리 생성."""
    cat = sample["category"]
    model_name = Path(sample["model_path"]).parent.name  # e.g., IKEA_MALM_3
    dir_name = f"{sample['sample_id']}_{model_name}"
    out_dir = VISUAL_DIR / cat / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def copy_gt_and_input(sample: dict, out_dir: Path):
    """GT OBJ + MTL + input image 복사 (이미 있으면 스킵)."""
    # GT model
    gt_src = PIX3D_DIR / sample["model_path"]
    gt_dst = out_dir / "gt.obj"
    if not gt_dst.exists() and gt_src.exists():
        shutil.copy2(gt_src, gt_dst)
        # MTL 파일도 복사
        mtl_src = gt_src.with_suffix(".mtl")
        if mtl_src.exists():
            shutil.copy2(mtl_src, out_dir / "gt.mtl")

    # Input image
    img_src = PIX3D_DIR / sample["img_path"]
    img_dst = out_dir / "input.jpg"
    if not img_dst.exists() and img_src.exists():
        shutil.copy2(img_src, img_dst)

    # Mask
    mask_src = PIX3D_DIR / sample["mask_path"]
    mask_dst = out_dir / "mask.png"
    if not mask_dst.exists() and mask_src.exists():
        shutil.copy2(mask_src, mask_dst)


class VisualPLYGenerator:
    def __init__(self, config_name: str):
        self.cfg = CONFIGS[config_name]
        self.config_name = config_name
        self.output_name = CONFIG_TO_FILENAME[config_name]
        self.logger = logging.LoggerAdapter(
            logging.getLogger(__name__),
            {"config": config_name},
        )
        self.pipe = None
        self.make_scene = None
        self.ready_gaussian = None

    def load_model(self):
        self.logger.info(f"Loading SAM-3D for config: {self.cfg.name}")

        sys.path.insert(0, str(SAM3D_NOTEBOOK_DIR))

        real_stdout = sys.stdout
        devnull = open(os.devnull, "w")
        try:
            sys.stdout = devnull
            from inference import Inference, make_scene, ready_gaussian_for_video_rendering

            sam3d = Inference(str(SAM3D_CONFIG), compile=False)
            self.pipe = sam3d._pipeline
            self.make_scene = make_scene
            self.ready_gaussian = ready_gaussian_for_video_rendering

            if hasattr(self.pipe, "models"):
                for name, model in self.pipe.models.items():
                    if model is not None and hasattr(model, "cuda"):
                        model.cuda()
                    if model is not None and hasattr(model, "eval"):
                        model.eval()
        finally:
            sys.stdout = real_stdout
            devnull.close()

        torch.set_grad_enabled(False)

        if self.cfg.vram_unload:
            self._unload_models()
        if self.cfg.ss_caching:
            self._setup_ss_caching()

        vram = torch.cuda.memory_allocated() / (1024**3)
        self.logger.info(f"Model VRAM: {vram:.2f}GB")

    def _unload_models(self):
        for name in ["slat_decoder_mesh", "slat_decoder_gs_4"]:
            if name in self.pipe.models and self.pipe.models[name] is not None:
                self.pipe.models[name].cpu()
                self.pipe.models[name] = None
                self.logger.info(f"Unloaded {name}")

        if hasattr(self.pipe, "depth_model") and self.pipe.depth_model is not None:
            depth = self.pipe.depth_model
            if hasattr(depth, "model") and hasattr(depth.model, "cpu"):
                depth.model.cpu()
            elif hasattr(depth, "cpu"):
                depth.cpu()
            del depth
            self.pipe.depth_model = None
            self.logger.info("Unloaded depth_model")

        gc.collect()
        torch.cuda.empty_cache()

    def _setup_ss_caching(self):
        try:
            from sam3d_objects.model.backbone.generator.flow_matching.cached_solver import CachedEuler
        except ImportError:
            fm_path = str(
                Path(SAM3D_NOTEBOOK_DIR).parent
                / "sam3d_objects" / "model" / "backbone" / "generator" / "flow_matching"
            )
            sys.path.insert(0, fm_path)
            from cached_solver import CachedEuler

        ss_gen = self.pipe.models["ss_generator"]
        ss_gen._solver = CachedEuler(
            cache_stride=self.cfg.ss_cache_stride,
            warmup_steps=self.cfg.ss_cache_warmup,
        )
        ss_gen._solver_method = "cached_euler"
        self.logger.info(f"SS caching: stride={self.cfg.ss_cache_stride}, warmup={self.cfg.ss_cache_warmup}")

    def warmup(self):
        self.logger.info("Warmup run...")
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy_mask = np.ones((256, 256), dtype=np.uint8) * 255
        pointmap = self._make_synthetic_pointmap(dummy_img)
        try:
            with torch.no_grad():
                self.pipe.run(
                    image=dummy_img, mask=dummy_mask, seed=SEED,
                    pointmap=pointmap, decode_formats=self.cfg.decode_formats,
                    stage1_inference_steps=self.cfg.stage1_steps,
                    stage2_inference_steps=self.cfg.stage2_steps,
                    with_mesh_postprocess=False, with_texture_baking=False,
                    with_layout_postprocess=False, use_vertex_color=True,
                )
        except Exception as e:
            self.logger.warning(f"Warmup error (ignored): {e}")
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info("Warmup done")

    def generate_ply(self, sample: dict, out_dir: Path) -> bool:
        ply_path = out_dir / f"{self.output_name}.ply"
        if ply_path.exists():
            self.logger.info(f"SKIP (exists): {ply_path}")
            return True

        img_path = PIX3D_DIR / sample["img_path"]
        mask_path = PIX3D_DIR / sample["mask_path"]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask_u8 = np.array(Image.open(mask_path).convert("L"))

        pointmap = self._make_synthetic_pointmap(image)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        gc.collect()
        torch.cuda.empty_cache()

        try:
            with torch.no_grad():
                output = self.pipe.run(
                    image=image, mask=mask_u8, seed=SEED,
                    pointmap=pointmap,
                    decode_formats=self.cfg.decode_formats,
                    stage1_inference_steps=self.cfg.stage1_steps,
                    stage2_inference_steps=self.cfg.stage2_steps,
                    with_mesh_postprocess=self.cfg.mesh_postprocess,
                    with_texture_baking=self.cfg.texture_baking,
                    with_layout_postprocess=False,
                    use_vertex_color=self.cfg.use_vertex_color,
                )

            scene_gs = self.make_scene(output, in_place=True)
            scene_gs = self.ready_gaussian(scene_gs, in_place=True, fix_alignment=False)
            scene_gs.save_ply(str(ply_path))
            self.logger.info(f"Saved: {ply_path}")
            return True

        except Exception as e:
            self.logger.error(f"FAIL sample {sample['sample_id']}: {e}")
            return False

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


def main():
    config_name = args.config
    output_name = CONFIG_TO_FILENAME[config_name]

    logger = logging.getLogger(__name__)
    adapter = logging.LoggerAdapter(logger, {"config": config_name})

    adapter.info("=" * 60)
    adapter.info(f"Visual PLY Generator — {config_name} → {output_name}.ply")
    adapter.info(f"  GPU: {args.gpu}")
    adapter.info("=" * 60)

    samples = load_visual_samples()
    adapter.info(f"Samples to process: {len(samples)}")

    # Copy GT + input for all samples first
    for s in samples:
        out_dir = make_output_dir(s)
        copy_gt_and_input(s, out_dir)
    adapter.info("GT + input files copied")

    # Load model and generate PLYs
    gen = VisualPLYGenerator(config_name)
    gen.load_model()
    gen.warmup()

    success = 0
    fail = 0
    t_total = time.perf_counter()

    for i, s in enumerate(samples):
        out_dir = make_output_dir(s)
        adapter.info(f"[{i+1}/{len(samples)}] {s['category']} / sample {s['sample_id']}")
        if gen.generate_ply(s, out_dir):
            success += 1
        else:
            fail += 1

    elapsed = time.perf_counter() - t_total
    adapter.info("=" * 60)
    adapter.info(f"Done: {success} success, {fail} fail in {elapsed:.1f}s")
    adapter.info(f"Output: {VISUAL_DIR}")


if __name__ == "__main__":
    main()
