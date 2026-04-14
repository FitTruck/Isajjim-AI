"""
Unified Benchmark Worker — --config flag로 baseline~o5 전환

Usage:
    python experiments/benchmark/worker_ablation.py \
        --gpu 0 --config baseline \
        --samples benchmark_samples.json --output results/original.csv

Configs: baseline, o1, o2, o4, o5
"""

import sys
import os
import argparse

# Parse args early to set CUDA_VISIBLE_DEVICES before any torch import
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--config", type=str, required=True, choices=["baseline", "o1", "o2", "o4", "o5", "o5c", "o5_slat", "o5_ss2", "o5_ss2_slat14"])
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

import numpy as np
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.float32)

from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import CONFIGS, PIX3D_DIR, SAM3D_CONFIG, SAM3D_NOTEBOOK_DIR, SEED, FLUSH_INTERVAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Worker:%(config)s] %(message)s")


class BenchmarkWorker:
    def __init__(self, config_name: str):
        self.cfg = CONFIGS[config_name]
        self.config_name = config_name
        self.logger = logging.LoggerAdapter(
            logging.getLogger(__name__),
            {"config": config_name},
        )
        self.pipe = None
        self.make_scene = None
        self.ready_gaussian = None

    def load_model(self):
        """SAM-3D 모델 로드 + config별 최적화 적용."""
        self.logger.info(f"Loading SAM-3D for config: {self.cfg.name}")

        sys.path.insert(0, str(SAM3D_NOTEBOOK_DIR))

        # Suppress stdout during model loading
        real_stdout = sys.stdout
        devnull = open(os.devnull, "w")
        try:
            sys.stdout = devnull

            from inference import Inference, make_scene, ready_gaussian_for_video_rendering

            # 항상 compile=False로 로드 (SAM3D 내부 _warmup에 run_layout_model 버그)
            sam3d = Inference(str(SAM3D_CONFIG), compile=False)
            self.pipe = sam3d._pipeline
            self.make_scene = make_scene
            self.ready_gaussian = ready_gaussian_for_video_rendering

            # Move all models to GPU
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

        # Log model VRAM before optimizations
        vram_loaded = torch.cuda.memory_allocated() / (1024**3)
        self.logger.info(f"Model VRAM after load: {vram_loaded:.2f}GB")

        if hasattr(self.pipe, "models"):
            self.logger.info(f"Loaded models: {list(self.pipe.models.keys())}")

        # ── VRAM Unload (O2) ──
        if self.cfg.vram_unload:
            self._unload_models()

        # ── SS Step Caching (O5) ──
        if self.cfg.ss_caching:
            self._setup_ss_caching()

        # ── SLAT Carving (from Fast-SAM3D) ──
        if self.cfg.slat_carving:
            self._setup_slat_carving()

        # ── Manual torch.compile (O6) ──
        if self.cfg.compile:
            self._setup_manual_compile()

        vram_final = torch.cuda.memory_allocated() / (1024**3)
        self.logger.info(f"Model VRAM after optimization: {vram_final:.2f}GB")

    def _unload_models(self):
        """미사용 모델 GPU에서 제거."""
        models_to_unload = []

        if "slat_decoder_mesh" in self.pipe.models:
            models_to_unload.append("slat_decoder_mesh")
        if "slat_decoder_gs_4" in self.pipe.models and self.pipe.models["slat_decoder_gs_4"] is not None:
            models_to_unload.append("slat_decoder_gs_4")

        for name in models_to_unload:
            model = self.pipe.models[name]
            if model is not None:
                model.cpu()
            self.pipe.models[name] = None
            self.logger.info(f"Unloaded {name} from GPU")

        # depth_model (MoGe)
        if hasattr(self.pipe, "depth_model") and self.pipe.depth_model is not None:
            depth = self.pipe.depth_model
            if hasattr(depth, "model") and hasattr(depth.model, "cpu"):
                depth.model.cpu()
            elif hasattr(depth, "cpu"):
                depth.cpu()
            del depth
            self.pipe.depth_model = None
            self.logger.info("Unloaded depth_model (MoGe) from GPU")

        gc.collect()
        torch.cuda.empty_cache()

    def _setup_manual_compile(self):
        """수동 torch.compile — SAM3D 내부 compile=True의 _warmup 버그를 우회."""
        self.logger.info("Manual torch.compile on critical modules (reduce-overhead)...")
        compile_start = time.time()
        compile_mode = "reduce-overhead"

        try:
            _dynamo = torch._dynamo
            _dynamo.config.cache_size_limit = 64
            _dynamo.config.accumulated_cache_size_limit = 2048
            _dynamo.config.capture_scalar_outputs = True

            # SS Generator backbone
            ss_gen = self.pipe.models["ss_generator"]
            if hasattr(ss_gen, "reverse_fn") and hasattr(ss_gen.reverse_fn, "inner_forward"):
                ss_gen.reverse_fn.inner_forward = torch.compile(
                    ss_gen.reverse_fn.inner_forward,
                    mode=compile_mode,
                    fullgraph=True,
                )
                self.logger.info("Compiled SS generator backbone")

            # SS Decoder
            ss_dec = self.pipe.models.get("ss_decoder")
            if ss_dec is not None:
                ss_dec.forward = torch.compile(
                    ss_dec.forward,
                    mode=compile_mode,
                    fullgraph=True,
                )
                self.logger.info("Compiled SS decoder")

            # Condition embedding
            if hasattr(self.pipe, "embed_condition"):
                self.pipe.embed_condition = torch.compile(
                    self.pipe.embed_condition,
                    mode=compile_mode,
                    fullgraph=False,
                )
                self.logger.info("Compiled condition embedding")

            self.logger.info(f"Compilation setup in {time.time() - compile_start:.1f}s")

            # AUTOTUNE warmup — 첫 실행으로 CUDA 커널 캐시 생성
            self.logger.info("Running AUTOTUNE warmup (first run ~2-5min)...")
            warmup_start = time.time()
            dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            dummy_mask = np.ones((512, 512), dtype=np.uint8) * 255
            dummy_pointmap = self._make_synthetic_pointmap(dummy_image, z=1.0)

            with torch.no_grad():
                _ = self.pipe.run(
                    image=dummy_image,
                    mask=dummy_mask,
                    seed=SEED,
                    pointmap=dummy_pointmap,
                    decode_formats=self.cfg.decode_formats,
                    stage1_inference_steps=self.cfg.stage1_steps,
                    stage2_inference_steps=self.cfg.stage2_steps,
                    with_mesh_postprocess=False,
                    with_texture_baking=False,
                    with_layout_postprocess=False,
                    use_vertex_color=True,
                )
            torch.cuda.empty_cache()
            self.logger.info(f"AUTOTUNE warmup completed in {time.time() - warmup_start:.1f}s")

        except Exception as e:
            self.logger.warning(f"Compile/warmup failed (non-fatal, continuing without compile): {e}")

    def _setup_ss_caching(self):
        """SS Generator Step Caching 설정."""
        try:
            from sam3d_objects.model.backbone.generator.flow_matching.cached_solver import CachedEuler
        except ImportError:
            try:
                fm_path = str(
                    Path(SAM3D_NOTEBOOK_DIR).parent
                    / "sam3d_objects" / "model" / "backbone" / "generator" / "flow_matching"
                )
                sys.path.insert(0, fm_path)
                from cached_solver import CachedEuler
            except ImportError:
                self.logger.warning("CachedEuler not found, skipping SS caching")
                return

        ss_gen = self.pipe.models["ss_generator"]
        ss_gen._solver = CachedEuler(
            cache_stride=self.cfg.ss_cache_stride,
            warmup_steps=self.cfg.ss_cache_warmup,
        )
        ss_gen._solver_method = "cached_euler"
        self.logger.info(
            f"SS Step Caching enabled (stride={self.cfg.ss_cache_stride}, "
            f"warmup={self.cfg.ss_cache_warmup})"
        )

    def _setup_slat_carving(self):
        """SLAT Generator에 Fast-SAM3D의 token carving 적용 (monkey-patch)."""
        FASTSAM3D_DIR = Path(__file__).resolve().parents[4] / "fast-sam3d"
        f3c_slat_dir = FASTSAM3D_DIR / "f3c_slat_end"
        cache_utils_dir = FASTSAM3D_DIR / "cache_utils_slat_end"
        fm_dir = FASTSAM3D_DIR / "sam3d_objects" / "model" / "backbone" / "generator" / "flow_matching"

        if not f3c_slat_dir.exists():
            self.logger.warning("fast-sam3d/f3c_slat_end not found, skipping SLAT carving")
            return

        # Add fast-sam3d paths for imports
        for p in [str(FASTSAM3D_DIR), str(fm_dir)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            from f3c_slat_end.f3c_leader import f3cLeader
            from f3c_slat_end.f3c_argparser import parse_f3c_args
            from f3c_slat_end.selection import AdvancedStabilityTracker
            from cache_utils_slat_end import cache_init
            from solver import Euler_end_slat
        except ImportError as e:
            self.logger.warning(f"SLAT carving import failed: {e}")
            return

        slat_gen = self.pipe.models["slat_generator"]

        # 1. Replace solver with Euler_end_slat
        slat_gen._solver = Euler_end_slat(
            thresh=self.cfg.slat_thresh,
            dir_weight=0.5,
            ret_steps=self.cfg.slat_warmup,
            full_steps=self.cfg.stage2_steps,
            carving_ratio=self.cfg.slat_carving_ratio,
        )
        slat_gen._solver_method = "euler_end_slat"

        # 2. Attach F3C components
        slat_gen.LEADER = f3cLeader()
        slat_gen.stability_tracker = AdvancedStabilityTracker()
        slat_gen.slat_params = {
            "slat_thresh": self.cfg.slat_thresh,
            "slat_warmup": self.cfg.slat_warmup,
            "slat_carving_ratio": self.cfg.slat_carving_ratio,
        }
        slat_gen.coords_scores = None
        slat_gen.map_tokens = None

        # 3. Parse f3c args for LEADER initialization
        slat_gen._f3c_args = parse_f3c_args()

        # 4. Monkey-patch generate / generate_iter to wire LEADER + stability_tracker
        import types

        original_generate_iter = slat_gen.generate_iter

        def patched_generate_iter(self_gen, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
            x_0 = self_gen._generate_noise(x_shape, x_device)
            t_seq = self_gen._prepare_t().to(x_device)

            # Set solver params from slat_params
            self_gen._solver.thresh = self_gen.slat_params["slat_thresh"]
            self_gen._solver.ret_steps = self_gen.slat_params["slat_warmup"]
            self_gen._solver.carving_ratio = self_gen.slat_params["slat_carving_ratio"]

            for x_t, t, v in self_gen._solver.solve_iter(
                self_gen._generate_dynamics,
                x_0,
                t_seq,
                self_gen.LEADER,
                self_gen.stability_tracker,
                *args_conditionals,
                **kwargs_conditionals,
            ):
                yield t, x_t, ()

        def patched_generate(self_gen, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
            B, N, C = x_shape

            # Initialize F3C state with adjusted steps
            f3c_args = self_gen._f3c_args
            f3c_args.effective_steps = self_gen.inference_steps
            f3c_args.euler_steps = self_gen.inference_steps
            # full_sampling_steps = floor(steps * 0.2) → 4 steps → 0, 최소 1
            import math
            f3c_args.full_sampling_steps = max(1, math.floor(self_gen.inference_steps * 0.4))
            f3c_args.full_sampling_end_steps = self_gen.inference_steps
            self_gen.LEADER.set_parameters(f3c_args)

            self_gen.stability_tracker.reset(
                device=x_device, num_tokens=N, latent_channels=C,
            )

            # Dummy coords_scores: uniform spatial score → motion-only carving
            if self_gen.coords_scores is None:
                self_gen.coords_scores = torch.zeros(N, 4, device=x_device)
            elif self_gen.coords_scores.shape[0] != N:
                self_gen.coords_scores = torch.zeros(N, 4, device=x_device)

            self_gen.stability_tracker.coords_scores = self_gen.coords_scores

            for _, xt, _ in self_gen.generate_iter(x_shape, x_device, *args_conditionals, **kwargs_conditionals):
                pass
            return xt

        slat_gen.generate_iter = types.MethodType(patched_generate_iter, slat_gen)
        slat_gen.generate = types.MethodType(patched_generate, slat_gen)

        self.logger.info(
            f"SLAT Carving enabled (ratio={self.cfg.slat_carving_ratio}, "
            f"thresh={self.cfg.slat_thresh}, warmup={self.cfg.slat_warmup})"
        )

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
            result["error"] = f"File not found"
            return result

        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_arr = np.array(Image.open(mask_path).convert("L"))
        mask_u8 = (mask_arr > 0).astype(np.uint8) * 255

        mask_pixels = int(np.sum(mask_u8 > 0))
        if mask_pixels == 0:
            result["error"] = "Empty mask"
            return result

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
                output = self.pipe.run(
                    image=image,
                    mask=mask_u8,
                    seed=SEED,
                    pointmap=pointmap,
                    decode_formats=self.cfg.decode_formats,
                    stage1_inference_steps=self.cfg.stage1_steps,
                    stage2_inference_steps=self.cfg.stage2_steps,
                    with_mesh_postprocess=self.cfg.mesh_postprocess,
                    with_texture_baking=self.cfg.texture_baking,
                    with_layout_postprocess=False,
                    use_vertex_color=self.cfg.use_vertex_color,
                )

            torch.cuda.synchronize()
            latency = time.perf_counter() - t_start
            vram_peak = torch.cuda.max_memory_allocated() / (1024**2)  # MB

            # Save PLY and calculate dimensions
            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                ply_path = tmp.name

            try:
                scene_gs = self.make_scene(output, in_place=True)
                scene_gs = self.ready_gaussian(scene_gs, in_place=True, fix_alignment=False)
                scene_gs.save_ply(ply_path)
                dims = self._calculate_obb_dimensions(ply_path)
            finally:
                # --save-ply-dir가 지정되면 PLY를 해당 디렉토리로 이동, 아니면 삭제
                if args.save_ply_dir and os.path.exists(ply_path):
                    save_dir = Path(args.save_ply_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    dest = save_dir / f"{sample['sample_id']}.ply"
                    import shutil
                    shutil.move(ply_path, dest)
                elif os.path.exists(ply_path):
                    os.unlink(ply_path)

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
    def _calculate_obb_dimensions(ply_path: str) -> dict:
        """PLY → PCA OBB → 정규화 → 크기순 정렬."""
        import trimesh

        pc = trimesh.load(ply_path)
        if isinstance(pc, trimesh.Scene):
            all_pts = []
            for geom in pc.geometry.values():
                if hasattr(geom, "vertices"):
                    all_pts.append(geom.vertices)
            points = np.vstack(all_pts) if all_pts else np.zeros((0, 3))
        elif hasattr(pc, "vertices"):
            points = np.array(pc.vertices)
        else:
            points = np.zeros((0, 3))

        if len(points) < 4:
            return {"dim_small": 0.0, "dim_mid": 0.0, "dim_large": 0.0}

        # PCA OBB
        centered = points - points.mean(axis=0)
        cov = np.cov(centered.T)
        _, eigenvectors = np.linalg.eigh(cov)
        rotated = centered @ eigenvectors
        obb_dims = rotated.max(axis=0) - rotated.min(axis=0)

        # 정규화 + 크기순 정렬 (ascending)
        max_dim = obb_dims.max()
        if max_dim < 1e-8:
            return {"dim_small": 0.0, "dim_mid": 0.0, "dim_large": 0.0}

        normalized = sorted((obb_dims / max_dim).tolist())

        # sorted: [smallest, middle, largest=1.0]
        return {
            "dim_small": normalized[0],
            "dim_mid": normalized[1],
            "dim_large": normalized[2],
        }


def load_completed_ids(output_path: str) -> set:
    """이미 처리된 sample_id 목록 (중단 복구용). 성공/실패 무관하게 스킵."""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(int(row["sample_id"]))
    return completed


def main():
    cfg = CONFIGS[args.config]
    logger = logging.LoggerAdapter(
        logging.getLogger("worker"),
        {"config": args.config},
    )

    logger.info("=" * 60)
    logger.info(f"Benchmark Worker: {cfg.name}")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  GPU: {args.gpu}")
    logger.info(f"  Samples: {args.samples}")
    logger.info(f"  Output: {args.output}")
    logger.info("=" * 60)

    # GPU check
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

    # 중단 복구: 이미 완료된 항목 스킵
    completed = load_completed_ids(args.output)
    if completed:
        logger.info(f"Resuming: {len(completed)} already completed, skipping")
    remaining = [s for s in samples if s["sample_id"] not in completed]
    logger.info(f"Remaining: {len(remaining)}")

    if not remaining:
        logger.info("All samples already completed!")
        return

    # Load model
    worker = BenchmarkWorker(args.config)
    worker.load_model()

    # Warmup (별도 dummy — 실제 샘플과 분리하여 latency 편향 방지)
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

    # Open CSV (context manager로 crash 안전성 확보)
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
    t_total_start = time.time()

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

            # Progress
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

            # Flush + fsync (장시간 실행 안정성)
            if done % FLUSH_INTERVAL == 0:
                csv_file.flush()
                os.fsync(csv_file.fileno())

    # Summary
    elapsed_total = time.time() - t_total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"완료: {cfg.name}")
    logger.info(f"  성공: {success_count}/{len(remaining)}")
    logger.info(f"  실패: {fail_count}")
    if latencies:
        logger.info(f"  평균 Latency: {np.mean(latencies):.1f}s")
        logger.info(f"  중간값 Latency: {np.median(latencies):.1f}s")
    logger.info(f"  총 소요: {elapsed_total/3600:.1f}h")
    logger.info(f"  결과: {args.output}")


if __name__ == "__main__":
    main()
