"""
Seed Variance Experiment Worker

GPU별 독립 프로세스로 SAM-3D를 실행하여 PLY + OBB 치수를 추출하는 워커.
persistent_3d_worker.py 패턴을 기반으로 실험용으로 수정.

Usage:
    python experiment_worker.py <gpu_id> <phase> <tasks_json_path> <output_csv_path>

    phase: "original" or "optimized"
"""

import sys
import os

# ============================================================================
# CRITICAL: Set environment variables BEFORE importing torch/spconv
# ============================================================================
if len(sys.argv) >= 2:
    gpu_id = int(sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME") or os.environ.get("CONDA_PREFIX") or "/usr/local/cuda"
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = os.environ.get("SPCONV_ALGO_TIME_LIMIT", "100")
os.environ["TORCH_CUDA_ARCH_LIST"] = "all"
os.environ["WARP_QUIET"] = "1"

# Persistent autotune cache
_autotune_cache_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache", "torch_compile"
)
os.makedirs(_autotune_cache_dir, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _autotune_cache_dir
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

# Thread limits
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import json
import csv
import time
import tempfile
import logging
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.float32)

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [Worker GPU{gpu_id}] %(message)s",
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAM3D_NOTEBOOK_DIR = PROJECT_ROOT / "sam-3d-objects" / "notebook"
SAM3D_CONFIG = PROJECT_ROOT / "sam-3d-objects" / "checkpoints" / "hf" / "pipeline.yaml"

# Add SAM-3D notebook to path
sys.path.insert(0, str(SAM3D_NOTEBOOK_DIR))


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
    """
    PLY 파일에서 OBB 기반 상대 치수(W, D, H) 계산.
    DimensionCalculator와 동일한 로직.
    """
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
        return {"width": 0.0, "depth": 0.0, "height": 0.0}

    # PCA-based OBB
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Project points onto principal axes
    rotated = centered @ eigenvectors
    obb_min = rotated.min(axis=0)
    obb_max = rotated.max(axis=0)
    obb_dims = obb_max - obb_min

    # Greedy mapping: OBB axes → coordinate axes (X=width, Y=height, Z=depth)
    similarity = np.abs(eigenvectors.T)  # (3, 3)

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

    return {"width": dims[0], "depth": dims[2], "height": dims[1]}


def add_rgb_to_ply(ply_path: str):
    """Post-process PLY to add RGB from SH coefficients (binary format)."""
    with open(ply_path, "rb") as f:
        data = f.read()

    text_data = data.decode("utf-8", errors="ignore")
    header_end = text_data.find("end_header")
    if header_end == -1:
        return

    header_text = text_data[:header_end + len("end_header")]
    header_lines = header_text.split("\n")

    vertex_count = 0
    properties = []
    property_types = {}

    for line in header_lines:
        line = line.strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            prop_type = parts[1]
            prop_name = parts[2]
            properties.append(prop_name)
            property_types[prop_name] = prop_type

    # Check if SH coefficients exist
    if "f_dc_0" not in properties:
        return

    numpy_dtype = []
    for prop_name in properties:
        if property_types[prop_name] == "float":
            numpy_dtype.append((prop_name, "<f4"))
        elif property_types[prop_name] in ["uchar", "uint8"]:
            numpy_dtype.append((prop_name, "u1"))

    binary_start = len(header_text.encode("utf-8")) + 1
    binary_data = data[binary_start:]

    vertices = np.frombuffer(binary_data, dtype=np.dtype(numpy_dtype), count=vertex_count)

    SH0 = 0.282095
    f_dc = np.column_stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]
    ).astype(np.float32)

    f_dc_tensor = torch.from_numpy(f_dc).cuda()
    rgb = torch.clamp(f_dc_tensor * SH0 + 0.5, 0.0, 1.0)
    rgb_u8 = (rgb * 255).to(torch.uint8).cpu().numpy()

    # Build new PLY with RGB
    new_header_lines = []
    rgb_added = False
    for line in header_lines:
        new_header_lines.append(line)
        if line.strip().startswith("property") and "f_dc_2" in line and not rgb_added:
            new_header_lines.append("property uchar red")
            new_header_lines.append("property uchar green")
            new_header_lines.append("property uchar blue")
            rgb_added = True

    new_header = "\n".join(new_header_lines) + "\n"

    # Build binary data
    import struct
    with open(ply_path, "wb") as f:
        f.write(new_header.encode("utf-8"))
        for i in range(vertex_count):
            for prop_name in properties:
                if property_types[prop_name] == "float":
                    f.write(struct.pack("<f", float(vertices[prop_name][i])))
                elif property_types[prop_name] in ["uchar", "uint8"]:
                    f.write(struct.pack("B", int(vertices[prop_name][i])))
                if prop_name == "f_dc_2":
                    f.write(struct.pack("BBB", rgb_u8[i, 0], rgb_u8[i, 1], rgb_u8[i, 2]))


class ExperimentWorker:
    """SAM-3D 실험 워커"""

    def __init__(self, phase: str):
        self.phase = phase
        self.pipe = None
        self.make_scene = None
        self.ready_gaussian = None

    def load_model(self):
        """SAM-3D 모델 로드"""
        logger.info(f"Loading SAM-3D model (phase={self.phase})...")
        start = time.time()

        from inference import (
            Inference, make_scene, ready_gaussian_for_video_rendering,
        )

        compile_model = False
        self.inference = Inference(str(SAM3D_CONFIG), compile=compile_model)
        self.pipe = self.inference._pipeline

        # Scene utilities (module-level functions in inference.py)
        self.make_scene = make_scene
        self.ready_gaussian = ready_gaussian_for_video_rendering

        # Setup SS Step Caching if optimized phase
        if self.phase == "optimized":
            self._setup_ss_caching()

        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.1f}s")

        # Log GPU memory
        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info(f"GPU memory after model load: {mem_mb:.0f}MB")

    def _setup_ss_caching(self):
        """Setup SS Generator Step Caching for optimized phase."""
        try:
            from sam3d_objects.model.backbone.generator.flow_matching.cached_solver import CachedEuler
        except ImportError:
            try:
                sys.path.insert(0, str(PROJECT_ROOT / "sam-3d-objects" / "sam3d_objects" / "model" / "backbone" / "generator" / "flow_matching"))
                from cached_solver import CachedEuler
            except ImportError:
                logger.warning("CachedEuler not found, skipping SS caching")
                return

        try:
            ss_gen = self.pipe.models["ss_generator"]
            ss_gen._solver = CachedEuler(cache_stride=3, warmup_steps=2)
            ss_gen._solver_method = "cached_euler"
            logger.info("SS Step Caching enabled (stride=3, warmup=2)")
        except Exception as e:
            logger.warning(f"SS Caching setup failed: {e}")

    def run_inference(self, image_path: str, mask_path: str, seed: int) -> dict:
        """
        단일 이미지에 대해 SAM-3D 추론 실행.

        Returns:
            {"width": float, "depth": float, "height": float, "time": float, "success": bool}
        """
        start = time.time()

        try:
            # Load image and mask
            image_pil = Image.open(image_path).convert("RGB")
            image = np.array(image_pil)

            mask_pil = Image.open(mask_path).convert("L")
            mask = np.array(mask_pil)
            mask_u8 = (mask > 0).astype(np.uint8) * 255

            if mask_u8.sum() == 0:
                raise ValueError("Empty mask")

            # Set seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Clear GPU cache
            torch.cuda.empty_cache()

            # Create pointmap (synthetic for both phases - MoGe not available)
            # Note: MoGe is not installed, using synthetic pinhole pointmap
            # This is consistent with production usage and avoids intrinsics recovery failure
            pointmap = make_synthetic_pointmap(image, z=1.0)

            # Config based on phase
            if self.phase == "original":
                s1_steps = 25
                s2_steps = 25
                mesh_pp = True
                tex_bake = True
            else:
                s1_steps = 14
                s2_steps = 4
                mesh_pp = False
                tex_bake = False

            # Run inference
            with torch.no_grad():
                output = self.pipe.run(
                    image=image,
                    mask=mask_u8,
                    seed=seed,
                    pointmap=pointmap,
                    decode_formats=["gaussian"],
                    stage1_inference_steps=s1_steps,
                    stage2_inference_steps=s2_steps,
                    with_mesh_postprocess=mesh_pp,
                    with_texture_baking=tex_bake,
                    with_layout_postprocess=False,
                    use_vertex_color=True,
                )

            torch.cuda.synchronize()

            # Extract Gaussian and save PLY
            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                ply_path = tmp.name

            try:
                scene_gs = self.make_scene(output, in_place=True)
                scene_gs = self.ready_gaussian(scene_gs, in_place=True, fix_alignment=False)
                scene_gs.save_ply(ply_path)

                # Post-process PLY
                try:
                    add_rgb_to_ply(ply_path)
                except Exception:
                    pass

                # Calculate OBB dimensions
                dims = calculate_obb_dimensions(ply_path)
                elapsed = time.time() - start

                return {
                    "width": dims["width"],
                    "depth": dims["depth"],
                    "height": dims["height"],
                    "time": elapsed,
                    "success": True,
                    "error": "",
                }
            finally:
                if os.path.exists(ply_path):
                    os.unlink(ply_path)

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "width": 0.0,
                "depth": 0.0,
                "height": 0.0,
                "time": elapsed,
                "success": False,
                "error": str(e),
            }


def main():
    if len(sys.argv) < 5:
        print("Usage: python experiment_worker.py <gpu_id> <phase> <tasks_json> <output_csv>")
        print("  phase: 'original' or 'optimized'")
        sys.exit(1)

    gpu_id = int(sys.argv[1])
    phase = sys.argv[2]
    tasks_json_path = sys.argv[3]
    output_csv_path = sys.argv[4]

    assert phase in ("original", "optimized"), f"Invalid phase: {phase}"

    # Load tasks
    with open(tasks_json_path, 'r') as f:
        tasks = json.load(f)

    logger.info(f"Phase: {phase}, GPU: {gpu_id}, Tasks: {len(tasks)}")

    # Initialize worker
    worker = ExperimentWorker(phase=phase)
    worker.load_model()

    # Process tasks
    results = []
    for i, task in enumerate(tasks):
        category = task["category"]
        img_path = task["img_path"]
        mask_path = task["mask_path"]
        seed = task["seed"]
        sample_id = task.get("sample_id", "")

        logger.info(f"[{i+1}/{len(tasks)}] {category}/{sample_id} seed={seed}")

        result = worker.run_inference(img_path, mask_path, seed)

        results.append({
            "category": category,
            "sample_id": sample_id,
            "seed": seed,
            "width": result["width"],
            "depth": result["depth"],
            "height": result["height"],
            "time_seconds": result["time"],
            "success": result["success"],
            "error": result["error"],
        })

        # Save intermediate results
        if (i + 1) % 10 == 0 or i == len(tasks) - 1:
            _save_csv(results, output_csv_path)
            logger.info(f"  Saved {len(results)} results to {output_csv_path}")

    logger.info(f"Done. Total: {len(results)} tasks, {sum(1 for r in results if r['success'])} succeeded")


def _save_csv(results: list, path: str):
    """Save results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["category", "sample_id", "seed", "width", "depth", "height",
                   "time_seconds", "success", "error"]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
