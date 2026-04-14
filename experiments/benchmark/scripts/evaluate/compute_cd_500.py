"""
500개 전체 Chamfer Distance 계산 (OBB 정렬 + sign search)

입력:
  results/ply_for_cd/{original,fastsam3d,ours}/{sample_id}.ply
  gt_dimensions.json (GT model path 참조)
  experiments/seed_variance/data/pix3d/model/  (GT OBJ)

출력:
  results/cd_evaluation.csv  — 샘플별 CD
  stdout — 요약 테이블
"""

import csv
import json
import numpy as np
import trimesh
from collections import defaultdict
from itertools import product as iprod
from pathlib import Path
from scipy.spatial import cKDTree

BENCH_DIR = Path(__file__).resolve().parent.parent.parent
PLY_BASE = BENCH_DIR / "results" / "ply_for_cd"
GT_PATH = BENCH_DIR / "data" / "gt_dimensions.json"
SAMPLES_PATH = BENCH_DIR / "data" / "benchmark_samples.json"
PIX3D_DIR = BENCH_DIR.parent / "seed_variance" / "data" / "pix3d"
OUTPUT_CSV = BENCH_DIR / "results" / "cd_evaluation.csv"

VARIANTS = ["original", "fastsam3d", "ours"]
N_SURFACE_SAMPLES = 10000
N_CD_SAMPLES = 8000


def load_points(path: Path) -> np.ndarray:
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        all_pts = [g.vertices for g in mesh.geometry.values() if hasattr(g, "vertices")]
        return np.vstack(all_pts) if all_pts else np.zeros((0, 3))
    return np.array(mesh.vertices) if hasattr(mesh, "vertices") else np.zeros((0, 3))


def sample_mesh_surface(path: Path, n: int = N_SURFACE_SAMPLES) -> np.ndarray:
    mesh = trimesh.load(str(path), process=False, force="mesh")
    if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
        pts, _ = trimesh.sample.sample_surface(mesh, n)
        return pts
    return np.array(mesh.vertices) if hasattr(mesh, "vertices") else np.zeros((0, 3))


def obb_align(points: np.ndarray) -> np.ndarray:
    c = points - points.mean(axis=0)
    _, ev = np.linalg.eigh(np.cov(c.T))
    r = c @ ev
    mx = (r.max(axis=0) - r.min(axis=0)).max()
    if mx < 1e-8:
        return r
    r = r / mx
    r -= (r.max(axis=0) + r.min(axis=0)) / 2
    return r


def chamfer_distance_l2(p1: np.ndarray, p2: np.ndarray) -> float:
    """Symmetric Chamfer Distance (CD-L2): mean of squared nearest-neighbor distances."""
    d1, _ = cKDTree(p2).query(p1)
    d2, _ = cKDTree(p1).query(p2)
    return (np.square(d1).mean() + np.square(d2).mean()) / 2


def best_cd(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    pa, ga = obb_align(pred_pts), obb_align(gt_pts)
    # Subsample ONCE, then test all 8 sign flips with the SAME points
    if len(pa) > N_CD_SAMPLES:
        pa = pa[np.random.choice(len(pa), N_CD_SAMPLES, replace=False)]
    if len(ga) > N_CD_SAMPLES:
        ga = ga[np.random.choice(len(ga), N_CD_SAMPLES, replace=False)]
    return min(chamfer_distance_l2(pa * np.array(s), ga) for s in iprod([-1, 1], repeat=3))


def main():
    np.random.seed(42)

    samples = json.load(open(SAMPLES_PATH))
    print(f"Samples: {len(samples)}")

    # Pre-load GT surfaces (735 unique → cache by model_path)
    gt_cache: dict[str, np.ndarray] = {}

    rows = []
    variant_cds: dict[str, list[float]] = {v: [] for v in VARIANTS}
    cat_cds: dict[str, dict[str, list[float]]] = defaultdict(lambda: {v: [] for v in VARIANTS})

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        cat = sample["category"]
        model_path = sample["model_path"]

        # GT surface
        if model_path not in gt_cache:
            gt_obj = PIX3D_DIR / model_path
            if gt_obj.exists():
                gt_cache[model_path] = sample_mesh_surface(gt_obj)
            else:
                gt_cache[model_path] = np.zeros((0, 3))

        gt_pts = gt_cache[model_path]
        if len(gt_pts) < 10:
            continue

        row = {"sample_id": sid, "category": cat, "model_path": model_path}

        for v in VARIANTS:
            ply_path = PLY_BASE / v / f"{sid}.ply"
            if not ply_path.exists():
                row[f"cd_{v}"] = ""
                continue

            pred_pts = load_points(ply_path)
            if len(pred_pts) < 10:
                row[f"cd_{v}"] = ""
                continue

            cd = best_cd(pred_pts, gt_pts)
            row[f"cd_{v}"] = f"{cd:.8f}"  # raw CD-L2 value
            variant_cds[v].append(cd)
            cat_cds[cat][v].append(cd)

        rows.append(row)

        if (i + 1) % 50 == 0 or i + 1 == len(samples):
            print(f"[{i+1}/{len(samples)}] processed")

    # Save CSV
    fieldnames = ["sample_id", "category", "model_path"] + [f"cd_{v}" for v in VARIANTS]
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {OUTPUT_CSV}")

    # Summary
    print(f"\n{'='*70}")
    print("### Chamfer Distance — CD-L2 (×10⁻³, ↓, lower is better)")
    print(f"\n| Variant | N | CD Mean (×10⁻³) | CD Median (×10⁻³) | CD Std (×10⁻³) |")
    print(f"|---------|---|-----------------|-------------------|----------------|")
    for v in VARIANTS:
        a = np.array(variant_cds[v]) * 1000  # scale to ×10⁻³
        label = {"original": "Original", "fastsam3d": "Fast-SAM3D", "ours": "Ours"}[v]
        print(f"| {label:12s} | {len(a)//1} | {a.mean():.3f} | {np.median(a):.3f} | {a.std():.3f} |")

    # Per-category
    print(f"\n### 카테고리별 CD-L2 (×10⁻³)")
    print(f"\n| Category | N | Original | Fast-SAM3D | Ours |")
    print(f"|----------|---|----------|------------|------|")
    for cat in sorted(cat_cds):
        n = len(cat_cds[cat]["original"]) if cat_cds[cat]["original"] else 0
        vals = []
        for v in VARIANTS:
            a = cat_cds[cat][v]
            vals.append(f"{np.mean(a)*1000:.3f}" if a else "—")
        print(f"| {cat:12s} | {n:3d} | {vals[0]} | {vals[1]} | {vals[2]} |")


if __name__ == "__main__":
    main()
