"""
3종 비교 결과 집계 + 평가 지표(RDE, DA@τ, VE) 계산

입력:
  results/original.csv   (R1)
  results/fastsam3d.csv  (R2)
  results/ours.csv       (R6)
  gt_dimensions.json
  benchmark_samples.json

출력:
  results/summary_3way.csv          — 전체 성능 비교 (Table 1)
  results/evaluation_3way.csv       — 치수 추정 정확도 (Table 2)
  results/category_latency.csv      — 카테고리별 Latency (Table 4)
  results/category_rde.csv          — 카테고리별 RDE
  stdout                            — 논문용 마크다운 테이블
"""

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BENCH_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BENCH_DIR / "results"
GT_PATH = BENCH_DIR / "data" / "gt_dimensions.json"
SAMPLES_PATH = BENCH_DIR / "data" / "benchmark_samples.json"

VARIANTS = {
    "Original": RESULTS_DIR / "original.csv",
    "Fast-SAM3D": RESULTS_DIR / "fastsam3d.csv",
    "Ours": RESULTS_DIR / "ours.csv",
}

DA_THRESHOLDS = [5, 10, 20]  # DA@τ (%)


# ================================================================
# Data structures
# ================================================================
@dataclass
class SampleResult:
    sample_id: int
    category: str
    model_path: str
    dim_small: float
    dim_mid: float
    dim_large: float
    latency: float
    vram_peak_mb: float
    success: bool


def load_csv(path: Path) -> list[SampleResult]:
    results = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(SampleResult(
                sample_id=int(row["sample_id"]),
                category=row["category"],
                model_path=row["model_path"],
                dim_small=float(row["dim_small"]) if row["success"] == "True" else 0.0,
                dim_mid=float(row["dim_mid"]) if row["success"] == "True" else 0.0,
                dim_large=float(row["dim_large"]) if row["success"] == "True" else 0.0,
                latency=float(row["latency_seconds"]) if row["latency_seconds"] else 0.0,
                vram_peak_mb=float(row["vram_peak_mb"]) if row["vram_peak_mb"] else 0.0,
                success=row["success"] == "True",
            ))
    return results


def load_gt() -> dict[str, list[float]]:
    """model_path -> [dim_small, dim_mid, dim_large] (normalized, sorted)"""
    with open(GT_PATH) as f:
        raw = json.load(f)
    return {k: v["normalized_sorted"] for k, v in raw.items()}


# ================================================================
# Metrics
# ================================================================
def compute_rde(gt: list[float], pred: list[float]) -> float:
    """Relative Dimension Error (mean of 3 axes)"""
    total = 0.0
    for g, p in zip(gt, pred):
        if g > 1e-6:
            total += abs(g - p) / g
    return total / 3.0


def compute_ve(gt: list[float], pred: list[float]) -> float:
    """Volume Error"""
    gt_vol = gt[0] * gt[1] * gt[2]
    pred_vol = pred[0] * pred[1] * pred[2]
    if gt_vol > 1e-9:
        return abs(gt_vol - pred_vol) / gt_vol
    return 0.0


def compute_metrics(
    results: list[SampleResult],
    gt_dims: dict[str, list[float]],
) -> dict:
    """Compute all metrics for a variant."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    latencies = [r.latency for r in successful]
    vram_peaks = [r.vram_peak_mb for r in successful]

    rdes = []
    ves = []
    matched = 0
    unmatched_models = set()

    for r in successful:
        gt = gt_dims.get(r.model_path)
        if gt is None:
            unmatched_models.add(r.model_path)
            continue
        matched += 1
        pred = [r.dim_small, r.dim_mid, r.dim_large]
        rdes.append(compute_rde(gt, pred))
        ves.append(compute_ve(gt, pred))

    da_scores = {}
    for tau in DA_THRESHOLDS:
        threshold = tau / 100.0
        da_scores[tau] = sum(1 for rde in rdes if rde <= threshold) / len(rdes) * 100 if rdes else 0

    return {
        "total": len(results),
        "success": len(successful),
        "fail": len(failed),
        "success_rate": len(successful) / len(results) * 100,
        "matched": matched,
        "unmatched_models": len(unmatched_models),
        # Latency
        "latency_mean": sum(latencies) / len(latencies) if latencies else 0,
        "latency_median": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "latency_std": (sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies)) ** 0.5 if latencies else 0,
        "latency_min": min(latencies) if latencies else 0,
        "latency_max": max(latencies) if latencies else 0,
        "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        # VRAM
        "vram_mean": sum(vram_peaks) / len(vram_peaks) if vram_peaks else 0,
        "vram_max": max(vram_peaks) if vram_peaks else 0,
        "vram_min": min(vram_peaks) if vram_peaks else 0,
        # Quality
        "rde_mean": sum(rdes) / len(rdes) * 100 if rdes else 0,
        "rde_median": sorted(rdes)[len(rdes) // 2] * 100 if rdes else 0,
        "rde_std": (sum((x - sum(rdes) / len(rdes)) ** 2 for x in rdes) / len(rdes)) ** 0.5 * 100 if rdes else 0,
        "da_scores": da_scores,
        "ve_mean": sum(ves) / len(ves) * 100 if ves else 0,
        "ve_median": sorted(ves)[len(ves) // 2] * 100 if ves else 0,
        # Raw lists for category breakdown
        "_rdes": rdes,
        "_ves": ves,
        "_latencies_by_sample": {r.sample_id: r.latency for r in successful},
        "_rdes_by_sample": {},
    }


def compute_category_stats(
    results: list[SampleResult],
    gt_dims: dict[str, list[float]],
) -> dict[str, dict]:
    """Per-category metrics."""
    by_cat: dict[str, list[SampleResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    stats = {}
    for cat, cat_results in sorted(by_cat.items()):
        successful = [r for r in cat_results if r.success]
        latencies = [r.latency for r in successful]
        rdes = []
        for r in successful:
            gt = gt_dims.get(r.model_path)
            if gt is None:
                continue
            pred = [r.dim_small, r.dim_mid, r.dim_large]
            rdes.append(compute_rde(gt, pred))

        stats[cat] = {
            "n": len(cat_results),
            "success": len(successful),
            "latency_mean": sum(latencies) / len(latencies) if latencies else 0,
            "latency_std": (sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies)) ** 0.5 if latencies else 0,
            "rde_mean": sum(rdes) / len(rdes) * 100 if rdes else 0,
            "rde_std": (sum((x - sum(rdes) / len(rdes)) ** 2 for x in rdes) / len(rdes)) ** 0.5 * 100 if rdes else 0,
        }
    return stats


# ================================================================
# Output
# ================================================================
def print_table1(all_metrics: dict[str, dict]) -> str:
    """Table 1: Performance comparison (3-way)"""
    baseline_lat = all_metrics["Original"]["latency_mean"]
    lines = []
    lines.append("### Table 1: 전체 성능 비교 (3종, N=500)")
    lines.append("")
    lines.append("| Variant | Latency (s) | Speedup | Latency P95 (s) | Peak VRAM (GB) | Success Rate |")
    lines.append("|---------|------------|---------|-----------------|----------------|--------------|")
    for name in ["Original", "Fast-SAM3D", "Ours"]:
        m = all_metrics[name]
        speedup = baseline_lat / m["latency_mean"] if m["latency_mean"] > 0 else 0
        lines.append(
            f"| {name} | {m['latency_mean']:.2f} +/- {m['latency_std']:.2f} | "
            f"{speedup:.1f}x | {m['latency_p95']:.2f} | "
            f"{m['vram_max'] / 1024:.2f} | {m['success_rate']:.1f}% |"
        )
    return "\n".join(lines)


def print_table2(all_metrics: dict[str, dict]) -> str:
    """Table 2: Dimension accuracy"""
    lines = []
    lines.append("### Table 2: 치수 추정 정확도 (vs Pix3D GT)")
    lines.append("")
    lines.append("| Variant | RDE (%, ↓) | DA@5 (%, ↑) | DA@10 (%, ↑) | DA@20 (%, ↑) | VE (%, ↓) |")
    lines.append("|---------|-----------|-------------|--------------|--------------|----------|")
    for name in ["Original", "Fast-SAM3D", "Ours"]:
        m = all_metrics[name]
        da = m["da_scores"]
        lines.append(
            f"| {name} | {m['rde_mean']:.2f} +/- {m['rde_std']:.2f} | "
            f"{da[5]:.1f} | {da[10]:.1f} | {da[20]:.1f} | "
            f"{m['ve_mean']:.2f} |"
        )
    return "\n".join(lines)


def print_table4(all_cat_stats: dict[str, dict[str, dict]]) -> str:
    """Table 4: Per-category latency"""
    categories = list(all_cat_stats["Original"].keys())
    lines = []
    lines.append("### Table 4: 카테고리별 Latency (N=500)")
    lines.append("")
    lines.append("| Category | N | Original (s) | Fast-SAM3D (s) | Ours (s) |")
    lines.append("|----------|---|-------------|---------------|---------|")
    for cat in categories:
        n = all_cat_stats["Original"][cat]["n"]
        orig = all_cat_stats["Original"][cat]["latency_mean"]
        fast = all_cat_stats["Fast-SAM3D"][cat]["latency_mean"]
        ours = all_cat_stats["Ours"][cat]["latency_mean"]
        lines.append(f"| {cat} | {n} | {orig:.2f} | {fast:.2f} | {ours:.2f} |")
    return "\n".join(lines)


def print_table_cat_rde(all_cat_stats: dict[str, dict[str, dict]]) -> str:
    """Category-wise RDE"""
    categories = list(all_cat_stats["Original"].keys())
    lines = []
    lines.append("### 카테고리별 RDE (%)")
    lines.append("")
    lines.append("| Category | N | Original | Fast-SAM3D | Ours |")
    lines.append("|----------|---|----------|------------|------|")
    for cat in categories:
        n = all_cat_stats["Original"][cat]["n"]
        orig = all_cat_stats["Original"][cat]["rde_mean"]
        fast = all_cat_stats["Fast-SAM3D"][cat]["rde_mean"]
        ours = all_cat_stats["Ours"][cat]["rde_mean"]
        lines.append(f"| {cat} | {n} | {orig:.2f} | {fast:.2f} | {ours:.2f} |")
    return "\n".join(lines)


def save_summary_csv(all_metrics: dict[str, dict]):
    path = RESULTS_DIR / "summary_3way.csv"
    baseline_lat = all_metrics["Original"]["latency_mean"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "variant", "n_total", "n_success", "n_fail", "success_rate",
            "latency_mean", "latency_std", "latency_median", "latency_p95",
            "latency_min", "latency_max", "speedup",
            "vram_mean_gb", "vram_max_gb",
            "rde_mean_pct", "rde_std_pct", "rde_median_pct",
            "da5_pct", "da10_pct", "da20_pct",
            "ve_mean_pct", "ve_median_pct",
        ])
        for name in ["Original", "Fast-SAM3D", "Ours"]:
            m = all_metrics[name]
            speedup = baseline_lat / m["latency_mean"] if m["latency_mean"] > 0 else 0
            w.writerow([
                name, m["total"], m["success"], m["fail"],
                f"{m['success_rate']:.1f}",
                f"{m['latency_mean']:.3f}", f"{m['latency_std']:.3f}",
                f"{m['latency_median']:.3f}", f"{m['latency_p95']:.3f}",
                f"{m['latency_min']:.3f}", f"{m['latency_max']:.3f}",
                f"{speedup:.2f}",
                f"{m['vram_mean'] / 1024:.3f}", f"{m['vram_max'] / 1024:.3f}",
                f"{m['rde_mean']:.3f}", f"{m['rde_std']:.3f}", f"{m['rde_median']:.3f}",
                f"{m['da_scores'][5]:.1f}", f"{m['da_scores'][10]:.1f}", f"{m['da_scores'][20]:.1f}",
                f"{m['ve_mean']:.3f}", f"{m['ve_median']:.3f}",
            ])
    print(f"Saved: {path}")


def save_category_csv(all_cat_stats: dict[str, dict[str, dict]]):
    path = RESULTS_DIR / "category_stats_3way.csv"
    categories = list(all_cat_stats["Original"].keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "n", "variant", "latency_mean", "latency_std", "rde_mean_pct", "rde_std_pct"])
        for cat in categories:
            for name in ["Original", "Fast-SAM3D", "Ours"]:
                s = all_cat_stats[name][cat]
                w.writerow([
                    cat, s["n"], name,
                    f"{s['latency_mean']:.3f}", f"{s['latency_std']:.3f}",
                    f"{s['rde_mean']:.3f}", f"{s['rde_std']:.3f}",
                ])
    print(f"Saved: {path}")


# ================================================================
# Main
# ================================================================
def main():
    gt_dims = load_gt()
    print(f"GT dimensions loaded: {len(gt_dims)} unique models")

    all_metrics: dict[str, dict] = {}
    all_cat_stats: dict[str, dict[str, dict]] = {}
    all_results: dict[str, list[SampleResult]] = {}

    for name, path in VARIANTS.items():
        if not path.exists():
            print(f"SKIP: {path} not found")
            continue
        results = load_csv(path)
        all_results[name] = results
        metrics = compute_metrics(results, gt_dims)
        all_metrics[name] = metrics
        all_cat_stats[name] = compute_category_stats(results, gt_dims)
        print(f"Loaded {name}: {metrics['success']}/{metrics['total']} success, "
              f"{metrics['matched']} GT matched")

    print("\n" + "=" * 70)
    print(print_table1(all_metrics))
    print()
    print(print_table2(all_metrics))
    print()
    print(print_table4(all_cat_stats))
    print()
    print(print_table_cat_rde(all_cat_stats))

    # Additional details
    print("\n" + "=" * 70)
    print("### 추가 통계")
    print()
    for name in ["Original", "Fast-SAM3D", "Ours"]:
        m = all_metrics[name]
        print(f"**{name}**:")
        print(f"  Latency: mean={m['latency_mean']:.2f}s, median={m['latency_median']:.2f}s, "
              f"min={m['latency_min']:.2f}s, max={m['latency_max']:.2f}s")
        print(f"  VRAM Peak: mean={m['vram_mean']/1024:.2f}GB, max={m['vram_max']/1024:.2f}GB, "
              f"min={m['vram_min']/1024:.2f}GB")
        print(f"  RDE: mean={m['rde_mean']:.2f}%, median={m['rde_median']:.2f}%, std={m['rde_std']:.2f}%")
        print(f"  VE: mean={m['ve_mean']:.2f}%, median={m['ve_median']:.2f}%")
        print(f"  GT matched: {m['matched']}, unmatched models: {m['unmatched_models']}")
        print()

    # Save CSVs
    save_summary_csv(all_metrics)
    save_category_csv(all_cat_stats)

    # Ours vs Original dimension comparison (sample-level)
    print("=" * 70)
    print("### Ours vs Original: 치수 차이 분석")
    if "Original" in all_results and "Ours" in all_results:
        orig_map = {r.sample_id: r for r in all_results["Original"] if r.success}
        ours_map = {r.sample_id: r for r in all_results["Ours"] if r.success}
        common = set(orig_map.keys()) & set(ours_map.keys())
        diffs = []
        for sid in common:
            o, u = orig_map[sid], ours_map[sid]
            gt = gt_dims.get(o.model_path)
            if gt is None:
                continue
            rde_o = compute_rde(gt, [o.dim_small, o.dim_mid, o.dim_large])
            rde_u = compute_rde(gt, [u.dim_small, u.dim_mid, u.dim_large])
            diffs.append(rde_u - rde_o)
        if diffs:
            mean_diff = sum(diffs) / len(diffs) * 100
            pos = sum(1 for d in diffs if d > 0)  # Ours worse
            neg = sum(1 for d in diffs if d < 0)  # Ours better
            zero = sum(1 for d in diffs if abs(d) < 1e-6)
            print(f"  Paired samples: {len(diffs)}")
            print(f"  Mean RDE difference (Ours - Original): {mean_diff:+.3f}%p")
            print(f"  Ours better: {neg} ({neg/len(diffs)*100:.1f}%), "
                  f"Ours worse: {pos} ({pos/len(diffs)*100:.1f}%), "
                  f"Equal: {zero} ({zero/len(diffs)*100:.1f}%)")


if __name__ == "__main__":
    main()
