"""
SS=2 vs SS=14 비교 분석 (동일 52개 샘플)

Usage:
    conda run -n sam3d-objects python experiments/benchmark/compare_ss2_vs_ss14.py
"""

import csv
import json
import numpy as np
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BENCH_DIR / "results"
GT_PATH = BENCH_DIR / "data" / "gt_dimensions.json"

SS2_PATH = RESULTS_DIR / "test_o5_ss2_52.csv"
SS14_PATH = RESULTS_DIR / "test_o5_ss14_52.csv"


def load_results(path):
    results = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            sid = int(row["sample_id"])
            results[sid] = {
                "category": row["category"],
                "model_path": row["model_path"],
                "dims": [float(row["dim_small"]), float(row["dim_mid"]), float(row["dim_large"])],
                "latency": float(row["latency_seconds"]) if row["latency_seconds"] else 0,
                "vram_peak": float(row["vram_peak_mb"]) if row["vram_peak_mb"] else 0,
                "success": row["success"] == "True",
            }
    return results


def compute_rde(gt, pred):
    return sum(abs(g - p) / g for g, p in zip(gt, pred)) / 3


def compute_ve(gt, pred):
    gt_vol = gt[0] * gt[1] * gt[2]
    pred_vol = pred[0] * pred[1] * pred[2]
    return abs(gt_vol - pred_vol) / gt_vol if gt_vol > 1e-9 else 0


def main():
    gt_all = json.load(open(GT_PATH))
    gt_dims = {k: v["normalized_sorted"] for k, v in gt_all.items()}

    ss2 = load_results(SS2_PATH)
    ss14 = load_results(SS14_PATH)

    print(f"SS=2  samples: {len(ss2)} (success: {sum(1 for v in ss2.values() if v['success'])})")
    print(f"SS=14 samples: {len(ss14)} (success: {sum(1 for v in ss14.values() if v['success'])})")

    common = sorted(set(ss2) & set(ss14))
    common = [s for s in common if ss2[s]["success"] and ss14[s]["success"]]
    print(f"Common successful: {len(common)}")

    # ── Latency comparison ──
    lat2 = np.array([ss2[s]["latency"] for s in common])
    lat14 = np.array([ss14[s]["latency"] for s in common])
    vram2 = np.array([ss2[s]["vram_peak"] for s in common])
    vram14 = np.array([ss14[s]["vram_peak"] for s in common])

    print("\n" + "=" * 70)
    print("### Latency (seconds)")
    print(f"  SS=14: mean={lat14.mean():.3f}, median={np.median(lat14):.3f}, std={lat14.std():.3f}")
    print(f"  SS=2:  mean={lat2.mean():.3f}, median={np.median(lat2):.3f}, std={lat2.std():.3f}")
    print(f"  Speedup: {lat14.mean() / lat2.mean():.2f}x")
    print(f"  Saved per sample: {lat14.mean() - lat2.mean():.3f}s")

    print("\n### VRAM Peak (MB)")
    print(f"  SS=14: mean={vram14.mean():.1f}, max={vram14.max():.1f}")
    print(f"  SS=2:  mean={vram2.mean():.1f}, max={vram2.max():.1f}")

    # ── RDE comparison ──
    rde2_list, rde14_list = [], []
    ve2_list, ve14_list = [], []
    rde_diffs = []  # ss2 - ss14

    for sid in common:
        gt = gt_dims.get(ss2[sid]["model_path"])
        if gt is None:
            continue

        rde2 = compute_rde(gt, ss2[sid]["dims"])
        rde14 = compute_rde(gt, ss14[sid]["dims"])
        ve2 = compute_ve(gt, ss2[sid]["dims"])
        ve14 = compute_ve(gt, ss14[sid]["dims"])

        rde2_list.append(rde2)
        rde14_list.append(rde14)
        ve2_list.append(ve2)
        ve14_list.append(ve14)
        rde_diffs.append(rde2 - rde14)

    rde2 = np.array(rde2_list) * 100
    rde14 = np.array(rde14_list) * 100
    ve2 = np.array(ve2_list) * 100
    ve14 = np.array(ve14_list) * 100
    diffs = np.array(rde_diffs) * 100

    print(f"\n### RDE (%, lower is better) — N={len(rde2)}")
    print(f"  SS=14: mean={rde14.mean():.3f}, median={np.median(rde14):.3f}, std={rde14.std():.3f}")
    print(f"  SS=2:  mean={rde2.mean():.3f}, median={np.median(rde2):.3f}, std={rde2.std():.3f}")
    print(f"  Δ(SS2-SS14): mean={diffs.mean():+.3f}%p, |mean|={np.abs(diffs).mean():.3f}%p")

    da_thresholds = [5, 10, 20]
    print("\n### DA@τ (%, higher is better)")
    for tau in da_thresholds:
        t = tau / 100.0
        da2 = (rde2 / 100 <= t).sum() / len(rde2) * 100
        da14 = (rde14 / 100 <= t).sum() / len(rde14) * 100
        print(f"  DA@{tau:2d}: SS=14={da14:.1f}%, SS=2={da2:.1f}%, Δ={da2-da14:+.1f}%p")

    print(f"\n### VE (%, lower is better)")
    print(f"  SS=14: mean={ve14.mean():.3f}, median={np.median(ve14):.3f}")
    print(f"  SS=2:  mean={ve2.mean():.3f}, median={np.median(ve2):.3f}")

    # ── Paired analysis ──
    print(f"\n### Paired analysis (SS=2 vs SS=14)")
    better = (diffs < 0).sum()
    worse = (diffs > 0).sum()
    within_1 = (np.abs(diffs) < 1).sum()
    within_2 = (np.abs(diffs) < 2).sum()
    print(f"  SS=2 better: {better}/{len(diffs)} ({better/len(diffs)*100:.1f}%)")
    print(f"  SS=2 worse:  {worse}/{len(diffs)} ({worse/len(diffs)*100:.1f}%)")
    print(f"  Within 1%p:  {within_1}/{len(diffs)} ({within_1/len(diffs)*100:.1f}%)")
    print(f"  Within 2%p:  {within_2}/{len(diffs)} ({within_2/len(diffs)*100:.1f}%)")

    # ── Per-axis ──
    print(f"\n### Per-axis comparison")
    axis_names = ["small", "mid", "large"]
    for i, name in enumerate(axis_names):
        diffs_axis = []
        for sid in common:
            gt = gt_dims.get(ss2[sid]["model_path"])
            if gt is None:
                continue
            err2 = abs(gt[i] - ss2[sid]["dims"][i]) / gt[i] * 100
            err14 = abs(gt[i] - ss14[sid]["dims"][i]) / gt[i] * 100
            diffs_axis.append(err2 - err14)
        da = np.array(diffs_axis)
        print(f"  {name:6s}: SS=2-SS=14 = {da.mean():+.3f}%p (|mean|={np.abs(da).mean():.3f}%p)")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("### Summary Table")
    print()
    print("| Metric | SS=14 (Ours) | SS=2 (Ours+FastSS) | Δ |")
    print("|--------|-------------|-------------------|---|")
    print(f"| Latency (s) | {lat14.mean():.2f} | {lat2.mean():.2f} | {lat14.mean()/lat2.mean():.2f}x faster |")
    print(f"| VRAM Peak (GB) | {vram14.max()/1024:.2f} | {vram2.max()/1024:.2f} | {(vram14.max()-vram2.max())/1024:+.2f} |")
    print(f"| RDE (%) | {rde14.mean():.2f} | {rde2.mean():.2f} | {diffs.mean():+.2f}%p |")
    for tau in da_thresholds:
        t = tau / 100.0
        da2 = (rde2 / 100 <= t).sum() / len(rde2) * 100
        da14 = (rde14 / 100 <= t).sum() / len(rde14) * 100
        print(f"| DA@{tau} (%) | {da14:.1f} | {da2:.1f} | {da2-da14:+.1f}%p |")
    print(f"| VE (%) | {ve14.mean():.2f} | {ve2.mean():.2f} | {(ve2-ve14).mean():+.2f}%p |")


if __name__ == "__main__":
    main()
