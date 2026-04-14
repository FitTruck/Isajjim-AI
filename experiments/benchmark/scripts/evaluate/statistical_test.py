"""
3종 비교 통계 검정 — Wilcoxon Signed-Rank Test (paired, two-sided)

입력:
  results/original.csv, results/fastsam3d.csv, results/ours.csv
  gt_dimensions.json

출력:
  - RDE + VE 에 대한 Wilcoxon signed-rank test (Bonferroni 보정)
  - Cohen's dz (paired effect size)
  - Shapiro-Wilk 정규성 검정 (Wilcoxon 사용 근거)

논문 사용:
  "통계적으로 유의미한 차이가 없다" 주장의 근거
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

# ================================================================
# Paths
# ================================================================
BASE = Path(__file__).resolve().parent.parent.parent
GT_PATH = BASE / "data" / "gt_dimensions.json"

VARIANTS = {
    "Original": BASE / "results" / "original.csv",
    "Fast-SAM3D": BASE / "results" / "fastsam3d.csv",
    "Ours": BASE / "results" / "ours.csv",
}

PAIRS = [
    ("Original", "Ours"),
    ("Original", "Fast-SAM3D"),
    ("Ours", "Fast-SAM3D"),
]

ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / len(PAIRS)  # 0.0167


# ================================================================
# Metrics (collect_results.py와 동일)
# ================================================================
def compute_rde(gt: list[float], pred: list[float]) -> float:
    """Relative Dimension Error (mean of 3 axes)."""
    total = 0.0
    for g, p in zip(gt, pred):
        if g > 1e-6:
            total += abs(g - p) / g
    return total / 3.0


def compute_ve(gt: list[float], pred: list[float]) -> float:
    """Volume Error."""
    gt_vol = gt[0] * gt[1] * gt[2]
    pred_vol = pred[0] * pred[1] * pred[2]
    if gt_vol > 1e-9:
        return abs(gt_vol - pred_vol) / gt_vol
    return 0.0


# ================================================================
# Data loading
# ================================================================
def load_gt() -> dict[str, list[float]]:
    with open(GT_PATH) as f:
        raw = json.load(f)
    return {k: v["normalized_sorted"] for k, v in raw.items()}


def load_sample_metrics(csv_path: Path, gt_dims: dict) -> dict[int, dict]:
    """sample_id -> {"rde": float, "ve": float}"""
    results = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["success"] != "True":
                continue
            sid = int(row["sample_id"])
            gt = gt_dims.get(row["model_path"])
            if gt is None:
                continue
            pred = [float(row["dim_small"]), float(row["dim_mid"]), float(row["dim_large"])]
            results[sid] = {
                "rde": compute_rde(gt, pred),
                "ve": compute_ve(gt, pred),
            }
    return results


# ================================================================
# Statistical tests
# ================================================================
def cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's dz for paired samples: mean(diff) / std(diff)."""
    diff = b - a
    std = diff.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return diff.mean() / std


def interpret_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    return "Large"


def run_test(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str, metric_name: str):
    """Run Wilcoxon signed-rank test and report."""
    diff = (b - a) * 100  # to percentage points

    # Wilcoxon signed-rank test (two-sided)
    try:
        w_stat, p_val = stats.wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        w_stat, p_val = 0, 1.0

    d = cohens_dz(a, b)
    sig_raw = p_val < ALPHA
    sig_bonf = p_val < BONFERRONI_ALPHA

    return {
        "pair": f"{a_name} vs {b_name}",
        "metric": metric_name,
        "mean_diff_pp": diff.mean(),
        "w_stat": w_stat,
        "p_value": p_val,
        "sig_raw": sig_raw,
        "sig_bonferroni": sig_bonf,
        "cohens_d": d,
        "d_interp": interpret_d(d),
    }


# ================================================================
# Main
# ================================================================
def main():
    gt_dims = load_gt()
    print(f"GT dimensions: {len(gt_dims)} unique models")

    # Load per-sample metrics
    all_metrics = {name: load_sample_metrics(path, gt_dims) for name, path in VARIANTS.items()}

    # Common samples across all 3 variants
    common = sorted(set.intersection(*[set(m.keys()) for m in all_metrics.values()]))
    n = len(common)
    print(f"Common paired samples: {n}")
    print(f"Bonferroni-corrected α: {BONFERRONI_ALPHA:.4f} (k={len(PAIRS)} comparisons)")
    print()

    # Build paired arrays
    arrays = {}
    for name in VARIANTS:
        arrays[name] = {
            "rde": np.array([all_metrics[name][sid]["rde"] for sid in common]),
            "ve": np.array([all_metrics[name][sid]["ve"] for sid in common]),
        }

    # ── Summary ──
    print("=" * 70)
    print(f"### Descriptive Statistics (N={n})")
    print()
    print(f"| Variant      | RDE Mean (%) | RDE Std (%) | RDE Median (%) | VE Mean (%) | VE Median (%) |")
    print(f"|--------------|-------------|------------|----------------|-------------|---------------|")
    for name in ["Original", "Fast-SAM3D", "Ours"]:
        rde = arrays[name]["rde"] * 100
        ve = arrays[name]["ve"] * 100
        print(
            f"| {name:12s} | {rde.mean():11.2f} | {rde.std():10.2f} | "
            f"{np.median(rde):14.2f} | {ve.mean():11.2f} | {np.median(ve):13.2f} |"
        )

    # ── Normality test (justification for Wilcoxon) ──
    print()
    print("=" * 70)
    print("### Shapiro-Wilk Normality Test on Paired Differences")
    print("(p < 0.05 → non-normal → Wilcoxon justified over paired t-test)")
    print()
    for a_name, b_name in PAIRS:
        for metric in ["rde", "ve"]:
            diff = arrays[b_name][metric] - arrays[a_name][metric]
            w_sw, p_sw = stats.shapiro(diff[:5000])
            normal = "Normal" if p_sw > 0.05 else "Non-normal"
            print(f"  {a_name} vs {b_name} ({metric.upper()}): W={w_sw:.4f}, p={p_sw:.2e} → {normal}")

    # ── Wilcoxon tests ──
    print()
    print("=" * 70)
    print(f"### Wilcoxon Signed-Rank Test (N={n}, two-sided)")
    print(f"    α = {ALPHA}, Bonferroni α = {BONFERRONI_ALPHA:.4f}")
    print()

    for metric_name, metric_key in [("RDE", "rde"), ("VE", "ve")]:
        print(f"#### {metric_name}")
        print()
        print(f"| Comparison               | ΔMean (%p) | W stat   | p-value     | Sig (α=.05) | Sig (Bonf) | Cohen's dz | Effect   |")
        print(f"|--------------------------|-----------|----------|-------------|-------------|------------|-----------|----------|")

        for a_name, b_name in PAIRS:
            r = run_test(
                arrays[a_name][metric_key],
                arrays[b_name][metric_key],
                a_name, b_name, metric_name,
            )
            p_str = f"{r['p_value']:.2e}" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}"
            sig_raw = "YES" if r["sig_raw"] else "No"
            sig_bonf = "YES" if r["sig_bonferroni"] else "No"
            print(
                f"| {r['pair']:24s} | {r['mean_diff_pp']:+8.3f}  | {r['w_stat']:8.0f} | "
                f"{p_str:11s} | {sig_raw:11s} | {sig_bonf:10s} | {r['cohens_d']:+8.4f}  | {r['d_interp']:8s} |"
            )
        print()

    # ── Ours vs Original detail ──
    print("=" * 70)
    print("### Ours vs Original — Detailed Comparison")
    diff_rde = (arrays["Ours"]["rde"] - arrays["Original"]["rde"]) * 100
    diff_ve = (arrays["Ours"]["ve"] - arrays["Original"]["ve"]) * 100

    for name, diff in [("RDE", diff_rde), ("VE", diff_ve)]:
        better = np.sum(diff < 0)
        worse = np.sum(diff > 0)
        equal = np.sum(np.abs(diff) < 1e-6)
        print(f"\n  {name} difference (Ours − Original):")
        print(f"    Mean: {diff.mean():+.3f}%p, Std: {diff.std():.3f}%p")
        print(f"    Ours better: {better} ({better/n*100:.1f}%)")
        print(f"    Ours worse:  {worse} ({worse/n*100:.1f}%)")
        print(f"    Equal:       {equal} ({equal/n*100:.1f}%)")

    # ── Paper-ready summary ──
    print()
    print("=" * 70)
    print("### Paper-Ready Summary")
    print()

    r_rde = run_test(arrays["Original"]["rde"], arrays["Ours"]["rde"], "Original", "Ours", "RDE")
    r_ve = run_test(arrays["Original"]["ve"], arrays["Ours"]["ve"], "Original", "Ours", "VE")

    rde_sig = "significant" if r_rde["sig_bonferroni"] else "no significant"
    ve_sig = "significant" if r_ve["sig_bonferroni"] else "no significant"

    print(
        f"Wilcoxon signed-rank tests on {n} paired samples with Bonferroni correction "
        f"(α={BONFERRONI_ALPHA:.4f}, k={len(PAIRS)}):"
    )
    print(
        f"  - RDE: {rde_sig} difference "
        f"(W={r_rde['w_stat']:.0f}, p={r_rde['p_value']:.4f}, Cohen's dz={r_rde['cohens_d']:+.3f} [{r_rde['d_interp']}])"
    )
    print(
        f"  - VE:  {ve_sig} difference "
        f"(W={r_ve['w_stat']:.0f}, p={r_ve['p_value']:.4f}, Cohen's dz={r_ve['cohens_d']:+.3f} [{r_ve['d_interp']}])"
    )
    if r_ve["sig_bonferroni"] and r_ve["d_interp"] == "Negligible":
        ve_diff = (arrays["Ours"]["ve"] - arrays["Original"]["ve"]).mean() * 100
        print(
            f"\n  Note: VE is statistically significant but effect size is negligible "
            f"(dz={r_ve['cohens_d']:+.3f} < 0.2). "
            f"Mean difference is only {ve_diff:+.1f}%p — no practical significance."
        )


if __name__ == "__main__":
    main()
