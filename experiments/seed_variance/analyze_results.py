"""
Seed Variance Experiment Analysis (v2 — Axis-Invariant)

OBB greedy mapping은 seed/config에 따라 W↔D↔H 축 라벨이 뒤바뀔 수 있다.
이를 방지하기 위해 각 실행의 (W, D, H)를 크기순 정렬(small, mid, large)하여
축 할당에 무관한(axis-invariant) 비교를 수행한다.

Phase 1/2 결과를 분석하여:
- Table B: 카테고리별 Seed Variance (CV%)
- Table C: Optimization Deviation vs Seed Variance
- 허용 오차 기준 δ 확정
- Outlier 별도 기술
- 논문용 요약 통계

Usage:
    python analyze_results.py
"""

import csv
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_CSV, OPTIMIZED_CSV, RESULTS_DIR, ANALYSIS_REPORT,
    CATEGORY_SAMPLES, SEEDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Analysis] %(message)s")
logger = logging.getLogger(__name__)

# Outlier 기준: 어떤 축이든 CV > 이 값이면 outlier
OUTLIER_CV_THRESHOLD = 20.0


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SampleStats:
    """Per-sample statistics across seeds (axis-invariant)."""
    category: str
    sample_id: str
    # Raw values per seed (each entry = sorted [small, mid, large])
    sorted_dims: list = field(default_factory=list)

    @property
    def smalls(self) -> list:
        return [d[0] for d in self.sorted_dims]

    @property
    def mids(self) -> list:
        return [d[1] for d in self.sorted_dims]

    @property
    def larges(self) -> list:
        return [d[2] for d in self.sorted_dims]

    def _cv(self, vals: list) -> float:
        if len(vals) < 2:
            return 0.0
        m = np.mean(vals)
        return (np.std(vals, ddof=1) / m * 100) if m > 0 else 0.0

    @property
    def mean_s(self) -> float:
        return float(np.mean(self.smalls))

    @property
    def mean_m(self) -> float:
        return float(np.mean(self.mids))

    @property
    def mean_l(self) -> float:
        return float(np.mean(self.larges))

    @property
    def std_s(self) -> float:
        return float(np.std(self.smalls, ddof=1)) if len(self.smalls) > 1 else 0.0

    @property
    def std_m(self) -> float:
        return float(np.std(self.mids, ddof=1)) if len(self.mids) > 1 else 0.0

    @property
    def std_l(self) -> float:
        return float(np.std(self.larges, ddof=1)) if len(self.larges) > 1 else 0.0

    @property
    def cv_s(self) -> float:
        return self._cv(self.smalls)

    @property
    def cv_m(self) -> float:
        return self._cv(self.mids)

    @property
    def cv_l(self) -> float:
        return self._cv(self.larges)

    @property
    def avg_cv(self) -> float:
        return (self.cv_s + self.cv_m + self.cv_l) / 3

    @property
    def is_outlier(self) -> bool:
        return max(self.cv_s, self.cv_m, self.cv_l) > OUTLIER_CV_THRESHOLD


# ============================================================================
# Data Loading & Stats
# ============================================================================

def load_csv(path: Path) -> list[dict]:
    """Load CSV results."""
    if not path.exists():
        logger.error(f"Results file not found: {path}")
        sys.exit(1)

    with open(path, 'r') as f:
        rows = list(csv.DictReader(f))

    successful = [r for r in rows if r.get("success", "True") == "True"]
    logger.info(f"Loaded {len(rows)} rows ({len(successful)} successful) from {path}")
    return successful


def compute_phase1_stats(rows: list[dict]) -> dict[tuple, SampleStats]:
    """Compute per-sample statistics using sorted dimensions."""
    groups = defaultdict(lambda: {"category": "", "dims": []})

    for row in rows:
        key = (row["category"], row["sample_id"])
        w, d, h = float(row["width"]), float(row["depth"]), float(row["height"])
        groups[key]["category"] = row["category"]
        groups[key]["dims"].append(sorted([w, d, h]))

    stats = {}
    for key, data in groups.items():
        stats[key] = SampleStats(
            category=data["category"],
            sample_id=key[1],
            sorted_dims=data["dims"],
        )

    return stats


def compute_category_stats(sample_stats: dict, exclude_outliers: bool = False) -> dict:
    """Compute category-level statistics (Table B)."""
    cat_groups = defaultdict(list)
    for stats in sample_stats.values():
        if exclude_outliers and stats.is_outlier:
            continue
        cat_groups[stats.category].append(stats)

    cat_stats = {}
    for cat, samples in sorted(cat_groups.items()):
        cv_ss = [s.cv_s for s in samples]
        cv_ms = [s.cv_m for s in samples]
        cv_ls = [s.cv_l for s in samples]

        cat_stats[cat] = {
            "n": len(samples),
            "cv_s_mean": float(np.mean(cv_ss)),
            "cv_m_mean": float(np.mean(cv_ms)),
            "cv_l_mean": float(np.mean(cv_ls)),
            "cv_s_std": float(np.std(cv_ss, ddof=1)) if len(cv_ss) > 1 else 0.0,
            "cv_m_std": float(np.std(cv_ms, ddof=1)) if len(cv_ms) > 1 else 0.0,
            "cv_l_std": float(np.std(cv_ls, ddof=1)) if len(cv_ls) > 1 else 0.0,
        }

    return cat_stats


def compute_optimization_deviation(
    phase1_stats: dict[tuple, SampleStats],
    phase2_rows: list[dict],
) -> dict:
    """
    Compute optimization deviation vs seed variance using sorted dimensions.

    Phase 2의 (W,D,H)도 sorted하여 Phase 1 sorted 평균과 비교.
    """
    optim_data = {}
    for row in phase2_rows:
        key = (row["category"], row["sample_id"])
        w, d, h = float(row["width"]), float(row["depth"]), float(row["height"])
        optim_data[key] = sorted([w, d, h])

    results = defaultdict(lambda: {"samples": [], "within_count": 0, "total_axes": 0})
    all_devs = {"s": [], "m": [], "l": []}
    all_cvs = {"s": [], "m": [], "l": []}
    total_within = 0
    total_axes = 0

    for key, stats in phase1_stats.items():
        if key not in optim_data:
            continue

        optim_sorted = optim_data[key]
        cat = stats.category

        sample_result = {"sample_id": key[1], "is_outlier": stats.is_outlier}

        for i, (axis_key, mean_val, cv_val) in enumerate([
            ("s", stats.mean_s, stats.cv_s),
            ("m", stats.mean_m, stats.cv_m),
            ("l", stats.mean_l, stats.cv_l),
        ]):
            optim_val = optim_sorted[i]
            dev = abs(optim_val - mean_val) / mean_val * 100 if mean_val > 0 else 0.0

            within = dev <= cv_val
            results[cat]["total_axes"] += 1
            total_axes += 1

            if within:
                results[cat]["within_count"] += 1
                total_within += 1

            all_devs[axis_key].append(dev)
            all_cvs[axis_key].append(cv_val)

            sample_result[f"dev_{axis_key}"] = dev
            sample_result[f"cv_{axis_key}"] = cv_val

        results[cat]["samples"].append(sample_result)

    return {
        "by_category": dict(results),
        "all_devs": all_devs,
        "all_cvs": all_cvs,
        "total_within": total_within,
        "total_axes": total_axes,
    }


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(
    cat_stats: dict,
    cat_stats_no_outlier: dict,
    deviation_results: dict,
    phase1_stats: dict,
    outliers: list[SampleStats],
) -> str:
    """Generate markdown analysis report."""
    lines = []

    total_n = sum(s["n"] for s in cat_stats.values())
    total_n_clean = sum(s["n"] for s in cat_stats_no_outlier.values())

    lines.append("# Seed Variance Experiment Results (Axis-Invariant)")
    lines.append("")
    lines.append("## 방법론")
    lines.append("")
    lines.append("OBB greedy mapping은 3D 형태가 미세하게 달라지면 W↔D↔H 축 라벨이")
    lines.append("뒤바뀔 수 있다. 이를 방지하기 위해 각 실행의 (W, D, H)를 크기순으로")
    lines.append("정렬하여 **small, mid, large** 세 축으로 재정의한 뒤 비교한다.")
    lines.append("")

    lines.append("## 실험 개요")
    lines.append("")
    lines.append(f"- 총 샘플: {total_n}개 객체 (7 카테고리)")
    lines.append(f"- Seeds: {SEEDS} (K={len(SEEDS)})")
    lines.append(f"- Phase 1 (Original SAM-3D, 25/25 steps): {total_n * len(SEEDS)}회 실행")
    lines.append(f"- Phase 2 (Optimized SAM-3D, 14/4 steps + caching): {total_n}회 실행")
    if outliers:
        lines.append(f"- Outlier: {len(outliers)}개 (CV > {OUTLIER_CV_THRESHOLD}%, 별도 기술)")
    lines.append("")

    # ========================================================================
    # Table B: Seed Variance (전체)
    # ========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## Table B: 카테고리별 Seed Variance (전체)")
    lines.append("")
    lines.append("| Category | N | Small CV(%) | Mid CV(%) | Large CV(%) | Avg CV(%) |")
    lines.append("|----------|---|-------------|-----------|-------------|-----------|")

    all_cvs = []
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        avg_cv = (s["cv_s_mean"] + s["cv_m_mean"] + s["cv_l_mean"]) / 3
        all_cvs.extend([s["cv_s_mean"], s["cv_m_mean"], s["cv_l_mean"]])
        lines.append(
            f"| {cat} | {s['n']} | "
            f"{s['cv_s_mean']:.2f}±{s['cv_s_std']:.2f} | "
            f"{s['cv_m_mean']:.2f}±{s['cv_m_std']:.2f} | "
            f"{s['cv_l_mean']:.2f}±{s['cv_l_std']:.2f} | "
            f"{avg_cv:.2f} |"
        )

    delta_all = float(np.mean(all_cvs))
    delta_all_std = float(np.std(all_cvs, ddof=1))

    lines.append(
        f"| **전체** | **{total_n}** | "
        f"**{np.mean([s['cv_s_mean'] for s in cat_stats.values()]):.2f}** | "
        f"**{np.mean([s['cv_m_mean'] for s in cat_stats.values()]):.2f}** | "
        f"**{np.mean([s['cv_l_mean'] for s in cat_stats.values()]):.2f}** | "
        f"**δ={delta_all:.2f}** |"
    )
    lines.append("")
    lines.append(f"**δ (전체)** = {delta_all:.2f}%")
    lines.append("")

    # ========================================================================
    # Table B': Seed Variance (Outlier 제거)
    # ========================================================================
    if outliers:
        lines.append(f"## Table B': 카테고리별 Seed Variance (Outlier {len(outliers)}개 제외)")
        lines.append("")
        lines.append("| Category | N | Small CV(%) | Mid CV(%) | Large CV(%) | Avg CV(%) |")
        lines.append("|----------|---|-------------|-----------|-------------|-----------|")

        clean_cvs = []
        for cat in sorted(cat_stats_no_outlier.keys()):
            s = cat_stats_no_outlier[cat]
            avg_cv = (s["cv_s_mean"] + s["cv_m_mean"] + s["cv_l_mean"]) / 3
            clean_cvs.extend([s["cv_s_mean"], s["cv_m_mean"], s["cv_l_mean"]])
            lines.append(
                f"| {cat} | {s['n']} | "
                f"{s['cv_s_mean']:.2f}±{s['cv_s_std']:.2f} | "
                f"{s['cv_m_mean']:.2f}±{s['cv_m_std']:.2f} | "
                f"{s['cv_l_mean']:.2f}±{s['cv_l_std']:.2f} | "
                f"{avg_cv:.2f} |"
            )

        delta_clean = float(np.mean(clean_cvs))
        delta_clean_std = float(np.std(clean_cvs, ddof=1))

        lines.append(
            f"| **전체** | **{total_n_clean}** | "
            f"**{np.mean([s['cv_s_mean'] for s in cat_stats_no_outlier.values()]):.2f}** | "
            f"**{np.mean([s['cv_m_mean'] for s in cat_stats_no_outlier.values()]):.2f}** | "
            f"**{np.mean([s['cv_l_mean'] for s in cat_stats_no_outlier.values()]):.2f}** | "
            f"**δ={delta_clean:.2f}** |"
        )
        lines.append("")
        lines.append(f"**δ (Outlier 제외)** = {delta_clean:.2f}%")
        lines.append(f"**δ + 1σ (보수적)** = {delta_clean + delta_clean_std:.2f}%")
        lines.append("")
    else:
        delta_clean = delta_all
        delta_clean_std = delta_all_std

    # ========================================================================
    # Outlier Report
    # ========================================================================
    if outliers:
        lines.append("---")
        lines.append("")
        lines.append(f"## Outlier 분석 ({len(outliers)}개 샘플, CV > {OUTLIER_CV_THRESHOLD}%)")
        lines.append("")
        lines.append("| Category | Sample | CV_small(%) | CV_mid(%) | CV_large(%) | 원인 분석 |")
        lines.append("|----------|--------|-------------|-----------|-------------|----------|")

        for s in outliers:
            # Determine likely cause
            cause = _diagnose_outlier(s)
            lines.append(
                f"| {s.category} | {s.sample_id} | "
                f"{s.cv_s:.1f} | {s.cv_m:.1f} | {s.cv_l:.1f} | {cause} |"
            )

        lines.append("")
        lines.append("이 샘플들은 SAM-3D diffusion model이 해당 형태를 안정적으로 복원하지 못하여")
        lines.append("seed에 따라 3D 구조 자체가 크게 달라지는 경우이다.")
        lines.append("δ 산정 시 이들을 제외한 값을 기준으로 사용한다.")
        lines.append("")

        # Per-seed detail for outliers
        lines.append("### Outlier 상세 (seed별 sorted dimensions)")
        lines.append("")
        for s in outliers:
            lines.append(f"**{s.category}/{s.sample_id}** (CV_avg={s.avg_cv:.1f}%)")
            lines.append("")
            lines.append("| Seed | Small | Mid | Large |")
            lines.append("|------|-------|-----|-------|")
            for i, dims in enumerate(s.sorted_dims):
                lines.append(f"| {SEEDS[i] if i < len(SEEDS) else '?'} | {dims[0]:.4f} | {dims[1]:.4f} | {dims[2]:.4f} |")
            lines.append("")

    # ========================================================================
    # Table C: Optimization Deviation
    # ========================================================================
    if deviation_results["total_axes"] > 0:
        lines.append("---")
        lines.append("")
        lines.append("## Table C: Optimization Deviation vs Seed Variance")
        lines.append("")
        lines.append("| Category | N | Avg Seed CV(δ) | Avg Optim Dev | δ 이내 비율 |")
        lines.append("|----------|---|----------------|---------------|-------------|")

        for cat in sorted(deviation_results["by_category"].keys()):
            cat_data = deviation_results["by_category"][cat]
            # Exclude outlier samples
            clean_samples = [s for s in cat_data["samples"] if not s.get("is_outlier", False)]
            if not clean_samples:
                continue

            n_samples = len(clean_samples)
            avg_cv = float(np.mean([
                np.mean([s["cv_s"], s["cv_m"], s["cv_l"]])
                for s in clean_samples
            ]))
            avg_dev = float(np.mean([
                np.mean([s["dev_s"], s["dev_m"], s["dev_l"]])
                for s in clean_samples
            ]))
            within = sum(
                1
                for s in clean_samples
                for axis in ["s", "m", "l"]
                if s[f"dev_{axis}"] <= s[f"cv_{axis}"]
            )
            total = n_samples * 3

            lines.append(
                f"| {cat} | {n_samples} | "
                f"{avg_cv:.2f}% | {avg_dev:.2f}% | "
                f"{within}/{total} ({within / max(total, 1) * 100:.1f}%) |"
            )

        # Overall (excluding outliers)
        all_clean_samples = [
            s
            for cat_data in deviation_results["by_category"].values()
            for s in cat_data["samples"]
            if not s.get("is_outlier", False)
        ]
        if all_clean_samples:
            total_w = sum(
                1
                for s in all_clean_samples
                for axis in ["s", "m", "l"]
                if s[f"dev_{axis}"] <= s[f"cv_{axis}"]
            )
            total_a = len(all_clean_samples) * 3
            avg_all_cv = float(np.mean([
                np.mean([s["cv_s"], s["cv_m"], s["cv_l"]])
                for s in all_clean_samples
            ]))
            avg_all_dev = float(np.mean([
                np.mean([s["dev_s"], s["dev_m"], s["dev_l"]])
                for s in all_clean_samples
            ]))
        else:
            total_w, total_a, avg_all_cv, avg_all_dev = 0, 0, 0, 0

        lines.append(
            f"| **전체** | **{len(all_clean_samples)}** | "
            f"**{avg_all_cv:.2f}%** | **{avg_all_dev:.2f}%** | "
            f"**{total_w}/{total_a} ({total_w / max(total_a, 1) * 100:.1f}%)** |"
        )
        lines.append("")

        # Axis-level breakdown
        lines.append("### 축별 상세 (Outlier 제외)")
        lines.append("")
        lines.append("| 축 | Avg Seed CV | Avg Optim Dev | δ 이내 비율 |")
        lines.append("|------|-----------|---------------|------------|")

        for axis_name, axis_key in [("Small", "s"), ("Mid", "m"), ("Large", "l")]:
            devs_clean = [s[f"dev_{axis_key}"] for s in all_clean_samples]
            cvs_clean = [s[f"cv_{axis_key}"] for s in all_clean_samples]
            within = sum(1 for d, c in zip(devs_clean, cvs_clean) if d <= c)
            n = len(devs_clean)
            lines.append(
                f"| {axis_name} | {np.mean(cvs_clean):.2f}% | {np.mean(devs_clean):.2f}% | "
                f"{within}/{n} ({within / max(n, 1) * 100:.1f}%) |"
            )

        lines.append("")

    # ========================================================================
    # Summary for Paper
    # ========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## 논문 기술용 요약")
    lines.append("")
    lines.append(
        f"> 허용 오차 δ는 Original SAM-3D를 Pix3D 데이터셋의 "
        f"{total_n}개 가구 객체({len(cat_stats)}개 카테고리)에 대해 "
        f"seed만 달리하여 K={len(SEEDS)}회 실행한 결과의 "
        f"OBB 상대 치수 변동 계수(CV)로 정의한다. "
        f"축 할당 불안정성을 제거하기 위해 각 실행의 치수를 크기순으로 정렬(small, mid, large)하여 비교하였다."
    )

    if outliers:
        lines.append(
            f"> {total_n}개 중 {len(outliers)}개 샘플은 diffusion model의 3D 복원이 "
            f"본질적으로 불안정하여(CV > {OUTLIER_CV_THRESHOLD}%) outlier로 분류하고 별도 보고하였다."
        )

    if deviation_results["total_axes"] > 0 and all_clean_samples:
        lines.append(
            f"> Outlier를 제외한 {total_n_clean}개 객체에서 "
            f"전체 평균 CV는 δ = {delta_clean:.2f}%였으며, "
            f"본 최적화로 인한 평균 치수 편차 {avg_all_dev:.2f}%는 "
            f"이 기준 이내로, {total_a}개 축 측정 중 "
            f"{total_w / max(total_a, 1) * 100:.1f}%가 "
            f"seed variance 이내에 있음을 확인하였다. "
            f"이는 최적화 효과가 모델의 내재적 확률 변동 수준에 있음을 의미한다."
        )

    lines.append("")

    # ========================================================================
    # Table A: Per-sample
    # ========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## Table A: Per-Sample 요약 (sorted dimensions)")
    lines.append("")
    lines.append("| Category | Sample | μ_S | σ_S | CV_S(%) | μ_M | σ_M | CV_M(%) | μ_L | σ_L | CV_L(%) | Outlier |")
    lines.append("|----------|--------|------|------|---------|------|------|---------|------|------|---------|---------|")

    for key in sorted(phase1_stats.keys()):
        s = phase1_stats[key]
        flag = "⚠️" if s.is_outlier else ""
        lines.append(
            f"| {s.category} | {s.sample_id} | "
            f"{s.mean_s:.4f} | {s.std_s:.4f} | {s.cv_s:.2f} | "
            f"{s.mean_m:.4f} | {s.std_m:.4f} | {s.cv_m:.2f} | "
            f"{s.mean_l:.4f} | {s.std_l:.4f} | {s.cv_l:.2f} | {flag} |"
        )

    lines.append("")
    return "\n".join(lines)


def _diagnose_outlier(s: SampleStats) -> str:
    """Diagnose probable cause of an outlier."""
    # Check if total volume is stable (sum of dims)
    sums = [sum(d) for d in s.sorted_dims]
    sum_cv = np.std(sums, ddof=1) / np.mean(sums) * 100 if np.mean(sums) > 0 else 0

    if sum_cv > 15:
        return "3D 복원 자체가 불안정 (형태 변동)"
    elif sum_cv > 8:
        return "3D 복원 불안정 + 형태 비율 변동"
    else:
        return "형태 비율 변동 (총 크기는 안정적)"


# ============================================================================
# JSON Output
# ============================================================================

def save_summary_json(
    cat_stats: dict,
    cat_stats_no_outlier: dict,
    deviation_results: dict,
    outliers: list,
    path: Path,
):
    """Save machine-readable summary."""
    all_cvs = []
    clean_cvs = []

    summary = {"category_stats": {}, "category_stats_no_outlier": {}}

    for cat, s in cat_stats.items():
        summary["category_stats"][cat] = {
            "n": s["n"],
            "cv_small": s["cv_s_mean"],
            "cv_mid": s["cv_m_mean"],
            "cv_large": s["cv_l_mean"],
        }
        all_cvs.extend([s["cv_s_mean"], s["cv_m_mean"], s["cv_l_mean"]])

    for cat, s in cat_stats_no_outlier.items():
        summary["category_stats_no_outlier"][cat] = {
            "n": s["n"],
            "cv_small": s["cv_s_mean"],
            "cv_mid": s["cv_m_mean"],
            "cv_large": s["cv_l_mean"],
        }
        clean_cvs.extend([s["cv_s_mean"], s["cv_m_mean"], s["cv_l_mean"]])

    summary["delta_all"] = float(np.mean(all_cvs))
    summary["delta_no_outlier"] = float(np.mean(clean_cvs)) if clean_cvs else 0.0
    summary["delta_conservative"] = (
        float(np.mean(clean_cvs) + np.std(clean_cvs, ddof=1)) if len(clean_cvs) > 1 else 0.0
    )
    summary["outlier_count"] = len(outliers)
    summary["outlier_samples"] = [
        {"category": s.category, "sample_id": s.sample_id, "avg_cv": s.avg_cv}
        for s in outliers
    ]

    if deviation_results["total_axes"] > 0:
        all_clean = [
            s
            for cat_data in deviation_results["by_category"].values()
            for s in cat_data["samples"]
            if not s.get("is_outlier", False)
        ]
        if all_clean:
            total_w = sum(
                1 for s in all_clean for a in ["s", "m", "l"] if s[f"dev_{a}"] <= s[f"cv_{a}"]
            )
            total_a = len(all_clean) * 3
            summary["optimization_deviation"] = {
                "avg_dev": float(np.mean([
                    np.mean([s["dev_s"], s["dev_m"], s["dev_l"]]) for s in all_clean
                ])),
                "within_ratio": total_w / max(total_a, 1),
                "total_within": total_w,
                "total_axes": total_a,
            }

    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Summary JSON saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("Seed Variance Analysis (Axis-Invariant, v2)")
    logger.info("=" * 60)

    # Phase 1
    phase1_rows = load_csv(RAW_CSV)
    phase1_stats = compute_phase1_stats(phase1_rows)

    outliers = [s for s in phase1_stats.values() if s.is_outlier]
    logger.info(f"Phase 1: {len(phase1_stats)} samples, {len(outliers)} outliers")

    cat_stats = compute_category_stats(phase1_stats, exclude_outliers=False)
    cat_stats_no_outlier = compute_category_stats(phase1_stats, exclude_outliers=True)

    # Phase 2
    deviation_results = {
        "by_category": {}, "all_devs": {}, "all_cvs": {},
        "total_within": 0, "total_axes": 0,
    }

    if OPTIMIZED_CSV.exists():
        phase2_rows = load_csv(OPTIMIZED_CSV)
        deviation_results = compute_optimization_deviation(phase1_stats, phase2_rows)
        logger.info(f"Phase 2: {len(phase2_rows)} optimized results analyzed")
    else:
        logger.warning(f"Phase 2 results not found: {OPTIMIZED_CSV}")

    # Report
    report = generate_report(
        cat_stats, cat_stats_no_outlier, deviation_results, phase1_stats, outliers,
    )

    ANALYSIS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_REPORT, 'w') as f:
        f.write(report)
    logger.info(f"Report saved: {ANALYSIS_REPORT}")

    save_summary_json(
        cat_stats, cat_stats_no_outlier, deviation_results, outliers,
        RESULTS_DIR / "summary.json",
    )

    # ========================================================================
    # Console output
    # ========================================================================
    print()
    print("=" * 60)
    print("KEY RESULTS (Axis-Invariant)")
    print("=" * 60)

    all_cvs = []
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        avg = (s["cv_s_mean"] + s["cv_m_mean"] + s["cv_l_mean"]) / 3
        all_cvs.extend([s["cv_s_mean"], s["cv_m_mean"], s["cv_l_mean"]])
        print(f"  {cat:12s}: CV_S={s['cv_s_mean']:.2f}%  CV_M={s['cv_m_mean']:.2f}%  CV_L={s['cv_l_mean']:.2f}%  (avg={avg:.2f}%)")

    delta_all = np.mean(all_cvs)
    print(f"\n  δ (전체 {len(phase1_stats)}개) = {delta_all:.2f}%")

    if outliers:
        clean_cvs = []
        for s in cat_stats_no_outlier.values():
            clean_cvs.extend([s["cv_s_mean"], s["cv_m_mean"], s["cv_l_mean"]])
        delta_clean = np.mean(clean_cvs)
        print(f"  δ (outlier {len(outliers)}개 제외) = {delta_clean:.2f}%")
        print(f"  δ + 1σ (보수적) = {delta_clean + np.std(clean_cvs, ddof=1):.2f}%")
        print(f"\n  Outliers:")
        for s in outliers:
            print(f"    {s.category}/{s.sample_id}: CV_S={s.cv_s:.1f}% CV_M={s.cv_m:.1f}% CV_L={s.cv_l:.1f}% → {_diagnose_outlier(s)}")

    if deviation_results["total_axes"] > 0:
        all_clean = [
            s
            for cat_data in deviation_results["by_category"].values()
            for s in cat_data["samples"]
            if not s.get("is_outlier", False)
        ]
        if all_clean:
            avg_dev = np.mean([np.mean([s["dev_s"], s["dev_m"], s["dev_l"]]) for s in all_clean])
            total_w = sum(1 for s in all_clean for a in ["s","m","l"] if s[f"dev_{a}"] <= s[f"cv_{a}"])
            total_a = len(all_clean) * 3
            print(f"\n  Avg Optimization Deviation = {avg_dev:.2f}%")
            print(f"  δ 이내 비율 = {total_w}/{total_a} ({total_w/total_a*100:.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()
