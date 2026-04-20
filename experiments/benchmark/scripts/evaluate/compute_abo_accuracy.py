"""
ABO 정확도 집계 → 표 2 (전체), 표 3 (카테고리별)

입력:
- worker 출력 CSV: dim_small / dim_mid / dim_large (상대, 크기순 정렬)
  + sample_id, success, latency_seconds, vram_peak_mb
- abo_samples_500.json: sample_id ↔ base_name, GT dimensions (mm)

계산:
1. Relative Dimension MAPE (%) — GT를 크기순 정렬한 비율과 비교, KB 독립
2. KB 매칭으로 절대 치수 산출 (ai/processors/8_absolute_volume_calculate.py 활용)
3. Volume MAPE (%) — 절대 부피 |V_pred - V_GT| / V_GT
4. Per-axis Absolute MAPE (%) — 크기순 정렬 후 W/D/H 매칭
5. Success rate (%) — SAM-3D 성공 비율
6. Absolute ↔ Relative gap (%pt) — KB 매칭 기여도

출력:
- abo_accuracy_summary.csv (표 2)
- abo_accuracy_by_category.csv (표 3)

Usage:
    python experiments/benchmark/scripts/evaluate/compute_abo_accuracy.py \
        --proposed-csv results/abo_proposed.csv \
        --baseline-csv results/abo_baseline_b.csv \
        --output-dir results/
"""

import argparse
import csv
import importlib.util
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ABO_SAMPLES_JSON, PROJECT_ROOT, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ABO:eval] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# AbsoluteVolumeCalculator 동적 로드 (8_* prefix 때문에 직접 import 불가)
# ============================================================
def _load_abs_vol_calculator():
    mod_path = PROJECT_ROOT / "ai" / "processors" / "8_absolute_volume_calculate.py"
    spec = importlib.util.spec_from_file_location("abs_vol_calc", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PROJECT_ROOT))
    spec.loader.exec_module(mod)
    return mod.AbsoluteVolumeCalculator


# ============================================================
# Metrics
# ============================================================
def sorted_triplet(w: float, d: float, h: float) -> tuple[float, float, float]:
    """Ascending sort (small, mid, large)."""
    return tuple(sorted((w, d, h)))


def mape(pred: float, gt: float) -> float:
    """MAPE = |pred - gt| / gt * 100. gt<=0 이면 NaN (집계에서 제외)."""
    if gt <= 0 or not math.isfinite(gt) or not math.isfinite(pred):
        return float("nan")
    return 100.0 * abs(pred - gt) / gt


def _clean(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def relative_ratios(small: float, mid: float, large: float) -> tuple[float, float, float]:
    """최장축=1 정규화 → (small/large, mid/large, 1.0)."""
    if large <= 0:
        return (0.0, 0.0, 0.0)
    return (small / large, mid / large, 1.0)


def per_axis_mape(
    pred_sorted: tuple[float, float, float],
    gt_sorted: tuple[float, float, float],
) -> float:
    """크기순 정렬 매칭 후 세 축 MAPE 평균 (NaN 제외)."""
    vals = _clean([mape(p, g) for p, g in zip(pred_sorted, gt_sorted)])
    return mean(vals) if vals else float("nan")


def relative_dimension_mape(
    pred_sorted: tuple[float, float, float],
    gt_sorted: tuple[float, float, float],
) -> float:
    """KB 독립 상대 비율 MAPE.

    최장축=1 정규화 후 small/mid 두 비율의 MAPE 평균.
    (large 비율은 항상 1이므로 제외.)
    """
    pred_ratios = relative_ratios(*pred_sorted)
    gt_ratios = relative_ratios(*gt_sorted)
    vals = _clean([mape(pred_ratios[i], gt_ratios[i]) for i in (0, 1)])
    return mean(vals) if vals else float("nan")


# ============================================================
# Load helpers
# ============================================================
def load_worker_csv(path: Path) -> dict[int, dict]:
    """sample_id → row."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                sid = int(row["sample_id"])
            except (ValueError, KeyError):
                continue
            if sid < 0:  # warmup dummy
                continue
            out[sid] = row
    return out


def load_abo_samples() -> dict[int, dict]:
    with open(ABO_SAMPLES_JSON) as f:
        arr = json.load(f)
    return {s["sample_id"]: s for s in arr if "sample_id" in s}


# ============================================================
# Per-sample computation
# ============================================================
def compute_sample_metrics(
    worker_row: dict,
    sample_meta: dict,
    calc,
) -> dict | None:
    """단일 샘플의 모든 지표 계산. 실패 시 None."""
    success = str(worker_row.get("success", "")).lower() == "true"
    if not success:
        return None

    try:
        dim_small = float(worker_row["dim_small"])
        dim_mid = float(worker_row["dim_mid"])
        dim_large = float(worker_row["dim_large"])
    except (ValueError, KeyError, TypeError):
        return None

    if dim_large <= 0:
        return None

    base_name = sample_meta["base_name"]
    gt_sorted = sorted_triplet(
        sample_meta["gt_width_mm"],
        sample_meta["gt_depth_mm"],
        sample_meta["gt_height_mm"],
    )
    pred_sorted_rel = (dim_small, dim_mid, dim_large)

    # 1) Relative Dimension MAPE (KB 독립)
    rel_mape = relative_dimension_mape(pred_sorted_rel, gt_sorted)

    # 2) KB 매칭으로 절대 치수 산출
    # AbsoluteVolumeCalculator.calculate_absolute_volume 는 rel_width/depth/height 인자를 받음.
    # 크기순 정렬한 값을 그대로 넣으면 내부에서 다시 정렬하므로 순서 무관.
    result = calc.calculate_absolute_volume(
        label=base_name,
        type_name=None,  # 자동 best-match
        rel_width=dim_small,
        rel_depth=dim_mid,
        rel_height=dim_large,
    )
    pred_abs_sorted = sorted_triplet(result.width_mm, result.depth_mm, result.height_mm)
    pred_volume_m3 = result.volume_m3

    # 3) GT 부피 (m³) — gt mm³를 m³로
    gt_volume_m3 = (gt_sorted[0] * gt_sorted[1] * gt_sorted[2]) * 1e-9

    # 4) Volume MAPE
    vol_mape = mape(pred_volume_m3, gt_volume_m3)

    # 5) Per-axis Absolute MAPE
    axis_mape = per_axis_mape(pred_abs_sorted, gt_sorted)

    # gap: Volume MAPE − Relative MAPE. 둘 중 하나 NaN 이면 gap 도 NaN.
    if math.isfinite(vol_mape) and math.isfinite(rel_mape):
        gap = vol_mape - rel_mape
    else:
        gap = float("nan")

    # YOLOE 전처리 단계의 detected_base 가 expected 와 매칭되었는지
    matched_expected = bool(sample_meta.get("matched_expected", True))

    return {
        "sample_id": sample_meta["sample_id"],
        "base_name": base_name,
        "matched_expected": matched_expected,
        "volume_mape": vol_mape,
        "relative_mape": rel_mape,
        "per_axis_mape": axis_mape,
        "gap": gap,
        "pred_volume_m3": pred_volume_m3,
        "gt_volume_m3": gt_volume_m3,
        "pred_small_mm": pred_abs_sorted[0],
        "pred_mid_mm": pred_abs_sorted[1],
        "pred_large_mm": pred_abs_sorted[2],
        "gt_small_mm": gt_sorted[0],
        "gt_mid_mm": gt_sorted[1],
        "gt_large_mm": gt_sorted[2],
        "rel_small": dim_small,
        "rel_mid": dim_mid,
        "rel_large": dim_large,
        "matched_type": result.matched_type or "",
        "latency_seconds": float(worker_row.get("latency_seconds", 0) or 0),
        "vram_peak_mb": float(worker_row.get("vram_peak_mb", 0) or 0),
    }


# ============================================================
# Aggregate
# ============================================================
def aggregate_metrics(rows: list[dict]) -> dict:
    """한 설정의 전체 집계 (mean±std, median)."""
    if not rows:
        return {k: 0.0 for k in [
            "n", "volume_mape_mean", "volume_mape_std", "volume_mape_median",
            "relative_mape_mean", "relative_mape_std", "relative_mape_median",
            "per_axis_mape_mean", "per_axis_mape_std",
            "gap_mean", "gap_median",
            "latency_mean", "vram_peak_mean",
        ]}

    def agg(values_raw: list[float]) -> tuple[float, float, float]:
        values = _clean(values_raw)
        if not values:
            return (float("nan"), float("nan"), float("nan"))
        m = mean(values)
        s = stdev(values) if len(values) > 1 else 0.0
        md = median(values)
        return (m, s, md)

    v_m, v_s, v_md = agg([r["volume_mape"] for r in rows])
    r_m, r_s, r_md = agg([r["relative_mape"] for r in rows])
    p_m, p_s, _ = agg([r["per_axis_mape"] for r in rows])
    g_m, _, g_md = agg([r["gap"] for r in rows])
    lat_m, _, _ = agg([r["latency_seconds"] for r in rows])
    vram_m, _, _ = agg([r["vram_peak_mb"] for r in rows])

    # 추가: matched_expected 비율, subtype 분포
    matched_count = sum(1 for r in rows if r.get("matched_expected"))
    subtype_counter: dict[str, int] = {}
    for r in rows:
        t = r.get("matched_type", "")
        subtype_counter[t] = subtype_counter.get(t, 0) + 1

    return {
        "n": len(rows),
        "n_matched_expected": matched_count,
        "matched_expected_rate": 100.0 * matched_count / max(1, len(rows)),
        "volume_mape_mean": v_m,
        "volume_mape_std": v_s,
        "volume_mape_median": v_md,
        "relative_mape_mean": r_m,
        "relative_mape_std": r_s,
        "relative_mape_median": r_md,
        "per_axis_mape_mean": p_m,
        "per_axis_mape_std": p_s,
        "gap_mean": g_m,
        "gap_median": g_md,
        "latency_mean": lat_m,
        "vram_peak_mean": vram_m,
        "subtype_counter": subtype_counter,
    }


def aggregate_by_category(rows: list[dict]) -> dict[str, dict]:
    """base_name 별 집계."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["base_name"]].append(r)
    return {bn: aggregate_metrics(lst) for bn, lst in by_cat.items()}


# ============================================================
# Success rate (explicit)
# ============================================================
def compute_success_rate(worker_rows: dict[int, dict], total: int) -> float:
    if total == 0:
        return 0.0
    success_count = sum(
        1 for r in worker_rows.values()
        if str(r.get("success", "")).lower() == "true"
    )
    return 100.0 * success_count / total


# ============================================================
# Output
# ============================================================
def write_summary_csv(
    path: Path,
    per_config: dict[str, dict],
    success_rates: dict[str, float],
) -> None:
    """표 2 — 전체 집계."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "config",
            "n_success",
            "volume_mape_mean(%)",
            "volume_mape_std(%)",
            "volume_mape_median(%)",
            "relative_mape_mean(%)",
            "relative_mape_std(%)",
            "relative_mape_median(%)",
            "per_axis_mape_mean(%)",
            "gap_mean(%pt)",
            "gap_median(%pt)",
            "success_rate(%)",
            "latency_mean(s)",
            "vram_peak_mean(MB)",
        ])
        for cfg_name, agg in per_config.items():
            w.writerow([
                cfg_name,
                agg["n"],
                f"{agg['volume_mape_mean']:.2f}",
                f"{agg['volume_mape_std']:.2f}",
                f"{agg['volume_mape_median']:.2f}",
                f"{agg['relative_mape_mean']:.2f}",
                f"{agg['relative_mape_std']:.2f}",
                f"{agg['relative_mape_median']:.2f}",
                f"{agg['per_axis_mape_mean']:.2f}",
                f"{agg['gap_mean']:.2f}",
                f"{agg['gap_median']:.2f}",
                f"{success_rates.get(cfg_name, 0.0):.1f}",
                f"{agg['latency_mean']:.2f}",
                f"{agg['vram_peak_mean']:.0f}",
            ])
    logger.info(f"Saved (표 2): {path}")


def write_category_csv(
    path: Path,
    per_config_by_cat: dict[str, dict[str, dict]],
) -> None:
    """표 3 — 카테고리별 gap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    all_cats = sorted({
        bn for cat_dict in per_config_by_cat.values() for bn in cat_dict
    })
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["category"]
        for cfg in per_config_by_cat:
            header.extend([
                f"{cfg}_n",
                f"{cfg}_volume_mape(%)",
                f"{cfg}_relative_mape(%)",
                f"{cfg}_gap(%pt)",
            ])
        w.writerow(header)
        for cat in all_cats:
            row = [cat]
            for cfg, cat_dict in per_config_by_cat.items():
                agg = cat_dict.get(cat, {})
                row.extend([
                    agg.get("n", 0),
                    f"{agg.get('volume_mape_mean', 0):.2f}",
                    f"{agg.get('relative_mape_mean', 0):.2f}",
                    f"{agg.get('gap_mean', 0):.2f}",
                ])
            w.writerow(row)
    logger.info(f"Saved (표 3): {path}")


def write_per_sample_csv(path: Path, rows_by_config: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_fields = [
        "config", "sample_id", "base_name", "matched_type",
        "volume_mape", "relative_mape", "per_axis_mape", "gap",
        "pred_volume_m3", "gt_volume_m3",
        "pred_small_mm", "pred_mid_mm", "pred_large_mm",
        "gt_small_mm", "gt_mid_mm", "gt_large_mm",
        "rel_small", "rel_mid", "rel_large",
        "latency_seconds", "vram_peak_mb",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        for cfg, rows in rows_by_config.items():
            for r in rows:
                out = {"config": cfg}
                for k in all_fields:
                    if k == "config":
                        continue
                    v = r.get(k, "")
                    if isinstance(v, float):
                        out[k] = f"{v:.4f}"
                    else:
                        out[k] = v
                w.writerow(out)
    logger.info(f"Saved (per-sample): {path}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposed-csv", type=Path,
                    default=RESULTS_DIR / "abo_proposed.csv")
    ap.add_argument("--baseline-csv", type=Path,
                    default=RESULTS_DIR / "abo_baseline_b.csv")
    ap.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--only-matched-expected", action="store_true",
                    help="YOLOE 가 expected base_name 과 일치한 샘플만 집계 (detection 오차 배제)")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("ABO 정확도 집계 (표 2 / 표 3)")
    logger.info("=" * 60)

    # Load AbsoluteVolumeCalculator
    AbsVolCalc = _load_abs_vol_calculator()
    calc = AbsVolCalc()

    # Load samples metadata
    samples = load_abo_samples()
    logger.info(f"Loaded {len(samples)} ABO samples (metadata)")

    # Load worker CSVs
    per_config_rows: dict[str, list[dict]] = {}
    per_config_success: dict[str, float] = {}

    for cfg_name, csv_path in [
        ("abo_proposed", args.proposed_csv),
        ("abo_baseline_b", args.baseline_csv),
    ]:
        if not csv_path.exists():
            logger.warning(f"{cfg_name}: CSV missing ({csv_path}), skipping")
            continue

        raw_rows = load_worker_csv(csv_path)
        logger.info(f"{cfg_name}: loaded {len(raw_rows)} worker rows")

        metric_rows = []
        for sid, row in raw_rows.items():
            meta = samples.get(sid)
            if not meta:
                continue
            m = compute_sample_metrics(row, meta, calc)
            if m is None:
                continue
            if args.only_matched_expected and not m["matched_expected"]:
                continue
            metric_rows.append(m)

        per_config_rows[cfg_name] = metric_rows
        per_config_success[cfg_name] = compute_success_rate(raw_rows, len(samples))
        logger.info(
            f"{cfg_name}: {len(metric_rows)} valid metric rows "
            f"(success rate={per_config_success[cfg_name]:.1f}%)"
        )

    if not per_config_rows:
        logger.error("No CSV loaded. Abort.")
        sys.exit(1)

    # Aggregate
    per_config_agg = {k: aggregate_metrics(v) for k, v in per_config_rows.items()}
    per_config_by_cat = {k: aggregate_by_category(v) for k, v in per_config_rows.items()}

    # Print summary
    logger.info("\n=== 전체 집계 ===")
    for cfg, a in per_config_agg.items():
        logger.info(
            f"{cfg}: n={a['n']} | VolMAPE={a['volume_mape_mean']:.2f}±{a['volume_mape_std']:.2f} "
            f"| RelMAPE={a['relative_mape_mean']:.2f}±{a['relative_mape_std']:.2f} "
            f"| gap={a['gap_mean']:.2f} | success={per_config_success[cfg]:.1f}% "
            f"| match_expected={a['matched_expected_rate']:.1f}%"
        )

    # Subtype agreement: 두 설정에서 같은 sample_id 가 같은 matched_type 을 얻었는지
    if len(per_config_rows) == 2 and "abo_proposed" in per_config_rows and "abo_baseline_b" in per_config_rows:
        p_map = {r["sample_id"]: r["matched_type"] for r in per_config_rows["abo_proposed"]}
        b_map = {r["sample_id"]: r["matched_type"] for r in per_config_rows["abo_baseline_b"]}
        common = set(p_map) & set(b_map)
        agree = sum(1 for sid in common if p_map[sid] == b_map[sid])
        disagree = len(common) - agree
        if common:
            logger.info(
                f"Subtype agreement (Proposed vs Baseline-B): "
                f"{agree}/{len(common)} = {100.0 * agree / len(common):.1f}% "
                f"(disagree={disagree})"
            )

    # Write CSVs
    write_summary_csv(
        args.output_dir / "abo_accuracy_summary.csv",
        per_config_agg,
        per_config_success,
    )
    write_category_csv(
        args.output_dir / "abo_accuracy_by_category.csv",
        per_config_by_cat,
    )
    write_per_sample_csv(
        args.output_dir / "abo_per_sample.csv",
        per_config_rows,
    )

    # Markdown 출력 (표 2/3 복사용)
    md_path = args.output_dir / "abo_tables.md"
    with open(md_path, "w") as f:
        f.write("# ABO Accuracy Tables\n\n")
        f.write("## 표 2 — 전체 집계\n\n")
        f.write("| 지표 | Baseline-B | Proposed |\n|---|---|---|\n")

        def fmt(cfg: str, field: str, prec: int = 2) -> str:
            v = per_config_agg.get(cfg, {}).get(field)
            if v is None:
                return "—"
            return f"{v:.{prec}f}"

        rows = [
            ("n", "n", 0),
            ("Volume MAPE (%) mean", "volume_mape_mean", 2),
            ("Volume MAPE (%) median", "volume_mape_median", 2),
            ("Relative Dimension MAPE (%) mean", "relative_mape_mean", 2),
            ("Relative Dimension MAPE (%) median", "relative_mape_median", 2),
            ("Per-axis Absolute MAPE (%) mean", "per_axis_mape_mean", 2),
            ("Success rate (%)", None, 1),
            ("Latency (s)", "latency_mean", 2),
            ("VRAM peak (MB)", "vram_peak_mean", 0),
            ("Absolute − Relative gap (%pt)", "gap_mean", 2),
        ]
        for name, field, prec in rows:
            if field is None:
                b = f"{per_config_success.get('abo_baseline_b', 0.0):.1f}"
                p = f"{per_config_success.get('abo_proposed', 0.0):.1f}"
            else:
                b = fmt("abo_baseline_b", field, prec)
                p = fmt("abo_proposed", field, prec)
            f.write(f"| {name} | {b} | {p} |\n")

        f.write("\n## 표 3 — 카테고리별 (Proposed)\n\n")
        f.write("| category | n | Volume MAPE (%) | Relative MAPE (%) | gap (%pt) |\n")
        f.write("|---|---|---|---|---|\n")
        prop_by_cat = per_config_by_cat.get("abo_proposed", {})
        for bn in sorted(prop_by_cat):
            a = prop_by_cat[bn]
            f.write(
                f"| {bn} | {a['n']} | {a['volume_mape_mean']:.2f} | "
                f"{a['relative_mape_mean']:.2f} | {a['gap_mean']:.2f} |\n"
            )
    logger.info(f"Saved (markdown): {md_path}")


if __name__ == "__main__":
    main()
