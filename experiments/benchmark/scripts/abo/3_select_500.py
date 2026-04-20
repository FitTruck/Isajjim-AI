"""
ABO 500 샘플 층화 추출

- abo_valid_listings.jsonl + abo_kb_mapping.json 로드
- item_dimensions 유효성 필터: height / length / width 모두 `normalized_value` 있고 양수
- KB base_name 별 층화 추출 (카테고리당 최소 25개 기준)
- 결과: abo_samples_500.json (sample_id, item_id, base_name, product_type, image_id, 3dmodel_id,
  gt_width_mm, gt_depth_mm, gt_height_mm)

Usage:
    python experiments/benchmark/scripts/abo/3_select_500.py [--target-n 500] [--min-per-cat 25]
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    ABO_KB_MAPPING_JSON,
    ABO_METADATA_DIR,
    ABO_SAMPLES_JSON,
    SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ABO:select] %(message)s")
logger = logging.getLogger(__name__)

# 단위 → mm 변환 계수
UNIT_TO_MM = {
    "millimeters": 1.0,
    "centimeters": 10.0,
    "centimetres": 10.0,
    "cm": 10.0,
    "mm": 1.0,
    "meters": 1000.0,
    "metres": 1000.0,
    "m": 1000.0,
    "inches": 25.4,
    "inch": 25.4,
    "in": 25.4,
    "feet": 304.8,
    "foot": 304.8,
    "ft": 304.8,
}

# item_dimensions 합리성 검증 범위 (mm): 가구는 10mm ~ 5000mm 이내
MIN_DIM_MM = 10.0
MAX_DIM_MM = 5000.0


def normalize(pt: str) -> str:
    return pt.strip().lower().replace(" ", "_").replace("-", "_")


def extract_dim_mm(dim_obj: dict | None) -> float | None:
    """item_dimensions 의 한 축 ({normalized_value: {unit, value}}) → mm."""
    if not isinstance(dim_obj, dict):
        return None
    nv = dim_obj.get("normalized_value") or {}
    val = nv.get("value")
    unit = (nv.get("unit") or "").strip().lower()
    if val is None or unit not in UNIT_TO_MM:
        # normalized_value 가 없으면 raw value 로 fallback
        rv = dim_obj.get("value")
        ru = (dim_obj.get("unit") or "").strip().lower()
        if rv is None or ru not in UNIT_TO_MM:
            return None
        val, unit = rv, ru
    try:
        mm = float(val) * UNIT_TO_MM[unit]
    except (TypeError, ValueError):
        return None
    if not (MIN_DIM_MM <= mm <= MAX_DIM_MM):
        return None
    return mm


def parse_dimensions(item_dims: dict | None) -> tuple[float, float, float] | None:
    """ABO item_dimensions → (width_mm, depth_mm, height_mm).

    ABO 는 `length` 를 depth 대신 사용. 매핑: length → depth.
    """
    if not isinstance(item_dims, dict):
        return None
    w = extract_dim_mm(item_dims.get("width"))
    d = extract_dim_mm(item_dims.get("length"))  # ABO length = depth
    h = extract_dim_mm(item_dims.get("height"))
    if w is None or d is None or h is None:
        return None
    return (w, d, h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-n", type=int, default=500)
    ap.add_argument("--min-per-cat", type=int, default=25)
    args = ap.parse_args()

    random.seed(SEED)

    logger.info("=" * 60)
    logger.info(f"ABO 500 샘플 층화 추출 (target={args.target_n})")
    logger.info("=" * 60)

    with open(ABO_KB_MAPPING_JSON) as f:
        mapping = json.load(f)
    abo_to_base = mapping["abo_to_base_name"]

    # 3dmodels.csv 로드 → 3dmodel_id → path 매핑
    models_csv = ABO_METADATA_DIR / "3dmodels.csv"
    import csv
    model_path_lookup: dict[str, str] = {}
    model_extent_lookup: dict[str, tuple[float, float, float]] = {}
    with open(models_csv) as f:
        for row in csv.DictReader(f):
            mid = row.get("3dmodel_id") or row.get("id") or ""
            p = row.get("path") or ""
            if mid and p:
                model_path_lookup[mid] = p
                try:
                    ex = float(row.get("extent_x", 0) or 0)
                    ey = float(row.get("extent_y", 0) or 0)
                    ez = float(row.get("extent_z", 0) or 0)
                    if ex > 0 and ey > 0 and ez > 0:
                        model_extent_lookup[mid] = (ex, ey, ez)
                except ValueError:
                    pass

    logger.info(f"Loaded 3dmodels.csv: {len(model_path_lookup)} entries")

    # images.csv 로드 → image_id → path
    images_csv = ABO_METADATA_DIR / "images.csv"
    image_path_lookup: dict[str, str] = {}
    with open(images_csv) as f:
        for row in csv.DictReader(f):
            iid = row.get("image_id") or ""
            p = row.get("path") or ""
            if iid and p:
                image_path_lookup[iid] = p
    logger.info(f"Loaded images.csv: {len(image_path_lookup)} entries")

    # listings 파싱 + 필터
    listings_jsonl = ABO_METADATA_DIR / "abo_valid_listings.jsonl"
    candidates_by_base: dict[str, list[dict]] = defaultdict(list)
    total_checked = 0
    filtered_no_map = 0
    filtered_no_dim = 0
    filtered_no_image = 0
    filtered_no_mesh = 0

    with open(listings_jsonl) as f:
        for line in f:
            total_checked += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            pt = normalize(obj.get("product_type") or "")
            base_name = abo_to_base.get(pt)
            if not base_name:
                filtered_no_map += 1
                continue

            dims = parse_dimensions(obj.get("item_dimensions"))
            if dims is None:
                filtered_no_dim += 1
                continue

            mid = obj.get("3dmodel_id")
            if not mid or mid not in model_path_lookup:
                filtered_no_mesh += 1
                continue

            iid = obj.get("main_image_id")
            if not iid or iid not in image_path_lookup:
                # fallback: other_image_id 의 첫 항목
                other = obj.get("other_image_id") or []
                iid = next((x for x in other if x in image_path_lookup), None)
            if not iid:
                filtered_no_image += 1
                continue

            w, d, h = dims
            candidates_by_base[base_name].append({
                "item_id": obj.get("item_id"),
                "product_type": pt,
                "base_name": base_name,
                "image_id": iid,
                "image_path": image_path_lookup[iid],
                "3dmodel_id": mid,
                "model_path": model_path_lookup[mid],
                "gt_width_mm": w,
                "gt_depth_mm": d,
                "gt_height_mm": h,
                "model_extent": model_extent_lookup.get(mid),
            })

    logger.info(
        f"Checked={total_checked}, no_map={filtered_no_map}, no_dim={filtered_no_dim}, "
        f"no_image={filtered_no_image}, no_mesh={filtered_no_mesh}"
    )
    logger.info(f"Candidate pool per base_name:")
    for bn, lst in sorted(candidates_by_base.items(), key=lambda x: -len(x[1])):
        logger.info(f"  {bn}: {len(lst)}")

    # 층화 추출: pool >= min-per-cat 만 대상
    valid_bases = {
        bn: lst for bn, lst in candidates_by_base.items()
        if len(lst) >= args.min_per_cat
    }
    excluded = [bn for bn in candidates_by_base if bn not in valid_bases]
    if excluded:
        logger.warning(f"Excluded (pool < {args.min_per_cat}): {excluded}")

    total_pool = sum(len(lst) for lst in valid_bases.values())
    if total_pool == 0:
        logger.error("No valid base_name meets min-per-cat threshold")
        sys.exit(1)

    # 비례 할당 (최소 min-per-cat 보장)
    alloc: dict[str, int] = {}
    remaining = args.target_n
    for bn, lst in valid_bases.items():
        alloc[bn] = max(args.min_per_cat, int(args.target_n * len(lst) / total_pool))
    # 총합이 target 과 차이나면 조정
    over = sum(alloc.values()) - args.target_n
    if over > 0:
        # 가장 큰 pool 에서 차감
        bns_sorted = sorted(valid_bases.keys(), key=lambda b: -len(valid_bases[b]))
        for bn in bns_sorted:
            if over <= 0:
                break
            reducible = alloc[bn] - args.min_per_cat
            take = min(reducible, over)
            alloc[bn] -= take
            over -= take
    elif over < 0:
        # 부족분 → pool 큰 순으로 채움
        bns_sorted = sorted(valid_bases.keys(), key=lambda b: -len(valid_bases[b]))
        for bn in bns_sorted:
            if over >= 0:
                break
            capacity = len(valid_bases[bn]) - alloc[bn]
            take = min(capacity, -over)
            alloc[bn] += take
            over += take
        if over < 0:
            logger.warning(
                f"Pool 부족으로 {-over} 샘플 미달 — total 할당={sum(alloc.values())}, target={args.target_n}"
            )

    # 실제 추출
    samples: list[dict] = []
    for bn, n in alloc.items():
        pool = valid_bases[bn]
        n = min(n, len(pool))
        picked = random.sample(pool, n)
        samples.extend(picked)
        logger.info(f"  {bn}: {n} selected")

    random.shuffle(samples)
    for i, s in enumerate(samples):
        s["sample_id"] = i

    ABO_SAMPLES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(ABO_SAMPLES_JSON, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {ABO_SAMPLES_JSON} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
