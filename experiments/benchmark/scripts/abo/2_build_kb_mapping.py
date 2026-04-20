"""
ABO product_type ↔ 프로젝트 KB base_name 매핑 테이블 생성

- 계획서 §4.1 KB ↔ ABO 매핑 (예시) 을 정적 딕셔너리로 기록
- 실제 product_type 분포를 확인하여 매핑 커버리지 보고
- 결과: abo_kb_mapping.json

Usage:
    python experiments/benchmark/scripts/abo/2_build_kb_mapping.py
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ABO_KB_MAPPING_JSON, ABO_METADATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ABO:map] %(message)s")
logger = logging.getLogger(__name__)

# ABO product_type → 프로젝트 KB base_name
# - ABO product_type 은 UPPERCASE_UNDERSCORE 또는 CamelCase 섞임.
# - 소문자 정규화 후 매핑.
# - 계획서 §4.1 + ABO 실제 product_type 분포 기반.
ABO_TO_BASE_NAME: dict[str, str] = {
    # --- 소파/안락의자 ---
    "sofa": "SOFA",
    "couch": "SOFA",
    "loveseat": "SOFA",
    "sectional_sofa": "SOFA",
    "sofa_sectional": "SOFA",
    # --- 침대 ---
    "bed": "BED",
    "bed_frame": "BED",
    "bunk_bed": "BED",
    "daybed": "BED",
    # --- 의자 ---
    "chair": "CHAIR_STOOL",
    "arm_chair": "CHAIR_STOOL",
    "armchair": "CHAIR_STOOL",
    "office_chair": "CHAIR_STOOL",
    "stool": "CHAIR_STOOL",
    "bar_stool": "CHAIR_STOOL",
    "rocking_chair": "CHAIR_STOOL",
    "folding_chair": "CHAIR_STOOL",
    "accent_chair": "CHAIR_STOOL",
    # --- 식탁 ---
    "dining_table": "DINING_TABLE",
    "kitchen_table": "DINING_TABLE",
    # --- 커피 테이블 ---
    "coffee_table": "COFFEE_TABLE",
    "cocktail_table": "COFFEE_TABLE",
    "end_table": "COFFEE_TABLE",
    "side_table": "COFFEE_TABLE",
    "accent_table": "COFFEE_TABLE",
    # --- 책상 ---
    "desk": "DESK",
    "writing_desk": "DESK",
    "computer_desk": "DESK",
    "office_desk": "DESK",
    # --- 책장 ---
    "bookcase": "BOOKSHELF",
    "bookshelf": "BOOKSHELF",
    # --- 옷장 ---
    "wardrobe": "WARDROBE",
    "armoire": "WARDROBE",
    # --- 서랍장 ---
    "chest_of_drawers": "DRAWER",
    "dresser": "DRAWER",
    # --- 나이트스탠드 ---
    "nightstand": "NIGHTSTAND",
    "bedside_table": "NIGHTSTAND",
    # --- TV 스탠드 ---
    "tv_stand": "TV_STAND",
    "media_console": "TV_STAND",
    "tv_cabinet": "TV_STAND",
    # --- 캐비닛/수납장 ---
    "cabinet": "CABINET",
    "storage_cabinet": "CABINET",
    "filing_cabinet": "CABINET",
    # --- 식기장 ---
    "china_cabinet": "DISH_CABINET",
    "kitchen_cabinet": "DISH_CABINET",
    # --- 진열장 ---
    "shelf": "DISPLAY_SHELF",
    "shelving_unit": "DISPLAY_SHELF",
    "display_stand": "DISPLAY_SHELF",
    # --- 화장대 ---
    "vanity": "VANITY_TABLE",
    "dressing_table": "VANITY_TABLE",
    # --- 거울 ---
    "mirror": "MIRROR",
    # --- 수납 박스 ---
    "storage_box": "STORAGE_BOX",
    "storage_bin": "STORAGE_BOX",
    "basket": "STORAGE_BOX",
}


def normalize(pt: str) -> str:
    return pt.strip().lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    logger.info("=" * 60)
    logger.info("ABO ↔ KB base_name 매핑 생성")
    logger.info("=" * 60)

    # 실제 listings 에서 product_type 분포 확인 (옵션)
    listings = ABO_METADATA_DIR / "abo_valid_listings.jsonl"
    pt_counter: Counter[str] = Counter()
    if listings.exists():
        with open(listings) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pt = normalize(obj.get("product_type") or "")
                if pt:
                    pt_counter[pt] += 1
        logger.info(f"Loaded {sum(pt_counter.values())} listings, {len(pt_counter)} unique product_types")

    # 커버리지 분석
    covered_count = 0
    uncovered_top: list[tuple[str, int]] = []
    if pt_counter:
        for pt, cnt in pt_counter.most_common():
            if pt in ABO_TO_BASE_NAME:
                covered_count += cnt
            else:
                if len(uncovered_top) < 20:
                    uncovered_top.append((pt, cnt))

        total = sum(pt_counter.values())
        logger.info(
            f"Coverage: {covered_count}/{total} "
            f"({100.0 * covered_count / max(1, total):.1f}%)"
        )
        logger.info("Top 20 uncovered product_types (검토용):")
        for pt, cnt in uncovered_top:
            logger.info(f"  {pt}: {cnt}")

    # base_name 별 매핑 rules 역집계
    by_base = {}
    for pt, bn in ABO_TO_BASE_NAME.items():
        by_base.setdefault(bn, []).append(pt)

    mapping = {
        "version": "1.0",
        "note": "ABO product_type (lowercase_underscore) → KB base_name",
        "abo_to_base_name": ABO_TO_BASE_NAME,
        "base_name_to_abo": by_base,
    }

    ABO_KB_MAPPING_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(ABO_KB_MAPPING_JSON, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {ABO_KB_MAPPING_JSON}")

    # 카테고리 요약
    logger.info(f"Mapped {len(ABO_TO_BASE_NAME)} product_types → {len(by_base)} base_names")
    for bn, pts in sorted(by_base.items()):
        logger.info(f"  {bn}: {len(pts)} product_types")


if __name__ == "__main__":
    main()
