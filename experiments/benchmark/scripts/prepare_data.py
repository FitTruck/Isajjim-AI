"""
Phase 3: 데이터 준비

1. Pix3D 필터링 (truncated=False, occluded=False, mask_pixels >= 1000)
2. 500개 층화 추출 → benchmark_samples.json
3. GT 치수 생성: model.obj → AABB → 정규화 → gt_dimensions.json

Usage:
    python experiments/benchmark/prepare_data.py
"""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    GT_DIMENSIONS_JSON,
    MIN_MASK_PIXELS,
    PIX3D_DIR,
    SAMPLES_JSON,
    SEED,
    STRATIFIED_SAMPLES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PrepData] %(message)s")
logger = logging.getLogger(__name__)


def load_and_filter_pix3d() -> list[dict]:
    """pix3d.json 로드 후 품질 필터링."""
    pix3d_json = PIX3D_DIR / "pix3d.json"
    with open(pix3d_json) as f:
        data = json.load(f)

    logger.info(f"Total entries: {len(data)}")

    filtered = []
    for entry in data:
        if entry.get("truncated", False):
            continue
        if entry.get("occluded", False):
            continue

        # 마스크 파일 존재 확인
        mask_path = PIX3D_DIR / entry["mask"]
        if not mask_path.exists():
            continue

        # 이미지 파일 존재 확인
        img_path = PIX3D_DIR / entry["img"]
        if not img_path.exists():
            continue

        # GT model 존재 확인
        model_path = PIX3D_DIR / entry["model"]
        if not model_path.exists():
            continue

        filtered.append(entry)

    logger.info(f"After filtering (truncated=False, occluded=False): {len(filtered)}")

    # 카테고리별 통계
    from collections import Counter
    cats = Counter(e["category"] for e in filtered)
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count}")

    return filtered


def check_mask_pixels(mask_path: Path) -> int:
    """마스크 픽셀 수 확인."""
    from PIL import Image
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    return int(np.sum(arr > 0))


def stratified_sample(entries: list[dict]) -> list[dict]:
    """카테고리별 비율 유지하여 층화 추출."""
    random.seed(SEED)

    by_category = {}
    for e in entries:
        cat = e["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(e)

    samples = []
    for cat, n_target in STRATIFIED_SAMPLES.items():
        pool = by_category.get(cat, [])
        if len(pool) < n_target:
            logger.warning(f"{cat}: only {len(pool)} available, target {n_target}")
            n_target = len(pool)

        # 마스크 픽셀 수 필터링
        valid = []
        for e in pool:
            mask_path = PIX3D_DIR / e["mask"]
            px = check_mask_pixels(mask_path)
            if px >= MIN_MASK_PIXELS:
                valid.append(e)

        if len(valid) < n_target:
            logger.warning(f"{cat}: only {len(valid)} valid after mask filter, target {n_target}")
            n_target = min(n_target, len(valid))

        selected = random.sample(valid, n_target)
        samples.extend(selected)
        logger.info(f"  {cat}: {n_target} sampled (from {len(valid)} valid)")

    logger.info(f"Total samples: {len(samples)}")
    return samples


def build_samples_json(samples: list[dict]) -> list[dict]:
    """benchmark_samples.json 형식으로 변환."""
    result = []
    for i, entry in enumerate(samples):
        result.append({
            "sample_id": i,
            "category": entry["category"],
            "img_path": entry["img"],
            "mask_path": entry["mask"],
            "model_path": entry["model"],
        })
    return result


def compute_gt_dimensions(samples: list[dict]) -> dict:
    """GT mesh에서 AABB 치수 계산 → 정규화 → 크기순 정렬."""
    # unique model paths
    unique_models = sorted(set(s["model_path"] for s in samples))
    logger.info(f"Unique GT models to process: {len(unique_models)}")

    gt_dims = {}
    errors = 0
    for model_rel in unique_models:
        model_path = PIX3D_DIR / model_rel
        try:
            mesh = trimesh.load(str(model_path), force="mesh")
            vertices = np.array(mesh.vertices)

            if len(vertices) < 4:
                logger.warning(f"Too few vertices in {model_rel}: {len(vertices)}")
                errors += 1
                continue

            # AABB
            aabb_min = vertices.min(axis=0)
            aabb_max = vertices.max(axis=0)
            dims = aabb_max - aabb_min  # (w, d, h) in canonical axes

            # 정규화: max(w,d,h)로 나누기
            max_dim = dims.max()
            if max_dim < 1e-8:
                logger.warning(f"Zero dimensions in {model_rel}")
                errors += 1
                continue

            normalized = dims / max_dim  # [0, 1] 범위

            # 크기순 정렬 (ascending)
            sorted_dims = sorted(normalized.tolist())

            gt_dims[model_rel] = {
                "raw": dims.tolist(),
                "normalized_sorted": sorted_dims,
                "n_vertices": len(vertices),
            }

        except Exception as e:
            logger.error(f"Failed to load {model_rel}: {e}")
            errors += 1

    logger.info(f"GT dimensions computed: {len(gt_dims)} success, {errors} errors")
    return gt_dims


def main():
    logger.info("=" * 60)
    logger.info("Phase 3: 데이터 준비")
    logger.info("=" * 60)

    # 1. 필터링
    entries = load_and_filter_pix3d()

    # 2. 층화 추출
    samples = stratified_sample(entries)
    samples_json = build_samples_json(samples)

    # 3. benchmark_samples.json 저장
    SAMPLES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLES_JSON, "w") as f:
        json.dump(samples_json, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {SAMPLES_JSON} ({len(samples_json)} samples)")

    # 4. GT 치수 생성
    gt_dims = compute_gt_dimensions(samples_json)

    with open(GT_DIMENSIONS_JSON, "w") as f:
        json.dump(gt_dims, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {GT_DIMENSIONS_JSON} ({len(gt_dims)} models)")

    # 5. 요약
    logger.info(f"\n{'='*60}")
    logger.info("데이터 준비 완료")
    logger.info(f"  Samples: {SAMPLES_JSON}")
    logger.info(f"  GT Dims: {GT_DIMENSIONS_JSON}")
    logger.info(f"  Total samples: {len(samples_json)}")
    logger.info(f"  Unique GT models: {len(gt_dims)}")


if __name__ == "__main__":
    main()
