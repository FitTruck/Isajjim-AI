"""
ABO 메타데이터 다운로드 + 파싱

- `listings_{0-f}.json.gz` 16 shard 다운로드
- item_dimensions + 3dmodel_id 보유 항목만 필터
- 3dmodels.csv.gz, images.csv.gz 메타 다운로드
- 결과: abo_valid_listings.jsonl (NDJSON), 3dmodels_lookup.csv, images_lookup.csv

Usage:
    python experiments/benchmark/scripts/abo/1_fetch_metadata.py
"""

import gzip
import io
import json
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ABO_METADATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ABO:meta] %(message)s")
logger = logging.getLogger(__name__)

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"
SHARDS = list("0123456789abcdef")


def _get(url: str, timeout: int = 120) -> bytes:
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            logger.warning(f"Fetch fail ({attempt+1}/3) {url}: {e}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Gave up on {url}")


def fetch_and_filter_listings(out_jsonl: Path) -> int:
    """16 shard 순회 → item_dimensions + 3dmodel_id 있는 항목만 NDJSON으로 저장."""
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped_no_dim = 0
    skipped_no_3d = 0

    with open(out_jsonl, "w") as fo:
        for s in SHARDS:
            url = f"{BASE}/listings/metadata/listings_{s}.json.gz"
            logger.info(f"Shard {s}: fetching...")
            raw = _get(url)
            t0 = time.time()
            shard_kept = 0
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                for line in gz:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not obj.get("item_dimensions"):
                        skipped_no_dim += 1
                        continue
                    if not obj.get("3dmodel_id"):
                        skipped_no_3d += 1
                        continue
                    # product_type 은 list[dict(value, language_tag)] 형식 — 첫 영어 값만 저장
                    pt_list = obj.get("product_type") or []
                    pt_value = None
                    for pt in pt_list:
                        if isinstance(pt, dict):
                            pt_value = pt.get("value")
                            if pt_value:
                                break
                    if not pt_value:
                        continue
                    # 경량화: 필요한 필드만 저장
                    slim = {
                        "item_id": obj.get("item_id"),
                        "product_type": pt_value,
                        "item_dimensions": obj.get("item_dimensions"),
                        "main_image_id": obj.get("main_image_id"),
                        "other_image_id": obj.get("other_image_id") or [],
                        "3dmodel_id": obj.get("3dmodel_id"),
                    }
                    fo.write(json.dumps(slim, ensure_ascii=False) + "\n")
                    shard_kept += 1
            kept += shard_kept
            logger.info(
                f"Shard {s}: kept {shard_kept} ({time.time()-t0:.1f}s), total={kept}"
            )

    logger.info(
        f"Filtering complete. kept={kept}, skipped_no_dim={skipped_no_dim}, "
        f"skipped_no_3d={skipped_no_3d}"
    )
    return kept


def fetch_csv(path: str, out_path: Path) -> None:
    url = f"{BASE}/{path}"
    logger.info(f"Fetching {path}...")
    raw = _get(url)
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        data = gz.read()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)
    logger.info(f"  saved: {out_path} ({len(data)/1e6:.1f} MB)")


def main() -> None:
    logger.info("=" * 60)
    logger.info("ABO 메타데이터 다운로드")
    logger.info("=" * 60)

    listings_jsonl = ABO_METADATA_DIR / "abo_valid_listings.jsonl"
    models_csv = ABO_METADATA_DIR / "3dmodels.csv"
    images_csv = ABO_METADATA_DIR / "images.csv"

    # 1) listings shards
    if listings_jsonl.exists() and listings_jsonl.stat().st_size > 0:
        logger.info(f"Exists, skip listings: {listings_jsonl}")
    else:
        fetch_and_filter_listings(listings_jsonl)

    # 2) 3dmodels lookup
    if models_csv.exists():
        logger.info(f"Exists, skip 3dmodels: {models_csv}")
    else:
        fetch_csv("3dmodels/metadata/3dmodels.csv.gz", models_csv)

    # 3) images lookup
    if images_csv.exists():
        logger.info(f"Exists, skip images: {images_csv}")
    else:
        fetch_csv("images/metadata/images.csv.gz", images_csv)

    logger.info("Done.")


if __name__ == "__main__":
    main()
