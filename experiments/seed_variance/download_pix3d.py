"""
Pix3D 데이터셋 다운로드 및 샘플링

1. pix3d.json 메타데이터 다운로드
2. 카테고리별 균등 샘플링 (총 50개)
3. 선택된 이미지 + GT 마스크만 다운로드 (디스크 절약)
"""

import json
import os
import sys
import random
import logging
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from urllib.error import URLError
import zipfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PIX3D_DIR, DATA_DIR, SAMPLE_LIST,
    CATEGORY_SAMPLES, MIN_IMAGE_SIZE, MAX_TRUNCATION_RATIO, MIN_MASK_PIXELS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PIX3D_URL = "http://pix3d.csail.mit.edu/data/pix3d.zip"
PIX3D_JSON_URL = "http://pix3d.csail.mit.edu/data/pix3d.json"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download file with progress."""
    if dest.exists():
        logger.info(f"Already exists: {dest}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {desc or url} → {dest}")

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                print(f"\r  {pct:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="", flush=True)

        urlretrieve(url, str(dest), reporthook=_progress)
        print()  # newline after progress
        return True
    except (URLError, OSError) as e:
        logger.error(f"Download failed: {e}")
        return False


def download_pix3d_full() -> bool:
    """Download full Pix3D dataset (zip)."""
    zip_path = DATA_DIR / "pix3d.zip"

    if PIX3D_DIR.exists() and (PIX3D_DIR / "pix3d.json").exists():
        logger.info("Pix3D already extracted")
        return True

    if not zip_path.exists():
        if not download_file(PIX3D_URL, zip_path, "Pix3D dataset"):
            return False

    logger.info("Inspecting zip structure...")
    PIX3D_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()

        # Detect zip prefix (could be "pix3d/", "", or other)
        json_files = [n for n in names if n.endswith("pix3d.json")]
        if not json_files:
            logger.error("pix3d.json not found in zip!")
            return False

        # Determine prefix from json location
        json_path = json_files[0]
        prefix = json_path.replace("pix3d.json", "")
        logger.info(f"Zip prefix: '{prefix}', entries: {len(names)}")

        extracted = 0
        skipped = 0
        for member in names:
            if not member.startswith(prefix):
                continue

            rel_path = member[len(prefix):]
            if not rel_path:
                continue

            # Skip 3D models to save disk space (only need img + mask + metadata)
            if rel_path.startswith("model/"):
                skipped += 1
                continue

            target = PIX3D_DIR / rel_path
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1

                if extracted % 1000 == 0:
                    logger.info(f"  Extracted {extracted} files...")

    logger.info(f"Extracted {extracted} files, skipped {skipped} model files")

    # Verify extraction
    if (PIX3D_DIR / "pix3d.json").exists():
        # Clean up zip to save space
        zip_path.unlink()
        logger.info("Pix3D extracted successfully (3D models skipped)")
        return True
    else:
        logger.error("pix3d.json not found after extraction!")
        return False


def load_pix3d_metadata() -> list:
    """Load pix3d.json metadata."""
    json_path = PIX3D_DIR / "pix3d.json"

    if not json_path.exists():
        # Try downloading just the JSON first
        logger.info("pix3d.json not found, attempting full download...")
        if not download_pix3d_full():
            raise FileNotFoundError("Cannot download Pix3D dataset")

    with open(json_path, 'r') as f:
        return json.load(f)


def filter_candidates(metadata: list, category: str) -> list:
    """Filter Pix3D entries for a category based on quality criteria."""
    candidates = []

    for entry in metadata:
        if entry.get("category") != category:
            continue

        # Check truncation (가려짐)
        truncated = entry.get("truncated", False)
        if truncated:
            continue

        # Check image path exists
        img_path = PIX3D_DIR / entry.get("img", "")
        mask_path = PIX3D_DIR / entry.get("mask", "")

        if not img_path.exists() or not mask_path.exists():
            continue

        # Check image size (lazy - from metadata if available, otherwise skip)
        img_size = entry.get("img_size", [0, 0])
        if isinstance(img_size, list) and len(img_size) >= 2:
            w, h = img_size[0], img_size[1]
            if min(w, h) < MIN_IMAGE_SIZE:
                continue

        # Check occlusion ratio
        occluded = entry.get("occluded", False)
        if occluded:
            continue

        candidates.append(entry)

    return candidates


def sample_diverse(candidates: list, n: int, seed: int = 2024) -> list:
    """
    Sample n diverse entries from candidates.
    Prioritize shape diversity by using different 3D models.
    """
    if len(candidates) <= n:
        return candidates

    # Group by 3D model to ensure shape diversity
    model_groups = {}
    for entry in candidates:
        model_id = entry.get("model", "unknown")
        if model_id not in model_groups:
            model_groups[model_id] = []
        model_groups[model_id].append(entry)

    rng = random.Random(seed)
    selected = []

    # First pass: one from each unique model
    model_keys = list(model_groups.keys())
    rng.shuffle(model_keys)

    for model_id in model_keys:
        if len(selected) >= n:
            break
        entries = model_groups[model_id]
        # Pick the entry with least occlusion / best quality
        entry = rng.choice(entries)
        selected.append(entry)

    # If we need more, add from remaining
    if len(selected) < n:
        remaining = [e for e in candidates if e not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[:n - len(selected)])

    return selected[:n]


def select_samples() -> list:
    """Select 50 samples across 7 categories."""
    metadata = load_pix3d_metadata()
    logger.info(f"Loaded {len(metadata)} Pix3D entries")

    all_selected = []

    for category, n_samples in CATEGORY_SAMPLES.items():
        candidates = filter_candidates(metadata, category)
        logger.info(f"  {category}: {len(candidates)} candidates → sampling {n_samples}")

        if len(candidates) < n_samples:
            logger.warning(f"  {category}: only {len(candidates)} candidates (need {n_samples})")
            n_samples = len(candidates)

        selected = sample_diverse(candidates, n_samples)

        for entry in selected:
            all_selected.append({
                "category": category,
                "img": entry["img"],
                "mask": entry["mask"],
                "model": entry.get("model", ""),
                "img_size": entry.get("img_size", []),
            })

        logger.info(f"  {category}: selected {len(selected)} samples")

    logger.info(f"Total selected: {len(all_selected)} samples")
    return all_selected


def verify_samples(samples: list) -> list:
    """Verify all sample files exist and masks have sufficient pixels."""
    from PIL import Image
    import numpy as np

    verified = []
    for sample in samples:
        img_path = PIX3D_DIR / sample["img"]
        mask_path = PIX3D_DIR / sample["mask"]

        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue
        if not mask_path.exists():
            logger.warning(f"Mask not found: {mask_path}")
            continue

        # Check mask pixel count
        try:
            mask = np.array(Image.open(mask_path).convert("L"))
            pixel_count = np.sum(mask > 0)
            if pixel_count < MIN_MASK_PIXELS:
                logger.warning(f"Mask too small ({pixel_count} pixels): {mask_path}")
                continue
            sample["mask_pixels"] = int(pixel_count)
        except Exception as e:
            logger.warning(f"Cannot read mask {mask_path}: {e}")
            continue

        verified.append(sample)

    logger.info(f"Verified: {len(verified)}/{len(samples)} samples")
    return verified


def main():
    """Main: download Pix3D and select samples."""
    logger.info("=" * 60)
    logger.info("Pix3D Download & Sampling")
    logger.info("=" * 60)

    # Step 1: Download
    if not download_pix3d_full():
        logger.error("Failed to download Pix3D")
        sys.exit(1)

    # Step 2: Select samples
    samples = select_samples()

    # Step 3: Verify
    samples = verify_samples(samples)

    # Step 4: Save sample list
    SAMPLE_LIST.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_LIST, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    logger.info(f"Sample list saved: {SAMPLE_LIST}")
    logger.info(f"Total verified samples: {len(samples)}")

    # Summary
    from collections import Counter
    cats = Counter(s["category"] for s in samples)
    for cat, count in sorted(cats.items()):
        logger.info(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
