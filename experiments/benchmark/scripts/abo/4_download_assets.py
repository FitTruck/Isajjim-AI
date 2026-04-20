"""
ABO 이미지 + 3D 메시 다운로드 (500 샘플)

- abo_samples_500.json 로드
- 이미지: images/original/{path} (SAM-3D 품질 확보 위해 original 사용)
- 메시: 3dmodels/original/{path} (GLB)
- 기존 파일 skip, 실패 로그

Usage:
    python experiments/benchmark/scripts/abo/4_download_assets.py [--images-only] [--meshes-only] [--variant original|small]
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ABO_IMAGES_DIR, ABO_MESHES_DIR, ABO_SAMPLES_JSON

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ABO:dl] %(message)s")
logger = logging.getLogger(__name__)

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"


def download_file(url: str, dest: Path, timeout: int = 120) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True, "skip(exists)"
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, timeout=timeout)
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            tmp.rename(dest)
            return True, "ok"
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            return False, str(e)[:120]
    return False, "gave up"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-only", action="store_true")
    ap.add_argument("--meshes-only", action="store_true")
    ap.add_argument("--image-variant", choices=["original", "small"], default="original",
                    help="original (고해상도, SAM-3D 품질 권장) or small (256px)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with open(ABO_SAMPLES_JSON) as f:
        samples = json.load(f)
    logger.info(f"Loaded {len(samples)} samples")

    # 작업 목록 생성
    tasks: list[tuple[str, str, Path]] = []  # (kind, url, dest)
    if not args.meshes_only:
        for s in samples:
            ip = s["image_path"]  # e.g. "14/14fe8812.jpg"
            url = f"{BASE}/images/{args.image_variant}/{ip}"
            dest = ABO_IMAGES_DIR / ip  # 해시 서브디렉토리 유지
            tasks.append(("img", url, dest))
    if not args.images_only:
        for s in samples:
            mp = s["model_path"]  # e.g. "L/B01N2PLWIL.glb"
            url = f"{BASE}/3dmodels/original/{mp}"
            dest = ABO_MESHES_DIR / mp
            tasks.append(("mesh", url, dest))

    logger.info(f"Queued {len(tasks)} downloads ({args.workers} workers)")

    ok = 0
    fail = 0
    errors: list[tuple[str, str, str]] = []
    skipped = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_file, url, dest): (kind, url, dest)
            for kind, url, dest in tasks
        }
        for i, fut in enumerate(as_completed(futures)):
            kind, url, dest = futures[fut]
            success, msg = fut.result()
            if success:
                ok += 1
                if msg == "skip(exists)":
                    skipped += 1
            else:
                fail += 1
                errors.append((kind, url, msg))

            done = i + 1
            if done % 50 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / max(1e-3, elapsed)
                eta = (len(tasks) - done) / max(1e-3, rate)
                logger.info(
                    f"[{done}/{len(tasks)}] ok={ok} fail={fail} skip={skipped} "
                    f"rate={rate:.1f}/s eta={eta:.0f}s"
                )

    elapsed = time.time() - t0
    logger.info(f"\nDownload done: ok={ok}, fail={fail}, skip={skipped}, elapsed={elapsed:.0f}s")
    if errors:
        logger.warning(f"Failures (showing first 10):")
        for kind, url, msg in errors[:10]:
            logger.warning(f"  [{kind}] {url} → {msg}")


if __name__ == "__main__":
    main()
