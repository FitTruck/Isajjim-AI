"""
worker_ablation.py가 소비할 수 있는 포맷으로 ABO 샘플 변환.

- abo_samples_500.json 로드
- mask_ok=True 인 샘플만 필터
- Pix3D 형식 (sample_id, category, img_path, mask_path, model_path) 로 변환
- 경로는 모두 절대 경로 → worker 의 PIX3D_DIR 베이스 무시

결과: abo_worker_samples.json

Usage:
    python experiments/benchmark/scripts/abo/6_prepare_worker_samples.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    ABO_DATA_DIR,
    ABO_IMAGES_DIR,
    ABO_MASKS_DIR,
    ABO_MESHES_DIR,
    ABO_SAMPLES_JSON,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ABO:prep] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = ABO_DATA_DIR / "abo_worker_samples.json"


def main() -> None:
    with open(ABO_SAMPLES_JSON) as f:
        samples = json.load(f)

    filtered = []
    skipped_no_mask = 0
    skipped_no_image = 0
    skipped_no_mesh = 0

    for s in samples:
        if not s.get("mask_ok"):
            skipped_no_mask += 1
            continue

        img_abs = (ABO_IMAGES_DIR / s["image_path"]).resolve()
        mask_abs = (ABO_MASKS_DIR / f"{s['item_id']}.png").resolve()
        mesh_abs = (ABO_MESHES_DIR / s["model_path"]).resolve()

        if not img_abs.exists():
            skipped_no_image += 1
            continue
        if not mask_abs.exists():
            skipped_no_mask += 1
            continue
        # mesh 없어도 SAM-3D 추론은 가능. nCD 측정에만 필요.

        filtered.append({
            "sample_id": s["sample_id"],
            "category": s["base_name"],
            "img_path": str(img_abs),
            "mask_path": str(mask_abs),
            "model_path": str(mesh_abs) if mesh_abs.exists() else "",
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Prepared {len(filtered)} worker samples "
        f"(skipped: no_mask={skipped_no_mask}, no_image={skipped_no_image})"
    )
    logger.info(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
