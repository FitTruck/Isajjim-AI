"""
ABO 이미지에 YOLOE-seg 적용 → 마스크 생성

- abo_samples_500.json 로드
- 각 이미지에 YOLOE-seg 실행 (KB 전체 클래스 세팅)
- base_name 이 기대 카테고리와 일치하는 탐지 중 가장 큰 박스 선택
- 실패 시 fallback: (1) KB 내 임의 탐지 중 최대 bbox, (2) bbox 없음 → skip
- 마스크 저장: ABO_MASKS_DIR/{item_id}.png (binary 0/255)
- 결과: abo_samples_500.json 업데이트 (mask_ok, mask_path, detected_label, bbox, confidence)

Usage:
    python experiments/benchmark/scripts/abo/5_yoloe_preprocess.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Set env before torch import
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    ABO_IMAGES_DIR,
    ABO_MASKS_DIR,
    ABO_SAMPLES_JSON,
    PROJECT_ROOT,
    SEED,
)

# 재현성
torch.manual_seed(SEED)
np.random.seed(SEED)

# KB 정적 로드 (AI 파이프라인 __init__ 우회)
sys.path.insert(0, str(PROJECT_ROOT))
from ai.data.knowledge_base import (
    FURNITURE_DB,
    get_all_yolo_classes,
    get_yolo_class_mapping,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ABO:yolo] %(message)s")
logger = logging.getLogger(__name__)

YOLOE_MODEL = str(PROJECT_ROOT / "yoloe-26x-seg.pt")
CONFIDENCE = 0.15  # ABO catalog shot 은 clean 하므로 낮춰도 false positive 적음
MIN_MASK_PIXELS = 100


def load_yoloe():
    from ultralytics import YOLOE
    logger.info(f"Loading YOLOE from {YOLOE_MODEL}")
    model = YOLOE(YOLOE_MODEL)
    classes = get_all_yolo_classes()
    logger.info(f"Setting {len(classes)} KB classes")
    model.set_classes(classes)
    if torch.cuda.is_available():
        model.to("cuda")
    return model


def yolo_class_to_base_name() -> dict[str, str]:
    """{YOLO class name: base_name} 역매핑."""
    mapping = get_yolo_class_mapping()  # {yolo_cls: {base_name, db_key}}
    return {k: v["base_name"] for k, v in mapping.items()}


def process_image(
    model,
    image_path: Path,
    expected_base: str,
    yolo_to_base: dict[str, str],
) -> dict | None:
    """단일 이미지 추론 + 최적 detection 선택.

    Returns:
      {
        "mask": np.ndarray(H,W) uint8 0/255,
        "bbox": [x1,y1,x2,y2],
        "confidence": float,
        "detected_label": str,
        "detected_base": str,
        "matched_expected": bool,
      } or None
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.warning(f"Image load fail: {image_path} → {e}")
        return None

    W, H = img.size
    try:
        results = model.predict(img, conf=CONFIDENCE, verbose=False)[0]
    except Exception as e:
        logger.warning(f"YOLOE predict fail: {image_path} → {e}")
        return None

    if results.boxes is None or len(results.boxes) == 0:
        return None

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    names = [model.names[int(c)] for c in class_ids]
    bases = [yolo_to_base.get(n) for n in names]

    masks_raw = results.masks.data.cpu().numpy() if results.masks is not None else None

    # 1) expected_base 와 일치하는 detection 우선
    def box_area(b):
        return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))

    candidates = []
    for i in range(len(boxes)):
        ba = box_area(boxes[i])
        if ba <= 0:
            continue
        candidates.append((i, bases[i], ba, scores[i]))

    # 우선순위: 기대 match → 면적 최대 / 폴백: KB 내 → 면적 최대
    matched = [c for c in candidates if c[1] == expected_base]
    pick = None
    matched_flag = False
    if matched:
        pick = max(matched, key=lambda c: c[2])
        matched_flag = True
    else:
        in_kb = [c for c in candidates if c[1] is not None]
        if in_kb:
            pick = max(in_kb, key=lambda c: c[2])

    if pick is None:
        return None

    idx, det_base, area, conf = pick
    bbox = boxes[idx].tolist()

    # 마스크 추출
    if masks_raw is None:
        # segmentation 없을 때 → bbox 로 fill 대체 (품질 떨어지지만 fallback)
        mask = np.zeros((H, W), dtype=np.uint8)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        mask[y1:y2, x1:x2] = 255
    else:
        m = masks_raw[idx]
        if m.shape != (H, W):
            import cv2
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = ((m > 0.5).astype(np.uint8)) * 255

    if int(np.sum(mask > 0)) < MIN_MASK_PIXELS:
        return None

    return {
        "mask": mask,
        "bbox": bbox,
        "confidence": float(conf),
        "detected_label": names[idx],
        "detected_base": det_base,
        "matched_expected": matched_flag,
    }


def main() -> None:
    logger.info("=" * 60)
    logger.info("ABO YOLOE 마스크 전처리")
    logger.info("=" * 60)

    with open(ABO_SAMPLES_JSON) as f:
        samples = json.load(f)
    logger.info(f"Loaded {len(samples)} samples")

    ABO_MASKS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_yoloe()
    yolo_to_base = yolo_class_to_base_name()

    ok = 0
    fail = 0
    mismatch = 0
    t0 = time.time()

    updated_samples = []
    for i, s in enumerate(samples):
        img_path = ABO_IMAGES_DIR / s["image_path"]
        if not img_path.exists():
            s["mask_ok"] = False
            s["mask_error"] = "image_missing"
            updated_samples.append(s)
            fail += 1
            continue

        expected_base = s["base_name"]
        out_mask = ABO_MASKS_DIR / f"{s['item_id']}.png"

        if out_mask.exists():
            # 재사용. 메타만 복원
            s.setdefault("mask_ok", True)
            s["mask_path"] = str(out_mask.relative_to(ABO_MASKS_DIR.parent))
            updated_samples.append(s)
            ok += 1
            continue

        det = process_image(model, img_path, expected_base, yolo_to_base)
        if det is None:
            s["mask_ok"] = False
            s["mask_error"] = "no_detection"
            updated_samples.append(s)
            fail += 1
        else:
            Image.fromarray(det["mask"]).save(out_mask)
            s["mask_ok"] = True
            s["mask_path"] = str(out_mask.relative_to(ABO_MASKS_DIR.parent))
            s["bbox"] = det["bbox"]
            s["confidence"] = det["confidence"]
            s["detected_label"] = det["detected_label"]
            s["detected_base"] = det["detected_base"]
            s["matched_expected"] = det["matched_expected"]
            updated_samples.append(s)
            ok += 1
            if not det["matched_expected"]:
                mismatch += 1

        done = i + 1
        if done % 50 == 0 or done == len(samples):
            elapsed = time.time() - t0
            rate = done / max(1e-3, elapsed)
            eta = (len(samples) - done) / max(1e-3, rate)
            logger.info(
                f"[{done}/{len(samples)}] ok={ok} fail={fail} mismatch={mismatch} "
                f"rate={rate:.1f}/s eta={eta:.0f}s"
            )

    with open(ABO_SAMPLES_JSON, "w") as f:
        json.dump(updated_samples, f, indent=2, ensure_ascii=False)

    logger.info(
        f"\nDone. ok={ok} fail={fail} mismatch(class)={mismatch} "
        f"elapsed={time.time()-t0:.0f}s"
    )
    logger.info(f"Samples json updated: {ABO_SAMPLES_JSON}")


if __name__ == "__main__":
    main()
