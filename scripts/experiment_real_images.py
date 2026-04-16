"""
실제 실내 이미지 검증: AR 보존 개선이 다양한 실내 사진에서도 유효한지 확인

Pix3D와 다른 점:
  - 한 이미지에 여러 가구
  - 다양한 각도, 조명, 가림
  - 실제 스마트폰 촬영 (4:3, 16:9 등)

GT가 없으므로 "물리적 타당성 점수"로 평가:
  1. 치수 범위 타당성 (50mm~4000mm)
  2. 비율 타당성 (aspect ratio가 0.1~1.0 범위)
  3. 신뢰도
  4. squash vs AR 보존 간 치수 차이 분석

Usage:
    conda run -n sam3d-objects python /path/to/experiment_real_images.py
"""

import os, sys, time, csv
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

os.chdir(str(ROOT / "boxer"))
sys.path.insert(0, str(ROOT / "boxer"))
sys.path.insert(1, str(ROOT))

from boxernet.boxernet import BoxerNet
from loaders.base_loader import BaseLoader
from utils.tw.camera import get_pinhole_camera
from utils.tw.pose import PoseTW

os.chdir(str(ROOT))

from ai.processors import YoloDetector, AbsoluteVolumeCalculator
from moge.model.v1 import MoGeModel


# 가구 카테고리별 물리적 타당 범위 (mm)
PLAUSIBLE_RANGES = {
    "bed": {"min_long": 1500, "max_long": 2500, "min_short": 200, "max_short": 1000},
    "sofa": {"min_long": 800, "max_long": 3500, "min_short": 300, "max_short": 1200},
    "chair": {"min_long": 400, "max_long": 1500, "min_short": 200, "max_short": 800},
    "desk": {"min_long": 600, "max_long": 2000, "min_short": 200, "max_short": 1000},
    "table": {"min_long": 400, "max_long": 2500, "min_short": 200, "max_short": 1200},
    "nightstand": {"min_long": 300, "max_long": 800, "min_short": 200, "max_short": 600},
    "cabinet": {"min_long": 500, "max_long": 2500, "min_short": 200, "max_short": 1000},
    "monitor": {"min_long": 400, "max_long": 1800, "min_short": 50, "max_short": 500},
    "default": {"min_long": 100, "max_long": 4000, "min_short": 50, "max_short": 2000},
}

def get_category(label: str) -> str:
    label_lower = label.lower()
    for cat in PLAUSIBLE_RANGES:
        if cat in label_lower:
            return cat
    return "default"

def plausibility_score(dims_mm: List[float], category: str) -> float:
    """물리적 타당성 점수 (0~1, 1이 최고)"""
    s = sorted(dims_mm)
    if s[2] <= 0:
        return 0.0

    ranges = PLAUSIBLE_RANGES.get(category, PLAUSIBLE_RANGES["default"])
    score = 1.0

    # 가장 긴 축 검사
    if s[2] < ranges["min_long"] or s[2] > ranges["max_long"]:
        score *= 0.3
    # 가장 짧은 축 검사
    if s[0] < ranges["min_short"] or s[0] > ranges["max_short"]:
        score *= 0.5
    # aspect ratio 검사 (극단적 비율 감점)
    ar = s[0] / s[2]
    if ar < 0.02 or ar > 0.95:
        score *= 0.5

    return score


class ImprovedBoxer:
    def __init__(self, device="cuda"):
        self.device = device
        ckpt = str(ROOT / "boxer" / "ckpts" / "boxernet_hw960in4x6d768-wssxpf9p.ckpt")
        self.model = BoxerNet.load_from_checkpoint(ckpt, device=device)
        self.model.eval()
        self._R_yz = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float32)
        self._t = np.zeros(3, dtype=np.float32)

    @torch.no_grad()
    def predict_batch(self, image, bboxes, depth_map=None, preserve_ar=False):
        """여러 bbox를 한 번에 추론"""
        if not bboxes:
            return []

        W, H = image.size
        tgt = 960

        if preserve_ar:
            scale = tgt / max(W, H)
            nw, nh = int(W * scale), int(H * scale)
            img_r = image.resize((nw, nh), Image.BILINEAR)
            img_pad = Image.new("RGB", (tgt, tgt), (0, 0, 0))
            px, py = (tgt - nw) // 2, (tgt - nh) // 2
            img_pad.paste(img_r, (px, py))
        else:
            img_pad = image.resize((tgt, tgt), Image.BILINEAR)
            scale = None
            px, py = 0, 0

        img_t = torch.from_numpy(np.array(img_pad.convert("RGB"))).permute(2,0,1).unsqueeze(0).float()/255.0
        img_t = img_t.to(self.device)

        f = max(W, H) * 0.8
        if preserve_ar:
            fs = f * scale
            cxs, cys = W/2*scale+px, H/2*scale+py
        else:
            fs = f * (tgt / max(W, H))
            cxs, cys = tgt/2, tgt/2

        cam = get_pinhole_camera([fs, fs, cxs, cys], tgt, tgt).to(self.device)
        R, t = self._R_yz.copy(), self._t.copy()
        pose = PoseTW.from_Rt(torch.from_numpy(R), torch.from_numpy(t)).to(self.device)

        if depth_map is not None:
            if preserve_ar:
                dr = cv2.resize(depth_map.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST)
                dr_pad = np.zeros((tgt, tgt), dtype=np.float32)
                dr_pad[py:py+nh, px:px+nw] = dr
            else:
                dr_pad = cv2.resize(depth_map.astype(np.float32), (tgt, tgt), interpolation=cv2.INTER_NEAREST)
            sdp = BaseLoader.sdp_from_depth(dr_pad, fs, fs, cxs, cys, R, t, 10000).to(self.device)
        else:
            sdp = torch.zeros(0, 3, device=self.device)

        # 모든 bbox → Boxer 포맷
        bb_list = []
        for x1, y1, x2, y2 in bboxes:
            if preserve_ar:
                bb_list.append([x1*scale+px, x2*scale+px, y1*scale+py, y2*scale+py])
            else:
                sx, sy = tgt/W, tgt/H
                bb_list.append([x1*sx, x2*sx, y1*sy, y2*sy])
        bb2d = torch.tensor([bb_list], dtype=torch.float32, device=self.device)

        out = self.model({
            "img0": img_t, "cam0": cam, "T_world_rig0": pose,
            "rotated0": torch.tensor([False], device=self.device),
            "sdp_w": sdp, "bb2d": bb2d,
        })
        obbs = out.get("obbs_pr_w")
        if obbs is None:
            return [None] * len(bboxes)

        results = []
        for i in range(min(len(bboxes), obbs.shape[-2])):
            obb = obbs[..., i, :]
            dims = sorted([float(d.abs())*1000 for d in obb.bb3_diagonal.squeeze()])
            conf = float(obb.prob.squeeze())
            results.append((dims, conf))
        return results


def run():
    out_dir = ROOT / "experiments" / "boxer_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  실제 실내 이미지 검증: squash vs AR 보존")
    print("=" * 70)

    # Models
    print("\n  Loading models...")
    yolo = YoloDetector(device_id=0)
    boxer = ImprovedBoxer()
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").cuda().eval()
    abs_calc = AbsoluteVolumeCalculator()
    print(f"  GPU: {torch.cuda.memory_allocated()/1e6:.0f}MB\n")

    image_dir = ROOT / "ai" / "imgs"
    images = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png')])
    print(f"  {len(images)} images\n")

    all_rows = []

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        ar = max(W, H) / min(W, H)
        print(f"--- {img_path.name} ({W}x{H}, AR={ar:.2f}) ---")

        # YOLOE 탐지
        results = yolo.detect_smart(image, return_masks=False)
        if not results or len(results["boxes"]) == 0:
            print("  No detections")
            continue

        # 필터링
        detections = []
        for j, (box, score, label) in enumerate(zip(results["boxes"], results["scores"], results["labels"])):
            x1, y1, x2, y2 = map(int, box)
            if (x2-x1) < 20 or (y2-y1) < 20:
                continue
            detections.append({"bbox": [x1, y1, x2, y2], "label": label, "conf": float(score)})

        if not detections:
            continue

        bboxes = [d["bbox"] for d in detections]

        # MoGe depth
        img_moge = image.resize((512, 512))
        img_t = torch.from_numpy(np.array(img_moge)).permute(2,0,1).float()/255.0
        with torch.no_grad():
            moge_out = moge.infer(img_t.cuda(), force_projection=False)
        moge_depth = moge_out["depth"].cpu().numpy()

        # Boxer: squash (기존)
        t0 = time.perf_counter()
        results_squash = boxer.predict_batch(image, bboxes, depth_map=moge_depth, preserve_ar=False)
        time_squash = (time.perf_counter() - t0) * 1000

        # Boxer: AR 보존
        t0 = time.perf_counter()
        results_ar = boxer.predict_batch(image, bboxes, depth_map=moge_depth, preserve_ar=True)
        time_ar = (time.perf_counter() - t0) * 1000

        print(f"  {len(detections)} objects (squash={time_squash:.0f}ms, AR={time_ar:.0f}ms)")

        for j, det in enumerate(detections):
            label = det["label"]
            cat = get_category(label)

            # Rule-based
            rule = abs_calc.calculate_absolute_volume(label.upper(), None, 1.0, 1.5, 0.8)
            rule_dims = sorted([rule.width_mm, rule.depth_mm, rule.height_mm])
            rule_plaus = plausibility_score(rule_dims, cat)

            # Boxer squash
            if j < len(results_squash) and results_squash[j]:
                sq_dims, sq_conf = results_squash[j]
                sq_plaus = plausibility_score(sq_dims, cat)
            else:
                sq_dims, sq_conf, sq_plaus = [0,0,0], 0, 0

            # Boxer AR
            if j < len(results_ar) and results_ar[j]:
                ar_dims, ar_conf = results_ar[j]
                ar_plaus = plausibility_score(ar_dims, cat)
            else:
                ar_dims, ar_conf, ar_plaus = [0,0,0], 0, 0

            # 치수 변화량 (squash vs AR)
            if sq_dims[2] > 0 and ar_dims[2] > 0:
                dim_change = np.mean([abs(s-a)/max(s,1) for s, a in zip(sq_dims, ar_dims)]) * 100
            else:
                dim_change = 0

            row = {
                "image": img_path.name, "label": label, "category": cat,
                "img_ar": ar, "yolo_conf": det["conf"],
                "sq_dims": sq_dims, "sq_conf": sq_conf, "sq_plaus": sq_plaus,
                "ar_dims": ar_dims, "ar_conf": ar_conf, "ar_plaus": ar_plaus,
                "rule_dims": rule_dims, "rule_plaus": rule_plaus,
                "dim_change_pct": dim_change,
            }
            all_rows.append(row)

            better = "AR" if ar_plaus > sq_plaus else ("SQ" if sq_plaus > ar_plaus else "TIE")
            print(f"  [{j}] {label} ({cat})")
            print(f"      Squash: {sq_dims[0]:.0f}x{sq_dims[1]:.0f}x{sq_dims[2]:.0f}mm conf={sq_conf:.2f} plaus={sq_plaus:.2f}")
            print(f"      AR:     {ar_dims[0]:.0f}x{ar_dims[1]:.0f}x{ar_dims[2]:.0f}mm conf={ar_conf:.2f} plaus={ar_plaus:.2f}  [{better}]")
            print(f"      Rule:   {rule_dims[0]:.0f}x{rule_dims[1]:.0f}x{rule_dims[2]:.0f}mm plaus={rule_plaus:.2f}")
            print(f"      치수변화: {dim_change:.1f}%")

    # 종합 분석
    print(f"\n{'='*70}")
    print(f"  종합 분석 ({len(all_rows)} objects)")
    print(f"{'='*70}")

    # 물리적 타당성 점수
    sq_plaus = [r["sq_plaus"] for r in all_rows]
    ar_plaus = [r["ar_plaus"] for r in all_rows]
    ru_plaus = [r["rule_plaus"] for r in all_rows]

    print(f"\n  --- 물리적 타당성 점수 (1.0이 최고) ---")
    print(f"  {'Method':<20} | {'Mean':<8} | {'Median':<8} | {'>0.5':<8} | {'>0.8':<8}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for name, scores in [("Boxer (squash)", sq_plaus), ("Boxer (AR 보존)", ar_plaus), ("Rule-based", ru_plaus)]:
        arr = np.array(scores)
        print(f"  {name:<20} | {arr.mean():>6.3f}  | {np.median(arr):>6.3f}  | "
              f"{(arr>0.5).sum()/len(arr)*100:>5.1f}% | {(arr>0.8).sum()/len(arr)*100:>5.1f}%")

    # AR에 따른 개선 분석
    print(f"\n  --- 이미지 AR별 치수 변화량 ---")
    square = [r for r in all_rows if r["img_ar"] < 1.1]
    non_square = [r for r in all_rows if r["img_ar"] >= 1.1]
    print(f"  정방형 (AR<1.1): {len(square)}개 → 평균 치수 변화 {np.mean([r['dim_change_pct'] for r in square]):.1f}%")
    print(f"  비정방형 (AR≥1.1): {len(non_square)}개 → 평균 치수 변화 {np.mean([r['dim_change_pct'] for r in non_square]):.1f}%")

    # AR 보존이 squash보다 나은 비율
    ar_wins = sum(1 for r in all_rows if r["ar_plaus"] > r["sq_plaus"])
    sq_wins = sum(1 for r in all_rows if r["sq_plaus"] > r["ar_plaus"])
    ties = len(all_rows) - ar_wins - sq_wins
    print(f"\n  AR 보존 vs Squash: AR 승={ar_wins}, Squash 승={sq_wins}, 무승부={ties}")

    # 비정방형에서만
    if non_square:
        ar_wins_ns = sum(1 for r in non_square if r["ar_plaus"] > r["sq_plaus"])
        sq_wins_ns = sum(1 for r in non_square if r["sq_plaus"] > r["ar_plaus"])
        print(f"  비정방형만: AR 승={ar_wins_ns}, Squash 승={sq_wins_ns}")

    # 신뢰도 비교
    sq_confs = [r["sq_conf"] for r in all_rows if r["sq_conf"] > 0]
    ar_confs = [r["ar_conf"] for r in all_rows if r["ar_conf"] > 0]
    print(f"\n  --- Boxer 신뢰도 ---")
    print(f"  Squash: {np.mean(sq_confs):.3f} ± {np.std(sq_confs):.3f}")
    print(f"  AR:     {np.mean(ar_confs):.3f} ± {np.std(ar_confs):.3f}")

    # CSV
    csv_path = out_dir / "real_images_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "label", "category", "img_ar",
                     "sq_short", "sq_mid", "sq_long", "sq_conf", "sq_plaus",
                     "ar_short", "ar_mid", "ar_long", "ar_conf", "ar_plaus",
                     "rule_short", "rule_mid", "rule_long", "rule_plaus",
                     "dim_change_pct"])
        for r in all_rows:
            w.writerow([
                r["image"], r["label"], r["category"], f"{r['img_ar']:.2f}",
                *[f"{d:.0f}" for d in r["sq_dims"]], f"{r['sq_conf']:.3f}", f"{r['sq_plaus']:.3f}",
                *[f"{d:.0f}" for d in r["ar_dims"]], f"{r['ar_conf']:.3f}", f"{r['ar_plaus']:.3f}",
                *[f"{d:.0f}" for d in r["rule_dims"]], f"{r['rule_plaus']:.3f}",
                f"{r['dim_change_pct']:.1f}",
            ])
    print(f"\n  CSV: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run()
