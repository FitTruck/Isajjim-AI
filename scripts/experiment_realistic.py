"""
공정 비교 실험: YOLOE 탐지 라벨 기반 Rule vs Boxer

이전 실험의 문제:
  - Pix3D GT category를 규칙기반에 직접 전달 → 정답 라벨 = 유리
  - Boxer는 class-agnostic → 라벨 무관
  → 규칙기반에 유리한 불공정 비교

이번 실험:
  1. Pix3D 이미지에 YOLOE를 실제로 돌림
  2. YOLOE가 탐지한 bbox 중 GT bbox와 IoU가 높은 것을 매칭
  3. YOLOE의 라벨(오류 포함)을 규칙기반에 전달
  4. YOLOE의 bbox를 Boxer에 전달
  → YOLOE 분류 오류의 영향까지 포함한 공정 비교

Usage:
    conda run -n sam3d-objects python /path/to/experiment_realistic.py
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

os.chdir(str(ROOT / "boxer"))
sys.path.insert(0, str(ROOT / "boxer"))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(ROOT / "depth-anything-v2" / "metric_depth"))

from boxernet.boxernet import BoxerNet
from loaders.base_loader import BaseLoader
from utils.tw.camera import get_pinhole_camera
from utils.tw.pose import PoseTW

os.chdir(str(ROOT))

from ai.processors import YoloDetector, AbsoluteVolumeCalculator
from moge.model.v1 import MoGeModel

PIX3D_ROOT = ROOT / "experiments" / "seed_variance" / "data" / "pix3d"


def compute_iou(box1, box2):
    """IoU 계산. box format: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compute_aspect_ratio(dims):
    s = sorted(dims)
    return [s[0] / s[2], s[1] / s[2], 1.0] if s[2] > 0 else [0, 0, 0]


def aspect_ratio_error(pred, gt):
    return float(np.mean([abs(p - g) for p, g in zip(pred, gt)])) if pred and gt else 1.0


def load_samples(max_per_cat=30):
    with open(PIX3D_ROOT / "pix3d.json") as f:
        data = json.load(f)
    target_cats = {"bed", "sofa", "chair", "desk", "table", "bookcase", "wardrobe"}
    samples, cat_count, seen = [], defaultdict(int), {}
    for d in data:
        cat = d["category"]
        if cat not in target_cats or d["truncated"] or d["occluded"]:
            continue
        if cat_count[cat] >= max_per_cat:
            continue
        mp = d["model"]
        if mp not in seen:
            try:
                mesh = trimesh.load(str(PIX3D_ROOT / mp), force="mesh")
                seen[mp] = sorted((mesh.bounds[1] - mesh.bounds[0]).tolist())
            except Exception:
                continue
        samples.append({
            "img_path": str(PIX3D_ROOT / d["img"]),
            "category": cat,
            "gt_bbox": d["bbox"],  # [x1, y1, x2, y2]
            "img_size": d["img_size"],
            "focal_length": d["focal_length"],
            "gt_ratio": compute_aspect_ratio(seen[mp]),
        })
        cat_count[cat] += 1
    return samples


class BoxerPredictor:
    def __init__(self, device="cuda"):
        self.device = device
        ckpt = str(ROOT / "boxer" / "ckpts" / "boxernet_hw960in4x6d768-wssxpf9p.ckpt")
        self.model = BoxerNet.load_from_checkpoint(ckpt, device=device)
        self.model.eval()
        self._R_yz = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        self._t = np.zeros(3, dtype=np.float32)

    @torch.no_grad()
    def predict(self, image, bbox, focal_mm, depth_map=None):
        W, H = image.size
        tgt = 960
        sx, sy = tgt / W, tgt / H
        img_r = image.resize((tgt, tgt), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img_r.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_t = img_t.to(self.device)
        f_px = focal_mm * max(W, H) / 36.0
        f_s = f_px * (tgt / max(W, H))
        cam = get_pinhole_camera([f_s, f_s, tgt / 2, tgt / 2], tgt, tgt).to(self.device)
        R, t = self._R_yz.copy(), self._t.copy()
        pose = PoseTW.from_Rt(torch.from_numpy(R), torch.from_numpy(t)).to(self.device)

        if depth_map is not None:
            dr = cv2.resize(depth_map.astype(np.float32), (tgt, tgt), interpolation=cv2.INTER_NEAREST)
            sdp = BaseLoader.sdp_from_depth(dr, f_s, f_s, tgt / 2, tgt / 2, R, t, 10000).to(self.device)
        else:
            sdp = torch.zeros(0, 3, device=self.device)

        x1, y1, x2, y2 = bbox
        bb2d = torch.tensor([[[x1 * sx, x2 * sx, y1 * sy, y2 * sy]]], dtype=torch.float32, device=self.device)

        out = self.model({
            "img0": img_t, "cam0": cam, "T_world_rig0": pose,
            "rotated0": torch.tensor([False], device=self.device),
            "sdp_w": sdp, "bb2d": bb2d,
        })
        obbs = out.get("obbs_pr_w")
        if obbs is None or obbs.shape[-2] == 0:
            return None
        obb = obbs[..., 0, :]
        dims = sorted([float(d.abs()) for d in obb.bb3_diagonal.squeeze()])
        return (dims[2] * 1000, dims[0] * 1000, dims[1] * 1000,
                float(obb.bb3_volumes.squeeze().abs()), float(obb.prob.squeeze()))


def run():
    out_dir = ROOT / "experiments" / "boxer_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  공정 비교: YOLOE 실제 탐지 라벨 기반 Rule vs Boxer")
    print("=" * 70)

    samples = load_samples(max_per_cat=30)
    print(f"\n  {len(samples)} Pix3D samples loaded")

    # Models
    print("\n  Loading models...")
    yolo = YoloDetector(device_id=0)
    print("  [OK] YOLOE")
    boxer = BoxerPredictor()
    print("  [OK] BoxerNet")
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").cuda().eval()
    print("  [OK] MoGe")
    abs_calc = AbsoluteVolumeCalculator()
    print(f"  GPU: {torch.cuda.memory_allocated()/1e6:.0f}MB\n")

    # 통계
    stats = {
        "total": 0,
        "yolo_matched": 0,
        "yolo_no_match": 0,
        "yolo_correct_label": 0,
        "yolo_wrong_label": 0,
        "rule_fallback_100mm": 0,
    }

    # Pix3D category → YOLOE 기대 라벨 (대소문자 무관 부분 매칭)
    EXPECTED_LABELS = {
        "bed": ["bed"],
        "sofa": ["sofa", "couch"],
        "chair": ["chair", "stool", "armchair"],
        "desk": ["desk", "table"],
        "table": ["table", "desk", "dining"],
        "bookcase": ["bookcase", "shelf", "bookshelf", "cabinet"],
        "wardrobe": ["wardrobe", "cabinet", "closet"],
    }

    conditions = ["boxer_no_depth", "boxer_moge", "rule_yolo_label", "rule_gt_label"]
    results = {c: [] for c in conditions}
    rows = []

    for i, s in enumerate(samples):
        try:
            image = Image.open(s["img_path"]).convert("RGB")
        except Exception:
            continue

        gt_ratio = s["gt_ratio"]
        gt_bbox = s["gt_bbox"]
        W, H = s["img_size"]
        fl = s["focal_length"]
        gt_cat = s["category"]
        stats["total"] += 1

        # === YOLOE 탐지 ===
        yolo_results = yolo.detect_smart(image, return_masks=False)

        yolo_label = None
        yolo_bbox = gt_bbox  # fallback: GT bbox

        if yolo_results and len(yolo_results["boxes"]) > 0:
            # GT bbox와 IoU가 가장 높은 YOLOE 탐지를 매칭
            best_iou = 0
            best_idx = -1
            for j, box in enumerate(yolo_results["boxes"]):
                iou = compute_iou(gt_bbox, [int(x) for x in box])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= 0.3 and best_idx >= 0:
                yolo_label = yolo_results["labels"][best_idx]
                yolo_bbox = [int(x) for x in yolo_results["boxes"][best_idx]]
                stats["yolo_matched"] += 1

                # 라벨 정확도 체크
                expected = EXPECTED_LABELS.get(gt_cat, [])
                label_correct = any(exp in yolo_label.lower() for exp in expected)
                if label_correct:
                    stats["yolo_correct_label"] += 1
                else:
                    stats["yolo_wrong_label"] += 1
            else:
                stats["yolo_no_match"] += 1
        else:
            stats["yolo_no_match"] += 1

        # === MoGe depth ===
        img_moge = image.resize((512, 512))
        img_t = torch.from_numpy(np.array(img_moge)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            moge_out = moge.infer(img_t.cuda(), force_projection=False)
        moge_depth = moge_out["depth"].cpu().numpy()

        # === Boxer (no depth) ===
        out_nd = boxer.predict(image, yolo_bbox, fl, depth_map=None)
        if out_nd:
            dims_nd = sorted([out_nd[0], out_nd[1], out_nd[2]])
            are_nd = aspect_ratio_error(compute_aspect_ratio(dims_nd), gt_ratio)
            conf_nd = out_nd[4]
        else:
            are_nd, conf_nd, dims_nd = 1.0, 0, [0, 0, 0]
        results["boxer_no_depth"].append(are_nd)

        # === Boxer + MoGe ===
        out_mg = boxer.predict(image, yolo_bbox, fl, depth_map=moge_depth)
        if out_mg:
            dims_mg = sorted([out_mg[0], out_mg[1], out_mg[2]])
            are_mg = aspect_ratio_error(compute_aspect_ratio(dims_mg), gt_ratio)
            conf_mg = out_mg[4]
        else:
            are_mg, conf_mg, dims_mg = 1.0, 0, [0, 0, 0]
        results["boxer_moge"].append(are_mg)

        # === Rule-based with YOLOE label (실제 파이프라인) ===
        if yolo_label:
            rule_yolo = abs_calc.calculate_absolute_volume(
                label=yolo_label.upper(), type_name=None,
                rel_width=1.0, rel_depth=1.5, rel_height=0.8
            )
        else:
            # YOLOE 탐지 못함 → 규칙기반도 실패
            rule_yolo = abs_calc.calculate_absolute_volume(
                label="UNKNOWN", type_name=None,
                rel_width=1.0, rel_depth=1.5, rel_height=0.8
            )
        rule_yolo_dims = sorted([rule_yolo.width_mm, rule_yolo.depth_mm, rule_yolo.height_mm])
        are_rule_yolo = aspect_ratio_error(compute_aspect_ratio(rule_yolo_dims), gt_ratio)
        if rule_yolo_dims == sorted([100.0, 100.0, 100.0]):
            stats["rule_fallback_100mm"] += 1
        results["rule_yolo_label"].append(are_rule_yolo)

        # === Rule-based with GT label (이전 실험, 참고용) ===
        PIX3D_LABEL_MAP = {
            "bed": "BED", "sofa": "SOFA", "chair": "CHAIR_STOOL",
            "desk": "DESK", "table": "DINING_TABLE",
            "bookcase": "BOOKSHELF", "wardrobe": "WARDROBE",
        }
        rule_gt = abs_calc.calculate_absolute_volume(
            label=PIX3D_LABEL_MAP[gt_cat], type_name=None,
            rel_width=1.0, rel_depth=1.5, rel_height=0.8
        )
        rule_gt_dims = sorted([rule_gt.width_mm, rule_gt.depth_mm, rule_gt.height_mm])
        are_rule_gt = aspect_ratio_error(compute_aspect_ratio(rule_gt_dims), gt_ratio)
        results["rule_gt_label"].append(are_rule_gt)

        rows.append({
            "image": os.path.basename(s["img_path"]),
            "gt_category": gt_cat,
            "yolo_label": yolo_label or "NO_DETECTION",
            "label_correct": yolo_label is not None and any(
                exp in yolo_label.lower() for exp in EXPECTED_LABELS.get(gt_cat, [])
            ),
            "boxer_nd_are": are_nd, "boxer_nd_conf": conf_nd,
            "boxer_mg_are": are_mg, "boxer_mg_conf": conf_mg,
            "rule_yolo_are": are_rule_yolo,
            "rule_gt_are": are_rule_gt,
            "rule_yolo_dims": rule_yolo_dims,
        })

        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(samples)}...")

    # ================================================================
    # Results
    # ================================================================
    n = stats["total"]
    print(f"\n{'='*70}")
    print(f"  결과 ({n} samples)")
    print(f"{'='*70}")

    # YOLOE 탐지 통계
    print(f"\n  --- YOLOE 탐지 통계 ---")
    print(f"  매칭 성공 (IoU≥0.3): {stats['yolo_matched']}/{n} ({stats['yolo_matched']*100//n}%)")
    print(f"  매칭 실패:           {stats['yolo_no_match']}/{n} ({stats['yolo_no_match']*100//n}%)")
    print(f"  라벨 정확:           {stats['yolo_correct_label']}/{stats['yolo_matched']} "
          f"({stats['yolo_correct_label']*100//max(stats['yolo_matched'],1)}%)")
    print(f"  라벨 오류:           {stats['yolo_wrong_label']}/{stats['yolo_matched']} "
          f"({stats['yolo_wrong_label']*100//max(stats['yolo_matched'],1)}%)")
    print(f"  Rule 100mm fallback: {stats['rule_fallback_100mm']}/{n} ({stats['rule_fallback_100mm']*100//n}%)")

    # 전체 ARE
    labels_map = {
        "boxer_no_depth": "Boxer (no depth)",
        "boxer_moge": "Boxer + MoGe",
        "rule_yolo_label": "Rule (YOLOE label)",
        "rule_gt_label": "Rule (GT label, 참고)",
    }

    print(f"\n  --- 전체 Aspect Ratio Error ---")
    print(f"  {'Method':<28} | {'Mean ARE':<10} | {'Median':<10} | {'<0.10':<8}")
    print(f"  {'-'*28}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    for cond in conditions:
        arr = np.array(results[cond])
        print(f"  {labels_map[cond]:<28} | {arr.mean():>8.4f}  | {np.median(arr):>8.4f}  | "
              f"{(arr<0.10).sum()/len(arr)*100:>5.1f}%")

    # 카테고리별
    print(f"\n  --- 카테고리별 ARE ---")
    print(f"  {'Category':<12} | {'Boxer+MoGe':<12} | {'Rule(YOLOE)':<12} | {'Rule(GT)':<12} | {'YOLOE Label Acc':<16}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*16}")

    for cat in ['bed', 'sofa', 'chair', 'desk', 'table', 'bookcase', 'wardrobe']:
        cat_rows = [r for r in rows if r["gt_category"] == cat]
        if not cat_rows:
            continue
        mg = np.mean([r["boxer_mg_are"] for r in cat_rows])
        ry = np.mean([r["rule_yolo_are"] for r in cat_rows])
        rg = np.mean([r["rule_gt_are"] for r in cat_rows])
        correct = sum(1 for r in cat_rows if r["label_correct"])
        total = len(cat_rows)
        print(f"  {cat:<12} | {mg:>10.4f}  | {ry:>10.4f}  | {rg:>10.4f}  | {correct}/{total} ({correct*100//total}%)")

    # YOLOE 라벨 오류 → 규칙기반 실패 사례
    wrong_label_rows = [r for r in rows if not r["label_correct"] and r["yolo_label"] != "NO_DETECTION"]
    if wrong_label_rows:
        print(f"\n  --- YOLOE 라벨 오류 시 비교 (N={len(wrong_label_rows)}) ---")
        mg_errs = [r["boxer_mg_are"] for r in wrong_label_rows]
        ry_errs = [r["rule_yolo_are"] for r in wrong_label_rows]
        print(f"  Boxer+MoGe: ARE={np.mean(mg_errs):.4f}")
        print(f"  Rule(YOLOE): ARE={np.mean(ry_errs):.4f}")
        print(f"  → 라벨 오류 시 {'Boxer가 유리' if np.mean(mg_errs) < np.mean(ry_errs) else '규칙기반이 유리'}")

    # YOLOE 라벨 정확 시 비교
    correct_rows = [r for r in rows if r["label_correct"]]
    if correct_rows:
        print(f"\n  --- YOLOE 라벨 정확 시 비교 (N={len(correct_rows)}) ---")
        mg_errs = [r["boxer_mg_are"] for r in correct_rows]
        ry_errs = [r["rule_yolo_are"] for r in correct_rows]
        print(f"  Boxer+MoGe: ARE={np.mean(mg_errs):.4f}")
        print(f"  Rule(YOLOE): ARE={np.mean(ry_errs):.4f}")

    # CSV
    csv_path = out_dir / "realistic_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "gt_category", "yolo_label", "label_correct",
                     "boxer_nd_ARE", "boxer_mg_ARE", "rule_yolo_ARE", "rule_gt_ARE"])
        for r in rows:
            w.writerow([r["image"], r["gt_category"], r["yolo_label"], r["label_correct"],
                         f"{r['boxer_nd_are']:.4f}", f"{r['boxer_mg_are']:.4f}",
                         f"{r['rule_yolo_are']:.4f}", f"{r['rule_gt_are']:.4f}"])
    print(f"\n  CSV: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run()
