"""
Depth Ablation 실험: Boxer + 다양한 depth 입력 비교

5가지 조건:
  A. Boxer (depth 없음)
  B. Boxer + MoGe (relative depth, raw)
  C. Boxer + MoGe (scale/shift 보정)
  D. Boxer + DepthAnything V2 Metric (absolute depth)
  E. Rule-based (baseline)

MoGe scale/shift 보정:
  MoGe는 relative depth → bbox 영역의 평균 depth를 기준으로
  "이미지에서 bbox가 차지하는 비율 + focal_length"로 물체 거리를 추정하여 보정.
  d_metric = d_moge * (estimated_distance / d_moge_bbox_mean)

Usage:
    conda run -n sam3d-objects python /path/to/experiment_depth_ablation.py
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

from ai.processors import AbsoluteVolumeCalculator
from moge.model.v1 import MoGeModel
from depth_anything_v2.dpt import DepthAnythingV2

PIX3D_ROOT = ROOT / "experiments" / "seed_variance" / "data" / "pix3d"
DA_CKPT = str(Path.home() / ".cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Metric-Hypersim-Small/snapshots/3bc65d4e14a6786a61acec16453c50e12bf5f338/depth_anything_v2_metric_hypersim_vits.pth")

PIX3D_TO_LABEL = {
    "bed": "BED", "sofa": "SOFA", "chair": "CHAIR_STOOL",
    "desk": "DESK", "table": "DINING_TABLE",
    "bookcase": "BOOKSHELF", "wardrobe": "WARDROBE",
}


def compute_aspect_ratio(dims):
    s = sorted(dims)
    return [s[0] / s[2], s[1] / s[2], 1.0] if s[2] > 0 else [0, 0, 0]


def aspect_ratio_error(pred, gt):
    return float(np.mean([abs(p - g) for p, g in zip(pred, gt)])) if pred and gt else 1.0


def load_samples(max_per_cat=30):
    with open(PIX3D_ROOT / "pix3d.json") as f:
        data = json.load(f)
    samples, cat_count, seen = [], defaultdict(int), {}
    for d in data:
        cat = d["category"]
        if cat not in PIX3D_TO_LABEL or d["truncated"] or d["occluded"]:
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
            "category": cat, "bbox": d["bbox"],
            "img_size": d["img_size"],
            "focal_length": d["focal_length"],
            "gt_ratio": compute_aspect_ratio(seen[mp]),
        })
        cat_count[cat] += 1
    return samples


# ================================================================
# MoGe Scale/Shift 보정
# ================================================================

def correct_moge_scale(
    moge_depth: np.ndarray,
    bbox: List[int],
    img_w: int, img_h: int,
    focal_length_mm: float,
) -> np.ndarray:
    """
    MoGe relative depth를 metric scale로 보정.

    원리:
    1. bbox의 이미지 비율로 물체의 실제 크기를 추정 (가구 평균 1.0m 가정)
    2. focal_length + bbox_size로 물체까지 거리 추정:
       distance = (assumed_size * f_px) / bbox_size_px
    3. MoGe depth의 bbox 영역 중앙값으로 스케일 팩터 계산:
       scale = estimated_distance / moge_bbox_median
    4. 전체 depth에 scale 적용

    Args:
        moge_depth: MoGe 출력 depth map (H, W)
        bbox: [x1, y1, x2, y2]
        img_w, img_h: 원본 이미지 크기
        focal_length_mm: Pix3D focal length (mm)

    Returns:
        보정된 metric depth map (H, W)
    """
    x1, y1, x2, y2 = bbox
    dh, dw = moge_depth.shape

    # bbox를 depth map 좌표로 변환
    sx, sy = dw / img_w, dh / img_h
    bx1 = max(0, int(x1 * sx))
    by1 = max(0, int(y1 * sy))
    bx2 = min(dw, int(x2 * sx))
    by2 = min(dh, int(y2 * sy))

    # bbox 영역의 depth 중앙값
    bbox_depth = moge_depth[by1:by2, bx1:bx2]
    if bbox_depth.size == 0:
        return moge_depth
    moge_median = np.median(bbox_depth[bbox_depth > 0])
    if moge_median <= 0 or np.isnan(moge_median):
        return moge_depth

    # bbox 크기 (pixel)
    bbox_size_px = max(x2 - x1, y2 - y1)
    if bbox_size_px <= 0:
        return moge_depth

    # pixel focal length (36mm 센서 가정)
    f_px = focal_length_mm * max(img_w, img_h) / 36.0

    # 가구 평균 크기 가정 (1.0m) → 거리 추정
    assumed_object_size_m = 1.0
    estimated_distance = (assumed_object_size_m * f_px) / bbox_size_px

    # 스케일 팩터
    scale = estimated_distance / moge_median

    return moge_depth * scale


# ================================================================
# Boxer Predictor
# ================================================================

class BoxerPredictor:
    def __init__(self, device="cuda"):
        self.device = device
        ckpt = str(ROOT / "boxer" / "ckpts" / "boxernet_hw960in4x6d768-wssxpf9p.ckpt")
        self.model = BoxerNet.load_from_checkpoint(ckpt, device=device)
        self.model.eval()
        self._R_yz = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        self._t_zero = np.zeros(3, dtype=np.float32)

    @torch.no_grad()
    def predict(self, image: Image.Image, bbox, focal_length_mm,
                depth_map: Optional[np.ndarray] = None) -> Optional[Tuple]:
        W, H = image.size
        tgt = 960
        sx, sy = tgt / W, tgt / H
        img_r = image.resize((tgt, tgt), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img_r.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_t = img_t.to(self.device)

        f_px = focal_length_mm * max(W, H) / 36.0
        f_s = f_px * (tgt / max(W, H))
        cam = get_pinhole_camera([f_s, f_s, tgt / 2, tgt / 2], tgt, tgt).to(self.device)

        R, t = self._R_yz.copy(), self._t_zero.copy()
        pose = PoseTW.from_Rt(torch.from_numpy(R), torch.from_numpy(t)).to(self.device)

        if depth_map is not None:
            depth_resized = cv2.resize(depth_map.astype(np.float32), (tgt, tgt), interpolation=cv2.INTER_NEAREST)
            sdp = BaseLoader.sdp_from_depth(depth_resized, f_s, f_s, tgt / 2, tgt / 2, R, t, 10000).to(self.device)
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


# ================================================================
# Main
# ================================================================

def run():
    out_dir = ROOT / "experiments" / "boxer_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Depth Ablation: 5가지 depth 조건 비교")
    print("=" * 70)

    samples = load_samples(max_per_cat=30)
    print(f"\n  {len(samples)} samples loaded")

    # Models
    print("\n  Loading models...")
    boxer = BoxerPredictor()
    print("  [OK] BoxerNet")

    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").cuda().eval()
    print("  [OK] MoGe")

    da = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=20)
    da.load_state_dict(torch.load(DA_CKPT, map_location='cpu'))
    da = da.cuda().eval()
    print("  [OK] DepthAnything V2 Metric")

    abs_calc = AbsoluteVolumeCalculator()
    print(f"  GPU: {torch.cuda.memory_allocated()/1e6:.0f}MB\n")

    conditions = ["A_no_depth", "B_moge_raw", "C_moge_corrected", "D_da_metric", "E_rule_based"]
    results = {c: [] for c in conditions}
    rows = []

    for i, s in enumerate(samples):
        try:
            img_pil = Image.open(s["img_path"]).convert("RGB")
        except Exception:
            continue

        gt_ratio = s["gt_ratio"]
        W, H = s["img_size"]
        fl = s["focal_length"]

        # --- MoGe depth ---
        img_moge = img_pil.resize((512, 512))
        img_t = torch.from_numpy(np.array(img_moge)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            moge_out = moge.infer(img_t.cuda(), force_projection=False)
        moge_depth = moge_out["depth"].cpu().numpy()

        # --- MoGe corrected ---
        moge_corrected = correct_moge_scale(moge_depth, s["bbox"], W, H, fl)

        # --- DA V2 Metric depth ---
        img_cv = cv2.imread(s["img_path"])
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            da_depth = da.infer_image(img_cv)

        # --- 5 conditions ---
        row = {"image": os.path.basename(s["img_path"]), "category": s["category"]}

        for cond, depth_in in [
            ("A_no_depth", None),
            ("B_moge_raw", moge_depth),
            ("C_moge_corrected", moge_corrected),
            ("D_da_metric", da_depth),
        ]:
            out = boxer.predict(img_pil, s["bbox"], fl, depth_map=depth_in)
            if out:
                w, d, h, vol, conf = out
                dims = sorted([w, d, h])
                ratio = compute_aspect_ratio(dims)
                are = aspect_ratio_error(ratio, gt_ratio)
                results[cond].append({"are": are, "conf": conf, "dims": dims})
                row[f"{cond}_are"] = are
                row[f"{cond}_conf"] = conf
                row[f"{cond}_dims"] = dims
            else:
                results[cond].append({"are": 1.0, "conf": 0, "dims": [0, 0, 0]})
                row[f"{cond}_are"] = 1.0
                row[f"{cond}_conf"] = 0

        # E: Rule-based
        label = PIX3D_TO_LABEL[s["category"]]
        rule = abs_calc.calculate_absolute_volume(label, None, 1.0, 1.5, 0.8)
        rule_dims = sorted([rule.width_mm, rule.depth_mm, rule.height_mm])
        rule_are = aspect_ratio_error(compute_aspect_ratio(rule_dims), gt_ratio)
        results["E_rule_based"].append({"are": rule_are, "conf": 1.0, "dims": rule_dims})
        row["E_rule_based_are"] = rule_are

        rows.append(row)
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(samples)}...")

    # ================================================================
    # Results
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  결과 ({len(rows)} samples)")
    print(f"{'='*70}")

    labels = {
        "A_no_depth": "A. Boxer (no depth)",
        "B_moge_raw": "B. Boxer + MoGe raw",
        "C_moge_corrected": "C. Boxer + MoGe corrected",
        "D_da_metric": "D. Boxer + DA V2 Metric",
        "E_rule_based": "E. Rule-based",
    }

    print(f"\n  {'Method':<28} | {'Mean ARE':<10} | {'Median':<10} | {'<0.05':<7} | {'<0.10':<7} | {'Conf':<8}")
    print(f"  {'-'*28}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")

    for cond in conditions:
        ares = np.array([r["are"] for r in results[cond]])
        confs = [r["conf"] for r in results[cond] if r["conf"] > 0]
        u5 = (ares < 0.05).sum() / len(ares) * 100
        u10 = (ares < 0.10).sum() / len(ares) * 100
        c_mean = np.mean(confs) if confs else 0
        print(f"  {labels[cond]:<28} | {ares.mean():>8.4f}  | {np.median(ares):>8.4f}  | {u5:>5.1f}% | {u10:>5.1f}% | {c_mean:>6.3f}")

    # 카테고리별 best method
    print(f"\n  --- 카테고리별 Best Method ---")
    print(f"  {'Category':<12} | ", end="")
    for cond in conditions:
        print(f"{cond.split('_')[0]:<8} | ", end="")
    print("Best")
    print(f"  {'-'*12}-+-" + "-+-".join(["-" * 8] * 5) + "-+--------")

    cat_wins = defaultdict(int)
    for cat in ['bed', 'sofa', 'chair', 'desk', 'table', 'bookcase', 'wardrobe']:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        print(f"  {cat:<12} | ", end="")
        best_cond = None
        best_are = 999
        for cond in conditions:
            key = f"{cond}_are"
            ares = [r[key] for r in cat_rows]
            mean = np.mean(ares)
            if mean < best_are:
                best_are = mean
                best_cond = cond
            print(f"{mean:>6.4f}   | ", end="")
        short = best_cond.split("_")[0]
        cat_wins[best_cond] += 1
        print(short)

    print(f"\n  카테고리 승리:")
    for cond in conditions:
        print(f"    {labels[cond]}: {cat_wins.get(cond, 0)}")

    # CSV
    csv_path = out_dir / "depth_ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["image", "category"]
        for cond in conditions:
            header += [f"{cond}_ARE", f"{cond}_conf"]
        w.writerow(header)
        for r in rows:
            row_data = [r["image"], r["category"]]
            for cond in conditions:
                row_data += [f"{r.get(f'{cond}_are', 1.0):.4f}", f"{r.get(f'{cond}_conf', 0):.3f}"]
            w.writerow(row_data)
    print(f"\n  CSV: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run()
