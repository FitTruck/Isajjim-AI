"""
성능 개선 실험: 3가지 개선 적용

개선점:
  1. MoGe intrinsics → Boxer intrinsics (heuristic FOV 대체)
  2. Aspect ratio 보존 리사이즈 (960x960 squash → 패딩)
  3. 카테고리별 하이브리드 전략 (Boxer 강점 카테고리 + Rule 강점 카테고리)

비교:
  A. Boxer + MoGe (기존, heuristic intrinsics + squash)
  B. Boxer + MoGe (MoGe intrinsics + AR 보존)
  C. 하이브리드 (카테고리별 최적 선택)
  D. Rule-based (YOLOE label)

Usage:
    conda run -n sam3d-objects python /path/to/experiment_improved.py
"""

import csv, json, os, sys, time
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

from boxernet.boxernet import BoxerNet
from loaders.base_loader import BaseLoader
from utils.tw.camera import get_pinhole_camera
from utils.tw.pose import PoseTW

os.chdir(str(ROOT))

from ai.processors import YoloDetector, AbsoluteVolumeCalculator
from moge.model.v1 import MoGeModel

PIX3D_ROOT = ROOT / "experiments" / "seed_variance" / "data" / "pix3d"


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0


def compute_aspect_ratio(dims):
    s = sorted(dims)
    return [s[0]/s[2], s[1]/s[2], 1.0] if s[2] > 0 else [0, 0, 0]


def aspect_ratio_error(pred, gt):
    return float(np.mean([abs(p-g) for p, g in zip(pred, gt)])) if pred and gt else 1.0


def load_samples(max_per_cat=30):
    with open(PIX3D_ROOT / "pix3d.json") as f:
        data = json.load(f)
    cats = {"bed", "sofa", "chair", "desk", "table", "bookcase", "wardrobe"}
    samples, cc, seen = [], defaultdict(int), {}
    for d in data:
        cat = d["category"]
        if cat not in cats or d["truncated"] or d["occluded"] or cc[cat] >= max_per_cat:
            continue
        mp = d["model"]
        if mp not in seen:
            try:
                mesh = trimesh.load(str(PIX3D_ROOT / mp), force="mesh")
                seen[mp] = sorted((mesh.bounds[1] - mesh.bounds[0]).tolist())
            except:
                continue
        samples.append({
            "img_path": str(PIX3D_ROOT / d["img"]), "category": cat,
            "gt_bbox": d["bbox"], "img_size": d["img_size"],
            "focal_length": d["focal_length"],
            "gt_ratio": compute_aspect_ratio(seen[mp]),
        })
        cc[cat] += 1
    return samples


class ImprovedBoxerPredictor:
    """개선된 Boxer 추론기"""

    def __init__(self, device="cuda"):
        self.device = device
        ckpt = str(ROOT / "boxer" / "ckpts" / "boxernet_hw960in4x6d768-wssxpf9p.ckpt")
        self.model = BoxerNet.load_from_checkpoint(ckpt, device=device)
        self.model.eval()
        self._R_yz = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        self._t = np.zeros(3, dtype=np.float32)

    @torch.no_grad()
    def predict(self, image, bbox, depth_map=None,
                use_moge_intrinsics=False, moge_K=None,
                preserve_ar=False) -> Optional[Tuple]:
        """
        Args:
            use_moge_intrinsics: True이면 MoGe K 매트릭스 사용
            moge_K: MoGe가 추정한 3x3 intrinsics (정규화)
            preserve_ar: True이면 aspect ratio 보존 패딩
        """
        W, H = image.size
        tgt = 960

        if preserve_ar:
            # 비율 보존: 긴 쪽을 960에 맞추고 짧은 쪽은 패딩
            scale = tgt / max(W, H)
            new_w, new_h = int(W * scale), int(H * scale)
            img_resized = image.resize((new_w, new_h), Image.BILINEAR)
            # 검은색 패딩
            img_padded = Image.new("RGB", (tgt, tgt), (0, 0, 0))
            pad_x, pad_y = (tgt - new_w) // 2, (tgt - new_h) // 2
            img_padded.paste(img_resized, (pad_x, pad_y))
            sx = scale
            sy = scale
            cx_offset = pad_x
            cy_offset = pad_y
        else:
            img_padded = image.resize((tgt, tgt), Image.BILINEAR)
            sx, sy = tgt / W, tgt / H
            cx_offset, cy_offset = 0, 0

        img_t = torch.from_numpy(np.array(img_padded.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_t = img_t.to(self.device)

        # Intrinsics
        if use_moge_intrinsics and moge_K is not None:
            # MoGe intrinsics: 정규화된 값 → pixel 변환
            # MoGe는 이미지 크기에 대해 정규화 (0~1 범위)
            # fx_moge * orig_W = pixel focal length
            moge_K_np = moge_K.cpu().numpy() if torch.is_tensor(moge_K) else moge_K
            fx_orig = float(moge_K_np[0, 0]) * W
            fy_orig = float(moge_K_np[1, 1]) * H

            if preserve_ar:
                fx_s = fx_orig * scale
                fy_s = fy_orig * scale
                cx_s = float(moge_K_np[0, 2]) * W * scale + cx_offset
                cy_s = float(moge_K_np[1, 2]) * H * scale + cy_offset
            else:
                fx_s = fx_orig * (tgt / W)
                fy_s = fy_orig * (tgt / H)
                cx_s = float(moge_K_np[0, 2]) * tgt
                cy_s = float(moge_K_np[1, 2]) * tgt
        else:
            f_heuristic = max(W, H) * 0.8
            if preserve_ar:
                fx_s = fy_s = f_heuristic * scale
                cx_s = W / 2 * scale + cx_offset
                cy_s = H / 2 * scale + cy_offset
            else:
                fx_s = f_heuristic * (tgt / max(W, H))
                fy_s = fx_s
                cx_s = tgt / 2
                cy_s = tgt / 2

        cam = get_pinhole_camera([fx_s, fy_s, cx_s, cy_s], tgt, tgt).to(self.device)

        R, t = self._R_yz.copy(), self._t.copy()
        pose = PoseTW.from_Rt(torch.from_numpy(R), torch.from_numpy(t)).to(self.device)

        # SDP
        if depth_map is not None:
            if preserve_ar:
                dr = cv2.resize(depth_map.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                dr_padded = np.zeros((tgt, tgt), dtype=np.float32)
                dr_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = dr
            else:
                dr_padded = cv2.resize(depth_map.astype(np.float32), (tgt, tgt), interpolation=cv2.INTER_NEAREST)
            sdp = BaseLoader.sdp_from_depth(dr_padded, fx_s, fy_s, cx_s, cy_s, R, t, 10000).to(self.device)
        else:
            sdp = torch.zeros(0, 3, device=self.device)

        # Bbox
        x1, y1, x2, y2 = bbox
        if preserve_ar:
            bx1, bx2 = x1 * scale + cx_offset, x2 * scale + cx_offset
            by1, by2 = y1 * scale + cy_offset, y2 * scale + cy_offset
        else:
            bx1, bx2 = x1 * sx, x2 * sx
            by1, by2 = y1 * sy, y2 * sy
        bb2d = torch.tensor([[[bx1, bx2, by1, by2]]], dtype=torch.float32, device=self.device)

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
        return (dims[2]*1000, dims[0]*1000, dims[1]*1000,
                float(obb.bb3_volumes.squeeze().abs()), float(obb.prob.squeeze()))


# 하이브리드 전략: 카테고리별 최적 선택
# 이전 실험 결과 기반
BOXER_PREFERRED_CATS = {"desk", "table", "bookcase", "wardrobe"}  # Boxer 승률 70%+
RULE_PREFERRED_CATS = {"bed", "sofa", "chair"}  # Rule 승률 60%+


def run():
    out_dir = ROOT / "experiments" / "boxer_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  성능 개선 실험: MoGe intrinsics + AR 보존 + 하이브리드")
    print("=" * 70)

    samples = load_samples(max_per_cat=30)
    print(f"\n  {len(samples)} samples")

    print("\n  Loading models...")
    yolo = YoloDetector(device_id=0)
    boxer = ImprovedBoxerPredictor()
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").cuda().eval()
    abs_calc = AbsoluteVolumeCalculator()
    print(f"  GPU: {torch.cuda.memory_allocated()/1e6:.0f}MB\n")

    EXPECTED_LABELS = {
        "bed": ["bed"], "sofa": ["sofa", "couch"],
        "chair": ["chair", "stool", "armchair"],
        "desk": ["desk", "table"], "table": ["table", "desk", "dining"],
        "bookcase": ["bookcase", "shelf", "bookshelf", "cabinet"],
        "wardrobe": ["wardrobe", "cabinet", "closet"],
    }

    conditions = [
        "A_baseline",         # Boxer + MoGe (heuristic + squash) - 기존
        "B_moge_intrinsics",  # Boxer + MoGe (MoGe intrinsics + squash)
        "C_ar_preserve",      # Boxer + MoGe (heuristic + AR 보존)
        "D_both_improved",    # Boxer + MoGe (MoGe intrinsics + AR 보존)
        "E_hybrid",           # 하이브리드 (D + Rule 카테고리별)
        "F_rule",             # Rule-based (YOLOE label)
    ]
    results = {c: [] for c in conditions}
    rows = []

    for i, s in enumerate(samples):
        try:
            image = Image.open(s["img_path"]).convert("RGB")
        except:
            continue

        gt_ratio = s["gt_ratio"]
        W, H = s["img_size"]
        gt_cat = s["category"]

        # YOLOE
        yolo_res = yolo.detect_smart(image, return_masks=False)
        yolo_label, yolo_bbox = None, s["gt_bbox"]
        if yolo_res and len(yolo_res["boxes"]) > 0:
            best_iou, best_idx = 0, -1
            for j, box in enumerate(yolo_res["boxes"]):
                iou = compute_iou(s["gt_bbox"], [int(x) for x in box])
                if iou > best_iou:
                    best_iou, best_idx = iou, j
            if best_iou >= 0.3:
                yolo_label = yolo_res["labels"][best_idx]
                yolo_bbox = [int(x) for x in yolo_res["boxes"][best_idx]]

        # MoGe depth + intrinsics
        img_moge = image.resize((512, 512))
        img_t = torch.from_numpy(np.array(img_moge)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            moge_out = moge.infer(img_t.cuda(), force_projection=False)
        moge_depth = moge_out["depth"].cpu().numpy()
        moge_K = moge_out["intrinsics"]  # 3x3, 정규화

        # A: 기존 (heuristic intrinsics + squash)
        out_a = boxer.predict(image, yolo_bbox, depth_map=moge_depth,
                              use_moge_intrinsics=False, preserve_ar=False)

        # B: MoGe intrinsics + squash
        out_b = boxer.predict(image, yolo_bbox, depth_map=moge_depth,
                              use_moge_intrinsics=True, moge_K=moge_K, preserve_ar=False)

        # C: heuristic + AR 보존
        out_c = boxer.predict(image, yolo_bbox, depth_map=moge_depth,
                              use_moge_intrinsics=False, preserve_ar=True)

        # D: MoGe intrinsics + AR 보존 (둘 다 개선)
        out_d = boxer.predict(image, yolo_bbox, depth_map=moge_depth,
                              use_moge_intrinsics=True, moge_K=moge_K, preserve_ar=True)

        # F: Rule-based
        label_for_rule = (yolo_label or "UNKNOWN").upper()
        rule = abs_calc.calculate_absolute_volume(label_for_rule, None, 1.0, 1.5, 0.8)
        rule_dims = sorted([rule.width_mm, rule.depth_mm, rule.height_mm])
        rule_are = aspect_ratio_error(compute_aspect_ratio(rule_dims), gt_ratio)

        row = {"image": os.path.basename(s["img_path"]), "category": gt_cat}

        for cond, out in [("A_baseline", out_a), ("B_moge_intrinsics", out_b),
                           ("C_ar_preserve", out_c), ("D_both_improved", out_d)]:
            if out:
                dims = sorted([out[0], out[1], out[2]])
                are = aspect_ratio_error(compute_aspect_ratio(dims), gt_ratio)
                conf = out[4]
            else:
                are, conf = 1.0, 0
            results[cond].append(are)
            row[f"{cond}_are"] = are
            row[f"{cond}_conf"] = conf

        results["F_rule"].append(rule_are)
        row["F_rule_are"] = rule_are

        # E: 하이브리드 (D가 Boxer 대표)
        if gt_cat in BOXER_PREFERRED_CATS:
            hybrid_are = row["D_both_improved_are"]
        else:
            hybrid_are = rule_are
        results["E_hybrid"].append(hybrid_are)
        row["E_hybrid_are"] = hybrid_are

        rows.append(row)
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(samples)}...")

    # Results
    print(f"\n{'='*70}")
    print(f"  결과 ({len(rows)} samples)")
    print(f"{'='*70}")

    labels = {
        "A_baseline": "A. 기존 (heuristic+squash)",
        "B_moge_intrinsics": "B. + MoGe intrinsics",
        "C_ar_preserve": "C. + AR 보존 패딩",
        "D_both_improved": "D. + 둘 다 적용",
        "E_hybrid": "E. 하이브리드 (D+Rule)",
        "F_rule": "F. Rule (YOLOE label)",
    }

    print(f"\n  {'Method':<30} | {'Mean ARE':<10} | {'Median':<10} | {'<0.10':<8} | {'vs A':<8}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    baseline_are = np.mean(results["A_baseline"])
    for cond in conditions:
        arr = np.array(results[cond])
        improvement = (baseline_are - arr.mean()) / baseline_are * 100
        print(f"  {labels[cond]:<30} | {arr.mean():>8.4f}  | {np.median(arr):>8.4f}  | "
              f"{(arr<0.10).sum()/len(arr)*100:>5.1f}% | {improvement:>+5.1f}%")

    # 카테고리별
    print(f"\n  --- 카테고리별 ARE ---")
    print(f"  {'Cat':<10} | {'A.기존':<8} | {'B.Intrin':<8} | {'C.AR':<8} | {'D.Both':<8} | {'E.Hybrid':<8} | {'F.Rule':<8}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for cat in ['bed', 'sofa', 'chair', 'desk', 'table', 'bookcase', 'wardrobe']:
        cr = [r for r in rows if r["category"] == cat]
        if not cr:
            continue
        vals = []
        for cond in conditions:
            key = f"{cond}_are"
            vals.append(np.mean([r[key] for r in cr]))
        best_idx = np.argmin(vals)
        line = f"  {cat:<10}"
        for j, v in enumerate(vals):
            marker = " *" if j == best_idx else "  "
            line += f" | {v:>5.4f}{marker}"
        print(line)

    # CSV
    csv_path = out_dir / "improved_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["image", "category"]
        for c in conditions:
            header.append(f"{c}_ARE")
        w.writerow(header)
        for r in rows:
            row_data = [r["image"], r["category"]]
            for c in conditions:
                row_data.append(f"{r[f'{c}_are']:.4f}")
            w.writerow(row_data)
    print(f"\n  CSV: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run()
