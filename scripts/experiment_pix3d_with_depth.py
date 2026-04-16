"""
Pix3D 실험: MoGe depth 유무에 따른 Boxer 성능 비교

비교 대상:
  1. Boxer (depth 없음)     — 이전 실험과 동일
  2. Boxer + MoGe depth     — MoGe pointmap → SDP → Boxer
  3. Rule-based (baseline)  — 기존 규칙기반

평가 지표: GT 3D 모델 비율 대비 Aspect Ratio Error (ARE)

Usage:
    conda run -n sam3d-objects python /path/to/experiment_pix3d_with_depth.py
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# Boxer imports (CWD = boxer/)
os.chdir(str(ROOT / "boxer"))
sys.path.insert(0, str(ROOT / "boxer"))
sys.path.insert(1, str(ROOT))

from boxernet.boxernet import BoxerNet
from loaders.base_loader import BaseLoader
from utils.tw.camera import get_pinhole_camera
from utils.tw.pose import PoseTW

os.chdir(str(ROOT))

# AI & MoGe imports
from ai.processors import AbsoluteVolumeCalculator
from moge.model.v1 import MoGeModel

PIX3D_ROOT = ROOT / "experiments" / "seed_variance" / "data" / "pix3d"

PIX3D_TO_LABEL = {
    "bed": "BED", "sofa": "SOFA", "chair": "CHAIR_STOOL",
    "desk": "DESK", "table": "DINING_TABLE",
    "bookcase": "BOOKSHELF", "wardrobe": "WARDROBE",
}


def compute_aspect_ratio(dims):
    s = sorted(dims)
    if s[2] == 0:
        return [0, 0, 0]
    return [s[0] / s[2], s[1] / s[2], 1.0]


def aspect_ratio_error(pred, gt):
    if not pred or not gt:
        return 1.0
    return float(np.mean([abs(p - g) for p, g in zip(pred, gt)]))


def load_pix3d_samples(max_per_cat=30):
    with open(PIX3D_ROOT / "pix3d.json") as f:
        data = json.load(f)

    samples = []
    cat_count = defaultdict(int)
    seen_models = {}

    for d in data:
        cat = d["category"]
        if cat not in PIX3D_TO_LABEL:
            continue
        if d["truncated"] or d["occluded"]:
            continue
        if cat_count[cat] >= max_per_cat:
            continue

        model_path = d["model"]
        if model_path not in seen_models:
            try:
                mesh = trimesh.load(str(PIX3D_ROOT / model_path), force="mesh")
                aabb = mesh.bounds[1] - mesh.bounds[0]
                seen_models[model_path] = sorted(aabb.tolist())
            except Exception:
                continue

        gt_dims = seen_models[model_path]
        samples.append({
            "img_path": str(PIX3D_ROOT / d["img"]),
            "category": cat,
            "bbox": d["bbox"],
            "img_size": d["img_size"],
            "focal_length": d["focal_length"],
            "rot_mat": d["rot_mat"],
            "trans_mat": d["trans_mat"],
            "gt_ratio": compute_aspect_ratio(gt_dims),
            "gt_dims": gt_dims,
        })
        cat_count[cat] += 1

    return samples


class BoxerWithDepth:
    """Boxer 추론기 — MoGe depth 옵션 지원"""

    def __init__(self, device="cuda"):
        self.device = device
        ckpt = str(ROOT / "boxer" / "ckpts" / "boxernet_hw960in4x6d768-wssxpf9p.ckpt")
        print("[Boxer] Loading BoxerNet...")
        self.model = BoxerNet.load_from_checkpoint(ckpt, device=device)
        self.model.eval()

        self._R_yz = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        self._t_zero = np.zeros(3, dtype=np.float32)

    @torch.no_grad()
    def predict(self, image: Image.Image, bbox: List[int],
                focal_length: Optional[float] = None,
                rot_mat=None, trans_mat=None,
                moge_output: Optional[dict] = None) -> Optional[Tuple]:
        """
        Args:
            focal_length: Pix3D focal length (mm)
            rot_mat: Pix3D 3x3 rotation (object→camera), gravity 정보 포함
            trans_mat: Pix3D 3D translation (camera coords)
            moge_output: MoGe 추론 결과 {'depth': (H,W), 'intrinsics': (3,3)}
        """
        W_orig, H_orig = image.size
        target = 960
        sx, sy = target / W_orig, target / H_orig
        image_r = image.resize((target, target), Image.BILINEAR)

        img_np = np.array(image_r.convert("RGB"))
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_t = img_t.to(self.device)

        # Intrinsics: Pix3D focal_length(mm) → pixel focal length
        if focal_length is not None:
            f_px = focal_length * max(W_orig, H_orig) / 36.0
            f_scaled = f_px * (target / max(W_orig, H_orig))
        else:
            f_scaled = target * 0.8
        fx = fy = float(f_scaled)
        cx_val = target / 2.0
        cy_val = target / 2.0

        cam = get_pinhole_camera(
            params=[fx, fy, cx_val, cy_val], width=target, height=target
        ).to(self.device)

        # Pose: Pix3D GT rot_mat/trans_mat 사용 (있으면)
        if rot_mat is not None and trans_mat is not None:
            # Pix3D: R은 object→camera 회전, t는 camera 좌표계 translation
            # Boxer T_world_rig: world(=object) → camera 변환의 역 = camera → world
            R_obj2cam = np.array(rot_mat, dtype=np.float32)
            t_cam = np.array(trans_mat, dtype=np.float32)
            # camera → world (= inverse)
            R_wc = R_obj2cam.T  # R^T
            t_wc = -R_wc @ t_cam
            pose = PoseTW.from_Rt(
                torch.from_numpy(R_wc), torch.from_numpy(t_wc)
            ).to(self.device)
        else:
            R_wc = self._R_yz.copy()
            t_wc = self._t_zero.copy()
            pose = PoseTW.from_Rt(
                torch.from_numpy(R_wc), torch.from_numpy(t_wc)
            ).to(self.device)

        # SDP: MoGe depth → semi-dense points
        if moge_output is not None:
            import cv2
            depth_np = moge_output["depth"].cpu().numpy().astype(np.float32)
            depth_resized = cv2.resize(depth_np, (target, target), interpolation=cv2.INTER_NEAREST)
            sdp_w = BaseLoader.sdp_from_depth(
                depth_resized,
                fx=fx, fy=fy, cx=cx_val, cy=cy_val,
                R_wc=R_wc if rot_mat is not None else self._R_yz,
                t_wc=t_wc if trans_mat is not None else self._t_zero,
                num_samples=10000,
            ).to(self.device)
        else:
            sdp_w = torch.zeros(0, 3, device=self.device)

        # Bbox
        x1, y1, x2, y2 = bbox
        bb2d = torch.tensor(
            [[[x1 * sx, x2 * sx, y1 * sy, y2 * sy]]],
            dtype=torch.float32, device=self.device
        )

        datum = {
            "img0": img_t,
            "cam0": cam,
            "T_world_rig0": pose,
            "rotated0": torch.tensor([False], device=self.device),
            "sdp_w": sdp_w,
            "bb2d": bb2d,
        }

        output = self.model(datum)
        obbs = output.get("obbs_pr_w")
        if obbs is None or obbs.shape[-2] == 0:
            return None

        obb = obbs[..., 0, :]
        diag = obb.bb3_diagonal.squeeze()
        dims = sorted([float(d.abs()) for d in diag])
        vol = float(obb.bb3_volumes.squeeze().abs())
        conf = float(obb.prob.squeeze())

        return (dims[2] * 1000, dims[0] * 1000, dims[1] * 1000, vol, conf)


def run_experiment():
    output_dir = ROOT / "experiments" / "boxer_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Pix3D: GT Pose + MoGe Depth 유무에 따른 Boxer 성능 비교")
    print("=" * 70)

    # 데이터 로드
    print("\n[1/5] Pix3D 데이터 로드...")
    samples = load_pix3d_samples(max_per_cat=30)
    cat_counts = defaultdict(int)
    for s in samples:
        cat_counts[s["category"]] += 1
    print(f"  총 {len(samples)}개 샘플: {dict(cat_counts)}")

    # 모델 로드
    print("\n[2/5] BoxerNet 로드...")
    boxer = BoxerWithDepth(device="cuda")

    print("\n[3/5] MoGe 로드...")
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").cuda().eval()
    print(f"  MoGe loaded, GPU: {torch.cuda.memory_allocated()/1e6:.0f}MB")

    abs_calc = AbsoluteVolumeCalculator()

    # 실험 실행
    print(f"\n[4/5] 실험 실행 ({len(samples)} samples)...")

    results = {
        "boxer_no_depth": defaultdict(list),
        "boxer_moge": defaultdict(list),
        "rule_based": defaultdict(list),
    }
    all_rows = []

    for i, s in enumerate(samples):
        try:
            image = Image.open(s["img_path"]).convert("RGB")
        except Exception:
            continue

        gt_ratio = s["gt_ratio"]
        cat = s["category"]
        label = PIX3D_TO_LABEL[cat]

        # --- MoGe depth 추론 ---
        img_for_moge = image.resize((512, 512))
        img_t = torch.from_numpy(np.array(img_for_moge)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            moge_out = moge.infer(img_t.cuda(), force_projection=False)

        # --- Boxer (depth 없음, GT pose) ---
        t0 = time.perf_counter()
        out_no_depth = boxer.predict(
            image, s["bbox"], s["focal_length"],
            rot_mat=s["rot_mat"], trans_mat=s["trans_mat"],
            moge_output=None
        )
        time_no_depth = (time.perf_counter() - t0) * 1000

        # --- Boxer + MoGe depth + GT pose ---
        t0 = time.perf_counter()
        out_moge = boxer.predict(
            image, s["bbox"], s["focal_length"],
            rot_mat=s["rot_mat"], trans_mat=s["trans_mat"],
            moge_output=moge_out
        )
        time_moge = (time.perf_counter() - t0) * 1000

        # --- Rule-based ---
        rule = abs_calc.calculate_absolute_volume(label, None, 1.0, 1.5, 0.8)
        rule_dims = sorted([rule.width_mm, rule.depth_mm, rule.height_mm])
        rule_ratio = compute_aspect_ratio(rule_dims)
        rule_are = aspect_ratio_error(rule_ratio, gt_ratio)

        # Boxer no depth
        if out_no_depth:
            w, d, h, vol, conf = out_no_depth
            nd_dims = sorted([w, d, h])
            nd_ratio = compute_aspect_ratio(nd_dims)
            nd_are = aspect_ratio_error(nd_ratio, gt_ratio)
            nd_conf = conf
        else:
            nd_dims = [0, 0, 0]
            nd_ratio = [0, 0, 0]
            nd_are = 1.0
            nd_conf = 0

        # Boxer + MoGe
        if out_moge:
            w, d, h, vol, conf = out_moge
            mg_dims = sorted([w, d, h])
            mg_ratio = compute_aspect_ratio(mg_dims)
            mg_are = aspect_ratio_error(mg_ratio, gt_ratio)
            mg_conf = conf
        else:
            mg_dims = [0, 0, 0]
            mg_ratio = [0, 0, 0]
            mg_are = 1.0
            mg_conf = 0

        results["boxer_no_depth"][cat].append(nd_are)
        results["boxer_moge"][cat].append(mg_are)
        results["rule_based"][cat].append(rule_are)

        all_rows.append({
            "image": os.path.basename(s["img_path"]),
            "category": cat,
            "gt_ratio": gt_ratio,
            "nd_are": nd_are, "nd_conf": nd_conf, "nd_dims": nd_dims,
            "mg_are": mg_are, "mg_conf": mg_conf, "mg_dims": mg_dims,
            "rule_are": rule_are, "rule_dims": rule_dims,
            "time_nd_ms": time_no_depth, "time_mg_ms": time_moge,
        })

        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(samples)}...")

    # 결과 분석
    print(f"\n[5/5] 결과 분석")
    print("=" * 70)

    all_nd = [r["nd_are"] for r in all_rows]
    all_mg = [r["mg_are"] for r in all_rows]
    all_ru = [r["rule_are"] for r in all_rows]

    print(f"\n  --- 전체 Aspect Ratio Error (낮을수록 좋음) ---")
    print(f"  {'Method':<22} | {'Mean ARE':<10} | {'Std':<10} | {'Median':<10} | {'<0.05':<8} | {'<0.10':<8}")
    print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    for name, errs in [
        ("Boxer (no depth)", all_nd),
        ("Boxer + MoGe depth", all_mg),
        ("Rule-based", all_ru),
    ]:
        arr = np.array(errs)
        u5 = (arr < 0.05).sum() / len(arr) * 100
        u10 = (arr < 0.10).sum() / len(arr) * 100
        print(f"  {name:<22} | {arr.mean():>8.4f}  | {arr.std():>8.4f}  | {np.median(arr):>8.4f}  | {u5:>5.1f}%  | {u10:>5.1f}%")

    # 개선도
    nd_mean = np.mean(all_nd)
    mg_mean = np.mean(all_mg)
    improvement = (nd_mean - mg_mean) / nd_mean * 100
    print(f"\n  MoGe depth 추가 효과: ARE {nd_mean:.4f} → {mg_mean:.4f} ({improvement:+.1f}%)")

    # Boxer 신뢰도 비교
    nd_confs = [r["nd_conf"] for r in all_rows if r["nd_conf"] > 0]
    mg_confs = [r["mg_conf"] for r in all_rows if r["mg_conf"] > 0]
    print(f"\n  --- Boxer 신뢰도 ---")
    print(f"  No depth: {np.mean(nd_confs):.3f} ± {np.std(nd_confs):.3f}")
    print(f"  + MoGe:   {np.mean(mg_confs):.3f} ± {np.std(mg_confs):.3f}")

    # 카테고리별
    print(f"\n  --- 카테고리별 ARE ---")
    print(f"  {'Category':<12} | {'N':<5} | {'No Depth':<10} | {'+ MoGe':<10} | {'Rule':<10} | {'Best':<12}")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    wins = {"boxer_no_depth": 0, "boxer_moge": 0, "rule_based": 0}
    for cat in ['bed', 'sofa', 'chair', 'desk', 'table', 'bookcase', 'wardrobe']:
        nd_errs = results["boxer_no_depth"].get(cat, [])
        mg_errs = results["boxer_moge"].get(cat, [])
        ru_errs = results["rule_based"].get(cat, [])
        if not nd_errs:
            continue

        nd_m, mg_m, ru_m = np.mean(nd_errs), np.mean(mg_errs), np.mean(ru_errs)
        best_val = min(nd_m, mg_m, ru_m)
        if best_val == mg_m:
            best = "Boxer+MoGe"
            wins["boxer_moge"] += 1
        elif best_val == nd_m:
            best = "Boxer"
            wins["boxer_no_depth"] += 1
        else:
            best = "Rule"
            wins["rule_based"] += 1

        print(f"  {cat:<12} | {len(nd_errs):<5} | {nd_m:>8.4f}  | {mg_m:>8.4f}  | {ru_m:>8.4f}  | {best:<12}")

    print(f"\n  카테고리 승리: Boxer+MoGe={wins['boxer_moge']}, Boxer(no depth)={wins['boxer_no_depth']}, Rule={wins['rule_based']}")

    # 처리 시간
    nd_times = [r["time_nd_ms"] for r in all_rows]
    mg_times = [r["time_mg_ms"] for r in all_rows]
    print(f"\n  --- 처리 시간 (Boxer 추론만) ---")
    print(f"  No depth: {np.mean(nd_times):.1f}ms/obj")
    print(f"  + MoGe:   {np.mean(mg_times):.1f}ms/obj")

    # Boxer + MoGe 절대 치수 분포
    print(f"\n  --- Boxer + MoGe 절대 치수 분포 (mm) ---")
    for cat in ['bed', 'sofa', 'chair', 'desk', 'table']:
        cat_rows = [r for r in all_rows if r["category"] == cat and max(r["mg_dims"]) > 0]
        if not cat_rows:
            continue
        arr = np.array([r["mg_dims"] for r in cat_rows])
        print(f"  {cat:<8}: short={arr[:,0].mean():.0f}±{arr[:,0].std():.0f}, "
              f"mid={arr[:,1].mean():.0f}±{arr[:,1].std():.0f}, "
              f"long={arr[:,2].mean():.0f}±{arr[:,2].std():.0f}mm")

    # CSV
    csv_path = output_dir / "pix3d_moge_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image", "category",
            "gt_ratio_s", "gt_ratio_m",
            "nd_ARE", "nd_conf", "nd_short", "nd_mid", "nd_long",
            "mg_ARE", "mg_conf", "mg_short", "mg_mid", "mg_long",
            "rule_ARE", "rule_short", "rule_mid", "rule_long",
        ])
        for r in all_rows:
            w.writerow([
                r["image"], r["category"],
                f"{r['gt_ratio'][0]:.4f}", f"{r['gt_ratio'][1]:.4f}",
                f"{r['nd_are']:.4f}", f"{r['nd_conf']:.3f}",
                f"{r['nd_dims'][0]:.0f}", f"{r['nd_dims'][1]:.0f}", f"{r['nd_dims'][2]:.0f}",
                f"{r['mg_are']:.4f}", f"{r['mg_conf']:.3f}",
                f"{r['mg_dims'][0]:.0f}", f"{r['mg_dims'][1]:.0f}", f"{r['mg_dims'][2]:.0f}",
                f"{r['rule_are']:.4f}",
                f"{r['rule_dims'][0]:.0f}", f"{r['rule_dims'][1]:.0f}", f"{r['rule_dims'][2]:.0f}",
            ])
    print(f"\n  CSV: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
