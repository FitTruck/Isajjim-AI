"""
Pix3D GT 기반 V2.5 vs V3 정량 비교 실험

Pix3D 데이터셋:
- 실제 가구 이미지 + GT 2D bbox + GT 3D OBJ 모델 + 카메라 파라미터
- 3D 모델은 정규화 스케일 → 비율(aspect ratio)로 형상 정확도 평가
- Pix3D의 실제 focal_length를 Boxer에 제공하여 공정 비교

평가 지표:
1. Aspect Ratio Error: GT 3D 모델 비율 vs 예측 비율
2. Boxer 신뢰도 분포
3. 카테고리별 분석

Usage:
    conda run -n sam3d-objects python /path/to/experiment_pix3d.py
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
os.chdir(str(ROOT / "boxer"))
sys.path.insert(0, str(ROOT / "boxer"))
sys.path.insert(1, str(ROOT))

from boxernet.boxernet import BoxerNet
from utils.tw.camera import get_pinhole_camera
from utils.tw.pose import PoseTW

os.chdir(str(ROOT))

from ai.processors import AbsoluteVolumeCalculator

PIX3D_ROOT = ROOT / "experiments" / "seed_variance" / "data" / "pix3d"

# Pix3D 카테고리 → 우리 시스템 라벨 매핑
PIX3D_TO_LABEL = {
    "bed": "BED",
    "sofa": "SOFA",
    "chair": "CHAIR_STOOL",
    "desk": "DESK",
    "table": "DINING_TABLE",
    "bookcase": "BOOKSHELF",
    "wardrobe": "WARDROBE",
}


@dataclass
class Pix3DSample:
    """Pix3D 샘플"""
    img_path: str
    category: str
    bbox: List[int]               # [x1, y1, x2, y2]
    img_size: List[int]           # [W, H]
    focal_length: float
    model_path: str
    gt_aspect_ratio: List[float]  # 정렬된 비율 [short/long, mid/long, 1.0]
    gt_dims_model: List[float]    # 모델 좌표 치수 (정렬, m)
    truncated: bool
    occluded: bool


@dataclass
class PredictionResult:
    """예측 결과"""
    # Boxer
    boxer_dims_mm: List[float] = field(default_factory=list)  # [w, d, h] sorted
    boxer_aspect_ratio: List[float] = field(default_factory=list)
    boxer_confidence: float = 0
    boxer_time_ms: float = 0
    # 규칙기반
    rule_dims_mm: List[float] = field(default_factory=list)
    rule_aspect_ratio: List[float] = field(default_factory=list)
    rule_type: Optional[str] = None


def compute_aspect_ratio(dims: List[float]) -> List[float]:
    """치수를 정렬 후 비율로 변환 [short/long, mid/long, 1.0]"""
    s = sorted(dims)
    if s[2] == 0:
        return [0, 0, 0]
    return [s[0] / s[2], s[1] / s[2], 1.0]


def aspect_ratio_error(pred: List[float], gt: List[float]) -> float:
    """두 aspect ratio 간 평균 절대 오차 (0~1)"""
    if not pred or not gt:
        return 1.0
    return float(np.mean([abs(p - g) for p, g in zip(pred, gt)]))


def load_pix3d_samples(max_per_cat: int = 30) -> List[Pix3DSample]:
    """Pix3D에서 유효한 샘플 로드 (비가림, 비절단)"""
    with open(PIX3D_ROOT / "pix3d.json") as f:
        data = json.load(f)

    samples = []
    cat_count = defaultdict(int)
    seen_models = {}  # model_path → gt_dims (중복 로드 방지)

    for d in data:
        cat = d["category"]
        if cat not in PIX3D_TO_LABEL:
            continue
        if d["truncated"] or d["occluded"]:
            continue
        if cat_count[cat] >= max_per_cat:
            continue

        model_path = d["model"]
        full_model = str(PIX3D_ROOT / model_path)

        # GT 치수 계산 (모델 로드 캐싱)
        if model_path not in seen_models:
            try:
                mesh = trimesh.load(full_model, force="mesh")
                aabb = mesh.bounds[1] - mesh.bounds[0]
                seen_models[model_path] = sorted(aabb.tolist())
            except Exception:
                continue

        gt_dims = seen_models[model_path]
        gt_ratio = compute_aspect_ratio(gt_dims)

        bbox = d["bbox"]  # [x1, y1, x2, y2]

        samples.append(Pix3DSample(
            img_path=str(PIX3D_ROOT / d["img"]),
            category=cat,
            bbox=bbox,
            img_size=d["img_size"],
            focal_length=d["focal_length"],
            model_path=model_path,
            gt_aspect_ratio=gt_ratio,
            gt_dims_model=gt_dims,
            truncated=d["truncated"],
            occluded=d["occluded"],
        ))
        cat_count[cat] += 1

    return samples


class BoxerPredictor:
    """Pix3D 실험용 Boxer 추론기"""

    def __init__(self, device="cuda"):
        self.device = device
        ckpt = str(ROOT / "boxer" / "ckpts" / "boxernet_hw960in4x6d768-wssxpf9p.ckpt")
        print(f"[Boxer] Loading from {ckpt}")
        self.model = BoxerNet.load_from_checkpoint(ckpt, device=device)
        self.model.eval()

        self._R_yz = torch.tensor(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32
        )

    @torch.no_grad()
    def predict(self, image: Image.Image, bbox: List[int],
                focal_length: Optional[float] = None) -> Optional[Tuple]:
        """단일 객체 추론. Pix3D focal_length 사용 가능."""
        W_orig, H_orig = image.size
        target = 960
        sx, sy = target / W_orig, target / H_orig
        image_r = image.resize((target, target), Image.BILINEAR)

        img_np = np.array(image_r.convert("RGB"))
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_t = img_t.to(self.device)

        # intrinsics: Pix3D focal_length 사용 시 스케일 변환
        if focal_length is not None:
            # Pix3D focal_length는 mm 단위, 센서 크기 기준
            # 일반적으로 Pix3D 이미지는 render → pixel focal length ≈ fl * W / 36
            # 36mm = full-frame sensor width
            f_px = focal_length * max(W_orig, H_orig) / 36.0
            f_scaled = f_px * (target / max(W_orig, H_orig))
        else:
            f_scaled = target * 0.8

        cam = get_pinhole_camera(
            params=[f_scaled, f_scaled, target / 2.0, target / 2.0],
            width=target, height=target
        ).to(self.device)

        R = self._R_yz.clone()
        t = torch.zeros(3, dtype=torch.float32)
        pose = PoseTW.from_Rt(R, t).to(self.device)

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
            "sdp_w": torch.zeros(0, 3, device=self.device),
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

        return (dims[2] * 1000, dims[0] * 1000, dims[1] * 1000,  # w, d, h (mm)
                vol, conf)


def run_experiment():
    output_dir = ROOT / "experiments" / "boxer_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Pix3D GT 기반 V2.5 vs V3 비교 실험")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[1/4] Pix3D 데이터 로드...")
    samples = load_pix3d_samples(max_per_cat=30)
    cat_counts = defaultdict(int)
    for s in samples:
        cat_counts[s.category] += 1
    print(f"  총 {len(samples)}개 샘플: {dict(cat_counts)}")

    # 2. 모델 로드
    print("\n[2/4] Boxer 로드...")
    boxer = BoxerPredictor(device="cuda")

    abs_calc = AbsoluteVolumeCalculator()

    # 3. 실험 실행
    print(f"\n[3/4] 실험 실행...")

    results_by_cat: Dict[str, List[dict]] = defaultdict(list)
    all_csv_rows = []

    total_boxer_time = 0
    total_samples = 0
    errors_boxer = []
    errors_rule = []

    for i, sample in enumerate(samples):
        try:
            image = Image.open(sample.img_path).convert("RGB")
        except Exception:
            continue

        # --- Boxer 추론 (Pix3D focal_length 사용) ---
        t0 = time.perf_counter()
        boxer_out = boxer.predict(image, sample.bbox, focal_length=sample.focal_length)
        boxer_ms = (time.perf_counter() - t0) * 1000
        total_boxer_time += boxer_ms

        boxer_dims_mm = [0, 0, 0]
        boxer_ratio = [0, 0, 0]
        boxer_conf = 0

        if boxer_out:
            w, d, h, vol, conf = boxer_out
            boxer_dims_mm = sorted([w, d, h])
            boxer_ratio = compute_aspect_ratio(boxer_dims_mm)
            boxer_conf = conf

        # --- 규칙기반 ---
        label = PIX3D_TO_LABEL.get(sample.category, "")
        rule_result = abs_calc.calculate_absolute_volume(
            label=label, type_name=None,
            rel_width=1.0, rel_depth=1.5, rel_height=0.8  # 일반적 가구 비율 가정
        )
        rule_dims_mm = sorted([rule_result.width_mm, rule_result.depth_mm, rule_result.height_mm])
        rule_ratio = compute_aspect_ratio(rule_dims_mm)

        # --- Aspect Ratio Error 계산 ---
        gt_ratio = sample.gt_aspect_ratio
        boxer_ar_error = aspect_ratio_error(boxer_ratio, gt_ratio)
        rule_ar_error = aspect_ratio_error(rule_ratio, gt_ratio)

        errors_boxer.append(boxer_ar_error)
        errors_rule.append(rule_ar_error)

        entry = {
            "image": os.path.basename(sample.img_path),
            "category": sample.category,
            "gt_ratio": gt_ratio,
            "gt_dims_model": sample.gt_dims_model,
            "boxer_dims_mm": boxer_dims_mm,
            "boxer_ratio": boxer_ratio,
            "boxer_conf": boxer_conf,
            "boxer_ar_error": boxer_ar_error,
            "rule_type": rule_result.matched_type,
            "rule_dims_mm": rule_dims_mm,
            "rule_ratio": rule_ratio,
            "rule_ar_error": rule_ar_error,
            "boxer_ms": boxer_ms,
        }
        results_by_cat[sample.category].append(entry)
        all_csv_rows.append(entry)
        total_samples += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(samples)} processed...")

    # 4. 결과 분석
    print(f"\n[4/4] 결과 분석 ({total_samples} samples)")
    print("=" * 70)

    # 전체 요약
    print(f"\n  --- 전체 Aspect Ratio Error (낮을수록 좋음) ---")
    print(f"  {'Method':<14} | {'Mean ARE':<10} | {'Std':<10} | {'Median':<10} | {'<0.05':<8} | {'<0.10':<8}")
    print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    for name, errs in [("Boxer (V3)", errors_boxer), ("Rule-based", errors_rule)]:
        arr = np.array(errs)
        under5 = (arr < 0.05).sum() / len(arr) * 100
        under10 = (arr < 0.10).sum() / len(arr) * 100
        print(f"  {name:<14} | {arr.mean():>8.4f}  | {arr.std():>8.4f}  | {np.median(arr):>8.4f}  | {under5:>5.1f}%  | {under10:>5.1f}%")

    # Boxer 신뢰도 분포
    confs = [e["boxer_conf"] for e in all_csv_rows if e["boxer_conf"] > 0]
    print(f"\n  --- Boxer 신뢰도 ---")
    print(f"  Mean: {np.mean(confs):.3f} ± {np.std(confs):.3f}")
    print(f"  Range: [{min(confs):.3f}, {max(confs):.3f}]")
    print(f"  >0.5: {sum(1 for c in confs if c > 0.5)}/{len(confs)} ({sum(1 for c in confs if c > 0.5)/len(confs)*100:.0f}%)")

    # 카테고리별 분석
    print(f"\n  --- 카테고리별 Aspect Ratio Error ---")
    print(f"  {'Category':<12} | {'N':<5} | {'Boxer ARE':<12} | {'Rule ARE':<12} | {'Winner':<8} | {'Boxer Conf':<10}")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*10}")

    boxer_wins = 0
    rule_wins = 0
    ties = 0

    for cat in ['bed', 'sofa', 'chair', 'desk', 'table', 'bookcase', 'wardrobe']:
        entries = results_by_cat.get(cat, [])
        if not entries:
            continue
        b_errs = [e["boxer_ar_error"] for e in entries]
        r_errs = [e["rule_ar_error"] for e in entries]
        b_confs = [e["boxer_conf"] for e in entries]

        b_mean = np.mean(b_errs)
        r_mean = np.mean(r_errs)

        if b_mean < r_mean - 0.005:
            winner = "Boxer"
            boxer_wins += 1
        elif r_mean < b_mean - 0.005:
            winner = "Rule"
            rule_wins += 1
        else:
            winner = "Tie"
            ties += 1

        print(f"  {cat:<12} | {len(entries):<5} | {b_mean:>10.4f}  | {r_mean:>10.4f}  | {winner:<8} | {np.mean(b_confs):>8.3f}")

    print(f"\n  카테고리 승리: Boxer={boxer_wins}, Rule={rule_wins}, Tie={ties}")

    # 속도
    print(f"\n  --- 처리 시간 ---")
    print(f"  Boxer: {total_boxer_time/max(total_samples,1):.1f}ms/obj (총 {total_boxer_time/1000:.1f}s)")
    print(f"  Rule-based: <0.1ms/obj")

    # Boxer 치수 분포
    print(f"\n  --- Boxer 절대 치수 분포 (mm) ---")
    for cat in ['bed', 'sofa', 'chair', 'desk', 'table']:
        entries = results_by_cat.get(cat, [])
        if not entries:
            continue
        dims = [e["boxer_dims_mm"] for e in entries if max(e["boxer_dims_mm"]) > 0]
        if not dims:
            continue
        arr = np.array(dims)  # (N, 3) sorted [short, mid, long]
        print(f"  {cat:<8}: short={arr[:,0].mean():.0f}±{arr[:,0].std():.0f}, "
              f"mid={arr[:,1].mean():.0f}±{arr[:,1].std():.0f}, "
              f"long={arr[:,2].mean():.0f}±{arr[:,2].std():.0f}mm")

    # CSV 저장
    csv_path = output_dir / "pix3d_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "category",
            "gt_ratio_short", "gt_ratio_mid",
            "boxer_W_mm", "boxer_D_mm", "boxer_H_mm",
            "boxer_ratio_short", "boxer_ratio_mid", "boxer_conf",
            "boxer_ARE",
            "rule_type", "rule_W_mm", "rule_D_mm", "rule_H_mm",
            "rule_ratio_short", "rule_ratio_mid",
            "rule_ARE",
            "boxer_ms",
        ])
        for e in all_csv_rows:
            writer.writerow([
                e["image"], e["category"],
                f"{e['gt_ratio'][0]:.4f}", f"{e['gt_ratio'][1]:.4f}",
                f"{e['boxer_dims_mm'][2]:.0f}", f"{e['boxer_dims_mm'][0]:.0f}", f"{e['boxer_dims_mm'][1]:.0f}",
                f"{e['boxer_ratio'][0]:.4f}", f"{e['boxer_ratio'][1]:.4f}", f"{e['boxer_conf']:.3f}",
                f"{e['boxer_ar_error']:.4f}",
                e["rule_type"],
                f"{e['rule_dims_mm'][2]:.0f}", f"{e['rule_dims_mm'][0]:.0f}", f"{e['rule_dims_mm'][1]:.0f}",
                f"{e['rule_ratio'][0]:.4f}", f"{e['rule_ratio'][1]:.4f}",
                f"{e['rule_ar_error']:.4f}",
                f"{e['boxer_ms']:.1f}",
            ])
    print(f"\n  CSV: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
