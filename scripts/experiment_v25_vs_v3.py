"""
V2.5 vs V3 Pipeline 비교 실험

테스트 이미지에 대해:
  1. YOLOE-seg로 객체 탐지 (bbox + label)
  2. 규칙기반(V2.5): label + 상대비율 → 하드코딩 치수
  3. Boxer(V3): bbox + 이미지 → 절대 3D OBB
  4. 두 방식의 치수를 비교 분석

Usage:
    conda run -n sam3d-objects python scripts/experiment_v25_vs_v3.py

출력:
    - 콘솔 비교 테이블
    - experiments/boxer_comparison/results.csv
"""

import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# 프로젝트 루트
ROOT = Path(__file__).resolve().parent.parent

# Boxer는 CWD가 boxer/ 디렉토리여야 내부 utils를 찾음
# 먼저 boxer/ 로 이동해서 Boxer를 import한 뒤, 원래 경로로 복원
_original_cwd = os.getcwd()
os.chdir(str(ROOT / "boxer"))
sys.path.insert(0, str(ROOT / "boxer"))
sys.path.insert(1, str(ROOT))

# --- Boxer import (CWD=boxer/) ---
from boxernet.boxernet import BoxerNet
from utils.tw.camera import get_pinhole_camera
from utils.tw.pose import PoseTW

# CWD 복원
os.chdir(_original_cwd)

# --- AI 모듈 import ---
from ai.config import Config
from ai.processors import YoloDetector, AbsoluteVolumeCalculator
from ai.data.furniture_dimensions import FURNITURE_TYPES, get_valid_type_or_none


# ================================================================
# 데이터 구조
# ================================================================

@dataclass
class DetectionResult:
    """YOLOE 탐지 결과"""
    image_name: str
    obj_idx: int
    label: str
    bbox: List[int]            # [x1, y1, x2, y2]
    confidence: float
    subtype_name: Optional[str] = None


@dataclass
class DimensionComparison:
    """치수 비교 결과"""
    image_name: str
    obj_idx: int
    label: str
    confidence: float
    # 규칙기반 (V2.5)
    rule_type: Optional[str] = None
    rule_width_mm: float = 0
    rule_depth_mm: float = 0
    rule_height_mm: float = 0
    rule_volume_m3: float = 0
    # Boxer (V3)
    boxer_width_mm: float = 0
    boxer_depth_mm: float = 0
    boxer_height_mm: float = 0
    boxer_volume_m3: float = 0
    boxer_confidence: float = 0
    # 차이
    width_diff_pct: float = 0
    depth_diff_pct: float = 0
    height_diff_pct: float = 0
    volume_diff_pct: float = 0


# ================================================================
# Boxer 추론기 (실험용 직접 구현)
# ================================================================

class BoxerPredictor:
    """실험용 Boxer 추론기"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        ckpt_path = str(ROOT / "boxer" / "ckpts" / "boxernet_hw960in4x6d768-wssxpf9p.ckpt")
        print(f"[Boxer] Loading model from {ckpt_path}")
        self.model = BoxerNet.load_from_checkpoint(ckpt_path, device=device)
        self.model.eval()
        print(f"[Boxer] Model loaded, GPU mem: {torch.cuda.memory_allocated()/1e6:.0f}MB")

        # Y-Z swap (SUN-RGBD omni_loader fallback)
        self._R_yz = torch.tensor(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32
        )

    @torch.no_grad()
    def predict(self, image: Image.Image, bboxes_xyxy: List[List[int]]):
        """
        Returns: list of (width_m, depth_m, height_m, volume_m3, confidence)
        """
        if not bboxes_xyxy:
            return []

        W_orig, H_orig = image.size

        # BoxerNet은 이미지가 patch_size(16)의 배수여야 함
        # 체크포인트 기본: 960x960
        target_hw = 960
        scale_x = target_hw / W_orig
        scale_y = target_hw / H_orig
        image_resized = image.resize((target_hw, target_hw), Image.BILINEAR)

        # 이미지 텐서
        img_np = np.array(image_resized.convert("RGB"))
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_t = img_t.to(self.device)

        # Intrinsics (리사이즈 후 기준)
        focal = target_hw * 0.8
        camera = get_pinhole_camera(
            params=[focal, focal, target_hw / 2.0, target_hw / 2.0],
            width=target_hw, height=target_hw
        ).to(self.device)

        # Pose (Y-Z swap)
        R = self._R_yz.clone()
        t = torch.zeros(3, dtype=torch.float32)
        pose = PoseTW.from_Rt(R, t).to(self.device)

        # bbox 변환: 원본 좌표 → 리사이즈 좌표, [x1,y1,x2,y2] → [xmin,xmax,ymin,ymax]
        bb2d = torch.tensor(
            [[[x1 * scale_x, x2 * scale_x, y1 * scale_y, y2 * scale_y]
              for x1, y1, x2, y2 in bboxes_xyxy]],
            dtype=torch.float32, device=self.device
        )

        datum = {
            "img0": img_t,
            "cam0": camera,
            "T_world_rig0": pose,
            "rotated0": torch.tensor([False], device=self.device),
            "sdp_w": torch.zeros(0, 3, device=self.device),
            "bb2d": bb2d,
        }

        output = self.model(datum)
        obbs = output.get("obbs_pr_w")

        if obbs is None:
            return [None] * len(bboxes_xyxy)

        results = []
        for i in range(min(len(bboxes_xyxy), obbs.shape[-2])):
            obb_i = obbs[..., i, :]
            diag = obb_i.bb3_diagonal.squeeze()
            dims = sorted([float(diag[0].abs()), float(diag[1].abs()), float(diag[2].abs())])
            # 정렬: 가장 짧은 = depth, 중간 = height, 가장 긴 = width
            d_m, h_m, w_m = dims[0], dims[1], dims[2]
            vol = float(obb_i.bb3_volumes.squeeze().abs())
            conf = float(obb_i.prob.squeeze())
            results.append((w_m, d_m, h_m, vol, conf))

        return results


# ================================================================
# 실험 실행
# ================================================================

def run_experiment():
    output_dir = ROOT / "experiments" / "boxer_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. YOLOE 탐지기 로드
    print("=" * 60)
    print("  V2.5 vs V3 비교 실험")
    print("=" * 60)
    print("\n[1/4] YOLOE 로드...")
    detector = YoloDetector(device_id=0)

    # 2. Boxer 로드
    print("\n[2/4] Boxer 로드...")
    boxer = BoxerPredictor(device="cuda")

    # 3. 규칙기반 계산기
    abs_calc = AbsoluteVolumeCalculator()

    # 4. 이미지 처리
    image_dir = ROOT / "ai" / "imgs"
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in (".jpg", ".png")])
    print(f"\n[3/4] {len(image_files)}개 이미지 처리...")

    all_comparisons: List[DimensionComparison] = []
    total_yoloe_time = 0
    total_boxer_time = 0
    total_rule_time = 0

    for img_file in image_files:
        print(f"\n--- {img_file.name} ---")
        image = Image.open(img_file).convert("RGB")

        # YOLOE 탐지
        t0 = time.perf_counter()
        results = detector.detect_smart(image, return_masks=True)
        yoloe_elapsed = time.perf_counter() - t0
        total_yoloe_time += yoloe_elapsed

        if results is None or len(results["boxes"]) == 0:
            print(f"  No objects detected")
            continue

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        # 유효한 탐지만 필터링
        detections = []
        for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = map(int, box)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
            detections.append(DetectionResult(
                image_name=img_file.name,
                obj_idx=idx,
                label=label,
                bbox=[x1, y1, x2, y2],
                confidence=float(score),
            ))

        if not detections:
            continue

        print(f"  YOLOE: {len(detections)} objects ({yoloe_elapsed:.2f}s)")

        # Boxer 추론 (전체 bbox 한 번에)
        bboxes = [d.bbox for d in detections]
        t0 = time.perf_counter()
        boxer_results = boxer.predict(image, bboxes)
        boxer_elapsed = time.perf_counter() - t0
        total_boxer_time += boxer_elapsed
        print(f"  Boxer: {len(boxer_results)} results ({boxer_elapsed:.2f}s)")

        # 각 객체별 비교
        for i, det in enumerate(detections):
            comp = DimensionComparison(
                image_name=det.image_name,
                obj_idx=det.obj_idx,
                label=det.label,
                confidence=det.confidence,
            )

            # 규칙기반 (V2.5) - 가상 상대치수로 계산
            # 실제 파이프라인에서는 SAM3D PLY에서 상대치수를 추출하지만,
            # 여기서는 규칙기반의 "best match"만 사용 (상대 치수 1:1:1)
            t0 = time.perf_counter()
            # 가장 기본적인 타입 매칭 (label → first subtype)
            rule_result = abs_calc.calculate_absolute_volume(
                label=det.label.upper(),
                type_name=None,  # best_match 찾기
                rel_width=1.0, rel_depth=1.0, rel_height=1.0
            )
            total_rule_time += time.perf_counter() - t0

            comp.rule_type = rule_result.matched_type
            comp.rule_width_mm = rule_result.width_mm
            comp.rule_depth_mm = rule_result.depth_mm
            comp.rule_height_mm = rule_result.height_mm
            comp.rule_volume_m3 = rule_result.volume_m3

            # Boxer (V3)
            if i < len(boxer_results) and boxer_results[i] is not None:
                w_m, d_m, h_m, vol, conf = boxer_results[i]
                comp.boxer_width_mm = round(w_m * 1000, 1)
                comp.boxer_depth_mm = round(d_m * 1000, 1)
                comp.boxer_height_mm = round(h_m * 1000, 1)
                comp.boxer_volume_m3 = round(vol, 6)
                comp.boxer_confidence = round(conf, 3)

                # 차이 계산 (규칙기반 대비)
                if comp.rule_width_mm > 0:
                    comp.width_diff_pct = round(
                        abs(comp.boxer_width_mm - comp.rule_width_mm) / comp.rule_width_mm * 100, 1
                    )
                if comp.rule_depth_mm > 0:
                    comp.depth_diff_pct = round(
                        abs(comp.boxer_depth_mm - comp.rule_depth_mm) / comp.rule_depth_mm * 100, 1
                    )
                if comp.rule_height_mm > 0:
                    comp.height_diff_pct = round(
                        abs(comp.boxer_height_mm - comp.rule_height_mm) / comp.rule_height_mm * 100, 1
                    )
                if comp.rule_volume_m3 > 0:
                    comp.volume_diff_pct = round(
                        abs(comp.boxer_volume_m3 - comp.rule_volume_m3) / comp.rule_volume_m3 * 100, 1
                    )

            all_comparisons.append(comp)

            print(f"  [{i}] {det.label} (conf={det.confidence:.2f})")
            print(f"      Rule: {comp.rule_width_mm:.0f}x{comp.rule_depth_mm:.0f}x{comp.rule_height_mm:.0f}mm = {comp.rule_volume_m3:.3f}m³ (type={comp.rule_type})")
            print(f"      Boxer: {comp.boxer_width_mm:.0f}x{comp.boxer_depth_mm:.0f}x{comp.boxer_height_mm:.0f}mm = {comp.boxer_volume_m3:.3f}m³ (conf={comp.boxer_confidence:.2f})")
            print(f"      Diff: W={comp.width_diff_pct:.1f}% D={comp.depth_diff_pct:.1f}% H={comp.height_diff_pct:.1f}% V={comp.volume_diff_pct:.1f}%")

    # ================================================================
    # 결과 분석
    # ================================================================
    print("\n" + "=" * 70)
    print("  실험 결과 요약")
    print("=" * 70)

    n = len(all_comparisons)
    if n == 0:
        print("  No objects to compare")
        return

    # 기본 통계
    boxer_confs = [c.boxer_confidence for c in all_comparisons if c.boxer_confidence > 0]
    w_diffs = [c.width_diff_pct for c in all_comparisons if c.boxer_width_mm > 0]
    d_diffs = [c.depth_diff_pct for c in all_comparisons if c.boxer_depth_mm > 0]
    h_diffs = [c.height_diff_pct for c in all_comparisons if c.boxer_height_mm > 0]
    v_diffs = [c.volume_diff_pct for c in all_comparisons if c.boxer_volume_m3 > 0]

    print(f"\n  총 객체 수: {n}")
    print(f"  총 이미지: {len(image_files)}")
    print(f"  Boxer 유효 추론: {len(boxer_confs)}/{n}")
    print(f"  Boxer 평균 신뢰도: {np.mean(boxer_confs):.3f} ± {np.std(boxer_confs):.3f}" if boxer_confs else "  N/A")

    print(f"\n  --- 규칙기반(V2.5) vs Boxer(V3) 치수 차이 ---")
    print(f"  {'지표':<14} | {'평균 차이(%)':<14} | {'표준편차(%)':<14} | {'최소':<8} | {'최대':<8}")
    print(f"  {'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}-+-{'-'*8}")

    for name, diffs in [("Width", w_diffs), ("Depth", d_diffs), ("Height", h_diffs), ("Volume", v_diffs)]:
        if diffs:
            print(f"  {name:<14} | {np.mean(diffs):>12.1f}% | {np.std(diffs):>12.1f}% | {min(diffs):>6.1f}% | {max(diffs):>6.1f}%")
        else:
            print(f"  {name:<14} | {'N/A':>12} | {'N/A':>12} | {'N/A':>6} | {'N/A':>6}")

    print(f"\n  --- 처리 시간 ---")
    print(f"  YOLOE 전체: {total_yoloe_time:.2f}s ({total_yoloe_time/max(len(image_files),1):.2f}s/img)")
    print(f"  Boxer 전체: {total_boxer_time:.2f}s ({total_boxer_time/max(len(image_files),1):.2f}s/img)")
    print(f"  규칙기반 전체: {total_rule_time:.4f}s ({total_rule_time/max(n,1)*1000:.2f}ms/obj)")

    # Boxer 치수 분포
    print(f"\n  --- Boxer 치수 분포 ---")
    b_widths = [c.boxer_width_mm for c in all_comparisons if c.boxer_width_mm > 0]
    b_depths = [c.boxer_depth_mm for c in all_comparisons if c.boxer_depth_mm > 0]
    b_heights = [c.boxer_height_mm for c in all_comparisons if c.boxer_height_mm > 0]
    if b_widths:
        print(f"  Width:  {np.mean(b_widths):.0f} ± {np.std(b_widths):.0f}mm (range: {min(b_widths):.0f}-{max(b_widths):.0f})")
        print(f"  Depth:  {np.mean(b_depths):.0f} ± {np.std(b_depths):.0f}mm (range: {min(b_depths):.0f}-{max(b_depths):.0f})")
        print(f"  Height: {np.mean(b_heights):.0f} ± {np.std(b_heights):.0f}mm (range: {min(b_heights):.0f}-{max(b_heights):.0f})")

    # 라벨별 분석
    print(f"\n  --- 라벨별 Boxer 신뢰도 & 치수 차이 ---")
    label_groups: Dict[str, List[DimensionComparison]] = {}
    for c in all_comparisons:
        label_groups.setdefault(c.label, []).append(c)

    print(f"  {'라벨':<20} | {'Count':<6} | {'Boxer Conf':<12} | {'Vol Diff(%)':<12}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*12}-+-{'-'*12}")
    for label in sorted(label_groups.keys()):
        group = label_groups[label]
        confs = [c.boxer_confidence for c in group if c.boxer_confidence > 0]
        vdiffs = [c.volume_diff_pct for c in group if c.boxer_volume_m3 > 0]
        avg_conf = np.mean(confs) if confs else 0
        avg_vdiff = np.mean(vdiffs) if vdiffs else 0
        print(f"  {label:<20} | {len(group):<6} | {avg_conf:>10.3f}  | {avg_vdiff:>10.1f}%")

    # CSV 저장
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "obj_idx", "label", "yoloe_conf",
            "rule_type", "rule_W_mm", "rule_D_mm", "rule_H_mm", "rule_vol_m3",
            "boxer_W_mm", "boxer_D_mm", "boxer_H_mm", "boxer_vol_m3", "boxer_conf",
            "W_diff%", "D_diff%", "H_diff%", "V_diff%",
        ])
        for c in all_comparisons:
            writer.writerow([
                c.image_name, c.obj_idx, c.label, c.confidence,
                c.rule_type, c.rule_width_mm, c.rule_depth_mm, c.rule_height_mm, c.rule_volume_m3,
                c.boxer_width_mm, c.boxer_depth_mm, c.boxer_height_mm, c.boxer_volume_m3, c.boxer_confidence,
                c.width_diff_pct, c.depth_diff_pct, c.height_diff_pct, c.volume_diff_pct,
            ])
    print(f"\n  CSV 저장: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
