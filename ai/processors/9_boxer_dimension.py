"""
Boxer Dimension Estimator (V3)

Meta의 Boxer 모델을 사용하여 2D bbox로부터 절대 3D 치수를 추정합니다.
BoxerNet은 class-agnostic으로, 입력 bbox 영역의 이미지만 보고 3D OBB를 추론합니다.

V3 개선사항 (실험 결과 반영):
  1. AR 보존 리사이즈: 960x960 squash → 패딩 (ARE 30% 개선)
  2. MoGe depth → SDP: SAM3D의 MoGe를 재활용하여 depth 입력 (ARE 13% 개선)
  3. MoGe intrinsics 옵션: MoGe가 추정한 focal length 사용 가능

파이프라인 흐름:
  이미지 → MoGe (depth + intrinsics) → AR 보존 960x960 패딩 → Boxer → 절대 치수
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ai.config import Config

logger = logging.getLogger(__name__)

# Boxer lazy import (설치 안 된 환경 호환)
_BOXER_AVAILABLE = False
try:
    from boxernet.boxernet import BoxerNet
    from loaders.base_loader import BaseLoader
    from utils.tw.camera import get_pinhole_camera
    from utils.tw.pose import PoseTW
    _BOXER_AVAILABLE = True
except ImportError:
    logger.warning(
        "[BoxerDimensionEstimator] Boxer not installed. "
        "Run: bash scripts/setup_boxer.sh"
    )


@dataclass
class BoxerResult:
    """Boxer 추론 결과 (단일 객체)"""
    width_m: float          # 절대 너비 (m)
    depth_m: float          # 절대 깊이 (m)
    height_m: float         # 절대 높이 (m)
    volume_m3: float        # 절대 부피 (m³)
    confidence: float       # 신뢰도 (0~1)
    center_world: List[float]  # 3D 중심 좌표 (m)


class BoxerDimensionEstimator:
    """
    Boxer 기반 절대 치수 추정기 (V3).

    실험 결과 반영:
    - AR 보존 패딩으로 비정방형 이미지 왜곡 방지 (Pix3D ARE 0.1076 → 0.0757)
    - MoGe depth를 SDP로 변환하여 Boxer에 제공 (ARE 13% 추가 개선)
    - Y-Z swap gravity heuristic 유지 (GT pose보다 성능 우수 확인)

    Usage:
        estimator = BoxerDimensionEstimator(device_id=0)
        results = estimator.predict(image, [[x1,y1,x2,y2], ...], depth_map=moge_depth)
    """

    TARGET_SIZE = 960  # BoxerNet 입력 크기 (patch_size=16의 배수)
    SDP_NUM_SAMPLES = 10000  # Semi-dense points 샘플 수

    def __init__(self, device_id: int = 0, checkpoint_path: Optional[str] = None):
        if not _BOXER_AVAILABLE:
            raise ImportError(
                "Boxer not installed. Run: bash scripts/setup_boxer.sh"
            )

        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

        if checkpoint_path is None:
            checkpoint_path = "./boxer/ckpts/boxernet_hw960in4x6d768-wssxpf9p.ckpt"

        logger.info(f"[BoxerDimensionEstimator] Loading model on {self.device}")
        self.model = BoxerNet.load_from_checkpoint(
            checkpoint_path, device=self.device
        )
        self.model.eval()
        logger.info("[BoxerDimensionEstimator] Model loaded")

        # Y-Z swap (SUN-RGBD omni_loader fallback)
        # 실험 결과: GT pose보다 이 heuristic이 Boxer에 더 적합
        self._R_yz = np.array(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32
        )
        self._t_zero = np.zeros(3, dtype=np.float32)

    def _prepare_image_ar(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, float, int, int, int, int]:
        """
        AR 보존 리사이즈 + 패딩.

        960x960 squash 대신 aspect ratio를 유지하고 검은색으로 패딩합니다.
        실험 결과: ARE 0.1076 → 0.0757 (30% 개선)

        Returns:
            (img_tensor, scale, new_w, new_h, pad_x, pad_y)
        """
        W, H = image.size
        tgt = self.TARGET_SIZE
        scale = tgt / max(W, H)
        new_w, new_h = int(W * scale), int(H * scale)

        img_resized = image.resize((new_w, new_h), Image.BILINEAR)
        img_padded = Image.new("RGB", (tgt, tgt), (0, 0, 0))
        pad_x = (tgt - new_w) // 2
        pad_y = (tgt - new_h) // 2
        img_padded.paste(img_resized, (pad_x, pad_y))

        img_np = np.array(img_padded.convert("RGB"))
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.float() / 255.0

        return img_tensor.to(self.device), scale, new_w, new_h, pad_x, pad_y

    def _build_intrinsics(
        self, W: int, H: int, scale: float, pad_x: int, pad_y: int,
        moge_intrinsics: Optional[torch.Tensor] = None,
    ):
        """
        Boxer용 카메라 intrinsics 생성.

        Args:
            W, H: 원본 이미지 크기
            scale: AR 보존 리사이즈 배율
            pad_x, pad_y: 패딩 오프셋
            moge_intrinsics: MoGe가 추정한 3x3 K (정규화). None이면 heuristic 사용.

        Returns:
            (CameraTW, fx_scaled, fy_scaled, cx_scaled, cy_scaled)
        """
        tgt = self.TARGET_SIZE

        if moge_intrinsics is not None:
            # MoGe K: 정규화 → pixel → 리사이즈 + 패딩 보정
            K = moge_intrinsics.cpu().numpy() if torch.is_tensor(moge_intrinsics) else moge_intrinsics
            fx = float(K[0, 0]) * W * scale
            fy = float(K[1, 1]) * H * scale
            cx = float(K[0, 2]) * W * scale + pad_x
            cy = float(K[1, 2]) * H * scale + pad_y
        else:
            # Heuristic FOV
            f = max(W, H) * Config.BOXER_FOCAL_LENGTH_FACTOR * scale
            fx = fy = f
            cx = W / 2.0 * scale + pad_x
            cy = H / 2.0 * scale + pad_y

        cam = get_pinhole_camera(
            params=[fx, fy, cx, cy], width=tgt, height=tgt
        ).to(self.device)

        return cam, fx, fy, cx, cy

    def _build_sdp(
        self, depth_map: Optional[np.ndarray],
        W_orig: int, H_orig: int,
        new_w: int, new_h: int, pad_x: int, pad_y: int,
        fx: float, fy: float, cx: float, cy: float,
    ) -> torch.Tensor:
        """
        MoGe depth map → Boxer semi-dense points.

        BaseLoader.sdp_from_depth() 패턴을 사용합니다.
        depth는 AR 보존 리사이즈 + 패딩에 맞춰 변환합니다.

        Args:
            depth_map: MoGe 출력 depth (H_moge, W_moge). None이면 빈 텐서.

        Returns:
            (N, 3) float32 world-space points
        """
        if depth_map is None:
            return torch.zeros(0, 3, device=self.device)

        import cv2
        tgt = self.TARGET_SIZE

        # depth를 AR 보존 크기로 리사이즈 + 패딩
        depth_resized = cv2.resize(
            depth_map.astype(np.float32), (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )
        depth_padded = np.zeros((tgt, tgt), dtype=np.float32)
        depth_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = depth_resized

        # BaseLoader.sdp_from_depth: depth → (N, 3) world points
        sdp = BaseLoader.sdp_from_depth(
            depth_padded,
            fx=fx, fy=fy, cx=cx, cy=cy,
            R_wc=self._R_yz, t_wc=self._t_zero,
            num_samples=self.SDP_NUM_SAMPLES,
        )

        return sdp.to(self.device)

    def _convert_bboxes(
        self, bboxes_xyxy: List[List[float]],
        scale: float, pad_x: int, pad_y: int,
    ) -> torch.Tensor:
        """
        YOLOE bbox를 AR 보존 좌표 + Boxer 포맷으로 변환.

        YOLOE: [x1, y1, x2, y2] (원본 이미지 좌표)
        Boxer: [xmin, xmax, ymin, ymax] (패딩된 960x960 좌표)
        """
        if not bboxes_xyxy:
            return torch.zeros(1, 0, 4, device=self.device)

        boxer_bbs = []
        for x1, y1, x2, y2 in bboxes_xyxy:
            bx1 = x1 * scale + pad_x
            bx2 = x2 * scale + pad_x
            by1 = y1 * scale + pad_y
            by2 = y2 * scale + pad_y
            boxer_bbs.append([bx1, bx2, by1, by2])

        return torch.tensor(
            [boxer_bbs], dtype=torch.float32, device=self.device
        )

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        bboxes_xyxy: List[List[float]],
        depth_map: Optional[np.ndarray] = None,
        moge_intrinsics: Optional[torch.Tensor] = None,
    ) -> List[BoxerResult]:
        """
        이미지와 YOLOE bbox로부터 절대 3D 치수를 추정합니다.

        Args:
            image: 원본 PIL 이미지
            bboxes_xyxy: YOLOE 탐지 bbox [[x1,y1,x2,y2], ...]
            depth_map: MoGe depth map (H, W) numpy array. None이면 depth 없이 추론.
            moge_intrinsics: MoGe 추정 intrinsics (3x3 tensor). None이면 heuristic.

        Returns:
            각 bbox에 대한 BoxerResult 리스트
        """
        if not bboxes_xyxy:
            return []

        W, H = image.size

        # 1. AR 보존 리사이즈 + 패딩
        img_tensor, scale, new_w, new_h, pad_x, pad_y = self._prepare_image_ar(image)

        # 2. Intrinsics
        cam, fx, fy, cx, cy = self._build_intrinsics(
            W, H, scale, pad_x, pad_y, moge_intrinsics
        )

        # 3. Pose (Y-Z swap)
        pose = PoseTW.from_Rt(
            torch.from_numpy(self._R_yz.copy()),
            torch.from_numpy(self._t_zero.copy()),
        ).to(self.device)

        # 4. SDP (MoGe depth → semi-dense points)
        sdp = self._build_sdp(
            depth_map, W, H, new_w, new_h, pad_x, pad_y, fx, fy, cx, cy
        )

        # 5. Bbox 변환 (AR 보존 좌표)
        bb2d = self._convert_bboxes(bboxes_xyxy, scale, pad_x, pad_y)

        # 6. BoxerNet 추론
        datum = {
            "img0": img_tensor,
            "cam0": cam,
            "T_world_rig0": pose,
            "rotated0": torch.tensor([False], device=self.device),
            "sdp_w": sdp,
            "bb2d": bb2d,
        }

        output = self.model(datum)

        # 7. 결과 파싱
        obbs = output.get("obbs_pr_w")
        if obbs is None:
            logger.warning("[BoxerDimensionEstimator] No OBBs returned")
            return []

        results = []
        for i in range(min(len(bboxes_xyxy), obbs.shape[-2])):
            obb_i = obbs[..., i, :]

            diagonal = obb_i.bb3_diagonal.squeeze()
            dims = sorted([float(d.abs()) for d in diagonal])
            d_m, h_m, w_m = dims[0], dims[1], dims[2]

            vol = float(obb_i.bb3_volumes.squeeze())
            center = obb_i.bb3_center_world.squeeze().tolist()
            conf = float(obb_i.prob.squeeze())

            results.append(BoxerResult(
                width_m=w_m,
                depth_m=d_m,
                height_m=h_m,
                volume_m3=abs(vol),
                confidence=conf,
                center_world=center if isinstance(center, list) else [0, 0, 0],
            ))

        logger.info(
            f"[BoxerDimensionEstimator] Predicted {len(results)} objects, "
            f"confidences: {[f'{r.confidence:.2f}' for r in results]}"
        )

        return results


# 모듈 레벨 싱글톤
_estimator: Optional[BoxerDimensionEstimator] = None


def get_boxer_estimator(device_id: int = 0) -> Optional[BoxerDimensionEstimator]:
    """BoxerDimensionEstimator 싱글톤 반환. Boxer 미설치 시 None."""
    global _estimator
    if _estimator is None and _BOXER_AVAILABLE:
        try:
            _estimator = BoxerDimensionEstimator(device_id=device_id)
        except Exception as e:
            logger.error(f"[BoxerDimensionEstimator] Init failed: {e}")
            return None
    return _estimator


def is_boxer_available() -> bool:
    """Boxer 사용 가능 여부"""
    return _BOXER_AVAILABLE
