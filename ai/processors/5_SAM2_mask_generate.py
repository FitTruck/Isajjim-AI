"""
Step 5: SAM2 Mask Generation

SAM 2 (Segment Anything Model 2)를 사용하여 객체 마스크를 생성합니다.
- 단일 포인트 세그멘테이션: segment()
- 다중 포인트 세그멘테이션: segment_binary()
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Optional, Tuple, Any


class SAM2MaskGenerator:
    """
    SAM 2를 사용한 마스크 생성기

    Usage:
        generator = SAM2MaskGenerator(model, processor, device)
        masks = generator.segment(image, x, y)
        masked_image = generator.segment_binary(image, points)
    """

    def __init__(self, model, processor, device):
        """
        Args:
            model: SAM2Model 인스턴스
            processor: Sam2Processor 인스턴스
            device: torch.device
        """
        self.model = model
        self.processor = processor
        self.device = device

    def segment(
        self,
        image: Image.Image,
        x: float,
        y: float,
        multimask_output: bool = True,
        mask_threshold: float = 0.0,
        invert_mask: bool = False
    ) -> Dict[str, Any]:
        """
        단일 포인트 기반 세그멘테이션

        Args:
            image: PIL 이미지
            x: X 좌표
            y: Y 좌표
            multimask_output: 다중 마스크 출력 여부
            mask_threshold: 마스크 임계값
            invert_mask: 마스크 반전 여부

        Returns:
            {
                "masks": [{"mask": base64, "mask_shape": tuple, "score": float}, ...],
                "input_point": [x, y],
                "image_shape": [height, width]
            }
        """
        import torch

        image_np = np.array(image)

        # 입력 포인트 형식: [[[[x, y]]]]
        input_points = [[[[x, y]]]]
        input_labels = [[[1]]]  # 1 = positive click

        # 프로세서 처리
        inputs = self.processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)

        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 후처리
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]

        # 스코어 추출
        scores = (
            outputs.iou_preds[0].cpu().numpy().tolist()
            if hasattr(outputs, "iou_preds")
            else [0.95] * masks.shape[0]
        )

        # 마스크 리스트 생성
        mask_list = []
        for i in range(masks.shape[0]):
            mask = masks[i].numpy()
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                mask = mask[0] if mask.ndim > 2 else mask

            # 임계값 적용
            mask = (mask > mask_threshold).astype(np.uint8) * 255

            # 형태학적 스무딩
            mask = self._apply_morphology(mask)

            # 마스크 반전
            if invert_mask:
                mask = 255 - mask

            # base64 인코딩
            mask_base64 = self._encode_mask(mask)

            mask_list.append({
                "mask": mask_base64,
                "mask_shape": mask.shape,
                "score": float(scores[i]) if i < len(scores) else 0.95
            })

        return {
            "masks": mask_list,
            "input_point": [x, y],
            "image_shape": [image.height, image.width]
        }

    def segment_binary(
        self,
        image: Image.Image,
        points: List[Dict[str, float]],
        previous_mask: Optional[str] = None,
        mask_threshold: float = 0.0
    ) -> Tuple[str, float]:
        """
        다중 포인트 기반 세그멘테이션 (마스크 유니온)

        Args:
            image: PIL 이미지
            points: [{"x": float, "y": float}, ...]
            previous_mask: 이전 마스크 base64 (선택적)
            mask_threshold: 마스크 임계값

        Returns:
            (mask_base64, score)
        """
        import torch

        image_pil_array = np.array(image)

        # 각 포인트에서 마스크 수집
        all_masks = []
        best_score = 0.0

        for point in points:
            # 단일 포인트 처리
            input_points = [[[[point["x"], point["y"]]]]]
            input_labels = [[[1]]]

            inputs = self.processor(
                images=image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )[0]

            scores = (
                outputs.iou_preds[0].cpu().numpy()
                if hasattr(outputs, "iou_preds")
                else np.array([0.95] * masks.shape[0])
            )

            # 최고 점수 마스크 선택
            best_mask_idx = np.argmax(scores)
            point_mask = masks[best_mask_idx].numpy()

            point_mask = np.squeeze(point_mask)
            if point_mask.ndim != 2:
                point_mask = point_mask[0] if point_mask.ndim > 2 else point_mask

            point_mask = (point_mask > mask_threshold).astype(np.uint8) * 255

            all_masks.append(point_mask)
            best_score = max(best_score, float(scores[best_mask_idx]))

        # 모든 마스크 유니온
        mask = all_masks[0].copy()
        for i in range(1, len(all_masks)):
            mask = np.maximum(mask, all_masks[i])

        # 이전 마스크 추가
        if previous_mask:
            try:
                mask_data = base64.b64decode(previous_mask)
                prev_mask_pil = Image.open(io.BytesIO(mask_data)).convert("L")
                prev_mask_array = np.array(prev_mask_pil)
                mask = np.maximum(mask, prev_mask_array)
            except Exception:
                pass

        mask = (mask > mask_threshold).astype(np.uint8) * 255

        # 형태학적 스무딩 (가벼운 버전)
        mask = self._apply_morphology_light(mask)

        # 반전 체크 (평균 > 127이면 반전)
        if mask.mean() > 127:
            mask = 255 - mask

        # 크기 맞춤
        if image_pil_array.shape[:2] != mask.shape:
            mask = cv2.resize(
                mask,
                (image_pil_array.shape[1], image_pil_array.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # 마스크 적용하여 이미지 생성
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
        masked_image = (image_pil_array.astype(np.float32) * mask_3ch).astype(np.uint8)

        # base64 인코딩
        masked_image_pil = Image.fromarray(masked_image, mode="RGB")
        buffer = io.BytesIO()
        masked_image_pil.save(buffer, format="PNG")
        buffer.seek(0)
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return mask_base64, best_score

    def segment_from_center(
        self,
        image: Image.Image,
        center_point: Tuple[float, float]
    ) -> Optional[str]:
        """
        중심점에서 마스크 생성 (FurniturePipeline용)

        Args:
            image: PIL 이미지
            center_point: (x, y) 중심점

        Returns:
            마스크 base64 또는 None
        """
        try:
            result = self.segment(
                image,
                center_point[0],
                center_point[1],
                multimask_output=True,
                mask_threshold=0.0
            )

            if result["masks"]:
                # 가장 높은 점수의 마스크 반환
                best_mask = max(result["masks"], key=lambda m: m["score"])
                return best_mask["mask"]

            return None
        except Exception as e:
            print(f"[SAM2MaskGenerator] segment_from_center error: {e}")
            return None

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """형태학적 스무딩 적용 (표준 버전)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = (mask > 127).astype(np.uint8) * 255
        return mask

    def _apply_morphology_light(self, mask: np.ndarray) -> np.ndarray:
        """형태학적 스무딩 적용 (가벼운 버전 - 다중 포인트용)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = (mask > 127).astype(np.uint8) * 255
        return mask

    def _encode_mask(self, mask: np.ndarray) -> str:
        """마스크를 base64로 인코딩"""
        mask_image = Image.fromarray(mask, mode="L")
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
