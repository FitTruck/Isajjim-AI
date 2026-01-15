"""
Stage 3: CLIP 세부 분류

CLIP(Contrastive Language-Image Pre-training) 모델을 활용하여
YOLO가 탐지한 객체의 세부 유형(subtype)을 분류합니다.

예: "침대" → "퀸 사이즈 침대" / "킹 사이즈 침대"
"""

import os
import sys
from typing import Dict, List, Optional
from PIL import Image

# AI module import
from ai.config import Config

# Type hint
from typing import TYPE_CHECKING

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("[ClipClassifier] transformers/torch not installed")


class ClipClassifier:
    """
    CLIP 기반 세부 분류기

    AI Logic Step 3: 객체 세부 유형 분류

    YOLO가 탐지한 객체 크롭 이미지를 받아서
    가능한 세부 유형(subtypes) 중 가장 적합한 것을 선택합니다.
    """

    def __init__(self, model_id: str = None, device_id: Optional[int] = None):
        """
        Args:
            model_id: CLIP 모델 ID (None이면 Config에서 가져옴)
            device_id: GPU 디바이스 ID (None이면 기본값 사용)
        """
        self.model_id = model_id or Config.CLIP_MODEL_ID

        # Multi-GPU 지원: 디바이스 설정
        self.device_id = device_id
        self._device = Config.get_device(device_id)

        self.model = None
        self.processor = None

        self._load_model()

    def _load_model(self):
        """모델 로드"""
        if not HAS_CLIP:
            print("[ClipClassifier] CLIP not available")
            return

        print(f"[ClipClassifier] Loading CLIP on {self._device}: {self.model_id}")
        try:
            self.model = CLIPModel.from_pretrained(
                self.model_id,
                use_safetensors=True
            ).to(self._device)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            print(f"[ClipClassifier] CLIP loaded successfully on {self._device}")
        except Exception as e:
            print(f"[ClipClassifier] Failed to load CLIP: {e}")

    def classify(
        self,
        crop_image: Image.Image,
        candidates: List[Dict]
    ) -> Optional[Dict]:
        """
        크롭 이미지를 후보 유형들과 비교하여 분류합니다.

        Args:
            crop_image: 객체 크롭 이미지 (PIL)
            candidates: 후보 리스트
                [
                    {"name": "퀸 사이즈 침대", "prompt": "a queen size bed", ...},
                    {"name": "킹 사이즈 침대", "prompt": "a king size bed", ...},
                    ...
                ]

        Returns:
            가장 적합한 후보 딕셔너리 + score 필드 추가
            예: {"name": "퀸 사이즈 침대", "prompt": "...", "score": 0.85, ...}
        """
        if self.model is None or not candidates:
            return None

        # 프롬프트 추출
        prompts = [c['prompt'] for c in candidates]

        # CLIP 입력 준비
        inputs = self.processor(
            text=prompts,
            images=crop_image,
            return_tensors="pt",
            padding=True
        ).to(self._device)

        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

        # 최고 점수 후보 선택
        best_idx = probs.argmax().item()
        score = probs[0][best_idx].item()

        result = candidates[best_idx].copy()
        result['score'] = score
        result['all_scores'] = {
            c['name']: float(probs[0][i].item())
            for i, c in enumerate(candidates)
        }

        return result

    def classify_with_head_crop(
        self,
        crop_image: Image.Image,
        candidates: List[Dict],
        head_ratio: float = 0.25
    ) -> Optional[Dict]:
        """
        이미지 상단 일부만 사용하여 분류합니다.

        타워형 에어컨 vs 옷장 같이 상단 특징이 중요한 경우 사용합니다.

        Args:
            crop_image: 객체 크롭 이미지
            candidates: 후보 리스트
            head_ratio: 상단에서 사용할 비율 (0.25 = 상단 25%)

        Returns:
            분류 결과
        """
        w, h = crop_image.size

        if h < 50:  # 너무 작으면 전체 사용
            return self.classify(crop_image, candidates)

        # 상단 부분만 크롭
        head_crop = crop_image.crop((0, 0, w, int(h * head_ratio)))

        return self.classify(head_crop, candidates)

    def classify_multi_view(
        self,
        crop_images: List[Image.Image],
        candidates: List[Dict]
    ) -> Optional[Dict]:
        """
        여러 크롭 이미지의 점수를 평균하여 분류합니다.

        같은 객체의 여러 뷰가 있는 경우 더 정확한 분류가 가능합니다.

        Args:
            crop_images: 크롭 이미지 리스트
            candidates: 후보 리스트

        Returns:
            분류 결과
        """
        if self.model is None or not candidates or not crop_images:
            return None

        all_probs = []

        for crop in crop_images:
            prompts = [c['prompt'] for c in candidates]
            inputs = self.processor(
                text=prompts,
                images=crop,
                return_tensors="pt",
                padding=True
            ).to(self._device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                all_probs.append(probs)

        # 평균 확률 계산
        avg_probs = torch.stack(all_probs).mean(dim=0)
        best_idx = avg_probs.argmax().item()
        score = avg_probs[0][best_idx].item()

        result = candidates[best_idx].copy()
        result['score'] = score

        return result


class ACRefiner:
    """
    에어컨 특수 분류기

    타워형 에어컨과 옷장을 구분하기 위한 특수 로직을 적용합니다.
    """

    def __init__(self, classifier: ClipClassifier):
        """
        Args:
            classifier: ClipClassifier 인스턴스
        """
        self.classifier = classifier

    def refine(
        self,
        bbox: List[int],
        image_size: tuple,
        crop: Image.Image,
        subtypes: List[Dict]
    ) -> Dict:
        """
        에어컨 타입을 정제합니다.

        Args:
            bbox: [x1, y1, x2, y2]
            image_size: (width, height)
            crop: 크롭 이미지
            subtypes: 후보 서브타입 리스트

        Returns:
            분류 결과
        """
        x1, y1, x2, y2 = bbox
        img_w, img_h = image_size
        box_w = x2 - x1
        box_h = y2 - y1

        # 종횡비로 타워형 여부 판단
        aspect_ratio = box_h / box_w if box_w > 0 else 1

        # 타워형은 세로가 길고 벽면 가까이 있음
        is_likely_tower = (
            aspect_ratio > 2.0 and  # 세로로 긴 형태
            (x1 < img_w * 0.1 or x2 > img_w * 0.9)  # 좌우 벽면 근처
        )

        if is_likely_tower:
            # 타워형 에어컨으로 판단될 가능성이 높으면
            # 상단 특징으로 재분류
            return self.classifier.classify_with_head_crop(crop, subtypes)
        else:
            # 일반 분류
            return self.classifier.classify(crop, subtypes)
