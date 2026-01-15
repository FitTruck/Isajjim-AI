"""
Furniture Analysis Pipeline

전체 AI 로직을 통합한 파이프라인 오케스트레이터:

[DeCl Stages - 1~4]
1. Firebase Storage URL에서 이미지 가져오기
2. YOLO-World + SAHI로 객체 탐지
3. CLIP으로 세부 유형 분류
4. DB 대조하여 is_movable 결정

[External API - 5~9]
5. SAM2로 마스크 생성
6. SAM-3D로 3D 변환
7. 부피/치수 계산
8. DB 규격 대조하여 절대 치수 계산
9. 최종 JSON 응답 반환
"""

import os
import sys
import io
import base64
import tempfile
import uuid
import asyncio
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np
from dataclasses import dataclass, field

# AI processors import
from ai.processors import (
    ImageFetcher,
    YoloWorldDetector,
    ClipClassifier,
    ACRefiner,
    MovabilityChecker,
    VolumeCalculator,
    SAM3DConverter
)

# GPU pool manager import
from ai.gpu import GPUPoolManager, get_gpu_pool

# Config import
from ai.config import Config

# Knowledge base import
from ai.data.knowledge_base import FURNITURE_DB

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class DetectedObject:
    """탐지된 객체 정보"""
    id: int
    label: str                              # 한글 라벨
    db_key: str                             # FURNITURE_DB 키
    subtype_name: Optional[str] = None
    bbox: List[int] = field(default_factory=list)
    center_point: List[float] = field(default_factory=list)
    confidence: float = 0.0
    is_movable: bool = True
    crop_image: Optional[Image.Image] = None
    mask_base64: Optional[str] = None

    # 3D 정보 (SAM-3D 처리 후)
    ply_url: Optional[str] = None
    glb_url: Optional[str] = None
    gif_url: Optional[str] = None

    # 치수 정보
    relative_dimensions: Optional[Dict] = None
    absolute_dimensions: Optional[Dict] = None


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    image_id: str
    image_url: str
    objects: List[DetectedObject] = field(default_factory=list)
    total_volume_liters: float = 0.0
    movable_volume_liters: float = 0.0
    processing_time_seconds: float = 0.0
    status: str = "pending"
    error: Optional[str] = None


class FurniturePipeline:
    """
    가구 분석 통합 파이프라인

    AI Logic Stages:
        Stage 1: ImageFetcher - Firebase URL → PIL Image
        Stage 2: YoloWorldDetector - 객체 탐지 (바운딩 박스, 1차 클래스)
        Stage 3: ClipClassifier - 세부 유형 분류
        Stage 4: MovabilityChecker - DB 대조 → is_movable 결정

    External API:
        SAM2 API - 마스크 생성
        SAM-3D API - 3D 모델 생성 및 부피 계산
    """

    def __init__(
        self,
        sam2_api_url: str = "http://localhost:8000",
        enable_3d_generation: bool = True,
        use_sahi: bool = True,
        device_id: Optional[int] = None,
        gpu_pool: Optional[GPUPoolManager] = None
    ):
        """
        Args:
            sam2_api_url: SAM2/SAM-3D API 서버 URL
            enable_3d_generation: 3D 생성 활성화 여부
            use_sahi: SAHI 슬라이싱 탐지 사용 여부
            device_id: GPU 디바이스 ID (None이면 기본값 사용)
            gpu_pool: GPU 풀 매니저 (Multi-GPU 처리용)
        """
        self.sam2_api_url = sam2_api_url.rstrip('/')
        self.enable_3d_generation = enable_3d_generation

        # Multi-GPU 지원
        self.device_id = device_id
        self._device = Config.get_device(device_id)
        self.gpu_pool = gpu_pool

        print(f"[FurniturePipeline] Initializing on device: {self._device}")

        # Stage 1: 이미지 가져오기
        self.fetcher = ImageFetcher()

        # Stage 2: YOLO-World 탐지 - device_id 전달
        self.detector = YoloWorldDetector(use_sahi=use_sahi, device_id=device_id)

        # Stage 3: CLIP 분류 - device_id 전달
        self.classifier = ClipClassifier(device_id=device_id)
        self.ac_refiner = ACRefiner(self.classifier)

        # Stage 4: is_movable 판단
        self.movability_checker = MovabilityChecker()

        # Volume 계산기
        self.volume_calculator = VolumeCalculator()

        # SAM-3D 변환기 - device_id 전달
        self.sam3d_converter = SAM3DConverter(device_id=device_id)

        # 검색 클래스 설정
        search_classes = self.movability_checker.get_search_classes()
        self.detector.set_classes(search_classes)

        # 클래스 매핑 (동의어 → DB 키)
        self.class_map = self.movability_checker.class_map

        print(f"[FurniturePipeline] Initialized with {len(search_classes)} search classes on {self._device}")

    # =========================================================================
    # Stage 1: 이미지 가져오기
    # =========================================================================

    async def fetch_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Stage 1: Firebase URL에서 이미지 가져오기"""
        return await self.fetcher.fetch_async(url)

    def fetch_image_from_url_sync(self, url: str) -> Optional[Image.Image]:
        """Stage 1 (동기): Firebase URL에서 이미지 가져오기"""
        return self.fetcher.fetch_sync(url)

    # =========================================================================
    # Stage 2-4: 객체 탐지 → 분류 → is_movable 결정
    # =========================================================================

    def detect_objects(self, image: Image.Image) -> List[DetectedObject]:
        """
        Stage 2-4 통합: 이미지에서 객체를 탐지하고 분류합니다.

        Args:
            image: PIL 이미지

        Returns:
            DetectedObject 리스트
        """
        img_w, img_h = image.size
        detected_objects = []

        # Stage 2: YOLO-World 탐지
        results = self.detector.detect_smart(image)

        if results is None or len(results["boxes"]) == 0:
            return detected_objects

        boxes = results["boxes"]
        scores = results["scores"]
        classes = results["classes"]

        for idx, (box, score, cls_idx) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = map(int, box)

            # 너무 작은 박스 필터링
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            # 클래스 라벨 가져오기
            cls_int = int(cls_idx)
            detected_label = self.detector.get_label_for_class(cls_int)
            if not detected_label:
                continue

            # DB 키 매핑
            db_key = self.class_map.get(detected_label.lower())
            if not db_key:
                continue

            # 객체 크롭
            crop = image.crop((x1, y1, x2, y2))

            # 중심점 계산
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Stage 3-4: 분류 및 is_movable 결정
            final_info = self._classify_and_check_movability(
                db_key, crop, [x1, y1, x2, y2], (img_w, img_h), float(score)
            )

            obj = DetectedObject(
                id=idx,
                label=final_info['label'],
                db_key=db_key,
                subtype_name=final_info.get('subtype_name'),
                bbox=[x1, y1, x2, y2],
                center_point=[center_x, center_y],
                confidence=final_info['confidence'],
                is_movable=final_info['is_movable'],
                crop_image=crop
            )

            detected_objects.append(obj)

        return detected_objects

    def _classify_and_check_movability(
        self,
        db_key: str,
        crop: Image.Image,
        bbox: List[int],
        image_size: Tuple[int, int],
        detection_score: float
    ) -> Dict:
        """
        Stage 3-4: 객체 분류 및 is_movable 판단

        Args:
            db_key: DB 키
            crop: 크롭 이미지
            bbox: 바운딩 박스
            image_size: 원본 이미지 크기
            detection_score: 탐지 신뢰도

        Returns:
            {
                "label": str,
                "subtype_name": str,
                "is_movable": bool,
                "confidence": float
            }
        """
        db_info = FURNITURE_DB.get(db_key, {})
        subtypes = db_info.get('subtypes', [])

        # Stage 3: CLIP 분류
        classification_result = None
        if subtypes:
            if db_info.get('check_strategy') == "AC_STRATEGY":
                classification_result = self.ac_refiner.refine(
                    bbox, image_size, crop, subtypes
                )
            else:
                classification_result = self.classifier.classify(crop, subtypes)

        # Stage 4: is_movable 판단
        if classification_result:
            movability = self.movability_checker.check_with_classification(
                db_key, classification_result
            )
        else:
            movability = self.movability_checker.check(db_key)

        return {
            "label": movability.label,
            "subtype_name": movability.subtype_name,
            "is_movable": movability.is_movable,
            "confidence": classification_result.get('score', detection_score) if classification_result else detection_score
        }

    # =========================================================================
    # External API: SAM2 마스크 생성
    # =========================================================================

    async def generate_mask(self, image: Image.Image, point: List[float]) -> Optional[str]:
        """
        SAM2 API를 호출하여 마스크를 생성합니다.

        Args:
            image: 원본 이미지
            point: [x, y] 중심점

        Returns:
            Base64 인코딩된 마스크
        """
        if not HAS_AIOHTTP:
            return None

        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.sam2_api_url}/segment",
                    json={
                        "image": image_b64,
                        "x": point[0],
                        "y": point[1],
                        "multimask_output": True
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        return None

                    result = await response.json()

                    if result.get("success") and result.get("masks"):
                        masks = result["masks"]
                        best_mask = max(masks, key=lambda m: m.get("score", 0))
                        return best_mask.get("mask")

                    return None

        except Exception as e:
            print(f"[FurniturePipeline] Mask generation error: {e}")
            return None

    # =========================================================================
    # External API: SAM-3D 3D 생성
    # =========================================================================

    async def generate_3d(
        self,
        image: Image.Image,
        mask_b64: str,
        seed: int = 42
    ) -> Optional[Dict]:
        """
        SAM-3D API를 호출하여 3D 모델을 생성합니다.

        Args:
            image: 원본 이미지
            mask_b64: Base64 인코딩된 마스크

        Returns:
            {"task_id": str, "ply_b64": str, "glb_b64": str, ...}
        """
        if not self.enable_3d_generation or not HAS_AIOHTTP:
            return None

        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            async with aiohttp.ClientSession() as session:
                # 3D 생성 요청
                async with session.post(
                    f"{self.sam2_api_url}/generate-3d",
                    json={"image": image_b64, "mask": mask_b64, "seed": seed},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        return None

                    result = await response.json()
                    task_id = result.get("task_id")
                    if not task_id:
                        return None

                # 결과 폴링
                for _ in range(120):
                    await asyncio.sleep(5)

                    async with session.get(
                        f"{self.sam2_api_url}/generate-3d-status/{task_id}",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as status_response:
                        if status_response.status != 200:
                            continue

                        status = await status_response.json()

                        if status.get("status") == "completed":
                            return {
                                "task_id": task_id,
                                "ply_b64": status.get("ply_b64"),
                                "glb_b64": status.get("mesh_b64"),
                                "gif_b64": status.get("gif_b64"),
                                "mesh_url": status.get("mesh_url")
                            }
                        elif status.get("status") == "failed":
                            return None

                return None

        except Exception as e:
            print(f"[FurniturePipeline] 3D generation error: {e}")
            return None

    # =========================================================================
    # 치수 계산
    # =========================================================================

    def calculate_dimensions(
        self,
        obj: DetectedObject,
        ply_path: Optional[str] = None,
        glb_path: Optional[str] = None
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        3D 모델에서 치수를 계산하고 DB와 대조합니다.

        Returns:
            (relative_dimensions, absolute_dimensions)
        """
        relative_dims = None

        if ply_path and os.path.exists(ply_path):
            relative_dims = self.volume_calculator.calculate_from_ply(ply_path)
        elif glb_path and os.path.exists(glb_path):
            relative_dims = self.volume_calculator.calculate_from_glb(glb_path)

        if relative_dims is None:
            return None, None

        # DB에서 참조 치수 가져오기
        ref_dims = self.movability_checker.get_reference_dimensions(
            obj.db_key,
            obj.subtype_name,
            relative_dims.get("ratio")
        )

        if ref_dims is None:
            return relative_dims, None

        # 절대 치수 계산
        absolute_dims = self.volume_calculator.scale_to_absolute(relative_dims, ref_dims)

        return relative_dims, absolute_dims

    # =========================================================================
    # 통합 처리
    # =========================================================================

    async def process_single_image(
        self,
        image_url: str,
        enable_mask: bool = True,
        enable_3d: bool = True
    ) -> PipelineResult:
        """
        단일 이미지를 처리합니다.

        Args:
            image_url: Firebase Storage 이미지 URL
            enable_mask: SAM2 마스크 생성 활성화
            enable_3d: SAM-3D 3D 생성 활성화

        Returns:
            PipelineResult
        """
        import time
        start_time = time.time()

        result = PipelineResult(
            image_id=str(uuid.uuid4()),
            image_url=image_url,
            status="processing"
        )

        try:
            # Stage 1: 이미지 가져오기
            image = await self.fetch_image_from_url(image_url)
            if image is None:
                result.status = "failed"
                result.error = "Failed to fetch image"
                return result

            # Stage 2-4: 객체 탐지 및 분류
            detected_objects = self.detect_objects(image)

            # Stage 5-8: 각 객체에 대해 마스크 및 3D 생성
            for obj in detected_objects:
                if enable_mask:
                    mask_b64 = await self.generate_mask(image, obj.center_point)
                    obj.mask_base64 = mask_b64

                if enable_3d and self.enable_3d_generation and obj.mask_base64:
                    gen_result = await self.generate_3d(image, obj.mask_base64)

                    if gen_result:
                        obj.glb_url = gen_result.get("mesh_url")

                        if gen_result.get("ply_b64"):
                            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                                tmp.write(base64.b64decode(gen_result["ply_b64"]))
                                ply_path = tmp.name

                            rel_dims, abs_dims = self.calculate_dimensions(obj, ply_path=ply_path)
                            obj.relative_dimensions = rel_dims
                            obj.absolute_dimensions = abs_dims

                            os.unlink(ply_path)

            # 결과 집계
            result.objects = detected_objects
            result.total_volume_liters = sum(
                o.absolute_dimensions.get("volume_liters", 0)
                for o in detected_objects if o.absolute_dimensions
            )
            result.movable_volume_liters = sum(
                o.absolute_dimensions.get("volume_liters", 0)
                for o in detected_objects if o.absolute_dimensions and o.is_movable
            )
            result.status = "completed"

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            import traceback
            traceback.print_exc()

        result.processing_time_seconds = time.time() - start_time
        return result

    async def process_multiple_images(
        self,
        image_urls: List[str],
        enable_mask: bool = True,
        enable_3d: bool = True,
        max_concurrent: Optional[int] = None
    ) -> List[PipelineResult]:
        """
        여러 이미지를 GPU에 분배하여 병렬 처리합니다.

        Multi-GPU 모드에서는 각 이미지를 다른 GPU에 라운드로빈 방식으로 분배합니다.
        """
        # GPU 풀 가져오기
        pool = self.gpu_pool or get_gpu_pool()

        # max_concurrent가 None이면 GPU 수만큼 설정
        if max_concurrent is None:
            max_concurrent = len(pool.gpu_ids) if pool.gpu_ids else 3

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_gpu(url: str) -> PipelineResult:
            """GPU를 할당받아 이미지 처리"""
            async with semaphore:
                try:
                    async with pool.gpu_context(task_id=url[:50]) as gpu_id:
                        # 해당 GPU용 파이프라인 생성
                        # Note: 이미 초기화된 self를 재사용하거나 새로 생성
                        if self.device_id == gpu_id:
                            # 같은 GPU면 self 재사용
                            return await self.process_single_image(url, enable_mask, enable_3d)
                        else:
                            # 다른 GPU면 새 파이프라인 생성
                            pipeline = FurniturePipeline(
                                sam2_api_url=self.sam2_api_url,
                                enable_3d_generation=self.enable_3d_generation,
                                device_id=gpu_id,
                                gpu_pool=pool
                            )
                            return await pipeline.process_single_image(url, enable_mask, enable_3d)
                except Exception as e:
                    return PipelineResult(
                        image_id=str(uuid.uuid4()),
                        image_url=url,
                        status="failed",
                        error=str(e)
                    )

        tasks = [process_with_gpu(url) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                processed_results.append(PipelineResult(
                    image_id=str(uuid.uuid4()),
                    image_url=image_urls[i],
                    status="failed",
                    error=str(r)
                ))
            else:
                processed_results.append(r)

        return processed_results

    def to_json_response(self, results: List[PipelineResult]) -> Dict:
        """결과를 JSON 응답 형식으로 변환합니다."""
        all_objects = []

        for result in results:
            for obj in result.objects:
                obj_data = {
                    "label": obj.label,
                    "is_movable": obj.is_movable,
                    "confidence": round(obj.confidence, 3),
                    "bbox": obj.bbox,
                    "center_point": obj.center_point
                }

                if obj.absolute_dimensions:
                    dims = obj.absolute_dimensions
                    obj_data.update({
                        "width": dims.get("width", 0),
                        "depth": dims.get("depth", 0),
                        "height": dims.get("height", 0),
                        "volume": dims.get("volume_liters", 0),
                        "ratio": dims.get("ratio", {"w": 1, "h": 1, "d": 1})
                    })

                if obj.glb_url:
                    obj_data["mesh_url"] = obj.glb_url

                all_objects.append(obj_data)

        total_volume = sum(r.total_volume_liters for r in results)
        movable_volume = sum(r.movable_volume_liters for r in results)
        total_objects = sum(len(r.objects) for r in results)
        movable_objects = sum(len([o for o in r.objects if o.is_movable]) for r in results)

        return {
            "objects": all_objects,
            "summary": {
                "total_objects": total_objects,
                "movable_objects": movable_objects,
                "fixed_objects": total_objects - movable_objects,
                "total_volume_liters": round(total_volume, 2),
                "movable_volume_liters": round(movable_volume, 2),
                "images_processed": len(results),
                "images_failed": sum(1 for r in results if r.status == "failed")
            }
        }
