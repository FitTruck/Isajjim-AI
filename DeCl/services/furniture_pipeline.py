"""
Furniture Analysis Pipeline Service

전체 AI 로직을 통합한 파이프라인 서비스:
1. Firebase Storage URL에서 이미지 가져오기
2. YOLO-World + SAHI로 객체 탐지
3. CLIP으로 세부 유형 분류
4. is_movable 결정
5. SAM2로 마스크 생성
6. SAM-3D로 3D 변환 및 부피 계산
7. DB 규격 대조하여 절대 치수 계산
8. 최종 JSON 응답 반환
"""

import os
import sys
import io
import base64
import tempfile
import uuid
import asyncio
import aiohttp
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

# DeCl 경로 추가
decl_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if decl_path not in sys.path:
    sys.path.insert(0, decl_path)

# DeCl 모듈 임포트
from config import Config
from data.knowledge_base import (
    FURNITURE_DB,
    get_dimensions_for_subtype,
    estimate_size_variant
)
from utils.image_ops import ImageUtils
from utils.volume_calculator import VolumeCalculator, estimate_dimensions_from_aspect_ratio
from models.classifier import ClipClassifier
from models.refiners.ac_refiner import ACRefiner

# SAHI 탐지기 (없으면 표준 탐지기 사용)
try:
    from models.sahi_detector import EnhancedDetector
    HAS_ENHANCED_DETECTOR = True
except ImportError:
    from models.detector import YoloDetector as EnhancedDetector
    HAS_ENHANCED_DETECTOR = False
    print("[FurniturePipeline] Using standard YOLO detector (SAHI not available)")


@dataclass
class DetectedObject:
    """탐지된 객체 정보"""
    id: int
    label: str  # 한글 라벨
    db_key: str  # FURNITURE_DB 키
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

    Firebase Storage URL → YOLO + SAHI → CLIP → SAM2 → SAM-3D → Volume
    """

    def __init__(
        self,
        sam2_api_url: str = "http://localhost:8000",
        enable_3d_generation: bool = True,
        use_sahi: bool = True
    ):
        """
        Args:
            sam2_api_url: SAM2 API 서버 URL
            enable_3d_generation: 3D 생성 활성화 여부
            use_sahi: SAHI 슬라이싱 탐지 사용 여부
        """
        self.sam2_api_url = sam2_api_url.rstrip('/')
        self.enable_3d_generation = enable_3d_generation
        self.use_sahi = use_sahi

        print("[FurniturePipeline] Initializing components...")

        # 컴포넌트 초기화
        self.detector = EnhancedDetector(use_sahi=use_sahi) if HAS_ENHANCED_DETECTOR else EnhancedDetector()
        self.classifier = ClipClassifier()
        self.ac_refiner = ACRefiner(self.classifier)
        self.volume_calculator = VolumeCalculator()

        # 검색 클래스 및 매핑 설정
        self.search_classes = []
        self.class_map = {}
        for key, info in FURNITURE_DB.items():
            for syn in info['synonyms']:
                self.search_classes.append(syn)
                self.class_map[syn] = key

        self.detector.set_classes(self.search_classes)

        print(f"[FurniturePipeline] Initialized with {len(self.search_classes)} search classes")

    async def fetch_image_from_url(self, url: str) -> Optional[Image.Image]:
        """
        URL에서 이미지를 가져옵니다.

        Args:
            url: 이미지 URL (Firebase Storage 또는 일반 URL)

        Returns:
            PIL 이미지
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        print(f"[FurniturePipeline] Failed to fetch image: HTTP {response.status}")
                        return None

                    image_data = await response.read()
                    image = Image.open(io.BytesIO(image_data)).convert("RGB")
                    return image

        except Exception as e:
            print(f"[FurniturePipeline] Error fetching image from {url}: {e}")
            return None

    def fetch_image_from_url_sync(self, url: str) -> Optional[Image.Image]:
        """동기 버전의 이미지 가져오기"""
        import requests
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return None
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"[FurniturePipeline] Error: {e}")
            return None

    def detect_objects(self, image: Image.Image) -> List[DetectedObject]:
        """
        이미지에서 객체를 탐지합니다.

        Args:
            image: PIL 이미지

        Returns:
            탐지된 객체 리스트
        """
        img_w, img_h = image.size
        detected_objects = []

        # YOLO + SAHI 탐지
        results = self.detector.detect_smart(image)

        if results is None:
            return detected_objects

        boxes = results["boxes"]
        scores = results["scores"]
        classes = results["classes"]

        for idx, (box, score, cls_idx) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = map(int, box)

            # 너무 작은 박스 필터링
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            # YOLO 라벨 → DB 키 매핑
            cls_int = int(cls_idx)
            if cls_int not in self.detector.model.names:
                continue  # Skip unknown class indices
            detected_label = self.detector.model.names[cls_int]
            db_key = self.class_map.get(detected_label)

            if not db_key:
                continue

            db_info = FURNITURE_DB[db_key]

            # 객체 크롭
            crop = image.crop((x1, y1, x2, y2))

            # 중심점 계산 (SAM2 prompt용)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 세부 분류 (CLIP 또는 AC Refiner)
            final_info = self._classify_object(
                db_info, db_key, crop,
                [x1, y1, x2, y2], (img_w, img_h)
            )

            obj = DetectedObject(
                id=idx,
                label=final_info['name'],
                db_key=db_key,
                subtype_name=final_info.get('name'),
                bbox=[x1, y1, x2, y2],
                center_point=[center_x, center_y],
                confidence=float(final_info.get('score', score)),
                is_movable=final_info.get('is_movable', True),
                crop_image=crop
            )

            detected_objects.append(obj)

        return detected_objects

    def _classify_object(
        self,
        db_info: Dict,
        db_key: str,
        crop: Image.Image,
        bbox: List[int],
        image_size: Tuple[int, int]
    ) -> Dict:
        """객체 세부 분류"""
        if 'subtypes' in db_info and db_info['subtypes']:
            if db_info.get('check_strategy') == "AC_STRATEGY":
                return self.ac_refiner.refine(
                    bbox, image_size, crop, db_info['subtypes']
                )
            else:
                return self.classifier.classify(crop, db_info['subtypes'])
        else:
            return {
                "name": db_info.get('base_name', db_key),
                "score": 1.0,
                "is_movable": db_info.get('is_movable', True)
            }

    async def generate_mask(self, image: Image.Image, point: List[float]) -> Optional[str]:
        """
        SAM2 API를 호출하여 마스크를 생성합니다.

        Args:
            image: 원본 이미지
            point: [x, y] 중심점

        Returns:
            Base64 인코딩된 마스크
        """
        try:
            # 이미지를 base64로 인코딩
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # SAM2 API 호출
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
                        print(f"[FurniturePipeline] SAM2 API error: {response.status}")
                        return None

                    result = await response.json()

                    if result.get("success") and result.get("masks"):
                        # 가장 높은 점수의 마스크 선택
                        masks = result["masks"]
                        best_mask = max(masks, key=lambda m: m.get("score", 0))
                        return best_mask.get("mask")

                    return None

        except Exception as e:
            print(f"[FurniturePipeline] Error generating mask: {e}")
            return None

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
            {
                "task_id": str,
                "ply_url": str,
                "glb_url": str,
                "gif_url": str
            }
        """
        if not self.enable_3d_generation:
            return None

        try:
            # 이미지를 base64로 인코딩
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            async with aiohttp.ClientSession() as session:
                # 3D 생성 요청
                async with session.post(
                    f"{self.sam2_api_url}/generate-3d",
                    json={
                        "image": image_b64,
                        "mask": mask_b64,
                        "seed": seed
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        return None

                    result = await response.json()
                    task_id = result.get("task_id")

                    if not task_id:
                        return None

                # 결과 폴링 (최대 10분)
                for _ in range(120):  # 5초 * 120 = 10분
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
                            print(f"[FurniturePipeline] 3D generation failed: {status.get('error')}")
                            return None

                return None

        except Exception as e:
            print(f"[FurniturePipeline] Error in 3D generation: {e}")
            return None

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

        # PLY 또는 GLB에서 상대 치수 계산
        if ply_path and os.path.exists(ply_path):
            relative_dims = self.volume_calculator.calculate_from_ply(ply_path)
        elif glb_path and os.path.exists(glb_path):
            relative_dims = self.volume_calculator.calculate_from_glb(glb_path)

        if relative_dims is None:
            return None, None

        # DB에서 참조 치수 가져오기
        ref_dims = get_dimensions_for_subtype(obj.db_key, obj.subtype_name)

        if ref_dims is None:
            return relative_dims, None

        # 비율로 사이즈 변형 추정 (variants가 있는 경우)
        aspect_ratio = relative_dims.get("ratio", {})
        variant_name, matched_dims = estimate_size_variant(
            obj.db_key, obj.subtype_name, aspect_ratio
        )

        if matched_dims:
            ref_dims = matched_dims

        # 절대 치수 계산
        absolute_dims = self.volume_calculator.scale_to_absolute(relative_dims, ref_dims)

        return relative_dims, absolute_dims

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
            # 1. 이미지 가져오기
            image = await self.fetch_image_from_url(image_url)
            if image is None:
                result.status = "failed"
                result.error = "Failed to fetch image"
                return result

            # 2. 객체 탐지
            detected_objects = self.detect_objects(image)

            # 3. 각 객체에 대해 마스크 및 3D 생성
            for obj in detected_objects:
                # 3a. SAM2 마스크 생성
                if enable_mask:
                    mask_b64 = await self.generate_mask(image, obj.center_point)
                    obj.mask_base64 = mask_b64

                # 3b. SAM-3D 3D 생성
                if enable_3d and self.enable_3d_generation and obj.mask_base64:
                    gen_result = await self.generate_3d(image, obj.mask_base64)

                    if gen_result:
                        obj.glb_url = gen_result.get("mesh_url")

                        # 임시 파일로 저장하여 부피 계산
                        if gen_result.get("ply_b64"):
                            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                                tmp.write(base64.b64decode(gen_result["ply_b64"]))
                                ply_path = tmp.name

                            rel_dims, abs_dims = self.calculate_dimensions(obj, ply_path=ply_path)
                            obj.relative_dimensions = rel_dims
                            obj.absolute_dimensions = abs_dims

                            os.unlink(ply_path)

            # 4. 결과 집계
            result.objects = detected_objects
            result.total_volume_liters = sum(
                o.absolute_dimensions.get("volume_liters", 0)
                for o in detected_objects
                if o.absolute_dimensions
            )
            result.movable_volume_liters = sum(
                o.absolute_dimensions.get("volume_liters", 0)
                for o in detected_objects
                if o.absolute_dimensions and o.is_movable
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
        max_concurrent: int = 3
    ) -> List[PipelineResult]:
        """
        여러 이미지를 동시에 처리합니다.

        Args:
            image_urls: Firebase Storage 이미지 URL 리스트 (5~10개)
            enable_mask: SAM2 마스크 생성 활성화
            enable_3d: SAM-3D 3D 생성 활성화
            max_concurrent: 최대 동시 처리 수

        Returns:
            PipelineResult 리스트
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(url: str):
            async with semaphore:
                return await self.process_single_image(url, enable_mask, enable_3d)

        tasks = [process_with_semaphore(url) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외를 PipelineResult로 변환
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
        """
        결과를 JSON 응답 형식으로 변환합니다.

        Returns:
            {
                "objects": [...],
                "summary": {...}
            }
        """
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

        # 요약 정보
        total_volume = sum(r.total_volume_liters for r in results)
        movable_volume = sum(r.movable_volume_liters for r in results)
        total_objects = sum(len(r.objects) for r in results)
        movable_objects = sum(
            len([o for o in r.objects if o.is_movable])
            for r in results
        )

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
