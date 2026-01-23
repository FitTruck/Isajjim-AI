"""
Furniture Analysis Pipeline

전체 AI 로직을 통합한 파이프라인 오케스트레이터:

[V2 파이프라인 - CLIP/SAHI/SAM2 제거]
1. Firebase Storage URL에서 이미지 가져오기
2. YOLOE-seg로 객체 탐지 (bbox + class + 세그멘테이션 마스크)
3. DB 대조하여 is_movable 결정 (YOLO 클래스로 직접 매칭)
4. YOLOE-seg 마스크를 SAM-3D에 직접 전달 (SAM2 제거)
5. SAM3D로 3D 변환
6. 객체별 부피 계산 (trimesh)
7. 상대적 부피 계산 후 JSON 응답

변경 이유 (2024-01 테스트 결과):
- YOLOE-seg 마스크가 SAM2보다 객체 전체를 더 정확하게 커버
- SAM2 API 호출 제거로 latency 감소
- 파이프라인 단순화
"""

import os
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
    YoloDetector,
    MovabilityChecker,
    VolumeCalculator,
    SAM3DConverter
)

# GPU pool manager import
from ai.gpu import GPUPoolManager, get_gpu_pool
from ai.gpu import SAM3DWorkerPool, get_sam3d_worker_pool

# Config import
from ai.config import Config

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
    subtype_name: Optional[str] = None      # CLIP 제거로 항상 None
    bbox: List[int] = field(default_factory=list)
    center_point: List[float] = field(default_factory=list)
    confidence: float = 0.0
    crop_image: Optional[Image.Image] = None
    mask_base64: Optional[str] = None

    # 세그멘테이션 마스크 (YOLOE-seg에서 출력)
    yolo_mask: Optional[np.ndarray] = None

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
    processing_time_seconds: float = 0.0
    status: str = "pending"
    error: Optional[str] = None
    user_image_id: Optional[int] = None  # 사용자 지정 이미지 ID (TDD Section 4.1)


class FurniturePipeline:
    """
    가구 분석 통합 파이프라인 V2

    [V2 파이프라인 - SAM2 제거]
    AI Logic Stages:
        Stage 1: ImageFetcher - Firebase URL → PIL Image
        Stage 2: YoloDetector - 객체 탐지 (bbox, class, 세그멘테이션 마스크)
        Stage 3: MovabilityChecker - DB 대조 → is_movable 결정
        Stage 4: YOLOE-seg 마스크 → SAM-3D 직접 전달

    External API:
        SAM-3D API - 3D 모델 생성 및 부피 계산

    제거된 컴포넌트:
        - CLIP: YOLO 클래스로 직접 DB 매칭
        - SAHI: 단일 모델 추론으로 단순화
        - SAM2: YOLOE-seg 마스크가 더 정확 (2024-01 테스트 결과)
    """

    def __init__(
        self,
        sam2_api_url: str = "http://localhost:8000",  # 하위 호환성 유지
        enable_3d_generation: bool = True,
        device_id: Optional[int] = None,
        gpu_pool: Optional[GPUPoolManager] = None
    ):
        """
        Args:
            sam2_api_url: SAM-3D API 서버 URL (SAM2 제거, 이름 유지는 하위 호환성)
            enable_3d_generation: 3D 생성 활성화 여부
            device_id: GPU 디바이스 ID (None이면 기본값 사용)
            gpu_pool: GPU 풀 매니저 (Multi-GPU 처리용)
        """
        self.api_url = sam2_api_url.rstrip('/')
        self.sam2_api_url = self.api_url  # 하위 호환성
        self.enable_3d_generation = enable_3d_generation

        # Multi-GPU 지원
        self.device_id = device_id
        self._device = Config.get_device(device_id)
        self.gpu_pool = gpu_pool

        print(f"[FurniturePipeline] Initializing on device: {self._device}")

        # Stage 1: 이미지 가져오기
        self.fetcher = ImageFetcher()

        # Stage 2: YOLO-E 탐지 - device_id 전달
        self.detector = YoloDetector(device_id=device_id)

        # Stage 3: is_movable 판단
        self.movability_checker = MovabilityChecker()

        # Volume 계산기
        self.volume_calculator = VolumeCalculator()

        # SAM-3D 변환기 - device_id 전달
        self.sam3d_converter = SAM3DConverter(device_id=device_id)

        # 클래스 매핑 (동의어 → DB 키)
        self.class_map = self.movability_checker.class_map

        print(f"[FurniturePipeline] Initialized with {len(self.class_map)} class mappings on {self._device}")

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
    # Stage 2-3: 객체 탐지 → is_movable 결정
    # =========================================================================

    def detect_objects(self, image: Image.Image) -> List[DetectedObject]:
        """
        Stage 2-3 통합: 이미지에서 객체를 탐지하고 is_movable을 결정합니다.

        CLIP 제거 - YOLO 클래스로 직접 DB 매칭
        SAHI 제거 - 단일 모델 추론

        Args:
            image: PIL 이미지

        Returns:
            DetectedObject 리스트
        """
        img_w, img_h = image.size
        detected_objects = []

        # Stage 2: YOLO-E 탐지 (세그멘테이션 마스크 포함)
        results = self.detector.detect_smart(image, return_masks=True)

        # 출력에서 제외할 클래스 필터링 (예: Kitchen Island, Floor)
        results = self.detector.filter_excluded_classes(results)

        if results is None or len(results["boxes"]) == 0:
            return detected_objects

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]
        masks = results.get("masks", [])

        for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = map(int, box)

            # 너무 작은 박스 필터링
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            # DB 키 매핑
            db_key = self.class_map.get(label.lower())
            if not db_key:
                # DB에 없는 클래스도 기본값으로 처리
                db_key = label.lower()

            # 객체 크롭
            crop = image.crop((x1, y1, x2, y2))

            # 중심점 계산
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Stage 3: is_movable 판단 (CLIP 없이 DB 직접 매칭)
            movability = self.movability_checker.check_from_label(label, float(score))

            # 세그멘테이션 마스크 (있는 경우)
            yolo_mask = None
            if masks and idx < len(masks):
                yolo_mask = masks[idx]

            obj = DetectedObject(
                id=idx,
                label=movability.label,
                db_key=db_key,
                subtype_name=None,  # CLIP 제거로 서브타입 없음
                bbox=[x1, y1, x2, y2],
                center_point=[center_x, center_y],
                confidence=float(score),
                crop_image=crop,
                yolo_mask=yolo_mask
            )

            detected_objects.append(obj)

        return detected_objects

    # =========================================================================
    # YOLOE-seg 마스크 변환 (V2 파이프라인)
    # =========================================================================

    def _yolo_mask_to_base64(self, yolo_mask: np.ndarray) -> str:
        """
        YOLOE-seg 마스크를 base64로 변환합니다.

        Args:
            yolo_mask: numpy array (H, W), uint8, 0/255 값

        Returns:
            Base64 인코딩된 마스크 PNG
        """
        mask_pil = Image.fromarray(yolo_mask, mode="L")
        buffer = io.BytesIO()
        mask_pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # =========================================================================
    # External API: SAM-3D 3D 생성
    # =========================================================================

    async def generate_3d(
        self,
        image: Image.Image,
        mask_b64: str,
        seed: int = 42,
        skip_gif: bool = True,
        max_image_size: int = 512
    ) -> Optional[Dict]:
        """
        SAM-3D API를 호출하여 3D 모델을 생성합니다.

        Args:
            image: 원본 이미지
            mask_b64: Base64 인코딩된 마스크
            seed: 랜덤 시드
            skip_gif: GIF 렌더링 스킵 (부피 계산 최적화)
            max_image_size: 최대 이미지 크기 (속도 최적화)

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
                # 3D 생성 요청 (최적화 옵션 포함)
                async with session.post(
                    f"{self.sam2_api_url}/generate-3d",
                    json={
                        "image": image_b64,
                        "mask": mask_b64,
                        "seed": seed,
                        "skip_gif": skip_gif,
                        "max_image_size": max_image_size
                    },
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

        # 절대 치수 계산은 백엔드에서 처리 - 상대 치수만 반환
        return relative_dims, None

    # =========================================================================
    # 통합 처리
    # =========================================================================

    def _image_to_base64(self, image: Image.Image) -> str:
        """PIL 이미지를 base64로 변환"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    async def _parallel_3d_generation(
        self,
        image: Image.Image,
        objects_with_masks: List[Tuple[int, DetectedObject]]
    ) -> Dict[int, Dict]:
        """
        여러 객체를 병렬로 3D 변환 (Worker Pool 사용)

        Args:
            image: 원본 이미지
            objects_with_masks: [(object_id, DetectedObject), ...] 마스크가 있는 객체들

        Returns:
            {object_id: {"ply_b64": str, ...}, ...}
        """
        sam3d_pool = get_sam3d_worker_pool()

        if sam3d_pool is None or not sam3d_pool.is_ready():
            # Worker Pool 미사용 - 기존 순차 처리
            print("[FurniturePipeline] SAM3D Worker Pool not available, falling back to sequential")
            results = {}
            for obj_id, obj in objects_with_masks:
                gen_result = await self.generate_3d(image, obj.mask_base64)
                if gen_result:
                    results[obj_id] = gen_result
            return results

        # Worker Pool 사용 - 병렬 처리
        print(f"[FurniturePipeline] Parallel 3D generation for {len(objects_with_masks)} objects")

        # 이미지를 base64로 변환 (한 번만)
        image_b64 = self._image_to_base64(image)

        # 작업 목록 생성
        tasks = []
        for obj_id, obj in objects_with_masks:
            tasks.append({
                "task_id": f"obj_{obj_id}",
                "image_b64": image_b64,
                "mask_b64": obj.mask_base64,
                "seed": 42,
                "skip_gif": True
            })

        # 병렬 제출
        worker_results = await sam3d_pool.submit_tasks_parallel(tasks)

        # 결과 매핑
        results = {}
        for i, (obj_id, obj) in enumerate(objects_with_masks):
            worker_result = worker_results[i]
            if worker_result.success:
                results[obj_id] = {
                    "ply_b64": worker_result.ply_b64,
                    "ply_size_bytes": worker_result.ply_size_bytes,
                    "gif_b64": worker_result.gif_b64,
                    "mesh_url": worker_result.mesh_url
                }
                print(f"[FurniturePipeline] Object {obj_id} 3D generated: {worker_result.ply_size_bytes} bytes")
            else:
                print(f"[FurniturePipeline] Object {obj_id} 3D failed: {worker_result.error}")

        return results

    async def process_single_image(
        self,
        image_url: str,
        enable_mask: bool = True,
        enable_3d: bool = True,
        use_parallel_3d: bool = True
    ) -> PipelineResult:
        """
        단일 이미지를 처리합니다 (V2 파이프라인).

        Args:
            image_url: Firebase Storage 이미지 URL
            enable_mask: 마스크 활성화 (V2: YOLOE-seg 마스크 사용)
            enable_3d: SAM-3D 3D 생성 활성화
            use_parallel_3d: 병렬 3D 생성 사용 (Worker Pool)

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

            # Stage 2-3: 객체 탐지 및 is_movable 결정
            detected_objects = self.detect_objects(image)

            # Stage 4: 마스크 준비 (V2: YOLOE-seg 마스크 직접 사용)
            objects_with_masks = []
            for obj in detected_objects:
                if enable_mask:
                    if obj.yolo_mask is not None:
                        mask_b64 = self._yolo_mask_to_base64(obj.yolo_mask)
                        obj.mask_base64 = mask_b64
                        objects_with_masks.append((obj.id, obj))
                    else:
                        print(f"[FurniturePipeline] No YOLOE-seg mask for {obj.label}, skipping 3D")
                        obj.mask_base64 = None

            # Stage 5-6: 3D 생성 (병렬 또는 순차)
            if enable_3d and self.enable_3d_generation and objects_with_masks:
                if use_parallel_3d:
                    # 병렬 3D 생성 (Worker Pool)
                    gen_results = await self._parallel_3d_generation(image, objects_with_masks)
                else:
                    # 순차 3D 생성 (기존 방식)
                    gen_results = {}
                    for obj_id, obj in objects_with_masks:
                        gen_result = await self.generate_3d(image, obj.mask_base64)
                        if gen_result:
                            gen_results[obj_id] = gen_result

                # 결과를 객체에 매핑
                for obj in detected_objects:
                    if obj.id in gen_results:
                        gen_result = gen_results[obj.id]
                        obj.glb_url = gen_result.get("mesh_url")

                        if gen_result.get("ply_b64"):
                            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                                tmp.write(base64.b64decode(gen_result["ply_b64"]))
                                ply_path = tmp.name

                            rel_dims, abs_dims = self.calculate_dimensions(obj, ply_path=ply_path)
                            obj.relative_dimensions = rel_dims
                            obj.absolute_dimensions = abs_dims

                            os.unlink(ply_path)

            # 결과 집계 (상대 부피 사용 - 절대 부피는 백엔드에서 계산)
            result.objects = detected_objects
            result.total_volume_liters = sum(
                o.relative_dimensions.get("volume", 0)
                for o in detected_objects if o.relative_dimensions
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
        사전 초기화된 파이프라인이 있으면 재사용하여 모델 로드 오버헤드를 방지합니다.
        """
        # GPU 풀 가져오기
        pool = self.gpu_pool or get_gpu_pool()

        # max_concurrent가 None이면 GPU 수만큼 설정
        if max_concurrent is None:
            max_concurrent = len(pool.gpu_ids) if pool.gpu_ids else 3

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_gpu(url: str) -> PipelineResult:
            """GPU를 할당받아 이미지 처리 - 사전 초기화된 파이프라인 사용"""
            async with semaphore:
                try:
                    # 사전 초기화된 파이프라인이 있는지 확인
                    if pool.has_pipeline(pool.gpu_ids[0] if pool.gpu_ids else 0):
                        # 사전 초기화된 파이프라인 사용
                        async with pool.pipeline_context(task_id=url[:50]) as (gpu_id, pipeline):
                            print(f"[FurniturePipeline] Processing on GPU {gpu_id} (pre-initialized)")
                            return await pipeline.process_single_image(url, enable_mask, enable_3d)
                    else:
                        # 사전 초기화가 안 된 경우 기존 방식 사용
                        async with pool.gpu_context(task_id=url[:50]) as gpu_id:
                            if self.device_id == gpu_id:
                                # 같은 GPU면 self 재사용
                                return await self.process_single_image(url, enable_mask, enable_3d)
                            else:
                                # 다른 GPU면 새 파이프라인 생성 (비효율적이지만 폴백)
                                print(f"[FurniturePipeline] Warning: Creating new pipeline for GPU {gpu_id} (not pre-initialized)")
                                pipeline = FurniturePipeline(
                                    sam2_api_url=self.sam2_api_url,
                                    enable_3d_generation=self.enable_3d_generation,
                                    device_id=gpu_id,
                                    gpu_pool=pool
                                )
                                return await pipeline.process_single_image(url, enable_mask, enable_3d)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
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

    async def process_multiple_images_with_ids(
        self,
        image_items: List[Tuple[int, str]],
        enable_mask: bool = True,
        enable_3d: bool = True,
        max_concurrent: Optional[int] = None
    ) -> List[PipelineResult]:
        """
        여러 이미지를 사용자 지정 ID와 함께 GPU에 분배하여 병렬 처리합니다 (TDD Section 4.1).

        Args:
            image_items: [(user_image_id, url), ...] 형태의 이미지 목록
            enable_mask: 마스크 활성화
            enable_3d: 3D 생성 활성화
            max_concurrent: 최대 동시 처리 수

        Returns:
            List[PipelineResult] - 각 결과에 user_image_id가 설정됨
        """
        # GPU 풀 가져오기
        pool = self.gpu_pool or get_gpu_pool()

        # max_concurrent가 None이면 GPU 수만큼 설정
        if max_concurrent is None:
            max_concurrent = len(pool.gpu_ids) if pool.gpu_ids else 3

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_gpu(user_image_id: int, url: str) -> PipelineResult:
            """GPU를 할당받아 이미지 처리 - 사전 초기화된 파이프라인 사용"""
            async with semaphore:
                try:
                    # 사전 초기화된 파이프라인이 있는지 확인
                    if pool.has_pipeline(pool.gpu_ids[0] if pool.gpu_ids else 0):
                        # 사전 초기화된 파이프라인 사용
                        async with pool.pipeline_context(task_id=f"img_{user_image_id}") as (gpu_id, pipeline):
                            print(f"[FurniturePipeline] Processing image {user_image_id} on GPU {gpu_id} (pre-initialized)")
                            result = await pipeline.process_single_image(url, enable_mask, enable_3d)
                            result.user_image_id = user_image_id
                            return result
                    else:
                        # 사전 초기화가 안 된 경우 기존 방식 사용
                        async with pool.gpu_context(task_id=f"img_{user_image_id}") as gpu_id:
                            if self.device_id == gpu_id:
                                result = await self.process_single_image(url, enable_mask, enable_3d)
                            else:
                                print(f"[FurniturePipeline] Warning: Creating new pipeline for GPU {gpu_id} (not pre-initialized)")
                                pipeline = FurniturePipeline(
                                    sam2_api_url=self.sam2_api_url,
                                    enable_3d_generation=self.enable_3d_generation,
                                    device_id=gpu_id,
                                    gpu_pool=pool
                                )
                                result = await pipeline.process_single_image(url, enable_mask, enable_3d)
                            result.user_image_id = user_image_id
                            return result
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return PipelineResult(
                        image_id=str(uuid.uuid4()),
                        image_url=url,
                        status="failed",
                        error=str(e),
                        user_image_id=user_image_id
                    )

        tasks = [process_with_gpu(user_id, url) for user_id, url in image_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                user_id, url = image_items[i]
                processed_results.append(PipelineResult(
                    image_id=str(uuid.uuid4()),
                    image_url=url,
                    status="failed",
                    error=str(r),
                    user_image_id=user_id
                ))
            else:
                processed_results.append(r)

        return processed_results

    def to_json_response(self, results: List[PipelineResult]) -> Dict:
        """
        TDD 문서에 맞는 JSON 응답을 생성합니다 (단일 이미지용, 하위 호환성 유지).

        Output format:
        {
            "objects": [
                {
                    "label": "box",
                    "width": 30.5,      # mm
                    "depth": 20.0,      # mm
                    "height": 15.2,     # mm
                    "volume": 0.00926   # m³
                }
            ]
        }
        """
        all_objects = []

        for result in results:
            for obj in result.objects:
                if obj.relative_dimensions:
                    dims = obj.relative_dimensions
                    bbox = dims.get("bounding_box", {})

                    # VolumeCalculator returns normalized values (relative dimensions)
                    # Absolute volume is calculated by backend using knowledge base
                    volume = dims.get("volume", 0)

                    all_objects.append({
                        "label": obj.label,
                        "width": round(bbox.get("width", 0), 2),
                        "depth": round(bbox.get("depth", 0), 2),
                        "height": round(bbox.get("height", 0), 2),
                        "volume": round(volume, 6)
                    })

        return {"objects": all_objects}

    def to_json_response_v2(self, results: List[PipelineResult]) -> Dict:
        """
        TDD 문서 Section 4.1에 맞는 JSON 응답을 생성합니다 (다중 이미지용).

        Output format (TDD_PIPELINE_V2.md Section 4.1):
        {
            "results": [
                {
                    "image_id": 101,
                    "objects": [
                        {"label": "sofa", "width": 200.0, "depth": 90.0, "height": 85.0, "volume": 1.53},
                        ...
                    ]
                },
                ...
            ]
        }
        """
        results_list = []

        for result in results:
            objects_list = []

            for obj in result.objects:
                if obj.relative_dimensions:
                    dims = obj.relative_dimensions
                    bbox = dims.get("bounding_box", {})

                    # VolumeCalculator returns normalized values (relative dimensions)
                    # Absolute volume is calculated by backend using knowledge base
                    volume = dims.get("volume", 0)

                    objects_list.append({
                        "label": obj.label,
                        "width": round(bbox.get("width", 0), 2),
                        "depth": round(bbox.get("depth", 0), 2),
                        "height": round(bbox.get("height", 0), 2),
                        "volume": round(volume, 6)
                    })

            results_list.append({
                "image_id": result.user_image_id,
                "objects": objects_list
            })

        return {"results": results_list}
