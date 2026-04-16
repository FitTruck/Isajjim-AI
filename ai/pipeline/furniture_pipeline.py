"""
Furniture Analysis Pipeline

전체 AI 로직을 통합한 파이프라인 오케스트레이터:

[V2.5 파이프라인 - CLIP/SAHI/SAM2 제거 + GCS 업로드]
1. Firebase Storage URL에서 이미지 가져오기
2. YOLOE-seg로 객체 탐지 (bbox + class + 세그멘테이션 마스크)
3. DB 대조하여 is_movable 결정 (YOLO 클래스로 직접 매칭)
4. YOLOE-seg 마스크를 SAM-3D에 직접 전달 (SAM2 제거)
5. SAM3D로 3D 변환
6. 객체별 부피 계산 (trimesh)
7. PLY 파일 GCS 업로드 및 절대 부피 계산 후 JSON 응답

변경 이유 (2024-01 테스트 결과):
- YOLOE-seg 마스크가 SAM2보다 객체 전체를 더 정확하게 커버
- SAM2 API 호출 제거로 latency 감소
- 파이프라인 단순화

V2.5 추가 (2026-02):
- PLY 파일 GCS 업로드 및 ply_url 응답 추가
- 절대 부피 계산 AI 서버에서 수행
"""

import os
import io
import base64
import tempfile
import uuid
import asyncio
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from PIL import Image
import numpy as np
from dataclasses import dataclass, field

# Type checking only imports (avoid circular imports)
if TYPE_CHECKING:
    from api.services.gcs_storage import GCSStorageService

# AI processors import
from ai.processors import (
    ImageFetcher,
    YoloDetector,
    MovabilityChecker,
    DimensionCalculator,
    AbsoluteVolumeCalculator,
    AbsoluteVolumeResult,
    PLYPreprocessor,
    DimensionStrategy,
    DimensionMode,
    is_boxer_available,
    BoxerDimensionEstimator,
    BoxerResult,
)

# GPU pool manager import
from ai.gpu import GPUPoolManager, get_gpu_pool
from ai.gpu import get_sam3d_worker_pool

# Config import
from ai.config import Config

# Knowledge base import for subtype exclusion
from ai.data.knowledge_base import get_excluded_subtype_names


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
    ply_url: Optional[str] = None           # GCS 업로드 URL (V2.5)
    ply_b64: Optional[str] = None           # PLY base64 데이터 (임시 저장용)
    glb_url: Optional[str] = None
    gif_url: Optional[str] = None

    # 치수 정보
    relative_dimensions: Optional[Dict] = None
    absolute_dimensions: Optional["AbsoluteVolumeResult"] = None  # V3: Boxer 또는 규칙기반


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    image_id: str
    image_url: str
    objects: List[DetectedObject] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    status: str = "pending"
    error: Optional[str] = None
    user_image_id: Optional[int] = None  # 사용자 지정 이미지 ID (TDD Section 4.1)


class FurniturePipeline:
    """
    가구 분석 통합 파이프라인 V2.5

    [V2.5 파이프라인 - SAM2 제거 + GCS 업로드]
    AI Logic Stages:
        Stage 1: ImageFetcher - Firebase URL → PIL Image
        Stage 2: YoloDetector - 객체 탐지 (bbox, class, 세그멘테이션 마스크)
        Stage 3: MovabilityChecker - DB 대조 → is_movable 결정
        Stage 4: YOLOE-seg 마스크 → SAM-3D 직접 전달
        Stage 5: PLY GCS 업로드 및 절대 부피 계산

    External API:
        SAM-3D API - 3D 모델 생성 및 부피 계산

    제거된 컴포넌트:
        - CLIP: YOLO 클래스로 직접 DB 매칭
        - SAHI: 단일 모델 추론으로 단순화
        - SAM2: YOLOE-seg 마스크가 더 정확 (2024-01 테스트 결과)
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        enable_3d_generation: bool = True,
        device_id: Optional[int] = None,
        gpu_pool: Optional[GPUPoolManager] = None,
        gcs_service: Optional["GCSStorageService"] = None,
        estimate_id: Optional[int] = None
    ):
        """
        Args:
            api_url: API 서버 URL (현재 사용 안함, 하위 호환성 유지)
            enable_3d_generation: 3D 생성 활성화 여부
            device_id: GPU 디바이스 ID (None이면 기본값 사용)
            gpu_pool: GPU 풀 매니저 (Multi-GPU 처리용)
            gcs_service: GCS 업로드 서비스 (None이면 업로드 안함)
            estimate_id: 견적 ID (GCS 파일명에 사용)
        """
        self.api_url = api_url.rstrip('/')
        self.enable_3d_generation = enable_3d_generation

        # Multi-GPU 지원
        self.device_id = device_id
        self._device = Config.get_device(device_id)
        self.gpu_pool = gpu_pool

        # GCS 업로드 서비스 (V2.5)
        self.gcs_service = gcs_service
        self.estimate_id = estimate_id

        print(f"[FurniturePipeline] Initializing on device: {self._device}")

        # Stage 1: 이미지 가져오기
        self.fetcher = ImageFetcher()

        # Stage 2: YOLO-E 탐지 - device_id 전달
        self.detector = YoloDetector(device_id=device_id)

        # Stage 3: is_movable 판단
        self.movability_checker = MovabilityChecker()

        # 치수 계산기
        self.dimension_calculator = DimensionCalculator()

        # V3: Boxer 절대 치수 추정 + MoGe depth
        self.boxer_estimator: Optional[BoxerDimensionEstimator] = None
        self._moge_model = None
        self.dim_strategy = DimensionStrategy(
            mode=DimensionMode.RULE_BASED_ONLY  # 기본값: 기존 규칙기반
        )

        if Config.ENABLE_BOXER and is_boxer_available():
            try:
                self.boxer_estimator = BoxerDimensionEstimator(device_id=device_id or 0)
                self.dim_strategy = DimensionStrategy(
                    mode=DimensionMode.BOXER_WITH_FALLBACK
                )
                print(f"[FurniturePipeline] Boxer enabled on {self._device}")

                # MoGe depth 모델 로드 (Boxer SDP용)
                try:
                    from moge.model.v1 import MoGeModel
                    self._moge_model = MoGeModel.from_pretrained(
                        "Ruicheng/moge-vitl"
                    ).to(self._device).eval()
                    print(f"[FurniturePipeline] MoGe depth enabled for Boxer")
                except Exception as e:
                    print(f"[FurniturePipeline] MoGe not available, Boxer without depth: {e}")
            except Exception as e:
                print(f"[FurniturePipeline] Boxer init failed, using rule-based: {e}")

        # 클래스 매핑 (동의어 → DB 키)
        self.class_map = self.movability_checker.class_map

        print(f"[FurniturePipeline] Initialized with {len(self.class_map)} class mappings on {self._device}")

    # =========================================================================
    # Stage 1: 이미지 가져오기
    # =========================================================================

    async def fetch_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Stage 1: Firebase URL에서 이미지 가져오기"""
        return await self.fetcher.fetch_async(url)

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

            # Subtype 탐지 (knowledge_base.py에 subtypes가 있는 경우만)
            subtype_name = self.detector.detect_subtype(crop, label)

            obj = DetectedObject(
                id=idx,
                label=movability.label,
                db_key=db_key,
                subtype_name=subtype_name,
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
    # MoGe depth 추론 (Boxer SDP용)
    # =========================================================================

    def _run_moge(self, image: Image.Image):
        """
        MoGe 단안 depth 추정. Boxer의 SDP 입력으로 사용.

        Returns:
            (depth_map, intrinsics): depth (H,W) numpy, intrinsics (3,3) tensor
        """
        import torch as _torch

        img_resized = image.resize((512, 512))
        img_np = np.array(img_resized.convert("RGB"))
        img_tensor = _torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        with _torch.no_grad():
            output = self._moge_model.infer(
                img_tensor.to(self._device), force_projection=False
            )

        depth = output["depth"].cpu().numpy()
        intrinsics = output.get("intrinsics")

        return depth, intrinsics

    # =========================================================================
    # 치수 계산
    # =========================================================================

    def calculate_dimensions(
        self,
        ply_path: Optional[str] = None,
        glb_path: Optional[str] = None
    ) -> Optional[Dict]:
        """
        3D 모델에서 치수를 계산합니다.

        Returns:
            relative_dimensions (절대 치수는 백엔드에서 계산)
        """
        if ply_path and os.path.exists(ply_path):
            return self.dimension_calculator.calculate_from_ply(ply_path)
        elif glb_path and os.path.exists(glb_path):
            return self.dimension_calculator.calculate_from_glb(glb_path)
        return None

    # =========================================================================
    # 통합 처리
    # =========================================================================

    def _image_to_base64(self, image: Image.Image) -> str:
        """PIL 이미지를 base64로 변환"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    async def generate_3d(
        self,
        image: Image.Image,
        mask_b64: str
    ) -> Optional[Dict]:
        """
        단일 객체에 대한 3D 생성 (Worker Pool 사용)

        Args:
            image: 원본 이미지
            mask_b64: 마스크 base64 문자열

        Returns:
            {"ply_b64": str, "ply_size_bytes": int, ...} or None
        """
        sam3d_pool = get_sam3d_worker_pool()

        if sam3d_pool is None or not sam3d_pool.is_ready():
            print("[FurniturePipeline] SAM3D Worker Pool not available")
            return None

        # 이미지를 base64로 변환
        image_b64 = self._image_to_base64(image)

        # 단일 작업 생성
        tasks = [{
            "task_id": "single_obj",
            "image_b64": image_b64,
            "mask_b64": mask_b64,
            "seed": 42
        }]

        # Worker Pool에 제출
        results = await sam3d_pool.submit_tasks_parallel(tasks)

        if results and results[0].success:
            return {
                "ply_b64": results[0].ply_b64,
                "ply_size_bytes": results[0].ply_size_bytes
            }

        return None

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
            # Worker Pool 미사용 - 3D 생성 불가
            print("[FurniturePipeline] SAM3D Worker Pool not available, skipping 3D generation")
            return {}

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
                "seed": 42
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
                    "ply_size_bytes": worker_result.ply_size_bytes
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

            # Stage 5-6: SAM3D 3D 생성 + Boxer 절대 치수 (병렬)
            if enable_3d and self.enable_3d_generation and objects_with_masks:
                # --- MoGe depth 추론 (Boxer SDP용, SAM3D와 공유 가능) ---
                moge_depth = None
                moge_intrinsics = None
                if self.boxer_estimator and self._moge_model is not None:
                    try:
                        moge_depth, moge_intrinsics = await asyncio.to_thread(
                            self._run_moge, image
                        )
                    except Exception as e:
                        print(f"[FurniturePipeline] MoGe failed: {e}")

                # --- 병렬 실행: SAM3D + Boxer ---
                sam3d_task = self._parallel_3d_generation(image, objects_with_masks)

                # Boxer 추론 (별도 스레드, MoGe depth 전달)
                boxer_results_map: Dict[int, BoxerResult] = {}
                boxer_task = None
                id_list: List[int] = []
                if self.boxer_estimator:
                    bboxes_for_boxer = [
                        (obj_id, obj.bbox)
                        for obj_id, obj in objects_with_masks
                    ]
                    bbox_list = [bbox for _, bbox in bboxes_for_boxer]
                    id_list = [obj_id for obj_id, _ in bboxes_for_boxer]
                    boxer_task = asyncio.to_thread(
                        self.boxer_estimator.predict,
                        image, bbox_list,
                        depth_map=moge_depth,
                        moge_intrinsics=moge_intrinsics,
                    )

                # 병렬 대기
                if boxer_task:
                    gen_results, boxer_raw = await asyncio.gather(
                        sam3d_task, boxer_task,
                        return_exceptions=True
                    )
                    # Boxer 결과 매핑 (인덱스 기반 1:1)
                    if not isinstance(boxer_raw, Exception) and boxer_raw:
                        for i, obj_id in enumerate(id_list):
                            if i < len(boxer_raw):
                                boxer_results_map[obj_id] = boxer_raw[i]
                    elif isinstance(boxer_raw, Exception):
                        print(f"[FurniturePipeline] Boxer failed: {boxer_raw}")
                    # gen_results 예외/타입 처리
                    if isinstance(gen_results, Exception):
                        print(f"[FurniturePipeline] SAM3D failed: {gen_results}")
                        gen_results = {}
                    elif not isinstance(gen_results, dict):
                        print(f"[FurniturePipeline] SAM3D unexpected type: {type(gen_results)}")
                        gen_results = {}
                else:
                    gen_results = await sam3d_task

                # GCS 업로드 작업 목록 (병렬 처리용)
                gcs_upload_tasks = []

                # 결과를 객체에 매핑 + 절대 치수 계산
                for obj in detected_objects:
                    if obj.id in gen_results:
                        gen_result = gen_results[obj.id]

                        if gen_result.get("ply_b64"):
                            # PLY base64 임시 저장 (GCS 업로드용)
                            obj.ply_b64 = gen_result["ply_b64"]

                            # 상대 치수 계산 (fallback용 + PLY 전처리용)
                            with tempfile.NamedTemporaryFile(suffix=".ply", delete=True) as tmp:
                                tmp.write(base64.b64decode(gen_result["ply_b64"]))
                                tmp.flush()
                                obj.relative_dimensions = self.calculate_dimensions(ply_path=tmp.name)

                            # 절대 치수 결정 (Boxer 우선 → fallback)
                            boxer_res = boxer_results_map.get(obj.id)
                            rel_dims = obj.relative_dimensions or {}
                            bbox_dims = rel_dims.get("bounding_box", {})

                            obj.absolute_dimensions = self.dim_strategy.resolve(
                                boxer_result=boxer_res,
                                label=obj.label,
                                type_name=obj.subtype_name,
                                rel_width=bbox_dims.get("width", 0),
                                rel_depth=bbox_dims.get("depth", 0),
                                rel_height=bbox_dims.get("height", 0),
                            )

                            # GCS 업로드 작업 추가
                            if self.gcs_service and obj.ply_b64:
                                filename = self.gcs_service.generate_unique_filename(
                                    label=obj.label,
                                    estimate_id=self.estimate_id,
                                    image_id=result.user_image_id,
                                    object_idx=obj.id
                                )
                                gcs_upload_tasks.append((obj, obj.ply_b64, filename))

                # PLY 전처리 및 GCS 병렬 업로드
                if gcs_upload_tasks:
                    print(f"[FurniturePipeline] Processing {len(gcs_upload_tasks)} PLY files")

                    preprocessor = None
                    if Config.PLY_ENABLE_PREPROCESSING:
                        preprocessor = PLYPreprocessor(
                            max_points=Config.PLY_MAX_POINTS,
                            convert_to_yup=Config.PLY_CONVERT_TO_YUP,
                            enable_alignment=Config.PLY_ENABLE_ALIGNMENT,
                            enable_scaling=Config.PLY_ENABLE_SCALING,
                            enable_downsampling=Config.PLY_ENABLE_DOWNSAMPLING
                        )

                    upload_items = []

                    for (obj, ply_b64, filename) in gcs_upload_tasks:
                        processed_ply_b64 = ply_b64  # 기본값: 원본

                        abs_result = obj.absolute_dimensions
                        if preprocessor and abs_result:
                            try:
                                processed_ply_b64, preprocess_result = preprocessor.process(
                                    ply_b64=ply_b64,
                                    target_width_mm=abs_result.width_mm,
                                    target_depth_mm=abs_result.depth_mm,
                                    target_height_mm=abs_result.height_mm
                                )

                                if preprocess_result.success:
                                    print(f"[FurniturePipeline] PLY preprocessed for {obj.label}: "
                                          f"{preprocess_result.original_points} → {preprocess_result.processed_points} points, "
                                          f"{preprocess_result.original_size_bytes} → {preprocess_result.processed_size_bytes} bytes")
                                else:
                                    print(f"[FurniturePipeline] PLY preprocessing failed for {obj.label}: {preprocess_result.message}")
                                    processed_ply_b64 = ply_b64
                            except Exception as e:
                                print(f"[FurniturePipeline] PLY preprocessing error for {obj.label}: {e}")
                                processed_ply_b64 = ply_b64

                        upload_items.append((obj, processed_ply_b64, filename))

                    # GCS 병렬 업로드
                    print(f"[FurniturePipeline] Uploading {len(upload_items)} PLY files to GCS")
                    upload_coros = [
                        self.gcs_service.upload_ply_base64(ply_b64, filename)
                        for (obj, ply_b64, filename) in upload_items
                    ]
                    upload_results = await asyncio.gather(*upload_coros, return_exceptions=True)

                    # 업로드 결과를 객체에 매핑
                    for i, (obj, ply_b64, filename) in enumerate(upload_items):
                        upload_result = upload_results[i]
                        if isinstance(upload_result, Exception):
                            print(f"[FurniturePipeline] GCS upload failed for {obj.label}: {upload_result}")
                        else:
                            obj.ply_url = upload_result
                            print(f"[FurniturePipeline] GCS upload success: {obj.ply_url}")
                        # 메모리 정리 - 업로드 후 base64 데이터 삭제
                        obj.ply_b64 = None

            # 결과 집계
            result.objects = detected_objects
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
        # 내부적으로 process_multiple_images_with_ids 사용 (중복 코드 제거)
        image_items = [(i, url) for i, url in enumerate(image_urls)]
        return await self.process_multiple_images_with_ids(
            image_items, enable_mask, enable_3d, max_concurrent
        )

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
                            # GCS 서비스와 estimate_id를 현재 파이프라인에서 복사 (V2.5)
                            pipeline.gcs_service = self.gcs_service
                            pipeline.estimate_id = self.estimate_id
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
                                    api_url=self.api_url,
                                    enable_3d_generation=self.enable_3d_generation,
                                    device_id=gpu_id,
                                    gpu_pool=pool,
                                    gcs_service=self.gcs_service,
                                    estimate_id=self.estimate_id
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
                    "type": "STORAGE_BOX",
                    "width": 30.5,
                    "depth": 20.0,
                    "height": 15.2
                }
            ]
        }

        Note: volume 필드는 제거됨 (백엔드에서 절대 부피 계산)
        """
        all_objects = []
        excluded_subtypes = get_excluded_subtype_names()

        for result in results:
            for obj in result.objects:
                # Skip excluded subtypes
                if obj.subtype_name and obj.subtype_name in excluded_subtypes:
                    continue

                if obj.relative_dimensions:
                    dims = obj.relative_dimensions
                    bbox = dims.get("bounding_box", {})

                    all_objects.append({
                        "label": obj.label,
                        "type": obj.subtype_name,
                        "width": round(bbox.get("width", 0), 2),
                        "depth": round(bbox.get("depth", 0), 2),
                        "height": round(bbox.get("height", 0), 2)
                    })

        return {"objects": all_objects}

    def to_json_response_v2(self, results: List[PipelineResult]) -> Dict:
        """
        TDD 문서 Section 4.1에 맞는 JSON 응답을 생성합니다 (다중 이미지용).

        Output format (TDD_PIPELINE_V2.md Section 4.1 - Updated):
        {
            "results": [
                {
                    "image_id": 101,
                    "objects": [
                        {
                            "label": "SOFA",
                            "type": "THREE_SEATER_SOFA",
                            "width": 1000.0,   # mm (절대 치수)
                            "depth": 3000.0,   # mm (절대 치수)
                            "height": 900.0,   # mm (절대 치수)
                            "volume": 2.7      # m³ (절대 부피)
                        },
                        ...
                    ]
                },
                ...
            ]
        }

        Note: 절대 치수와 부피는 AI 서버에서 계산 (Backend fallback 가능)
        """
        results_list = []
        excluded_subtypes = get_excluded_subtype_names()
        abs_calc = AbsoluteVolumeCalculator()

        for result in results:
            objects_list = []

            for obj in result.objects:
                # Skip excluded subtypes
                if obj.subtype_name and obj.subtype_name in excluded_subtypes:
                    continue

                if obj.relative_dimensions:
                    # V3: 미리 계산된 절대 치수 사용 (Boxer 또는 규칙기반)
                    abs_result = obj.absolute_dimensions
                    if abs_result is None:
                        # Fallback: process_single_image에서 계산 안 된 경우
                        dims = obj.relative_dimensions
                        bbox = dims.get("bounding_box", {})
                        abs_result = abs_calc.calculate_absolute_volume(
                            label=obj.label,
                            type_name=obj.subtype_name,
                            rel_width=bbox.get("width", 0),
                            rel_depth=bbox.get("depth", 0),
                            rel_height=bbox.get("height", 0)
                        )

                    objects_list.append({
                        "label": obj.label,
                        "type": abs_result.matched_type,
                        "width": abs_result.width_mm,
                        "depth": abs_result.depth_mm,
                        "height": abs_result.height_mm,
                        "volume": abs_result.volume_m3,
                        "ply_url": obj.ply_url,  # GCS URL (V2.5)
                        "center_x": round(obj.center_point[0], 1) if obj.center_point else 0.0,
                        "center_y": round(obj.center_point[1], 1) if obj.center_point else 0.0
                    })

            results_list.append({
                "image_id": result.user_image_id,
                "objects": objects_list
            })

        return {"results": results_list}
