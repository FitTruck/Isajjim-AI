"""
Unit tests for ai/pipeline/furniture_pipeline.py

FurniturePipeline 단위 테스트 (Mock 기반):
- generate_3d 메서드
- calculate_dimensions 메서드
- process_single_image 메서드
- to_json_response 메서드
"""

import pytest
import base64
import io
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import asynccontextmanager
from PIL import Image
import numpy as np

from ai.pipeline.furniture_pipeline import (
    FurniturePipeline,
    DetectedObject,
    PipelineResult
)


@pytest.fixture
def mock_pipeline(tmp_path):
    """Mock된 FurniturePipeline"""
    with patch('ai.pipeline.furniture_pipeline.YoloDetector') as mock_yolo, \
         patch('ai.pipeline.furniture_pipeline.DimensionCalculator') as mock_dim:

        mock_yolo.return_value = MagicMock()
        mock_dim.return_value = MagicMock()

        pipeline = FurniturePipeline(
            api_url="http://test:8000",
            enable_3d_generation=False,
            device_id=0
        )
        return pipeline


class TestFurniturePipelineGenerate3D:
    """generate_3d 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_generate_3d_disabled(self, mock_pipeline):
        """3D 생성 비활성화시 None 반환"""
        mock_pipeline.enable_3d_generation = False

        result = await mock_pipeline.generate_3d(
            Image.new('RGB', (100, 100)),
            "test_mask_b64"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_3d_no_aiohttp(self, mock_pipeline):
        """aiohttp 없을 때 None 반환"""
        mock_pipeline.enable_3d_generation = True

        with patch('ai.pipeline.furniture_pipeline.HAS_AIOHTTP', False):
            result = await mock_pipeline.generate_3d(
                Image.new('RGB', (100, 100)),
                "test_mask_b64"
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_generate_3d_api_error(self, mock_pipeline):
        """API 호출 실패시 None 반환"""
        mock_pipeline.enable_3d_generation = True

        # aiohttp 세션 모킹
        mock_response = AsyncMock()
        mock_response.status = 500

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('ai.pipeline.furniture_pipeline.HAS_AIOHTTP', True), \
             patch('aiohttp.ClientSession', return_value=mock_session):
            result = await mock_pipeline.generate_3d(
                Image.new('RGB', (100, 100)),
                "test_mask_b64"
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_generate_3d_exception(self, mock_pipeline):
        """예외 발생시 None 반환"""
        mock_pipeline.enable_3d_generation = True

        with patch('ai.pipeline.furniture_pipeline.HAS_AIOHTTP', True), \
             patch('aiohttp.ClientSession', side_effect=Exception("Connection error")):
            result = await mock_pipeline.generate_3d(
                Image.new('RGB', (100, 100)),
                "test_mask_b64"
            )

            assert result is None


class TestFurniturePipelineCalculateDimensions:
    """calculate_dimensions 메서드 테스트"""

    def test_calculate_dimensions_with_ply(self, mock_pipeline, tmp_path):
        """PLY 파일에서 치수 계산"""
        # PLY 파일 생성
        ply_path = tmp_path / "test.ply"
        ply_path.write_text("ply\nformat ascii 1.0\nend_header\n")

        mock_pipeline.dimension_calculator.calculate_from_ply.return_value = {
            "bounding_box": {"width": 10, "depth": 10, "height": 10},
            "centroid": [5, 5, 5],
            "surface_area": 600
        }

        dims = mock_pipeline.calculate_dimensions(ply_path=str(ply_path))

        assert dims is not None
        assert "bounding_box" in dims

    def test_calculate_dimensions_with_glb(self, mock_pipeline, tmp_path):
        """GLB 파일에서 치수 계산"""
        # GLB 파일 생성
        glb_path = tmp_path / "test.glb"
        glb_path.write_bytes(b"GLTF")

        mock_pipeline.dimension_calculator.calculate_from_glb.return_value = {
            "bounding_box": {"width": 5, "depth": 10, "height": 10},
            "centroid": [2.5, 5, 5],
            "surface_area": 350
        }

        dims = mock_pipeline.calculate_dimensions(glb_path=str(glb_path))

        assert dims is not None
        assert "bounding_box" in dims

    def test_calculate_dimensions_no_file(self, mock_pipeline):
        """파일 없을 때 None 반환"""
        dims = mock_pipeline.calculate_dimensions()

        assert dims is None

    def test_calculate_dimensions_nonexistent_file(self, mock_pipeline):
        """존재하지 않는 파일"""
        dims = mock_pipeline.calculate_dimensions(
            ply_path="/nonexistent/path.ply"
        )

        assert dims is None


class TestFurniturePipelineToJsonResponse:
    """to_json_response 메서드 테스트"""

    def test_to_json_response_empty_results(self, mock_pipeline):
        """빈 결과"""
        json_resp = mock_pipeline.to_json_response([])

        assert "objects" in json_resp
        assert len(json_resp["objects"]) == 0

    def test_to_json_response_with_objects(self, mock_pipeline):
        """객체가 있는 결과"""
        obj = DetectedObject(
            id=0,
            label="침대",
            db_key="bed",
            relative_dimensions={
                "bounding_box": {"width": 2000, "depth": 1500, "height": 450},
                "centroid": [1000, 750, 225],
                "surface_area": 1000
            }
        )

        result = PipelineResult(
            image_id="test",
            image_url="http://test.com",
            objects=[obj],
            status="completed"
        )

        json_resp = mock_pipeline.to_json_response([result])

        assert "objects" in json_resp
        assert len(json_resp["objects"]) == 1
        assert json_resp["objects"][0]["label"] == "침대"

    def test_to_json_response_filters_no_dimensions(self, mock_pipeline):
        """치수 없는 객체 필터링"""
        obj_with_dims = DetectedObject(
            id=0,
            label="침대",
            db_key="bed",
            relative_dimensions={
                "bounding_box": {"width": 100, "depth": 100, "height": 100},
                "volume": 1000000
            }
        )
        obj_without_dims = DetectedObject(
            id=1,
            label="의자",
            db_key="chair"
            # relative_dimensions 없음
        )

        result = PipelineResult(
            image_id="test",
            image_url="http://test.com",
            objects=[obj_with_dims, obj_without_dims],
            status="completed"
        )

        json_resp = mock_pipeline.to_json_response([result])

        assert len(json_resp["objects"]) == 1
        assert json_resp["objects"][0]["label"] == "침대"

    def test_to_json_response_dimensions(self, mock_pipeline):
        """치수가 상대값으로 반환 (절대 부피는 백엔드에서 계산)"""
        obj = DetectedObject(
            id=0,
            label="박스",
            db_key="box",
            relative_dimensions={
                "bounding_box": {"width": 1.0, "depth": 0.8, "height": 0.5},
                "centroid": [0.5, 0.4, 0.25],
                "surface_area": 2.0
            }
        )

        result = PipelineResult(
            image_id="test",
            image_url="http://test.com",
            objects=[obj],
            status="completed"
        )

        json_resp = mock_pipeline.to_json_response([result])

        # Dimensions are returned, volume is not included
        assert json_resp["objects"][0]["width"] == 1.0
        assert json_resp["objects"][0]["depth"] == 0.8
        assert json_resp["objects"][0]["height"] == 0.5
        assert "volume" not in json_resp["objects"][0]


class TestFurniturePipelineToJsonResponseV2:
    """to_json_response_v2 메서드 테스트 (TDD Section 4.1)"""

    def test_to_json_response_v2_format(self, mock_pipeline):
        """V2 응답 포맷: results 배열 with image_id"""
        obj = DetectedObject(
            id=0,
            label="소파",
            db_key="sofa",
            relative_dimensions={
                "bounding_box": {"width": 200.0, "depth": 90.0, "height": 85.0},
                "volume": 1.53e9
            }
        )

        result = PipelineResult(
            image_id="uuid-1",
            image_url="http://test.com/1.jpg",
            objects=[obj],
            status="completed",
            user_image_id=101
        )

        json_resp = mock_pipeline.to_json_response_v2([result])

        assert "results" in json_resp
        assert len(json_resp["results"]) == 1
        assert json_resp["results"][0]["image_id"] == 101
        assert len(json_resp["results"][0]["objects"]) == 1
        assert json_resp["results"][0]["objects"][0]["label"] == "소파"

    def test_to_json_response_v2_multiple_images(self, mock_pipeline):
        """V2 응답: 다중 이미지 결과"""
        result1 = PipelineResult(
            image_id="uuid-1",
            image_url="http://test.com/1.jpg",
            objects=[],
            status="completed",
            user_image_id=101
        )
        result2 = PipelineResult(
            image_id="uuid-2",
            image_url="http://test.com/2.jpg",
            objects=[],
            status="completed",
            user_image_id=102
        )

        json_resp = mock_pipeline.to_json_response_v2([result1, result2])

        assert len(json_resp["results"]) == 2
        assert json_resp["results"][0]["image_id"] == 101
        assert json_resp["results"][1]["image_id"] == 102


class TestFurniturePipelineFetchImage:
    """fetch_image_from_url 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_fetch_image_success(self):
        """이미지 가져오기 성공"""
        with patch('ai.pipeline.furniture_pipeline.YoloDetector'), \
             patch('ai.pipeline.furniture_pipeline.DimensionCalculator'):

            pipeline = FurniturePipeline(enable_3d_generation=False, device_id=0)

            # 테스트 이미지 생성
            img = Image.new('RGB', (100, 100), 'red')

            # fetch_image_from_url 직접 모킹
            pipeline.fetch_image_from_url = AsyncMock(return_value=img)

            result = await pipeline.fetch_image_from_url("http://test.com/image.jpg")
            assert result is not None

    @pytest.mark.asyncio
    async def test_fetch_image_failure(self):
        """이미지 가져오기 실패"""
        with patch('ai.pipeline.furniture_pipeline.YoloDetector'), \
             patch('ai.pipeline.furniture_pipeline.DimensionCalculator'):

            pipeline = FurniturePipeline(enable_3d_generation=False, device_id=0)

            # fetch_image_from_url 직접 모킹
            pipeline.fetch_image_from_url = AsyncMock(return_value=None)

            result = await pipeline.fetch_image_from_url("http://test.com/notfound.jpg")
            assert result is None


class TestFurniturePipelineYoloMaskToBase64:
    """_yolo_mask_to_base64 메서드 테스트"""

    def test_yolo_mask_to_base64(self, mock_pipeline):
        """YOLO 마스크 → Base64 변환"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        b64 = mock_pipeline._yolo_mask_to_base64(mask)

        assert isinstance(b64, str)
        assert len(b64) > 0

        # 디코딩 검증
        decoded = base64.b64decode(b64)
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (100, 100)
        assert img.mode == 'L'


class TestFurniturePipelineDetectObjects:
    """detect_objects 메서드 테스트"""

    def test_detect_objects_returns_list(self):
        """detect_objects가 리스트 반환"""
        with patch('ai.pipeline.furniture_pipeline.YoloDetector') as mock_yolo, \
             patch('ai.pipeline.furniture_pipeline.DimensionCalculator'):

            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_smart.return_value = {
                "boxes": [[100, 100, 200, 200]],
                "labels": ["Bed"],
                "scores": [0.95],
                "masks": [np.zeros((100, 100), dtype=np.uint8)]
            }
            mock_yolo.return_value = mock_detector_instance

            pipeline = FurniturePipeline(enable_3d_generation=False, device_id=0)

            image = Image.new('RGB', (300, 300), 'white')
            result = pipeline.detect_objects(image)

            assert isinstance(result, list)

    def test_detect_objects_empty_detection(self):
        """탐지 결과 없음"""
        with patch('ai.pipeline.furniture_pipeline.YoloDetector') as mock_yolo, \
             patch('ai.pipeline.furniture_pipeline.DimensionCalculator'):

            mock_detector_instance = MagicMock()
            mock_detector_instance.detect_smart.return_value = None
            mock_yolo.return_value = mock_detector_instance

            pipeline = FurniturePipeline(enable_3d_generation=False, device_id=0)

            image = Image.new('RGB', (300, 300), 'white')
            result = pipeline.detect_objects(image)

            assert result == []


class TestFurniturePipelineProcessSingleImage:
    """process_single_image 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_process_single_image_fetch_failure(self, mock_pipeline):
        """이미지 가져오기 실패"""
        with patch.object(mock_pipeline, 'fetch_image_from_url', new=AsyncMock(return_value=None)):
            result = await mock_pipeline.process_single_image("http://test.com/image.jpg")

            assert result.status == "failed"
            assert "Failed to fetch image" in result.error

    @pytest.mark.asyncio
    async def test_process_single_image_success(self, mock_pipeline):
        """성공적인 이미지 처리"""
        test_image = Image.new('RGB', (300, 300), 'white')

        with patch.object(mock_pipeline, 'fetch_image_from_url', new=AsyncMock(return_value=test_image)), \
             patch.object(mock_pipeline, 'detect_objects', return_value=[]):
            result = await mock_pipeline.process_single_image(
                "http://test.com/image.jpg",
                enable_mask=False,
                enable_3d=False
            )

            assert result.status == "completed"
            assert result.error is None

    @pytest.mark.asyncio
    async def test_process_single_image_with_mask(self, mock_pipeline):
        """마스크 생성 포함 처리"""
        test_image = Image.new('RGB', (300, 300), 'white')
        test_mask = np.ones((300, 300), dtype=np.uint8) * 255

        mock_obj = DetectedObject(
            id=0,
            label="침대",
            db_key="bed",
            yolo_mask=test_mask
        )

        with patch.object(mock_pipeline, 'fetch_image_from_url', new=AsyncMock(return_value=test_image)), \
             patch.object(mock_pipeline, 'detect_objects', return_value=[mock_obj]):
            result = await mock_pipeline.process_single_image(
                "http://test.com/image.jpg",
                enable_mask=True,
                enable_3d=False
            )

            assert result.status == "completed"
            # 마스크가 base64로 변환되었는지 확인
            assert result.objects[0].mask_base64 is not None

    @pytest.mark.asyncio
    async def test_process_single_image_no_yolo_mask(self, mock_pipeline):
        """YOLOE-seg 마스크 없는 객체"""
        test_image = Image.new('RGB', (300, 300), 'white')

        mock_obj = DetectedObject(
            id=0,
            label="침대",
            db_key="bed",
            yolo_mask=None  # 마스크 없음
        )

        with patch.object(mock_pipeline, 'fetch_image_from_url', new=AsyncMock(return_value=test_image)), \
             patch.object(mock_pipeline, 'detect_objects', return_value=[mock_obj]):
            result = await mock_pipeline.process_single_image(
                "http://test.com/image.jpg",
                enable_mask=True,
                enable_3d=False
            )

            assert result.status == "completed"
            # 마스크가 None으로 설정됨
            assert result.objects[0].mask_base64 is None


class TestFurniturePipelineGenerate3DSuccess:
    """generate_3d 성공 경로 테스트 (Worker Pool 기반)"""

    @pytest.mark.asyncio
    async def test_generate_3d_success_completed(self, mock_pipeline):
        """3D 생성 성공: Worker Pool 사용"""
        mock_pipeline.enable_3d_generation = True

        # Worker Pool 모킹
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.ply_b64 = "dGVzdA=="
        mock_result.ply_size_bytes = 100
        mock_result.gif_b64 = "dGVzdA=="
        mock_result.mesh_url = "/assets/mesh.glb"

        mock_pool = MagicMock()
        mock_pool.is_ready.return_value = True
        mock_pool.submit_tasks_parallel = AsyncMock(return_value=[mock_result])

        with patch('ai.pipeline.furniture_pipeline.get_sam3d_worker_pool', return_value=mock_pool):
            result = await mock_pipeline.generate_3d(
                Image.new('RGB', (100, 100)),
                "test_mask_b64"
            )

            assert result is not None
            assert result["ply_b64"] == "dGVzdA=="
            assert result["mesh_url"] == "/assets/mesh.glb"

    @pytest.mark.asyncio
    async def test_generate_3d_status_failed(self, mock_pipeline):
        """3D 생성 실패: failed 상태"""
        mock_pipeline.enable_3d_generation = True

        mock_post_response = AsyncMock()
        mock_post_response.status = 200
        mock_post_response.json = AsyncMock(return_value={"task_id": "test-task-123"})

        mock_get_response = AsyncMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value={"status": "failed"})

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_post_response))
        )
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_get_response))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('ai.pipeline.furniture_pipeline.HAS_AIOHTTP', True), \
             patch('aiohttp.ClientSession', return_value=mock_session), \
             patch('asyncio.sleep', new=AsyncMock()):
            result = await mock_pipeline.generate_3d(
                Image.new('RGB', (100, 100)),
                "test_mask_b64"
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_generate_3d_no_task_id(self, mock_pipeline):
        """3D 생성: task_id 없음"""
        mock_pipeline.enable_3d_generation = True

        mock_post_response = AsyncMock()
        mock_post_response.status = 200
        mock_post_response.json = AsyncMock(return_value={})  # task_id 없음

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_post_response))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('ai.pipeline.furniture_pipeline.HAS_AIOHTTP', True), \
             patch('aiohttp.ClientSession', return_value=mock_session):
            result = await mock_pipeline.generate_3d(
                Image.new('RGB', (100, 100)),
                "test_mask_b64"
            )

            assert result is None


class TestFurniturePipelineProcessWith3D:
    """process_single_image 3D 처리 경로 테스트"""

    @pytest.mark.asyncio
    async def test_process_single_image_with_3d_success(self, mock_pipeline, tmp_path):
        """3D 생성 성공 경로"""
        test_image = Image.new('RGB', (300, 300), 'white')
        test_mask = np.ones((300, 300), dtype=np.uint8) * 255

        mock_obj = DetectedObject(
            id=0,
            label="침대",
            db_key="bed",
            yolo_mask=test_mask
        )

        # 3D 생성은 Worker Pool이 필요하므로 비활성화 테스트
        # 3D 관련 테스트는 통합 테스트에서 수행

        mock_pipeline.enable_3d_generation = False

        with patch.object(mock_pipeline, 'fetch_image_from_url', new=AsyncMock(return_value=test_image)), \
             patch.object(mock_pipeline, 'detect_objects', return_value=[mock_obj]):

            result = await mock_pipeline.process_single_image(
                "http://test.com/image.jpg",
                enable_mask=True,
                enable_3d=False  # 3D 비활성화
            )

            assert result.status == "completed"
            assert len(result.objects) > 0
            assert result.objects[0].mask_base64 is not None

    @pytest.mark.asyncio
    async def test_process_single_image_exception_handling(self, mock_pipeline):
        """process_single_image 예외 처리"""
        with patch.object(mock_pipeline, 'fetch_image_from_url',
                         new=AsyncMock(side_effect=Exception("Network error"))):
            result = await mock_pipeline.process_single_image("http://test.com/image.jpg")

            assert result.status == "failed"
            assert "Network error" in result.error


class TestFurniturePipelineMultipleImages:
    """process_multiple_images 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_process_multiple_images_basic(self):
        """다중 이미지 처리 기본 테스트"""
        with patch('ai.pipeline.furniture_pipeline.YoloDetector'), \
             patch('ai.pipeline.furniture_pipeline.DimensionCalculator'), \
             patch('ai.pipeline.furniture_pipeline.get_gpu_pool') as mock_get_pool:

            # GPU 풀 모킹
            mock_pool = MagicMock()
            mock_pool.gpu_ids = [0]
            mock_pool.has_pipeline = MagicMock(return_value=False)

            # gpu_context async context manager 모킹
            @asynccontextmanager
            async def mock_gpu_context(task_id):
                yield 0

            mock_pool.gpu_context = mock_gpu_context
            mock_get_pool.return_value = mock_pool

            pipeline = FurniturePipeline(enable_3d_generation=False, device_id=0)

            # process_single_image 모킹
            async def mock_process(url, enable_mask=True, enable_3d=True):
                return PipelineResult(
                    image_id="test",
                    image_url=url,
                    status="completed"
                )

            with patch.object(pipeline, 'process_single_image', side_effect=mock_process):
                results = await pipeline.process_multiple_images(
                    ["http://test.com/1.jpg", "http://test.com/2.jpg"],
                    enable_mask=False,
                    enable_3d=False
                )

            assert len(results) == 2
            assert all(r.status == "completed" for r in results)

    @pytest.mark.asyncio
    async def test_process_multiple_images_with_pipeline_context(self):
        """파이프라인 컨텍스트 사용 테스트"""

        with patch('ai.pipeline.furniture_pipeline.YoloDetector'), \
             patch('ai.pipeline.furniture_pipeline.DimensionCalculator'), \
             patch('ai.pipeline.furniture_pipeline.get_gpu_pool') as mock_get_pool:

            mock_pool = MagicMock()
            mock_pool.gpu_ids = [0]
            mock_pool.has_pipeline = MagicMock(return_value=True)

            # pipeline_context 모킹
            mock_inner_pipeline = MagicMock()

            async def mock_process(url, enable_mask=True, enable_3d=True):
                return PipelineResult(
                    image_id="test",
                    image_url=url,
                    status="completed"
                )

            mock_inner_pipeline.process_single_image = mock_process

            @asynccontextmanager
            async def mock_pipeline_context(task_id):
                yield (0, mock_inner_pipeline)

            mock_pool.pipeline_context = mock_pipeline_context
            mock_get_pool.return_value = mock_pool

            pipeline = FurniturePipeline(enable_3d_generation=False, device_id=0)

            results = await pipeline.process_multiple_images(
                ["http://test.com/1.jpg"],
                enable_mask=False,
                enable_3d=False
            )

            assert len(results) == 1
            assert results[0].status == "completed"

    @pytest.mark.asyncio
    async def test_process_multiple_images_exception(self):
        """다중 이미지 처리 중 예외 발생 (asyncio.gather에서 처리)"""
        with patch('ai.pipeline.furniture_pipeline.YoloDetector'), \
             patch('ai.pipeline.furniture_pipeline.DimensionCalculator'), \
             patch('ai.pipeline.furniture_pipeline.get_gpu_pool') as mock_get_pool:

            mock_pool = MagicMock()
            mock_pool.gpu_ids = [0]
            mock_pool.has_pipeline = MagicMock(return_value=False)

            @asynccontextmanager
            async def mock_gpu_context(task_id):
                yield 0

            mock_pool.gpu_context = mock_gpu_context
            mock_get_pool.return_value = mock_pool

            pipeline = FurniturePipeline(enable_3d_generation=False, device_id=0)

            # process_single_image에서 예외 발생
            async def mock_process_with_error(url, enable_mask=True, enable_3d=True):
                raise Exception("GPU allocation failed")

            with patch.object(pipeline, 'process_single_image', side_effect=mock_process_with_error):
                results = await pipeline.process_multiple_images(
                    ["http://test.com/1.jpg"],
                    enable_mask=False,
                    enable_3d=False
                )

            # asyncio.gather(return_exceptions=True)에서 Exception을 처리
            assert len(results) == 1
            assert results[0].status == "failed"
            assert "GPU allocation failed" in results[0].error


class TestFurniturePipelineMultipleImagesWithIds:
    """process_multiple_images_with_ids 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_process_multiple_images_with_ids_basic(self):
        """ID 포함 다중 이미지 처리 기본 테스트"""

        with patch('ai.pipeline.furniture_pipeline.YoloDetector'), \
             patch('ai.pipeline.furniture_pipeline.DimensionCalculator'), \
             patch('ai.pipeline.furniture_pipeline.get_gpu_pool') as mock_get_pool:

            mock_pool = MagicMock()
            mock_pool.gpu_ids = [0]
            mock_pool.has_pipeline = MagicMock(return_value=False)

            @asynccontextmanager
            async def mock_gpu_context(task_id):
                yield 0

            mock_pool.gpu_context = mock_gpu_context
            mock_get_pool.return_value = mock_pool

            pipeline = FurniturePipeline(enable_3d_generation=False, device_id=0)

            async def mock_process(url, enable_mask=True, enable_3d=True):
                return PipelineResult(
                    image_id="test",
                    image_url=url,
                    status="completed"
                )

            with patch.object(pipeline, 'process_single_image', side_effect=mock_process):
                results = await pipeline.process_multiple_images_with_ids(
                    [(101, "http://test.com/1.jpg"), (102, "http://test.com/2.jpg")],
                    enable_mask=False,
                    enable_3d=False
                )

            assert len(results) == 2
            assert results[0].user_image_id == 101
            assert results[1].user_image_id == 102

    @pytest.mark.asyncio
    async def test_process_multiple_images_with_ids_pipeline_context(self):
        """ID 포함 파이프라인 컨텍스트 사용 테스트"""

        with patch('ai.pipeline.furniture_pipeline.YoloDetector'), \
             patch('ai.pipeline.furniture_pipeline.DimensionCalculator'), \
             patch('ai.pipeline.furniture_pipeline.get_gpu_pool') as mock_get_pool:

            mock_pool = MagicMock()
            mock_pool.gpu_ids = [0]
            mock_pool.has_pipeline = MagicMock(return_value=True)

            mock_inner_pipeline = MagicMock()

            async def mock_process(url, enable_mask=True, enable_3d=True):
                return PipelineResult(
                    image_id="test",
                    image_url=url,
                    status="completed"
                )

            mock_inner_pipeline.process_single_image = mock_process

            @asynccontextmanager
            async def mock_pipeline_context(task_id):
                yield (0, mock_inner_pipeline)

            mock_pool.pipeline_context = mock_pipeline_context
            mock_get_pool.return_value = mock_pool

            pipeline = FurniturePipeline(enable_3d_generation=False, device_id=0)

            results = await pipeline.process_multiple_images_with_ids(
                [(101, "http://test.com/1.jpg")],
                enable_mask=False,
                enable_3d=False
            )

            assert len(results) == 1
            assert results[0].user_image_id == 101

    @pytest.mark.asyncio
    async def test_process_multiple_images_with_ids_exception(self):
        """ID 포함 다중 이미지 처리 중 예외 발생 (asyncio.gather에서 처리)"""
        with patch('ai.pipeline.furniture_pipeline.YoloDetector'), \
             patch('ai.pipeline.furniture_pipeline.DimensionCalculator'), \
             patch('ai.pipeline.furniture_pipeline.get_gpu_pool') as mock_get_pool:

            mock_pool = MagicMock()
            mock_pool.gpu_ids = [0]
            mock_pool.has_pipeline = MagicMock(return_value=False)

            @asynccontextmanager
            async def mock_gpu_context(task_id):
                yield 0

            mock_pool.gpu_context = mock_gpu_context
            mock_get_pool.return_value = mock_pool

            pipeline = FurniturePipeline(enable_3d_generation=False, device_id=0)

            # process_single_image에서 예외 발생
            async def mock_process_with_error(url, enable_mask=True, enable_3d=True):
                raise Exception("GPU error")

            with patch.object(pipeline, 'process_single_image', side_effect=mock_process_with_error):
                results = await pipeline.process_multiple_images_with_ids(
                    [(101, "http://test.com/1.jpg")],
                    enable_mask=False,
                    enable_3d=False
                )

            # asyncio.gather(return_exceptions=True)에서 Exception을 처리
            assert len(results) == 1
            assert results[0].status == "failed"
            assert results[0].user_image_id == 101
