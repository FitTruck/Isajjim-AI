"""
Tests for ai/gpu/gpu_pool_manager.py

GPUPoolManager 클래스의 단위 테스트:
- 초기화
- GPU 획득/반환
- 라운드로빈 할당
- 헬스체크
- 파이프라인 관리
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from ai.gpu.gpu_pool_manager import (
    GPUPoolManager,
    GPUInfo,
    get_gpu_pool,
    initialize_gpu_pool,
)


class TestGPUInfo:
    """GPUInfo 데이터클래스 테스트"""

    def test_create_gpu_info(self):
        """GPUInfo 생성"""
        info = GPUInfo(device_id=0)
        assert info.device_id == 0
        assert info.is_available is True
        assert info.current_task_id is None
        assert info.memory_used_mb == 0.0
        assert info.error_count == 0
        assert info.pipeline is None

    def test_modify_gpu_info(self):
        """GPUInfo 수정"""
        info = GPUInfo(device_id=1)
        info.is_available = False
        info.current_task_id = "task_123"
        info.memory_used_mb = 1024.5

        assert info.is_available is False
        assert info.current_task_id == "task_123"
        assert info.memory_used_mb == 1024.5


class TestGPUPoolManagerInit:
    """GPUPoolManager 초기화 테스트"""

    def test_init_with_gpu_ids(self):
        """GPU ID 지정 초기화"""
        pool = GPUPoolManager(gpu_ids=[0, 1])
        assert pool.gpu_ids == [0, 1]
        assert len(pool._gpus) == 2
        assert 0 in pool._gpus
        assert 1 in pool._gpus

    def test_init_empty_gpu_list(self):
        """빈 GPU 리스트로 초기화"""
        pool = GPUPoolManager(gpu_ids=[])
        assert pool.gpu_ids == []
        assert len(pool._gpus) == 0

    def test_init_default_settings(self):
        """기본 설정 확인"""
        pool = GPUPoolManager(gpu_ids=[0])
        assert pool.max_retries == 3
        assert pool.health_check_interval == 30.0
        assert pool.wait_timeout == 300.0

    def test_init_custom_settings(self):
        """사용자 정의 설정"""
        pool = GPUPoolManager(
            gpu_ids=[0],
            max_retries=5,
            health_check_interval=60.0,
            wait_timeout=600.0
        )
        assert pool.max_retries == 5
        assert pool.health_check_interval == 60.0
        assert pool.wait_timeout == 600.0


class TestGPUPoolManagerAcquireRelease:
    """GPU 획득/반환 테스트"""

    @pytest.mark.asyncio
    async def test_acquire_single_gpu(self):
        """단일 GPU 획득"""
        pool = GPUPoolManager(gpu_ids=[0])

        # 헬스체크 모킹
        pool._check_gpu_health = AsyncMock(return_value=True)

        gpu_id = await pool.acquire(task_id="test_task")

        assert gpu_id == 0
        assert pool._gpus[0].is_available is False
        assert pool._gpus[0].current_task_id == "test_task"

    @pytest.mark.asyncio
    async def test_release_gpu(self):
        """GPU 반환"""
        pool = GPUPoolManager(gpu_ids=[0])
        pool._gpus[0].is_available = False
        pool._gpus[0].current_task_id = "test_task"

        await pool.release(0)

        assert pool._gpus[0].is_available is True
        assert pool._gpus[0].current_task_id is None

    @pytest.mark.asyncio
    async def test_acquire_no_gpus_raises_error(self):
        """GPU가 없으면 에러"""
        pool = GPUPoolManager(gpu_ids=[])

        with pytest.raises(RuntimeError, match="No GPUs available"):
            await pool.acquire()

    @pytest.mark.asyncio
    async def test_round_robin_allocation(self):
        """라운드로빈 할당"""
        pool = GPUPoolManager(gpu_ids=[0, 1, 2])
        pool._check_gpu_health = AsyncMock(return_value=True)

        # 첫 번째 획득
        gpu1 = await pool.acquire(task_id="task1")
        await pool.release(gpu1)

        # 두 번째 획득 - 다음 GPU
        gpu2 = await pool.acquire(task_id="task2")
        await pool.release(gpu2)

        # 세 번째 획득 - 다음 GPU
        gpu3 = await pool.acquire(task_id="task3")

        # 라운드로빈으로 순차 할당
        assert gpu1 == 0
        assert gpu2 == 1
        assert gpu3 == 2

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """타임아웃 테스트"""
        pool = GPUPoolManager(gpu_ids=[0], wait_timeout=0.1)
        pool._check_gpu_health = AsyncMock(return_value=True)

        # GPU를 먼저 사용 중으로 설정
        pool._gpus[0].is_available = False

        with pytest.raises(RuntimeError, match="Timeout waiting for GPU"):
            await pool.acquire()


class TestGPUPoolManagerContext:
    """gpu_context 컨텍스트 매니저 테스트"""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """컨텍스트 매니저 사용"""
        pool = GPUPoolManager(gpu_ids=[0])
        pool._check_gpu_health = AsyncMock(return_value=True)

        async with pool.gpu_context(task_id="ctx_task") as gpu_id:
            assert gpu_id == 0
            assert pool._gpus[0].is_available is False

        # 컨텍스트 종료 후 반환됨
        assert pool._gpus[0].is_available is True

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self):
        """컨텍스트에서 예외 발생 시에도 반환"""
        pool = GPUPoolManager(gpu_ids=[0])
        pool._check_gpu_health = AsyncMock(return_value=True)

        with pytest.raises(ValueError):
            async with pool.gpu_context(task_id="error_task") as gpu_id:
                assert pool._gpus[0].is_available is False
                raise ValueError("Test error")

        # 예외 후에도 GPU 반환됨
        assert pool._gpus[0].is_available is True


class TestGPUPoolManagerStatus:
    """상태 조회 테스트"""

    def test_get_status(self):
        """상태 조회"""
        pool = GPUPoolManager(gpu_ids=[0, 1])
        pool._gpus[0].is_available = False
        pool._gpus[0].current_task_id = "task_1"
        pool._gpus[0].memory_used_mb = 512.5

        status = pool.get_status()

        assert status["total_gpus"] == 2
        assert status["available_gpus"] == 1  # GPU 1만 available
        assert status["gpus"][0]["available"] is False
        assert status["gpus"][0]["task_id"] == "task_1"
        assert status["gpus"][1]["available"] is True

    def test_is_available(self):
        """GPU 가용성 확인"""
        pool = GPUPoolManager(gpu_ids=[0, 1])
        pool._gpus[0].is_available = False

        assert pool.is_available(0) is False
        assert pool.is_available(1) is True
        assert pool.is_available(99) is False  # 존재하지 않는 GPU


class TestGPUPoolManagerHealthCheck:
    """헬스체크 테스트"""

    @pytest.mark.asyncio
    async def test_health_check_skip_recent(self):
        """최근 체크된 GPU는 스킵"""
        pool = GPUPoolManager(gpu_ids=[0], health_check_interval=30.0)

        import time
        pool._gpus[0].last_health_check = time.time()  # 방금 체크됨

        result = await pool._check_gpu_health(0)
        assert result is True

    @pytest.mark.asyncio
    @patch('ai.gpu.gpu_pool_manager.HAS_TORCH', False)
    async def test_health_check_no_torch(self):
        """torch가 없으면 체크 스킵"""
        pool = GPUPoolManager(gpu_ids=[0])

        result = await pool._check_gpu_health(0)
        assert result is True


class TestGPUPoolManagerPipeline:
    """파이프라인 관리 테스트"""

    def test_register_pipeline(self):
        """파이프라인 등록"""
        pool = GPUPoolManager(gpu_ids=[0])
        mock_pipeline = MagicMock()

        pool.register_pipeline(0, mock_pipeline)

        assert pool._gpus[0].pipeline is mock_pipeline

    def test_get_pipeline(self):
        """파이프라인 조회"""
        pool = GPUPoolManager(gpu_ids=[0])
        mock_pipeline = MagicMock()
        pool._gpus[0].pipeline = mock_pipeline

        result = pool.get_pipeline(0)
        assert result is mock_pipeline

        result = pool.get_pipeline(99)  # 존재하지 않는 GPU
        assert result is None

    def test_has_pipeline(self):
        """파이프라인 존재 여부"""
        pool = GPUPoolManager(gpu_ids=[0, 1])
        pool._gpus[0].pipeline = MagicMock()

        assert pool.has_pipeline(0) is True
        assert pool.has_pipeline(1) is False
        assert pool.has_pipeline(99) is False

    @pytest.mark.asyncio
    async def test_initialize_pipelines(self):
        """파이프라인 사전 초기화"""
        pool = GPUPoolManager(gpu_ids=[0, 1])

        mock_factory = MagicMock(side_effect=[
            MagicMock(name="pipeline_0"),
            MagicMock(name="pipeline_1")
        ])

        await pool.initialize_pipelines(mock_factory)

        assert mock_factory.call_count == 2
        assert pool._gpus[0].pipeline is not None
        assert pool._gpus[1].pipeline is not None

    @pytest.mark.asyncio
    async def test_initialize_pipelines_with_error(self):
        """파이프라인 초기화 오류 (스킵)"""
        pool = GPUPoolManager(gpu_ids=[0, 1])

        def factory(gpu_id):
            if gpu_id == 0:
                raise RuntimeError("Init error")
            return MagicMock()

        await pool.initialize_pipelines(factory, skip_on_error=True)

        assert pool._gpus[0].pipeline is None  # 실패
        assert pool._gpus[1].pipeline is not None  # 성공

    @pytest.mark.asyncio
    async def test_pipeline_context(self):
        """파이프라인 컨텍스트"""
        pool = GPUPoolManager(gpu_ids=[0])
        pool._check_gpu_health = AsyncMock(return_value=True)
        mock_pipeline = MagicMock()
        pool._gpus[0].pipeline = mock_pipeline

        async with pool.pipeline_context(task_id="pipe_task") as (gpu_id, pipeline):
            assert gpu_id == 0
            assert pipeline is mock_pipeline

    @pytest.mark.asyncio
    async def test_pipeline_context_no_pipeline(self):
        """파이프라인 없으면 에러"""
        pool = GPUPoolManager(gpu_ids=[0])
        pool._check_gpu_health = AsyncMock(return_value=True)
        # 파이프라인 등록 안 함

        with pytest.raises(RuntimeError, match="No pipeline registered"):
            async with pool.pipeline_context():
                pass

    def test_get_pipelines_status(self):
        """파이프라인 상태 조회"""
        pool = GPUPoolManager(gpu_ids=[0, 1])
        pool._gpus[0].pipeline = MagicMock()

        status = pool.get_pipelines_status()

        assert status["total_gpus"] == 2
        assert status["initialized_pipelines"] == 1
        assert status["gpus"][0]["has_pipeline"] is True
        assert status["gpus"][1]["has_pipeline"] is False


class TestGPUPoolManagerErrorHandling:
    """에러 처리 테스트"""

    def test_reset_error_count(self):
        """에러 카운트 리셋"""
        pool = GPUPoolManager(gpu_ids=[0])
        pool._gpus[0].error_count = 5

        pool.reset_error_count(0)

        assert pool._gpus[0].error_count == 0

    @pytest.mark.asyncio
    async def test_skip_gpu_with_errors(self):
        """에러가 많은 GPU 스킵"""
        pool = GPUPoolManager(gpu_ids=[0, 1], max_retries=3)
        pool._check_gpu_health = AsyncMock(return_value=True)
        pool._gpus[0].error_count = 3  # max_retries에 도달

        gpu_id = await pool.acquire()

        # GPU 0은 스킵, GPU 1 할당
        assert gpu_id == 1


class TestGlobalPool:
    """글로벌 풀 테스트"""

    def test_initialize_gpu_pool(self):
        """글로벌 풀 초기화"""
        pool = initialize_gpu_pool(gpu_ids=[0, 1])

        assert pool is not None
        assert pool.gpu_ids == [0, 1]

    def test_get_gpu_pool(self):
        """글로벌 풀 가져오기"""
        # 먼저 초기화
        initialize_gpu_pool(gpu_ids=[0])

        pool = get_gpu_pool()

        assert pool is not None
        assert isinstance(pool, GPUPoolManager)

    def test_get_gpu_pool_auto_create(self):
        """글로벌 풀 자동 생성"""
        import ai.gpu.gpu_pool_manager as module
        module._global_pool = None  # 리셋

        pool = get_gpu_pool()

        assert pool is not None
