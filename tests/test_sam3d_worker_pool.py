"""
SAM3D Worker Pool 테스트

ai/gpu/sam3d_worker_pool.py 커버리지 향상을 위한 단위/통합 테스트
"""

import asyncio
import subprocess
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
import pytest

from ai.gpu.sam3d_worker_pool import (
    SAM3DWorkerPool,
    WorkerInfo,
    get_sam3d_worker_pool,
    initialize_sam3d_worker_pool,
    shutdown_sam3d_worker_pool,
)
from ai.subprocess.worker_protocol import (
    MessageType,
    ResultMessage,
    InitMessage,
    TaskMessage,
)


class TestWorkerInfo:
    """WorkerInfo 데이터클래스 테스트"""

    def test_default_values(self):
        """기본값으로 WorkerInfo 생성"""
        info = WorkerInfo(worker_id=0, gpu_id=0)
        assert info.worker_id == 0
        assert info.gpu_id == 0
        assert info.process is None
        assert info.is_ready is False
        assert info.is_busy is False
        assert info.current_task_id is None
        assert info.error_count == 0
        assert info.last_activity == 0.0

    def test_custom_values(self):
        """사용자 정의 값으로 WorkerInfo 생성"""
        mock_process = MagicMock()
        info = WorkerInfo(
            worker_id=1,
            gpu_id=2,
            process=mock_process,
            is_ready=True,
            is_busy=True,
            current_task_id="task_123",
            error_count=5,
            last_activity=100.0
        )
        assert info.worker_id == 1
        assert info.gpu_id == 2
        assert info.process == mock_process
        assert info.is_ready is True
        assert info.is_busy is True
        assert info.current_task_id == "task_123"
        assert info.error_count == 5
        assert info.last_activity == 100.0


class TestSAM3DWorkerPoolInit:
    """SAM3DWorkerPool 초기화 테스트"""

    def test_init_with_gpu_ids(self):
        """GPU ID 목록으로 초기화"""
        pool = SAM3DWorkerPool(gpu_ids=[0, 1, 2])
        assert pool.gpu_ids == [0, 1, 2]
        assert len(pool._workers) == 3
        assert 0 in pool._workers
        assert 1 in pool._workers
        assert 2 in pool._workers

    def test_init_auto_detect_gpus(self):
        """GPU 자동 감지 - 실제 환경 테스트"""
        # 실제 torch 사용
        import torch
        if torch.cuda.is_available():
            pool = SAM3DWorkerPool(gpu_ids=None)
            assert len(pool.gpu_ids) == torch.cuda.device_count()
        else:
            pool = SAM3DWorkerPool(gpu_ids=None)
            assert pool.gpu_ids == [0]

    def test_init_explicit_gpu_ids(self):
        """명시적 GPU ID로 초기화"""
        pool = SAM3DWorkerPool(gpu_ids=[2, 3])
        assert pool.gpu_ids == [2, 3]

    def test_init_with_custom_timeout(self):
        """커스텀 타임아웃 설정"""
        pool = SAM3DWorkerPool(
            gpu_ids=[0],
            init_timeout=300.0,
            task_timeout=600.0
        )
        assert pool.init_timeout == 300.0
        assert pool.task_timeout == 600.0

    def test_init_worker_script_path(self):
        """워커 스크립트 경로 설정"""
        pool = SAM3DWorkerPool(gpu_ids=[0], worker_script="/custom/path/worker.py")
        assert pool.worker_script == "/custom/path/worker.py"

    def test_init_python_executable(self):
        """Python 실행 경로 설정"""
        pool = SAM3DWorkerPool(gpu_ids=[0], python_executable="/usr/bin/python3")
        assert pool.python_executable == "/usr/bin/python3"


class TestSAM3DWorkerPoolStatus:
    """SAM3DWorkerPool 상태 조회 테스트"""

    def test_get_status_initial(self):
        """초기 상태 조회"""
        pool = SAM3DWorkerPool(gpu_ids=[0, 1])
        status = pool.get_status()

        assert status["total_workers"] == 2
        assert status["ready_workers"] == 0
        assert status["busy_workers"] == 0
        assert 0 in status["workers"]
        assert 1 in status["workers"]

    def test_get_status_with_ready_workers(self):
        """준비된 워커가 있는 상태"""
        pool = SAM3DWorkerPool(gpu_ids=[0, 1])
        pool._workers[0].is_ready = True
        pool._workers[1].is_ready = True
        pool._workers[1].is_busy = True
        pool._workers[1].current_task_id = "task_1"

        status = pool.get_status()
        assert status["ready_workers"] == 2
        assert status["busy_workers"] == 1
        assert status["workers"][0]["is_ready"] is True
        assert status["workers"][1]["is_busy"] is True
        assert status["workers"][1]["current_task_id"] == "task_1"

    def test_is_ready_false_when_not_started(self):
        """시작되지 않은 상태"""
        pool = SAM3DWorkerPool(gpu_ids=[0])
        assert pool.is_ready() is False

    def test_is_ready_true_when_started_and_workers_ready(self):
        """시작되고 워커가 준비된 상태"""
        pool = SAM3DWorkerPool(gpu_ids=[0])
        pool._started = True
        pool._workers[0].is_ready = True
        assert pool.is_ready() is True

    def test_is_ready_false_when_no_workers_ready(self):
        """시작됐지만 워커가 준비 안됨"""
        pool = SAM3DWorkerPool(gpu_ids=[0])
        pool._started = True
        pool._workers[0].is_ready = False
        assert pool.is_ready() is False


class TestSAM3DWorkerPoolSubmit:
    """SAM3DWorkerPool 작업 제출 테스트"""

    @pytest.mark.asyncio
    async def test_submit_task_no_available_workers(self):
        """워커 없을 때 작업 제출"""
        pool = SAM3DWorkerPool(gpu_ids=[0], task_timeout=0.1)
        pool._started = True
        # 워커가 준비되지 않음

        result = await pool.submit_task(
            task_id="test_task",
            image_b64="base64_image",
            mask_b64="base64_mask"
        )

        assert result.success is False
        assert "No available workers" in result.error

    @pytest.mark.asyncio
    async def test_acquire_worker_round_robin(self):
        """라운드로빈 워커 할당"""
        pool = SAM3DWorkerPool(gpu_ids=[0, 1, 2])
        pool._workers[0].is_ready = True
        pool._workers[1].is_ready = True
        pool._workers[2].is_ready = True

        # 첫 번째 할당
        worker1 = await pool._acquire_worker("task_1")
        assert worker1.gpu_id == 0

        # 다음 라운드로빈 위치 확인 후 첫 번째 워커 반환
        await pool._release_worker(0)

        # 두 번째 할당
        worker2 = await pool._acquire_worker("task_2")
        assert worker2.gpu_id == 1
        await pool._release_worker(1)

        # 세 번째 할당
        worker3 = await pool._acquire_worker("task_3")
        assert worker3.gpu_id == 2

    @pytest.mark.asyncio
    async def test_release_worker(self):
        """워커 반환"""
        pool = SAM3DWorkerPool(gpu_ids=[0])
        pool._workers[0].is_ready = True
        pool._workers[0].is_busy = True
        pool._workers[0].current_task_id = "task_1"

        await pool._release_worker(0)

        assert pool._workers[0].is_busy is False
        assert pool._workers[0].current_task_id is None

    @pytest.mark.asyncio
    async def test_release_worker_invalid_gpu(self):
        """잘못된 GPU ID로 워커 반환"""
        pool = SAM3DWorkerPool(gpu_ids=[0])
        # 존재하지 않는 GPU ID로 반환 시도
        await pool._release_worker(999)  # 에러 없이 통과해야 함


class TestSAM3DWorkerPoolParallel:
    """SAM3DWorkerPool 병렬 작업 테스트"""

    @pytest.mark.asyncio
    async def test_submit_tasks_parallel_no_tasks(self):
        """빈 작업 목록"""
        pool = SAM3DWorkerPool(gpu_ids=[0])
        results = await pool.submit_tasks_parallel([])
        assert results == []

    @pytest.mark.asyncio
    async def test_submit_tasks_parallel_exception_handling(self):
        """예외 발생 시 처리"""
        pool = SAM3DWorkerPool(gpu_ids=[0], task_timeout=0.1)
        pool._started = True

        tasks = [
            {"task_id": "task_1", "image_b64": "img1", "mask_b64": "mask1"},
            {"task_id": "task_2", "image_b64": "img2", "mask_b64": "mask2"},
        ]

        results = await pool.submit_tasks_parallel(tasks)

        assert len(results) == 2
        # 워커가 준비 안됐으므로 실패
        for result in results:
            assert result.success is False


class TestSAM3DWorkerPoolShutdown:
    """SAM3DWorkerPool 종료 테스트"""

    @pytest.mark.asyncio
    async def test_shutdown_no_workers(self):
        """워커 없이 종료"""
        pool = SAM3DWorkerPool(gpu_ids=[0])
        await pool.shutdown()
        assert pool._started is False

    @pytest.mark.asyncio
    async def test_shutdown_with_process(self):
        """프로세스가 있는 상태로 종료"""
        pool = SAM3DWorkerPool(gpu_ids=[0])

        # Mock process
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.flush = MagicMock()
        mock_process.wait = MagicMock(return_value=0)

        pool._workers[0].process = mock_process
        pool._started = True

        await pool.shutdown()

        assert pool._started is False
        assert pool._workers[0].process is None
        mock_process.stdin.write.assert_called()
        mock_process.wait.assert_called()

    @pytest.mark.asyncio
    async def test_shutdown_process_timeout(self):
        """종료 시 타임아웃"""
        pool = SAM3DWorkerPool(gpu_ids=[0])

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.flush = MagicMock()
        mock_process.wait = MagicMock(side_effect=subprocess.TimeoutExpired(cmd="test", timeout=5))
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()

        pool._workers[0].process = mock_process
        pool._started = True

        await pool.shutdown()

        mock_process.terminate.assert_called()


class TestSAM3DWorkerPoolStartWorkers:
    """SAM3DWorkerPool 워커 시작 테스트"""

    @pytest.mark.asyncio
    async def test_start_workers_already_started(self):
        """이미 시작된 상태에서 재시작 시도"""
        pool = SAM3DWorkerPool(gpu_ids=[0])
        pool._started = True

        # 이미 시작됐으므로 early return
        await pool.start_workers()
        assert pool._started is True


class TestGlobalFunctions:
    """글로벌 함수 테스트"""

    def test_get_sam3d_worker_pool_not_initialized(self):
        """초기화 전 글로벌 풀 조회"""
        import ai.gpu.sam3d_worker_pool as module
        original = module._global_sam3d_pool
        module._global_sam3d_pool = None

        result = get_sam3d_worker_pool()
        assert result is None

        module._global_sam3d_pool = original

    def test_get_sam3d_worker_pool_initialized(self):
        """초기화 후 글로벌 풀 조회"""
        import ai.gpu.sam3d_worker_pool as module
        original = module._global_sam3d_pool

        mock_pool = MagicMock()
        module._global_sam3d_pool = mock_pool

        result = get_sam3d_worker_pool()
        assert result == mock_pool

        module._global_sam3d_pool = original

    @pytest.mark.asyncio
    async def test_shutdown_sam3d_worker_pool_not_initialized(self):
        """초기화 안된 상태에서 종료"""
        import ai.gpu.sam3d_worker_pool as module
        original = module._global_sam3d_pool
        module._global_sam3d_pool = None

        await shutdown_sam3d_worker_pool()  # 에러 없이 통과

        module._global_sam3d_pool = original

    @pytest.mark.asyncio
    async def test_shutdown_sam3d_worker_pool_initialized(self):
        """초기화된 상태에서 종료"""
        import ai.gpu.sam3d_worker_pool as module
        original = module._global_sam3d_pool

        mock_pool = MagicMock()
        mock_pool.shutdown = AsyncMock()
        module._global_sam3d_pool = mock_pool

        await shutdown_sam3d_worker_pool()

        mock_pool.shutdown.assert_called_once()
        assert module._global_sam3d_pool is None

        module._global_sam3d_pool = original


class TestWaitForResult:
    """결과 대기 테스트"""

    @pytest.mark.asyncio
    async def test_wait_for_result_no_process(self):
        """프로세스 없이 결과 대기"""
        pool = SAM3DWorkerPool(gpu_ids=[0])
        pool._workers[0].process = None

        result = await pool._wait_for_result(0, "task_1")

        assert result.success is False
        assert "Worker not running" in result.error

    @pytest.mark.asyncio
    async def test_wait_for_result_process_terminated(self):
        """프로세스가 종료된 경우"""
        pool = SAM3DWorkerPool(gpu_ids=[0], task_timeout=0.5)

        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = MagicMock(return_value="")
        mock_process.poll = MagicMock(return_value=1)  # 종료됨

        pool._workers[0].process = mock_process

        result = await pool._wait_for_result(0, "task_1")

        assert result.success is False
        assert "terminated" in result.error


class TestResultMessage:
    """ResultMessage 테스트"""

    def test_result_message_success(self):
        """성공 결과 메시지"""
        msg = ResultMessage(
            task_id="test_task",
            success=True,
            ply_b64="base64_ply_data",
            processing_time_seconds=1.5
        )
        assert msg.task_id == "test_task"
        assert msg.success is True
        assert msg.ply_b64 == "base64_ply_data"
        assert msg.processing_time_seconds == 1.5

    def test_result_message_failure(self):
        """실패 결과 메시지"""
        msg = ResultMessage(
            task_id="test_task",
            success=False,
            error="Processing failed"
        )
        assert msg.task_id == "test_task"
        assert msg.success is False
        assert msg.error == "Processing failed"
