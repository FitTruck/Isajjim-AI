"""
GPU Pool Manager for Multi-GPU Parallel Processing

Multi-GPU 병렬 처리를 위한 GPU 풀 매니저입니다.

Features:
- 라운드로빈 GPU 할당
- Thread-safe 획득/반환
- GPU 헬스체크
- 자동 페일오버
- GPU별 파이프라인 사전 초기화 및 재사용
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class GPUInfo:
    """GPU 상태 정보"""
    device_id: int
    is_available: bool = True
    current_task_id: Optional[str] = None
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    last_health_check: float = 0.0
    error_count: int = 0
    pipeline: Any = None  # 사전 초기화된 FurniturePipeline


class GPUPoolManager:
    """
    Multi-GPU 리소스 풀 매니저

    라운드로빈 방식으로 GPU를 할당하고, 헬스체크를 통해
    안정적인 GPU 사용을 보장합니다.

    Usage:
        pool = GPUPoolManager(gpu_ids=[0, 1, 2, 3])

        async with pool.gpu_context() as gpu_id:
            # gpu_id가 이 컨텍스트에서 전용으로 사용됨
            pipeline.process(device_id=gpu_id)
    """

    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        max_retries: int = 3,
        health_check_interval: float = 30.0,
        wait_timeout: float = 300.0
    ):
        """
        Args:
            gpu_ids: 사용할 GPU ID 목록 (None이면 자동 감지)
            max_retries: GPU 오류 시 최대 재시도 횟수
            health_check_interval: 헬스체크 간격 (초)
            wait_timeout: GPU 대기 최대 시간 (초)
        """
        # GPU 자동 감지
        if gpu_ids is None:
            if HAS_TORCH and torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
            else:
                gpu_ids = []

        self.gpu_ids = gpu_ids
        self.max_retries = max_retries
        self.health_check_interval = health_check_interval
        self.wait_timeout = wait_timeout

        # GPU 레지스트리
        self._gpus: Dict[int, GPUInfo] = {}
        for gpu_id in gpu_ids:
            self._gpus[gpu_id] = GPUInfo(device_id=gpu_id)
            # 초기 메모리 정보 설정
            if HAS_TORCH and torch.cuda.is_available():
                try:
                    props = torch.cuda.get_device_properties(gpu_id)
                    self._gpus[gpu_id].memory_total_mb = props.total_memory / 1024 / 1024
                except Exception:
                    pass

        # 라운드로빈 카운터
        self._next_gpu_index = 0

        # Thread-safe를 위한 락
        self._allocation_lock = asyncio.Lock()

        print(f"[GPUPoolManager] Initialized with {len(gpu_ids)} GPUs: {gpu_ids}")

    async def acquire(self, task_id: Optional[str] = None) -> int:
        """
        라운드로빈 방식으로 GPU를 할당합니다.

        Args:
            task_id: 태스크 식별자 (로깅/디버깅용)

        Returns:
            gpu_id: 할당된 GPU 디바이스 ID

        Raises:
            RuntimeError: 사용 가능한 GPU가 없을 때
        """
        if not self.gpu_ids:
            raise RuntimeError("No GPUs available in pool")

        start_time = time.time()

        while True:
            async with self._allocation_lock:
                # 모든 GPU 순회하며 사용 가능한 것 찾기
                for _ in range(len(self.gpu_ids)):
                    gpu_id = self.gpu_ids[self._next_gpu_index]
                    self._next_gpu_index = (self._next_gpu_index + 1) % len(self.gpu_ids)

                    gpu_info = self._gpus[gpu_id]

                    if gpu_info.is_available and gpu_info.error_count < self.max_retries:
                        # GPU 헬스체크
                        if await self._check_gpu_health(gpu_id):
                            gpu_info.is_available = False
                            gpu_info.current_task_id = task_id
                            print(f"[GPUPoolManager] Acquired GPU {gpu_id} for task {task_id}")
                            return gpu_id

            # 타임아웃 체크
            elapsed = time.time() - start_time
            if elapsed > self.wait_timeout:
                raise RuntimeError(f"Timeout waiting for GPU (waited {elapsed:.1f}s)")

            # 잠시 대기 후 재시도
            await asyncio.sleep(0.5)

    async def release(self, gpu_id: int):
        """
        GPU를 풀에 반환합니다.

        Args:
            gpu_id: 반환할 GPU ID
        """
        if gpu_id in self._gpus:
            task_id = self._gpus[gpu_id].current_task_id
            self._gpus[gpu_id].is_available = True
            self._gpus[gpu_id].current_task_id = None
            print(f"[GPUPoolManager] Released GPU {gpu_id} (was task {task_id})")

    async def _check_gpu_health(self, gpu_id: int) -> bool:
        """
        GPU 상태를 체크합니다.

        Args:
            gpu_id: 체크할 GPU ID

        Returns:
            True if healthy, False otherwise
        """
        if not HAS_TORCH or not torch.cuda.is_available():
            return True  # torch 없으면 체크 스킵

        try:
            # 현재 시간 체크 - 최근에 체크했으면 스킵
            now = time.time()
            if now - self._gpus[gpu_id].last_health_check < self.health_check_interval:
                return True

            # GPU에 작은 텐서 할당하여 동작 확인
            with torch.cuda.device(gpu_id):
                test_tensor = torch.zeros(1, device=f"cuda:{gpu_id}")
                del test_tensor
                torch.cuda.empty_cache()

            # 메모리 사용량 업데이트
            self._gpus[gpu_id].memory_used_mb = (
                torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
            )
            self._gpus[gpu_id].last_health_check = now

            return True

        except Exception as e:
            self._gpus[gpu_id].error_count += 1
            print(f"[GPUPoolManager] GPU {gpu_id} health check failed: {e}")
            return False

    @asynccontextmanager
    async def gpu_context(self, task_id: Optional[str] = None):
        """
        GPU 획득을 위한 async 컨텍스트 매니저

        Usage:
            async with pool.gpu_context(task_id="image_1") as gpu_id:
                # gpu_id 사용
                process_on_gpu(gpu_id)
            # 자동으로 GPU 반환됨
        """
        gpu_id = await self.acquire(task_id)
        try:
            yield gpu_id
        finally:
            await self.release(gpu_id)

    def get_status(self) -> Dict:
        """
        풀 상태를 반환합니다.

        Returns:
            {
                "total_gpus": 4,
                "available_gpus": 2,
                "gpus": {
                    0: {"available": true, "task_id": null, "memory_mb": 1024, ...},
                    ...
                }
            }
        """
        return {
            "total_gpus": len(self.gpu_ids),
            "available_gpus": sum(1 for g in self._gpus.values() if g.is_available),
            "gpus": {
                gpu_id: {
                    "available": info.is_available,
                    "task_id": info.current_task_id,
                    "memory_used_mb": round(info.memory_used_mb, 2),
                    "memory_total_mb": round(info.memory_total_mb, 2),
                    "error_count": info.error_count,
                    "last_health_check": info.last_health_check
                }
                for gpu_id, info in self._gpus.items()
            }
        }

    def reset_error_count(self, gpu_id: int):
        """GPU 에러 카운트를 리셋합니다."""
        if gpu_id in self._gpus:
            self._gpus[gpu_id].error_count = 0

    def is_available(self, gpu_id: int) -> bool:
        """특정 GPU가 사용 가능한지 확인합니다."""
        if gpu_id not in self._gpus:
            return False
        return self._gpus[gpu_id].is_available

    # =========================================================================
    # 파이프라인 사전 초기화 기능
    # =========================================================================

    def register_pipeline(self, gpu_id: int, pipeline: Any):
        """
        GPU에 사전 초기화된 파이프라인을 등록합니다.

        Args:
            gpu_id: GPU ID
            pipeline: 초기화된 FurniturePipeline 인스턴스
        """
        if gpu_id in self._gpus:
            self._gpus[gpu_id].pipeline = pipeline
            print(f"[GPUPoolManager] Registered pipeline for GPU {gpu_id}")

    def get_pipeline(self, gpu_id: int) -> Optional[Any]:
        """
        GPU에 등록된 파이프라인을 가져옵니다.

        Args:
            gpu_id: GPU ID

        Returns:
            등록된 FurniturePipeline 또는 None
        """
        if gpu_id in self._gpus:
            return self._gpus[gpu_id].pipeline
        return None

    def has_pipeline(self, gpu_id: int) -> bool:
        """GPU에 파이프라인이 등록되어 있는지 확인합니다."""
        if gpu_id not in self._gpus:
            return False
        return self._gpus[gpu_id].pipeline is not None

    async def initialize_pipelines(
        self,
        pipeline_factory: Callable[[int], Any],
        skip_on_error: bool = True
    ):
        """
        모든 GPU에 파이프라인을 사전 초기화합니다.

        Args:
            pipeline_factory: GPU ID를 받아 파이프라인을 생성하는 팩토리 함수
                              예: lambda gpu_id: FurniturePipeline(device_id=gpu_id)
            skip_on_error: 에러 발생 시 해당 GPU 건너뛰기

        Example:
            await pool.initialize_pipelines(
                lambda gpu_id: FurniturePipeline(device_id=gpu_id)
            )
        """
        print(f"[GPUPoolManager] Initializing pipelines for {len(self.gpu_ids)} GPUs...")

        for gpu_id in self.gpu_ids:
            try:
                print(f"[GPUPoolManager] Creating pipeline for GPU {gpu_id}...")
                pipeline = pipeline_factory(gpu_id)
                self.register_pipeline(gpu_id, pipeline)
            except Exception as e:
                print(f"[GPUPoolManager] Failed to initialize pipeline for GPU {gpu_id}: {e}")
                if not skip_on_error:
                    raise

        initialized_count = sum(1 for g in self._gpus.values() if g.pipeline is not None)
        print(f"[GPUPoolManager] Initialized {initialized_count}/{len(self.gpu_ids)} pipelines")

    @asynccontextmanager
    async def pipeline_context(self, task_id: Optional[str] = None):
        """
        사전 초기화된 파이프라인과 함께 GPU를 획득합니다.

        Usage:
            async with pool.pipeline_context(task_id="image_1") as (gpu_id, pipeline):
                result = await pipeline.process_single_image(url)
            # 자동으로 GPU 반환됨

        Yields:
            (gpu_id, pipeline) 튜플
        """
        gpu_id = await self.acquire(task_id)
        try:
            pipeline = self.get_pipeline(gpu_id)
            if pipeline is None:
                raise RuntimeError(f"No pipeline registered for GPU {gpu_id}")
            yield gpu_id, pipeline
        finally:
            await self.release(gpu_id)

    def get_pipelines_status(self) -> Dict:
        """파이프라인 초기화 상태를 반환합니다."""
        return {
            "total_gpus": len(self.gpu_ids),
            "initialized_pipelines": sum(1 for g in self._gpus.values() if g.pipeline is not None),
            "gpus": {
                gpu_id: {
                    "has_pipeline": info.pipeline is not None,
                    "available": info.is_available
                }
                for gpu_id, info in self._gpus.items()
            }
        }


# Singleton 인스턴스
_global_pool: Optional[GPUPoolManager] = None


def get_gpu_pool() -> GPUPoolManager:
    """
    글로벌 GPU 풀을 가져옵니다.

    풀이 초기화되지 않았으면 자동으로 생성합니다.
    """
    global _global_pool
    if _global_pool is None:
        _global_pool = GPUPoolManager()
    return _global_pool


def initialize_gpu_pool(gpu_ids: Optional[List[int]] = None) -> GPUPoolManager:
    """
    글로벌 GPU 풀을 초기화합니다.

    Args:
        gpu_ids: 사용할 GPU ID 목록 (None이면 자동 감지)

    Returns:
        초기화된 GPUPoolManager 인스턴스
    """
    global _global_pool
    _global_pool = GPUPoolManager(gpu_ids=gpu_ids)
    return _global_pool
