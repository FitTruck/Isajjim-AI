"""
SAM3D Worker Pool Manager

Persistent Worker 프로세스 풀을 관리하여 모델 로딩 오버헤드를 제거합니다.

Features:
- GPU당 하나의 워커 프로세스 관리
- 모델을 한 번만 로드하고 재사용
- 라운드로빈 작업 분배
- 자동 워커 재시작
"""

import asyncio
import subprocess
import sys
import os
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

# Import protocol
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.subprocess.worker_protocol import (
    MessageType, TaskMessage, ResultMessage, InitMessage,
    HeartbeatMessage, ShutdownMessage, parse_message
)


@dataclass
class WorkerInfo:
    """워커 프로세스 정보"""
    worker_id: int
    gpu_id: int
    process: Optional[subprocess.Popen] = None
    is_ready: bool = False
    is_busy: bool = False
    current_task_id: Optional[str] = None
    error_count: int = 0
    last_activity: float = 0.0


class SAM3DWorkerPool:
    """
    SAM-3D Worker Pool Manager

    GPU당 하나의 persistent 워커 프로세스를 관리합니다.
    워커는 모델을 미리 로드하고, 작업 요청이 오면 즉시 처리합니다.

    Usage:
        pool = SAM3DWorkerPool(gpu_ids=[0, 1, 2, 3])
        await pool.start_workers()

        # 작업 제출
        result = await pool.submit_task(task_id, image_b64, mask_b64)

        # 종료
        await pool.shutdown()
    """

    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        worker_script: Optional[str] = None,
        python_executable: Optional[str] = None,
        init_timeout: float = 120.0,
        task_timeout: float = 300.0
    ):
        """
        Args:
            gpu_ids: 사용할 GPU ID 목록 (None이면 자동 감지)
            worker_script: 워커 스크립트 경로
            python_executable: Python 실행 경로
            init_timeout: 워커 초기화 타임아웃 (초)
            task_timeout: 작업 타임아웃 (초)
        """
        # GPU 자동 감지
        if gpu_ids is None:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_ids = list(range(torch.cuda.device_count()))
                else:
                    gpu_ids = [0]
            except ImportError:
                gpu_ids = [0]

        self.gpu_ids = gpu_ids
        self.init_timeout = init_timeout
        self.task_timeout = task_timeout

        # 워커 스크립트 경로
        if worker_script is None:
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            worker_script = os.path.join(script_dir, "subprocess", "persistent_3d_worker.py")
        self.worker_script = worker_script

        # Python 실행 경로
        if python_executable is None:
            sam3d_python = os.path.expanduser("~/miniconda3/envs/sam3d-objects/bin/python")
            python_executable = sam3d_python if os.path.exists(sam3d_python) else sys.executable
        self.python_executable = python_executable

        # 워커 레지스트리
        self._workers: Dict[int, WorkerInfo] = {}
        for i, gpu_id in enumerate(gpu_ids):
            self._workers[gpu_id] = WorkerInfo(worker_id=i, gpu_id=gpu_id)

        # 라운드로빈 카운터
        self._next_worker_index = 0

        # 작업 결과 대기용 Future 맵
        self._result_futures: Dict[str, asyncio.Future] = {}

        # 동기화 락
        self._allocation_lock = asyncio.Lock()
        self._started = False

        print(f"[SAM3DWorkerPool] Initialized with {len(gpu_ids)} workers for GPUs: {gpu_ids}")

    async def start_workers(self):
        """모든 워커 프로세스를 시작하고 모델 로딩을 기다립니다."""
        if self._started:
            print("[SAM3DWorkerPool] Workers already started")
            return

        print(f"[SAM3DWorkerPool] Starting {len(self.gpu_ids)} workers...")
        start_time = time.time()

        # 워커 프로세스 시작
        for gpu_id in self.gpu_ids:
            await self._start_worker(gpu_id)

        # 모든 워커 초기화 대기
        init_tasks = [
            self._wait_for_worker_init(gpu_id)
            for gpu_id in self.gpu_ids
        ]

        results = await asyncio.gather(*init_tasks, return_exceptions=True)

        # 결과 확인
        ready_count = 0
        for gpu_id, result in zip(self.gpu_ids, results):
            if isinstance(result, Exception):
                print(f"[SAM3DWorkerPool] Worker {gpu_id} init failed: {result}")
            elif result:
                ready_count += 1

        elapsed = time.time() - start_time
        print(f"[SAM3DWorkerPool] {ready_count}/{len(self.gpu_ids)} workers ready in {elapsed:.2f}s")

        self._started = True

        if ready_count == 0:
            raise RuntimeError("No workers were initialized successfully")

    async def _start_worker(self, gpu_id: int):
        """단일 워커 프로세스 시작"""
        worker_info = self._workers[gpu_id]

        # 기존 프로세스 정리
        if worker_info.process is not None:
            try:
                worker_info.process.terminate()
                worker_info.process.wait(timeout=5)
            except Exception:
                pass

        cmd = [
            self.python_executable,
            self.worker_script,
            str(worker_info.worker_id),
            str(gpu_id)
        ]

        print(f"[SAM3DWorkerPool] Starting worker {worker_info.worker_id} on GPU {gpu_id}")

        # 프로세스 시작
        worker_info.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        worker_info.last_activity = time.time()

        # stderr 로그 리더 시작
        asyncio.create_task(self._read_worker_stderr(gpu_id))

    async def _read_worker_stderr(self, gpu_id: int):
        """워커의 stderr 로그를 읽어서 출력"""
        worker_info = self._workers[gpu_id]
        if worker_info.process is None:
            return

        try:
            while worker_info.process.poll() is None:
                line = await asyncio.get_event_loop().run_in_executor(
                    None,
                    worker_info.process.stderr.readline
                )
                if line:
                    print(f"[Worker {gpu_id}] {line.rstrip()}")
        except Exception:
            pass

    async def _wait_for_worker_init(self, gpu_id: int) -> bool:
        """워커 초기화 완료 대기"""
        worker_info = self._workers[gpu_id]
        if worker_info.process is None:
            return False

        start_time = time.time()

        try:
            while time.time() - start_time < self.init_timeout:
                # stdout에서 INIT 메시지 읽기
                line = await asyncio.get_event_loop().run_in_executor(
                    None,
                    worker_info.process.stdout.readline
                )

                if not line:
                    await asyncio.sleep(0.1)
                    continue

                msg = parse_message(line)
                if msg["type"] == MessageType.INIT:
                    init_msg = InitMessage.from_dict(msg["data"])
                    if init_msg.model_loaded:
                        worker_info.is_ready = True
                        print(f"[SAM3DWorkerPool] Worker {gpu_id} initialized successfully")
                        return True
                    else:
                        print(f"[SAM3DWorkerPool] Worker {gpu_id} init error: {init_msg.error}")
                        return False

            print(f"[SAM3DWorkerPool] Worker {gpu_id} init timeout")
            return False

        except Exception as e:
            print(f"[SAM3DWorkerPool] Worker {gpu_id} init exception: {e}")
            return False

    async def submit_task(
        self,
        task_id: str,
        image_b64: str,
        mask_b64: str,
        seed: int = 42,
        skip_gif: bool = True
    ) -> ResultMessage:
        """
        3D 생성 작업을 워커에 제출합니다.

        Args:
            task_id: 작업 식별자
            image_b64: Base64 인코딩된 이미지
            mask_b64: Base64 인코딩된 마스크
            seed: 랜덤 시드
            skip_gif: GIF 렌더링 스킵

        Returns:
            ResultMessage: 처리 결과
        """
        # 사용 가능한 워커 찾기
        worker_info = await self._acquire_worker(task_id)

        if worker_info is None:
            return ResultMessage(
                task_id=task_id,
                success=False,
                error="No available workers"
            )

        try:
            # 작업 메시지 생성
            task_msg = TaskMessage(
                task_id=task_id,
                image_b64=image_b64,
                mask_b64=mask_b64,
                seed=seed,
                skip_gif=skip_gif
            )

            # 워커에 작업 전송
            worker_info.process.stdin.write(task_msg.to_json() + "\n")
            worker_info.process.stdin.flush()

            # 결과 대기
            result = await self._wait_for_result(worker_info.gpu_id, task_id)

            return result

        except Exception as e:
            print(f"[SAM3DWorkerPool] Task {task_id} failed: {e}")
            return ResultMessage(
                task_id=task_id,
                success=False,
                error=str(e)
            )
        finally:
            await self._release_worker(worker_info.gpu_id)

    async def _acquire_worker(self, task_id: str) -> Optional[WorkerInfo]:
        """라운드로빈으로 사용 가능한 워커 할당"""
        start_time = time.time()

        while time.time() - start_time < self.task_timeout:
            async with self._allocation_lock:
                # 모든 워커 순회
                for _ in range(len(self.gpu_ids)):
                    gpu_id = self.gpu_ids[self._next_worker_index]
                    self._next_worker_index = (self._next_worker_index + 1) % len(self.gpu_ids)

                    worker_info = self._workers[gpu_id]

                    if worker_info.is_ready and not worker_info.is_busy:
                        worker_info.is_busy = True
                        worker_info.current_task_id = task_id
                        worker_info.last_activity = time.time()
                        print(f"[SAM3DWorkerPool] Acquired worker {gpu_id} for task {task_id}")
                        return worker_info

            # 잠시 대기 후 재시도
            await asyncio.sleep(0.5)

        return None

    async def _release_worker(self, gpu_id: int):
        """워커 반환"""
        if gpu_id in self._workers:
            task_id = self._workers[gpu_id].current_task_id
            self._workers[gpu_id].is_busy = False
            self._workers[gpu_id].current_task_id = None
            print(f"[SAM3DWorkerPool] Released worker {gpu_id} (was task {task_id})")

    async def _wait_for_result(self, gpu_id: int, task_id: str) -> ResultMessage:
        """워커로부터 결과 대기"""
        worker_info = self._workers[gpu_id]
        if worker_info.process is None:
            return ResultMessage(task_id=task_id, success=False, error="Worker not running")

        start_time = time.time()

        try:
            while time.time() - start_time < self.task_timeout:
                # stdout에서 결과 읽기
                line = await asyncio.get_event_loop().run_in_executor(
                    None,
                    worker_info.process.stdout.readline
                )

                if not line:
                    # 프로세스가 종료되었는지 확인
                    if worker_info.process.poll() is not None:
                        return ResultMessage(
                            task_id=task_id,
                            success=False,
                            error="Worker process terminated"
                        )
                    await asyncio.sleep(0.1)
                    continue

                msg = parse_message(line)
                if msg["type"] == MessageType.RESULT:
                    result = ResultMessage.from_dict(msg["data"])
                    if result.task_id == task_id:
                        return result

            return ResultMessage(
                task_id=task_id,
                success=False,
                error=f"Task timeout ({self.task_timeout}s)"
            )

        except Exception as e:
            return ResultMessage(
                task_id=task_id,
                success=False,
                error=str(e)
            )

    async def submit_tasks_parallel(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[ResultMessage]:
        """
        여러 작업을 병렬로 제출합니다.

        Args:
            tasks: [{"task_id": str, "image_b64": str, "mask_b64": str, ...}, ...]

        Returns:
            List[ResultMessage]: 각 작업의 결과
        """
        async def submit_one(task: Dict) -> ResultMessage:
            return await self.submit_task(
                task_id=task["task_id"],
                image_b64=task["image_b64"],
                mask_b64=task["mask_b64"],
                seed=task.get("seed", 42),
                skip_gif=task.get("skip_gif", True)
            )

        results = await asyncio.gather(
            *[submit_one(t) for t in tasks],
            return_exceptions=True
        )

        # 예외를 ResultMessage로 변환
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ResultMessage(
                    task_id=tasks[i]["task_id"],
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def shutdown(self):
        """모든 워커 프로세스 종료"""
        print(f"[SAM3DWorkerPool] Shutting down {len(self.gpu_ids)} workers...")

        for gpu_id, worker_info in self._workers.items():
            if worker_info.process is not None:
                try:
                    # SHUTDOWN 메시지 전송
                    shutdown_msg = ShutdownMessage(reason="pool_shutdown")
                    worker_info.process.stdin.write(shutdown_msg.to_json() + "\n")
                    worker_info.process.stdin.flush()

                    # 정상 종료 대기
                    worker_info.process.wait(timeout=5)
                except Exception:
                    # 강제 종료
                    try:
                        worker_info.process.terminate()
                        worker_info.process.wait(timeout=2)
                    except Exception:
                        worker_info.process.kill()

                worker_info.process = None
                worker_info.is_ready = False

        self._started = False
        print("[SAM3DWorkerPool] All workers stopped")

    def get_status(self) -> Dict:
        """워커 풀 상태 반환"""
        return {
            "total_workers": len(self.gpu_ids),
            "ready_workers": sum(1 for w in self._workers.values() if w.is_ready),
            "busy_workers": sum(1 for w in self._workers.values() if w.is_busy),
            "workers": {
                gpu_id: {
                    "worker_id": info.worker_id,
                    "is_ready": info.is_ready,
                    "is_busy": info.is_busy,
                    "current_task_id": info.current_task_id,
                    "error_count": info.error_count,
                    "process_alive": info.process is not None and info.process.poll() is None
                }
                for gpu_id, info in self._workers.items()
            }
        }

    def is_ready(self) -> bool:
        """워커 풀이 준비되었는지 확인"""
        return self._started and any(w.is_ready for w in self._workers.values())


# 글로벌 인스턴스
_global_sam3d_pool: Optional[SAM3DWorkerPool] = None
_init_lock: Optional[asyncio.Lock] = None
_initializing: bool = False


def get_sam3d_worker_pool() -> Optional[SAM3DWorkerPool]:
    """글로벌 SAM3D Worker Pool을 가져옵니다."""
    return _global_sam3d_pool


async def get_or_create_sam3d_worker_pool(
    gpu_ids: Optional[List[int]] = None
) -> Optional[SAM3DWorkerPool]:
    """
    글로벌 SAM3D Worker Pool을 가져오거나 없으면 생성합니다 (Lazy Initialization).

    이 함수는 첫 3D 생성 요청 시 호출되어 워커 풀을 초기화합니다.
    서버 시작 시점이 아닌 실제 사용 시점에 초기화하여 GPU 충돌을 방지합니다.

    Args:
        gpu_ids: 사용할 GPU ID 목록 (None이면 자동 감지)

    Returns:
        SAM3DWorkerPool 인스턴스 또는 초기화 실패 시 None
    """
    global _global_sam3d_pool, _init_lock, _initializing

    # 이미 초기화된 경우 바로 반환
    if _global_sam3d_pool is not None and _global_sam3d_pool._started:
        return _global_sam3d_pool

    # Lock 초기화 (한 번만)
    if _init_lock is None:
        _init_lock = asyncio.Lock()

    async with _init_lock:
        # Double-check after acquiring lock
        if _global_sam3d_pool is not None and _global_sam3d_pool._started:
            return _global_sam3d_pool

        # 이미 초기화 중인 경우 대기
        if _initializing:
            print("[SAM3DWorkerPool] Initialization already in progress, waiting...")
            # 초기화 완료 대기 (최대 180초)
            for _ in range(180):
                await asyncio.sleep(1)
                if _global_sam3d_pool is not None and _global_sam3d_pool._started:
                    return _global_sam3d_pool
            return None

        _initializing = True
        print("[SAM3DWorkerPool] Lazy initialization started...")

        try:
            # GPU ID 자동 감지
            if gpu_ids is None:
                try:
                    from ai.config import Config
                    gpu_ids = Config.get_available_gpus()
                except Exception:
                    import torch
                    if torch.cuda.is_available():
                        gpu_ids = list(range(torch.cuda.device_count()))
                    else:
                        gpu_ids = [0]

            _global_sam3d_pool = SAM3DWorkerPool(gpu_ids=gpu_ids)
            await _global_sam3d_pool.start_workers()

            print(f"[SAM3DWorkerPool] Lazy initialization complete: {_global_sam3d_pool.get_status()}")
            return _global_sam3d_pool

        except Exception as e:
            print(f"[SAM3DWorkerPool] Lazy initialization failed: {e}")
            import traceback
            traceback.print_exc()
            _global_sam3d_pool = None
            return None

        finally:
            _initializing = False


async def initialize_sam3d_worker_pool(
    gpu_ids: Optional[List[int]] = None
) -> SAM3DWorkerPool:
    """
    글로벌 SAM3D Worker Pool을 초기화합니다.

    Args:
        gpu_ids: 사용할 GPU ID 목록

    Returns:
        초기화된 SAM3DWorkerPool 인스턴스
    """
    global _global_sam3d_pool

    if _global_sam3d_pool is not None:
        await _global_sam3d_pool.shutdown()

    _global_sam3d_pool = SAM3DWorkerPool(gpu_ids=gpu_ids)
    await _global_sam3d_pool.start_workers()

    return _global_sam3d_pool


async def shutdown_sam3d_worker_pool():
    """글로벌 SAM3D Worker Pool을 종료합니다."""
    global _global_sam3d_pool

    if _global_sam3d_pool is not None:
        await _global_sam3d_pool.shutdown()
        _global_sam3d_pool = None
