# Common Patterns

## API 응답 형식

```python
from pydantic import BaseModel
from typing import Optional, Generic, TypeVar

T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None

class PaginatedResponse(ApiResponse[T]):
    total: int
    page: int
    limit: int
```

## 의존성 주입 패턴 (FastAPI)

```python
from fastapi import Depends
from functools import lru_cache

class Settings:
    gpu_ids: list[int] = [0, 1, 2, 3]
    confidence_threshold: float = 0.5

@lru_cache
def get_settings() -> Settings:
    return Settings()

@app.post("/analyze")
async def analyze(settings: Settings = Depends(get_settings)):
    ...
```

## 비동기 컨텍스트 매니저 패턴

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def gpu_context(pool: GPUPoolManager, task_id: str):
    """GPU 리소스 자동 해제를 보장하는 컨텍스트 매니저"""
    gpu_id = await pool.acquire(task_id)
    try:
        yield gpu_id
    finally:
        await pool.release(gpu_id)

# 사용
async with gpu_context(pool, "task_1") as gpu_id:
    result = await process_on_gpu(gpu_id)
```

## Repository 패턴

```python
from abc import ABC, abstractmethod
from typing import Optional

class Repository(ABC):
    @abstractmethod
    async def find_all(self, filters: dict = None) -> list:
        pass

    @abstractmethod
    async def find_by_id(self, id: str) -> Optional[dict]:
        pass

    @abstractmethod
    async def create(self, data: dict) -> dict:
        pass

    @abstractmethod
    async def update(self, id: str, data: dict) -> dict:
        pass

    @abstractmethod
    async def delete(self, id: str) -> None:
        pass
```

## 파이프라인 패턴 (이 프로젝트)

```python
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class PipelineStep:
    name: str
    processor: Callable
    required: bool = True

class Pipeline:
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    async def execute(self, input_data: Any) -> Any:
        result = input_data
        for step in self.steps:
            try:
                result = await step.processor(result)
            except Exception as e:
                if step.required:
                    raise
                logger.warning(f"Optional step {step.name} failed: {e}")
        return result
```

## 싱글톤 패턴 (GPU Pool)

```python
from typing import Optional

class GPUPoolManager:
    _instance: Optional["GPUPoolManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# 또는 모듈 레벨 인스턴스
_gpu_pool: Optional[GPUPoolManager] = None

def get_gpu_pool() -> GPUPoolManager:
    global _gpu_pool
    if _gpu_pool is None:
        raise RuntimeError("GPU pool not initialized")
    return _gpu_pool
```

## 데코레이터 패턴 (로깅/타이밍)

```python
import functools
import time
import logging

logger = logging.getLogger(__name__)

def log_execution_time(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper

@log_execution_time
async def process_image(image_url: str):
    ...
```
