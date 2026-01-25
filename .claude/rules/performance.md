# Performance Optimization

## GPU 메모리 관리

```python
import torch
import gc

def clear_gpu_memory():
    """GPU 메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# 큰 작업 후 항상 호출
result = process_large_image(image)
clear_gpu_memory()
```

## 비동기 처리 최적화

```python
import asyncio

# GOOD: 병렬 실행
async def process_multiple_images(urls: list[str]):
    tasks = [process_single_image(url) for url in urls]
    return await asyncio.gather(*tasks, return_exceptions=True)

# BAD: 순차 실행
async def process_multiple_images_slow(urls: list[str]):
    results = []
    for url in urls:
        result = await process_single_image(url)  # 하나씩 기다림
        results.append(result)
    return results
```

## 이미지 처리 최적화

```python
import cv2
import numpy as np

# 메모리 효율적인 이미지 로딩
def load_image_efficient(path: str, max_size: int = 1024):
    """큰 이미지를 효율적으로 로드"""
    img = cv2.imread(path)
    h, w = img.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    return img
```

## 모델 로딩 최적화

```python
# 서버 시작 시 한 번만 로드
class ModelManager:
    _models: dict = {}

    @classmethod
    def get_model(cls, name: str):
        if name not in cls._models:
            cls._models[name] = load_model(name)
        return cls._models[name]

# FastAPI lifespan으로 사전 로드
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    ModelManager.get_model("yoloe")
    yield
    # Shutdown
    clear_gpu_memory()
```

## 배치 처리

```python
def process_batch(items: list, batch_size: int = 32):
    """배치 단위로 처리하여 메모리 효율 향상"""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = model.predict(batch)
        results.extend(batch_results)
        clear_gpu_memory()  # 배치 간 메모리 정리
    return results
```

## 캐싱 전략

```python
from functools import lru_cache
from cachetools import TTLCache

# 함수 결과 캐싱
@lru_cache(maxsize=100)
def get_furniture_info(class_name: str) -> dict:
    return knowledge_base.lookup(class_name)

# TTL 캐시 (시간 기반 만료)
image_cache = TTLCache(maxsize=50, ttl=300)  # 5분

def get_cached_image(url: str):
    if url not in image_cache:
        image_cache[url] = fetch_image(url)
    return image_cache[url]
```

## 프로파일링

```bash
# cProfile로 병목 찾기
python -m cProfile -s cumtime api.py

# line_profiler로 라인별 분석
kernprof -l -v script.py

# memory_profiler로 메모리 사용량
python -m memory_profiler script.py

# py-spy로 실시간 프로파일링
py-spy top --pid <PID>
```

## 빌드 트러블슈팅

빌드/린트 실패 시:
1. `ruff check .`로 린트 에러 확인
2. `mypy ai/`로 타입 에러 확인
3. 에러 메시지 분석
4. 점진적으로 수정
5. 각 수정 후 검증
