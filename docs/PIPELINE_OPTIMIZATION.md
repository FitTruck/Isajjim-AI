# Pipeline Optimization Guide

이 문서는 SAM3D-API 파이프라인에 적용된 최적화 기법들을 분석하고 정리합니다.

## 목차

1. [V2 파이프라인 아키텍처 최적화](#1-v2-파이프라인-아키텍처-최적화)
2. [2단계 병렬 처리 아키텍처](#2-2단계-병렬-처리-아키텍처)
3. [Multi-GPU 병렬 처리 (1단계)](#3-multi-gpu-병렬-처리-1단계)
4. [SAM-3D Worker Pool (2단계)](#4-sam-3d-worker-pool-2단계)
5. [SAM-3D 추론 최적화](#5-sam-3d-추론-최적화)
6. [환경 변수 최적화](#6-환경-변수-최적화)
7. [프로세스 격리](#7-프로세스-격리)
8. [Synthetic Pinhole Pointmap](#8-synthetic-pinhole-pointmap)
9. [초기 대비 최적화 효과 분석](#9-초기-대비-최적화-효과-분석)

---

## 1. V2 파이프라인 아키텍처 최적화

### 변경 사항

V1에서 V2로 파이프라인을 단순화하여 불필요한 단계를 제거했습니다.

```
[V1 파이프라인]
YOLO detect → center_point → SAM2 → mask → CLIP 분류 → SAM-3D
(5단계, 3회 API 호출)

[V2 파이프라인]
YOLOE-seg detect → mask (직접) → SAM-3D
(2단계, 2회 API 호출)
```

### 제거된 컴포넌트

| 컴포넌트 | V1 역할 | 제거 이유 |
|---------|--------|----------|
| **SAM2** | center point에서 마스크 생성 | YOLOE-seg가 더 정확한 마스크 제공 |
| **CLIP** | 세부 유형 분류 | YOLO 클래스로 직접 DB 매칭 가능 |
| **SAHI** | 작은 객체 탐지 | YOLOE-seg로 충분한 탐지율 |

### 코드 위치

- `ai/pipeline/furniture_pipeline.py:1-19` - V2 파이프라인 설명

### 효과

- **Latency 감소**: SAM2 API 호출 제거 (~2-5초)
- **코드 단순화**: 의존성 및 유지보수 복잡도 감소
- **마스크 품질 향상**: YOLOE-seg가 객체 전체를 더 정확하게 커버

---

## 2. 2단계 병렬 처리 아키텍처

현재 파이프라인은 **2단계 병렬 처리**를 적용하여 이미지와 객체를 동시에 처리합니다.

### 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     1단계: 이미지 병렬 처리                               │
│                     (GPUPoolManager - YOLOE)                            │
│                                                                         │
│   img1 → GPU0 (YOLOE) ─┐                                                │
│   img2 → GPU1 (YOLOE) ─┼─► 각 이미지별 객체 탐지 + 마스크 생성             │
│   img3 → GPU2 (YOLOE) ─┤                                                │
│   img4 → GPU3 (YOLOE) ─┘                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     2단계: 객체 병렬 처리                                 │
│                     (SAM3DWorkerPool)                                   │
│                                                                         │
│   img1: [obj1, obj2, obj3] ──┬──► Worker0 (GPU0) → obj1 3D 생성         │
│                              ├──► Worker1 (GPU1) → obj2 3D 생성         │
│                              └──► Worker2 (GPU2) → obj3 3D 생성         │
│                                                                         │
│   img2: [obj1, obj2] ────────┬──► Worker3 (GPU3) → obj1 3D 생성         │
│                              └──► Worker0 (GPU0) → obj2 3D 생성         │
│                                        ...                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     3단계: 결과 취합                                     │
│                                                                         │
│   image_id로 그룹핑 → 각 이미지별 objects 리스트 → JSON 응답 반환          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 처리 흐름

```
요청: 4개 이미지 (각 3개 객체 = 총 12개 객체)

1단계 (YOLOE - 이미지 병렬):
┌────────────────────────────────────────────────────┐
│  GPU0: img1 → [obj1, obj2, obj3]  ─┐              │
│  GPU1: img2 → [obj1, obj2, obj3]  ─┼── 동시 처리   │
│  GPU2: img3 → [obj1, obj2, obj3]  ─┤   (~1초)     │
│  GPU3: img4 → [obj1, obj2, obj3]  ─┘              │
└────────────────────────────────────────────────────┘
                      │
                      ▼
2단계 (SAM-3D - 객체 병렬):
┌────────────────────────────────────────────────────┐
│  12개 객체를 4개 Worker에 분배                      │
│                                                    │
│  라운드 1: obj1,2,3,4 → Worker0,1,2,3 (동시)       │
│  라운드 2: obj5,6,7,8 → Worker0,1,2,3 (동시)       │
│  라운드 3: obj9,10,11,12 → Worker0,1,2,3 (동시)    │
│                                                    │
│  총 3 라운드 × 26초 = ~78초                        │
└────────────────────────────────────────────────────┘
                      │
                      ▼
3단계 (취합):
┌────────────────────────────────────────────────────┐
│  image_id별로 결과 그룹핑                           │
│  → img1: [obj1, obj2, obj3]                        │
│  → img2: [obj1, obj2, obj3]                        │
│  → img3: [obj1, obj2, obj3]                        │
│  → img4: [obj1, obj2, obj3]                        │
└────────────────────────────────────────────────────┘
```

### 코드 구현

#### 1단계: 이미지 병렬 처리

```python
# ai/pipeline/furniture_pipeline.py:453-526
async def process_multiple_images(self, image_urls, ...):
    pool = self.gpu_pool or get_gpu_pool()

    async def process_with_gpu(url):
        async with pool.pipeline_context(task_id=url) as (gpu_id, pipeline):
            return await pipeline.process_single_image(url)

    # 모든 이미지 동시 처리
    results = await asyncio.gather(*[process_with_gpu(url) for url in image_urls])
```

#### 2단계: 객체 병렬 처리

```python
# ai/pipeline/furniture_pipeline.py:388-450
async def _parallel_3d_generation(self, image, objects_with_masks):
    sam3d_pool = get_sam3d_worker_pool()

    # 작업 목록 생성 (객체별)
    tasks = []
    for obj_id, obj in objects_with_masks:
        tasks.append({
            "task_id": f"obj_{obj_id}",
            "image_b64": image_b64,
            "mask_b64": obj.mask_base64,
        })

    # 모든 객체 동시 제출
    worker_results = await sam3d_pool.submit_tasks_parallel(tasks)
```

### 효과

| 처리 방식 | 4 이미지 × 3 객체 | 효율성 |
|----------|------------------|--------|
| 완전 순차 | 12 × 150초 = 1800초 | 기준 |
| 이미지만 병렬 | 3 × 150초 = 450초 | 4배 |
| **2단계 병렬** | 1초 + 78초 = **79초** | **23배** |

---

## 3. Multi-GPU 병렬 처리 (1단계)

### 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Server                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    GPUPoolManager                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │   │
│  │  │  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │     │   │
│  │  │ YOLOE   │  │ YOLOE   │  │ YOLOE   │  │ YOLOE   │     │   │
│  │  │(사전로드)│  │(사전로드)│  │(사전로드)│  │(사전로드)│     │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                    Round-robin GPU allocation                    │
│                              ▼                                   │
│         img1→GPU0, img2→GPU1, img3→GPU2, img4→GPU3              │
└─────────────────────────────────────────────────────────────────┘
```

### 핵심 기능

#### 3.1 라운드로빈 GPU 할당

```python
# ai/gpu/gpu_pool_manager.py:101-143
async def acquire(self, task_id: Optional[str] = None) -> int:
    async with self._allocation_lock:
        for _ in range(len(self.gpu_ids)):
            gpu_id = self.gpu_ids[self._next_gpu_index]
            self._next_gpu_index = (self._next_gpu_index + 1) % len(self.gpu_ids)

            if gpu_info.is_available:
                gpu_info.is_available = False
                return gpu_id
```

#### 3.2 파이프라인 사전 초기화

서버 시작 시 각 GPU에 YOLOE 모델을 미리 로드합니다.

```python
# ai/gpu/gpu_pool_manager.py:289-320
async def initialize_pipelines(self, pipeline_factory, skip_on_error=True):
    for gpu_id in self.gpu_ids:
        pipeline = pipeline_factory(gpu_id)  # YOLOE 로드
        self.register_pipeline(gpu_id, pipeline)
```

#### 3.3 컨텍스트 매니저

```python
# 자동 GPU 획득/반환
async with pool.pipeline_context(task_id="image_1") as (gpu_id, pipeline):
    result = await pipeline.process_single_image(url)
# GPU 자동 반환
```

### 코드 위치

- `ai/gpu/gpu_pool_manager.py` - GPU Pool Manager 구현
- `ai/config.py:9-20` - Multi-GPU 설정

### 효과

- **처리량 증가**: N개 GPU로 N배 병렬 처리
- **모델 로드 시간 제거**: 요청당 3-5초 절약
- **GPU 활용률 최적화**: 유휴 GPU 최소화

---

## 4. SAM-3D Worker Pool (2단계)

### 문제점

SAM-3D는 spconv 라이브러리의 GPU 상태 충돌 문제로 메인 프로세스에서 직접 로드할 수 없습니다. 기존 방식(매 요청마다 subprocess)은 모델 로딩 오버헤드(3-5초)가 발생했습니다.

### 해결책: Persistent Worker Pool

```
서버 시작 시:
  Worker 0 (GPU 0) 시작 → SAM-3D 모델 로드 → 대기
  Worker 1 (GPU 1) 시작 → SAM-3D 모델 로드 → 대기
  Worker 2 (GPU 2) 시작 → SAM-3D 모델 로드 → 대기
  Worker 3 (GPU 3) 시작 → SAM-3D 모델 로드 → 대기

요청 처리:
  API Server ──JSON──► Worker (이미 모델 로드됨) ──JSON──► 결과
              (stdin)                               (stdout)
```

### 구현 방식

```python
# ai/gpu/sam3d_worker_pool.py:44-60
class SAM3DWorkerPool:
    """
    GPU당 하나의 persistent 워커 프로세스를 관리합니다.
    워커는 모델을 미리 로드하고, 작업 요청이 오면 즉시 처리합니다.
    """

    async def submit_tasks_parallel(self, tasks):
        """여러 작업을 병렬로 제출"""
        results = await asyncio.gather(
            *[self.submit_task(**t) for t in tasks],
            return_exceptions=True
        )
        return results
```

### 통신 프로토콜

```python
# ai/subprocess/worker_protocol.py
MessageType:
  - INIT: 워커 초기화 완료 알림
  - TASK: 3D 생성 작업 요청
  - RESULT: 작업 결과 반환
  - HEARTBEAT: 워커 상태 확인
  - SHUTDOWN: 워커 종료 요청
```

### 코드 위치

- `ai/gpu/sam3d_worker_pool.py` - Worker Pool Manager
- `ai/subprocess/persistent_3d_worker.py` - Persistent Worker 구현
- `ai/subprocess/worker_protocol.py` - JSON 통신 프로토콜

### 효과

- **모델 로드 시간 제거**: 요청당 3-5초 절약
- **객체 병렬 처리**: 여러 객체 동시 3D 생성
- **리소스 효율**: GPU 메모리 재사용

---

## 5. SAM-3D 추론 최적화

### 5.1 불필요한 후처리 비활성화

```python
# ai/subprocess/persistent_3d_worker.py:631-644
output = pipe.run(
    image=image,
    mask=mask,
    seed=seed,
    pointmap=pointmap,
    decode_formats=["gaussian", "glb", "mesh"],
    with_mesh_postprocess=False,     # 비활성화: 20-40초 절약
    with_texture_baking=False,       # 비활성화: 30-60초 절약
    with_layout_postprocess=False,   # 비활성화: 2-5초 절약
    use_vertex_color=True,
)
```

| 옵션 | 기본값 | 변경값 | 절약 시간 |
|------|--------|--------|----------|
| `with_texture_baking` | True | **False** | 30-60초 |
| `with_mesh_postprocess` | True | **False** | 20-40초 |
| `with_layout_postprocess` | True | **False** | 2-5초 |

### 5.2 GIF 렌더링 스킵

Gaussian-only 모드에서는 GIF 렌더링이 자동으로 스킵됩니다.

```python
# ai/subprocess/persistent_3d_worker.py
GAUSSIAN_ONLY_MODE = True  # GIF/GLB/Mesh 모두 스킵

# 효과: 15-30초 절약
```

### 5.3 Inference Steps 감소

```python
# ai/subprocess/persistent_3d_worker.py:58-62
# Stage1 (Sparse Structure): 12~16 사이 최적값
STAGE1_INFERENCE_STEPS = 14  # 기본값 25 → 14 (속도/정확도 균형)

# Stage2 (SLAT): 4 steps로 충분
STAGE2_INFERENCE_STEPS = 4   # 기본값 12 → 4 (치수 오차 0.5% 이내, 30% 속도 향상)
```

#### Stage1 Steps 테스트 결과

| Steps | 부피 오차 | 속도 향상 | 권장 |
|-------|----------|----------|------|
| 25 | baseline | 1.00x | - |
| 20 | +5.47% | 1.23x | ⚠️ 주의 |
| 16 | ~+1% | ~1.4x | ✅ 권장 범위 |
| **14** | **~+1.5%** | **~1.5x** | ✅ **권장 (현재 설정)** |
| 12 | +11.04% | 1.65x | ⚠️ 주의 |
| 10 | +15.09% | 1.84x | ❌ 비권장 |

**결론**: `stage1_steps=14`가 속도/정확도 균형점 (12~16 사이 최적값)

### 5.4 Binary PLY 포맷

```python
# ai/subprocess/persistent_3d_worker.py:66
USE_BINARY_PLY = True

# 효과: 파일 크기 70% 감소, 쓰기 속도 50% 향상
```

### 5.5 이미지 다운샘플링 (비활성화)

```python
# ai/subprocess/persistent_3d_worker.py:57
MAX_IMAGE_SIZE = None  # 비활성화

# 이유: 다운샘플링이 부피 정확도에 91.7% 영향
# 특히 작은 객체에서 최대 576% 부피 차이 발생
```

### 5.6 Gaussian-only 모드

```python
# ai/subprocess/persistent_3d_worker.py:68-71
GAUSSIAN_ONLY_MODE = True  # GLB/Mesh 생성 스킵, decode_formats=["gaussian"]

# 효과: 37.4% 속도 향상, 부피 오차 0.005% (무시 가능)
# 부피 계산만 필요한 경우 권장 (현재 활성화됨)
```

### 5.7 in_place=True 최적화

```python
# ai/subprocess/persistent_3d_worker.py:455-458
# deepcopy 제거로 메모리/속도 최적화
scene_gs = self.make_scene(output, in_place=True)
scene_gs = self.ready_gaussian_for_video_rendering(
    scene_gs, in_place=True, fix_alignment=False
)

# 효과: 메모리 복사 오버헤드 제거, ~5-10% 속도 향상
```

### 5.8 torch.compile 활성화

```python
# ai/subprocess/persistent_3d_worker.py:317-321
ENABLE_COMPILE = True  # True = 추론 10-20% 빠름, False = 빠른 시작 (테스트용)

# 워커 초기화 시 CUDA 커널 컴파일
self.sam3d_inference = Inference(config_path, compile=ENABLE_COMPILE, device="cuda")

# 효과:
# - 초기화 시: warmup으로 추가 시간 소요
# - 추론 시: CUDA 커널 재사용으로 ~10-20% 속도 향상
# - Persistent Worker이므로 초기화 비용은 서버 시작 시 1회만 발생
```

### 효과 요약

| Phase | 최적화 | 효과 | 부피 영향 |
|-------|--------|------|----------|
| 1 | 이미지 다운샘플링 비활성화 | 부피 정확도 유지 | 91.7% 영향 방지 |
| 2 | Stage1 Steps (25→14) | **~50% 속도 향상** | **~1.5%** |
| 2 | Stage2 Steps (12→4) | ~30% 속도 향상 | ~0.5% |
| 3 | Binary PLY | 쓰기 50% 빠름 | 없음 |
| 5 | Gaussian-only 모드 | **37.4% 속도 향상** | 0.005% |
| - | in_place=True | 5-10% 속도 향상 | 없음 |
| - | torch.compile | 10-20% 속도 향상 | 없음 |
| - | 후처리 비활성화 | 52-105초 절약 | 없음 |
| - | GIF 스킵 | 15-30초 절약 | 없음 |

**총 예상 성능 향상**: 단일 객체 기준 ~2-3배 빠름 (부피 정확도 유지)

---

## 6. 환경 변수 최적화

### 6.1 스레드 폭발 방지

```python
# ai/subprocess/persistent_3d_worker.py:32-37
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# PyTorch 스레드 제한
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
```

**문제**: 기본 설정에서 각 라이브러리가 CPU 코어 수만큼 스레드 생성 → 스레드 폭발

**해결**: 스레드 수를 4개로 제한하여 컨텍스트 스위칭 오버헤드 감소

### 6.2 spconv 튜닝 시간 제한

```python
# ai/subprocess/persistent_3d_worker.py:29
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "100"  # 100ms 제한

# 문제: spconv가 최적 알고리즘을 찾기 위해 무한 튜닝
# 해결: 튜닝 시간을 100ms로 제한
```

### 6.3 CUDA 디바이스 격리

```python
# Multi-GPU 환경에서 특정 GPU만 보이게 설정
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# SPCONV_TUNE_DEVICE는 항상 0 (remap 되므로)
os.environ["SPCONV_TUNE_DEVICE"] = "0"
```

---

## 7. 프로세스 격리

### 문제점

spconv 라이브러리는 GPU 상태를 유지하며, 같은 프로세스에서 여러 번 로드하면 충돌이 발생합니다.

### 해결책: Persistent Worker Pool + Subprocess 격리

```python
# ai/gpu/sam3d_worker_pool.py - 워커 프로세스 시작
process = subprocess.Popen(
    [sys.executable, worker_script, "--gpu-id", str(gpu_id)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env  # CUDA_VISIBLE_DEVICES로 GPU 격리
)
```

### 장점

1. **GPU 메모리 자동 해제**: subprocess 종료 시 메모리 완전 해제
2. **상태 격리**: 한 요청의 실패가 다른 요청에 영향 없음
3. **spconv 충돌 방지**: 매번 새로운 프로세스에서 로드

---

## 8. Synthetic Pinhole Pointmap

### 문제점

SAM-3D의 MoGe 모듈이 카메라 intrinsics를 추정할 때 실패하거나 NaN/Inf 값을 생성하는 경우가 있었습니다.

### 해결책

```python
# ai/subprocess/persistent_3d_worker.py:213-251
def make_synthetic_pointmap(image, z=1.0, f=None):
    """
    Create a simple pinhole-camera pointmap:
      X = (u - cx) / f * Z
      Y = (v - cy) / f * Z
      Z = constant depth
    """
    H, W = image.shape[:2]
    if f is None:
        f = 0.9 * max(H, W)  # 이미지 크기 기반 focal length

    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    Z = np.full((H, W), z, dtype=np.float32)
    X = (uu - cx) / f * Z
    Y = (vv - cy) / f * Z

    return torch.from_numpy(np.stack([X, Y, Z], axis=-1))
```

### 효과

- **안정성 향상**: MoGe intrinsics recovery 실패 방지
- **NaN/Inf 제거**: 유효한 좌표값 보장
- **일관된 결과**: 모든 이미지에서 동일한 방식으로 pointmap 생성

---

## 9. 초기 대비 최적화 효과 분석

### 9.1 단일 객체 처리 시간 비교

#### 초기 상태 (V1 파이프라인, 최적화 없음)

| 단계 | 작업 | 시간 |
|------|------|------|
| 1 | YOLO 탐지 (매번 로드) | 4-6초 |
| 2 | SAM2 마스크 생성 | 2-5초 |
| 3 | CLIP 분류 | 1-2초 |
| 4 | SAM-3D 모델 로드 | 3-5초 |
| 5 | SAM-3D 추론 (steps=12) | ~35초 |
| 6 | texture_baking | 30-60초 |
| 7 | mesh_postprocess | 20-40초 |
| 8 | layout_postprocess | 2-5초 |
| 9 | GIF 렌더링 | 15-30초 |
| | **총합** | **112-188초 (~150초)** |

#### 현재 상태 (V2 파이프라인, 최적화 적용)

| 단계 | 작업 | 시간 |
|------|------|------|
| 1 | YOLOE-seg 탐지 (사전 로드) | 0.5-1초 |
| 2 | SAM-3D 모델 로드 (Worker Pool) | 0초 |
| 3 | SAM-3D 추론 (stage1=14, stage2=4, compile=True) | ~6-7초 |
| 4 | Gaussian-only 디코딩 | ~0.5초 |
| 5 | 후처리/GIF | 0초 (비활성화/스킵) |
| | **총합** | **~7-8초** |

**적용된 최적화**:
- Stage1 Steps: 25 → 14 (~50% 빠름)
- Stage2 Steps: 12 → 4 (~30% 빠름)
- torch.compile: 10-20% 빠름
- in_place=True: 5-10% 빠름
- Gaussian-only: 37% 빠름

#### 단일 객체 최적화 효과

```
초기:  ████████████████████████████████████████████████████  ~150초
현재:  ███                                                    ~7-8초

절약:  약 142-143초 (95% 감소)
속도:  약 19-21배 향상
```

---

### 9.2 다중 이미지/객체 처리 시간 비교

#### 시나리오: 4개 이미지, 이미지당 3개 객체 (총 12개 객체), 4 GPU

##### 초기 (완전 순차 처리)

```
img1 → YOLO(5s) → SAM2(3s) → CLIP(1s) → obj1(150s) → obj2(150s) → obj3(150s) = 459초
img2 → YOLO(5s) → SAM2(3s) → CLIP(1s) → obj1(150s) → obj2(150s) → obj3(150s) = 459초
img3 → YOLO(5s) → SAM2(3s) → CLIP(1s) → obj1(150s) → obj2(150s) → obj3(150s) = 459초
img4 → YOLO(5s) → SAM2(3s) → CLIP(1s) → obj1(150s) → obj2(150s) → obj3(150s) = 459초
────────────────────────────────────────────────────────────────────────────────
총: ~1836초 (30.6분)
```

##### 현재 (2단계 병렬 처리 + 추론 최적화)

```
1단계 (YOLOE 병렬):
  GPU0: img1 ─┐
  GPU1: img2 ─┼── ~1초 (동시)
  GPU2: img3 ─┤
  GPU3: img4 ─┘

2단계 (SAM-3D 객체 병렬):
  12개 객체 → 4 Workers (라운드로빈)
  라운드 1: obj1,2,3,4 → ~7초
  라운드 2: obj5,6,7,8 → ~7초
  라운드 3: obj9,10,11,12 → ~7초
  총: ~21초

3단계 (취합): ~0초
────────────────────────────────────────────────────────────────────────────────
총: ~22초 (0.4분)
```

##### 비교

```
초기:  ████████████████████████████████████████████████████████████  1836초 (30.6분)
현재:  █                                                              22초 (0.4분)

절약:  1814초 (98.8% 감소)
속도:  약 83배 향상
```

---

### 9.3 규모별 최적화 효과

| 시나리오 | 초기 | 현재 | 절약 | 배수 |
|----------|------|------|------|------|
| 1 이미지 × 1 객체 | 150초 | 8초 | 94.7% | **~19배** |
| 1 이미지 × 3 객체 | 450초 | 21초 | 95.3% | **~21배** |
| 4 이미지 × 1 객체 | 600초 | 9초 | 98.5% | **~67배** |
| 4 이미지 × 3 객체 | 1836초 | 22초 | 98.8% | **~83배** |
| 10 이미지 × 3 객체 | 4590초 | 54초 | 98.8% | **~85배** |
| 10 이미지 × 5 객체 | 7650초 | 89초 | 98.8% | **~86배** |

> **Note**: 현재 시간은 stage1=14, stage2=4, compile=True, Gaussian-only 모드 적용 기준

---

### 9.4 최적화 기여도 분석

각 최적화 기법이 전체 성능 향상에 기여한 정도를 분석합니다.

#### 단일 객체 기준 (150초 → 8초)

| 최적화 | 절약 시간 | 기여도 |
|--------|----------|--------|
| 후처리 비활성화 (texture_baking 등) | 52-105초 | **37-43%** |
| GIF 스킵 | 15-30초 | **11-13%** |
| V2 파이프라인 (SAM2/CLIP 제거) | 3-7초 | **2-3%** |
| 모델 사전 로드 (YOLOE + SAM-3D) | 7-11초 | **5-6%** |
| Stage1 Steps 감소 (25→14) | ~4초 | **~3%** |
| Stage2 Steps 감소 (12→4) | ~4초 | **~3%** |
| Gaussian-only 모드 | ~9초 | **~6%** |
| torch.compile | ~3-5초 | **~3%** |
| in_place=True | ~1-2초 | **~1%** |
| **합계** | **~142초** | **~95%** |

#### 다중 이미지/객체 기준 (추가 최적화)

| 최적화 | 효과 |
|--------|------|
| 이미지 병렬 처리 (GPUPoolManager) | N개 GPU → N배 속도 |
| 객체 병렬 처리 (SAM3DWorkerPool) | M개 객체 → ceil(M/N)배 속도 |
| **복합 효과** | **~23배 속도 향상** |

---

### 9.5 시각적 요약

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        최적화 전후 비교 (4 이미지 × 3 객체)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  초기 (V1, 순차):                                                           │
│  ████████████████████████████████████████████████████████████  1836초       │
│  |-------- 30.6분 --------|                                                 │
│                                                                             │
│  현재 (V2, 2단계 병렬 + 추론 최적화):                                         │
│  █  22초                                                                    │
│  |22초|                                                                     │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  개선율: 98.8% 감소 (83배 빠름)                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 9.6 결론

| 측면 | 초기 | 현재 | 개선율 |
|------|------|------|--------|
| **단일 객체 처리** | ~150초 | ~8초 | **95% 감소** |
| **다중 이미지/객체 처리** (4×3) | ~1836초 | ~22초 | **98.8% 감소** |
| **처리 속도** | 1배 | 83배 | **83배 향상** |
| **파이프라인 단계** | 5단계 | 2단계 | **60% 감소** |
| **모델 로드 오버헤드** | 7-11초/요청 | 0초 | **100% 제거** |
| **GPU 활용률** | 단일 GPU | N GPU 병렬 | **N배 향상** |

#### 적용된 최적화 설정 (2026-03-18 업데이트)

```python
# ai/subprocess/persistent_3d_worker.py
MAX_IMAGE_SIZE = None           # Phase 1: 다운샘플링 비활성화 (부피 정확도 유지)
STAGE1_INFERENCE_STEPS = 14     # Phase 2: Stage1 (25→14, 속도/정확도 균형)
STAGE2_INFERENCE_STEPS = 4      # Phase 2: Stage2 (12→4, 치수 오차 0.5% 이내, 30% 빠름)
USE_BINARY_PLY = True           # Phase 3: Binary PLY (70% 작음, 50% 빠름)
GAUSSIAN_ONLY_MODE = True       # Phase 5: Gaussian-only (37.4% 빠름, 0.005% 오차)
ENABLE_COMPILE = True           # Phase C: torch.compile reduce-overhead (20-30% 빠름)
ENABLE_SS_STEP_CACHING = True   # Phase A: SS Step Caching (stride=3, 1.5x 빠름)
ENABLE_SLAT_STEP_CACHING = False # Phase B: SLaT Step Caching (비활성화, 품질 리스크)
in_place=True                   # make_scene/ready_gaussian에서 deepcopy 제거 (5-10% 빠름)
```

---

## 10. 이미지 전처리 최적화

### 10.1 CLAHE 객체 캐싱

YOLOE 탐지 시 저조도/저대비 이미지 개선을 위해 CLAHE(Contrast Limited Adaptive Histogram Equalization)를 적용합니다.

#### 문제점

```python
# 기존: 매번 CLAHE 객체 생성 (5-10ms 오버헤드)
def apply_clahe(cv2_image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # 매번 생성
    cl = clahe.apply(l)
```

#### 해결책: 싱글톤 캐싱

```python
# ai/utils/image_ops.py
class ImageUtils:
    _clahe = None  # 클래스 레벨 캐싱

    @staticmethod
    def apply_clahe(cv2_image):
        # CLAHE 객체 캐싱 (최초 1회만 생성)
        if ImageUtils._clahe is None:
            ImageUtils._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = ImageUtils._clahe.apply(l)
```

#### 효과

| 항목 | 기존 | 최적화 후 | 개선 |
|------|------|----------|------|
| CLAHE 생성 시간 | 5-10ms/호출 | 0ms (캐싱) | **100% 제거** |
| 10개 이미지 처리 | 50-100ms | ~0ms | **50-100ms 절약** |

> **Note**: CLAHE는 `detect_smart()` 앙상블 탐지에서 사용됩니다. 이미지당 1회 호출되므로 다중 이미지 처리 시 누적 효과가 있습니다.

---

## 11. VRAM 최적화 (2026-03-18)

### 문제점

SAM-3D 모델이 L4 GPU (22GB) VRAM의 ~21GB를 사용하여:
- YOLOE와 동일 GPU에 동시 탑재 불가
- 2 GPU 환경에서 1 GPU는 YOLOE 전용, 1 GPU는 SAM-3D 전용으로 사용해야 함
- 병렬 SAM-3D 처리 불가

### 해결책: Gaussian-only 모드에서 불필요한 모델 GPU 언로드

```python
# ai/subprocess/persistent_3d_worker.py (initialize 메서드)

# 1. Mesh decoder (~4GB) - Gaussian-only에서 절대 사용 안 함
pipe.models["slat_decoder_mesh"].cpu()

# 2. GS 4-channel decoder (~3GB) - 기본 GS decoder로 충분
pipe.models["slat_decoder_gs_4"].cpu()

# 3. Depth model/MoGe (~3GB) - synthetic pointmap 사용 중
pipe.depth_model.model.cpu()
pipe.depth_model = None
```

### 효과

| 항목 | 변경 전 | 변경 후 | 절약 |
|------|---------|---------|------|
| SAM-3D VRAM | ~21GB | **11.25GB** | **~10GB (48%)** |
| YOLOE VRAM | 0.36GB | 0.36GB | - |
| 합계 (동일 GPU) | **OOM** | **11.61GB / 22GB** | 2GPU 병렬 가능 |

### MoGe (depth_model) 비활성화 근거

MoGe는 2D→3D 깊이 추정 모델. `make_synthetic_pointmap()`이 대신 사용되므로 MoGe는 호출되지 않음:

```python
# SAM3D 내부 (inference_pipeline_pointmap.py:268)
if pointmap is None:
    output = self.depth_model(loaded_image)  # pointmap 없을 때만 호출
```

worker가 항상 synthetic pointmap을 제공하므로 depth_model은 GPU에서 제거해도 안전.

**비교 테스트 결과 (MoGe vs Synthetic):**

| 객체 | 항목 | Synthetic | MoGe | 차이 |
|------|------|-----------|------|------|
| Nightstand | W/D/H | 0.76/0.54/1.00 | 0.77/0.55/1.00 | 1-1.4% |
| Bed | W/D/H | 0.77/0.84/0.40 | 0.75/0.85/0.39 | 1-2.6% |
| Television | W/D/H | 1.04/0.02/0.58 | 1.23/0.02/0.69 | 18% |

TV 차이는 극히 얇은 객체(depth=0.02)의 깊이 추정 민감도 차이이며, Synthetic이 안정적.

### 코드 위치

- `ai/subprocess/persistent_3d_worker.py:347-390` - VRAM cleanup 로직

---

## 12. Fast-SAM3D 기반 추론 가속 (2026-03-18)

Fast-SAM3D (arXiv:2602.05293) 논문의 기법을 적용한 training-free 추론 가속.

### 12.1 Phase C: torch.compile + AUTOTUNE 캐시 영속화

#### 문제점

1. SAM3D 내부 `compile=True` 사용 시 `_warmup()`에서 `run_layout_model` 버그 발생
2. `torch.compile(mode="max-autotune")`의 첫 실행 AUTOTUNE에 10분+ 소요
3. 워커 재시작 시 AUTOTUNE 캐시 손실

#### 해결책

```python
# 1. SAM3D는 compile=False로 로드
self.sam3d_inference = Inference(config_path, compile=False)

# 2. 핵심 모듈만 수동 torch.compile 적용
compile_mode = "reduce-overhead"  # max-autotune보다 빠른 첫 실행

# SS Generator backbone (14 steps × 3 CFG calls = 42회 호출, 가장 빈번)
ss_gen.reverse_fn.inner_forward = torch.compile(
    ss_gen.reverse_fn.inner_forward, mode=compile_mode, fullgraph=True
)

# SS Decoder
ss_dec.forward = torch.compile(ss_dec.forward, mode=compile_mode, fullgraph=True)

# Condition Embedding (fullgraph=False: PointPatchEmbed 호환성)
pipe.embed_condition = torch.compile(
    pipe.embed_condition, mode=compile_mode, fullgraph=False
)

# 3. AUTOTUNE 캐시 영속화
os.environ["TORCHINDUCTOR_CACHE_DIR"] = ".cache/torch_compile"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

# 4. 자체 warmup (SAM3D 내부 warmup 우회)
_ = pipe.run(dummy_image, dummy_mask, seed=42, pointmap=dummy_pointmap, ...)
```

#### 효과

| 항목 | 변경 전 | 변경 후 | 개선 |
|------|---------|---------|------|
| Bed 추론 시간 | 25.6s | 18.7s | **1.37x** |
| Television 추론 시간 | 10.0s | 7.6s | **1.31x** |
| 첫 실행 warmup | N/A | ~280s (캐시 있을 때) | 1회 비용 |
| AUTOTUNE 캐시 | 재시작 시 손실 | **영속** (2,259 파일) | 재컴파일 방지 |

### 12.2 Phase A: SS Generator Step Caching

#### 배경

SS Generator (ShortCut 모델)은 PointmapCFG를 통해 매 step마다 3회 backbone 호출:
```
v_t = PointmapCFG(x_t, t)
    = y_cond + strength_pm * (y_cond - y_no_pm) + strength * (y_no_pm - y_uncond)
```
14 steps × 3 calls = 42회 backbone 호출 → 전체 추론 시간의 ~50%

#### 해결책: CachedEuler 솔버

논문의 Modality-Aware Step Caching을 단순화한 CachedEuler 솔버:
- **Full step**: 기존대로 `dynamics_fn()` 호출 (3회 backbone)
- **Cached step**: 이전 step의 velocity를 재사용 (0회 backbone)

```python
# ai/subprocess/cached_solver.py
class CachedEuler(ODESolver):
    def __init__(self, cache_stride=3, warmup_steps=2):
        ...

    def _is_full_step(self, step_idx):
        if step_idx < self.warmup_steps:
            return True
        return ((step_idx - self.warmup_steps) % self.cache_stride) == 0

    def solve_iter(self, dynamics_fn, x_init, times, *args, **kwargs):
        x_t = x_init
        cached_velocity = None
        for step_idx, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
            dt = t1 - t0
            if self._is_full_step(step_idx) or cached_velocity is None:
                velocity = dynamics_fn(x_t, t0, *args, **kwargs)
                cached_velocity = velocity
            else:
                velocity = cached_velocity
            x_t = linear_approximation_step(x_t, dt, velocity)
            yield x_t, t0
```

#### Step 패턴 (14 steps, stride=3, warmup=2)

```
Step:  0  1  2  3  4  5  6  7  8  9  10 11 12 13
Type:  F  F  F  C  C  F  C  C  F  C  C  F  C  C
       ^  ^  ^           ^           ^
     warmup  |-- stride=3 패턴 반복 --|

F=Full (3 backbone calls), C=Cached (0 calls)
Full steps: 6개, Cached steps: 8개
Backbone calls: 6×3 = 18 (기존 42에서 57% 감소)
```

#### 런타임 솔버 교체

```python
# ai/subprocess/persistent_3d_worker.py (initialize 메서드)
from cached_solver import CachedEuler
ss_gen = pipe.models["ss_generator"]
ss_gen._solver = CachedEuler(cache_stride=3, warmup_steps=2)
```

FlowMatching/ShortCut의 `generate_iter()`가 `self._solver.solve_iter()`를 호출하므로,
solver 교체만으로 캐싱이 적용됨. upstream 코드 수정 불필요.

#### 효과 (compile=False 기준)

| 객체 | Baseline | Phase A | Speedup | W err | D err | H err |
|------|----------|---------|---------|-------|-------|-------|
| Nightstand | 19.1s | 16.5s | **1.16x** | 2.3% | 0.1% | 1.9% |
| Bed | 25.6s | 17.9s | **1.43x** | 0.1% | 0.2% | 0.3% |
| Television | 10.0s | 5.3s | **1.90x** | 1.2% | 2.7% | *9.4% |

*Television H 오차: 캐싱 없이도 4.3% 자연 변동 (depth=0.018인 극박 객체)

### 12.3 Phase B: SLaT Generator Step Caching (비활성화)

SLaT Generator는 4 steps, CFG 비활성화 (strength=0)로 step당 1회 backbone 호출만 수행.
stride=2로 캐싱 시 TV 등 얇은 객체에서 5%+ 치수 오차 발생 → **비활성화 유지**.

4 steps에서 캐싱 효과가 제한적 (4→3 calls, ~25% 감소)이고 품질 리스크가 높아
투자 대비 효과가 낮음.

### 12.4 stdout 오염 방지

Warp/kaolin 라이브러리가 import 시 stdout으로 초기화 메시지를 출력하여
JSON 프로토콜을 깨뜨리는 문제 해결:

```python
# 환경변수로 Warp 출력 억제
os.environ["WARP_QUIET"] = "1"

# 모델 로딩/추론 시 stdout을 /dev/null로 리다이렉트
def suppress_stdout():
    send_message._real_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

# JSON 메시지 전송 시 자동 복원
def send_message(msg_obj):
    if hasattr(send_message, '_real_stdout'):
        sys.stdout = send_message._real_stdout
    print(msg_obj.to_json(), flush=True)
```

### 12.5 종합 최적화 효과 (2026-03-18)

```
┌──────────────────────────────────────────────────────────────────────┐
│ 최적화 전후 비교 (L4 GPU, 단일 객체 기준)                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  초기 (V1, 순차):                                                    │
│  ████████████████████████████████████████  ~150초                    │
│                                                                      │
│  V2 파이프라인 최적화 (2026-03-13):                                   │
│  ████████  ~20초                                                     │
│                                                                      │
│  V2.5 + Fast-SAM3D (2026-03-18):                                    │
│  █████  ~13초                                                        │
│                                                                      │
│  개선율: 150s → 13s = 11.5x 가속                                     │
│                                                                      │
│  VRAM: 21GB → 11.25GB (48% 절감, 동일 GPU에 YOLOE+SAM3D 탑재)       │
└──────────────────────────────────────────────────────────────────────┘
```

| 시나리오 | 2026-03-13 | 2026-03-18 | 개선 |
|----------|-----------|-----------|------|
| 1 객체 | ~20초 | ~13초 | 1.5x |
| 3 객체 (1 GPU) | ~60초 | ~40초 | 1.5x |
| 3 객체 (2 GPU) | ~40초 (역할분리) | ~27초 (동일GPU) | 1.5x |
| VRAM/GPU | 21GB | 11.25GB | 48% 절감 |

---

## 13. Multi-GPU 확장성 및 GPU 스펙별 성능 예측 (2026-03-18)

### 13.1 Work-Stealing 스케줄링

기존 라운드로빈 스케줄링을 Event 기반 work-stealing으로 개선:

```python
# 변경 전 (폴링, 0.5초 지연)
while time.time() - start < timeout:
    worker = try_acquire()
    if worker: return worker
    await asyncio.sleep(0.5)  # 최대 0.5초 유휴

# 변경 후 (Event, 즉시 할당)
while time.time() - start < timeout:
    worker = try_acquire()
    if worker: return worker
    await self._worker_available.wait()  # 워커 반환 즉시 깨어남
```

워커 반환 시 `_worker_available.set()`으로 대기 중인 task에 즉시 시그널.
객체 크기가 다를 때 (TV 5초 vs Bed 17초) **먼저 끝난 GPU가 다음 작업을 즉시 가져감**.

### 13.2 GPU 수별 처리 시간 예측

#### 조건: 3 이미지, 12 객체, 평균 13초/객체

```
1단계 (YOLOE): ~2초 (이미지 수 / GPU 수 라운드, 매우 빠름)
2단계 (SAM-3D): 객체 수 / GPU 수 라운드 × 13초
3단계 (후처리): ~2초
```

| GPU 수 | SAM-3D 라운드 | 예상 총 시간 | 1GPU 대비 |
|--------|-------------|-------------|----------|
| 1 | 12 | **~160초** | 1.0x |
| 2 | 6 | **~82초** | 1.9x |
| **4** | **3** | **~43초** | **3.7x** |
| 6 | 2 | ~30초 | 5.3x |
| 8 | 2 (4 GPU 유휴) | ~30초 | 5.3x |

> **Note**: 8 GPU에서 12 객체는 GPU 4개가 유휴. 객체 수 ÷ GPU 수 = 라운드 수이므로,
> **12 객체 기준 최적 GPU 수는 4-6개**.

#### 규모별 최적 GPU 수

| 객체 수 | 최적 GPU | 예상 시간 | 비고 |
|---------|---------|----------|------|
| 3 | 2-3 | 15-27초 | 2 GPU면 충분 |
| 6 | 3-4 | 20-30초 | |
| 12 | 4-6 | 30-43초 | 4 GPU 권장 |
| 20 | 4-8 | 35-67초 | |
| 50 | 8+ | 82-160초 | GPU 선형 확장 |

### 13.3 양자화 실험 결과 (2026-03-18)

L4 GPU에서 양자화의 실질적 효과를 검증:

| 방법 | FP16 대비 속도 | 결과 |
|------|--------------|------|
| bitsandbytes INT8 | **0.36x (2.8배 느림)** | 양자화/역양자화 오버헤드 >> 연산 절약 |
| torch.compile inductor | 0.94x (동등) | FP16 Tensor Core 이미 최적 |

**결론**: L4의 FP16 Tensor Core (121 TFLOPS)가 이미 충분히 빠르며,
현재 PyTorch 생태계의 INT8 양자화 도구(bitsandbytes)는 runtime 오버헤드로 인해 순손실.
효과적인 양자화에는 `torch_tensorrt` 또는 `torchao`의 커널 레벨 INT8 fusion이 필요.

### 13.4 추가 최적화 실험 결과 (2026-03-18)

| 실험 | 결과 | 채택 |
|------|------|------|
| SS cache stride=4 | 속도 +6-11%, **치수 24-6108% 오차** | ❌ 품질 붕괴 |
| DINOv2 TensorRT/compile | 1회 27.5ms, 객체당 110ms (전체 0.7%) | ❌ 이미 빠름 |
| DINOv2 SS↔SLaT 캐싱 | weight/preprocessor 다름 | ❌ 구조적 불가 |
| bitsandbytes INT8 | FP16 대비 2.8배 느림 | ❌ 오버헤드 |
| SLaT step caching | 4 step에서 얇은 객체 품질 저하 | ❌ 비활성화 유지 |

> **현재 설정 (stride=3 + compile reduce-overhead)이 L4에서 training-free로 도달 가능한 최적점**.
> 추가 가속은 A100 GPU 또는 `torchao`/`torch_tensorrt` 도입이 필요.

### 13.5 GPU 스펙별 성능 비교 (L4 vs A100)

| 항목 | NVIDIA L4 | NVIDIA A100 80GB | 배수 |
|------|-----------|-----------------|------|
| **아키텍처** | Ada Lovelace (sm_89) | Ampere (sm_80) | - |
| **FP16 Tensor Core** | 121 TFLOPS | 312 TFLOPS | **2.6x** |
| **INT8 Tensor Core** | 242 TOPS | 624 TOPS | 2.6x |
| **메모리 대역폭** | 300 GB/s | 2,039 GB/s | **6.8x** |
| **VRAM** | 24 GB | 80 GB | 3.3x |
| **TDP** | 72W | 300W | 0.24x (효율적) |
| **가격 (GCP)** | ~$0.7/hr | ~$3.7/hr | 5.3x |

#### SAM-3D 추론 시간 예측 (객체당)

| 단계 | L4 (실측) | A100 (예측) | 근거 |
|------|----------|------------|------|
| SS Generator (14 steps) | ~8s | ~3-4s | FP16 2.6x + 메모리 대역폭 |
| SLaT Generator (4 steps) | ~3s | ~1-2s | FP16 2.6x |
| Condition Embedding | ~0.1s | ~0.05s | 이미 빠름 |
| GS Decoder | ~0.5s | ~0.2s | |
| **객체당 합계** | **~13s** | **~5-7s** | **~2x** |

> **Note**: 메모리 대역폭이 6.8x 차이나므로, memory-bound 연산(attention의 KV 읽기)에서
> A100이 FP16 TFLOPS 차이(2.6x)보다 더 큰 이점을 가질 수 있음.
> 최적화 문서 기준 A100 4GPU에서 객체당 ~7-8초 → 현재 최적화 적용 시 ~5-6초 예상.

#### GPU 수 × 스펙 조합별 12 객체 처리 시간

| 구성 | 객체당 | 12 객체 | 비용/hr |
|------|--------|---------|--------|
| L4 × 2 | ~13s | **~82초** | ~$1.4 |
| L4 × 4 | ~13s | **~43초** | ~$2.8 |
| A100 × 2 | ~6s | **~40초** | ~$7.4 |
| A100 × 4 | ~6s | **~22초** | ~$14.8 |

> **비용 효율**: L4 4대($2.8/hr, 43초) vs A100 2대($7.4/hr, 40초)
> → 유사한 성능에서 L4 4대가 **2.6배 저렴**

---

## 참고 파일

| 파일 | 설명 |
|------|------|
| `ai/utils/image_ops.py` | 이미지 전처리 유틸리티 (CLAHE 캐싱) |
| `ai/gpu/gpu_pool_manager.py` | YOLOE용 GPU Pool Manager (1단계) |
| `ai/gpu/sam3d_worker_pool.py` | SAM-3D Persistent Worker Pool (2단계) |
| `ai/subprocess/persistent_3d_worker.py` | SAM-3D Persistent 워커 (성능 최적화 설정 포함) |
| `ai/subprocess/cached_solver.py` | CachedEuler 솔버 (Fast-SAM3D Phase A) |
| `ai/subprocess/worker_protocol.py` | 워커-풀 통신 프로토콜 |
| `ai/pipeline/furniture_pipeline.py` | V2 파이프라인 오케스트레이터 |
| `ai/config.py` | Multi-GPU 설정 |
| `api/routes/furniture.py` | /analyze-furniture 엔드포인트 |
| `api/services/callback.py` | 비동기 Callback 서비스 |
| `docs/fast_sam3d.pdf` | Fast-SAM3D 논문 (arXiv:2602.05293) |
