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
# ai/subprocess/generate_3d_worker.py:631-644
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

부피 계산만 필요한 경우 GIF 렌더링을 스킵합니다.

```python
# ai/processors/6_SAM3D_convert.py:53
skip_gif: bool = True  # 기본값: GIF 스킵

# 효과: 15-30초 절약
```

### 5.3 Inference Steps 감소

```python
# ai/subprocess/persistent_3d_worker.py:58-62
# Stage1 (Sparse Structure): 테스트 결과 15 steps가 최적
STAGE1_INFERENCE_STEPS = 15  # 기본값 25 → 15

# Stage2 (SLAT): 8 steps가 최적
STAGE2_INFERENCE_STEPS = 8   # 기본값 12 → 8
```

#### Stage1 Steps 테스트 결과

| Steps | 부피 오차 | 속도 향상 | 권장 |
|-------|----------|----------|------|
| 25 | baseline | 1.00x | - |
| 20 | +5.47% | 1.23x | ⚠️ 주의 |
| **15** | **+1.31%** | **1.47x** | ✅ **권장** |
| 12 | +11.04% | 1.65x | ❌ 비권장 |
| 10 | +15.09% | 1.84x | ❌ 비권장 |

**결론**: `stage1_steps=15`가 최적의 균형점 (부피 오차 1.31%, 속도 47% 향상)

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
# ai/subprocess/persistent_3d_worker.py:67-70
GAUSSIAN_ONLY_MODE = True  # GLB/Mesh 생성 스킵

# 효과: 37.4% 속도 향상, 부피 오차 0.005% (무시 가능)
# 부피 계산만 필요한 경우 권장
```

### 5.7 in_place=True 최적화

```python
# ai/subprocess/persistent_3d_worker.py:457-461
# deepcopy 제거로 메모리/속도 최적화
scene_gs = self.make_scene(output, in_place=True)
scene_gs = self.ready_gaussian_for_video_rendering(
    scene_gs, in_place=True, fix_alignment=False
)

# 효과: 메모리 복사 오버헤드 제거, ~5-10% 속도 향상
```

### 5.8 torch.compile 활성화

```python
# ai/subprocess/persistent_3d_worker.py:320-325
# 워커 초기화 시 CUDA 커널 컴파일
self.sam3d_inference = Inference(config_path, compile=True, device="cuda")

# 효과:
# - 초기화 시: warmup 3회로 ~20-30초 추가
# - 추론 시: CUDA 커널 재사용으로 ~10-20% 속도 향상
# - Persistent Worker이므로 초기화 비용은 서버 시작 시 1회만 발생
```

### 효과 요약

| Phase | 최적화 | 효과 | 부피 영향 |
|-------|--------|------|----------|
| 1 | 이미지 다운샘플링 비활성화 | 부피 정확도 유지 | 91.7% 영향 방지 |
| 2 | Stage1 Steps (25→15) | **47% 속도 향상** | **1.31%** |
| 2 | Stage2 Steps (12→8) | 15-20% 속도 향상 | ~4% |
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
# ai/subprocess/generate_3d_worker.py:32-37
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
# ai/subprocess/generate_3d_worker.py:29
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

### 해결책: Subprocess 격리

```python
# ai/processors/6_SAM3D_convert.py:128-134
result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    timeout=timeout,
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
# ai/subprocess/generate_3d_worker.py:213-251
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
| 3 | SAM-3D 추론 (stage1=15, stage2=8, compile=True) | ~6-7초 |
| 4 | Gaussian-only 디코딩 | ~0.5초 |
| 5 | 후처리/GIF | 0초 (비활성화/스킵) |
| | **총합** | **~7-8초** |

**적용된 최적화**:
- Stage1 Steps: 25 → 15 (47% 빠름)
- Stage2 Steps: 12 → 8 (15-20% 빠름)
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

> **Note**: 현재 시간은 stage1=15, stage2=8, compile=True, Gaussian-only 모드 적용 기준

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
| Stage1 Steps 감소 (25→15) | ~4초 | **~3%** |
| Stage2 Steps 감소 (12→8) | ~3초 | **~2%** |
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

#### 적용된 최적화 설정 (2026-01)

```python
# ai/subprocess/persistent_3d_worker.py
MAX_IMAGE_SIZE = None           # Phase 1: 다운샘플링 비활성화
STAGE1_INFERENCE_STEPS = 15     # Phase 2: Stage1 (25→15, 47% 빠름, 1.3% 오차)
STAGE2_INFERENCE_STEPS = 8      # Phase 2: Stage2 (12→8, 15-20% 빠름)
USE_BINARY_PLY = True           # Phase 3: Binary PLY (70% 작음, 50% 빠름)
GAUSSIAN_ONLY_MODE = True       # Phase 5: Gaussian-only (37% 빠름)
compile=True                    # torch.compile (10-20% 빠름)
in_place=True                   # deepcopy 제거 (5-10% 빠름)
```

---

## 참고 파일

| 파일 | 설명 |
|------|------|
| `ai/gpu/gpu_pool_manager.py` | YOLOE용 GPU Pool Manager (1단계) |
| `ai/gpu/sam3d_worker_pool.py` | SAM-3D Persistent Worker Pool (2단계) |
| `ai/subprocess/generate_3d_worker.py` | SAM-3D subprocess 워커 |
| `ai/subprocess/persistent_3d_worker.py` | SAM-3D persistent 워커 |
| `ai/pipeline/furniture_pipeline.py` | V2 파이프라인 오케스트레이터 |
| `ai/config.py` | Multi-GPU 설정 |
