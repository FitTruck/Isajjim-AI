# Pipeline Optimization Guide

이 문서는 SAM3D-API 파이프라인에 적용된 최적화 기법들을 분석하고 정리합니다.

## 목차

**아키텍처 및 병렬 처리**
1. [V2 파이프라인 아키텍처 최적화](#1-v2-파이프라인-아키텍처-최적화)
2. [2단계 병렬 처리 아키텍처](#2-2단계-병렬-처리-아키텍처)
3. [Multi-GPU 병렬 처리 (1단계 — YOLOE)](#3-multi-gpu-병렬-처리-1단계)
4. [SAM-3D Worker Pool (2단계 — Persistent)](#4-sam-3d-worker-pool-2단계)

**추론 및 환경 최적화**
5. [SAM-3D 추론 최적화](#5-sam-3d-추론-최적화) (후처리/Steps/PLY/Gaussian-only/compile 등 8개)
6. [환경 변수 최적화](#6-환경-변수-최적화) (스레드/spconv/CUDA 격리)
7. [프로세스 격리](#7-프로세스-격리)
8. [Synthetic Pinhole Pointmap](#8-synthetic-pinhole-pointmap)

**성능 분석**
9. [초기 대비 최적화 효과 분석](#9-초기-대비-최적화-효과-분석)
10. [이미지 전처리 최적화 (CLAHE 캐싱)](#10-이미지-전처리-최적화)

**최신 최적화 (2026-03-18)**
11. [VRAM 최적화 (모델 언로드 48% 절감)](#11-vram-최적화-2026-03-18)
12. [Fast-SAM3D 기반 추론 가속](#12-fast-sam3d-기반-추론-가속-2026-03-18) (Phase A/B/C)
13. [Multi-GPU 확장성 및 GPU 스펙별 성능 예측](#13-multi-gpu-확장성-및-gpu-스펙별-성능-예측-2026-03-18)

**비교 및 요약**
14. [Fast-SAM3D 논문 vs 이삿짐 서비스 독자 최적화 구분](#14-fast-sam3d-논문-vs-이삿짐-서비스-독자-최적화-구분)

> 각 최적화 섹션 상단에는 **정의** 및 **원리** 블록쿼트가 있어 빠른 이해를 돕습니다.

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

- `ai/pipeline/furniture_pipeline.py:1-23` - V2.5 파이프라인 설명 (docstring)

### 효과

- **Latency 감소**: SAM2 API 호출 제거 (~2-5초)
- **코드 단순화**: 의존성 및 유지보수 복잡도 감소
- **마스크 품질 향상**: YOLOE-seg가 객체 전체를 더 정확하게 커버

---

## 2. 2단계 병렬 처리 아키텍처

> **정의**: 파이프라인의 서로 다른 단계에서 서로 다른 granularity (이미지 단위 vs 객체 단위)로 병렬성을 각각 적용하는 아키텍처 패턴.
>
> **원리**: 단일 granularity로만 병렬 처리하면 작업량이 병목 단계에 쏠려 GPU 활용률이 낮아집니다. 각 단계의 자연스러운 병렬 단위(YOLOE는 이미지 단위, SAM-3D는 객체 단위)에 맞게 병렬성을 개별 적용하면 전체 throughput이 각 단계의 병렬성을 곱한 값에 가까워집니다.

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
│              2단계: 객체 병렬 처리 (Event 기반 Work-Stealing)              │
│                     (SAM3DWorkerPool)                                   │
│                                                                         │
│   모든 객체 → asyncio.gather(N coroutines)                               │
│                           │                                             │
│                           ▼                                             │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  빈 Worker가 있으면 즉시 획득 / 없으면 Event 대기             │      │
│   │  Worker free → _worker_available.set() → 대기 코루틴 깨어남  │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   ※ 고정 라운드가 아닌 연속적 dispatch                                   │
│      먼저 끝난 GPU가 다음 작업을 즉시 가져감 (work-stealing)             │
│      → 객체 크기 불균등(TV 5초 vs Bed 17초)에 효율적                     │
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
2단계 (SAM-3D - Event 기반 Work-Stealing, L4 GPU):
┌────────────────────────────────────────────────────┐
│  12개 객체를 asyncio.gather로 일괄 제출             │
│                                                    │
│  초기: obj1..4 → Worker0..3 (즉시 dispatch)        │
│  obj5 이후: 워커가 free 되는 즉시 다음 작업 획득     │
│                                                    │
│  객체 크기가 불균등해도 GPU 유휴 시간 최소화         │
│  (고정 라운드 아님 - 자세한 동작은 Section 13.1)    │
│                                                    │
│  총 ~40-43초 (객체당 평균 ~13초 × ~3 라운드)       │
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
# ai/pipeline/furniture_pipeline.py:610-699 (process_multiple_images_with_ids)
async def process_multiple_images_with_ids(self, image_items, ...):
    pool = self.gpu_pool or get_gpu_pool()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_gpu(user_image_id, url):
        async with semaphore:
            async with pool.pipeline_context(task_id=f"img_{user_image_id}") as (gpu_id, pipeline):
                return await pipeline.process_single_image(url, ...)

    # 모든 이미지 동시 처리
    tasks = [process_with_gpu(user_id, url) for user_id, url in image_items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

#### 2단계: 객체 병렬 처리

```python
# ai/pipeline/furniture_pipeline.py:354-411
async def _parallel_3d_generation(self, image, objects_with_masks):
    sam3d_pool = get_sam3d_worker_pool()

    # 작업 목록 생성 (객체별)
    tasks = []
    for obj_id, obj in objects_with_masks:
        tasks.append({
            "task_id": f"obj_{obj_id}",
            "image_b64": image_b64,
            "mask_b64": obj.mask_base64,
            "seed": 42
        })

    # 모든 객체 동시 제출 → Event 기반 work-stealing으로 분배
    worker_results = await sam3d_pool.submit_tasks_parallel(tasks)
```

### 효과

4 이미지 × 3 객체 (총 12 객체), 4× L4 GPU 기준:

| 처리 방식 | 소요 시간 | 효율성 |
|----------|----------|--------|
| V1 완전 순차 | ~1836초 | 1배 (기준) |
| V1 이미지만 병렬 | ~459초 | ~4배 |
| V2.5 + Fast-SAM3D (현재) | **~43초** | **~43배** |

> 상세 breakdown은 [Section 9.2](#92-다중-이미지객체-처리-시간-비교) 참고.

---

## 3. Multi-GPU 병렬 처리 (1단계)

> **정의**: 여러 GPU를 리소스 풀로 관리하고, 요청마다 하나의 GPU를 할당/반환하는 리소스 풀링 패턴.
>
> **원리**: 딥러닝 모델은 GPU 1개에 고정(pinned)되는 경향이 있습니다 (weight가 특정 device에 올라감). 이를 여러 GPU에 복제해두고 요청마다 다른 GPU에 분배하면 이론상 N개 GPU → N배 throughput을 얻을 수 있습니다. 풀 매니저가 "사용 가능 여부" 상태를 추적하여 동시에 같은 GPU를 쓰지 않도록 보장합니다.

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

> **정의**: `N`개 리소스에 대해 `i % N` 순서로 순환하며 배분하는 가장 단순한 스케줄링 알고리즘.
>
> **원리**: 카운터 하나만 유지하면 되므로 구현과 동기화가 간단하고, 장기적으로 **균등 분배**를 보장합니다. 단점은 작업 크기가 불균등할 때 늦은 작업을 받은 GPU가 계속 밀리면서 전체 병목이 된다는 점 (Section 13.1의 work-stealing으로 해결).

```python
# ai/gpu/gpu_pool_manager.py:101-142
async def acquire(self, task_id: Optional[str] = None) -> int:
    start_time = time.time()
    while True:
        async with self._allocation_lock:
            for _ in range(len(self.gpu_ids)):
                gpu_id = self.gpu_ids[self._next_gpu_index]
                self._next_gpu_index = (self._next_gpu_index + 1) % len(self.gpu_ids)

                gpu_info = self._gpus[gpu_id]
                if gpu_info.is_available and gpu_info.error_count < self.max_retries:
                    if await self._check_gpu_health(gpu_id):
                        gpu_info.is_available = False
                        gpu_info.current_task_id = task_id
                        return gpu_id
        # 타임아웃 체크 후 0.5초 폴링 재시도
        if time.time() - start_time > self.wait_timeout:
            raise RuntimeError(...)
        await asyncio.sleep(0.5)
```

> **Note**: YOLOE GPU 풀(1단계)은 여전히 라운드로빈 + 0.5s 폴링 방식입니다.
> SAM3D Worker Pool(2단계)은 Event 기반 work-stealing으로 업그레이드되었습니다 ([Section 13.1](#131-work-stealing-스케줄링) 참고).

#### 3.2 파이프라인 사전 초기화

> **정의**: 딥러닝 모델을 서버 시작 시점에 GPU에 미리 로드해두고 요청마다 재사용하는 기법.
>
> **원리**: 모델 로딩은 weight file 읽기 + CUDA로의 tensor 전송 + kernel 컴파일로 3-5초가 소요됩니다. 이를 요청 경로에서 제거하고 startup 시점으로 옮기면 사용자가 느끼는 latency에서 완전히 제거됩니다. N개 GPU에 각각 모델 인스턴스를 유지해 병렬 처리도 보장합니다.

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

- **처리량 증가**: N개 GPU로 최대 N배 병렬 처리 (이미지 수 ≥ GPU 수일 때)
- **모델 로드 시간 제거**: 요청당 3-5초 절약 (YOLOE pre-initialization)
- **균등 분배**: 라운드로빈으로 이미지가 N개 GPU에 번갈아가며 할당됨

> YOLOE 추론은 이미지당 ~0.5-1초로 매우 빠르므로, 1단계에서는 라운드로빈의 "균등 분배" 특성이
> 더 중요합니다. 2단계 SAM-3D처럼 작업 크기 불균등(TV 5초 vs Bed 17초)이 크지 않기 때문에
> work-stealing으로의 업그레이드 우선순위는 낮습니다.

---

## 4. SAM-3D Worker Pool (2단계)

> **정의**: 서버 시작 시 워커 프로세스를 N개 spawn하고 종료하지 않은 채 IPC로 작업을 주고받는 **persistent process pool** 패턴.
>
> **원리**: subprocess spawn + 모델 로드 비용(3-5초/요청)을 제거하기 위해 프로세스를 살아있는 상태로 유지합니다. 메인 API와는 stdin/stdout JSON 프로토콜로 통신하여 언어/환경 독립적. 또한 spconv 같이 GPU 상태를 전역으로 유지하는 라이브러리의 충돌도 **프로세스 경계**로 격리되어 해결됩니다.

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
# ai/gpu/sam3d_worker_pool.py:42-117
class SAM3DWorkerPool:
    """
    GPU당 하나의 persistent 워커 프로세스를 관리합니다.
    워커는 모델을 미리 로드하고, 작업 요청이 오면 Event 기반 work-stealing으로
    대기 중인 작업에 즉시 dispatch합니다.
    """

    # ai/gpu/sam3d_worker_pool.py:406-445
    async def submit_tasks_parallel(self, tasks):
        """여러 작업을 병렬로 제출 (내부적으로 Event 기반 dispatch)"""
        results = await asyncio.gather(
            *[submit_one(t) for t in tasks],
            return_exceptions=True
        )
        return results
```

> **스케줄링 방식**: 초기에는 라운드로빈 + 0.5초 폴링이었으나, 2026-03-18부터
> Event 기반 work-stealing으로 변경되어 객체 크기가 불균등할 때 GPU 유휴 시간을 최소화합니다.
> 자세한 동작은 [Section 13.1](#131-work-stealing-스케줄링) 참고.

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

> **정의**: SAM-3D 파이프라인의 선택적 후처리 단계(`texture_baking`, `mesh_postprocess`, `layout_postprocess`)를 끄는 설정.
>
> **원리**: 이들 후처리는 Gaussian latent 출력 이후에 실행되는 **선택적** 단계로, 시각적 품질(텍스처 굽기, 메시 리메시, 레이아웃 정규화)을 향상시키는 목적입니다. 본 서비스는 부피 계산에만 PLY를 사용하므로 시각적 품질이 불필요하고, 해당 단계를 스킵하면 52-105초 절약됩니다.

```python
# ai/subprocess/persistent_3d_worker.py:635-648
output = self.pipe.run(
    image=image,
    mask=mask_u8,
    seed=task.seed,
    pointmap=pointmap,
    decode_formats=decode_formats,          # Gaussian-only 모드: ["gaussian"]
    stage1_inference_steps=STAGE1_INFERENCE_STEPS,  # 14
    stage2_inference_steps=STAGE2_INFERENCE_STEPS,  # 4
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

### 5.2 GIF/비디오 렌더링 미구현

이삿짐 서비스는 부피(m³) 계산만 필요하므로 워커에 **GIF/비디오 렌더링 코드가 아예 없습니다**.
SAM-3D 원본에서는 렌더링 기능을 제공하지만, 본 프로젝트의 `persistent_3d_worker.py`는 PLY 저장만 수행합니다.

```python
# ai/subprocess/persistent_3d_worker.py — process_task() 내부
scene_gs = self.make_scene(output, in_place=True)
scene_gs = self.ready_gaussian_for_video_rendering(scene_gs, in_place=True, fix_alignment=False)
scene_gs.save_ply(output_ply_path)  # ← PLY만 저장, GIF/비디오 생성 없음
```

> **참고**: `ready_gaussian_for_video_rendering`은 이름과 달리 렌더링이 아닌 **정규화만 수행**합니다.
> 실제 GIF/MP4 생성은 추가 단계가 필요하지만 본 프로젝트에서는 수행하지 않습니다.

### 5.3 Inference Steps 감소

> **정의**: Diffusion 모델의 denoising step 수를 기본값보다 줄여 추론 속도를 높이는 기법.
>
> **원리**: Diffusion 모델은 노이즈에서 출력까지 N번의 iterative step으로 생성합니다. 각 step이 backbone forward pass를 포함하므로 총 시간은 `O(N)`. N을 절반으로 줄이면 시간도 거의 절반. 단, 너무 줄이면 품질(본 서비스에선 **부피 정확도**)이 급격히 저하되므로 experimental sweet spot을 찾아야 합니다. Stage1은 3D 형상, Stage2는 텍스처/디테일을 담당하며 각각 다른 최적점이 있습니다.

```python
# ai/subprocess/persistent_3d_worker.py:73-74
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

> **정의**: PLY 파일을 사람이 읽을 수 있는 ASCII 대신 **little-endian binary**로 저장하는 포맷 선택.
>
> **원리**: ASCII PLY는 각 float을 `"1.23456789 "`처럼 문자열로 직렬화하여 vertex당 ~50-100 bytes가 필요합니다. Binary PLY는 `float32` 4 bytes 고정 + `uint8` 1 byte로 vertex당 고정 15 bytes. 크기 ~70% 감소 + 파싱/포맷팅 오버헤드 제거로 I/O 속도 ~50% 향상. base64 인코딩하여 JSON으로 전송하는 본 파이프라인에서는 payload 크기 감소가 네트워크 latency로도 직결됩니다.

```python
# ai/subprocess/persistent_3d_worker.py:78
USE_BINARY_PLY = True

# 효과: 파일 크기 70% 감소, 쓰기 속도 50% 향상
```

### 5.5 이미지 다운샘플링 (비활성화)

> **정의**: 일반적으로는 입력 이미지 해상도를 낮춰 추론 속도를 높이는 기법이지만, 본 서비스에서는 **의도적으로 비활성화**한 결정.
>
> **원리**: SAM-3D는 내부적으로 이미지를 518×518로 리사이즈하므로 사전 다운샘플링이 전처리 시간을 줄여줍니다. 그러나 본 서비스의 실험 결과 다운샘플링이 **부피 오차의 91.7%를 차지**하는 지배적 요인이었고, 특히 작은 객체(pillow, lamp)에서 최대 576% 오차가 발생했습니다. 이삿짐 견적은 부피가 금액에 직결되므로 속도 이득을 포기하고 정확도를 택했습니다.

```python
# ai/subprocess/persistent_3d_worker.py:67
MAX_IMAGE_SIZE = None  # 비활성화

# 이유: 다운샘플링이 부피 정확도에 91.7% 영향
# 특히 작은 객체에서 최대 576% 부피 차이 발생
```

### 5.6 Gaussian-only 모드

> **정의**: SAM-3D의 출력 포맷 중 3D Gaussian Splatting만 생성하고 Mesh/GLB 디코딩을 건너뛰는 모드.
>
> **원리**: SAM-3D는 같은 latent로부터 여러 출력 포맷을 생성합니다 (gaussian, mesh, glb). Mesh 디코딩은 별도 transformer(`slat_decoder_mesh`)를 통과하는 추가 단계입니다. 본 서비스는 부피 계산을 위해 **point cloud만** 필요하고, Gaussian Splatting의 각 점이 이미 3D 좌표를 가지므로 이것으로 OBB 부피를 계산할 수 있습니다. Mesh 관련 단계를 완전히 스킵하여 37.4% 속도 향상 + VRAM 절감을 동시에 달성.

```python
# ai/subprocess/persistent_3d_worker.py:80-83
GAUSSIAN_ONLY_MODE = True  # GLB/Mesh 생성 스킵, decode_formats=["gaussian"]

# 효과: 37.4% 속도 향상, 부피 오차 0.005% (무시 가능)
# 부피 계산만 필요한 경우 권장 (현재 활성화됨)
```

### 5.7 in_place=True 최적화

> **정의**: 함수가 새 객체를 만들지 않고 입력 객체를 직접 수정(mutate)하는 방식.
>
> **원리**: `make_scene()`과 `ready_gaussian_for_video_rendering()`의 기본 동작은 안전성을 위해 `deepcopy`로 입력을 복사한 뒤 수정합니다. 그러나 Gaussian tensor는 수백 MB 크기이므로 deepcopy에 메모리 할당 + 복사 시간이 누적됩니다. 본 파이프라인은 입력 객체를 더 이상 사용하지 않으므로 in-place mutation이 안전하고, ~5-10% 속도 향상 + 메모리 피크 감소 효과를 얻습니다.

```python
# ai/subprocess/persistent_3d_worker.py:652-656
# deepcopy 제거로 메모리/속도 최적화
scene_gs = self.make_scene(output, in_place=True)
scene_gs = self.ready_gaussian_for_video_rendering(
    scene_gs, in_place=True, fix_alignment=False
)

# 효과: 메모리 복사 오버헤드 제거, ~5-10% 속도 향상
```

### 5.8 torch.compile 활성화

> **정의**: PyTorch 2.0+의 JIT 컴파일러로, Python 레벨의 모델 코드를 추적하여 최적화된 CUDA 커널로 변환하는 기법.
>
> **원리**: PyTorch의 eager mode는 매 연산마다 Python 인터프리터 오버헤드 + 개별 CUDA 커널 호출이 발생합니다. `torch.compile`은 TorchDynamo로 graph capture → TorchInductor가 **fused CUDA kernel**을 생성합니다. 효과는 (1) Python overhead 제거, (2) operator fusion (여러 연산이 하나의 커널로 병합), (3) kernel AUTOTUNE (여러 구현 중 최적 선택). Persistent worker 환경에서는 첫 실행 시에만 컴파일 비용을 지불하고 이후 모든 요청이 빠른 경로를 사용합니다.

```python
# ai/subprocess/persistent_3d_worker.py:361 — SAM3D 로드 시 compile=False
# (SAM3D 내부 _warmup()의 run_layout_model 버그 우회)
self.sam3d_inference = Inference(config_path, compile=False, device="cuda")

# ai/subprocess/persistent_3d_worker.py:468 — Phase C에서 수동으로 선별 compile
ENABLE_COMPILE = True
if ENABLE_COMPILE:
    compile_mode = "reduce-overhead"  # max-autotune은 첫 실행 10분+
    ss_gen.reverse_fn.inner_forward = torch.compile(..., mode=compile_mode, fullgraph=True)
    ss_dec.forward = torch.compile(..., mode=compile_mode, fullgraph=True)
    self.pipe.embed_condition = torch.compile(..., mode=compile_mode, fullgraph=False)
    # 자체 warmup으로 AUTOTUNE 캐시 생성 (Section 12.1 참고)
```

> **주의**: SAM3D는 항상 `compile=False`로 로드합니다. 전체 모델을 `compile=True`로 로드 시
> `_warmup()`에서 `run_layout_model` 버그가 발생하므로, Phase C에서 핵심 모듈만 수동으로
> `torch.compile`을 적용합니다. 자세한 내용은 [Section 12.1](#121-phase-c-torchcompile--autotune-캐시-영속화) 참고.

**효과** (Phase C 단독 측정, baseline: compile=False, Step Caching off):
- 초기화 시: warmup으로 추가 시간 소요 (AUTOTUNE 캐시 영속화로 재시작 시 재사용)
- 추론 시: CUDA 커널 재사용으로 **Bed 25.6→18.7s (1.37x), TV 10.0→7.6s (1.31x)**
- Persistent Worker이므로 초기화 비용은 서버 시작 시 1회만 발생
- 실제 배포에서는 Phase A(SS Step Caching)와 조합 적용 → 객체당 ~13초 (Section 9.1 참고)

### 효과 요약

| Phase | 최적화 | 효과 | 부피 영향 |
|-------|--------|------|----------|
| 1 | 이미지 다운샘플링 비활성화 | 부피 정확도 유지 | 91.7% 영향 방지 |
| 2 | Stage1 Steps (25→14) | **~50% 속도 향상** | **~1.5%** |
| 2 | Stage2 Steps (12→4) | ~30% 속도 향상 | ~0.5% |
| 3 | Binary PLY | 쓰기 50% 빠름 | 없음 |
| 5 | Gaussian-only 모드 | **37.4% 속도 향상** | 0.005% |
| - | in_place=True | 5-10% 속도 향상 | 없음 |
| - | torch.compile (Phase C) | Bed 1.37x / TV 1.31x | 없음 |
| - | SS Step Caching (Phase A) | Bed 1.43x / TV 1.90x | ~1-3% |
| - | 후처리 비활성화 | 52-105초 절약 | 없음 |

> **Note**: GIF/비디오 렌더링은 본 파이프라인에 **구현되어 있지 않으므로** 별도 최적화 대상이 아닙니다.
> V1에서 있던 GIF 15-30초 오버헤드는 V2에서 해당 코드 경로 자체가 제거된 상태입니다 (Section 5.2 참고).

**총 예상 성능 향상** (Fast-SAM3D 적용 후, 단일 객체 L4 GPU 기준): **~11-12배 빠름** (150초 → ~13초, 부피 정확도 유지)

---

## 6. 환경 변수 최적화

### 6.1 스레드 폭발 방지

> **정의**: BLAS/OpenMP 기반 라이브러리들이 자동으로 CPU 코어 수만큼 스레드를 생성하는 것을 환경 변수로 제한하는 기법.
>
> **원리**: OpenMP, OpenBLAS, MKL 등은 기본적으로 `nproc()`(시스템 CPU 코어 수)만큼 worker thread를 spawn합니다. 다중 워커 환경에서는 `워커 수 × 스레드 수 × 라이브러리 수` = 수백~수천 개 스레드가 같은 CPU 코어를 두고 경쟁하게 되어, 실제 연산 시간보다 **context switch 오버헤드가 압도적**이 됩니다. 4개로 제한하면 스레드 경쟁이 줄고 캐시 친화적이 되어 오히려 빨라집니다.

```python
# ai/subprocess/persistent_3d_worker.py:45-49
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# PyTorch 스레드 제한 (line 55-56)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
```

**문제**: 기본 설정에서 각 라이브러리가 CPU 코어 수만큼 스레드 생성 → 스레드 폭발

**해결**: 스레드 수를 4개로 제한하여 컨텍스트 스위칭 오버헤드 감소

### 6.2 spconv 튜닝 시간 제한

> **정의**: spconv(sparse convolution) 라이브러리의 알고리즘 auto-tuning에 시간 제한을 두는 환경 변수.
>
> **원리**: spconv는 입력 shape에 대한 최적 sparse convolution algorithm을 찾기 위해 여러 커널 variant를 벤치마킹합니다. 기본 설정에서는 튜닝이 수렴하지 않아 초반 추론이 매우 느려지거나 무한 루프에 걸릴 수 있습니다. `SPCONV_ALGO_TIME_LIMIT=100`(ms)으로 제한하면 충분히 빠른 알고리즘을 찾는 즉시 튜닝을 종료합니다.

```python
# ai/subprocess/persistent_3d_worker.py:30
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "100"  # 100ms 제한

# 문제: spconv가 최적 알고리즘을 찾기 위해 무한 튜닝
# 해결: 튜닝 시간을 100ms로 제한
```

### 6.3 CUDA 디바이스 격리

> **정의**: `CUDA_VISIBLE_DEVICES` 환경 변수를 사용하여 프로세스가 인지하는 GPU를 한 개로 제한하는 기법.
>
> **원리**: CUDA 드라이버는 이 환경 변수에 명시된 GPU만 노출하고, **항상 device 0으로 remap**합니다. 즉 워커 0은 GPU 2를, 워커 1은 GPU 3을 보지만 둘 다 자신의 입장에서는 "device 0"입니다. 이 덕분에 (1) 멀티 워커가 서로의 GPU를 침범할 수 없고, (2) `cuda:0`을 하드코딩한 라이브러리(spconv 등)도 호환되며, (3) `SPCONV_TUNE_DEVICE=0`처럼 디바이스 ID도 항상 0으로 통일할 수 있습니다.

```python
# Multi-GPU 환경에서 특정 GPU만 보이게 설정
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# SPCONV_TUNE_DEVICE는 항상 0 (remap 되므로)
os.environ["SPCONV_TUNE_DEVICE"] = "0"
```

---

## 7. 프로세스 격리

> **정의**: 작업 단위를 OS 프로세스 경계로 분리하는 아키텍처 패턴.
>
> **원리**: Python 인터프리터의 GIL, CUDA context, spconv 같은 라이브러리의 전역 상태 등은 하나의 프로세스 내에서 격리하기 어렵습니다. 이를 별도 프로세스로 분리하면 (1) 메모리 누수가 프로세스 종료 시 자동 회수되고, (2) 한 프로세스의 crash가 다른 프로세스에 영향을 주지 않으며, (3) 각 프로세스가 독립된 CUDA context를 가져 GPU 상태 충돌이 없어집니다. 비용은 IPC overhead이지만, 본 케이스에서는 작업당 ~7-13초의 GPU 연산이라 IPC 비용이 무시할 수준입니다.

### 문제점

spconv 라이브러리는 GPU 상태를 유지하며, 같은 프로세스에서 여러 번 로드하면 충돌이 발생합니다.
또한 SAM-3D는 전용 conda 환경(`sam3d-objects`)의 Python을 사용해야 합니다.

### 해결책: Persistent Worker Pool + Subprocess 격리

```python
# ai/gpu/sam3d_worker_pool.py:98-101 — SAM-3D 전용 Python 선택
sam3d_python = os.path.expanduser("~/miniconda3/envs/sam3d-objects/bin/python")
python_executable = sam3d_python if os.path.exists(sam3d_python) else sys.executable

# ai/gpu/sam3d_worker_pool.py:168-185 — 워커 프로세스 시작 (positional args)
cmd = [
    self.python_executable,
    self.worker_script,
    str(worker_info.worker_id),   # argv[1] = worker_id
    str(gpu_id)                   # argv[2] = gpu_id (워커가 자체적으로 CUDA_VISIBLE_DEVICES 설정)
]
worker_info.process = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)
```

```python
# ai/subprocess/persistent_3d_worker.py:21-25 — 워커에서 GPU 격리 설정
if len(sys.argv) >= 3:
    gpu_id = int(sys.argv[2])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # torch/spconv import 이전에 설정
```

### 장점

1. **spconv 충돌 방지**: 서버 시작 시 GPU당 1회만 SAM-3D 로드 → 같은 프로세스에서 중복 로드 이슈 회피
2. **상태 격리**: 한 요청의 실패가 다른 요청/메인 프로세스에 영향 없음
3. **전용 conda 환경 사용**: 메인 API는 일반 Python, 워커는 `sam3d-objects` 환경으로 분리
4. **Persistent 구조**: 워커는 서버 라이프타임 동안 살아 있으며 (매 요청마다 재생성 아님),
   stdin/stdout JSON 프로토콜로 작업을 주고받음 → 모델 로딩 오버헤드 0

---

## 8. Synthetic Pinhole Pointmap

> **정의**: depth estimation 모델 대신 **균일 깊이(uniform depth) + pinhole 카메라 모델**로 합성한 3D point map.
>
> **원리**: SAM-3D는 2D 이미지에서 3D를 복원할 때 각 픽셀의 3D 위치(pointmap)를 입력으로 받습니다. 원래는 MoGe(monocular depth) 모델이 깊이를 예측하지만, 간헐적으로 NaN/Inf가 발생하고 ~3GB VRAM을 사용합니다. 본 서비스는 **절대 깊이가 아닌 상대 비율**만 필요하므로(부피 계산을 위한 OBB는 회전/스케일 불변), `Z=1` 균일 평면 + pinhole 공식 `X = (u-cx)/f * Z`로 합성해도 충분합니다. 안정성 + 메모리 절감을 동시 달성.

### 문제점

SAM-3D의 MoGe 모듈이 카메라 intrinsics를 추정할 때 실패하거나 NaN/Inf 값을 생성하는 경우가 있었습니다.

### 해결책

```python
# ai/subprocess/persistent_3d_worker.py:174-192
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

> **벤치마크 기준**: L4 GPU, 단일 객체 기준. 수치는 [Section 12.5](#125-종합-최적화-효과-2026-03-18)의 Fast-SAM3D 적용 후 측정값과 일관됩니다.

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

#### 현재 상태 (V2.5 + Fast-SAM3D, 2026-03-18 이후)

| 단계 | 작업 | 시간 |
|------|------|------|
| 1 | YOLOE-seg 탐지 (사전 로드) | 0.5-1초 |
| 2 | SAM-3D 모델 로드 (Persistent Worker Pool) | 0초 |
| 3 | SAM-3D 추론 (stage1=14, stage2=4, SS Step Caching, torch.compile) | ~11-12초 |
| 4 | Gaussian-only 디코딩 | ~0.5초 |
| 5 | 후처리 | 0초 (비활성화) |
| | **총합** | **~13초** (L4 기준) |

**적용된 최적화 (V2.2 + V2.5 + Fast-SAM3D)**:
- Stage1 Steps: 25 → 14 (~50% 빠름, V2.2)
- Stage2 Steps: 12 → 4 (~30% 빠름, V2.2)
- Gaussian-only 모드: 37.4% 빠름 (V2.2)
- in_place=True: 5-10% 빠름 (V2.2)
- VRAM 모델 언로드: 21GB → 11.25GB (Fast-SAM3D, 2026-03-18)
- torch.compile Phase C (reduce-overhead): Bed 1.37x, TV 1.31x (Fast-SAM3D)
- SS Step Caching Phase A (stride=3): Bed 1.43x, TV 1.90x (Fast-SAM3D)

#### 단일 객체 최적화 효과

```
초기:  ████████████████████████████████████████████████████  ~150초
현재:  █████                                                  ~13초

절약:  약 137초 (91% 감소)
속도:  약 11.5배 향상 (L4 GPU 기준)
```

> **참고**: V2.2 시점(2026-01) 벤치마크에서는 ~20초였고, Fast-SAM3D 적용(2026-03-18) 후 ~13초로 추가 개선되었습니다.

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

##### 현재 (2단계 병렬 처리 + Fast-SAM3D)

```
1단계 (YOLOE 병렬, 라운드로빈):
  GPU0: img1 ─┐
  GPU1: img2 ─┼── ~1초 (동시)
  GPU2: img3 ─┤
  GPU3: img4 ─┘

2단계 (SAM-3D, Event 기반 Work-Stealing):
  12개 객체 → asyncio.gather로 일괄 제출
  초기: obj1..4 → Worker0..3 (즉시 dispatch)
  이후: 워커가 free 되는 즉시 다음 작업 획득 (크기 차이 흡수)
  총: ~40-43초 (객체당 평균 ~13초, 4 GPU에서 3 라운드)

3단계 (취합): ~2초
────────────────────────────────────────────────────────────────────────────────
총: ~43초 (0.72분)
```

> **Note**: Work-stealing 덕분에 객체 크기가 불균등해도 (TV 5초 vs Bed 17초) GPU 유휴
> 시간이 거의 없습니다. 고정 라운드로빈이라면 각 라운드가 최대 크기 객체에 의해 제한되지만,
> work-stealing은 빠른 GPU가 작은 작업 여러 개를 연속 처리합니다.

##### 비교

```
초기:  ████████████████████████████████████████████████████████████  1836초 (30.6분)
현재:  ██                                                             43초 (0.72분)

절약:  1793초 (97.7% 감소)
속도:  약 43배 향상
```

---

### 9.3 규모별 최적화 효과

4 GPU(L4) + Fast-SAM3D(stride=3, compile reduce-overhead) + Event 기반 work-stealing 기준:

| 시나리오 | 초기 | 현재 | 절약 | 배수 |
|----------|------|------|------|------|
| 1 이미지 × 1 객체 | 150초 | ~13초 | 91.3% | **~11배** |
| 1 이미지 × 3 객체 | 450초 | ~15초 | 96.7% | **~30배** |
| 4 이미지 × 1 객체 | 600초 | ~15초 | 97.5% | **~40배** |
| 4 이미지 × 3 객체 | 1836초 | ~43초 | 97.7% | **~43배** |
| 10 이미지 × 3 객체 | 4590초 | ~100초 | 97.8% | **~46배** |
| 10 이미지 × 5 객체 | 7650초 | ~165초 | 97.8% | **~46배** |

> **Note**: 현재 시간은 `stage1=14`, `stage2=4`, `Gaussian-only`, `ENABLE_SS_STEP_CACHING=True`,
> `ENABLE_COMPILE=True (reduce-overhead)` 설정에 L4 GPU 4개 + work-stealing 기준.
> 객체 수 ÷ GPU 수가 작아질수록(1 이미지 × 1 객체 등) YOLOE 오버헤드의 상대적 비중이 커져 배수가 감소합니다.
> 자세한 GPU 수별 예측은 [Section 13.2](#132-gpu-수별-처리-시간-예측) 참고.

---

### 9.4 최적화 기여도 분석

각 최적화 기법이 전체 성능 향상에 기여한 정도를 분석합니다.

#### 단일 객체 기준 (V1 150초 → V2.5+Fast-SAM3D 13초, L4 GPU)

| 최적화 | 절약 시간 | 기여도 | 버전 |
|--------|----------|--------|------|
| 후처리 비활성화 (texture_baking 등) | 52-105초 | **36-77%** | V2.0 |
| V1의 GIF 렌더링 단계 제거 | 15-30초 | **11-22%** | V2.0 |
| V2 파이프라인 (SAM2/CLIP 제거) | 3-7초 | **2-5%** | V2.0 |
| 모델 사전 로드 (YOLOE + SAM-3D) | 7-11초 | **5-8%** | V2.0 |
| Stage1 Steps 감소 (25→14) | ~4초 | **~3%** | V2.2 |
| Stage2 Steps 감소 (12→4) | ~4초 | **~3%** | V2.2 |
| Gaussian-only 모드 | ~9초 | **~7%** | V2.2 |
| in_place=True | ~1-2초 | **~1%** | V2.2 |
| torch.compile Phase C (reduce-overhead) | ~3-5초 | **~3%** | 2026-03-18 |
| SS Step Caching Phase A (stride=3) | ~5-7초 | **~4-5%** | 2026-03-18 |
| **합계** | **~137초** | **~91%** | |

> V1의 "GIF 렌더링 15-30초"는 V1 파이프라인에서 실제로 수행되던 단계였으며, V2에서는 해당
> 코드 경로 자체가 제거되었습니다 (Section 5.2 참고). 이는 "스킵"이 아니라 "미구현" 상태입니다.

#### 다중 이미지/객체 기준 (추가 병렬화 효과)

| 최적화 | 효과 |
|--------|------|
| 이미지 병렬 처리 (GPUPoolManager, 라운드로빈) | N개 GPU → 최대 N배 속도 |
| 객체 병렬 처리 (SAM3DWorkerPool + Work-Stealing) | M개 객체 → ~`ceil(M/N)` 라운드 (크기 불균등 흡수) |
| **복합 효과 (4 GPU, 4×3 시나리오)** | **~43배 속도 향상** |

---

### 9.5 결론

L4 GPU 기준, V1 대비 V2.5 + Fast-SAM3D 적용 후 측정/추정값:

| 측면 | 초기 (V1) | 현재 (V2.5 + Fast-SAM3D) | 개선율 |
|------|----------|--------------------------|--------|
| **단일 객체 처리** | ~150초 | ~13초 | **91% 감소 (11.5x)** |
| **다중 이미지/객체 처리** (4 이미지 × 3 객체, 4 GPU) | ~1836초 | ~43초 | **97.7% 감소 (43x)** |
| **파이프라인 단계** | 5단계 | 2단계 | **60% 감소** |
| **모델 로드 오버헤드** | 7-11초/요청 | 0초 (Persistent Worker Pool) | **100% 제거** |
| **SAM-3D VRAM** | ~21GB | 11.25GB (Gaussian-only 모델 언로드) | **48% 감소** |
| **GPU 스케줄링** | 단일 GPU 순차 | Multi-GPU + Event work-stealing | **N배 + 불균등 분포 흡수** |

#### 적용된 최적화 설정 (2026-03-18 업데이트)

```python
# ai/subprocess/persistent_3d_worker.py
MAX_IMAGE_SIZE = None            # Phase 1: 다운샘플링 비활성화 (부피 정확도 유지)
STAGE1_INFERENCE_STEPS = 14      # Phase 2: Stage1 (25→14, ~50% 빠름, ~1.5% 오차)
STAGE2_INFERENCE_STEPS = 4       # Phase 2: Stage2 (12→4, ~30% 빠름, <0.5% 오차)
USE_BINARY_PLY = True            # Phase 3: Binary PLY (파일 크기 -70%, 쓰기 +50%)
GAUSSIAN_ONLY_MODE = True        # Phase 5: Gaussian-only (+37.4% 빠름, 0.005% 오차)
ENABLE_COMPILE = True            # Phase C: torch.compile reduce-overhead
                                 #          (Bed 1.37x, TV 1.31x)
ENABLE_SS_STEP_CACHING = True    # Phase A: SS Step Caching (stride=3, warmup=2)
                                 #          (Nightstand 1.16x, Bed 1.43x, TV 1.90x)
ENABLE_SLAT_STEP_CACHING = False # Phase B: SLaT Step Caching (비활성화, 품질 리스크)
# in_place=True                  # make_scene/ready_gaussian deepcopy 제거 (5-10% 빠름)
```

---

## 10. 이미지 전처리 최적화

### 10.1 CLAHE 객체 캐싱

> **정의**: OpenCV의 CLAHE(Contrast Limited Adaptive Histogram Equalization) 객체를 매 호출마다 재생성하지 않고 클래스 변수에 **싱글톤**으로 캐싱하는 기법.
>
> **원리**: `cv2.createCLAHE()`는 lookup table과 내부 grid 구조체를 초기화하는데 5-10ms가 소요됩니다. 이는 단일 호출에서는 무시할 수준이지만, 다중 이미지 ensemble detection에서는 이미지마다 호출되어 누적됩니다. 객체가 stateless하므로(파라미터: clipLimit, tileGridSize는 고정) 한 번 만들어두고 재사용해도 안전합니다.

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

> **정의**: Gaussian-only 모드에서 호출되지 않는 SAM-3D 서브 모델들의 weight를 GPU에서 CPU로 이동(또는 삭제)시켜 VRAM을 확보하는 기법.
>
> **원리**: PyTorch 모델의 `.cpu()` 메서드는 weight tensor를 GPU memory에서 CPU RAM으로 이동시킵니다. 이후 `del` + `torch.cuda.empty_cache()`로 GPU 메모리 fragment까지 정리. SAM-3D는 8개 서브 모델로 구성되는데, 본 서비스에선 5개만 실행 경로에 포함되고 나머지(`slat_decoder_mesh`, `slat_decoder_gs_4`, `depth_model`)는 절대 호출되지 않으므로 GPU에 둘 필요가 없습니다. 약 10GB(48%) VRAM 절감으로 L4(22GB) 단일 GPU에 YOLOE + SAM-3D를 함께 탑재할 수 있게 됩니다.

### 문제점

SAM-3D 모델이 L4 GPU (24GB 공식 / ~22GB 실사용 가능) VRAM의 ~21GB를 사용하여:
- YOLOE와 동일 GPU에 동시 탑재 불가
- 2 GPU 환경에서 1 GPU는 YOLOE 전용, 1 GPU는 SAM-3D 전용으로 사용해야 함
- 병렬 SAM-3D 처리 불가

> L4 공식 스펙은 24GB GDDR6이지만, CUDA context, driver overhead 등을 제외한 실사용 가능 메모리는 약 22GB입니다.

### SAM-3D 모델 구성 및 GPU 로딩 현황

SAM-3D는 내부적으로 8개의 서브 모델로 구성됩니다. 이삿짐 서비스의 Gaussian-Only 모드에서는 이 중 5개만 GPU에 로딩하고, 나머지 3개는 언로딩하여 VRAM을 절감합니다.

#### GPU에 로딩하는 모델 (활성, 5개)

| 모델 | 파이프라인 단계 | 역할 | 파라미터 | VRAM |
|------|--------------|------|---------|------|
| **`image_cond_model`** (DINOv2) | Condition Embedding | 입력 이미지+마스크를 visual token으로 인코딩. 모든 후속 단계의 조건(condition)으로 사용됨 | ~300M | ~1GB |
| **`ss_generator`** | Stage 1: SS Generator | **3D 구조(voxel) 생성** — DiT Block ×24로 iterative denoising하여 coarse한 3D shape + layout(회전/위치/스케일) 생성 | ~1,034M | ~4GB |
| **`ss_decoder`** | Stage 1→2 변환 | SS Generator의 latent 출력을 sparse voxel로 변환. 3D-CNN 기반. Stage 2(SLaT)의 입력 조건으로 사용 | 소규모 | ~0.5GB |
| **`slat_generator`** | Stage 2: SLaT Generator | **텍스처/디테일 생성** — DiT Block ×24로 coarse voxel에 appearance 신호를 iterative denoising하여 세부 형상 완성 | ~600M | ~3GB |
| **`slat_decoder_gs`** | Stage 3: GS Decoder | **SLaT 출력 → 3D Gaussian Splatting(PLY)** 변환. Swin-Transformer ×12 기반. **부피 계산용 포인트 클라우드 생성에 필수** | ~91M | ~1GB |

#### GPU에서 언로딩한 모델 (비활성, CPU 이동 후 삭제, 3개)

| 모델 | 원래 역할 | VRAM | 언로딩 근거 |
|------|----------|------|-----------|
| **`slat_decoder_mesh`** | SLaT 출력 → **Mesh(GLB)** 변환. Swin-Transformer ×12. 텍스처 베이킹, 메시 후처리 포함 | **~3-4GB** | Gaussian-Only 모드에서 `decode_formats=["gaussian"]`만 사용하므로 mesh 디코딩 코드 경로 자체가 실행되지 않음 |
| **`slat_decoder_gs_4`** | SLaT 출력 → **4-channel GS** 변환 (고해상도 버전). 기본 `slat_decoder_gs`의 상위 버전 | **~2-3GB** | Stage2 inference steps가 적을 때(=4) 기본 `slat_decoder_gs`가 자동 선택됨. 4-channel 버전은 호출되지 않음 |
| **`depth_model`** (MoGe) | 입력 이미지 → **monocular depth 추정** → 3D pointmap 생성. 카메라 intrinsics 복원에 사용 | **~1-3GB** | `make_synthetic_pointmap()`으로 대체. SAM-3D 내부에서 `if pointmap is None`일 때만 호출되므로, pointmap을 항상 제공하면 미호출 |

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
| 합계 (동일 GPU) | **OOM** (~21.4GB > 22GB 가용) | **11.61GB / ~22GB** | 동일 GPU에 탑재 가능 |

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

- `ai/subprocess/persistent_3d_worker.py:369-412` - VRAM cleanup 로직

---

## 12. Fast-SAM3D 기반 추론 가속 (2026-03-18)

Fast-SAM3D (arXiv:2602.05293) 논문의 기법을 적용한 training-free 추론 가속.

> **상위 정의**: 모델을 재학습하지 않고(추가 학습 비용 0) inference path만 수정하여 속도를 높이는 **training-free acceleration** 기법군.
>
> **원리**: 기존 모델의 weight를 그대로 사용하면서 (1) 컴파일러 최적화(Phase C), (2) iterative step 중 일부 캐싱(Phase A), (3) 디코더 단순화(Gaussian-only)를 조합하여 정확도 손실을 최소화하면서 속도를 높입니다. Re-training이 필요한 quantization과 달리 즉시 적용 가능합니다.

### 12.1 Phase C: torch.compile + AUTOTUNE 캐시 영속화

> **정의**: SAM3D 내부 `compile=True` 옵션의 버그를 우회하여 **핵심 모듈만 수동으로** `torch.compile` 적용하고, 컴파일 결과 캐시를 디스크에 영속화하여 워커 재시작 시 재컴파일을 방지하는 기법.
>
> **원리**: `torch.compile`의 mode 옵션은 컴파일 비용 vs 속도 향상의 trade-off가 있습니다. `max-autotune`은 가장 빠른 커널을 찾기 위해 여러 변형을 벤치마킹하여 첫 실행에 10분+가 걸리고, `reduce-overhead`는 Python 오버헤드만 제거하여 ~2분에 끝납니다. 후자를 선택해 합리적인 성능 + 빠른 시작을 얻고, 결과는 `TORCHINDUCTOR_CACHE_DIR`에 저장하여 다음 워커 시작 시 재사용됩니다. 핵심 모듈만 컴파일하는 이유는 SAM3D의 일부 모듈(`PointPatchEmbed` 등)이 `fullgraph=True`와 호환되지 않아 전체 컴파일이 실패하기 때문입니다.

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

#### 효과 (Phase C 단독, baseline: compile=False + Step Caching off)

| 항목 | 변경 전 | 변경 후 | 개선 |
|------|---------|---------|------|
| Bed 추론 시간 | 25.6s | 18.7s | **1.37x** |
| Television 추론 시간 | 10.0s | 7.6s | **1.31x** |
| 워커 초기화 warmup | N/A | ~280s (첫 실행, AUTOTUNE 벤치마크) | 서버 시작 시 1회 |
| 워커 재시작 warmup | N/A | ~수십초 (캐시 재사용) | 재컴파일 방지 |
| AUTOTUNE 캐시 파일 수 | 0 | **2,259** (`.cache/torch_compile/`) | 영속 저장 |

> **Note**: 실제 배포는 Phase C + Phase A를 **함께** 사용하므로 단독 수치보다 더 빠릅니다.
> Phase A 단독 측정은 Section 12.2, 복합 결과는 Section 9.1(객체당 ~13초) 참고.

### 12.2 Phase A: SS Generator Step Caching

> **정의**: Diffusion solver의 velocity field가 인접 step 간에 **완만하게 변한다는 관찰**을 이용해, 일부 step에서 backbone 호출 없이 이전 step의 velocity를 재사용하는 기법.
>
> **원리**: Diffusion 모델은 N step의 iterative denoising으로 출력을 생성합니다. SS Generator는 step마다 PointmapCFG를 통해 backbone을 3회(conditional + no-pointmap + unconditional) 호출하여 14 steps × 3 = 42회. Velocity field가 매 step 크게 바뀌지 않으므로, warmup step 후 `1/stride`만 full 계산하고 나머지는 캐시된 velocity를 재사용해도 결과가 거의 동일합니다. stride=3, warmup=2 설정에서 backbone 호출이 42 → 18로 57% 감소.
>
> **수학적 근거**: Velocity field `v(x_t, t)`는 t에 대해 Lipschitz continuous하므로, 인접 step 간 변화가 작습니다. Linear approximation `x_{t+dt} ≈ x_t + v(x_t, t) * dt`에서 `v`를 한 step만큼 stale하게 사용해도 오차는 `O(dt)`로 작은 편입니다.

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

> **정의**: 워커 프로세스의 stdout을 JSON 프로토콜 전용으로 보호하기 위해, 라이브러리 import/모델 추론 구간 동안 stdout을 `/dev/null`로 리다이렉트하는 기법.
>
> **원리**: SAM3DWorkerPool은 워커와 stdin/stdout JSON 메시지로 통신합니다. 그러나 Warp, kaolin 같은 GPU 라이브러리는 `import` 시점에 "Warp 0.x.y initialized..." 같은 init 메시지를 stdout으로 출력합니다. 이 텍스트가 JSON 메시지 사이에 끼면 매니저 측에서 `json.loads()` 실패. 해결책은 (1) `WARP_QUIET=1` 환경 변수로 일부 라이브러리 출력 억제, (2) `sys.stdout = open(os.devnull, 'w')`로 일시적 우회, (3) `send_message()` 호출 직전에만 real stdout 복원.

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
│  V2.2 (Gaussian-only + Steps 14/4, 2026-01-25):                      │
│  ████████  ~20초                                                     │
│                                                                      │
│  V2.5 + Fast-SAM3D (Phase A/C + VRAM, 2026-03-18):                  │
│  █████  ~13초                                                        │
│                                                                      │
│  개선율: 150s → 13s = 11.5x 가속                                     │
│                                                                      │
│  VRAM: 21GB → 11.25GB (48% 절감, 동일 GPU에 YOLOE+SAM3D 탑재)       │
└──────────────────────────────────────────────────────────────────────┘
```

| 시나리오 | V2.2 (2026-01-25) | V2.5 + Fast-SAM3D (2026-03-18) | 개선 |
|----------|-------------------|--------------------------------|------|
| 1 객체 | ~20초 | ~13초 | 1.5x |
| 3 객체 (1 GPU) | ~60초 | ~40초 | 1.5x |
| 3 객체 (2 GPU, 역할분리 vs 동일GPU) | ~40초 | ~27초 | 1.5x |
| VRAM/GPU | 21GB | 11.25GB | 48% 절감 |

---

## 13. Multi-GPU 확장성 및 GPU 스펙별 성능 예측 (2026-03-18)

### 13.1 Work-Stealing 스케줄링

> **정의**: 유휴 워커가 대기 중인 작업 큐에서 다음 작업을 즉시 "훔쳐(steal)" 가는 동적 스케줄링 전략. 본 구현에서는 `asyncio.Event` 시그널 기반.
>
> **원리**: 작업 크기가 균일하면 라운드로빈 = 최적입니다. 그러나 실제로는 객체마다 SAM-3D 추론 시간이 크게 다르고(TV ~5초, Bed ~17초), 라운드로빈 + 폴링은 빠른 GPU가 다음 폴링 cycle까지 유휴 상태로 대기합니다. Work-stealing은 워커가 작업을 끝낸 즉시 (1) 자신을 free 상태로 마킹, (2) `_worker_available.set()` 시그널 송신, (3) 대기 중인 코루틴이 깨어나 즉시 다음 작업 획득. 이렇게 하면 빠른 GPU가 작은 작업 여러 개를 병렬로 처리할 수 있어 GPU 유휴 시간이 거의 0에 가까워집니다.
>
> **이점**: 최악의 경우(모든 작업이 같은 크기)에도 라운드로빈과 동일한 성능을 보장하면서, 일반적인 경우(작업 크기 분포가 있음)에는 makespan이 개선됩니다. 추가 비용은 `asyncio.Event` 한 개와 boolean 체크뿐.

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

#### 조건: 4 이미지 × 3 객체 (총 12 객체), 평균 13초/객체, L4 GPU

```
1단계 (YOLOE): ~1-2초 (이미지 수 / GPU 수 라운드, 매우 빠름)
2단계 (SAM-3D): ⌈객체 수 / GPU 수⌉ 라운드 × ~13초
3단계 (후처리 + 취합): ~2초
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

## 14. Fast-SAM3D 논문 vs 이삿짐 서비스 독자 최적화 구분

이 섹션은 Fast-SAM3D (arXiv:2602.05293) 논문에서 제안된 기법과, **이삿짐 부피 추정 서비스의 목적에 맞춰 독자적으로 개발한 최적화**를 명확히 구분합니다.

### 14.1 Fast-SAM3D 논문에 포함된 기법 (선행 연구)

| 기법 | 논문 섹션 | 설명 | 본 프로젝트 적용 |
|------|----------|------|-----------------|
| Modality-Aware Step Caching | §4.1 | Shape/Layout 토큰을 분리하여 shape은 Taylor 외삽, layout은 momentum-anchored smoothing | **단순화 적용**: CachedEuler 솔버 (stride=3, warmup=2)로 velocity 재사용만 구현. 논문의 shape/layout 토큰 분리 및 momentum-anchored smoothing은 미적용 |
| Joint Spatiotemporal Token Carving | §4.2 | 시공간 saliency 기반 토큰 pruning + 동적 adaptive step caching | **미적용**: Gaussian-Only 모드에서 SLaT 자체가 간소화되어 효과 제한적 |
| Spectral-Aware Token Aggregation | §4.3 | FFT 기반 기하학적 복잡도 분석으로 mesh 디코딩 토큰 축소 | **미적용**: mesh 디코딩 자체를 완전 제거 (Gaussian-Only 모드) |

> **요약**: 논문의 3가지 핵심 기법 중 Step Caching의 단순화 버전만 적용. 나머지 2가지(Token Carving, Token Aggregation)는 Gaussian-Only 전략으로 대체됨.

---

### 14.2 이삿짐 서비스 독자 최적화 (본 프로젝트 고유 기여)

Fast-SAM3D 논문에 없으며, **이삿짐 부피 추정 서비스의 도메인 요구사항**에 맞춰 독자적으로 설계·구현한 최적화입니다.
대부분은 본 문서의 앞 섹션에서 자세히 다뤘으므로, 여기서는 **독자 기여 관점**에서 요약하고 **새로운 항목(E-2, F-1, F-2)**만 전개합니다.

#### 독자 최적화 요약 (A-D, E-1, E-3, G)

| 카테고리 | 기법 | 핵심 차별점 | 본 문서 참조 |
|---------|------|------------|-------------|
| **A. 파이프라인 재설계** | Gaussian-Only 디코딩 | 논문은 mesh 디코딩 최적화, 본 프로젝트는 mesh 제거 | [Section 5.6](#56-gaussian-only-모드) |
| | V2 파이프라인 (SAM2/CLIP/SAHI 제거) | YOLOE-seg 단일 호출로 5단계→2단계 통합 | [Section 1](#1-v2-파이프라인-아키텍처-최적화) |
| | 이미지 다운샘플링 비활성화 | 부피 정확도 기준(91.7% 영향)으로 최적화 경계 결정 | [Section 5.5](#55-이미지-다운샘플링-비활성화) |
| **B. VRAM 최적화** | 불필요 모델 GPU 언로딩 (−48%) | 실행 경로에서 호출되지 않는 3개 모듈 CPU 이동 | [Section 11](#11-vram-최적화-2026-03-18) |
| | Synthetic Pinhole Pointmap | MoGe 대체로 NaN/Inf 제거 + MoGe 언로드 가능 | [Section 8](#8-synthetic-pinhole-pointmap) |
| **C. 시스템 아키텍처** | Persistent Worker Pool | spconv 충돌을 프로세스 격리로 해결 | [Section 4](#4-sam-3d-worker-pool-2단계) + [Section 7](#7-프로세스-격리) |
| | 2단계 병렬 처리 | 이미지·객체 단위의 독립적 병렬화 | [Section 2](#2-2단계-병렬-처리-아키텍처) |
| | Event 기반 Work-Stealing | `asyncio.Event`로 크기 불균등 흡수 | [Section 13.1](#131-work-stealing-스케줄링) |
| **D. 추론 엔진** | torch.compile 수동 적용 (Phase C) | SAM-3D 내부 버그 우회 + 선별적 `reduce-overhead` 컴파일 | [Section 12.1](#121-phase-c-torchcompile--autotune-캐시-영속화) |
| | Inference Steps 최적점 탐색 (14+4) | 부피 오차 기준 실험으로 14+4가 sweet spot | [Section 5.3](#53-inference-steps-감소) |
| | in_place=True (deepcopy 제거) | Gaussian tensor 복사 제거로 5-10% 향상 | [Section 5.7](#57-in_placetrue-최적화) |
| **E. I/O 최적화** | Binary PLY 포맷 | ASCII→Binary로 파일 ~70%↓, 쓰기 ~50%↑ | [Section 5.4](#54-binary-ply-포맷) |
| | stdout 오염 방지 | Warp/kaolin import 출력을 `/dev/null` 리다이렉트 | [Section 12.4](#124-stdout-오염-방지) |
| **G. 런타임 안정화** | 스레드 폭발 방지 | OpenMP/MKL/BLAS 스레드를 4개로 제한 | [Section 6.1](#61-스레드-폭발-방지) |
| | spconv 튜닝 시간 제한 | 무한 튜닝 방지 (`SPCONV_ALGO_TIME_LIMIT=100`) | [Section 6.2](#62-spconv-튜닝-시간-제한) |

> 위 14개 기법 모두 Fast-SAM3D 논문에 없거나(A/B/C/G) 논문이 다루지 않는 엔지니어링 세부사항(D/E).
> 아래는 본 문서에서 별도로 다루지 않은 **신규 항목** 3개만 상세 전개합니다.

---

#### E-2. PLY 전처리 파이프라인 (GCS 업로드 전)

**동기**: SAM-3D가 생성한 PLY는 프론트엔드(Three.js) 렌더링 요구사항과 맞지 않고 파일 크기도 큽니다
(임의 회전, Z-up 좌표계, ~2MB, 수십만 포인트). GCS에 올려 모바일/웹에서 다운받으려면 전처리가 필수입니다.

**전처리 파이프라인**:
```
SAM-3D PLY 원본
    ↓
1. OBB 기반 축 정렬 (임의 회전 보정)
    ↓
2. 바닥 배치 (Z-min = 0)
    ↓
3. 좌표계 변환 (Z-up → Y-up, Three.js 호환)
    ↓
4. 절대 치수 스케일링 (mm → m)
    ↓
5. Stride 다운샘플링 (max 72,000 points)
    ↓
GCS 업로드 (크기: ~2MB → ~290KB, 85% 감소)
```

**의의**: SAM-3D 논문은 3D 생성까지만 다루며 **배포 후 렌더링 최적화는 제시하지 않음**.
서비스 수준(Three.js 호환성, 모바일 대역폭)을 고려한 엔지니어링 기여.

**코드**: `ai/processors/ply_preprocessor.py`

---

#### F-1. OBB 기반 상대 치수 추출

**동기**: 축 정렬된 Bounding Box(AABB)는 회전된 가구에서 부정확한 치수를 산출합니다
(예: 45도 회전된 책상은 AABB가 실제보다 1.4배 커짐).

**핵심 발상**: 3D Gaussian 포인트 클라우드에서 PCA 기반 Oriented Bounding Box(OBB)를 계산하고,
greedy coordinate mapping으로 width/depth/height를 추출합니다.

```python
# 1. PCA로 principal components 계산
# 2. OBB extents 추출 (3개 축 길이)
# 3. Greedy similarity mapping:
#    - 회전 행렬의 |R^T| 유사도 행렬 계산
#    - 내림차순 정렬 후 greedy 할당 (중복 방지)
#    - → 고유한 (width, depth, height) 매핑
```

**의의**: SAM-3D 논문은 3D 재구성만 다루며 **치수 추출 방법론은 제시하지 않음**.
회전 불변 OBB + greedy mapping은 이삿짐 서비스의 부피 추정을 위한 고유 기여.

**코드**: `ai/processors/7_volume_calculate.py:166-213`

---

#### F-2. 절대 부피 계산 (표준 치수 DB 매칭)

**동기**: SAM-3D가 출력하는 3D 모델은 **상대 비율만** 정확하며 절대 크기 정보가 없습니다
(이미지에서 카메라까지 거리를 모르기 때문). 이삿짐 견적에는 절대 부피(m³)가 필요합니다.

**핵심 발상**: 52개 가구 타입별 표준 치수 DB를 구축하고, OBB에서 추출한 상대 비율을 매칭하여
"이 가구는 SOFA이고 상대 비율이 3:1:0.9이면 실제 치수는 3000×1000×900mm"와 같이 계산합니다.

```python
def calculate_absolute_volume(label, type_name, rel_w, rel_d, rel_h):
    # 1. 가구 타입 매칭 (label + type_name → 표준 치수)
    furniture_type = get_furniture_type(type_name)

    # 2. 고정 치수 사용 (width, depth)
    actual_width = furniture_type.width    # mm
    actual_depth = furniture_type.depth    # mm

    # 3. 가변 높이 계산 (height == -1인 경우)
    if furniture_type.height != -1:
        actual_height = furniture_type.height
    else:
        # 상대 비율로 높이 역산 (침대 등 높이 다양한 가구)
        sorted_rel = sorted([rel_w, rel_d, rel_h])
        scale_factor = max(furniture_type.width, furniture_type.depth) / sorted_rel[2]
        actual_height = sorted_rel[0] * scale_factor

    # 4. 절대 부피 (m³)
    return actual_width * actual_depth * actual_height * 1e-9
```

**의의**: SAM-3D 자체는 scale-free 3D 생성이므로 "이 가구 상대 비율 + 이 가구 카테고리"만으로
절대 치수를 역산하는 것은 **도메인 지식(가구 표준 치수)과 AI의 결합**입니다.

**데이터**: 52개 가구 타입 (`ai/data/furniture_dimensions.py`), YOLO 365 클래스 → 가구 매핑 (`ai/data/knowledge_base.py`)

**코드**: `ai/processors/8_absolute_volume_calculate.py`

---

### 14.3 기여 분류 요약

| 분류 | 기법 | 출처 | 효과 |
|------|------|------|------|
| **논문 적용** | SS Step Caching (단순화) | Fast-SAM3D §4.1 | 1.16-1.90x 가속 |
| **서비스 독자 A** | Gaussian-Only 디코딩 | 이삿짐 서비스 | 67-135초/객체 절약 |
| **서비스 독자 A** | V2 파이프라인 통합 | 이삿짐 서비스 | 3-7초/요청 절약 |
| **서비스 독자 A** | 다운샘플링 비활성화 결정 | 부피 정확도 분석 | 부피 오차 91.7% 방지 |
| **서비스 독자 B** | VRAM 모델 언로딩 | 이삿짐 서비스 | 48% VRAM 절감 |
| **서비스 독자 B** | Synthetic Pinhole Pointmap | 이삿짐 서비스 | 안정성 향상 (NaN/Inf 방지) + MoGe 언로드 가능 |
| **서비스 독자 C** | Persistent Worker Pool | 시스템 아키텍처 | 모델 로딩 100% 제거 |
| **서비스 독자 C** | 2단계 병렬 처리 | 시스템 아키텍처 | ~43x 다중 객체 가속 (4 GPU) |
| **서비스 독자 C** | Event-Based Work-Stealing | 시스템 아키텍처 | GPU 유휴 시간 최소화 |
| **서비스 독자 D** | torch.compile 수동 적용 | 엔지니어링 | 1.31-1.37x 가속 |
| **서비스 독자 D** | Steps 최적점 탐색 (14+4) | 부피 기준 실험 | 1.5x 가속, ±1.5% 오차 |
| **서비스 독자 D** | in_place=True | 엔지니어링 | 5-10% 가속 |
| **서비스 독자 E** | Binary PLY + 전처리 | I/O 최적화 | 85% 크기 감소 |
| **서비스 독자 E** | stdout 오염 방지 | 시스템 안정성 | JSON 프로토콜 보호 |
| **서비스 독자 F** | OBB 상대 치수 추출 | 부피 추정 핵심 | 회전 불변 치수 |
| **서비스 독자 F** | 절대 부피 계산 (52타입 DB) | 부피 추정 핵심 | 실제 m³ 부피 산출 |
| **서비스 독자 G** | 스레드/spconv 안정화 | 런타임 안정성 | 다중 워커 안정 |

> **결론**: Fast-SAM3D 논문의 3가지 기법 중 1가지(Step Caching)만 단순화하여 적용하고, **나머지 16가지 최적화는 이삿짐 부피 추정 서비스의 도메인 요구사항에 맞춰 독자적으로 설계·구현**한 것입니다.

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
