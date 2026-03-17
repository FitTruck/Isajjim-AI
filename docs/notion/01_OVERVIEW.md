# Isajjim AI - 프로젝트 Overview

## 한 줄 소개

2D 가구 이미지에서 객체를 탐지하고, 3D 모델을 생성하여 상대 치수를 계산하는 AI 파이프라인 서비스

---

## 서비스 목적

이사짐 견적 서비스에서 사용자가 업로드한 실내 사진을 분석하여:

1. 어떤 가구가 있는지 자동 탐지
2. 각 가구의 상대적 크기(width, depth, height) 산출
3. 백엔드에서 절대 치수/부피를 계산할 수 있도록 데이터 제공

---

## 파이프라인 V2 (현재)

```
사용자 이미지 (Firebase URL)
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. 이미지    │     │  2. YOLOE-seg│     │  3. SAM-3D   │     │  4. 치수 계산 │
│  다운로드     │────▶│  탐지 + 마스크│────▶│  3D 모델 생성 │────▶│  OBB 기반    │
│  (Firebase)  │     │  (365 classes)│     │ (Worker Pool)│     │  상대 치수    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  DB 매칭      │
                     │  라벨 반환    │
                     │  (base_name) │
                     └──────────────┘
```

### V1 → V2 핵심 변경

| 항목 | V1 | V2 (현재) |
|------|----|----|
| 탐지 모델 | YOLO-World + SAHI | **YOLOE-seg** (단일 모델) |
| 마스크 생성 | SAM2 (center point prompt) | **YOLOE-seg 마스크 직접 사용** |
| 분류 | CLIP → DB | **YOLO 클래스 → DB 직접 매칭** |
| 3D 워커 | 매 요청마다 subprocess 생성 | **Persistent Worker Pool** |
| 병렬 처리 | 이미지별 순차 | **이미지 + 객체 2단계 병렬** |
| 부피 계산 | AI에서 절대 부피 | **상대 치수만 반환** (절대값은 백엔드) |

---

## 핵심 아키텍처

### 2단계 병렬 처리

```
┌─────────────────────────────────────────────────────┐
│  1단계: 이미지 병렬 처리 (GPUPoolManager - YOLOE)    │
│                                                     │
│  img1 → GPU0 (YOLOE) ─┐                            │
│  img2 → GPU1 (YOLOE) ─┼─ 각 이미지에서 객체 탐지    │
│  img3 → GPU2 (YOLOE) ─┤   + 세그멘테이션 마스크      │
│  img4 → GPU3 (YOLOE) ─┘                            │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  2단계: 객체 병렬 처리 (SAM3DWorkerPool)             │
│                                                     │
│  탐지된 객체들을 라운드로빈으로 Worker에 분배          │
│  Worker0(GPU0) → obj1 3D │ Worker1(GPU1) → obj2 3D │
│  Worker2(GPU2) → obj3 3D │ Worker3(GPU3) → obj4 3D │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  3단계: 결과 취합 → Callback URL로 전송              │
└─────────────────────────────────────────────────────┘
```

### 프로세스 격리

| 프로세스 | 역할 | GPU 사용 |
|---------|------|---------|
| **API 서버** (메인) | FastAPI + YOLOE 탐지 | YOLOE 모델 per GPU |
| **SAM-3D Worker** (subprocess) | 3D 모델 생성 | SAM-3D 모델 per GPU |

spconv 라이브러리의 GPU 상태 충돌 문제로, SAM-3D는 반드시 별도 프로세스에서 실행해야 합니다.

---

## 성능 지표

### 벤치마크 (8 GPU, 8 이미지, 101 객체)

| 항목 | 값 |
|------|-----|
| 총 처리 시간 | ~3분 47초 (226초) |
| 객체당 평균 | **2.24초** |
| 부피 정확도 | ~4% 오차 이내 |

### V1 대비 개선

| 시나리오 | V1 (순차) | V2 (병렬+최적화) | 개선 |
|---------|----------|-----------------|------|
| 1 이미지 × 1 객체 | ~150초 | ~8초 | **19배** |
| 4 이미지 × 3 객체 | ~1836초 | ~22초 | **83배** |
| 10 이미지 × 5 객체 | ~7650초 | ~89초 | **86배** |

---

## 기술 스택

| 분류 | 기술 | 버전/설명 |
|------|------|----------|
| API 프레임워크 | FastAPI + Uvicorn | ASGI 서버 |
| 탐지 모델 | YOLOE-seg | `yoloe-26x-seg.pt`, Objects365 365 classes |
| 3D 생성 | SAM-3D (sam-3d-objects) | Facebook Research, Gaussian Splat |
| GPU 프레임워크 | PyTorch | >=2.1.0, CUDA 11.8+ |
| 3D 분석 | trimesh | OBB 기반 치수 계산 |
| HTTP 클라이언트 | aiohttp | 비동기 Firebase/Callback |
| 언어 | Python | 3.10+ |

### 인프라 요구사항

| 항목 | 최소 사양 |
|------|----------|
| GPU | CUDA 11.8+, 32GB+ VRAM (권장) |
| 디스크 | 50GB+ (모델 저장) |
| OS | Linux (Ubuntu 추천) |

---

## 핵심 파일 구조

```
api/                          # FastAPI 애플리케이션
├── app.py                    # 메인 앱, 라우터 등록, startup/shutdown
├── config.py                 # 환경변수, 디바이스, Callback 설정
├── models.py                 # Pydantic 요청 모델
├── routes/
│   ├── furniture.py          # 가구 분석 엔드포인트 4개
│   └── health.py             # /health, /gpu-status, /assets-list
└── services/
    └── callback.py           # 비동기 Callback 서비스 (retry 포함)

ai/                           # AI 모듈
├── config.py                 # GPU/모델/탐지 설정
├── gpu/
│   ├── gpu_pool_manager.py   # YOLOE용 Multi-GPU Pool (라운드로빈)
│   └── sam3d_worker_pool.py  # SAM-3D Persistent Worker Pool
├── processors/
│   ├── 1_firebase_images_fetch.py  # 이미지 다운로드
│   ├── 2_YOLO_detect.py           # YOLOE-seg 탐지
│   ├── 4_DB_movability_check.py   # DB 라벨 매핑
│   └── 7_volume_calculate.py      # OBB 치수 계산
├── pipeline/
│   └── furniture_pipeline.py      # V2 파이프라인 오케스트레이터
├── subprocess/
│   ├── persistent_3d_worker.py    # SAM-3D Worker (성능 최적화 설정)
│   └── worker_protocol.py         # JSON 통신 프로토콜
└── data/
    └── knowledge_base.py          # 가구 DB (24개 카테고리, 동의어 매핑)
```

---

## 외부 연동

| 서비스 | 용도 | 방향 |
|--------|------|------|
| Firebase Storage | 사용자 이미지 호스팅 | AI ← Firebase (이미지 다운로드) |
| 백엔드 API | 견적 시스템 | AI → 백엔드 (Callback 결과 전송) |
| sam-3d-objects | 3D 모델 생성 라이브러리 | 로컬 (subprocess) |
| HuggingFace | 모델 다운로드 | 초기 설정 시 1회 |

### Callback 흐름

```
백엔드 → POST /analyze-furniture (estimate_id, image_urls)
    ↓
AI 서버 → 즉시 {"success": true, "status": "processing"} 반환
    ↓ (백그라운드 처리)
AI 서버 → POST https://api.isajjim.kro.kr/api/v1/estimates/{estimateId}/callback
    ↓
백엔드 ← {"results": [{image_id, objects: [{label, type, width, depth, height}]}]}
```
