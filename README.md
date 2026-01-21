# YOLOE-seg + SAM-3D API (V2)

이사 서비스를 위한 가구 탐지 및 3D 모델 생성 통합 API 서비스입니다.

> **V2 (2026-01)**: YOLOE-seg 마스크 직접 사용, CLIP/SAHI/SAM2 제거로 파이프라인 단순화

---

## 주요 기능

- **YOLOE-seg 탐지**: Objects365 기반 365개 클래스 탐지 + 인스턴스 세그멘테이션 마스크
- **SAM-3D 3D 생성**: 2D 이미지 + YOLO 마스크에서 3D Gaussian Splat, PLY, GLB 메시 생성
- **부피 계산**: 3D 모델 bounding box 기반 상대 부피/치수 계산
- **Multi-GPU 병렬 처리**: 라운드 로빈 GPU 할당을 통한 병렬 이미지 처리

---

## 빠른 시작

### 1. 설치

```bash
# Hugging Face CLI 설치 및 인증
pip install 'huggingface-hub[cli]<1.0'
huggingface-cli login

# 환경 설정 스크립트 실행 (sam-3d-objects 클론, 의존성 설치)
source setup.sh
```

### 2. 서버 실행

```bash
# 개발 환경
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 프로덕션 환경
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```

### 3. 상태 확인

```bash
curl http://localhost:8000/health
curl http://localhost:8000/gpu-status
```

API 문서: http://localhost:8000/docs

---

## AI Logic 파이프라인 V2

```
Firebase URL → YOLOE-seg (bbox + mask) → DB 매칭 → SAM-3D → 부피 계산
```

| 단계 | 파일 | 설명 |
|------|------|------|
| 1 | `1_firebase_images_fetch.py` | Firebase Storage URL에서 이미지 다운로드 |
| 2 | `2_YOLO_detect.py` | YOLOE-seg 객체 탐지 (bbox + class + mask) |
| 3 | `4_DB_movability_check.py` | YOLO 클래스로 DB 매칭, 한국어 라벨 반환 |
| 4 | `6_SAM3D_convert.py` | YOLO 마스크 → SAM-3D 3D 모델 생성 |
| 5 | `7_volume_calculate.py` | trimesh로 상대 부피/치수 계산 |

### V1 → V2 변경사항

| 항목 | V1 | V2 |
|------|------|------|
| 탐지 모델 | yolov8l-world.pt | yoloe-26x-seg.pt |
| SAHI/CLIP | 사용 | **제거** |
| 마스크 생성 | SAM2 (center point) | **YOLO 마스크 직접 사용** |
| API 호출 | 3회 | 2회 |
| 부피 단위 | 절대값 (m³) | **상대값** (백엔드 계산) |

---

## API 엔드포인트

### 가구 분석 (주요 엔드포인트)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/analyze-furniture` | 다중 이미지 가구 분석 (Firebase URL) |
| POST | `/analyze-furniture-single` | 단일 이미지 가구 분석 |
| POST | `/analyze-furniture-base64` | Base64 이미지 가구 분석 |
| POST | `/detect-furniture` | 탐지만 (3D 없음, 빠른 응답) |

### 시스템

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 확인 |
| GET | `/gpu-status` | GPU 풀 상태 |
| POST | `/generate-3d` | 3D 생성 (task_id 반환) |
| GET | `/generate-3d-status/{task_id}` | 3D 생성 결과 폴링 |
| GET | `/assets-list` | 저장된 에셋 목록 |
| GET | `/assets/{filename}` | 에셋 다운로드 |

---

## 요청/응답 예시

### 가구 분석 요청

```bash
curl -X POST http://localhost:8000/analyze-furniture \
  -H 'Content-Type: application/json' \
  -d '{
    "image_urls": [
      {"id": 101, "url": "https://firebase-url-1.jpg"},
      {"id": 102, "url": "https://firebase-url-2.jpg"}
    ]
  }'
```

### 응답

```json
{
  "results": [
    {
      "image_id": 101,
      "objects": [
        {
          "label": "sofa",
          "width": 1.5,
          "depth": 0.8,
          "height": 0.6,
          "volume": 0.72
        }
      ]
    }
  ]
}
```

**단위 설명:**
- `width`, `depth`, `height`: **상대 길이** (3D 메시 bounding box, 모델 좌표계)
- `volume`: **상대 부피** (bounding box 부피, 모델 좌표계)

> 절대 부피/치수는 백엔드에서 Knowledge Base 실제 치수와 비율을 조합하여 계산

### 빠른 탐지 (3D 없음)

```bash
curl -X POST http://localhost:8000/detect-furniture \
  -H 'Content-Type: application/json' \
  -d '{"image":"<BASE64_IMAGE>"}'
```

```json
{
  "success": true,
  "objects": [
    {
      "label": "sofa",
      "bbox": [100, 200, 400, 500],
      "center_point": [250, 350],
      "confidence": 0.95
    }
  ],
  "total_objects": 1,
  "processing_time_seconds": 0.5
}
```

---

## 디렉토리 구조

```
sam3d-api/
├── api.py                      # FastAPI 메인 서버
├── requirements.txt            # Python 의존성
├── setup.sh                    # 환경 설정 스크립트
├── assets/                     # 생성된 PLY/GIF/GLB 에셋
├── docs/                       # 문서
│   ├── tdd/TDD_PIPELINE_V2.md  # 기술 설계 문서
│   └── qa/                     # QA 테스트 리포트
├── ai/                         # AI 모듈
│   ├── config.py               # 설정 (GPU_IDS, 모델 경로)
│   ├── gpu/                    # GPU 풀 매니저
│   ├── processors/             # 파이프라인 프로세서
│   ├── pipeline/               # 파이프라인 오케스트레이터
│   ├── subprocess/             # SAM-3D worker (GPU 격리)
│   ├── data/                   # Knowledge Base
│   └── utils/                  # 유틸리티
├── sam-3d-objects/             # Facebook Research SAM-3D (setup.sh로 클론)
└── tests/                      # 테스트
```

---

## 운영 가이드

### 환경 변수

`api.py`와 `generate_3d_worker.py`에서 **torch import 전** 자동 설정됩니다:

```bash
# GPU 설정 (spconv 튜닝 이슈 방지)
CUDA_HOME=/usr/local/cuda
SPCONV_TUNE_DEVICE=0
SPCONV_ALGO_TIME_LIMIT=100

# 스레드 제한 (스레드 폭발 방지)
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
MKL_NUM_THREADS=4

# macOS 호환성
PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 모니터링

```bash
# API 상태
curl http://localhost:8000/health

# GPU 풀 상태
curl http://localhost:8000/gpu-status
```

**GPU 상태 응답 예시:**

```json
{
  "total_gpus": 4,
  "available_gpus": 3,
  "pipelines_initialized": 4,
  "gpus": {
    "0": {"available": true, "task_id": null, "has_pipeline": true},
    "1": {"available": false, "task_id": "processing", "has_pipeline": true}
  }
}
```

### 트러블슈팅

| 문제 | 원인 | 해결 방법 |
|------|------|----------|
| spconv float64 오류 | torch import 전 dtype 미설정 | `torch.set_default_dtype(torch.float32)` 확인 |
| Intrinsics recovery failure | MoGe pointmap 실패 | synthetic pinhole pointmap 사용 (기본값) |
| GLB export AttributeError | mesh_data가 list | PLY fallback 사용 (자동 처리) |
| CUDA out of memory | GPU 메모리 부족 | 워커 수 줄이기, 이미지 해상도 축소 |
| Empty mask error | 세그멘테이션 실패 | 마스크 >100 픽셀 확인 |
| Pipeline not initialized | 시작 시 초기화 실패 | on-demand 생성 fallback (자동) |
| Subprocess timeout | 3D 생성 10분 초과 | GPU 성능 확인, 마스크 크기 확인 |

### 롤백 절차

```bash
# 서버 중지
pkill -f "uvicorn api:app"

# 이전 버전으로 롤백
git checkout HEAD~1

# 의존성 재설치 (필요시)
pip install -r requirements.txt

# 서버 재시작
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 에셋 정리

```bash
# 7일 이상 된 에셋 삭제
find assets/ -type f -mtime +7 -delete
```

---

## Docker 배포

### 사전 요구사항

```bash
# VM 설정
sudo bash scripts/vm-setup-docker.sh
sudo bash scripts/vm-setup-nvidia-toolkit.sh
sudo bash scripts/vm-setup-data.sh
```

### Docker Compose 실행

```bash
export GCP_PROJECT_ID=your-project-id
export IMAGE_TAG=latest

docker compose up -d
docker compose logs -f
```

### 볼륨 마운트

| 호스트 | 컨테이너 | 설명 |
|--------|----------|------|
| `/data/sam3d/sam-3d-objects` | `/data/sam3d/sam-3d-objects` | SAM-3D 체크포인트 |
| `/data/sam3d/models` | `/data/sam3d/models` | YOLO 모델 |
| `/data/sam3d/huggingface` | `/data/sam3d/huggingface` | HuggingFace 캐시 |
| `/data/sam3d/assets` | `/app/assets` | 생성된 에셋 |

### CI/CD (GitHub Actions)

`main` 브랜치 푸시 시 자동 배포:
1. Docker 이미지 빌드
2. GCP Artifact Registry 푸시
3. VM SSH 배포

필요한 Secrets: `GCP_PROJECT_ID`, `GCP_SA_KEY`, `VM_HOST`, `VM_SSH_KEY`, `VM_USER`

---

## 요구 사항

- Python 3.10+
- CUDA 11.8+ (GPU 권장)
- 32GB+ RAM
- 50GB+ 디스크 (모델 저장)

### 주요 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| fastapi | latest | API 프레임워크 |
| uvicorn | latest | ASGI 서버 |
| torch | >=2.1.0 | PyTorch |
| ultralytics | >=8.3.0 | YOLOE-seg |
| trimesh | latest | 3D 메시 분석 |
| aiohttp | latest | 비동기 HTTP |

---

## 문서

- [기술 설계 문서 (TDD)](docs/tdd/TDD_PIPELINE_V2.md) - 아키텍처, API 스펙
- [CLAUDE.md](CLAUDE.md) - Claude Code 가이드, 코드 수정 지침

---

## 라이선스

MIT
