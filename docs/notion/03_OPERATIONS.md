# Isajjim AI - 운영 가이드

## 1. 환경 설정

### 1.1 사전 요구사항

| 항목 | 버전 | 비고 |
|------|------|------|
| Python | 3.10+ | conda 환경 권장 |
| CUDA | 11.8+ | GPU 필수 |
| VRAM | 32GB+ | GPU당 (YOLOE + SAM-3D) |
| 디스크 | 50GB+ | 모델 + 체크포인트 |

### 1.2 초기 설치

```bash
# 1. HuggingFace CLI 설치 및 인증 (모델 다운로드용)
pip install 'huggingface-hub[cli]<1.0'
huggingface-cli login

# 2. 환경 설정 스크립트 (sam-3d-objects 클론, conda env, 체크포인트)
source setup.sh

# 3. 의존성 설치
pip install -r requirements.txt
```

### 1.3 필수 경로

SAM-3D는 다음 경로를 하드코딩으로 참조합니다:

| 경로 | 설명 |
|------|------|
| `./sam-3d-objects/notebook/` | inference.py (SAM-3D 파이프라인) |
| `./sam-3d-objects/checkpoints/hf/pipeline.yaml` | 파이프라인 설정 |
| `./assets/` | 생성된 PLY/GIF/GLB 저장 |

---

## 2. 서버 실행

### 2.1 개발 환경

```bash
# 자동 리로드 (코드 변경 시 재시작)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

### 2.2 프로덕션 환경

```bash
# Uvicorn 직접 실행 (workers=1 권장 - GPU 모델 공유 문제)
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info

# 또는 Gunicorn + Uvicorn
gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8000 api:app --log-level info
```

> **주의**: workers를 2 이상으로 설정하면 각 워커가 독립적으로 GPU 모델을 로드하여 VRAM이 부족할 수 있습니다. GPU 병렬 처리는 GPUPoolManager가 담당하므로 worker=1로 충분합니다.

### 2.3 서버 시작 시 자동 초기화

서버가 시작되면 `startup_event`에서 다음이 자동 실행됩니다:

1. **GPUPoolManager 초기화**: 가용 GPU 자동 감지, 라운드로빈 풀 생성
2. **YOLOE 파이프라인 사전 로드**: 각 GPU에 FurniturePipeline 1개씩 로드
3. **SAM3DWorkerPool 초기화**: 각 GPU에 Persistent Worker 프로세스 1개씩 시작
   - 각 워커가 SAM-3D 모델을 로드하고 대기 상태 진입
   - 초기화 타임아웃: 120초

**정상 시작 로그:**

```
Using device: cuda
GPU pool initialized with 8 GPUs: [0, 1, 2, 3, 4, 5, 6, 7]
Furniture pipelines pre-initialized for 8 GPUs
Initializing SAM3D Worker Pool with 8 GPUs...
SAM3D Worker Pool ready: {workers: 8, ready: 8}
```

---

## 3. Docker 배포

### 3.1 VM 사전 설정

```bash
sudo bash scripts/vm-setup-docker.sh         # Docker 설치
sudo bash scripts/vm-setup-nvidia-toolkit.sh  # NVIDIA Container Toolkit
sudo bash scripts/vm-setup-data.sh            # 데이터 디렉토리 설정
```

### 3.2 Docker Compose 실행

```bash
export GCP_PROJECT_ID=your-project-id
export IMAGE_TAG=latest

docker compose up -d
docker compose logs -f
```

### 3.3 볼륨 마운트

| 호스트 | 컨테이너 | 설명 |
|--------|----------|------|
| `/data/sam3d/sam-3d-objects` | `/data/sam3d/sam-3d-objects` | SAM-3D 체크포인트 |
| `/data/sam3d/models` | `/data/sam3d/models` | YOLO 모델 |
| `/data/sam3d/huggingface` | `/data/sam3d/huggingface` | HuggingFace 캐시 |
| `/data/sam3d/assets` | `/app/assets` | 생성된 에셋 |

### 3.4 CI/CD (GitHub Actions)

`main` 브랜치 푸시 시 자동 배포:

1. Docker 이미지 빌드
2. GCP Artifact Registry 푸시
3. VM SSH 배포

필요한 GitHub Secrets: `GCP_PROJECT_ID`, `GCP_SA_KEY`, `VM_HOST`, `VM_SSH_KEY`, `VM_USER`

---

## 4. 환경 변수

`api/config.py`와 `ai/subprocess/persistent_3d_worker.py`에서 **torch import 전** 자동 설정:

### 4.1 GPU/spconv 설정

| 변수 | 값 | 목적 |
|------|-----|------|
| `CUDA_HOME` | `/usr/local/cuda` | CUDA 경로 |
| `SPCONV_TUNE_DEVICE` | `0` | spconv 튜닝 디바이스 |
| `SPCONV_ALGO_TIME_LIMIT` | `100` | spconv 튜닝 시간 제한 (ms) |
| `CUDA_VISIBLE_DEVICES` | GPU별 설정 | Worker 프로세스 GPU 격리 |

### 4.2 스레드 제한

| 변수 | 값 | 목적 |
|------|-----|------|
| `OMP_NUM_THREADS` | `4` | OpenMP 스레드 수 |
| `OPENBLAS_NUM_THREADS` | `4` | OpenBLAS 스레드 수 |
| `MKL_NUM_THREADS` | `4` | Intel MKL 스레드 수 |
| `VECLIB_MAXIMUM_THREADS` | `4` | macOS Accelerate |
| `NUMEXPR_NUM_THREADS` | `4` | NumExpr 스레드 수 |

> 기본값을 사용하면 각 라이브러리가 CPU 코어 수만큼 스레드를 생성하여 "스레드 폭발" 발생

### 4.3 Callback 설정

| 변수 | 값 | 위치 |
|------|-----|------|
| `CALLBACK_URL_TEMPLATE` | `https://api.isajjim.kro.kr/api/v1/estimates/{estimateId}/callback` | `api/config.py` |
| `CALLBACK_TIMEOUT_SECONDS` | `30` | `api/config.py` |
| `CALLBACK_RETRY_COUNT` | `1` | `api/config.py` |

---

## 5. 모니터링

### 5.1 상태 확인 명령

```bash
# API 서버 상태
curl http://localhost:8000/health

# GPU 풀 상태 (파이프라인 초기화, GPU 사용률)
curl http://localhost:8000/gpu-status

# 에셋 목록 (생성된 PLY/GIF/GLB)
curl http://localhost:8000/assets-list
```

### 5.2 GPU 상태 해석

```json
{
  "total_gpus": 4,
  "available_gpus": 3,        // 현재 사용 가능한 GPU 수
  "pipelines_initialized": 4, // YOLOE 파이프라인 로드된 GPU 수
  "gpus": {
    "0": {
      "available": true,       // 현재 사용 가능 여부
      "task_id": null,         // 현재 처리 중인 작업 (null = 유휴)
      "memory_used_mb": 1024,  // GPU 메모리 사용량
      "has_pipeline": true     // YOLOE 파이프라인 로드 여부
    }
  }
}
```

- `available_gpus = 0`: 모든 GPU가 작업 중 → 새 요청은 대기 (timeout 300초)
- `has_pipeline: false`: 시작 시 초기화 실패 → on-demand 생성 (느림)
- `error_count > 0`: GPU 에러 발생 이력 확인 필요

### 5.3 로그 확인 포인트

| 로그 패턴 | 의미 |
|----------|------|
| `[GPUPoolManager]` | GPU 풀 acquire/release |
| `[SAM3DWorkerPool]` | Worker 시작/종료/재시작 |
| `[FurniturePipeline]` | 파이프라인 처리 상태 |
| `[Callback]` | Callback 전송 성공/실패 |
| `[Background]` | 백그라운드 작업 시작/완료 |

---

## 6. 성능 튜닝

### 6.1 SAM-3D 추론 설정

`ai/subprocess/persistent_3d_worker.py` 상단:

```python
MAX_IMAGE_SIZE = None           # None = 다운샘플링 비활성화 (정확도 우선)
STAGE1_INFERENCE_STEPS = 14     # 12~16 사이 (낮을수록 빠르지만 부정확)
STAGE2_INFERENCE_STEPS = 4      # 4~12 사이 (4도 치수 오차 0.5% 이내)
USE_BINARY_PLY = True           # Binary PLY (70% 작음, 50% 빠름)
GAUSSIAN_ONLY_MODE = True       # GLB/Mesh 스킵 (37.4% 빠름)
ENABLE_COMPILE = True           # torch.compile (10-20% 빠름, 초기화 느림)
```

### 6.2 설정별 영향도

| 설정 변경 | 속도 영향 | 정확도 영향 | 권장 |
|----------|----------|-----------|------|
| STAGE1: 14 → 12 | +15% 빠름 | 부피 오차 +11% | 비권장 |
| STAGE1: 14 → 16 | -10% 느림 | 부피 오차 -0.5% | 정확도 중요 시 |
| STAGE2: 4 → 8 | -15% 느림 | 치수 오차 ±0.3% | 불필요 |
| GAUSSIAN_ONLY: False | -37% 느림 | 동일 | GLB 필요 시만 |
| MAX_IMAGE_SIZE: 512 | +30% 빠름 | 부피 오차 +91% | 비권장 |

### 6.3 YOLOE 탐지 설정

`ai/config.py`:

| 설정 | 현재 값 | 설명 |
|------|---------|------|
| `CONF_THRESHOLD_MAIN` | 0.10 | 메인 탐지 임계값 |
| `CONF_THRESHOLD_SMALL` | 0.05 | 작은 객체 임계값 |
| `USE_CLAHE_ENHANCEMENT` | True | 저조도 이미지 대비 향상 |

---

## 7. 트러블슈팅

### 7.1 서버 시작 실패

| 증상 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError: spconv` | spconv 미설치 | `pip install spconv-cu118` (CUDA 버전에 맞게) |
| `RuntimeError: CUDA out of memory` | 시작 시 VRAM 부족 | GPU 수 줄이기 (`ai/config.py`의 `GPU_IDS`) |
| `Pipeline pre-initialization failed` | YOLOE 모델 없음 | `yoloe-26x-seg.pt` 모델 파일 확인 |
| `SAM3D Worker Pool initialization failed` | sam-3d-objects 경로 | `./sam-3d-objects/` 디렉토리 존재 확인 |

### 7.2 처리 중 오류

| 증상 | 원인 | 해결 |
|------|------|------|
| `spconv float64 error` | dtype 설정 누락 | `torch.set_default_dtype(torch.float32)` 확인 |
| `Intrinsics recovery failure` | MoGe pointmap 실패 | synthetic pinhole pointmap 사용 (기본값) |
| `Empty mask error` | 세그멘테이션 마스크 비어있음 | 마스크 >100 픽셀인지 확인 |
| `Worker not ready` | 워커 초기화 실패/타임아웃 | 로그 확인, 워커 자동 재시작 대기 |
| `Subprocess timeout` (300초) | 3D 생성 과도한 시간 | GPU 성능, 마스크 크기 확인 |
| `Callback failed` | 백엔드 서버 응답 없음 | Callback URL 접근성 확인 |

### 7.3 정확도 문제

| 증상 | 원인 | 해결 |
|------|------|------|
| 부피가 크게 다름 | 이미지 다운샘플링 활성화됨 | `MAX_IMAGE_SIZE = None` 확인 |
| width/depth 뒤바뀜 | OBB 좌표계 매핑 이슈 | Greedy 매핑 로직 확인 |
| 미탐지 객체 | confidence 임계값 높음 | `CONF_THRESHOLD_MAIN` 낮추기 |
| 오탐지 과다 | confidence 임계값 낮음 | `CONF_THRESHOLD_MAIN` 올리기 |

---

## 8. 운영 작업

### 8.1 에셋 정리

```bash
# 7일 이상 된 에셋 삭제
find assets/ -type f -mtime +7 -delete
```

### 8.2 서버 재시작

```bash
# 서버 중지
pkill -f "uvicorn api:app"

# 서버 시작
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```

### 8.3 롤백

```bash
pkill -f "uvicorn api:app"
git checkout HEAD~1
pip install -r requirements.txt  # 의존성 변경된 경우
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

### 8.4 Knowledge Base 업데이트

`ai/data/knowledge_base.py`의 `FURNITURE_DB`를 수정:

1. 새 카테고리 추가: DB key, base_name, synonyms 설정
2. 동의어 추가: 기존 카테고리의 `synonyms` 리스트에 추가
3. 서브타입 추가: `subtypes` 딕셔너리에 추가
4. 서버 재시작 필요 (모델 재로드는 불필요)

---

## 9. 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `fastapi` | API 프레임워크 |
| `uvicorn[standard]` | ASGI 서버 |
| `pydantic` | 요청/응답 검증 |
| `torch>=2.1.0` | PyTorch |
| `torchvision>=0.16.0` | 이미지 처리 |
| `ultralytics>=8.3.0` | YOLOE-seg |
| `trimesh` | 3D 메시 분석 (OBB 치수) |
| `Pillow>=10.0.0` | 이미지 처리 |
| `numpy>=1.24.0` | 수치 연산 |
| `opencv-python-headless>=4.9.0` | 이미지 전처리 (CLAHE) |
| `aiohttp` | 비동기 HTTP (Firebase, Callback) |
| `requests` | 동기 HTTP (fallback) |
| `omegaconf>=2.3.0` | SAM-3D 설정 파일 |
| `hydra-core>=1.3.2` | SAM-3D 설정 관리 |
