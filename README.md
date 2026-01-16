# YOLOE-seg + SAM-3D 생성 API (V2)

이사 서비스를 위한 가구 탐지 및 3D 모델 생성 통합 API 서비스입니다.

> **V2 (2024-01)**: SAM2 제거, YOLOE-seg 마스크 직접 사용으로 파이프라인 단순화

---

## 주요 기능

- **YOLOE-seg 탐지**: Objects365 기반 365개 클래스 탐지 + 인스턴스 세그멘테이션 마스크
- **SAM-3D 3D 생성**: 2D 이미지 + YOLO 마스크에서 3D Gaussian Splat, PLY, GLB 메시 생성
- **부피 계산**: 3D 모델 기반 가구 부피 및 치수 자동 계산
- **이동 가능 여부 판단**: Knowledge Base 대조를 통한 is_movable 결정
- **Multi-GPU 병렬 처리**: 라운드 로빈 GPU 할당을 통한 병렬 이미지 처리
- **SAM 2 (deprecated)**: `/segment` 엔드포인트에서 사용 가능 (V2 파이프라인에서는 미사용)

---

## 디렉토리 구조

```
sam3d-api/
├── api.py                      # FastAPI 메인 서버 (오케스트레이션)
├── ai/                         # AI 모듈
│   ├── processors/             # AI Logic 단계별 프로세서
│   │   ├── 1_firebase_images_fetch.py   # Step 1: 이미지 다운로드
│   │   ├── 2_YOLO_detect.py             # Step 2: YOLOE-seg 객체 탐지
│   │   ├── 4_DB_movability_check.py     # Step 3: is_movable 결정
│   │   ├── 5_SAM2_mask_generate.py      # Step 4: SAM2 마스크 생성
│   │   ├── 6_SAM3D_convert.py           # Step 5: SAM-3D 3D 변환
│   │   └── 7_volume_calculate.py        # Step 6: 부피/치수 계산
│   │
│   ├── pipeline/
│   │   └── furniture_pipeline.py        # 파이프라인 오케스트레이터
│   │
│   ├── subprocess/
│   │   └── generate_3d_worker.py        # SAM-3D subprocess (GPU 격리)
│   │
│   ├── gpu/
│   │   └── gpu_pool_manager.py          # Multi-GPU 풀 매니저
│   │
│   ├── data/
│   │   └── knowledge_base.py            # 가구 Knowledge Base (DB)
│   │
│   ├── utils/
│   │   └── image_ops.py                 # 이미지 유틸리티
│   │
│   └── config.py                        # 설정
│
├── sam-3d-objects/             # Facebook Research SAM-3D 레포 (setup.sh를 통해 외부에서 클론)
├── assets/                     # 생성된 PLY/GIF/GLB 에셋 저장
├── requirements.txt            # Python 의존성
└── setup.sh                    # 환경 설정 스크립트
```

---

## AI Logic 파이프라인 V2

전체 가구 분석 파이프라인은 5단계로 구성됩니다 (SAM2 제거로 단순화):

```
[V2 Pipeline]
Firebase URL → YOLOE-seg (마스크 포함) → DB 매칭 → SAM-3D → 부피 계산
```

| 단계 | 파일 | 설명 |
|------|------|------|
| 1 | `1_firebase_images_fetch.py` | Firebase Storage URL에서 이미지 다운로드 |
| 2 | `2_YOLO_detect.py` | YOLOE-seg로 객체 탐지 (bbox + class + **mask 직접 출력**) |
| 3 | `4_DB_movability_check.py` | YOLO 클래스로 직접 DB 매칭, is_movable 결정 |
| 4 | `6_SAM3D_convert.py` | YOLO 마스크 → SAM-3D로 3D 모델 생성 |
| 5 | `7_volume_calculate.py` | trimesh로 부피/치수 계산 및 DB 규격 대조 |

> **Note**: `5_SAM2_mask_generate.py`는 V2에서 deprecated 되었습니다. YOLOE-seg 마스크를 직접 사용합니다.

### 변경 사항 (V1 → V2)

| 항목 | V1 | V2 |
|------|------|------|
| 탐지 모델 | yolov8l-world.pt | yoloe-11x-seg.pt |
| SAHI | 사용 | **제거** |
| CLIP 분류 | 세부 유형 분류 | **제거** |
| **SAM2 마스크** | center point prompt | **제거 (YOLO 마스크 직접 사용)** |
| 분류 단계 | YOLO → CLIP → DB | YOLO → DB (직접) |
| API 호출 수 | 3회 | 2회 |

### V2 변경 이유

- **마스크 품질**: YOLOE-seg가 SAM2보다 객체 전체를 더 정확하게 커버 (테스트 결과)
- **속도**: SAM2 API 호출 제거로 latency 감소
- **단순화**: HTTP 호출 제거, 코드 복잡도 감소

---

## API 엔드포인트

### 기본 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 확인 |
| POST | `/segment` | 단일 포인트 세그멘테이션 |
| POST | `/segment-binary` | 다중 포인트 세그멘테이션 (마스크된 이미지 반환) |
| POST | `/generate-3d` | 비동기 3D 생성 (task_id 반환) |
| GET | `/generate-3d-status/{task_id}` | 3D 생성 결과 폴링 |
| GET | `/assets-list` | 저장된 에셋 목록 |

### 가구 분석 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/analyze-furniture` | 다중 이미지 가구 분석 (Firebase URL) |
| POST | `/analyze-furniture-single` | 단일 이미지 가구 분석 |
| POST | `/analyze-furniture-base64` | Base64 이미지 가구 분석 |
| POST | `/detect-furniture` | 가구 탐지만 수행 (3D 생성 없음, 빠른 응답) |

---

## 설치 방법

### 1. Hugging Face CLI 설치 및 인증

```bash
pip install 'huggingface-hub[cli]<1.0'
huggingface-cli login
```

### 2. 환경 설정 스크립트 실행

```bash
source setup.sh
```

이 스크립트는:
- `sam-3d-objects` 레포지토리 클론
- Conda 환경 생성 및 활성화
- 의존성 설치
- 체크포인트 다운로드

### 3. 필수 경로 확인

다음 경로가 존재해야 합니다:
- `./sam-3d-objects/notebook`
- `./sam-3d-objects/checkpoints/hf/pipeline.yaml`

---

## 실행 방법

### 개발 환경 (자동 리로드)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

### 기본 실행

```bash
python api.py
# 또는
uvicorn api:app --host 0.0.0.0 --port 8000 --log-level info
```

### 프로덕션 환경 (로컬)

Gunicorn + Uvicorn 워커:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 api:app --log-level info
```

또는 Uvicorn 멀티 워커:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```

**주의사항:**
- GPU 작업이 많은 경우 워커 수를 적절히 조절하세요
- 프로덕션에서는 NGINX 리버스 프록시 사용을 권장합니다

API 문서 확인: http://localhost:8000/docs

---

## Docker 배포

### 사전 요구사항

1. Docker 및 NVIDIA Container Toolkit 설치
2. 데이터 디렉토리 설정 (`/data/sam3d/`)

```bash
# VM 설정 스크립트 실행
sudo bash scripts/vm-setup-docker.sh
sudo bash scripts/vm-setup-nvidia-toolkit.sh
sudo bash scripts/vm-setup-data.sh
```

### Docker Compose로 실행

```bash
# 환경 변수 설정
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1
export IMAGE_TAG=latest

# 컨테이너 시작
docker compose up -d

# 로그 확인
docker compose logs -f

# 중지
docker compose down
```

### 볼륨 마운트

| 호스트 경로 | 컨테이너 경로 | 설명 |
|------------|--------------|------|
| `/data/sam3d/sam-3d-objects` | `/data/sam3d/sam-3d-objects` | SAM-3D 체크포인트 (읽기 전용) |
| `/data/sam3d/models` | `/data/sam3d/models` | YOLO 모델 (읽기 전용) |
| `/data/sam3d/huggingface` | `/data/sam3d/huggingface` | HuggingFace 캐시 |
| `/data/sam3d/assets` | `/app/assets` | 생성된 에셋 |

### CI/CD (GitHub Actions)

`main` 브랜치에 푸시하면 자동으로:
1. Docker 이미지 빌드
2. GCP Artifact Registry에 푸시
3. VM에 SSH로 배포

필요한 GitHub Secrets:
- `GCP_PROJECT_ID`: GCP 프로젝트 ID
- `GCP_SA_KEY`: 서비스 계정 JSON 키
- `VM_HOST`: VM 외부 IP
- `VM_SSH_KEY`: SSH 개인 키
- `VM_USER`: VM 사용자 이름

---

## 사용 예시

### 상태 확인

```bash
curl http://localhost:8000/health
```

### 단일 포인트 세그멘테이션

```bash
curl -X POST http://localhost:8000/segment \
  -H 'Content-Type: application/json' \
  -d '{"image":"<BASE64_IMAGE>","x":200,"y":150}'
```

### 다중 포인트 세그멘테이션

```bash
curl -X POST http://localhost:8000/segment-binary \
  -H 'Content-Type: application/json' \
  -d '{
    "image": "<BASE64_IMAGE>",
    "points": [{"x": 200, "y": 150}, {"x": 220, "y": 170}]
  }'
```

### 3D 생성

```bash
# 1. 3D 생성 요청
curl -X POST http://localhost:8000/generate-3d \
  -H 'Content-Type: application/json' \
  -d '{"image":"<BASE64_IMAGE>","mask":"<BASE64_MASK>","seed":42}'

# 2. 결과 폴링
curl http://localhost:8000/generate-3d-status/<task_id>
```

### 가구 탐지 (빠른 응답)

```bash
curl -X POST http://localhost:8000/detect-furniture \
  -H 'Content-Type: application/json' \
  -d '{"image":"<BASE64_IMAGE>"}'
```

**응답 예시:**

```json
{
  "success": true,
  "objects": [
    {
      "id": 0,
      "label": "소파",
      "db_key": "sofa",
      "subtype": "3인용 소파",
      "bbox": [100, 200, 400, 500],
      "center_point": [250, 350],
      "is_movable": true,
      "confidence": 0.95
    }
  ],
  "total_objects": 1,
  "movable_objects": 1
}
```

### Python 클라이언트 예시

```python
import base64
import requests

# 이미지 읽기 및 인코딩
with open('input.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

# 가구 탐지 요청
resp = requests.post('http://localhost:8000/detect-furniture', json={
    'image': img_b64
})
print(resp.json())
```

---

## 환경 변수 및 주의사항

### 자동 설정되는 환경 변수

`api.py`와 `generate_3d_worker.py`에서 자동으로 설정됩니다:

- `SPCONV_TUNE_DEVICE=0`
- `SPCONV_ALGO_TIME_LIMIT=100`
- `PYTORCH_ENABLE_MPS_FALLBACK=1` (macOS)

### GPU 격리

3D 생성은 subprocess(`ai/subprocess/generate_3d_worker.py`)에서 실행됩니다.
이는 spconv/SAM-3D의 GPU 상태 충돌을 방지하기 위함입니다.

### 대용량 파일 주의

PLY 파일은 ASCII 형식으로 저장되어 용량이 클 수 있습니다.
클라이언트에서 대용량 base64 페이로드를 처리할 수 있는지 확인하세요.

---

## 트러블슈팅

| 문제 | 해결 방법 |
|------|----------|
| 모델 로드 실패 | Hugging Face CLI 인증 확인 및 `setup.sh` 재실행 |
| 메모리 부족 | GPU 메모리 확인, 워커 수 줄이기 |
| spconv float64 오류 | torch import 전 환경 변수 설정 확인 |
| 3D 생성 타임아웃 | GPU 성능 확인, 타임아웃 값 증가 |

---

## 요구 사항

- Python 3.10+
- CUDA 지원 GPU (권장)
- macOS의 경우 MPS fallback 지원

주요 의존성:
- `fastapi`, `uvicorn`
- `torch`, `torchvision`
- `transformers` (SAM2)
- `ultralytics>=8.3.0` (YOLOE-seg)
- `trimesh` (3D 메시 분석)
- `opencv-python-headless`

---

## 라이선스

MIT
