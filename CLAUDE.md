# CLAUDE.md

Claude Code (claude.ai/code)가 이 레포지토리에서 작업할 때 참고하는 가이드입니다.

## 개요

FastAPI 기반 서비스로 다음을 통합합니다:
1. **SAM 2 (Segment Anything Model 2)** - Meta의 Hugging Face 이미지 세그멘테이션 모델
2. **Sam-3d-objects** - Facebook Research의 2D 이미지에서 3D 객체 생성 파이프라인
3. **AI 모듈** - 가구 탐지 및 분석 시스템 (한국어 지원)

API는 포인트 기반 세그멘테이션으로 2D 이미지를 받아 3D Gaussian Splat, PLY, GIF, GLB 메시를 생성합니다.

## 아키텍처

### 프로세스 격리 패턴

GPU/spconv 상태 충돌을 방지하기 위해 **subprocess 격리**를 사용합니다:
- `api.py` - SAM 2 세그멘테이션을 처리하는 메인 FastAPI 서버
- `ai/subprocess/generate_3d_worker.py` - Sam-3d-objects 3D 생성을 위한 격리된 subprocess

spconv는 지속적인 GPU 상태를 가지므로 같은 프로세스에서 모델을 로드하면 충돌이 발생하기 때문에 이 격리는 **필수적**입니다.

### 핵심 컴포넌트

1. **메인 API (`api.py`)**
   - 시작 시 SAM 2 모델을 로드하는 FastAPI 서버
   - 세그멘테이션 요청 처리 (/segment, /segment-binary)
   - 백그라운드 워커를 통한 비동기 3D 생성 작업 관리
   - /assets 엔드포인트에서 정적 에셋(PLY, GIF, GLB) 제공

2. **3D 생성 Subprocess (`ai/subprocess/generate_3d_worker.py`)**
   - 각 3D 생성 작업을 위한 새로운 Python 프로세스
   - Sam-3d-objects 파이프라인을 독립적으로 로드
   - Gaussian splat 생성, 회전 GIF 렌더링, GLB 메시 내보내기
   - intrinsics 복구 실패를 방지하기 위해 합성 핀홀 포인트맵 사용
   - 구형 조화(spherical harmonics)에서 RGB 색상을 추가하여 PLY 파일 후처리

3. **AI 모듈 (`ai/`)**
   - YOLO-World + SAHI를 사용한 가구 탐지 (작은 객체 탐지)
   - CLIP을 사용한 세부 가구 서브타입 분류
   - 이사 서비스를 위한 한국어 인터페이스
   - 부피 계산을 위한 가구 치수 Knowledge Base
   - 통합 파이프라인: Firebase URL → YOLO → CLIP → SAM2 → SAM-3D → 부피

### 고정 경로 의존성

Sam-3d-objects 통합은 **고정된 상대 경로**를 사용합니다 (환경 변수 아님):
- `./sam-3d-objects/notebook` - inference import에 필요
- `./sam-3d-objects/checkpoints/hf/pipeline.yaml` - 파이프라인 설정

이 경로들은 `api.py`와 `ai/subprocess/generate_3d_worker.py` 모두에 하드코딩되어 있습니다.

### 환경 변수 설정

**중요**: GPU 튜닝 문제를 방지하기 위해 여러 환경 변수가 **torch/spconv import 전에 설정되어야** 합니다:
- `SPCONV_TUNE_DEVICE=0`
- `SPCONV_ALGO_TIME_LIMIT=100` (무한 튜닝 방지)
- `OMP_NUM_THREADS=4` (스레드 폭발 방지)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` (macOS 호환성)

이들은 `api.py`와 `ai/subprocess/generate_3d_worker.py` 상단에서 import 전에 설정됩니다.

### 작업 저장 패턴

3D 생성은 비동기 작업을 추적하기 위해 인메모리 dict (`generation_tasks`)를 사용합니다:
- POST /generate-3d는 즉시 task_id를 반환 (게이트웨이 타임아웃 방지)
- 클라이언트가 GET /generate-3d-status/{task_id}로 결과 폴링
- 작업은 상태(queued/processing/completed/failed), 출력 파일, 메타데이터 저장

## 자주 사용하는 명령어

### 설정

```bash
# Hugging Face CLI 설치 및 인증
pip install 'huggingface-hub[cli]<1.0'
huggingface-cli login

# setup 스크립트 실행 (sam-3d-objects 클론, conda 환경 생성, 체크포인트 다운로드)
source setup.sh
```

### 개발

```bash
# 자동 리로드로 실행 (개발)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# 리로드 없이 실행
python api.py
```

### 프로덕션

```bash
# 여러 워커로 Uvicorn 사용
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info

# Gunicorn + Uvicorn 워커 클래스 사용
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 api:app --log-level info
```

**프로덕션 참고:**
- 사용 가능한 CPU/메모리에 따라 `--workers` 조정
- GPU 워크로드의 경우 같은 GPU를 두고 경쟁하는 워커가 너무 많지 않도록 주의
- 프로덕션에서는 프로세스 매니저(systemd, docker-compose)와 리버스 프록시(NGINX) 사용
- 시작 전 Sam-3d-objects 경로가 존재하는지 확인

### AI 모듈 (가구 탐지)

AI 모듈은 메인 API와 통합되어 있습니다:

```bash
# 독립 실행 (테스트용)
cd ai
python main.py  # imgs/ 디렉토리의 이미지 분석
```

## API 엔드포인트

### 핵심 엔드포인트
- `GET /health` - 모델 상태와 함께 헬스 체크
- `POST /segment` - 단일 포인트 세그멘테이션 (여러 마스크 반환)
- `POST /segment-binary` - 다중 포인트 세그멘테이션 (마스크된 PNG 반환)
- `POST /generate-3d` - 비동기 3D 생성 (task_id 반환)
- `GET /generate-3d-status/{task_id}` - 3D 생성 결과 폴링
- `GET /assets-list` - 저장된 PLY/GIF/GLB 파일 목록 조회
- `GET /assets/{filename}` - 정적 에셋 다운로드

### 가구 분석 엔드포인트 (AI 통합)
- `POST /analyze-furniture` - 전체 파이프라인: Firebase URLs → 탐지 → 3D → 부피
- `POST /analyze-furniture-single` - 단일 이미지 분석
- `POST /analyze-furniture-base64` - Base64 이미지 입력 (Firebase URL 없음)
- `POST /detect-furniture` - 탐지만 수행 (3D 없음, 빠른 응답)

모든 엔드포인트는 base64 인코딩된 이미지 또는 Firebase Storage URL을 사용합니다.

## 3D 생성 파이프라인

1. 클라이언트가 이미지 + 마스크를 POST /generate-3d로 전송
2. API가 task_id를 생성하고 백그라운드 워커 시작
3. 백그라운드 워커가 새 프로세스에서 generate_3d_worker.py 실행
4. Subprocess:
   - 고정된 설정 경로로 Sam-3d-objects 파이프라인 로드
   - 합성 핀홀 포인트맵 생성 (MoGe intrinsics 실패 방지)
   - decode_formats=["gaussian", "glb", "mesh"]로 파이프라인 실행
   - render_video()를 사용하여 360° 회전 GIF 렌더링
   - SH 계수에서 RGB 색상을 추가하여 PLY 내보내기
   - to_glb() 또는 네이티브 파이프라인 출력으로 텍스처 GLB 내보내기 시도
   - assets/ 폴더에 메타데이터와 함께 파일 저장
   - stdout 마커로 URL 반환 (GIF_DATA_START/END, MESH_URL_START/END, PLY_URL_START/END)
5. API가 subprocess stdout에서 URL 추출하고 작업 상태 업데이트
6. 클라이언트가 /generate-3d-status/{task_id}를 폴링하여 base64 인코딩된 파일 수신

## 가구 분석 파이프라인 (AI 통합)

`/analyze-furniture` 엔드포인트는 전체 AI Logic 파이프라인을 구현합니다:

1. **이미지 가져오기**: Firebase Storage URL에서 이미지 다운로드 (5-10장)
2. **객체 탐지**: YOLO-World + SAHI로 작은 객체 탐지
3. **분류**: CLIP으로 세부 서브타입 분류 (예: 퀸 침대 vs 킹 침대)
4. **이동 가능 여부 확인**: knowledge base와 비교하여 is_movable 결정
5. **마스크 생성**: SAM2가 중심점 프롬프트로 세그멘테이션 마스크 생성
6. **3D 생성**: SAM-3D가 마스크된 이미지를 3D 모델로 변환
7. **부피 계산**: trimesh가 상대적 치수를 위해 메시 분석
8. **절대 치수**: DB 가구 사양과 매칭하여 실제 측정값 산출

### 응답 형식
```json
{
  "objects": [
    {
      "label": "퀸 사이즈 침대",
      "width": 1500.0,
      "depth": 2000.0,
      "height": 450.0,
      "volume": 1.35,
      "ratio": {"w": 0.75, "h": 1.0, "d": 0.225},
      "is_movable": true
    }
  ],
  "summary": {
    "total_objects": 10,
    "movable_objects": 8,
    "total_volume_liters": 15.5,
    "movable_volume_liters": 12.3
  }
}
```

### 핵심 컴포넌트
- `ai/pipeline/furniture_pipeline.py` - 메인 파이프라인 오케스트레이터
- `ai/processors/2_Yolo-World_detect.py` - SAHI 강화 YOLO 탐지기
- `ai/processors/7_volume_calculate.py` - 메시 부피 분석
- `ai/data/knowledge_base.py` - 가구 치수 데이터베이스

## 코드 수정 가이드라인

### SAM 2 통합 수정 시
- api.py:segment_image와 api.py:segment_binary의 기존 마스크 처리 확인
- 마스크 품질을 위한 morphological smoothing (cv2.morphologyEx) 유지
- SAM 2의 4D 입력 형식 유지: [[[[x, y]]]]

### 3D 생성 수정 시
- api.py 메인 프로세스에서 Sam-3d-objects를 **절대** 로드하지 말 것 - 항상 subprocess 사용
- generate_3d_worker.py 상단의 환경 변수 유지
- MoGe/dummy 맵 대신 합성 핀홀 포인트맵 (make_synthetic_pointmap) 사용
- subprocess stdout에서 디버그 마커 확인 (PLY_URL_START/END 등)
- GLB 내보내기 신중하게 테스트 - to_glb()는 메시 데이터가 필요하며 AttributeError가 발생할 수 있음

### 새 엔드포인트 추가 시
- 5초 이상 걸리는 작업에는 background_tasks 사용
- 비동기 작업은 즉시 task_id 반환
- 결과를 generation_tasks dict 또는 영구 저장소에 저장

### PLY 파일 작업 시
- PLY 파일은 SH 계수에서 RGB를 추가하여 후처리됨 (add_rgb_to_ply)
- 호환성을 위해 ASCII 형식 사용 (파일이 클 수 있음)
- 클라이언트는 결과 폴링 시 대용량 base64 페이로드 처리 필요

### 3D 생성 디버깅 시
- subprocess stdout/stderr 로그 확인 (_generate_3d_background에서 출력)
- Sam-3d-objects 경로 존재 확인: ./sam-3d-objects/notebook 및 checkpoints
- 메모리 문제 확인 - 피크 GPU 메모리가 로그됨
- 마스크가 충분한 픽셀을 가지는지 검증 (100 이상 권장, subprocess 로그에 출력)

## 파일 구조

```
api.py                                  # 메인 FastAPI 서버 (SAM 2 + 작업 관리 + AI 통합)
requirements.txt                        # Python 의존성
setup.sh                                # 설정 스크립트 (sam-3d-objects 클론, conda 환경 생성)
assets/                                 # /assets/에서 제공되는 정적 파일 (PLY, GIF, GLB)
sam-3d-objects/                         # 클론된 Facebook Research 레포 (git에 없음)
  notebook/inference.py                 # Sam-3d-objects 파이프라인 클래스
  checkpoints/hf/pipeline.yaml          # 파이프라인 설정
ai/                                     # AI 모듈 (메인 API와 통합)
  __init__.py                           # 모듈 진입점
  main.py                               # 독립 CLI 진입점
  config.py                             # YOLO/CLIP 모델 설정
  processors/                           # AI Logic 단계별 프로세서
    1_firebase_images_fetch.py          # Step 1: Firebase에서 이미지 가져오기
    2_Yolo-World_detect.py              # Step 2: YOLO-World 객체 탐지
    3_CLIP_classify.py                  # Step 3: CLIP 세부 분류
    4_DB_movability_check.py            # Step 4: is_movable 판단
    5_SAM2_mask_generate.py             # Step 5: SAM2 마스크 생성
    6_SAM3D_convert.py                  # Step 6: SAM-3D 3D 변환
    7_volume_calculate.py               # Step 7: 부피/치수 계산
  pipeline/
    furniture_pipeline.py               # 파이프라인 오케스트레이터
  subprocess/
    generate_3d_worker.py               # 격리된 3D 생성 워커
  data/
    knowledge_base.py                   # 치수가 있는 가구 데이터베이스
  utils/
    image_ops.py                        # 이미지 처리 유틸리티
```

## 알려진 문제 및 해결 방법

1. **spconv float64 오류**: 모든 import 전에 torch.set_default_dtype(torch.float32) 설정으로 방지
2. **Intrinsics 복구 실패**: MoGe 대신 합성 핀홀 포인트맵 사용으로 방지
3. **GLB 내보내기 AttributeError**: mesh_data가 비메시 객체 리스트일 때 발생 - subprocess가 이제 이를 캐치하고 로깅
4. **메시 생성 CPU 스파이크**: Gaussian-to-mesh 변환은 기본적으로 비활성화 (CPU 집약적)
5. **빈 마스크 오류**: Subprocess가 마스크가 0픽셀 이상인지 검증하고 100픽셀 미만이면 경고

## 통합 참고

이 프로젝트는 [Sam3D Mobile](https://github.com/andrisgauracs/sam3d-mobile) 앱과 연동하도록 설계되었습니다.
