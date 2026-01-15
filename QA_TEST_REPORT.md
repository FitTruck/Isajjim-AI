# AI Logic Pipeline QA 테스트 리포트

> 테스트 일시: 2026-01-14
> 테스트 환경: NVIDIA A100-SXM4-40GB, Ubuntu Linux

---

## 목차

1. [개요](#1-개요)
2. [AI Logic 파이프라인 구조](#2-ai-logic-파이프라인-구조)
3. [테스트 환경 설정](#3-테스트-환경-설정)
4. [테스트 결과](#4-테스트-결과)
5. [API 응답 JSON 포맷](#5-api-응답-json-포맷)
6. [발견된 이슈 및 해결](#6-발견된-이슈-및-해결)
7. [성능 측정](#7-성능-측정)
8. [결론](#8-결론)

---

## 1. 개요

이 문서는 이사 서비스를 위한 가구 탐지 및 3D 변환 AI 파이프라인의 QA 테스트 결과를 정리합니다.

### 테스트 범위

| 구성요소 | 설명 | 테스트 상태 |
|----------|------|-------------|
| YOLO-World | 객체 탐지 | ✅ 완료 |
| SAHI | 소형 객체 탐지 | ⚠️ 미설치 |
| CLIP | 세부 유형 분류 | ✅ 완료 |
| Knowledge Base | 가구 DB 조회 | ✅ 완료 |
| SAM2 | 마스크 생성 | ✅ 완료 |
| SAM-3D | 3D 변환 | ✅ 완료 |
| Volume Calculator | 부피 계산 | ✅ 완료 |

---

## 2. AI Logic 파이프라인 구조

### 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Logic Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1] Firebase Storage URL (5~10장)                              │
│         │                                                        │
│         ▼                                                        │
│  [2] YOLO-World ──────────────────┐                             │
│         │                          │                             │
│         ▼                          ▼                             │
│  [3] SAHI (선택)              바운딩 박스                        │
│         │                     + 1차 클래스                       │
│         ▼                                                        │
│  [4] CLIP Classification                                         │
│         │                                                        │
│         ▼                                                        │
│  [5] Knowledge Base 조회 ──► is_movable 결정                    │
│         │                                                        │
│         ▼                                                        │
│  [6] 이동가능 객체 Crop                                          │
│         │                                                        │
│         ▼                                                        │
│  [7] SAM2 Mask 생성 (중심점 프롬프트)                           │
│         │                                                        │
│         ▼                                                        │
│  [8] SAM-3D 3D 변환                                              │
│         │                                                        │
│         ▼                                                        │
│  [9] trimesh 부피 계산 (상대 비율)                              │
│         │                                                        │
│         ▼                                                        │
│  [10] DB 규격 대조 ──► 절대 치수 계산                           │
│         │                                                        │
│         ▼                                                        │
│  [11] JSON 응답 반환                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 주요 파일 구조

```
sam3d-api/
├── api.py                      # FastAPI 메인 서버
├── generate_3d_subprocess.py   # SAM-3D 서브프로세스
├── DeCl/                       # 가구 탐지 모듈
│   ├── models/
│   │   ├── detector.py         # YOLO-World 탐지기
│   │   ├── classifier.py       # CLIP 분류기
│   │   └── sahi_detector.py    # SAHI 탐지기
│   ├── data/
│   │   └── knowledge_base.py   # 가구 DB (17종)
│   ├── services/
│   │   └── furniture_pipeline.py  # 통합 파이프라인
│   └── utils/
│       └── volume_calculator.py   # 부피 계산기
├── test_decl_pipeline.py       # DeCl 테스트 스크립트
└── test_sam_api.py             # SAM API 테스트 스크립트
```

---

## 3. 테스트 환경 설정

### 하드웨어

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA A100-SXM4-40GB |
| Platform | Linux (GCP) |

### 소프트웨어 버전

| 패키지 | 버전 | 비고 |
|--------|------|------|
| Python | 3.11 | conda 환경 |
| PyTorch | 2.5.1+cu121 | CUDA 12.1 |
| Kaolin | 0.17.0 | NVIDIA 3D 라이브러리 |
| Transformers | 4.57.3 | HuggingFace |
| OpenCV | 4.12.0 | 이미지 처리 |
| Ultralytics | - | YOLO-World |

### 중요 설정 사항

```python
# CLIP 로드 시 safetensors 사용 (torch < 2.6 보안 이슈 해결)
CLIPModel.from_pretrained(model_id, use_safetensors=True)
```

---

## 4. 테스트 결과

### 4.1 DeCl 파이프라인 테스트

#### Import 테스트
```
[OK] torch 2.5.1+cu121 (CUDA: True)
[OK] PIL (Pillow)
[OK] OpenCV 4.12.0
[OK] NumPy 1.26.4
[OK] DeCl.config (Device: cuda)
[OK] DeCl.knowledge_base (17 furniture types)
[OK] DeCl.models.detector
[OK] DeCl.models.classifier
[OK] DeCl.utils.volume_calculator
```

#### Knowledge Base 테스트

등록된 가구 유형 (17종):

| 가구 | 한글명 | 이동가능 | Subtypes |
|------|--------|----------|----------|
| air conditioner | 에어컨 | O | 3 (천장형/벽걸이/스탠드) |
| kitchen cabinet | 찬장/수납장 | X | 0 |
| bookshelf | 책장 | O | 0 |
| refrigerator | 냉장고 | O | 2 (일반/빌트인) |
| wardrobe | 장롱/수납장 | O | 3 |
| sofa | 소파 | O | 4 (1인/2인/3인/L자형) |
| bed | 침대 | O | 6 (싱글~킹/2층) |
| dining table | 식탁 | O | 3 (2인/4인/6인) |
| tv | TV | O | 5 (32~75인치) |
| desk | 책상 | O | 3 |
| chair | 의자 | O | 0 |
| washing machine | 세탁기 | O | 2 (드럼/통돌이) |
| dryer | 건조기 | O | 0 |
| box | 박스 | O | 4 variants |
| drawer | 서랍장 | O | 0 |
| mirror | 거울 | O | 0 |
| microwave | 전자레인지 | O | 0 |

#### YOLO-World 탐지 결과 (test.jpg)

```
이미지: 870x876px (거실)
탐지 시간: 0.59s
탐지 객체: 8개

1. sofa: confidence=0.943, bbox=[83,481,702,816]
2. mirror: confidence=0.667, bbox=[437,175,598,384]
3. desk: confidence=0.618, bbox=[57,542,198,658]
4. desk: confidence=0.587, bbox=[535,631,795,870]
5. wall cabinet: confidence=0.165, bbox=[106,53,218,504]
6. wall cabinet: confidence=0.155, bbox=[405,0,660,519]
7. wall cabinet: confidence=0.128, bbox=[1,96,111,583]
8. wall cabinet: confidence=0.120, bbox=[400,0,811,720]
```

#### CLIP 분류 결과

```
1. sofa → L자형 소파 (이동가능, score: 0.522)
2. desk → L자형 책상 (이동가능, score: 0.623)
3. desk → L자형 책상 (이동가능, score: 0.720)
```

#### Full Pipeline 결과 (test2.jpg)

```
이미지: 870x742px (다이닝룸)
탐지 객체: 12개
├── 이동가능: 7개
│   ├── 의자 5개
│   ├── 4인용 식탁 1개
│   └── 거울 1개
└── 고정: 5개
    └── 찬장/수납장 5개
```

### 4.2 SAM2 API 테스트

| 엔드포인트 | 상태 | 응답 시간 | Score |
|------------|------|-----------|-------|
| `/health` | ✅ | - | - |
| `/segment` | ✅ | 0.16s | 0.950 |
| `/segment-binary` | ✅ | 0.17s | 0.950 |

### 4.3 SAM-3D API 테스트

| 항목 | 결과 |
|------|------|
| 상태 | ✅ 성공 |
| 처리 시간 | 136.6s |
| 출력 파일 | GLB (1.1MB), PLY (23MB) |

---

## 5. API 응답 JSON 포맷

### 5.1 Detection Only (`/detect-furniture`)

```json
{
  "objects": [
    {
      "label": "찬장/수납장",
      "is_movable": false,
      "confidence": 0.915,
      "bbox": [161.3, 188.5, 299.5, 362.0]
    },
    {
      "label": "의자",
      "is_movable": true,
      "confidence": 0.911,
      "bbox": [565.1, 402.7, 769.7, 690.7]
    }
  ],
  "summary": {
    "total_objects": 12,
    "movable_objects": 7,
    "fixed_objects": 5
  }
}
```

### 5.2 Full Analysis (`/analyze-furniture`)

```json
{
  "objects": [
    {
      "label": "퀸 사이즈 침대",
      "width": 1500.0,
      "depth": 2000.0,
      "height": 450.0,
      "volume": 1.35,
      "ratio": {
        "w": 0.75,
        "h": 1.0,
        "d": 0.225
      },
      "is_movable": true,
      "confidence": 0.95,
      "bbox": [100, 200, 500, 600],
      "center_point": [300, 400],
      "mesh_url": "/assets/mesh_xxx.glb"
    }
  ],
  "summary": {
    "total_objects": 10,
    "movable_objects": 8,
    "fixed_objects": 2,
    "total_volume_liters": 15.5,
    "movable_volume_liters": 12.3,
    "images_processed": 5,
    "images_failed": 0
  }
}
```

### 5.3 SAM2 Segmentation (`/segment`)

```json
{
  "masks": [
    {
      "mask": "base64_encoded_png...",
      "score": 0.950
    }
  ],
  "input_point": [392, 648],
  "image_shape": [876, 870]
}
```

### 5.4 SAM2 Binary Mask (`/segment-binary`)

```json
{
  "success": true,
  "mask": "base64_encoded_png...",
  "score": 0.950
}
```

### 5.5 SAM-3D Generation

**요청 (`/generate-3d`)**
```json
{
  "task_id": "845860ec-24f1-4989-8ec9-e7883596702a",
  "status": "queued"
}
```

**완료 (`/generate-3d-status/{task_id}`)**
```json
{
  "status": "completed",
  "task_id": "845860ec-24f1-4989-8ec9-e7883596702a",
  "mesh_url": "/assets/mesh_b096d5df.glb",
  "ply_url": "/assets/ply_825968da.ply",
  "gif_url": null
}
```

---

## 6. 발견된 이슈 및 해결

### 6.1 CLIP 모델 로딩 실패

**문제**
```
ValueError: Due to a serious vulnerability issue in `torch.load`,
even with `weights_only=True`, we now require users to upgrade
torch to at least v2.6 (CVE-2025-32434)
```

**원인**
- torch 2.5.x 버전에서 transformers의 보안 정책 강화

**해결**
```python
# DeCl/models/classifier.py 수정
CLIPModel.from_pretrained(model_id, use_safetensors=True)
```

### 6.2 Kaolin 호환성 문제

**문제**
```
ImportError: kaolin/_C.so: undefined symbol: _ZN3c105ErrorC2E...
```

**원인**
- torch 2.9.1로 업그레이드 시 kaolin 바이너리 불일치

**해결**
- torch 2.5.1로 다운그레이드 (kaolin 0.17.0 호환 버전)

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 6.3 YOLO Detection 결과 형식

**문제**
- `detect_smart()` 반환값이 dict 형식이 아닌 raw 형식

**해결**
- 테스트 스크립트에서 결과 변환 로직 추가

```python
detections = []
boxes = raw_results.get('boxes', [])
scores = raw_results.get('scores', [])
classes = raw_results.get('classes', [])

for i in range(len(boxes)):
    detections.append({
        'bbox': boxes[i].tolist(),
        'confidence': float(scores[i]),
        'label': class_list[int(classes[i])]
    })
```

---

## 7. 성능 측정

### 모델 초기화 시간

| 모델 | 시간 |
|------|------|
| YOLO-World (yolov8l-world.pt) | 0.58s |
| CLIP (clip-vit-base-patch32) | 3.93s |
| SAM2.1 (sam2.1-hiera-large) | ~5s |

### 추론 시간

| 작업 | 시간 | 비고 |
|------|------|------|
| YOLO 탐지 (1장) | 0.59~1.10s | Ensemble 포함 |
| CLIP 분류 (1객체) | 0.04~0.10s | |
| SAM2 세그먼트 | 0.16~0.21s | |
| SAM-3D 3D 생성 | 136~147s | A100 기준 |

### 메모리 사용량

| 모델 | GPU 메모리 |
|------|------------|
| YOLO-World | ~3.5GB |
| CLIP | ~600MB |
| SAM2 | ~2.5GB |
| SAM-3D | ~10GB (피크) |

---

## 8. 결론

### 테스트 결과 요약

| 구성요소 | 상태 | 비고 |
|----------|------|------|
| YOLO-World 객체 탐지 | ✅ 정상 | 57개 가구 클래스 |
| CLIP 세부 유형 분류 | ✅ 정상 | safetensors 사용 |
| Knowledge Base 조회 | ✅ 정상 | 17종 가구, is_movable |
| SAM2 마스크 생성 | ✅ 정상 | score 0.95 |
| SAM-3D 3D 변환 | ✅ 정상 | GLB, PLY 출력 |
| Volume Calculator | ✅ 정상 | 상대→절대 변환 |
| JSON 응답 포맷 | ✅ 정상 | 스펙 일치 |

### 권장 사항

1. **SAHI 설치** - 작은 객체 탐지 향상
   ```bash
   pip install sahi
   ```

2. **로깅 개선** - print → logging 프레임워크

3. **단위 테스트 추가** - pytest 기반 자동화

### 생성된 테스트 파일

| 파일 | 용도 |
|------|------|
| `test_decl_pipeline.py` | DeCl 파이프라인 테스트 |
| `test_sam_api.py` | SAM2/SAM-3D API 테스트 |
| `test_mask_output.png` | SAM2 마스크 출력 |
| `test_masked_output.png` | 마스크 적용 이미지 |
| `assets/mesh_*.glb` | 3D GLB 메시 |
| `assets/ply_*.ply` | 3D PLY 포인트클라우드 |

---

**AI Logic 파이프라인이 정상적으로 구현되어 있으며, 모든 핵심 기능이 동작함을 확인했습니다.**
