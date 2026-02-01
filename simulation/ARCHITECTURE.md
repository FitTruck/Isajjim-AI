# 트럭 적재 시뮬레이션 아키텍처

## 개요

3D 트럭 적재 시뮬레이션 시스템으로, 가구를 트럭에 배치하는 기능을 제공합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (Three.js)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  PLY Loader │  │ Drag/Drop  │  │   3D Visualization      │  │
│  │  (Points)   │  │ Controls   │  │   (Container + Items)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP API
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Server (:8080)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   routes.py │  │  models.py  │  │   optimizer.py          │  │
│  │  (API 엔드) │  │  (Pydantic) │  │   obb_packer.py         │  │
│  │             │  │             │  │   ply_alignment.py      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Static Assets                             │
│  ┌─────────────┐  ┌─────────────────────────────────────────┐   │
│  │ assets/*.ply│  │         static/simulator.html           │   │
│  │ (3D Models) │  │         (Frontend SPA)                  │   │
│  └─────────────┘  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 디렉토리 구조

```
simulation/
├── __init__.py          # 모듈 초기화
├── run_server.py        # 프로덕션 서버 실행
├── test_server.py       # 독립 테스트 서버 (uvicorn)
├── routes.py            # FastAPI 라우터 (API 엔드포인트)
├── models.py            # Pydantic 데이터 모델
├── optimizer.py         # 3D Bin Packing 알고리즘 (BLF, py3dbp)
├── obb_packer.py        # OBB 기반 Extreme Points 알고리즘
├── ply_alignment.py     # PLY 정렬 서비스 (OBB 기반)
├── align_ply.py         # PLY 배치 정렬 스크립트
├── integration.py       # 메인 API 통합 유틸리티
├── browser_test.py      # 브라우저 테스트 스크립트
├── static/
│   └── simulator.html   # Three.js 기반 3D 시뮬레이터 UI
├── assets/
│   ├── *.ply            # 원본 3D 모델 파일들
│   └── aligned/         # 축 정렬된 PLY 파일들
├── docs/
│   ├── OBB_ALGORITHM.md             # OBB 알고리즘 설명
│   ├── PLY_ALIGNMENT.md             # PLY 정렬 가이드
│   ├── SIMULATION_API_FORMAT.md     # 백엔드 전송 JSON 포맷
│   └── LAFF_IMPLEMENTATION_BACKUP.md  # 이전 구현 백업
└── ARCHITECTURE.md      # 이 문서
```

## API 엔드포인트

### 기본 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/simulation/` | GET | 시뮬레이터 HTML 페이지 |
| `/simulation/trucks` | GET | 트럭 프리셋 목록 |
| `/simulation/data/{id}` | GET | 시뮬레이션 데이터 (가구 목록) |
| `/simulation/state/{id}` | POST | 시뮬레이션 상태 저장 |
| `/simulation/state/{id}` | GET | 저장된 상태 불러오기 |
| `/simulation/static/{filename}` | GET | 정적 파일 제공 |
| `/simulation/assets/{file}` | GET | PLY 에셋 파일 제공 |

### 최적화 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/simulation/optimize` | POST | BLF/py3dbp 최적화 |
| `/simulation/optimize-obb` | POST | OBB 기반 최적화 (단일 트럭) |
| `/simulation/optimize-obb-auto` | POST | 멀티 트럭 자동 최적화 |
| `/simulation/optimizer-status` | GET | 최적화 엔진 상태 |

### PLY 정렬 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/simulation/align-ply` | POST | PLY 객체 정렬 (Base64) |
| `/simulation/alignment-status` | GET | PLY 정렬 서비스 상태 |

## 최적화 알고리즘

### 1. BLF (Bottom-Left-Fill)
- `optimizer.py`에 구현
- Height map 기반 그리드 탐색
- 빠른 속도, 단순한 구현

### 2. py3dbp
- `optimizer.py`에서 호출
- 외부 라이브러리 사용 (`pip install py3dbp`)
- 검증된 3D bin packing 알고리즘

### 3. OBB (Extreme Points)
- `obb_packer.py`에 구현
- Extreme Points 알고리즘 기반
- 70% 지지 규칙, 수평 회전만 허용
- 뒤쪽부터 차곡차곡 적재
- 멀티 트럭 자동 선택 지원

## 트럭 프리셋

```python
TRUCK_PRESETS = {
    "1ton": TruckSpec("1톤 트럭", 1.7, 2.8, 1.7, 1000),
    "2.5ton": TruckSpec("2.5톤 트럭", 2.0, 4.3, 1.9, 2500),
    "5ton": TruckSpec("5톤 트럭", 2.3, 6.2, 2.4, 5000),
    "11ton": TruckSpec("11톤 트럭", 2.4, 9.0, 2.6, 11000),
}
```

## 실행 방법

```bash
# 테스트 서버 실행
python -m simulation.test_server

# 프로덕션 서버 실행
python -m simulation.run_server

# 브라우저에서 접속
http://localhost:8080
```

## 의존성

- **FastAPI**: API 서버
- **Three.js**: 3D 렌더링 (CDN)
- **PLYLoader**: PLY 파일 로딩 (Three.js addon)
- **py3dbp** (선택): 3D bin packing 라이브러리
- **Open3D** (선택): PLY 정렬 서비스

## 백엔드 전송 JSON 포맷

시뮬레이션 결과를 백엔드로 전송할 때 사용하는 JSON 포맷은 `docs/SIMULATION_API_FORMAT.md`를 참조하세요.

### 단위 요약

| 항목 | 단위 |
|------|------|
| 가구 치수 (width, depth, height) | **mm** |
| 트럭 규격 (spec.width/depth/height) | **m** |
| 배치 좌표 (placement.x/y/z) | **m** |
| 부피 (volume) | **m³** |
| 적재율 (utilization) | **%** |
