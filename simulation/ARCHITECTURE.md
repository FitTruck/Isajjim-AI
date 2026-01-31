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
│  │   routes.py │  │  models.py  │  │     optimizer.py        │  │
│  │  (API 엔드) │  │  (Pydantic) │  │  (BLF 알고리즘)         │  │
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
├── test_server.py       # 독립 테스트 서버 (uvicorn)
├── routes.py            # FastAPI 라우터 (API 엔드포인트)
├── models.py            # Pydantic 데이터 모델
├── optimizer.py         # 3D Bin Packing 알고리즘 (BLF)
├── integration.py       # 메인 API 통합 유틸리티
├── static/
│   └── simulator.html   # Three.js 기반 3D 시뮬레이터 UI
├── assets/
│   └── *.ply            # 3D 모델 파일들
├── docs/
│   └── LAFF_IMPLEMENTATION_BACKUP.md  # 이전 구현 백업
└── ARCHITECTURE.md      # 이 문서
```

## API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/simulation/` | GET | 시뮬레이터 HTML 페이지 |
| `/simulation/trucks` | GET | 트럭 프리셋 목록 |
| `/simulation/data/{id}` | GET | 시뮬레이션 데이터 (가구 목록) |
| `/simulation/optimize` | POST | 최적화 실행 |
| `/simulation/assets/{file}` | GET | PLY 파일 제공 |

## 트럭 프리셋

```python
TRUCK_PRESETS = {
    "1ton": TruckSpec("1톤 트럭", 1.6, 2.8, 1.6, 1000),
    "2.5ton": TruckSpec("2.5톤 트럭", 2.0, 4.3, 1.9, 2500),
    "5ton": TruckSpec("5톤 트럭", 2.3, 6.2, 2.4, 5000),
}
```

## 실행 방법

```bash
# 프로젝트 루트에서 실행
python -m simulation.test_server

# 브라우저에서 접속
http://localhost:8080
```

## 의존성

- **FastAPI**: API 서버
- **Three.js**: 3D 렌더링 (CDN)
- **PLYLoader**: PLY 파일 로딩 (Three.js addon)
