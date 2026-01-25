# Update Codemaps

코드베이스 구조를 분석하고 아키텍처 문서 갱신:

1. 모든 Python 소스 파일에서 import, export, 의존성 스캔

2. 다음 형식으로 토큰 효율적인 코드맵 생성:
   - codemaps/architecture.md - 전체 아키텍처
   - codemaps/api.md - API 엔드포인트 구조
   - codemaps/pipeline.md - AI 파이프라인 구조
   - codemaps/processors.md - 프로세서 모듈 구조

3. 이전 버전과 변경률 계산

4. 변경률 30% 초과 시 사용자 승인 요청

5. 각 코드맵에 갱신 타임스탬프 추가

6. .reports/codemap-diff.txt에 리포트 저장

## 분석 대상

```
ai/
├── __init__.py
├── config.py              # 설정
├── main.py                # CLI 엔트리포인트
├── gpu/                   # GPU 관리
│   └── gpu_pool_manager.py
├── pipeline/              # 파이프라인
│   └── furniture_pipeline.py
├── processors/            # 처리 단계
│   ├── 1_firebase_images_fetch.py
│   ├── 2_YOLO_detect.py
│   ├── 4_DB_movability_check.py
│   ├── 6_SAM3D_convert.py
│   └── 7_volume_calculate.py
├── subprocess/            # 서브프로세스 워커
│   └── generate_3d_worker.py
├── data/                  # 데이터/DB
│   └── knowledge_base.py
└── utils/                 # 유틸리티
    └── image_ops.py

api.py                     # FastAPI 메인 서버
```

## 코드맵 형식 예시

```markdown
# Pipeline Architecture

## 의존성 그래프

```
api.py
  └── ai/pipeline/furniture_pipeline.py
        ├── ai/processors/2_YOLO_detect.py
        ├── ai/processors/4_DB_movability_check.py
        ├── ai/processors/6_SAM3D_convert.py
        └── ai/processors/7_volume_calculate.py
```

## 모듈별 책임

| 모듈 | 책임 |
|------|------|
| furniture_pipeline.py | 파이프라인 오케스트레이션 |
| 2_YOLO_detect.py | YOLOE-seg 객체 탐지 |
| ... | ... |
```

## Python 분석 도구

```bash
# 의존성 그래프 생성
pydeps ai/ --max-bacon=2 --cluster

# import 분석
importlab ai/

# AST 기반 분석
python -c "import ast; ..."
```

고수준 구조에 집중하고, 구현 세부사항은 포함하지 마세요.
