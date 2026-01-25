# Update Documentation

소스 오브 트루스로부터 문서 동기화:

## 1. 소스 분석

1. requirements.txt 읽기
   - 모든 의존성 추출
   - 버전 정보 확인

2. api.py 엔드포인트 분석
   - FastAPI 라우트 추출
   - 요청/응답 형식 확인

3. ai/processors/ 파일 분석
   - 파이프라인 단계 확인
   - 입출력 형식 확인

## 2. 문서 업데이트

### README.md 업데이트
- 주요 기능 목록
- API 엔드포인트 테이블
- 빠른 시작 가이드
- 요청/응답 예시
- 운영 가이드 (환경변수, 트러블슈팅, 롤백)
- 의존성 목록

### CLAUDE.md 업데이트
- Overview 섹션
- Architecture 섹션
- API Endpoints 목록
- File Structure
- Code Modification Guidelines

### docs/tdd/TDD_PIPELINE_V2.md 업데이트
- Data Flow (입출력 형식)
- API Endpoints 스펙
- Component Details
- Dependencies

## 3. 검증

- 삭제된 파일 반영 (git status 확인)
- 단위 설명 일관성 (상대 길이/부피)
- SAM2/is_movable 등 제거된 기능 정리

## 4. 오래된 문서 식별

- 90일 이상 수정되지 않은 문서 찾기
- 수동 검토 목록 제공

## 5. 변경 요약 표시

## 문서 구조

```
README.md                  # 메인 문서 (설치, 실행, 운영)
CLAUDE.md                  # Claude Code 가이드 (아키텍처, 코드 수정)
docs/
├── tdd/
│   └── TDD_PIPELINE_V2.md # 기술 설계 문서 (API 스펙)
└── qa/                    # QA 테스트 리포트 (수동 관리)
```

## 소스 오브 트루스

- requirements.txt - Python 의존성
- api.py - API 엔드포인트 정의
- ai/processors/ - 파이프라인 프로세서
- ai/pipeline/ - 파이프라인 오케스트레이터

## 문서 갱신 시 주의사항

- docs/qa/는 수동 관리 (자동 갱신하지 않음)
- 생성된 문서가 아닌 **기존 문서 수정**
- git status로 삭제된 파일 확인 후 반영
