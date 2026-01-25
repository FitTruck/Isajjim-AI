# Code Review

커밋되지 않은 변경사항에 대한 포괄적 보안 및 품질 검토:

1. 변경된 파일 확인: `git diff --name-only HEAD`

2. 각 변경 파일에 대해 검사:

**보안 이슈 (CRITICAL):**
- 하드코딩된 시크릿, API 키, 토큰
- SQL 인젝션 취약점
- 경로 탐색 위험
- 입력 검증 누락
- 안전하지 않은 의존성
- subprocess 명령어 인젝션

**코드 품질 (HIGH):**
- 50줄 초과 함수
- 800줄 초과 파일
- 4단계 초과 중첩
- 에러 핸들링 누락
- print() 문 (로깅 대신 사용)
- TODO/FIXME 주석
- public API에 docstring 누락

**모범 사례 (MEDIUM):**
- 타입 힌트 누락
- 새 코드에 대한 테스트 누락
- bare except 사용 (Exception 명시 필요)
- mutable 기본 인자 (def func(items=[]))

## 리포트 생성

3. 다음 형식으로 리포트 생성:
   - 심각도: CRITICAL, HIGH, MEDIUM, LOW
   - 파일 위치와 라인 번호
   - 이슈 설명
   - 수정 제안

4. CRITICAL 또는 HIGH 이슈 발견 시 커밋 차단

## Python 특화 검사

```bash
# 타입 검사
mypy ai/ --ignore-missing-imports

# 린팅
ruff check .

# 포매팅 검사
ruff format --check .

# 보안 취약점 스캔
bandit -r ai/ -ll

# 복잡도 검사
radon cc ai/ -a -s
```

## FastAPI 특화 검사

- Pydantic 모델에 적절한 validation 있는지
- 엔드포인트에 적절한 status_code 설정
- 비동기 함수에서 블로킹 호출 여부
- 의존성 주입 올바르게 사용하는지

## 이 프로젝트 특화 검사

- GPU 리소스 acquire 후 반드시 release 하는지
- subprocess 호출 시 환경변수 올바르게 설정하는지
- 이미지/마스크 처리 시 메모리 해제 하는지
- Firebase URL 검증 여부

보안 취약점이 있는 코드는 절대 승인하지 마세요!
