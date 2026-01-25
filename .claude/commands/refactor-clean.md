# Refactor Clean

Python 프로젝트에서 데드 코드를 안전하게 식별하고 제거:

1. 데드 코드 분석 도구 실행:
   - vulture: 미사용 함수, 클래스, 변수 탐지
   - autoflake: 미사용 import 탐지
   - ruff check --select F401,F841: 미사용 import/변수 탐지
   - pip-check 또는 pipdeptree: 미사용 의존성 확인

2. 분석 결과를 심각도별로 분류:
   - SAFE: 테스트 파일, 미사용 유틸리티
   - CAUTION: API 엔드포인트, 프로세서
   - DANGER: 설정 파일, 메인 엔트리포인트

3. 안전한 삭제만 제안

4. 각 삭제 전:
   - pytest 전체 실행
   - 테스트 통과 확인
   - 변경 적용
   - 테스트 재실행
   - 실패 시 롤백

5. 정리된 항목 요약 표시

## 사용 도구

```bash
# 미사용 코드 탐지
vulture . --min-confidence 80

# 미사용 import 자동 제거
autoflake --in-place --remove-all-unused-imports --recursive .

# ruff로 미사용 import/변수 확인
ruff check --select F401,F841 .

# 의존성 트리 확인
pipdeptree --warn silence
```

## 주의사항

- __init__.py의 re-export는 vulture가 오탐할 수 있음
- FastAPI 엔드포인트 데코레이터는 직접 호출되지 않아도 사용 중
- Pydantic 모델 필드는 런타임에 사용됨

테스트 실행 없이 코드를 삭제하지 마세요!
