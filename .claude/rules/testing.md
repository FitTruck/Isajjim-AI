# Testing Requirements

## 최소 테스트 커버리지: 80%

테스트 유형 (모두 필수):
1. **단위 테스트** - 개별 함수, 유틸리티, 클래스
2. **통합 테스트** - API 엔드포인트, 파이프라인 프로세서
3. **E2E 테스트** - 전체 파이프라인 플로우 (pytest로 API 호출)

## 테스트 주도 개발

필수 워크플로:
1. 테스트 먼저 작성 (RED)
2. 테스트 실행 - 실패해야 함
3. 최소 구현 작성 (GREEN)
4. 테스트 실행 - 통과해야 함
5. 리팩토링 (IMPROVE)
6. 커버리지 확인 (80%+)

## pytest 명령어

```bash
# 전체 테스트 실행
pytest -v

# 커버리지와 함께 실행
pytest --cov=ai --cov-report=term-missing

# 특정 테스트만 실행
pytest tests/test_volume_calculator.py -v

# 실패 시 즉시 중단
pytest -x

# 마지막 실패 테스트만 재실행
pytest --lf

# 특정 마커만 실행
pytest -m "not slow"
```

## 테스트 구조

```python
# tests/test_example.py
import pytest
from ai.processors.example import process

class TestProcess:
    """process 함수 테스트"""

    def test_정상_입력_처리(self):
        """정상 입력이 올바르게 처리되는지 확인"""
        result = process(valid_input)
        assert result.status == "success"

    def test_빈_입력_에러(self):
        """빈 입력에 대해 적절한 에러 발생"""
        with pytest.raises(ValueError):
            process([])

    @pytest.mark.parametrize("input,expected", [
        (1, 1),
        (2, 4),
        (3, 9),
    ])
    def test_파라미터화_테스트(self, input, expected):
        """여러 입력에 대해 테스트"""
        assert process(input) == expected
```

## 테스트 실패 트러블슈팅

1. **tdd-guide** 에이전트 사용
2. 테스트 격리 확인
3. mock이 올바른지 확인
4. 테스트가 아닌 구현 수정 (테스트가 틀린 경우 제외)

## 에이전트 지원

- **tdd-guide** - 새 기능에 선제적으로 사용, 테스트 먼저 작성 강제
