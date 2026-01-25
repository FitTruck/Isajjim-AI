---
description: 테스트 주도 개발 워크플로 강제. 인터페이스 정의, 테스트 먼저 작성, 최소 구현. 80%+ 커버리지 보장.
---

# TDD Command

**tdd-guide** 에이전트를 호출하여 테스트 주도 개발 방법론을 강제합니다.

## 이 명령어가 하는 일

1. **인터페이스 정의** - 타입/클래스 먼저 정의
2. **테스트 먼저 작성** - 실패하는 테스트 작성 (RED)
3. **최소 구현** - 테스트 통과할 만큼만 작성 (GREEN)
4. **리팩토링** - 테스트 통과 유지하며 개선 (REFACTOR)
5. **커버리지 확인** - 80%+ 테스트 커버리지 보장

## 사용 시점

- 새 기능 구현
- 새 함수/클래스 추가
- 버그 수정 (버그 재현 테스트 먼저 작성)
- 기존 코드 리팩토링
- 핵심 비즈니스 로직 작성

## TDD 사이클

```
RED → GREEN → REFACTOR → REPEAT

RED:      실패하는 테스트 작성
GREEN:    테스트 통과할 최소 코드 작성
REFACTOR: 테스트 통과 유지하며 코드 개선
REPEAT:   다음 기능/시나리오
```

## 예제 사용법

```
User: /tdd 가구 볼륨 계산 함수가 필요해

Agent (tdd-guide):
# TDD 세션: 가구 볼륨 계산기

## Step 1: 인터페이스 정의 (SCAFFOLD)

```python
# ai/processors/volume_calculator.py
from dataclasses import dataclass

@dataclass
class MeshDimensions:
    width: float
    depth: float
    height: float

def calculate_volume(dimensions: MeshDimensions) -> float:
    """메시 치수로부터 볼륨을 계산합니다."""
    raise NotImplementedError
```

## Step 2: 실패하는 테스트 작성 (RED)

```python
# tests/test_volume_calculator.py
import pytest
from ai.processors.volume_calculator import calculate_volume, MeshDimensions

class TestCalculateVolume:
    def test_정상_치수로_볼륨_계산(self):
        dimensions = MeshDimensions(width=1.0, depth=2.0, height=0.5)

        volume = calculate_volume(dimensions)

        assert volume == 1.0  # 1.0 * 2.0 * 0.5 = 1.0

    def test_제로_치수는_제로_볼륨(self):
        dimensions = MeshDimensions(width=0, depth=2.0, height=0.5)

        volume = calculate_volume(dimensions)

        assert volume == 0.0

    def test_음수_치수는_예외_발생(self):
        dimensions = MeshDimensions(width=-1.0, depth=2.0, height=0.5)

        with pytest.raises(ValueError, match="음수 치수"):
            calculate_volume(dimensions)
```

## Step 3: 테스트 실행 - 실패 확인

```bash
pytest tests/test_volume_calculator.py -v

FAILED tests/test_volume_calculator.py::TestCalculateVolume::test_정상_치수로_볼륨_계산
    NotImplementedError

1 failed, 0 passed
```

✅ 예상대로 테스트 실패. 구현 준비 완료.

## Step 4: 최소 구현 (GREEN)

```python
# ai/processors/volume_calculator.py
def calculate_volume(dimensions: MeshDimensions) -> float:
    """메시 치수로부터 볼륨을 계산합니다."""
    if dimensions.width < 0 or dimensions.depth < 0 or dimensions.height < 0:
        raise ValueError("음수 치수는 허용되지 않습니다")

    return dimensions.width * dimensions.depth * dimensions.height
```

## Step 5: 테스트 실행 - 통과 확인

```bash
pytest tests/test_volume_calculator.py -v

PASSED tests/test_volume_calculator.py::TestCalculateVolume::test_정상_치수로_볼륨_계산
PASSED tests/test_volume_calculator.py::TestCalculateVolume::test_제로_치수는_제로_볼륨
PASSED tests/test_volume_calculator.py::TestCalculateVolume::test_음수_치수는_예외_발생

3 passed
```

✅ 모든 테스트 통과!

## Step 6: 커버리지 확인

```bash
pytest tests/test_volume_calculator.py --cov=ai/processors/volume_calculator --cov-report=term-missing

Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
ai/processors/volume_calculator      8      0   100%
---------------------------------------------------------------
TOTAL                                8      0   100%

Coverage: 100% ✅ (목표: 80%)
```

✅ TDD 세션 완료!
```

## TDD 모범 사례

**DO:**
- ✅ 구현 전에 테스트 먼저 작성
- ✅ 구현 전 테스트 실패 확인
- ✅ 테스트 통과할 최소 코드만 작성
- ✅ 테스트 통과 후에만 리팩토링
- ✅ 엣지 케이스와 에러 시나리오 추가
- ✅ 80%+ 커버리지 목표 (핵심 코드는 100%)

**DON'T:**
- ❌ 테스트 전에 구현 작성
- ❌ 변경 후 테스트 실행 생략
- ❌ 한 번에 너무 많은 코드 작성
- ❌ 실패하는 테스트 무시
- ❌ 구현 세부사항 테스트 (동작을 테스트)
- ❌ 모든 것을 모킹 (통합 테스트 선호)

## 포함할 테스트 유형

**단위 테스트** (함수 레벨):
- Happy path 시나리오
- 엣지 케이스 (빈 값, None, 최대값)
- 에러 조건
- 경계값

**통합 테스트** (컴포넌트 레벨):
- FastAPI 엔드포인트
- 파이프라인 프로세서
- 외부 서비스 호출

## 커버리지 요구사항

- **80% 최소** 모든 코드
- **100% 필수**:
  - 볼륨/치수 계산
  - GPU 할당 로직
  - 파이프라인 핵심 로직

## pytest 명령어

```bash
# 특정 테스트 파일 실행
pytest tests/test_volume_calculator.py -v

# 특정 테스트 클래스/함수 실행
pytest tests/test_volume_calculator.py::TestCalculateVolume::test_정상_치수로_볼륨_계산 -v

# 커버리지와 함께 실행
pytest --cov=ai --cov-report=term-missing

# 실패 시 즉시 중단
pytest -x

# 마지막 실패 테스트만 재실행
pytest --lf
```

## 다른 명령어와 연계

- `/plan`으로 먼저 구현 계획 수립
- `/tdd`로 테스트와 함께 구현
- `/test-coverage`로 커버리지 확인
- `/code-review`로 구현 검토
