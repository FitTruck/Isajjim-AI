# Coding Style

## 불변성 (CRITICAL)

가능한 불변 패턴 사용:

```python
# WRONG: 뮤테이션
def update_config(config, key, value):
    config[key] = value  # 원본 변경!
    return config

# CORRECT: 불변성
def update_config(config, key, value):
    return {**config, key: value}  # 새 객체 반환
```

## 파일 구조

작은 파일 여러 개 > 큰 파일 몇 개:
- 높은 응집도, 낮은 결합도
- 일반적으로 200-400줄, 최대 800줄
- 큰 모듈에서 유틸리티 분리
- 타입이 아닌 기능/도메인별로 구성

## 에러 핸들링

항상 에러를 포괄적으로 처리:

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = risky_operation()
    return result
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise RuntimeError("사용자 친화적 메시지") from e
```

## 입력 검증

항상 사용자 입력 검증 (Pydantic 사용):

```python
from pydantic import BaseModel, Field, validator

class FurnitureRequest(BaseModel):
    image_urls: list[str] = Field(..., min_length=1, max_length=10)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @validator('image_urls', each_item=True)
    def validate_url(cls, v):
        if not v.startswith('https://'):
            raise ValueError('HTTPS URL 필요')
        return v
```

## 타입 힌트

모든 public 함수에 타입 힌트 필수:

```python
from typing import Optional

def calculate_volume(
    width: float,
    depth: float,
    height: float,
    unit: str = "m"
) -> Optional[float]:
    """볼륨을 계산합니다."""
    if width <= 0 or depth <= 0 or height <= 0:
        return None
    return width * depth * height
```

## 코드 품질 체크리스트

작업 완료 전 확인:
- [ ] 코드가 읽기 쉽고 명확한 이름 사용
- [ ] 함수가 작음 (<50줄)
- [ ] 파일이 집중됨 (<800줄)
- [ ] 깊은 중첩 없음 (>4 레벨)
- [ ] 적절한 에러 핸들링
- [ ] print() 대신 logging 사용
- [ ] 하드코딩된 값 없음
- [ ] 타입 힌트 포함
