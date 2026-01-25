# Test Coverage

테스트 커버리지를 분석하고 누락된 테스트를 생성:

1. 커버리지와 함께 테스트 실행:
   ```bash
   pytest --cov=ai --cov=api --cov-report=term-missing --cov-report=html
   ```

2. 커버리지 리포트 분석 (htmlcov/index.html 또는 터미널 출력)

3. 80% 커버리지 미달 파일 식별

4. 커버되지 않은 각 파일에 대해:
   - 테스트되지 않은 코드 경로 분석
   - 함수에 대한 단위 테스트 생성
   - API 엔드포인트에 대한 통합 테스트 생성

5. 새 테스트 통과 확인

6. 변경 전/후 커버리지 지표 표시

7. 프로젝트 전체 80%+ 커버리지 달성 확인

## 집중해야 할 영역

- Happy path 시나리오
- 에러 핸들링
- 엣지 케이스 (None, 빈 리스트, 최대값)
- 경계 조건

## pytest-cov 명령어

```bash
# 전체 커버리지 측정
pytest --cov=ai --cov=api --cov-report=term-missing

# HTML 리포트 생성
pytest --cov=ai --cov-report=html
# 결과: htmlcov/index.html

# 특정 모듈만 커버리지 측정
pytest --cov=ai/processors --cov-report=term-missing

# 커버리지 임계값 설정 (미달 시 실패)
pytest --cov=ai --cov-fail-under=80

# 브랜치 커버리지 포함
pytest --cov=ai --cov-branch --cov-report=term-missing
```

## 이 프로젝트 핵심 모듈

우선순위가 높은 커버리지 대상:

1. **ai/pipeline/furniture_pipeline.py** - 파이프라인 오케스트레이터
2. **ai/processors/** - 각 처리 단계
3. **ai/gpu/gpu_pool_manager.py** - GPU 풀 관리
4. **ai/data/knowledge_base.py** - 가구 DB 매칭
5. **api.py** - FastAPI 엔드포인트

## 커버리지 목표

| 모듈 | 목표 |
|------|------|
| ai/processors/ | 80%+ |
| ai/pipeline/ | 80%+ |
| ai/gpu/ | 90%+ |
| api.py (엔드포인트) | 70%+ |
