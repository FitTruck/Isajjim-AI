# Implementation Plan: SAM2 Removal from V2 Pipeline

## Requirements Restatement

TDD 문서(TDD_PIPELINE_V2.md)에 따라 V2 파이프라인에서 SAM2는 deprecated 되었습니다:
- V2는 YOLOE-seg 마스크를 SAM-3D에 직접 전달합니다
- SAM2 API 호출이 제거되어 latency가 감소합니다
- `5_SAM2_mask_generate.py` 파일을 삭제하고 관련 참조를 정리합니다

## Impact Analysis

### Files to Modify

| File | Change Type | Risk |
|------|-------------|------|
| `ai/processors/5_SAM2_mask_generate.py` | **DELETE** | LOW |
| `ai/processors/__init__.py` | Remove SAM2 export | LOW |
| `ai/__init__.py` | Remove SAM2 export | LOW |
| `ai/pipeline/furniture_pipeline.py` | Remove SAM2 fallback | MEDIUM |
| `CLAUDE.md` | Update documentation | LOW |
| `README.md` | Update documentation | LOW |

### Files NOT Modified (SAM2 endpoints remain in api.py)

> **Important**: `/segment` and `/segment-binary` endpoints in `api.py` use SAM2 but are **separate** from the Furniture Analysis Pipeline. These endpoints are for standalone segmentation and **will be preserved**.

## Implementation Phases

### Phase 1: Remove SAM2 from ai/processors

**Step 1.1**: Edit `ai/processors/__init__.py`
- Remove `_stage5 = importlib.import_module('.5_SAM2_mask_generate', ...)`
- Remove `SAM2MaskGenerator = _stage5.SAM2MaskGenerator`
- Remove `'SAM2MaskGenerator'` from `__all__`
- Update docstring to remove SAM2 reference

**Step 1.2**: Delete `ai/processors/5_SAM2_mask_generate.py`

### Phase 2: Remove SAM2 from ai/__init__.py

**Step 2.1**: Edit `ai/__init__.py`
- Remove `SAM2MaskGenerator` from import
- Remove `SAM2MaskGenerator` from `__all__`
- Update docstring

### Phase 3: Update FurniturePipeline

**Step 3.1**: Edit `ai/pipeline/furniture_pipeline.py`
- Remove `generate_mask()` method (deprecated SAM2 fallback)
- Remove SAM2 fallback logic in processing loop
- Keep `sam2_api_url` parameter for SAM-3D API URL (rename to `api_url` for clarity)

### Phase 4: Update Documentation

**Step 4.1**: Edit `CLAUDE.md`
- Remove `5_SAM2_mask_generate.py` from file structure
- Update pipeline description

**Step 4.2**: Edit `README.md`
- Remove SAM2 reference from file structure

### Phase 5: Verification

**Step 5.1**: Run pytest
```bash
pytest tests/ -v -m "not slow"
```

**Step 5.2**: Verify imports work
```bash
python -c "from ai.processors import YoloDetector, SAM3DConverter, VolumeCalculator; print('OK')"
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking api.py /segment endpoints | api.py has its own SAM2 model loading, not affected |
| Breaking furniture pipeline | YOLOE-seg provides masks directly, SAM2 fallback unused |
| Import errors | Will test imports after changes |

## Estimated Complexity: LOW

- Changes are mostly deletions
- SAM2 is already deprecated and unused in V2 pipeline
- No new code required, only cleanup

## CONFIRMATION REQUIRED

Proceed with this plan? (yes/no/modify)
