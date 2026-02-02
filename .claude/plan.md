# Implementation Plan: PLY 전처리 파이프라인

## 요구사항 재확인

GCS에 PLY 파일을 저장하기 전에 다음 전처리가 필요합니다:

1. **축 정렬**: `simulation/ply_alignment.py`의 OBB 기반 정렬 방법 적용
2. **리사이징**: 절대 치수(mm)에 맞게 PLY 크기 조정
3. **다운샘플링**: `simulator.html`의 Stride 다운샘플링으로 용량 최소화

---

## 현재 아키텍처 분석

### 1. PLY 정렬 로직 (ply_alignment.py)

`PLYAlignmentService` 클래스가 이미 존재:
- OBB 기반 축 정렬 (`_align_to_floor`)
- Z-up → Y-up 좌표계 변환 (`_convert_to_yup`)
- Base64 입출력 지원 (`align_from_base64`)

```python
# 핵심 메서드
def align_from_base64(self, ply_base64: str) -> Tuple[str, AlignmentResult]:
    """Base64 PLY 정렬 후 Base64로 반환"""
```

### 2. Stride 다운샘플링 로직 (simulator.html)

JavaScript에서 구현된 다운샘플링:
```javascript
const MAX_POINTS = 50000;

function downsampleGeometry(geometry, maxPoints) {
  const totalPoints = positions.length / 3;
  const stride = Math.ceil(totalPoints / maxPoints);

  for (let i = 0; i < totalPoints; i += stride) {
    newPositions.push(positions[idx], positions[idx + 1], positions[idx + 2]);
    if (newColors) {
      newColors.push(colors[idx], colors[idx + 1], colors[idx + 2]);
    }
  }
}
```

### 3. 현재 PLY 처리 흐름 (furniture_pipeline.py)

```python
# 현재 흐름
gen_result["ply_b64"]  # SAM-3D 출력
    ↓
tempfile 저장 → 치수 계산
    ↓
GCS 업로드  # 전처리 없이 원본 업로드
```

---

## 구현 계획

### Phase 1: PLY 전처리 서비스 생성
**파일:** `ai/processors/ply_preprocessor.py` (신규)

통합 전처리 파이프라인:
```python
class PLYPreprocessor:
    def __init__(
        self,
        max_points: int = 50000,      # 최대 포인트 수
        convert_to_yup: bool = True   # Y-up 좌표계 변환
    ):
        self.max_points = max_points
        self.alignment_service = PLYAlignmentService(convert_to_yup)

    def process(
        self,
        ply_b64: str,
        target_width_mm: float,
        target_depth_mm: float,
        target_height_mm: float
    ) -> Tuple[str, PreprocessResult]:
        """
        PLY 전처리 파이프라인:
        1. 축 정렬 (OBB 기반)
        2. 스케일링 (절대 치수에 맞게)
        3. 다운샘플링 (Stride 방식)
        """
```

### Phase 2: Stride 다운샘플링 구현
**파일:** `ai/processors/ply_preprocessor.py`

JavaScript 로직을 Python으로 포팅:
```python
def _downsample_points(
    self,
    points: np.ndarray,
    colors: Optional[np.ndarray],
    max_points: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Stride 기반 포인트 다운샘플링.

    Args:
        points: (N, 3) 포인트 좌표
        colors: (N, 3) RGB 색상 (옵션)
        max_points: 최대 포인트 수

    Returns:
        다운샘플링된 (points, colors)
    """
    total_points = len(points)
    if total_points <= max_points:
        return points, colors

    stride = math.ceil(total_points / max_points)
    indices = np.arange(0, total_points, stride)

    new_points = points[indices]
    new_colors = colors[indices] if colors is not None else None

    return new_points, new_colors
```

### Phase 3: 스케일링 구현
**파일:** `ai/processors/ply_preprocessor.py`

절대 치수에 맞게 PLY 스케일 조정:
```python
def _scale_to_absolute_size(
    self,
    pcd: o3d.geometry.PointCloud,
    target_width_mm: float,
    target_depth_mm: float,
    target_height_mm: float
) -> o3d.geometry.PointCloud:
    """
    절대 치수에 맞게 스케일링 (균일 스케일 유지).

    현재 PLY 치수와 target 치수 비교 후 스케일 팩터 계산.
    """
    aabb = pcd.get_axis_aligned_bounding_box()
    current_size = aabb.get_max_bound() - aabb.get_min_bound()

    # mm → m 변환
    target_w = target_width_mm / 1000
    target_d = target_depth_mm / 1000
    target_h = target_height_mm / 1000

    # 균일 스케일 팩터 (비율 유지)
    scale_factors = [
        target_w / current_size[0],  # width
        target_h / current_size[1],  # height (Y-up)
        target_d / current_size[2]   # depth
    ]
    scale = min(scale_factors)  # 가장 작은 비율로 균일 스케일

    pcd.scale(scale, center=pcd.get_center())
    return pcd
```

### Phase 4: FurniturePipeline 통합
**파일:** `ai/pipeline/furniture_pipeline.py` (수정)

GCS 업로드 전에 전처리 적용:
```python
# process_single_image() 내부
if gen_result.get("ply_b64") and self.gcs_service:
    # 전처리 적용 (축 정렬 + 스케일링 + 다운샘플링)
    preprocessor = PLYPreprocessor(max_points=50000)
    processed_ply_b64, preprocess_result = preprocessor.process(
        ply_b64=gen_result["ply_b64"],
        target_width_mm=abs_result.width_mm,
        target_depth_mm=abs_result.depth_mm,
        target_height_mm=abs_result.height_mm
    )

    # 전처리된 PLY로 GCS 업로드
    obj.ply_url = await self.gcs_service.upload_ply_base64(
        processed_ply_b64, filename
    )
```

### Phase 5: 설정 상수 추가
**파일:** `ai/config.py` (수정)

```python
# PLY 전처리 설정
PLY_MAX_POINTS = 50000        # 다운샘플링 최대 포인트 수
PLY_CONVERT_TO_YUP = True     # Y-up 좌표계 변환 (Three.js 호환)
PLY_ENABLE_PREPROCESSING = True  # 전처리 활성화
```

---

## 데이터 흐름 (변경 후)

```
SAM-3D PLY (base64)
    ↓
1. 축 정렬 (OBB.R.T 역회전)
    ↓
2. 바닥 배치 (Z-min = 0)
    ↓
3. 좌표계 변환 (Z-up → Y-up)
    ↓
4. 스케일링 (절대 치수 mm → m)
    ↓
5. Stride 다운샘플링 (max_points)
    ↓
GCS 업로드
```

---

## 파일 변경 요약

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `ai/processors/ply_preprocessor.py` | **신규** | PLY 전처리 파이프라인 |
| `ai/processors/__init__.py` | 수정 | PLYPreprocessor export 추가 |
| `ai/pipeline/furniture_pipeline.py` | 수정 | GCS 업로드 전 전처리 적용 |
| `ai/config.py` | 수정 | PLY 전처리 설정 상수 추가 |

---

## 기술적 고려사항

### Open3D 의존성
- `ply_alignment.py`는 Open3D 사용
- Open3D 없는 환경에서는 정렬/스케일링 스킵 (다운샘플링만 적용)
- lazy import로 ImportError 방지

### 성능 최적화
- 다운샘플링으로 파일 크기 ~50-90% 감소 (포인트 수에 따라)
- 50,000 포인트 기준: 약 2MB → ~200KB

### 좌표계 호환성
- Y-up 변환으로 Three.js/WebGL 렌더러 호환
- simulator.html에서 추가 변환 불필요

---

## 리스크 및 고려사항

### HIGH
- **정밀도 손실**: 다운샘플링으로 세밀한 디테일 손실 가능
  - 완화: max_points 설정 가능하게 구현

### MEDIUM
- **스케일링 정확도**: 균일 스케일로 비율 유지하지만 정확한 치수 매칭 어려움
  - 완화: 가장 작은 비율 사용으로 트럭 충돌 방지

### LOW
- **Open3D 설치**: 일부 환경에서 설치 어려움
  - 완화: Open3D 없으면 다운샘플링만 적용

---

## 테스트 계획

1. **단위 테스트**
   - `test_ply_preprocessor.py`: 각 전처리 단계 테스트
   - 다운샘플링 결과 포인트 수 검증
   - 스케일링 결과 치수 검증

2. **통합 테스트**
   - GCS 업로드된 PLY 파일 검증
   - simulator.html에서 렌더링 테스트

---

## 구현 완료 (2026-02-02)

✅ **Phase 1**: `ai/processors/ply_preprocessor.py` 생성
✅ **Phase 2**: Stride 다운샘플링 구현 (numpy + Open3D 지원)
✅ **Phase 3**: 스케일링 구현 (절대 치수 기반 균일 스케일)
✅ **Phase 4**: `furniture_pipeline.py` 통합 (GCS 업로드 전 전처리 적용)
✅ **Phase 5**: `ai/config.py`에 설정 상수 추가
✅ **테스트**: `tests/test_ply_preprocessor.py` (14개 테스트 통과)
✅ **문서**: `CLAUDE.md` 업데이트
