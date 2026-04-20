# ABO 기반 절대 부피 정확도 평가 실험 계획서

> 논문 `docs/paper/Isajjim-framework.pdf` 실험 섹션(5장) 보강을 위한 추가 실험 계획.
> 기존 `EXPERIMENT_BENCHMARK.md`(SAM-3D 3종 벤치마크)와 상보적 역할이며, 본 문서는
> **최종 파이프라인(Proposed)의 산출 부피가 실세계 metric 치수에 얼마나 맞는가**를 검증한다.

---

## 1. 목적

현 논문 5장은 **추론 시간·VRAM**만 보고한다. 프레임워크의 핵심 주장("데이터 기반 객관적 절대 부피 산출")을 뒷받침하는 **정확도 지표가 전무**하며, 이는 심사에서 반드시 지적될 공백이다.

본 실험은 다음 세 질문에 동시에 답한다.

| 질문 | 대응 증거 |
|---|---|
| Q1. 제안 파이프라인이 실제 가구 치수와 얼마나 일치하는가? | ABO GT 기반 Volume MAPE, Per-axis MAPE |
| Q2. OBB 채택이 AABB 대비 실제로 기여하는가? | Baseline-A (AABB) 비교 |
| Q3. Gaussian-only 모드가 Full mesh 대비 정확도 손실 없이 속도·VRAM만 개선하는가? | Baseline-B (Full SAM-3D) 비교 |

이로써 **논문 4.3(SAM-3D Gaussian-only), 4.4(OBB), 4.5(절대 부피 변환)** 의 설계 선택 전부가 정량 증거로 방어된다.

---

## 2. 제약 조건

### 2.1 데이터셋이 충족해야 할 요건

| 요건 | 근거 |
|---|---|
| 단일 2D 이미지 + 가구 객체 | 논문 파이프라인 입력 포맷 |
| **Absolute metric scale(mm/cm/m) GT** | Volume MAPE 측정 필요. Pix3D는 이를 제공하지 않아 배제 (`feedback_pix3d_metric_scale.md` 참조) |
| 3D mesh/point cloud GT | nCD(normalized Chamfer) 측정 |
| 학술 사용 라이선스 | 재현성 |

### 2.2 카테고리 제약 — 프로젝트 KB 범위

평가 대상 가구는 반드시 YOLOE-seg가 인식하도록 설정된 프로젝트 KB 범위 안이어야 한다.

- 정의 위치: `ai/data/knowledge_base.py:17` `FURNITURE_DB`
- 허용 YOLO 클래스 추출: `get_all_yolo_classes()` (`ai/data/knowledge_base.py:409`)
- 총 27개 effective 카테고리 (`floor` 등 `exclude_from_output=True` 항목 제외)

| 그룹 | 개수 | 카테고리 |
|---|---|---|
| 가구 | 17 | sofa, bed, chair, dining table, coffee table, desk, bookshelf, wardrobe, drawer, nightstand, tv stand, cabinet, dish cabinet, display shelf, vanity table, mirror, storage box |
| 가전/전자 | 6 | refrigerator, washing machine, microwave/oven, air conditioner, monitor/tv, fan |
| 기타 대형 | 4 | piano, massage chair, treadmill, exercise bike |

---

## 3. 데이터셋 선택

### 3.1 후보 비교

| 데이터셋 | Metric GT | 가구 17 커버 | 가전 6 커버 | 기타 4 커버 | 결론 |
|---|---|---|---|---|---|
| **ABO** (Collins et al., CVPR'22) | ✔ (`item_dimensions`) | 거의 전부 | 일부 | 소수 | **메인** |
| 3D-FUTURE (Fu et al., 2021) | ✔ (cm) | 전부 | 없음 | 없음 | 가구 전용·가전 누락 → 예비 |
| OmniObject3D (2023) | ✔ (real scan) | 일부 | 일부 | 일부 | 가구 카테고리 얕음 |
| IKEA 3D (Lim et al., 2013) | ✔ | 일부 | 없음 | 없음 | 규모 소수(219) |
| Pix3D (현행) | ✘ | 9종 | 0 | 0 | scale 결함 + 카테고리 부족 |

### 3.2 채택: **ABO (Amazon Berkeley Objects)** + 자체 실측 보조

공개 데이터셋 단독으로 KB 27종을 완전 충족하는 것은 없다. 따라서 2-tier 구성.

| 역할 | 소스 | 대상 카테고리 | 목적 |
|---|---|---|---|
| **메인 (공개 재현성)** | ABO | 가구 17종 ∩ ABO 제품 | Volume/Per-axis MAPE·nCD 정량 평가 |
| **보조 (KB 커버리지)** | 자체 촬영+줄자 | 가전 6 + 기타 4 + 일부 가구 | in-the-wild 검증 및 KB 전체 커버 증명 |
| **유지 (profiling)** | Pix3D 500 | 현행 9종 | 기존 표 1(시간·VRAM) 연속성 |

ABO 채택 근거
1. `item_dimensions` field에 height/length/width가 metric으로 명시 → Volume GT 직접 산출.
2. product_type이 KB 가구 17종과 높은 겹침(소파·의자·침대·식탁·책상·서랍장 등).
3. catalog 이미지(배경 clean) + 일부 lifestyle shot 모두 존재 → YOLOE-seg 통과 가능.
4. 3D mesh 동봉 → nCD 측정 유지.
5. CC-BY-4.0, 공개 재현 용이.

ABO의 한계와 완화
- catalog shot은 in-the-wild 대비 도메인 갭 → **자체 실측 보조 세트**로 보완.
- 가전·운동기구 샘플 얕음 → **자체 실측이 이 블록을 전담**.

---

## 4. 데이터셋 구성 상세

### 4.1 ABO 서브셋 (메인, n=500)

#### 준비 파이프라인

```
ABO 3D subset (8K meshes) download
        │
        ▼
  metadata 파싱  ── product_type 추출
        │
        ▼
  KB 매핑 테이블 적용 (abo_kb_mapping.json)
        │
        ▼
  item_dimensions 유효성 필터
   (결측·단위 이상치·0 값 제외)
        │
        ▼
  카테고리별 유효 샘플 집계
   (카테고리당 최소 25개 확보 조건)
        │
        ▼
  비례 층화 추출 n=500
```

#### KB ↔ ABO 매핑 (예시)

| KB 키 (base_name) | ABO product_type 후보 |
|---|---|
| SOFA | Sofa, Couch, Loveseat, Sectional |
| BED | Bed, Bed Frame, Bunk Bed |
| CHAIR_STOOL | Chair, Armchair, Office Chair, Stool |
| DINING_TABLE | Dining Table, Kitchen Table |
| COFFEE_TABLE | Coffee Table, Cocktail Table |
| DESK | Desk, Writing Desk, Computer Desk |
| BOOKSHELF | Bookcase, Bookshelf |
| WARDROBE | Wardrobe, Armoire |
| DRAWER | Dresser, Chest of Drawers |
| NIGHTSTAND | Nightstand, Bedside Table |
| TV_STAND | TV Stand, Media Console |
| CABINET | Cabinet, Storage Cabinet, Filing Cabinet |
| DISH_CABINET | Kitchen Cabinet, China Cabinet (수작업 검수) |
| DISPLAY_SHELF | Display Stand, Shelving Unit |
| VANITY_TABLE | Vanity, Dressing Table |
| MIRROR | Mirror |
| STORAGE_BOX | Storage Box, Storage Bin |

실제 카테고리·샘플 수는 Phase 1(데이터 준비) 완료 후 확정하여 본 문서를 업데이트한다. 카테고리당 최소 샘플이 25 미만이면 해당 카테고리는 ABO 트랙에서 제외하고 자체 실측으로 이전한다.

### 4.2 자체 실측 세트 (보조, n=20~30)

목적: ABO가 커버하지 못하는 가전·운동기구 10종 + in-the-wild 현장 이미지 검증.

| 항목 | 내용 |
|---|---|
| 촬영 | 스마트폰 단일 이미지 (정면 우위 각도 1장/객체) |
| 측정 | 줄자 W·D·H (mm 단위), 0.5 cm 단위 반올림 |
| 카테고리 | 가전 6 + 기타 4 + 일부 이사 도메인 가구 |
| 샘플 수 | 카테고리당 2–3개, 총 20–30 |
| 마스크 | YOLOE-seg가 직접 생성 (별도 주입 없음) |
| 기록 형식 | `experiments/benchmark/data/inhouse_ground_truth.json` |

인-하우스 세트는 통계 유의성 주장에는 사용하지 않고, **KB 전체 카테고리 커버리지의 정성적 증거**로 논문에 표 주석과 함께 첨부한다.

### 4.3 Pix3D (유지, profiling 용도)

- 기존 500 샘플 구성 그대로(`EXPERIMENT_BENCHMARK.md` §데이터셋 참조)
- 용도: 논문 표 1(평균 추론 시간·VRAM Peak) 연속성 확보
- 정확도 평가에는 사용하지 않음

---

## 5. 평가 지표

### 5.1 지표 정의

| 지표 | 정의 | 방어 대상 |
|---|---|---|
| **Volume MAPE(%)** | `\|V_pred − V_GT\| / V_GT`, V = W·D·H | 논문 핵심 주장 (절대 부피) |
| **Per-axis MAPE(%)** (W, D, H) | 각 축 `\|pred − GT\| / GT`. pred·GT 모두 크기순(L≥M≥S) 정렬 후 매칭 | 4.4절 OBB 단계 |
| **nCD** (normalized Chamfer) | 예측·GT 각각 자기 OBB diagonal로 정규화 후 CD | 4.3절 Gaussian-only의 형상 왜곡 검출 |
| **Success rate(%)** | YOLOE-seg 검출 성공 · SAM-3D 산출 성공 비율 | 파이프라인 안정성 |

### 5.2 통계 처리

- 전체 집계: 평균 ± 표준편차, 중앙값.
- 카테고리별 breakdown: 평균, 중앙값.
- 설정 간 비교: paired Wilcoxon signed-rank test (n=500, 정규성 가정 회피).
- 보고 수준: Volume MAPE는 **<15%를 잠정 합격선**, Per-axis MAPE는 **<12%**를 가이드라인으로 설정(Phase 1 실측 후 재보정).

---

## 6. 실험 설계

### 6.1 3 설정 Ablation

| 설정 | 3D 복원 | 치수 산출 | 분리하려는 기여 |
|---|---|---|---|
| Baseline-A | SAM-3D Gaussian-only | **AABB** | OBB의 기여 |
| Baseline-B | **SAM-3D Full** (mesh + postprocess) | OBB | Gaussian-only 채택의 정확도 손실 여부 |
| **Proposed** | SAM-3D Gaussian-only | OBB | 제안 구성 |

### 6.2 설정별 SAM-3D 파라미터

`ai/subprocess/persistent_3d_worker.py` 기준.

| 파라미터 | Baseline-A | Baseline-B | Proposed |
|---|---|---|---|
| `STAGE1_INFERENCE_STEPS` | 14 | 25 (default) | 14 |
| `STAGE2_INFERENCE_STEPS` | 4 | 25 (default) | 4 |
| `decode_formats` | `["gaussian"]` | `["gaussian","mesh"]` | `["gaussian"]` |
| `mesh_postprocess` | False | True | False |
| `texture_baking` | False | True | False |
| `GAUSSIAN_ONLY_MODE` | True | False | True |
| 치수 산출 | AABB | OBB | OBB |

### 6.3 고정 조건

| 항목 | 값 |
|---|---|
| GPU | NVIDIA L4 ×1 (22 GB usable) |
| 샘플 | 동일 500 샘플(카테고리 층화), 설정 간 순서 동일 |
| Seed | ablation 스크립트 기본값(`experiments/benchmark/scripts/workers/worker_ablation.py`와 일치) |
| 마스크 | YOLOE-seg 출력 직접 사용(추가 보정 없음) |
| KB 매칭 | V2.5 경로 유지(`ai/processors/8_absolute_volume_calculate.py`) |

---

## 7. 실행 계획 (단계별)

### Phase 0 — 설계 합의 (완료 시점: 본 문서 커밋)

- [x] 데이터셋·지표·카테고리 범위 확정
- [x] 계획서 커밋

### Phase 1 — 데이터 준비

| 산출물 | 경로 | 설명 |
|---|---|---|
| ABO 메타데이터 덤프 | `experiments/benchmark/data/abo/metadata/` | 원본 JSON |
| KB↔ABO 매핑 테이블 | `experiments/benchmark/data/abo_kb_mapping.json` | 수작업 검수 포함 |
| 후보 샘플 리스트 | `experiments/benchmark/data/abo_candidates.csv` | 필터 통과 전체 |
| 최종 500 샘플 | `experiments/benchmark/data/abo_samples_500.json` | 층화 추출 |
| 자체 실측 GT | `experiments/benchmark/data/inhouse_ground_truth.json` | 20–30 |

### Phase 2 — 스크립트 확장

| 변경 | 위치 |
|---|---|
| ABO loader 추가 | `experiments/benchmark/scripts/evaluate/abo_loader.py` (신규) |
| Volume/Per-axis MAPE 계산 | `experiments/benchmark/scripts/evaluate/compute_accuracy.py` (신규) |
| nCD 정규화 통일(OBB diagonal) | `experiments/benchmark/scripts/evaluate/compute_cd_summary.py` 수정 |
| Baseline-A(AABB) 분기 | 동일 파일 또는 `compute_ablation.py`에 플래그 추가 |
| Baseline-B(Full SAM-3D) 분기 | worker에 `GAUSSIAN_ONLY_MODE=False` 플래그 전달 |
| 실행 스크립트 | `experiments/benchmark/run/run_abo_accuracy.sh` (신규) |

### Phase 3 — 실험 실행

| 설정 | 샘플 | 예상 시간 | 비고 |
|---|---|---|---|
| Proposed | 500 | ~50분 | 기존 최적화 설정 |
| Baseline-A (AABB) | 500 | ~50분 | 복원은 동일, 치수만 AABB |
| Baseline-B (Full) | 500 | ~2–3시간 | mesh + postprocess 활성 |
| 자체 실측 × 3 설정 | 20–30 | ~30분 | 최종 비교용 |

총 예상 실행 시간: 4–5시간 (L4 단일 GPU 기준).

### Phase 4 — 분석 및 논문 반영

| 산출물 | 경로 |
|---|---|
| 정확도 집계 CSV | `experiments/benchmark/results/abo_accuracy_summary.csv` |
| 카테고리별 breakdown | `experiments/benchmark/results/abo_accuracy_by_category.csv` |
| 유의성 검정 결과 | `experiments/benchmark/results/abo_pvalues.csv` |
| 실패 케이스 시각화 | `experiments/benchmark/results/abo_failure_cases/` |

---

## 8. 논문 반영 방안

### 8.1 5장 개편(안)

- **5.1 실험 환경** — 두 문단으로 분리
  1. 시간·VRAM profiling: Pix3D 500 (현행 유지).
  2. 정확도 평가: ABO 500 층화 + 자체 실측 20–30.
- **5.2 실험 결과** — 표 재구성
  - 표 1 (유지): 평균 추론 시간·VRAM Peak (Pix3D)
  - **표 2 (신규)**: 정확도 × 3 설정, 전체 집계 (Volume MAPE / Per-axis MAPE / nCD / Success rate)
  - **표 3 (신규, 선택)**: 카테고리별 Volume MAPE (ABO)
  - **표 4 (신규, 선택)**: 자체 실측 20–30 Volume MAPE (가전·운동기구 포함)
- **교차 참조**: 4.3(Gaussian-only 채택)·4.4(OBB 채택)·4.5(절대 부피 변환) 본문에 "표 2의 Baseline-B·Baseline-A 비교에서 유의성 확보" 형식 인용.

### 8.2 6장 결론 보강

한 문장 추가: "복원 품질은 Pix3D·ABO로, 절대 부피 정합성은 ABO와 자체 실측 세트로 분리 검증하여 Gaussian-only OBB 채택이 정확도 손실 없이 속도·VRAM만 개선함을 실증하였다."

---

## 9. 리스크 및 완화책

| 리스크 | 완화 |
|---|---|
| ABO `item_dimensions` 단위 불일치(inch/cm 혼재) | `normalized_value.unit` 기준 일괄 정규화, 이상치 IQR 기반 제외 |
| KB↔ABO 매핑 모호성(예: DISH_CABINET) | 수작업 검수 1회 + 부정확 매핑 샘플 제외. 재현을 위해 매핑 테이블 JSON 커밋 |
| 카테고리당 샘플 <25 | 해당 카테고리를 ABO 트랙에서 빼고 자체 실측으로 흡수 |
| Baseline-B(Full) VRAM OOM | Pix3D 환경과 동일하게 L4 22GB 내 수행 가능성 기확인(EXPERIMENT_BENCHMARK.md). OOM 시 batch=1로 강제 |
| Volume MAPE 합격선 미달 | 실패 원인 breakdown(OBB 축 정렬 실패 vs KB 매칭 스케일 팩터 오차) 보고로 전환 — 한계 인정이 더 학술적 |
| 자체 실측 촬영 편향(정면 편중) | 카메라 각도 2조건(정면·사선) 촬영 로그 남김, 평균 값 보고 |

---

## 10. 참조

- 논문 초안: `docs/paper/Isajjim-framework.pdf`
- 기존 SAM-3D 벤치마크: `docs/paper/EXPERIMENT_BENCHMARK.md`
- 기존 seed variance: `docs/paper/EXPERIMENT_SEED_VARIANCE.md`
- Knowledge Base: `ai/data/knowledge_base.py`
- 절대 부피 계산기: `ai/processors/8_absolute_volume_calculate.py`
- 3D Worker 설정: `ai/subprocess/persistent_3d_worker.py`
- Ablation 실행기: `experiments/benchmark/scripts/workers/worker_ablation.py`
- 데이터셋 논문: Collins et al., "ABO: Dataset and Benchmarks for Real-World 3D Object Understanding," CVPR 2022.
