# ABO 기반 절대 부피 정확도 평가 실험 계획서

> 논문 `docs/paper/Isajjim-framework.pdf` 실험 섹션(5장) 보강을 위한 추가 실험 계획.
> 기존 `EXPERIMENT_BENCHMARK.md`(SAM-3D 3종 벤치마크)와 상보적 역할이며, 본 문서는
> **최종 파이프라인(Proposed)의 산출 부피가 실세계 metric 치수에 얼마나 맞는가**를 검증한다.

---

## 1. 목적

현 논문 5장은 **추론 시간·VRAM**만 보고한다. 프레임워크의 핵심 주장("데이터 기반 객관적 절대 부피 산출")을 뒷받침하는 **정확도 지표가 전무**하며, 이는 심사에서 반드시 지적될 공백이다.

본 실험(포스터 버전)은 다음 세 질문에 동시에 답한다.

| 질문 | 대응 증거 |
|---|---|
| Q1. 제안 파이프라인이 실제 가구 치수와 얼마나 일치하는가? | ABO GT 기반 Volume MAPE, Per-axis Absolute MAPE |
| Q2. KB 기반 절대 부피 매칭이 충분한가, 한계가 얼마나 드러나는가? | **Absolute Volume MAPE ↔ Relative Dimension MAPE gap** |
| Q3. Gaussian-only 모드가 Full mesh 대비 정확도 손실 없이 속도·VRAM만 개선하는가? | Baseline-B (Gaussian+Mesh) 비교 |

이로써 **논문 4.3(SAM-3D Gaussian-only), 4.5(절대 부피 변환)** 의 설계 선택이 정량 증거로 방어되며, 4.4(OBB)는 Relative Dimension MAPE의 절대값(충분히 낮으면 OBB 품질이 실증됨)으로 본문 논증된다.

> **포스터 지면 고려**: OBB vs AABB 비교(Baseline-A)는 PCA 기반 OBB의 기하학적 우위가 textbook 수준이고, Relative Dimension MAPE 자체가 OBB 정렬 품질의 직접 지표이므로 별도 ablation 없이 본문 논증한다. 표 1개 절약.

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

### 3.2 채택: **ABO (Amazon Berkeley Objects)** — 단일 공개 데이터셋 전략 (포스터 버전)

포스터 지면 제약상 데이터셋을 **ABO 단일**로 단순화한다. 자체 실측 보조 세트는 향후 저널 확장 버전에서 추가한다.

| 역할 | 소스 | 대상 카테고리 | 목적 |
|---|---|---|---|
| **메인 (정확도 평가)** | ABO | 가구 17종 ∩ ABO 제품 | Volume/Relative/Per-axis MAPE·nCD 정량 평가 |
| **유지 (profiling)** | Pix3D 500 | 현행 9종 | 기존 표 1(시간·VRAM) 연속성 |

ABO 채택 근거
1. `item_dimensions` field에 height/length/width가 metric으로 명시 → Volume GT 직접 산출.
2. product_type이 KB 가구 17종과 높은 겹침(소파·의자·침대·식탁·책상·서랍장 등).
3. catalog 이미지(배경 clean) + 일부 lifestyle shot 모두 존재 → YOLOE-seg 통과 가능.
4. 3D mesh 동봉 → nCD 측정 유지.
5. CC-BY-4.0, 공개 재현 용이.

ABO의 한계 (논문 "한계 및 향후 과제"로 명시)
- catalog shot 중심 → in-the-wild 도메인 갭 존재.
- 가전·운동기구 카테고리 얕음 → KB 27종 중 ABO 미커버 블록은 본 평가 범위 외.
- 해당 블록에 대한 in-the-wild 검증은 저널 확장 버전의 자체 실측 세트로 다룬다.

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

실제 카테고리·샘플 수는 Phase 1(데이터 준비) 완료 후 확정하여 본 문서를 업데이트한다. 카테고리당 최소 샘플이 25 미만이면 해당 카테고리는 ABO 트랙에서 제외한다 (본 실험 범위 밖으로 명시).

### 4.2 Pix3D (유지, profiling 용도)

- 기존 500 샘플 구성 그대로(`EXPERIMENT_BENCHMARK.md` §데이터셋 참조)
- 용도: 논문 표 1(평균 추론 시간·VRAM Peak) 연속성 확보
- 정확도 평가에는 사용하지 않음

---

## 5. 평가 지표

### 5.1 지표 정의

| 지표 | 정의 | 방어 대상 |
|---|---|---|
| **Volume MAPE(%)** | `\|V_pred − V_GT\| / V_GT`, V = W·D·H. mm³ 단위 절대 부피 | 논문 핵심 주장 (절대 부피, 4.5절) |
| **Relative Dimension MAPE(%)** | 상대 치수 비율 `(w_rel, d_rel, h_rel)` 정확도. 최장축=1로 정규화 후 크기순(L≥M≥S) 매칭. **KB 매칭과 독립** | 4.3절 형상 복원 + 4.4절 OBB (KB 스케일 팩터 영향 배제) |
| **Per-axis Absolute MAPE(%)** (W, D, H) | 각 축 절대값(mm) `\|pred − GT\| / GT`. 크기순 정렬 매칭 | 4.4절 + 4.5절 결합 |
| **nCD** (normalized Chamfer) | 예측·GT 양측 모두 **GT OBB diagonal** 로 정규화 후 CD | 4.3절 Gaussian-only의 형상 왜곡 검출 |
| **Success rate(%)** | YOLOE-seg 검출 성공 · SAM-3D 산출 성공 비율 | 파이프라인 안정성 |

#### 핵심 분리: Absolute ↔ Relative gap = KB 매칭 기여도

Volume MAPE는 **형상 복원 + OBB 정렬 + KB 스케일 매칭** 이 모두 섞인 복합 지표이다. Relative Dimension MAPE는 **KB 매칭을 제거한** KB-독립 지표이므로, 두 지표의 **gap** 이 KB 매칭(4.5절)의 기여도를 정량화한다.

| 관측 패턴 | 해석 | 대응되는 논문 섹션 |
|---|---|---|
| Volume MAPE ≈ Relative MAPE | KB 매칭이 잘 작동. 남은 오차는 형상·OBB에서 기인 | 4.3, 4.4 건전성 |
| Volume MAPE ≫ Relative MAPE | 상대 비율은 정확하나 절대 스케일 팩터가 부정확 → **KB 매칭 한계** | **4.5절 한계, 결론의 향후 과제와 직접 정합** |
| 양측 모두 큼 | OBB 축 정렬 또는 Gaussian 형상 복원 자체 문제 | 4.3 / 4.4 재검토 |

이 gap 서사는 본 실험의 **메인 내러티브**로 채택하며, 논문 5장 본문과 6장 결론의 교량 역할을 한다 (§8.1 참조).

#### 5.1.1 KB 매칭 특성 (중요 주석)

`ai/processors/8_absolute_volume_calculate.py` 의 `find_best_match` 는 **상대 치수의 l2/l3 비율** 한 가지만으로 서브타입을 선택하며, `height != -1` 타입(대다수 KB 엔트리)에서는 **선택된 서브타입의 KB 표준 치수를 그대로 반환**한다. 즉:

- Proposed ↔ Baseline-B 간 l2/l3 비율이 **같은 subtype 을 선택하면** pred 절대 치수가 동일 → Volume MAPE 동일.
- 두 설정의 차이는 **서브타입 전환 경계 근처에서만** 드러난다.
- 따라서 **Subtype Agreement Rate** (Proposed 와 Baseline-B 가 같은 샘플에서 같은 서브타입을 선택한 비율) 를 보조 지표로 기록하여 이 특성을 명시한다.

이는 버그가 아니라 KB 매칭 방식의 구조적 성질이며, **gap 해석에 영향**:
- gap 이 작다 ≠ KB 가 잘 작동 (단순히 서브타입 전환이 적었을 수 있음)
- gap 이 크다 ≠ KB 한계 (서브타입 전환 + 표준 치수와 GT 의 거리 둘 다 원인 가능)

논문 본문에는 gap 해석 시 이 제한점을 각주로 명시한다.

### 5.2 통계 처리

- 전체 집계(**표 2**): **평균 ± 표준편차, 중앙값** (포스터 보고 기준).
- 카테고리별 breakdown(**표 3**): 카테고리별 Volume MAPE + Relative MAPE + gap. 카테고리별 KB 매칭 한계 편차를 드러냄.
- **Absolute ↔ Relative gap**: 샘플별 `Volume MAPE − Relative Dimension MAPE` 를 별도 집계하여 KB 매칭 기여도를 정량 보고 (표 2 마지막 행 + 표 3 카테고리별 열).
- 설정 간 유의성 검정(Wilcoxon signed-rank)은 저널 확장 시 추가. 포스터에서는 평균±std로 충분.
- 보고 수준 (Phase 1 실측 후 재보정):
  - Volume MAPE **<15%** 잠정 합격선
  - Relative Dimension MAPE **<10%** (KB 독립이므로 절대 지표보다 tight)
  - Per-axis Absolute MAPE **<12%**

### 5.3 단계별 오차 분해 (내부 분석용, 표 제외)

Volume MAPE가 합격선을 초과하는 실패 샘플에 대해 오차 원인이 어느 파이프라인 단계에서 발생했는지 정량적으로 분리한다. §5.1의 Absolute↔Relative gap 서사를 **로그-가법 분해** 로 확장한 형태이며, **포스터에는 표로 올리지 않고** 본문 한 문장("실패 케이스의 X%가 스케일 단계 주원인")으로만 요약한다.

#### 5.3.1 부피 오차의 로그-가법 분해

절대 부피는 스케일 팩터와 상대 비율의 곱으로 표현된다.

```
V_pred = k_pred × (w_rel_pred · d_rel_pred · h_rel_pred)
V_GT   = k_GT   × (w_rel_GT  · d_rel_GT  · h_rel_GT)
```

- `w_rel, d_rel, h_rel`: OBB로부터 산출한 상대 치수. 최장축을 1로 정규화.
- `k`: mm 단위 스케일 팩터. Proposed에서는 KB 표준 치수 매칭으로 결정.

로그 공간에서 부피 오차는 **스케일 성분** 과 **비율 성분** 으로 가법 분해된다.

```
log(V_pred / V_GT) = log(k_pred / k_GT) + Σᵢ log(r_pred,ᵢ / r_GT,ᵢ),   i ∈ {w, d, h}
```

이를 통해 각 샘플의 부피 오차에서 **KB 스케일 성분** 과 **OBB 비율 성분** 의 기여도를 개별 측정한다. §5.1의 gap 관측을 샘플 단위로 정량화한 확장판이다.

#### 5.3.2 단계별 지표

| 단계 | 오차 원인 | 지표 | 측정 소스 |
|---|---|---|---|
| (a) 검출 | miss / class 오분류 | Detection Rate, Class Accuracy (base_name 일치율) | YOLOE 출력 로그 |
| (b) 형상 복원 | Gaussian/Mesh 형상 왜곡 | nCD (GT OBB diagonal 정규화) | SAM-3D PLY ↔ ABO GT mesh |
| (c) 축 비율 | OBB 축 정렬 실패 | Ratio MAPE = meanᵢ \|r_pred,ᵢ − r_GT,ᵢ\| / r_GT,ᵢ (= Relative Dimension MAPE, §5.1) | OBB 치수 ↔ GT 치수 |
| (d) 절대 스케일 | KB 매칭 / 표준 치수 한계 | Scale MAPE = \|k_pred − k_GT\| / k_GT, 여기서 k_GT = V_GT / (w_rel_pred · d_rel_pred · h_rel_pred) | V_pred, V_GT |

**해석 규칙 (실패 원인 귀속)**
- (a)에서 실패한 샘플은 이후 단계 측정 불가 → 별도 집계.
- (b) nCD가 `EXPERIMENT_BENCHMARK.md` 벤치마크 대비 악화 → SAM-3D 복원 품질 저하.
- **(c) 낮음 & (d) 높음** → 상대 비율은 정확하나 절대 스케일이 부정확 → **4.5절 KB 매칭 한계** (결론의 향후 과제와 정합, §5.1 Absolute≫Relative 패턴과 등가).
- **(c) 높음 & (d) 낮음** → **4.4절 OBB 한계** (주축 정렬 실패).
- (c)(d) 모두 높음 → 복합 요인. 카테고리별 추가 분석.

이 매핑으로 **실패 케이스의 주원인 분포(%)** 를 표로 보고할 수 있으며, 논문 결론 섹션의 "향후 연구는 KB 매칭 고도화에 집중한다" 주장을 데이터로 받친다.

---

## 6. 실험 설계

### 6.1 2 설정 Ablation (포스터 버전)

| 설정 | 3D 복원 | 치수 산출 | 분리하려는 기여 |
|---|---|---|---|
| Baseline-B | **Gaussian + Mesh** (steps 14/4, postprocess·texture_baking 활성) | OBB | 메시 디코드 생략의 정확도 영향 (4.3절) |
| **Proposed** | Gaussian-only (steps 14/4) | OBB | 제안 구성 |

**변수 분리 원칙**: Proposed ↔ Baseline-B는 **복원 디코드 경로만 토글** (Gaussian-only → Gaussian+Mesh+Postprocess+Texture_baking). 추론 스텝 수(14/4)는 동일하게 고정하여 단일 요인 비교를 보장한다. 상세 근거는 §6.2.1.

4.4절(OBB)의 방어는 Baseline-A(AABB) 비교 대신 **Relative Dimension MAPE 절대값**(§5.1)으로 간접 논증한다: 해당 지표가 합격선(<10%) 이하라면 OBB 기반 상대 치수 산출이 충분한 정확도로 작동함이 입증된다.

### 6.2 설정별 SAM-3D 파라미터

`ai/subprocess/persistent_3d_worker.py` 기준.

| 파라미터 | Baseline-B | Proposed |
|---|---|---|
| `STAGE1_INFERENCE_STEPS` | **14** | 14 |
| `STAGE2_INFERENCE_STEPS` | **4** | 4 |
| `decode_formats` | `["gaussian","mesh"]` | `["gaussian"]` |
| `mesh_postprocess` | True | False |
| `texture_baking` | True | False |
| `GAUSSIAN_ONLY_MODE` | False | True |
| 치수 산출 | OBB | OBB |

#### 6.2.1 추론 스텝 수를 두 설정 동일하게 고정한 이유

논문 4.3절이 방어하려는 명제는 **"메시 디코드 및 후처리 생략의 효과"** 이다. Baseline-B에 SAM-3D 기본값(25/25)을 적용하면, Proposed 대비 정확도 차이가 **(i) 메시 디코드 경로 활성화** 때문인지 **(ii) 추론 스텝 수 증가** 때문인지 분리되지 않는다. 따라서 본 ablation은 **한 번에 하나의 축만 변경**한다는 원칙을 따른다.

- Proposed ↔ Baseline-B: 복원 디코드 경로만 토글 (Gaussian-only → Gaussian+Mesh+Postprocess+Texture_baking). 스텝 14/4 동일.

이 설계로 **단일 쌍 비교(B vs Proposed)** 가 논문 4.3절을 혼입 없이 방어한다.

#### 6.2.2 추론 스텝 수 질의 대응

"Proposed(14/4) 설정이 SAM-3D 기본값(25/25) 대비 정확도 손실이 미미하다"는 명제는 **본 ablation의 방어 대상이 아니다** (별도 축). 리뷰어 질의 시 기존 `experiments/benchmark/results/ablation_*.csv` 를 각주로 인용한다. 포스터 버전에서는 별도 측정을 수행하지 않는다.

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

### Phase 2 — 스크립트 확장

| 변경 | 위치 |
|---|---|
| ABO loader 추가 | `experiments/benchmark/scripts/evaluate/abo_loader.py` (신규) |
| Volume / Relative / Per-axis MAPE 계산 | `experiments/benchmark/scripts/evaluate/compute_accuracy.py` (신규) — §5.1 지표 동시 산출, **카테고리별 집계 포함** |
| nCD 정규화 통일 (**GT OBB diagonal**) | `experiments/benchmark/scripts/evaluate/compute_cd_summary.py` 수정 |
| Baseline-B(Gaussian+Mesh) 분기 | worker에 `GAUSSIAN_ONLY_MODE=False` 전달 (**스텝은 14/4 유지**, §6.2.1) |
| **Absolute ↔ Relative gap 집계** | `compute_accuracy.py` 서브 모듈. 전체·카테고리별 gap 산출 |
| 단계별 오차 분해 계산 (내부 분석) | `experiments/benchmark/scripts/evaluate/compute_error_breakdown.py` (신규) — §5.3. 본문 요약 한 문장용 수치 산출 |
| 실행 스크립트 | `experiments/benchmark/run/run_abo_accuracy.sh` (신규) |

### Phase 3 — 실험 실행

| 설정 | 샘플 | 예상 시간 | 비고 |
|---|---|---|---|
| Proposed | 500 | ~50분 | 기존 최적화 설정 |
| Baseline-B (Gaussian+Mesh) | 500 | ~75–90분 | 스텝 14/4 유지, mesh decode + postprocess + texture_baking 추가 |

총 예상 실행 시간: **약 2–2.5시간** (L4 단일 GPU 기준).

### Phase 4 — 분석 및 논문 반영

| 산출물 | 경로 | 사용처 |
|---|---|---|
| 전체 정확도 집계 | `experiments/benchmark/results/abo_accuracy_summary.csv` — Volume / Relative / Per-axis MAPE, nCD, Success rate × 2 설정 + **gap 행** | **표 2** |
| 카테고리별 집계 | `experiments/benchmark/results/abo_accuracy_by_category.csv` — 카테고리별 Volume MAPE / Relative MAPE / gap | **표 3** |
| 단계별 오차 분해 (내부) | `experiments/benchmark/results/abo_error_breakdown.csv` | 본문 요약 1문장 |
| 실패 케이스 시각화 (내부) | `experiments/benchmark/results/abo_failure_cases/` — 주원인 단계별 폴더 | 리뷰어 질의 대응용, 포스터 비게재 |

---

## 8. 논문 반영 방안

### 8.1 5장 개편(안) — 포스터 버전

- **5.1 실험 환경** — 두 문단으로 분리
  1. 시간·VRAM profiling: Pix3D 500 (현행 유지).
  2. 정확도 평가: ABO 500 층화, 2 설정 ablation (Proposed, Baseline-B).
- **5.2 실험 결과** — 표 구성
  - 표 1 (유지): 평균 추론 시간·VRAM Peak (Pix3D)
  - **표 2 (신규)**: 전체 정확도 집계 — Proposed vs Baseline-B × {Volume MAPE, Relative Dimension MAPE, Per-axis Absolute MAPE, nCD, Success rate, **Absolute−Relative gap**}
  - **표 3 (신규)**: 카테고리별 Volume MAPE / Relative MAPE / gap (ABO, Proposed 설정) — KB 매칭 한계의 카테고리 편차를 노출
- **본문 보강 (표 없이)**:
  - OBB 채택(4.4절) 방어: 표 2의 Relative Dimension MAPE 절대값이 <10% 합격선임을 본문 수치 인용
  - 실패 원인(§5.3): 실패 샘플의 주원인 단계 분포를 한 문장으로 요약 (예: "Volume MAPE 합격선 초과 샘플의 X%가 KB 스케일 단계 주원인")
- **교차 참조**:
  - 4.3(Gaussian-only) → **표 2의 Proposed vs Baseline-B 비교** (스텝 고정, 디코드 경로만 토글)
  - 4.4(OBB) → 표 2의 **Relative Dimension MAPE 절대값** 본문 인용
  - 4.5(절대 부피 변환) → **표 2 gap 행 + 표 3 카테고리별 gap** → KB 매칭 한계의 정량 근거

### 8.2 6장 결론 보강

두 문장 추가 권장.

1. "제안 프레임워크를 ABO 500 샘플에서 검증한 결과, Gaussian-only + OBB 구성이 Full SAM-3D 대비 Volume MAPE [x]%의 차이만을 보여 정확도 손실 없이 속도·VRAM을 개선함을 실증하였다."
2. "특히 Absolute Volume MAPE와 Relative Dimension MAPE의 gap이 [y]%로 관측되어, 남은 오차의 주요 원인이 **정적 KB 기반 스케일 매칭의 한계**에 있음을 정량적으로 확인하였다. 이는 기존 결론의 '딥러닝 기반 절대 크기 추정 모델' 제안과 정합하며, 향후 연구의 우선순위 근거가 된다."

---

## 9. 리스크 및 완화책

| 리스크 | 완화 |
|---|---|
| ABO `item_dimensions` 단위 불일치(inch/cm 혼재) | `normalized_value.unit` 기준 일괄 정규화, 이상치 IQR 기반 제외 |
| KB↔ABO 매핑 모호성(예: DISH_CABINET) | 수작업 검수 1회 + 부정확 매핑 샘플 제외. 재현을 위해 매핑 테이블 JSON 커밋 |
| 카테고리당 샘플 <25 | 해당 카테고리를 본 평가 범위에서 제외, 논문 한계 섹션 명시 |
| Baseline-B VRAM OOM | Pix3D 환경과 동일하게 L4 22GB 내 수행 가능성 기확인(EXPERIMENT_BENCHMARK.md). OOM 시 batch=1로 강제 |
| Volume MAPE 합격선 미달 | 실패 원인 breakdown(OBB 축 정렬 실패 vs KB 매칭 스케일 팩터 오차) 보고로 전환 — 한계 인정이 더 학술적 |
| ABO catalog 도메인 갭 | in-the-wild 검증은 저널 확장 버전의 과제로 명시, 포스터 한계 섹션에 1줄 언급 |

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
