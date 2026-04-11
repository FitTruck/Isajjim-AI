# 실험 계획: SAM-3D Seed Variance 측정

## 목적

최적화 전후 상대 치수 편차를 판단할 **허용 오차 기준(δ)**을 확립한다.
SAM-3D는 diffusion model이므로 같은 입력이라도 seed에 따라 출력이 다르다.
이 **내재적 확률 변동(inherent stochastic variance)**을 측정하여,
"최적화 편차가 이 변동 이내이면 모델 자체 노이즈 수준"이라고 판정한다.

---

## 실험 설계

### 독립 변수

- **seed**: 1, 2, 3, ..., K (K=10 권장)

### 통제 변수 (고정)

| 항목 | 값 | 비고 |
|------|-----|------|
| 모델 | Original SAM-3D | compile=False, 모든 decoder 활성 |
| `stage1_inference_steps` | 25 (기본값) | |
| `stage2_inference_steps` | 25 (기본값) | |
| `decode_formats` | `["gaussian"]` | Gaussian만 사용해도 OBB 계산 가능 |
| `with_mesh_postprocess` | True (기본값) | |
| `with_texture_baking` | True (기본값) | |
| `with_layout_postprocess` | False (기본값) | |
| `pointmap` | None (MoGe 사용) | Original 기본 동작 |
| 이미지 | 고정 (동일 이미지) | |
| 마스크 | 고정 (동일 마스크) | |

### 종속 변수 (측정)

- 각 실행의 Gaussian Splat PLY → PCA OBB → **상대 치수 (w, d, h)**
- K번 실행 결과에서:
  - **평균** μ_w, μ_d, μ_h
  - **표준편차** σ_w, σ_d, σ_h
  - **변동 계수** CV_w = σ_w / μ_w × 100%, CV_d, CV_h

---

## 테스트 이미지 선정

### 기준

1. **가구 카테고리 다양성**: 일반 형태(sofa, bed) + 극박 형태(TV) + 소형(nightstand) 포함
2. **마스크 품질**: YOLOE-seg로 탐지 가능한 명확한 객체
3. **이삿짐 서비스 대표성**: 실제 견적 이미지와 유사한 조건 (실내, 자연광, 부분 가려짐)

### 데이터셋 규모

**최소 50개 객체, 8개 이상 카테고리** 확보하여 통계적 신뢰성 확보.

| 카테고리 | 목표 수량 | 특성 | 비고 |
|---------|----------|------|------|
| Bed (침대) | 6-8장 | 대형, 납작 (h << w, d) | single/double/queen 다양하게 |
| Sofa (소파) | 6-8장 | 중형, 비대칭 (등받이) | 1인/2인/3인 포함 |
| Chair (의자) | 6-8장 | 소형, 다리 구조 | 사무용/식탁용/안락의자 |
| Table (테이블) | 6-8장 | 중형, 빈 공간 (다리 사이) | 식탁/책상/커피테이블 |
| Nightstand (협탁) | 4-6장 | 소형, 정육면체에 가까움 | |
| Television (TV) | 4-6장 | 극박 (depth ≈ 0.02) | 높은 variance 예상 |
| Bookshelf (책장) | 4-6장 | 세로로 긴 형태 | |
| Dresser/Wardrobe (서랍장/옷장) | 4-6장 | 대형, 직육면체 | |
| **합계** | **~50장** | **8 카테고리** | |

### 이미지 소스

| 소스 | 수량 | 장점 | 단점 |
|------|------|------|------|
| **이삿짐 서비스 실제 이미지** | 20-30장 | 실서비스 context, practical impact | 비공개, 재현 불가 |
| **공개 데이터셋 (Pix3D/ABO)** | 20-30장 | 재현 가능, 심사자 검증 가능 | 이삿짐 context 약함 |
| **합계** | **~50장** | hybrid로 양쪽 장점 | |

> **Pix3D** (pix3d.csail.mit.edu): chair, bed, desk, sofa, table, bookcase, wardrobe — 가구 7개 카테고리, 실제 이미지 + 정확한 3D 모델 pair
>
> **ABO** (Amazon Berkeley Objects): 가구 카테고리 풍부, 상품 사양에 실측 치수 포함

---

## 실행 절차

### Phase 1: Seed Variance 측정 (Original SAM-3D, 250회)

```
for each image_i in test_images:           # 50 images
    for seed_k in [1, 2, 3, 4, 5]:         # 5 seeds
        output = original_sam3d.run(
            image=image_i,
            mask=mask_i,
            seed=seed_k,
            stage1_inference_steps=25,   # 기본값
            stage2_inference_steps=25,   # 기본값
            decode_formats=["gaussian"],
            with_mesh_postprocess=True,  # 기본값
            with_texture_baking=True,    # 기본값
            with_layout_postprocess=False,  # 기본값
            pointmap=None,               # MoGe 사용 (기본)
        )
        ply = extract_gaussian_ply(output)
        obb = calculate_pca_obb(ply)
        record(image_i, seed_k, obb.width, obb.depth, obb.height)
```

### Phase 2: Optimization Deviation 측정 (Optimized SAM-3D, 50회)

```
for each image_i in test_images:           # 50 images
    # seed 고정 (예: seed=42)
    output = optimized_sam3d.run(
        image=image_i,
        mask=mask_i,
        seed=42,
        stage1_inference_steps=14,         # 최적화: 25→14
        stage2_inference_steps=4,          # 최적화: 25→4
        decode_formats=["gaussian"],       # Gaussian-only
        with_mesh_postprocess=False,       # 최적화: 비활성화
        with_texture_baking=False,         # 최적화: 비활성화
        with_layout_postprocess=False,     # 기본값 (변경 없음)
        pointmap=synthetic_pointmap,       # 최적화: MoGe→synthetic
        # + SS Step Caching (stride=3, warmup=2) 활성화
    )
    ply = extract_gaussian_ply(output)
    obb = calculate_pca_obb(ply)
    record_optimized(image_i, obb.width, obb.depth, obb.height)
```

### Phase 3: 비교

```
for each image_i:
    # Seed variance (Phase 1)
    μ_w = mean(phase1_results[image_i].widths)
    σ_w = std(phase1_results[image_i].widths)
    CV_w = σ_w / μ_w × 100%

    # Optimization deviation (Phase 2 vs Phase 1 mean)
    dev_w = |phase2_result[image_i].width - μ_w| / μ_w × 100%

    # 판정
    if dev_w <= CV_w:
        판정 = "이내 (seed 변동 수준)"
    else:
        판정 = "초과"
```

---

## 결과 기록 양식

### Table A: Raw 측정값 (Phase 1)

| Image | Seed | W | D | H |
|-------|------|-----|-----|-----|
| Nightstand | 1 | ? | ? | ? |
| Nightstand | 2 | ? | ? | ? |
| ... | ... | ... | ... | ... |
| Nightstand | 10 | ? | ? | ? |
| Bed | 1 | ? | ? | ? |
| ... | ... | ... | ... | ... |

### Table B: Seed Variance 요약

| Image | W μ | W σ | W CV(%) | D μ | D σ | D CV(%) | H μ | H σ | H CV(%) |
|-------|-----|-----|---------|-----|-----|---------|-----|-----|---------|
| Nightstand | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Bed | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Television | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Sofa | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Dining Table | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| **평균** | | | **δ_w** | | | **δ_d** | | | **δ_h** |

### Table C: Optimization Deviation vs Seed Variance

| Image | Axis | Seed CV (δ) | Optim Dev | δ 이내? | 비고 |
|-------|------|-------------|-----------|---------|------|
| Nightstand | W | ?% | ?% | ? | |
| Nightstand | D | ?% | ?% | ? | |
| Nightstand | H | ?% | ?% | ? | |
| Bed | W | ?% | ?% | ? | |
| Bed | D | ?% | ?% | ? | |
| Bed | H | ?% | ?% | ? | |
| TV | W | ?% | ?% | ? | |
| TV | D | ?% | ?% | ? | |
| TV | H | ?% | ?% | ? | 극박 객체 — CV 자체가 높을 것 |
| **전체** | **Avg** | **δ_avg** | **dev_avg** | ? | |

---

## 허용 오차 기준 확정 방법

### 기준 정의

```
δ = max(δ_w, δ_d, δ_h)  (각 축 CV의 최대값)

또는 보수적으로:
δ = mean(CV_w, CV_d, CV_h) + 1σ
```

### 논문에서의 기술

> "허용 오차 δ는 Original SAM-3D를 동일 입력에 대해 seed만 달리하여 K=10회 실행한
> 결과의 OBB 상대 치수 변동 계수(CV)로 정의한다. 측정 결과 δ = X.X%였으며,
> 본 최적화로 인한 치수 편차 Y.Y%는 이 기준 이내로, 최적화 효과가
> 모델의 내재적 확률 변동 수준에 있음을 확인하였다."

---

## 예상 결과 (가설)

기존 코드의 실험 데이터(Nightstand, Bed, TV)에서 관찰된 패턴:

| 객체 | 예상 Seed CV | 예상 Optim Dev | 근거 |
|------|-------------|---------------|------|
| Nightstand | 2-4% | 2.3% (W) | 일반 형태, moderate variance |
| Bed | 1-3% | 0.3% (H) | 대형 납작 객체, 안정적 |
| Television | 3-6% | *9.4% (H) | 극박(depth≈0.02), 자연 변동 4.3% |

TV H축은 seed variance도 높을 것으로 예상:
- 극박 객체의 "높이" 축은 noise에 민감
- Optim Dev 9.4%가 이 축의 seed CV보다 높을 수 있음
- 이 경우 "극박 객체의 H축은 모델 inherent limitation"으로 논의

---

## 추가 실험 (Optional)

### Ablation: 어느 최적화가 치수에 가장 영향을 주는가

각 최적화를 개별적으로 켜면서 seed=42 고정으로 Original 대비 편차 측정:

| Configuration | vs Original (seed=42) |
|--------------|----------------------|
| Gaussian-Only만 적용 | W? D? H? |
| + VRAM Unload | W? D? H? (= 0%, dead-code) |
| + Synthetic Pointmap만 | W? D? H? |
| + Steps 14/4만 | W? D? H? |
| + SS Caching만 | W? D? H? |
| 전부 적용 (Final) | W? D? H? |

이 ablation에서 "치수에 영향을 주는 것은 Synthetic Pointmap, Steps, Caching 3가지뿐"을
데이터로 입증할 수 있음.

### Seed 고정 vs 변동: Optimized SAM-3D의 seed variance도 측정

Optimized SAM-3D도 seed에 따라 변동이 있을 수 있음.
Original과 Optimized 양쪽의 seed variance를 비교하면:
- Optimized의 CV가 Original과 비슷하면: "최적화가 모델 안정성에 영향 없음"
- Optimized의 CV가 더 높으면: "최적화가 출력 분산을 키움" → 논의 필요

---

## 필요 리소스

| 항목 | 수량 | 비고 |
|------|------|------|
| GPU | L4 4대 (또는 A100 1-2대) | SAM-3D full model 로드 (~21GB/GPU) |
| 테스트 이미지 | **50장** (8 카테고리) | YOLOE-seg 마스크 포함 |
| Phase 1 실행 횟수 | 50 images × 5 seeds = **250회** | Original SAM-3D |
| Phase 2 실행 횟수 | 50 images × 1 seed = **50회** | Optimized SAM-3D |
| 예상 시간 (Phase 1, 단일 GPU) | 250 × ~150초 = **~10.4시간** | Original은 느림 |
| 예상 시간 (Phase 1, 4× GPU 병렬) | **~2.5시간** | 배치 병렬 실행 |
| 예상 시간 (Phase 2) | 50 × ~13초 = **~11분** | Optimized는 빠름 |
| **총 예상 시간 (4× GPU)** | **~3시간** | Phase 1이 지배적 |

> Phase 1이 느리므로 **밤에 배치로 돌리는 것** 권장.
> 4 GPU 병렬이면 ~3시간이면 끝남. 단일 GPU면 ~11시간.
>
> **통계적 신뢰성**: 50개 객체 × 5 seeds = 250개 측정점.
> 카테고리당 평균 6개 객체 → 카테고리별 분산도 보고 가능.
> 논문에서 "8개 가구 카테고리, 50개 객체에 대해 측정" 기술 가능.

---

## 실행 스크립트 (TODO)

```python
# TODO: 실제 실행 스크립트 작성
# experiments/seed_variance.py
#
# 1. Original SAM-3D 로드 (compile=False, full decoder)
# 2. 테스트 이미지 50장 + 마스크 로드
# 3. seed 1~5로 각각 실행 → PLY 저장 (250회)
# 4. PLY → PCA OBB → (w, d, h) 기록
# 5. CSV 출력
#
# 실행 방법:
#   python experiments/seed_variance.py \
#     --image-dir datasets/furniture_50/ \
#     --mask-dir datasets/furniture_50_masks/ \
#     --seeds 5 \
#     --output results/seed_variance.csv \
#     --gpu-ids 0,1,2,3
```

---

## 체크리스트

- [ ] **데이터셋 구축**: 50장 확보 (서비스 이미지 20-30장 + Pix3D/ABO 20-30장)
- [ ] 각 이미지에 대해 YOLOE-seg 마스크 생성
- [ ] GPU 서버에서 Original SAM-3D(25+25 steps, full decoder) 실행 가능 확인
- [ ] Phase 1 배치 실행 (250회, 4×GPU ~2.5시간)
- [ ] Phase 2 배치 실행 (50회, ~11분)
- [ ] Table B (Seed Variance 카테고리별) 작성 → δ 확정
- [ ] Table C (Optimization Deviation vs δ) 작성 → 판정
- [ ] 카테고리별 분석 (TV 등 극박 객체 별도 보고)
- [ ] 논문 Table 1/4에 반영
- [ ] (Optional) Ablation 실험
- [ ] (Optional) Optimized SAM-3D의 seed variance도 측정
