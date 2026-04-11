# 실험 계획: SAM-3D Seed Variance 측정

## 목적

최적화 전후 상대 치수 편차를 판단할 **허용 오차 기준(δ)**을 확립한다.
SAM-3D는 diffusion model이므로 같은 입력이라도 seed에 따라 출력이 다르다.
이 **내재적 확률 변동(inherent stochastic variance)**을 측정하여,
"최적화 편차가 이 변동 이내이면 모델 자체 노이즈 수준"이라고 판정한다.

---

## 데이터셋

### 소스: Pix3D (http://pix3d.csail.mit.edu/)

Pix3D를 메인 데이터셋으로 사용한다. 선정 이유:

| 장점 | 설명 |
|------|------|
| **이미지 + 마스크 쌍 제공** | pixel-level binary mask가 포함되어 YOLOE-seg 불필요 |
| **가구 카테고리 풍부** | bed, chair, sofa, table, desk, bookcase, wardrobe 7개 |
| **3D GT 모델 포함** | 추후 절대 치수 검증에도 활용 가능 (optional) |
| **공개 데이터셋** | 재현 가능, 심사자 검증 가능 |
| **학계 표준** | 3D 복원 논문에서 널리 사용 (CVPR 2018) |

### 데이터 구조

```
pix3d/
├── img/           ← RGB 이미지 (JPG/PNG)
│   ├── bed/
│   ├── chair/
│   ├── sofa/
│   ├── table/
│   ├── desk/
│   ├── bookcase/
│   └── wardrobe/
├── mask/          ← Binary mask (PNG, 0/255) — SAM-3D 입력으로 직접 사용
│   ├── bed/
│   ├── chair/
│   └── ...
├── model/         ← 3D .obj 파일 (optional, GT 검증용)
└── pix3d.json     ← 메타데이터 (카테고리, 이미지-마스크 pair 정보)
```

### 마스크 선택: Pix3D GT 마스크 사용

| 방법 | 설명 | 채택 |
|------|------|------|
| **Pix3D GT 마스크** | 데이터셋 제공 pixel-level mask | ✅ 채택 |
| YOLOE-seg 재생성 | 우리 파이프라인으로 마스크 생성 | ❌ |

**이유**: Seed variance 실험에서는 **마스크를 고정**해야 합니다 (seed만 바꾸는 게 목적).
Pix3D GT 마스크를 사용하면 마스크 품질이라는 변수를 완전히 제거하고,
순수하게 SAM-3D의 seed 변동만 격리할 수 있습니다.

### 샘플링 계획

Pix3D의 7개 가구 카테고리에서 **총 50개 객체**를 균등 샘플링:

| 카테고리 | Pix3D 보유 수 | 샘플 수 | 특성 |
|---------|-------------|---------|------|
| bed | 50+ | **8** | 대형, 납작 (h << w, d) |
| sofa | 100+ | **8** | 중형, 비대칭 (등받이) |
| chair | 1000+ | **8** | 소형, 다리 구조 (가장 다양) |
| table | 300+ | **8** | 중형, 빈 공간 (다리 사이) |
| desk | 200+ | **6** | table과 유사하나 서랍 등 차이 |
| bookcase | 50+ | **6** | 세로로 긴 형태 |
| wardrobe | 50+ | **6** | 대형, 직육면체 |
| **합계** | | **50** | **7 카테고리** |

**샘플링 기준**:
- 각 카테고리 내에서 **형태 다양성** 최대화 (같은 모양 반복 회피)
- 가려짐(occlusion)이 심하지 않은 이미지 우선 (SAM-3D 성공률 보장)
- 해상도 512×512 이상

> **Note**: Pix3D에는 TV(television) 카테고리가 없음.
> 극박 객체 테스트가 필요하면 이삿짐 서비스 이미지에서 TV 5장을 추가 확보하거나
> 별도 공개 이미지 사용. TV는 "보충 데이터"로 분리하여 보고.

---

## 실험 설계

### 독립 변수

- **seed**: 1, 2, 3, 4, 5 (K=5)

### 통제 변수 (고정)

| 항목 | 값 | 비고 |
|------|-----|------|
| 모델 | Original SAM-3D | compile=False, 모든 decoder 활성 |
| `stage1_inference_steps` | 25 (기본값) | |
| `stage2_inference_steps` | 25 (기본값) | |
| `decode_formats` | `["gaussian"]` | Gaussian만으로 OBB 계산 가능 |
| `with_mesh_postprocess` | True (기본값) | |
| `with_texture_baking` | True (기본값) | |
| `with_layout_postprocess` | False (기본값) | |
| `pointmap` | None (MoGe 사용) | Original 기본 동작 |
| 이미지 | 고정 (Pix3D 이미지) | |
| 마스크 | 고정 (**Pix3D GT mask**) | YOLOE 아닌 GT mask 사용 |

### 종속 변수 (측정)

- 각 실행의 Gaussian Splat PLY → PCA OBB → **상대 치수 (w, d, h)**
- K번 실행 결과에서:
  - **평균** μ_w, μ_d, μ_h
  - **표준편차** σ_w, σ_d, σ_h
  - **변동 계수** CV_w = σ_w / μ_w × 100%, CV_d, CV_h

---

## 실행 절차

### Phase 1: Seed Variance 측정 (Original SAM-3D, 250회)

```
for each image_i in pix3d_50:              # 50 images (Pix3D)
    mask_i = pix3d_gt_mask[image_i]        # Pix3D 제공 GT mask
    for seed_k in [1, 2, 3, 4, 5]:         # 5 seeds
        output = original_sam3d.run(
            image=image_i,
            mask=mask_i,                   # Pix3D GT mask (고정)
            seed=seed_k,
            stage1_inference_steps=25,     # 기본값
            stage2_inference_steps=25,     # 기본값
            decode_formats=["gaussian"],
            with_mesh_postprocess=True,    # 기본값
            with_texture_baking=True,      # 기본값
            with_layout_postprocess=False, # 기본값
            pointmap=None,                 # MoGe 사용 (기본)
        )
        ply = extract_gaussian_ply(output)
        obb = calculate_pca_obb(ply)
        record(image_i, seed_k, obb.width, obb.depth, obb.height)
```

### Phase 2: Optimization Deviation 측정 (Optimized SAM-3D, 50회)

```
for each image_i in pix3d_50:              # 50 images (동일)
    mask_i = pix3d_gt_mask[image_i]        # 동일 Pix3D GT mask
    output = optimized_sam3d.run(
        image=image_i,
        mask=mask_i,
        seed=42,                           # seed 고정
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

### Phase 3: 비교 및 판정

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

### Table A: Raw 측정값 (Phase 1, 250개 행)

| Category | Image | Seed | W | D | H |
|----------|-------|------|-----|-----|-----|
| bed | bed_001.jpg | 1 | ? | ? | ? |
| bed | bed_001.jpg | 2 | ? | ? | ? |
| ... | ... | ... | ... | ... | ... |
| bed | bed_001.jpg | 5 | ? | ? | ? |
| bed | bed_002.jpg | 1 | ? | ? | ? |
| ... | ... | ... | ... | ... | ... |

### Table B: 카테고리별 Seed Variance 요약

| Category | N | W CV(%) | D CV(%) | H CV(%) | 비고 |
|----------|---|---------|---------|---------|------|
| bed | 8 | ? | ? | ? | 납작 형태 |
| sofa | 8 | ? | ? | ? | 비대칭 |
| chair | 8 | ? | ? | ? | 다리 구조 |
| table | 8 | ? | ? | ? | 빈 공간 |
| desk | 6 | ? | ? | ? | |
| bookcase | 6 | ? | ? | ? | 세로 |
| wardrobe | 6 | ? | ? | ? | 직육면체 |
| **전체 평균** | **50** | **δ_w** | **δ_d** | **δ_h** | **허용 오차 기준** |

### Table C: Optimization Deviation vs Seed Variance (카테고리별)

| Category | N | Avg Seed CV (δ) | Avg Optim Dev | δ 이내 비율 | 비고 |
|----------|---|----------------|--------------|------------|------|
| bed | 8 | ?% | ?% | ?/24 축 | |
| sofa | 8 | ?% | ?% | ?/24 | |
| chair | 8 | ?% | ?% | ?/24 | |
| table | 8 | ?% | ?% | ?/24 | |
| desk | 6 | ?% | ?% | ?/18 | |
| bookcase | 6 | ?% | ?% | ?/18 | |
| wardrobe | 6 | ?% | ?% | ?/18 | |
| **전체** | **50** | **δ_avg** | **dev_avg** | **?/150 축** | |

> "δ 이내 비율": 50개 객체 × 3축(W,D,H) = 150개 측정 중 δ 이내인 비율.
> 논문에서 "150개 축 측정 중 X%가 seed variance 이내"로 기술.

---

## 허용 오차 기준 확정 방법

### 기준 정의

```
δ = 전체 50개 객체의 평균 CV (W, D, H 통합)

또는 보수적으로:
δ = mean(CV_all) + 1σ(CV_all)
```

### 논문에서의 기술

> "허용 오차 δ는 Original SAM-3D를 Pix3D 데이터셋의 50개 가구 객체(7개 카테고리)에 대해
> seed만 달리하여 K=5회 실행한 결과의 OBB 상대 치수 변동 계수(CV)로 정의한다.
> 측정 결과 전체 평균 CV는 δ = X.X%였으며, 본 최적화로 인한 평균 치수 편차 Y.Y%는
> 이 기준 이내로, 150개 축 측정 중 Z%가 seed variance 이내에 있음을 확인하였다.
> 이는 최적화 효과가 모델의 내재적 확률 변동 수준에 있음을 의미한다."

---

## 예상 결과 (가설)

기존 코드의 실험 데이터(Nightstand, Bed, TV)에서 관찰된 패턴 기반:

| 카테고리 | 예상 Seed CV | 예상 Optim Dev | 근거 |
|---------|-------------|---------------|------|
| bed | 1-3% | 0.1-0.3% | 대형 납작 객체, 안정적 |
| sofa | 2-4% | 1-2% | 비대칭 형태, moderate |
| chair | 3-5% | 2-3% | 다리 구조로 불안정 가능 |
| table | 2-4% | 1-2% | 빈 공간 있으나 형태 단순 |
| desk | 2-4% | 1-2% | table과 유사 |
| bookcase | 2-3% | 1-2% | 직육면체, 안정적 |
| wardrobe | 1-3% | 0.5-1% | 가장 단순한 형태 |

극박 객체(TV 등)는 Pix3D에 없으므로, 서비스 이미지로 보충 실험 시 별도 보고.

---

## 추가 실험 (Optional)

### Ablation: 어느 최적화가 치수에 가장 영향을 주는가

각 최적화를 개별적으로 적용하면서 seed=42 고정으로 Original 대비 편차 측정:

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

### Optimized SAM-3D의 seed variance도 측정

Optimized SAM-3D도 seed에 따라 변동이 있을 수 있음.
Original과 Optimized 양쪽의 seed variance를 비교하면:
- Optimized의 CV가 Original과 비슷하면: "최적화가 모델 안정성에 영향 없음"
- Optimized의 CV가 더 높으면: "최적화가 출력 분산을 키움" → 논의 필요

### 서비스 이미지 보충 (TV 등 극박 객체)

Pix3D에 없는 카테고리(TV, 냉장고 등)는 이삿짐 서비스 이미지에서 5-10장을 추가.
이 경우 마스크는 YOLOE-seg로 생성하되 seed variance 실험에서는 **마스크를 고정**(1회만 생성 후 재사용).
보충 데이터는 메인 결과와 분리하여 "서비스 이미지 보충 실험"으로 별도 보고.

---

## 필요 리소스

| 항목 | 수량 | 비고 |
|------|------|------|
| GPU | L4 4대 (또는 A100 1-2대) | SAM-3D full model 로드 (~21GB/GPU) |
| **데이터셋** | **Pix3D에서 50장** (7 카테고리) | **이미지 + GT mask 쌍** |
| Phase 1 실행 횟수 | 50 images × 5 seeds = **250회** | Original SAM-3D |
| Phase 2 실행 횟수 | 50 images × 1 seed = **50회** | Optimized SAM-3D |
| 예상 시간 (Phase 1, 단일 GPU) | 250 × ~150초 = **~10.4시간** | Original은 느림 |
| 예상 시간 (Phase 1, 4× GPU 병렬) | **~2.5시간** | 배치 병렬 실행 |
| 예상 시간 (Phase 2) | 50 × ~13초 = **~11분** | Optimized는 빠름 |
| **총 예상 시간 (4× GPU)** | **~3시간** | Phase 1이 지배적 |

> Phase 1이 느리므로 **밤에 배치로 돌리는 것** 권장.
> 4 GPU 병렬이면 ~3시간, 단일 GPU면 ~11시간.
>
> **통계적 신뢰성**: 50개 객체 × 5 seeds = 250개 측정점.
> 카테고리당 평균 7개 객체 → 카테고리별 분산도 보고 가능.
> 논문에서 "Pix3D 데이터셋의 7개 가구 카테고리, 50개 객체에 대해 측정" 기술.

---

## 실행 스크립트 (TODO)

```python
# TODO: 실제 실행 스크립트 작성
# experiments/seed_variance.py
#
# 1. Pix3D 데이터셋에서 50개 이미지 + GT mask 로드
# 2. Original SAM-3D 로드 (compile=False, full decoder)
# 3. seed 1~5로 각각 실행 → PLY 저장 (250회)
# 4. PLY → PCA OBB → (w, d, h) 기록
# 5. CSV 출력
#
# 데이터 준비:
#   # Pix3D 다운로드 (http://pix3d.csail.mit.edu/)
#   wget http://pix3d.csail.mit.edu/data/pix3d.zip
#   unzip pix3d.zip -d datasets/pix3d/
#
# 실행 방법:
#   python experiments/seed_variance.py \
#     --pix3d-dir datasets/pix3d/ \
#     --samples-per-category 8 \
#     --seeds 5 \
#     --output results/seed_variance.csv \
#     --gpu-ids 0,1,2,3
```

---

## 체크리스트

- [ ] Pix3D 데이터셋 다운로드 (http://pix3d.csail.mit.edu/)
- [ ] 7개 카테고리에서 50개 이미지 + GT mask 샘플링
- [ ] 샘플링 기준 확인 (형태 다양성, 가려짐 없음, 해상도 512+ 등)
- [ ] GPU 서버에서 Original SAM-3D (25+25 steps, full decoder) 실행 가능 확인
- [ ] Phase 1 배치 실행 (250회, 4×GPU ~2.5시간)
- [ ] Phase 2 배치 실행 (50회, ~11분)
- [ ] Table B (카테고리별 Seed Variance) 작성 → δ 확정
- [ ] Table C (Optimization Deviation vs δ) 작성 → 판정 비율 계산
- [ ] 논문 Table 1/4에 반영
- [ ] (Optional) Ablation: 각 최적화 개별 영향 측정
- [ ] (Optional) Optimized SAM-3D seed variance 측정
- [ ] (Optional) 서비스 이미지 보충 실험 (TV 등 극박 객체)
