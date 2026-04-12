# SAM-3D 3종 벤치마크 실험 계획서

## 목적

Pix3D 데이터셋을 활용하여 Original SAM-3D, Fast-SAM3D, Ours(최적화 SAM-3D) 3종의
**Latency**, **VRAM 점유율**, **치수 추정 정확도(RDE, DA@τ, VE)**를 비교 측정한다.
추가로 Ours의 5가지 최적화 각각의 기여도를 Ablation Study로 분리 측정한다.

---

## 비교 대상

> **원칙**: 각 variant의 **레포/파이프라인 default 설정**을 그대로 사용한다.
> 단, pointmap은 3종 모두 MoGe 대신 synthetic pinhole을 사용 (MoGe 미설치 환경 통일).

| 항목 | Original SAM-3D | Fast-SAM3D | Ours |
|------|----------------|------------|------|
| 출처 | facebook/sam-3d-objects | [GitHub](https://github.com/wlfeng0509/Fast-SAM3D) | 자체 최적화 |
| conda env | `sam3d-objects` | `fastsam3d` (별도) | `sam3d-objects` |
| stage1_steps | 25 (default) | 레포 기본값 | 14 |
| stage2_steps | 25 (default) | 레포 기본값 | 4 |
| decode_formats | **["gaussian", "mesh"]** (default) | 레포 기본값 | ["gaussian"] |
| pointmap | synthetic | 레포 기본값 | synthetic |
| step caching | off (default) | on (레포 기본) | SS only (stride=3) |
| torch.compile | off (_warmup 버그로 실행 불가) | 레포 기본 | on (reduce-overhead, 자체 warmup) |
| VRAM unload | off (default) | 레포 기본 | on |
| mesh_postprocess | **True** (default) | 레포 기본 | False |
| texture_baking | **True** (default) | 레포 기본 | False |
| use_vertex_color | **False** (default) | 레포 기본 | True |

### Original SAM-3D default 설정 근거

`sam-3d-objects/sam3d_objects/pipeline/inference_pipeline.py` 소스 확인 결과:
- `decode_formats`: `["gaussian", "mesh"]` (\_\_init\_\_ line 104)
- `ss_inference_steps` / `slat_inference_steps`: `25` / `25`
- `with_mesh_postprocess`: `True` (run() default)
- `with_texture_baking`: `True` (run() default)
- `use_vertex_color`: `False` (run() default)
- `compile_model`: pipeline.yaml에는 `True`이나, `_warmup()`에서 `run_layout_model` 미구현 버그로 crash → **실제 사용 시 `compile=False`** 필수

### 의존성 (Phase 0에서 확인)
- `nvdiffrast`: mesh_postprocess(hole filling)에 필요 → `pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast`
- `diff-gaussian-rasterization`: texture_baking에 필요 → Mip-Splatting 포크 사용 (kernel_size 지원) → `pip install --no-build-isolation git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization`

## 실험 환경

| 항목 | 사양 |
|------|------|
| VM | **2대** (동일 스펙) |
| GPU | NVIDIA L4 (**22GB** usable VRAM) **1개/VM** |
| 실행 방식 | VM별 단일 GPU 순차 실행, VM 간 병렬 |

## 데이터셋: Pix3D

### 전체 규모 (품질 필터링 후)

필터 조건: `truncated=False`, `occluded=False`

| 카테고리 | 전체 | 필터 후 |
|---------|------|---------|
| bed | 994 | 532 |
| bookcase | 361 | 228 |
| chair | 3,839 | 3,064 |
| desk | 700 | 428 |
| misc | 68 | 61 |
| sofa | 1,947 | 1,352 |
| table | 1,870 | 1,124 |
| tool | 47 | 46 |
| wardrobe | 243 | 186 |
| **합계** | **10,069** | **7,021** |

### 실험 규모: **500개** (9 카테고리 층화 추출)

| 카테고리 | 필터 후 N | 비율 | 샘플 수 |
|---------|----------|------|---------|
| chair | 3,064 | 43.6% | 218 |
| sofa | 1,352 | 19.3% | 96 |
| table | 1,124 | 16.0% | 80 |
| bed | 532 | 7.6% | 38 |
| desk | 428 | 6.1% | 30 |
| bookcase | 228 | 3.2% | 16 |
| wardrobe | 186 | 2.6% | 13 |
| misc | 61 | 0.9% | 5 |
| tool | 46 | 0.7% | 4 |
| **합계** | **7,021** | **100%** | **500** |

### 데이터 구조

```
experiments/seed_variance/data/pix3d/
├── pix3d.json         # 메타데이터 (10,069 entries)
├── img/               # RGB 이미지 (9 카테고리)
├── mask/              # Binary mask (동일 구조)
└── model/             # 3D GT mesh (735 unique OBJ, 1.1GB) ← 다운로드 완료
```

---

## 측정 항목

### 성능 지표 (Performance Metrics)

| 항목 | 단위 | 설명 |
|------|------|------|
| **Latency** | 초/객체 | `torch.cuda.synchronize()` 전후 시간 측정 |
| **VRAM Peak** | GB | `torch.cuda.max_memory_allocated()` |
| **Model VRAM** | GB | 모델 로딩 후 `torch.cuda.memory_allocated()` |
| **성공률** | % | OOM/에러 없이 완료된 비율 |

### 치수 추정 평가 지표 (Evaluation Metrics)

> **GT 기준**: Pix3D 3D GT mesh (model.obj).
> GT mesh는 canonical orientation이므로 AABB로 치수 추출, SAM-3D 출력은 PCA OBB로 추출.
> 양쪽 모두 max(w,d,h)로 정규화 후 크기순 정렬하여 축 매칭.
>
> Pix3D 내 unique 3D model: **735개** (10,069 이미지가 이 735개를 공유).

#### (1) Relative Dimension Error (RDE) — 상대 치수 오차

$$
RDE = \frac{1}{3} \left( \frac{|w - \hat{w}|}{w} + \frac{|h - \hat{h}|}{h} + \frac{|d - \hat{d}|}{d} \right)
$$

- $w, h, d$ : GT 치수 (Pix3D GT mesh → AABB → 정규화 → 크기순 정렬)
- $\hat{w}, \hat{h}, \hat{d}$ : 예측 치수 (SAM-3D PLY → PCA OBB → 정규화 → 크기순 정렬)

#### (2) Dimension Accuracy @ τ (DA@τ)

RDE ≤ τ인 샘플의 비율. **DA@5**, **DA@10**, **DA@20**.

#### (3) Volume Error (VE) — 부피 오차

$$
VE = \frac{|V - \hat{V}|}{V}, \quad V = w \times h \times d
$$

### GT 치수 생성 파이프라인

```
Pix3D model.obj (735 unique, canonical orientation)
    ↓
trimesh.load() → vertices
    ↓
AABB (min/max per axis) → (w, d, h)
    ↓
max(w,d,h) 정규화 → 크기순 정렬
    ↓
gt_dimensions.json
```

### 측정 프로토콜 (3종 + Ablation 동일)

1. 모델 로드 → GPU warmup (dummy 1회)
2. `torch.cuda.reset_peak_memory_stats()` (매 샘플 전)
3. `torch.cuda.synchronize()` → 시간 시작
4. 추론 실행 (seed=42 고정)
5. `torch.cuda.synchronize()` → 시간 종료
6. `torch.cuda.max_memory_allocated()` → VRAM peak 기록
7. PLY → PCA OBB → 정규화 → 크기순 정렬 → 치수(w, d, h) 저장
8. CSV에 기록 (10개마다 중간 저장)

---

## 시간 예상

> Phase 0 실측: Original default **75.4s**/obj, gaussian-only 22.5s/obj

### 실험 재활용 설계

3종 비교와 Ablation에서 동일한 설정의 실험을 **1회만 실행**하고 양쪽에서 재활용:

| 실행 ID | Configuration | 용도 | 예상 Latency | 500개 소요 |
|---------|---------------|------|-------------|-----------|
| R1 | Original default (Ablation Baseline) | 3종 비교 **Original** + Ablation **Baseline** | ~75s | ~10.4h |
| R2 | Fast-SAM3D (레포 default) | 3종 비교 **Fast-SAM3D** | ~17s | ~2.4h |
| R3 | + O1: Gaussian-Only | Ablation **O1** | ~23s | ~3.2h |
| R4 | + O2: VRAM Unload | Ablation **O2** | ~23s | ~3.2h |
| R5 | + O4: Steps Reduction | Ablation **O4** | ~15s | ~2.1h |
| R6 | Ours (= Ablation O5) | 3종 비교 **Ours** + Ablation **O5** | ~13s | ~1.8h |
| | | **총 고유 실행** | | **~23.1시간 (~1일)** |

재활용 효과:
- ~~3종 Original~~ = R1 (Ablation Baseline과 공유) → **10.4시간 절감**
- ~~3종 Ours~~ = R6 (Ablation O5와 공유) → **1.8시간 절감**
- **총 절감: ~12.2시간 (35%↓)**, 35.3시간 → **23.1시간**

### 2-VM 병렬 실행 배분

R1(Original, 10.4h)이 병목이므로 이를 기준으로 분배:

| | VM-A | VM-B |
|--|------|------|
| conda env | `sam3d-objects` | `sam3d-objects` + `fastsam3d` |
| 실행 | R1 → R6 | R3 → R4 → R5 → R2 |
| 내역 | Original(10.4h) + Ours(1.8h) | O1(3.2h) + O2(3.2h) + O4(2.1h) + Fast-SAM3D(2.4h) |
| 합계 | **~12.2h** | **~10.9h** |
| **벽시계** | | **~12.2시간 (반나절)** |

```
VM-A  ┃ R1 (Original/Baseline) ████████████████████████████████████ │ R6 (Ours) ████│
VM-B  ┃ R3 (O1) ████████│ R4 (O2) ████████│ R5 (O4) █████│ R2 (F-SAM3D) ██████│
      ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      0h        2h        4h        6h        8h       10h       12h
```

VM 셋업 요구사항:
- **공통**: Pix3D 데이터 (img/ + mask/ + model/), `benchmark_samples.json`, `sam3d-objects` conda env, SAM-3D 체크포인트
- **VM-A 추가**: nvdiffrast, diff-gaussian-rasterization (Mip-Splatting 포크) — Original mesh_pp+baking용
- **VM-B 추가**: `fastsam3d` conda env — Fast-SAM3D용

결과 수집: 실험 완료 후 양쪽 VM의 `results/*.csv`를 한 곳으로 모아서 `collect_results.py` 실행

---

## 실행 단계

### Phase 0: Original SAM-3D L4 동작 테스트 ✅ 완료

**실행 스크립트**: `experiments/benchmark/phase0_test_original.py`

**결과 (2026-04-12 실측)**:

모델 로딩 VRAM:
- VRAM allocated: **12.761 GB** (22GB 중 58%)
- 서브모델: ss_generator 3,678MB + slat_generator 1,148MB + ss_decoder 281MB + decoders 499MB = **5,606MB** (params)
- CUDA overhead: ~7,461MB

| 테스트 | decode_formats | mesh_pp / tex_bake | Avg Latency | Max VRAM Peak | VRAM % |
|--------|---------------|-------------------|-------------|---------------|--------|
| gaussian-only | ["gaussian"] | — | **22.5s** | **14.29 GB** | 65% |
| **Original default** | **["gaussian", "mesh"]** | **True / True** | **75.4s** | **17.99 GB** | **82%** |

핵심 발견:
- mesh decode + postprocess + texture baking = **+52.9s (235% 증가)**, +3.7 GB VRAM
- **Ours(~13s) vs Original default(~75s) = 5.8× speedup** (mesh_pp=False 기준 대비 훨씬 큰 차이)

### Phase 1: Fast-SAM3D 환경 구축

**목표**: Fast-SAM3D를 별도 conda env에서 동작 검증

1. 레포 클론: `git clone https://github.com/wlfeng0509/Fast-SAM3D.git fast-sam3d`
2. **별도 conda env 생성** (`fastsam3d`) — 의존성 충돌 방지
3. 기존 SAM-3D 체크포인트 심볼릭 링크
4. hydra 패치 적용
5. 단일 이미지 테스트
6. 출력 형식 확인 (PLY → OBB 치수 추출 가능 여부)

### Phase 2: 통합 벤치마크 스크립트 작성

```
experiments/benchmark/
├── config.py                 # 전체 설정 (6개 config: baseline/o1/o2/o4/o5 + fastsam3d)
├── run_all.sh                # 전체 실행 셸 스크립트 (R1→R6→R2 순서)
├── worker_ablation.py        # 통합 워커 (--config flag로 baseline~o5 전환)
├── worker_fastsam3d.py       # Fast-SAM3D 전용 워커 (fastsam3d env)
├── prepare_data.py           # 500개 층화 추출 + GT 치수 생성
├── collect_results.py        # 결과 집계 + RDE/DA@τ/VE 계산
└── results/
    ├── original.csv          # R1: 3종 Original + Ablation Baseline (공유)
    ├── ablation_o1.csv       # R3: Ablation +O1
    ├── ablation_o2.csv       # R4: Ablation +O2
    ├── ablation_o4.csv       # R5: Ablation +O4
    ├── ours.csv              # R6: 3종 Ours + Ablation +O5 (공유)
    ├── fastsam3d.csv         # R2: 3종 Fast-SAM3D
    └── summary.csv           # 종합 통계
```

CSV 출력 형식 (전체 동일):
```csv
category,sample_id,img_path,model_path,width,depth,height,latency_seconds,vram_peak_mb,success,error
```

### Phase 3: 데이터셋 준비

1. ~~Pix3D 3D model 다운로드~~ ✅ 완료 (735 unique OBJ, `model/` 디렉토리)
2. GT 치수 생성: model.obj → AABB → 정규화 → `gt_dimensions.json`
3. `pix3d.json`에서 전체 9 카테고리 필터링 (truncated=False, occluded=False)
4. 500개 층화 추출 → `benchmark_samples.json`
5. 마스크 pixel count 검증 (MIN_MASK_PIXELS=1000)

### Phase 4: 통합 실행 (3종 비교 + Ablation 공유)

> **재활용 설계**: 동일 설정은 1회만 실행, 3종 비교와 Ablation 양쪽에서 참조.
> R1(Original) = Ablation Baseline, R6(Ours) = Ablation O5.

#### 5가지 최적화 (Ablation 적용 순서)

| # | 최적화 | 설명 | 기대 효과 |
|---|--------|------|----------|
| O1 | **Gaussian-Only Decode** | mesh decoder 스킵, postprocess/baking 제거 | **Latency 75→23s (3.3×)**, VRAM 18→14 GB |
| O2 | **VRAM Model Unloading** | 미사용 decoder GPU에서 제거 | **Model VRAM 12.8→~6.5 GB (49%↓)** |
| O3 | **Synthetic Pinhole Pointmap** | MoGe 대신 synthetic (이미 적용 중) | VRAM ↓ (MoGe ~1GB 절감) |
| O4 | **Inference Steps Reduction** | stage1: 25→14, stage2: 25→4 | **Latency 23→~15s** |
| O5 | **SS Step Caching** | CachedEuler (stride=3, warmup=2) | **Latency 15→~13s** |

> O3(Synthetic Pointmap)은 Baseline에서도 이미 사용 중 (MoGe 미설치).
> 실측에서는 O2→O4로 직접 비교. O3 효과는 "MoGe VRAM ~1GB 절감"으로 기술.

#### 핵심 관찰

- **O1+O2는 수학적으로 동일한 PLY 출력** (dead-code path 제거) → RDE = 0%, VE = 0%
- **O4+O5에서 치수 편차 발생** → seed variance δ 이하인지 확인이 핵심

#### 실행 스크립트 (VM별)

```bash
# ═══════════════════════════════════════════════
# VM-A: run_vm_a.sh  (sam3d-objects env)
# ═══════════════════════════════════════════════

# R1: Original default = Ablation Baseline (~10.4h)
#   → 3종 비교 Original + Ablation Baseline 공유
conda run -n sam3d-objects python worker_ablation.py \
    --gpu 0 --config baseline \
    --samples benchmark_samples.json --output results/original.csv

# R6: Ours = Ablation O5 (~1.8h)
#   → 3종 비교 Ours + Ablation O5 공유
conda run -n sam3d-objects python worker_ablation.py \
    --gpu 0 --config o5 \
    --samples benchmark_samples.json --output results/ours.csv
```

```bash
# ═══════════════════════════════════════════════
# VM-B: run_vm_b.sh  (sam3d-objects + fastsam3d env)
# ═══════════════════════════════════════════════

# R3: + O1 Gaussian-Only (~3.2h)
conda run -n sam3d-objects python worker_ablation.py \
    --gpu 0 --config o1 \
    --samples benchmark_samples.json --output results/ablation_o1.csv

# R4: + O2 VRAM Unload (~3.2h)
conda run -n sam3d-objects python worker_ablation.py \
    --gpu 0 --config o2 \
    --samples benchmark_samples.json --output results/ablation_o2.csv

# R5: + O4 Steps Reduction (~2.1h)
conda run -n sam3d-objects python worker_ablation.py \
    --gpu 0 --config o4 \
    --samples benchmark_samples.json --output results/ablation_o4.csv

# R2: Fast-SAM3D (~2.4h)
conda run -n fastsam3d python worker_fastsam3d.py \
    --gpu 0 --samples benchmark_samples.json --output results/fastsam3d.csv
```

중단 복구: CSV에 이미 처리된 sample_id 확인하여 이어서 실행

#### 결과 파일 → 테이블 매핑

| 결과 파일 | 3종 비교 | Ablation |
|-----------|---------|----------|
| `results/original.csv` | **Original** | **Baseline** |
| `results/fastsam3d.csv` | **Fast-SAM3D** | — |
| `results/ablation_o1.csv` | — | **+O1** |
| `results/ablation_o2.csv` | — | **+O2** |
| `results/ablation_o4.csv` | — | **+O4** |
| `results/ours.csv` | **Ours** | **+O5** |

### Phase 5: 결과 집계 + 평가 지표 계산

산출물:
- `summary.csv`: variant별 평균 latency, VRAM, 성공률
- `evaluation.csv`: variant별 RDE, DA@5/10/20, VE
- `ablation_summary.csv`: Ablation 단계별 latency, VRAM, speedup, RDE
- `category_stats.csv`: 카테고리별 통계

---

## 예상 결과 형태

### Table 1: 전체 성능 비교 (3종)

| Variant | Latency (s) | Speedup | Peak VRAM (GB) | Model VRAM (GB) | VRAM ↓ |
|---------|-------------|---------|----------------|-----------------|--------|
| Original SAM-3D | **~75** | 1.0× | ~18.0 | 12.76 | — |
| Fast-SAM3D | ~17 | ~4.4× | ?? | ?? | ?% |
| **Ours** | **~13** | **~5.8×** | **~8.2** | **~6.5** | **64%↓** |

### Table 2: 치수 추정 정확도 (vs Pix3D GT)

| Variant | RDE (↓) | DA@5 (↑) | DA@10 (↑) | DA@20 (↑) | VE (↓) |
|---------|---------|----------|-----------|-----------|--------|
| Original SAM-3D | ?% | ?% | ?% | ?% | ?% |
| Fast-SAM3D | ?% | ?% | ?% | ?% | ?% |
| Ours | ?% | ?% | ?% | ?% | ?% |

### Table 3: 최적화 단계별 Ablation (논문 핵심 표)

| Configuration | Latency (s) | Speedup | VRAM Peak (GB) | Model VRAM (GB) | RDE (↓) | VE (↓) |
|---------------|-------------|---------|----------------|-----------------|---------|--------|
| Baseline (Original) | ~75 | 1.0× | ~18.0 | 12.76 | ?% | ?% |
| + O1: Gaussian-Only | ~23 | **3.3×** | ~14.3 | 12.76 | 0% | 0% |
| + O2: VRAM Unload | ~23 | 3.3× | ~? | **~6.5** | 0% | 0% |
| + O4: Steps 14/4 | ~15 | **5.0×** | ~? | ~6.5 | ?% | ?% |
| + O5: SS Caching (**Ours**) | **~13** | **5.8×** | **~8.2** | **~6.5** | **?%** | **?%** |

> O1: mesh decode + postprocess + baking 제거 → **단일 최대 기여 (75→23s, 3.3×)**
> O1+O2: dead-code path 제거 → RDE = 0% (수학적 보장)
> O4+O5: 추론 스텝 최적화 → seed variance δ 이하 편차 예상

### Table 4: 카테고리별 Latency (N=500)

| Category | N | Original | Fast-SAM3D | Ours |
|----------|---|----------|------------|------|
| chair | 218 | ? | ? | ? |
| sofa | 96 | ? | ? | ? |
| table | 80 | ? | ? | ? |
| bed | 38 | ? | ? | ? |
| desk | 30 | ? | ? | ? |
| bookcase | 16 | ? | ? | ? |
| wardrobe | 13 | ? | ? | ? |
| misc | 5 | ? | ? | ? |
| tool | 4 | ? | ? | ? |

---

## 리스크

| 리스크 | 심각도 | 대응 |
|--------|--------|------|
| ~~Original full decoder가 L4 VRAM 초과~~ | ~~HIGH~~ | ✅ **해소**: Phase 0에서 17.99GB/22GB 확인 |
| ~~Original 의존성 (nvdiffrast, diff-gaussian-rasterization)~~ | ~~HIGH~~ | ✅ **해소**: Mip-Splatting 포크 설치 완료 |
| Fast-SAM3D 환경이 기존 SAM-3D와 충돌 | HIGH | 별도 conda env (`fastsam3d`) |
| Fast-SAM3D 체크포인트 호환성 | MEDIUM | 동일 facebook/sam-3d-objects 체크포인트 사용 |
| ~~단일 GPU 총 실행 시간 (~1일)~~ | ~~MEDIUM~~ | **해소**: 2-VM 병렬로 ~12시간 (반나절) |
| 2-VM 간 데이터/환경 동기화 | LOW | 동일 Pix3D 데이터 + benchmark_samples.json 배포, 결과 CSV 수집 |
| Fast-SAM3D 입출력 형식 차이 | MEDIUM | 어댑터 작성 |

## 관련 파일

- `experiments/benchmark/phase0_test_original.py` — Phase 0 실험 스크립트
- `experiments/seed_variance/experiment_worker.py` — 기존 워커 (참고 템플릿)
- `experiments/seed_variance/config.py` — 기존 설정
- `ai/subprocess/persistent_3d_worker.py` — Ours 최적화 설정
- `docs/paper/KIISE_POSTER_PROPOSAL.md` — 논문 제안서 (Table 2 설계)
- `docs/PIPELINE_OPTIMIZATION.md` — 최적화 상세
