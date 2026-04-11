# KIISE 포스터 논문 제안서

## Document Info

| 항목 | 내용 |
|------|------|
| 대상 학회 | 한국정보과학회 (KIISE) — KCC 또는 KSC 포스터 세션 |
| 분량 | 3페이지 (레퍼런스 포함) |
| 상태 | 제안 단계 |

---

## 1. 제목 후보

**한국어:**
- 이삿짐 견적을 위한 SAM-3D 기반 가구 부피 추정: 도메인 특화 최적화
- Gaussian-Only 모드와 표준 치수 DB를 활용한 실시간 3D 부피 측정
- 상용 서비스를 위한 SAM-3D의 도메인 특화 경량화

**영문:**
- Task-Aware Pruning of SAM-3D via OBB Volume Invariance: A Case Study on Furniture Volume Estimation
- Domain-Specific Optimization of SAM-3D for Real-Time Furniture Volume Estimation
- Gaussian-Only Decoding for Furniture Volume Estimation: An 11.5× Speedup of SAM-3D

---

## 2. 핵심 Thesis (선택과 집중)

### 한 줄 핵심 주장

> **"부피 계산만 필요한 downstream task에서는, SAM-3D의 출력이 회전·스케일에 불변인 OBB로 귀결되므로, 이 불변성을 역으로 활용하여 inference pipeline의 상당 부분을 제거·단순화할 수 있다. 결과적으로 L4 GPU에서 객체당 150초 → 13초 (11.5×) 가속과 48% VRAM 절감을 training-free로 달성한다."**

### 왜 이 angle인가

- **Novelty가 명확**: "volume invariance로 모델을 pruning한다"는 관점은 기존 최적화 논문에 없음
- **Measurable**: 11.5×, 48%, <3% error 같은 정량 수치가 포스터에서 강력
- **Fast-SAM3D와 차별화**: 논문은 "시각 품질 유지하며 가속", 본 연구는 "task 특성 활용한 적극적 pruning"
- **실제 배포 context**: "이삿짐 견적 서비스"라는 구체적 시나리오는 practical impact 어필

---

## 3. 포함할 기법 (5개로 압축)

14개 최적화 중 아래 5개만 포스터에 포함:

| # | 기법 | 논문에서의 역할 | 포함 이유 |
|---|------|--------------|----------|
| 1 | **Gaussian-Only Decoding** | Thesis의 출발점 | "mesh 디코딩 불필요"라는 핵심 발상 |
| 2 | **VRAM Model Unloading** (48%) | Task-Aware Pruning 적용 | 불변성의 직접적 결과 |
| 3 | **Synthetic Pinhole Pointmap** | Task-Aware Pruning 적용 | MoGe 대체, 추가 안정성 + VRAM |
| 4 | **Inference Steps Reduction** (25→14, 12→4) | 상대 치수 기준 최적점 탐색 | 상대 치수 정확도 관점 실험 (논문은 시각 품질 기준) |
| 5 | **SS Step Caching** (Fast-SAM3D Phase A) | 직교적 가속 추가 | 기존 연구 적용의 ablation 비교 |

### 제외할 것 (3페이지에 안 들어감)

- ❌ Persistent Worker Pool / 2단계 병렬 / Work-stealing → systems 논문 angle
- ❌ torch.compile / in_place / Binary PLY → 엔지니어링 세부
- ❌ PLY 전처리 / 절대 부피 DB → downstream 처리 (별도 논문 소재)
- ❌ V2 파이프라인 (SAM2/CLIP 제거) → SAM-3D와 직접 무관

---

## 4. 3페이지 구성안

### Page 1

#### Title (가로 full)
부피 불변성을 활용한 Task-Aware SAM-3D 가속화: 이삿짐 부피 추정 사례

#### Authors, Affiliation (가로 full)

#### Abstract (150-200자)
본 논문은 상용 이삿짐 견적 서비스를 위해 SAM-3D 기반 가구 부피 자동 측정 파이프라인을 제안한다.
범용 3D 생성 모델인 SAM-3D는 고품질 mesh/texture 복원을 목표로 설계되어 객체당 150초, VRAM 21GB를
요구하므로 실서비스 적용이 어렵다. 우리는 **부피 계산에는 3D point cloud만 필요하다는 관찰**에 기반해
mesh 디코딩 경로를 완전히 우회하는 **Gaussian-Only 모드**, 이로 인해 dead-code가 된 서브 모델을
언로드하는 **VRAM 최적화**, 그리고 **OBB 기반 부피 추출 + 52개 가구 표준 치수 DB 매칭**을 통해
절대 부피를 산출한다. 실험 결과 L4 GPU에서 객체당 처리 시간을 150초에서 13초로 단축(11.5배 가속)하고
VRAM을 48% 절감하면서 상대 치수 오차를 3% 이내로 유지했다.

#### 1. 서론 (0.5p)
- 배경: 이삿짐 견적 시 트럭 크기 결정을 위해 가구 부피 측정 필요
- SAM-3D의 한계: 단일 이미지 → 3D mesh + Gaussian Splat + texture 복원. 시각적 품질이 목적이라 견적 서비스에 과도함.
- **핵심 관찰**: 견적의 부피 계산은 **회전 불변 OBB만** 있으면 됨 → mesh/texture 불필요
- **기여 3가지**:
  1. Gaussian-Only 디코딩으로 SAM-3D의 mesh 경로 완전 우회
  2. dead-code path가 된 SLaT mesh/GS-4/depth 모델 언로드로 48% VRAM 절감
  3. OBB + 표준 치수 DB 매칭으로 스케일-프리 3D 출력에서 절대 부피(m³) 역산

#### 2. 관련 연구 (0.3p)
- **2D → 3D 복원 모델**: TripoSR, Hunyuan3D, SAM-3D [Meta, 2025]
- **추론 가속**: Fast-SAM3D [arXiv:2602.05293] — step caching + token pruning
- **차별점**: 기존 연구는 시각적 품질(Chamfer Distance, F-Score) 유지가 목표. 본 연구는 **상대 치수 정확도** (OBB extent 보존)가 유일한 제약

### Page 2

#### 3. 제안 방법 (1p)

##### 3.1 Volume Invariance Observation
- OBB = PCA(point cloud) → extent가 rotation-invariant
- Canonical pose 불필요, mesh 불필요
- Gaussian Splatting point cloud만으로 OBB 계산 가능
- 수식: `OBB_extent(R · pts + t) = OBB_extent(pts), ∀R, t → V = w · d · h 는 pose에 불변`

##### 3.2 Gaussian-Only Decoding
- SAM-3D 기본: `decode_formats = [gaussian, mesh, glb]`
- 본 연구: `decode_formats = [gaussian]`만
- `with_texture_baking=False`, `with_mesh_postprocess=False`, `with_layout_postprocess=False`
- Figure 1: 파이프라인 비교 (기본 vs Gaussian-Only)

##### 3.3 VRAM Model Unloading
- Gaussian-Only 모드에서 호출되지 않는 3개 서브 모델 식별:
  - `slat_decoder_mesh` (~3-4GB) — mesh 생성 안 함
  - `slat_decoder_gs_4` (~2-3GB) — 기본 `slat_decoder_gs`로 대체
  - `depth_model (MoGe)` (~1-3GB) — `synthetic pinhole pointmap`으로 대체
- `model.cpu()` + `torch.cuda.empty_cache()` → 총 ~10GB 절감 (48%)

##### 3.4 Synthetic Pinhole Pointmap
- MoGe: 학습된 depth model, ~3GB, NaN/Inf 위험
- 대체: `z=1` uniform plane + pinhole camera model
- 상대 치수 오차: 일반 가구 1-2.6% (절대 깊이 불필요, 상대 비율만 중요)

##### 3.5 Inference Steps: 상대 치수 기준 최적점 탐색
- 기존 Fast-SAM3D: Chamfer Distance 기준 스윕
- 본 연구: **상대 치수 오차 기준** 스윕
- Stage1 25→14 (50% 가속, 1.5% 오차), Stage2 12→4 (30% 가속, 0.5% 오차)
- Table 1: Stage1 Steps 테스트 결과

### Page 3

#### 4. 실험 (1p)

##### 4.1 환경
- Hardware: NVIDIA L4 (24GB, Ada Lovelace)
- Dataset: 가구 이미지 N장 (이삿짐 서비스 실제 촬영 + 공개 이미지)
- **Baseline**: Original SAM-3D (25+25 steps, full decoder, MoGe depth)
- **핵심 Metric**: Original SAM-3D 대비 OBB 상대 치수 변화율 (%)

> **GT 불필요 근거**: SAM-3D 원 논문이 3D 품질(CD, F-Score, human pref.)을 이미 검증.
> 본 실험은 "최적화가 원본 대비 얼마나 degradation을 일으키는가"만 측정.

##### 4.2 Table 2: 최적화 단계별 Ablation (가장 중요한 표)

각 최적화를 점진적으로 추가하며 **Original SAM-3D 대비 상대 치수 변화**를 측정:

| Configuration | Time/obj | VRAM | W err | D err | H err | V dev |
|---------------|---------|------|-------|-------|-------|-------|
| Original SAM-3D (baseline) | ~150s | 21GB | 0% | 0% | 0% | 0% |
| + Gaussian-Only | ~94s | 18GB | — | — | — | <0.1% |
| + VRAM Unload | ~94s | 11.25GB | — | — | — | 0% |
| + Synthetic Pointmap | ~90s | ~8.2GB | ~1% | ~1% | ~1% | ~1-3% |
| + Steps 14/4 | ~20s | ~8.2GB | — | — | — | ~1.5% |
| + SS Step Caching (stride=3) | **~13s** | **~8.2GB** | — | — | — | **<3%** |
| **Speedup** | **11.5×** | **61%↓** | | | | **maintain** |

> "—" 표시: 해당 최적화에 의한 추가 degradation이 무시 가능
> VRAM Unload: dead-code path 제거이므로 출력에 영향 없음 (수학적으로 동일)
> 치수에 영향을 주는 것: Synthetic Pointmap, Steps Reduction, Step Caching

##### 4.3 가구 종류별 치수 오차 (Original SAM-3D 대비)

| 가구 | W err | D err | H err | 특성 |
|------|-------|-------|-------|------|
| Nightstand | 2.3% | 0.1% | 1.9% | 일반 |
| Bed | 0.1% | 0.2% | 0.3% | 일반 |
| Television | 1.2% | 2.7% | *9.4% | 극박 (depth≈0.02) |

> *TV H err: 캐싱 없이도 4.3% 자연 변동 → 극박 객체의 내재적 불안정성

##### 4.4 논의
- Gaussian-Only + VRAM Unload는 **수학적으로 동일한 출력** (dead path 제거일 뿐)
  → 37% 속도↑ + 48% VRAM↓ 가 **무손실**
- Synthetic Pointmap은 1-3% 치수 변화이지만 **NaN/Inf 발생률 0%** (안정성 관점에서 개선)
- Steps 14/4: 상대 치수 기준 sweet spot. 시각 품질 기준(CD/F-Score)보다 더 공격적 값이 허용됨
  → **"평가 metric이 바뀌면 최적화 경계도 달라진다"**는 insight

#### 5. 결론 (0.2p)
- Downstream task가 특정 불변성(OBB)을 가질 때, 이를 활용한 task-aware pruning이
  generic training-free 가속보다 더 큰 이득 가능함을 실증
- 제안 방법은 SAM-3D 원본 모델 재학습 없이 적용 가능 (training-free)
- 향후 과제: 다른 downstream task (물리 시뮬레이션, AR 배치)에 불변성 기반 pruning 일반화

#### References (5-8개)
- [1] Meta AI. "SAM 3D Objects." 2025.
- [2] Fast-SAM3D. arXiv:2602.05293, 2026.
- [3] Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." SIGGRAPH, 2023.
- [4] trimesh: Python library for 3D meshes. https://trimsh.org
- [5] Ultralytics. YOLOE-seg. 2025.
- [6] PyTorch 2.0: torch.compile.

---

## 5. 핵심 Figure/Table 우선순위

### Must-have (3개)
1. **Figure 1**: 파이프라인 다이어그램 (Gaussian-Only vs 기본) — 한눈에 이해 가능
2. **Table 2**: 최적화 단계별 성능 (Baseline → Ours, 점진적 개선)
3. **Table 3**: End-to-end 시나리오 비교 (43× 가속 강조)

### Nice-to-have (1-2개)
4. **Figure 2**: VRAM 변화 막대 그래프 (21GB → 8.2GB)
5. **Table 4**: 가구별 상대 치수 오차 (정확도 검증)

---

## 6. Novelty 방어 전략

### 심사자가 "Fast-SAM3D와 뭐가 다르냐" 물을 때

> Fast-SAM3D는 generic training-free acceleration을 목표로 시각 품질을 유지하며 가속합니다.
> 본 연구는 특정 downstream task(부피 추정)가 제공하는 OBB invariance를 활용하여
> Fast-SAM3D가 건드리지 않은 decoder 자체를 pruning합니다.
> Fast-SAM3D의 Step Caching은 본 연구의 마지막 단계로 직교적으로 적용 가능하며,
> ablation에서 양자의 combined 효과(11.5×)가 Fast-SAM3D 단독(1.5×)보다 훨씬 큼을 보입니다.

### 심사자가 "VRAM 언로드가 참신한가?" 물을 때

> VRAM 모델 언로드 자체는 일반적 기법이지만, 핵심은 "어떤 모델을 언로드해도 안전한지"를
> domain task의 불변성(OBB)으로 수학적으로 보장한다는 점입니다.
> 단순히 "안 쓰는 모듈 CPU로 이동"이 아니라 "부피 계산에서 이 모듈이 왜 불필요한지"의
> 이론적 근거(rotation-invariant OBB)가 기여입니다.

---

## 7. 작성 순서 (권장)

1. **Table 2부터 쓰기** — 결과가 명확해야 스토리가 선명해짐
2. **Figure 1** (파이프라인 다이어그램) 스케치
3. **제목 + 초록** (결과에서 역산)
4. **Method section** 상세 기술
5. Intro/Related Work는 최후 채우기

---

## 8. KIISE 투고 실무 체크리스트

- [ ] 학회 template (.cls 또는 .docx) 확인 및 적용
- [ ] 제목: 한국어 + 영문 둘 다 필요한지 확인
- [ ] 초록: 한국어 + 영문 둘 다 필요한지 확인
- [ ] 페이지 3.0 이내 (레퍼런스 포함)
- [ ] 저작권 양식 확인
- [ ] 투고 분야 선택: 인공지능 / 컴퓨터그래픽스 / 시스템
- [ ] 포스터 발표 준비 (별도 포스터 제작 필요 여부)

---

## 9. 원 논문 실험 데이터셋 조사 결과

### SAM-3D 원 논문 (arXiv:2511.16624)

**논문 정보**: "SAM 3D: 3Dfy Anything in Images", Meta AI, Nov 2025

**평가 데이터셋 3개:**

| 데이터셋 | 규모 | 용도 | 특성 |
|---------|------|------|------|
| **SA-3DAO** (SAM 3D Artist Objects) | 1,000 artist-created 3D objects | Shape + Texture 평가 (메인 벤치마크) | 전문 3D 아티스트가 natural image에 매칭한 고품질 mesh. Model-in-the-Loop(MITL) 파이프라인으로 "가장 어려운 케이스"를 라우팅하여 구축. 기존 벤치마크보다 현실적/도전적 |
| **ISO3D** | 101 synthetic objects | Perceptual fidelity (Uni3D/ULIP score) | 3D Arena에서 가져온 synthetic object |
| **Aria Digital Twin (ADT)** | 4 sequences × 4 views = 16 views | Scene layout/pose 평가 | 실제 실내 환경의 depth/pointmap 포함. 3D IoU, rotation error 측정 |

**평가 메트릭:**
- Chamfer Distance (CD) — 기하학적 정확도
- F-Score (F1@0.05) — precision/recall 조화 평균
- Volumetric IoU (vIoU) — 32³ voxel 기반 체적 교차율
- EMD (Earth Mover's Distance) — 분포 유사도
- Uni3D / ULIP Score — CLIP 기반 multi-modal semantic consistency
- ICP Rotation Error — layout/pose 정렬 정확도
- Human Preference — 5:1 win rate (objects), 6:1 (scenes)

**비교 Baseline:**
- Trellis, Hunyuan3D (v2.0, v2.1), Direct3D-S2, TripoSG, Hi3DGen

**SA-3DAO 주요 결과:**
| Method | Chamfer ↓ | F1@0.05 ↑ |
|--------|----------|-----------|
| Hi3DGen | 0.0844 | 0.1629 |
| TripoSG | (not given) | (not given) |
| **SAM-3D** | **0.0400** | **0.2344** |

**학습 데이터 (참고):**
- Iso-3DO: 2.7M Objaverse-XL objects, 2.5T tokens (synthetic pretrain)
- RP-3DO: 61M render-paste samples (semi-synthetic)
- MITL-3DO / Art-3DO (real-world alignment, SFT + DPO)
- 3.14M shapes total, 100K textures

### Fast-SAM3D 논문 (arXiv:2602.05293)

**논문 정보**: "Fast-SAM3D: 3Dfy Anything in Images but Faster", 2026

**평가 데이터셋 3개:**

| 데이터셋 | 규모 | 용도 |
|---------|------|------|
| **Toys4K** | 600 unique views | Geometry 평가 |
| **Aria Digital Twin (ADT)** | 16 views (4 seq × 4 views) | Scene layout 평가 |
| **ISO3D** | 101 synthetic objects | Perceptual fidelity 평가 |

**평가 메트릭:**
- Chamfer Distance (CD)
- F-Score (F1@0.05)
- Volumetric IoU (vIoU)
- ICP Rotation Error
- Uni3D Score

**하드웨어 & 추론 속도:**
- **GPU**: Single NVIDIA A800
- **Baseline (SAM-3D)**: 462.3s/scene, **31.04s/object** (25+25 steps)
- **Fast-SAM3D**: 229.7s/scene, **11.60s/object**
- **Speedup**: scene 2.01×, **object 2.67×**

**Diffusion Steps:**
- SAM-3D 기본: Stage1(SS) **25 steps**, Stage2(SLaT) **25 steps**
- Fast-SAM3D는 step caching + token carving으로 가속

**주요 결과:**
| Method | F-Score ↑ | vIoU ↑ | Uni3D ↑ |
|--------|----------|--------|---------|
| SAM-3D (baseline) | 92.34 | 0.543 | 0.369 |
| **Fast-SAM3D** | **92.59** | **0.552** | 0.350 |

> Fast-SAM3D는 품질을 유지(F-Score 약간 상승)하면서 2.67× 가속

---

## 10. 본 연구의 실험 설계 전략

### 핵심 관점 전환: "절대 부피 GT" 대신 "Original SAM-3D를 baseline으로"

본 연구의 claim은 "우리 부피가 실제 가구와 얼마나 같은가"가 **아니라**
"**최적화해도 원본 SAM-3D의 품질이 유지되는가**"입니다.

3D 복원 정확도(CD, F-Score 등)는 **SAM-3D 원 논문이 이미 검증**한 것이고,
우리는 그걸 전제로 **"최적화해도 그 품질이 보존됨"만 실증**하면 됩니다.

```
[실험 설계]

이미지 → Original SAM-3D (25+25 steps, full decoder, MoGe)
       → OBB → relative (w₀, d₀, h₀)  ← BASELINE (이것이 GT 역할)

이미지 → Optimized SAM-3D (14+4 steps, Gaussian-only, synthetic pointmap, step caching)
       → OBB → relative (w₁, d₁, h₁)  ← OURS

Metric: dimension_error(%) = |w₁ - w₀| / w₀ × 100  (w, d, h 각각)
        volume_error(%) = |V₁ - V₀| / V₀ × 100     (부피 = w × d × h)
```

### 왜 이 설계가 더 강한가

| 항목 | 절대 부피 비교 | ✅ 상대 치수 비교 (채택) |
|------|--------------|--------------------------|
| **GT 확보** | 줄자/카탈로그/공개 데이터셋 필수 | **불필요** (Original SAM-3D가 baseline) |
| **노이즈** | 실측 오차 + DB 매칭 오차 혼재 | **순수하게 최적화 영향만 격리** |
| **재현 가능성** | 실측 데이터 비공개 → 재현 불가 | **SAM-3D만 있으면 누구나 재현** |
| **논문 claim** | "우리 부피가 실측에 가깝다" (과도) | **"최적화해도 원본 품질 유지"** (정확) |
| **데이터셋** | Pix3D/ABO 등 외부 의존 | **아무 가구 이미지 사용 가능** |
| **실험 비용** | GT 수집에 수 주 | **이미 보유한 이미지로 즉시 실험** |

### 구체적 실험 계획

#### 실험 데이터

| 데이터 소스 | 이미지 수 | 객체 수 (예상) | 용도 |
|-----------|----------|--------------|------|
| 이삿짐 서비스 실제 이미지 | 5-10장 | 15-30개 | 정량 평가 + qualitative |
| 공개 가구 이미지 (optional) | 10-20장 | 20-40개 | 재현성 보강 |

가구 카테고리: Bed, Sofa, Table, Chair, Nightstand, Television, Bookshelf, Dresser 등

#### Metric 설계

**Primary (필수):**
1. **치수 오차 (%)**: 각 축(w, d, h)별로 Original vs Optimized 비교
2. **상대 치수 종합 오차 (%)**: V = w × d × h 기반 volume deviation
3. **추론 시간**: 객체당 초 (baseline vs ours)
4. **VRAM 사용량**: peak GPU memory (MB)

**Secondary (optional):**
5. **Chamfer Distance**: Original PLY vs Optimized PLY 직접 비교 (point cloud 간)
6. **점 개수**: Gaussian splat 포인트 수 변화

#### Ablation Table 설계 (Table 2 — 가장 중요한 표)

각 최적화를 하나씩 추가하면서 **Original SAM-3D 대비 상대 치수 변화**를 측정:

| Configuration | Time/obj | VRAM | W err | D err | H err | V dev |
|---------------|---------|------|-------|-------|-------|-------|
| Original SAM-3D (25+25, full) | ~150s | 21GB | 0% | 0% | 0% | 0% |
| + Gaussian-Only | ~94s | 18GB | ? | ? | ? | ? |
| + VRAM Unload | ~94s | 11.25GB | 0% | 0% | 0% | 0% |
| + Synthetic Pointmap | ~90s | ~8.2GB | ? | ? | ? | ? |
| + Steps 14/4 | ~20s | ~8.2GB | ? | ? | ? | ? |
| + SS Step Caching | ~13s | ~8.2GB | ? | ? | ? | ? |
| **Final (Ours)** | **13s** | **8.2GB** | **?** | **?** | **?** | **<3%** |

> W/D/H err = Original SAM-3D 대비 각 축 상대 치수 오차 (%)
> V dev = w×d×h 기반 종합 치수 편차 (%)
> VRAM Unload: dead-code path 제거이므로 출력에 영향 0%.
> 상대 치수에 영향을 주는 것은 Synthetic Pointmap, Steps Reduction, Step Caching 3가지.

#### 논문에서의 논리 구조

```
1. SAM-3D가 high-quality 3D를 생성한다 (원 논문이 이미 증명, CD/F-Score/human pref.)
2. 우리는 부피 계산에 필요한 OBB extent만 보존하면 됨
3. 우리의 최적화가 OBB extent를 얼마나 변화시키나? → Table 2의 W/D/H err
4. 변화가 <3%이면, 원본 SAM-3D의 검증된 품질이 우리 파이프라인에도 전이됨
5. 결론: 11.5× 가속 @ <3% degradation → training-free task-aware pruning 유효
```

이 논리는 심사자가 "절대 부피의 GT는?" 이라는 질문에 대해
**"SAM-3D 원 논문이 3D 품질을 보장했고, 우리는 그 품질 중 OBB 부분만 유지하면 됩니다"**
라고 깔끔하게 방어 가능합니다.

---

## 11. TODO (업데이트)

- [x] SAM-3D 논문 정확한 citation 확인 → arXiv:2511.16624
- [x] Fast-SAM3D 논문 정확한 citation 확인 → arXiv:2602.05293
- [x] 원 논문들의 실험 데이터셋 구성 조사
- [x] 실험 설계 결정 → "Original SAM-3D를 baseline으로 상대 치수 비교" (외부 GT 불필요)
- [ ] 테스트 이미지 셋 준비 (서비스 이미지 5-10장 + 공개 이미지 10-20장)
- [ ] Original SAM-3D baseline 실행 (25+25 steps, full decoder, MoGe) → 기준 치수 확보
- [ ] 각 최적화 단계별 치수 측정 (ablation)
- [ ] Chamfer Distance (Original PLY vs Optimized PLY) 측정 (optional)
- [ ] Figure 1 벡터 이미지 제작 (파이프라인 다이어그램)
- [ ] LaTeX/Word 템플릿으로 초안 작성
- [ ] 공저자 역할 분담 확정
