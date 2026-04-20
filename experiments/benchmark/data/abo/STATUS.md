# ABO 실험 현재 상태

## 완료
- [x] 스크립트 작성 (7개): `scripts/abo/1~6_*.py`, `scripts/evaluate/compute_abo_accuracy.py`
- [x] 코드 리뷰 1회 (code-reviewer 에이전트) — 수정 반영
- [x] 코드 리뷰 2회 (합성 데이터 유닛 테스트 8개 통과)
- [x] ABO 메타데이터 다운로드: 7792 valid listings
- [x] KB 매핑 (17 base_names): `abo_kb_mapping.json`
- [x] 500 샘플 층화 추출 → 실제 500 선택 (7 카테고리)
- [x] 이미지 + 메시 다운로드 (1000 파일, ~7.5GB)
- [x] YOLOE 마스크 전처리: 488/500 성공 (27% 클래스 mismatch)
- [x] worker 샘플 JSON: 488 샘플

## 카테고리 분포 (488)
| base_name | n |
|---|---|
| CHAIR_STOOL | 202 |
| SOFA | 158 |
| BED | 30 |
| DRAWER | 25 |
| CABINET | 25 |
| DESK | 25 |
| DISPLAY_SHELF | 23 |

## 블로커 (진행 중)
- pytorch3d + nvdiffrast 빌드 필요 (SAM-3D 의존성)
- 1차 빌드 실패: CUDA 12.8 vs glibc 헤더 충돌 (`cospi` 함수 선언 mismatch)
- 2차 빌드 실패: CUDA 12.4 nvcc 설치했으나 시스템 gcc>13 으로 거부
- **현재**: conda gcc-13 설치 후 3차 빌드 진행 중 (ninja -j 4)

## 남은 작업
- [ ] pytorch3d 빌드 완료 (진행 중, 예상 ~15-25분)
- [ ] nvdiffrast 빌드 (~2분)
- [ ] SAM-3D `abo_proposed` 488 샘플 (예상 ~50분)
- [ ] SAM-3D `abo_baseline_b` 488 샘플 (예상 ~75-90분)
- [ ] `compute_abo_accuracy.py` 실행 → 표 2, 표 3 생성

## 재개 명령
```bash
bash /tmp/resume_after_install.sh
```

## 설치 로그
- 1차: `/tmp/claude-1000/-home-rladlems1031-Isajjim-AI/*/tasks/bmf6x4m8m.output`
- 2차: `/tmp/claude-1000/-home-rladlems1031-Isajjim-AI/*/tasks/bki3cz22w.output`
- 3차 (현재): `/tmp/claude-1000/-home-rladlems1031-Isajjim-AI/*/tasks/bdlci1mqw.output`
