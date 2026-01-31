# PLY 객체 축 정렬 가이드

## 개요

3D 스캔 또는 SAM-3D로 생성된 PLY 파일은 임의의 방향으로 회전되어 있을 수 있습니다.
이 문서는 Open3D의 OBB(Oriented Bounding Box)를 사용하여 객체를 축 정렬하는 방법을 설명합니다.

## 문제점

정렬되지 않은 PLY 객체의 문제:
- 객체가 대각선으로 기울어져 보임
- 모서리로 서있음 (면이 아닌)
- 바닥에 붙지 않고 떠있거나 묻혀있음
- 시뮬레이션에서 비현실적인 배치

## 해결 방법: OBB 기반 축 정렬

### 알고리즘

```python
import open3d as o3d

def align_object_to_floor(ply_path: str):
    # 1. PLY 파일 로드
    pcd = o3d.io.read_point_cloud(ply_path)

    # 2. OBB(Oriented Bounding Box) 계산
    obb = pcd.get_oriented_bounding_box()

    # 3. OBB 회전 행렬의 역행렬(전치)로 회전 → 축 정렬
    center = pcd.get_center()
    pcd.rotate(obb.R.T, center=center)

    # 4. 바닥에 놓기 (Z-min = 0)
    aabb = pcd.get_axis_aligned_bounding_box()
    z_min = aabb.get_min_bound()[2]
    pcd.translate((0, 0, -z_min))

    return pcd
```

### 원리

1. **OBB 계산**: 포인트 클라우드를 가장 타이트하게 감싸는 회전된 박스를 찾음
2. **역회전 적용**: OBB의 회전 행렬 `R`의 전치(역행렬) `R.T`를 적용하여 축 정렬
3. **바닥 이동**: Z 최소값이 0이 되도록 이동하여 바닥에 배치

### 좌표계

- **Open3D**: Z-up (Z축이 위)
- **Three.js**: Y-up (Y축이 위)

Three.js에서 사용 시 X축 기준 -90도 회전 필요:
```javascript
geometry.rotateX(-Math.PI / 2);
```

## 사용 방법

### 1. 환경 설정

```bash
# Python 3.11 환경 생성 (open3d는 3.13 미지원)
conda create -n ply_align python=3.11 -y
conda activate ply_align
pip install open3d numpy
```

### 2. 단일 파일 정렬

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("input.ply")
obb = pcd.get_oriented_bounding_box()
pcd.rotate(obb.R.T, center=pcd.get_center())
z_min = pcd.get_axis_aligned_bounding_box().get_min_bound()[2]
pcd.translate((0, 0, -z_min))
o3d.io.write_point_cloud("output_aligned.ply", pcd)
```

### 3. 배치 정렬

```bash
# simulation 디렉토리에서 실행
conda run -n ply_align python align_ply.py
```

정렬된 파일은 `assets/aligned/` 디렉토리에 저장됩니다.

## 파일 구조

```
simulation/
├── assets/
│   ├── *.ply              # 원본 PLY 파일
│   └── aligned/           # 정렬된 PLY 파일
│       └── *.ply
├── align_ply.py           # 정렬 스크립트
└── docs/
    └── PLY_ALIGNMENT.md   # 이 문서
```

## 정렬 결과 확인

정렬 스크립트 실행 시 각 파일의 결과가 출력됩니다:

```
Processing: 12_SOFA_0.ply
  -> Saved: 12_SOFA_0.ply
     Size: 0.950 x 0.557 x 0.341
     Min Z: 0.000000 (should be ~0)
```

- **Size**: 정렬 후 AABB 크기 (X × Y × Z)
- **Min Z**: 바닥 위치 (0에 가까워야 함)

## 주의사항

1. **대칭 객체**: OBB는 객체의 주축을 찾지만, 대칭 객체는 여러 정렬이 가능할 수 있음
2. **비정형 객체**: 복잡한 형태의 객체는 OBB 정렬이 예상과 다를 수 있음
3. **파일 크기**: Open3D는 대용량 PLY 파일 처리 시 메모리 사용량이 높을 수 있음

## 관련 파일

- `simulation/align_ply.py` - 배치 정렬 스크립트
- `simulation/static/simulator.html` - Three.js 시뮬레이터 (좌표계 변환 포함)
- `simulation/routes.py` - PLY 파일 서빙 엔드포인트
