"""OBB Packer 단위 테스트"""

import pytest
from simulation.obb_packer import (
    OBBItem,
    PlacedBox,
    ExtremePoint,
    Orientation,
    CornerPosition,
    PackingResult,
    TRUCK_PRESETS_CM,
    normalize_to_floor,
    get_rotated_dims,
    check_boundary,
    check_overlap,
    check_support,
    calculate_2d_overlap,
    generate_new_extreme_points,
    is_ep_valid,
    get_corner_position,
    try_corner_placement,
    find_corner_placement,
    extreme_points_pack,
    select_smallest_fitting_truck,
    optimize_obb,
)


class TestNormalizeToFloor:
    """바닥 보정 테스트"""

    def test_원본_치수_유지(self):
        """원본 치수가 유지되는지 확인"""
        dims = (100, 200, 50)
        result = normalize_to_floor(dims)
        assert result == (100, 200, 50)

    def test_다양한_치수_유지(self):
        """다양한 치수 조합 테스트"""
        test_cases = [
            (50, 100, 200),
            (200, 50, 100),
            (100, 100, 100),
        ]
        for dims in test_cases:
            result = normalize_to_floor(dims)
            assert result == dims


class TestCheckSupport:
    """70% 지지 규칙 테스트"""

    def test_바닥_배치_항상_지지(self):
        """바닥 배치는 항상 지지됨"""
        assert check_support(0, 0, 0, 100, 100, []) is True

    def test_지지_박스_없으면_실패(self):
        """y > 0이고 아래에 박스 없으면 실패"""
        assert check_support(0, 50, 0, 100, 100, []) is False

    def test_충분한_지지(self):
        """70% 이상 지지되면 성공"""
        box = PlacedBox(
            item_id="support",
            x=0, y=0, z=0,
            width=100, depth=100, height=50,
            orientation=0
        )
        # 위에 올릴 박스 (같은 크기)
        assert check_support(0, 50, 0, 100, 100, [box]) is True

    def test_불충분한_지지(self):
        """70% 미만 지지면 실패"""
        box = PlacedBox(
            item_id="support",
            x=0, y=0, z=0,
            width=50, depth=50, height=50,
            orientation=0
        )
        # 위에 올릴 박스 (더 큰 크기 - 지지율 25%)
        assert check_support(0, 50, 0, 100, 100, [box]) is False


class TestCheckOverlap:
    """충돌 검사 테스트"""

    def test_빈_공간_충돌_없음(self):
        """배치된 박스 없으면 충돌 없음"""
        assert check_overlap(0, 0, 0, 100, 100, 50, []) is False

    def test_겹치는_박스(self):
        """겹치는 박스 감지"""
        box = PlacedBox(
            item_id="existing",
            x=0, y=0, z=0,
            width=100, depth=100, height=50,
            orientation=0
        )
        # 완전히 겹치는 위치
        assert check_overlap(0, 0, 0, 100, 100, 50, [box]) is True

    def test_인접한_박스_충돌_없음(self):
        """인접하지만 겹치지 않는 박스"""
        box = PlacedBox(
            item_id="existing",
            x=0, y=0, z=0,
            width=100, depth=100, height=50,
            orientation=0
        )
        # X 방향으로 이동 (충돌 없음)
        assert check_overlap(100, 0, 0, 100, 100, 50, [box]) is False


class TestCheckBoundary:
    """경계 검사 테스트"""

    def test_경계_내부(self):
        """트럭 내부 배치"""
        # 트럭: 180x420x180, 박스: 100x100x50
        assert check_boundary(0, 0, 0, 100, 100, 50, 180, 420, 180) is True

    def test_경계_초과_X(self):
        """X 경계 초과"""
        assert check_boundary(100, 0, 0, 100, 100, 50, 180, 420, 180) is False

    def test_경계_초과_높이(self):
        """높이 경계 초과"""
        assert check_boundary(0, 150, 0, 100, 100, 50, 180, 420, 180) is False


class TestOptimizeOBB:
    """OBB 최적화 통합 테스트"""

    def test_단일_아이템_배치(self):
        """단일 아이템 배치"""
        items = [{"id": "box1", "width": 1.0, "depth": 1.0, "height": 0.5}]
        result = optimize_obb(items, truck_type="2.5ton", unit="m")

        assert result.success is True
        assert len(result.placed_items) == 1
        assert len(result.unplaced_items) == 0

    def test_다중_아이템_배치(self):
        """다중 아이템 배치"""
        items = [
            {"id": "bed_001", "width": 1.0, "depth": 2.0, "height": 0.5},
            {"id": "bed_002", "width": 1.4, "depth": 2.0, "height": 0.5},
            {"id": "nightstand", "width": 0.5, "depth": 0.45, "height": 0.55},
        ]
        result = optimize_obb(items, truck_type="2.5ton", unit="m")

        assert result.success is True
        assert len(result.placed_items) == 3

    def test_자동_트럭_선택(self):
        """자동 트럭 선택"""
        items = [{"id": "small", "width": 0.5, "depth": 0.5, "height": 0.3}]
        result = optimize_obb(items, truck_type=None, unit="m")

        assert result.success is True
        assert result.truck_type == "1ton"  # 가장 작은 트럭 선택

    def test_cm_단위_처리(self):
        """cm 단위 처리"""
        items = [{"id": "box1", "width": 100, "depth": 100, "height": 50}]
        result = optimize_obb(items, truck_type="2.5ton", unit="cm")

        assert result.success is True
        # cm 단위로 반환
        assert result.placed_items[0].width == 100

    def test_큰_아이템_배치_실패(self):
        """트럭보다 큰 아이템 배치 실패"""
        items = [{"id": "huge", "width": 10.0, "depth": 10.0, "height": 5.0}]
        result = optimize_obb(items, truck_type="1ton", unit="m")

        assert result.success is False
        assert len(result.unplaced_items) == 1


class TestSelectSmallestFittingTruck:
    """트럭 자동 선택 테스트"""

    def test_1톤_충분(self):
        """1톤으로 충분한 경우"""
        items = [OBBItem(id="small", original_dims=(50, 50, 30))]
        truck_type, result = select_smallest_fitting_truck(items)

        assert truck_type == "1ton"
        assert result.success is True

    def test_2_5톤_필요(self):
        """2.5톤이 필요한 경우"""
        # 1톤 트럭 크기 초과하는 아이템
        items = [OBBItem(id="medium", original_dims=(170, 290, 50))]
        truck_type, result = select_smallest_fitting_truck(items)

        assert truck_type in ["2.5ton", "5ton"]
        assert result.success is True


class TestEPGeneration:
    """Extreme Point 생성 테스트"""

    def test_새_EP_생성(self):
        """배치 후 새 EP 생성"""
        box = PlacedBox(
            item_id="test",
            x=0, y=0, z=0,
            width=100, depth=100, height=50,
            orientation=0
        )
        eps = generate_new_extreme_points(box, 180, 420, 180)

        # 3개의 EP 생성 (+X, +Z, +Y)
        assert len(eps) == 3

    def test_EP_유효성(self):
        """EP 유효성 검사"""
        box = PlacedBox(
            item_id="test",
            x=0, y=0, z=0,
            width=100, depth=100, height=50,
            orientation=0
        )
        # 박스 외부 EP - 유효
        ep_outside = ExtremePoint(x=100, y=0, z=0)
        assert is_ep_valid(ep_outside, [box]) is True

        # 박스 내부 EP - 무효
        ep_inside = ExtremePoint(x=25, y=25, z=25)
        assert is_ep_valid(ep_inside, [box]) is False


class TestCornerPosition:
    """코너 위치 계산 테스트"""

    def test_4개_코너_위치(self):
        """4개 코너의 중심 좌표 계산"""
        truck_w, truck_d = 180, 420  # 2.5톤 트럭
        w, d = 100, 100  # 객체 크기

        # 뒤쪽 왼쪽: (-truck_w/2 + w/2, 0, -truck_d/2 + d/2)
        cx, cy, cz = get_corner_position(CornerPosition.LEFT_REAR, w, d, truck_w, truck_d)
        assert cx == -90 + 50  # -40
        assert cy == 0
        assert cz == -210 + 50  # -160

        # 뒤쪽 오른쪽: (truck_w/2 - w/2, 0, -truck_d/2 + d/2)
        cx, cy, cz = get_corner_position(CornerPosition.RIGHT_REAR, w, d, truck_w, truck_d)
        assert cx == 90 - 50  # 40
        assert cy == 0
        assert cz == -210 + 50  # -160

        # 앞쪽 왼쪽: (-truck_w/2 + w/2, 0, truck_d/2 - d/2)
        cx, cy, cz = get_corner_position(CornerPosition.LEFT_FRONT, w, d, truck_w, truck_d)
        assert cx == -90 + 50  # -40
        assert cy == 0
        assert cz == 210 - 50  # 160

        # 앞쪽 오른쪽: (truck_w/2 - w/2, 0, truck_d/2 - d/2)
        cx, cy, cz = get_corner_position(CornerPosition.RIGHT_FRONT, w, d, truck_w, truck_d)
        assert cx == 90 - 50  # 40
        assert cy == 0
        assert cz == 210 - 50  # 160


class TestTryCornerPlacement:
    """코너 배치 시도 테스트"""

    def test_빈_트럭_코너_배치(self):
        """빈 트럭에 코너 배치"""
        item = OBBItem(id="test", original_dims=(100, 100, 50))
        truck_dims = TRUCK_PRESETS_CM["2.5ton"]

        result = try_corner_placement(
            item, CornerPosition.LEFT_REAR, [],
            truck_dims["width"], truck_dims["depth"], truck_dims["height"]
        )

        assert result is not None
        corner, orientation, dims, pos = result
        assert corner == CornerPosition.LEFT_REAR
        assert pos[1] == 0  # 바닥

    def test_충돌하는_코너_배치_실패(self):
        """이미 박스가 있는 코너에 배치 실패"""
        truck_dims = TRUCK_PRESETS_CM["2.5ton"]
        truck_w, truck_d, truck_h = truck_dims["width"], truck_dims["depth"], truck_dims["height"]

        # 뒤쪽 왼쪽 코너에 박스 배치
        existing_box = PlacedBox(
            item_id="existing",
            x=-truck_w/2 + 50,
            y=0,
            z=-truck_d/2 + 50,
            width=100, depth=100, height=50,
            orientation=0
        )

        item = OBBItem(id="test", original_dims=(100, 100, 50))
        result = try_corner_placement(
            item, CornerPosition.LEFT_REAR, [existing_box],
            truck_w, truck_d, truck_h
        )

        assert result is None


class TestFindCornerPlacement:
    """코너 배치 탐색 테스트"""

    def test_첫번째_가용_코너_선택(self):
        """첫 번째 가용 코너 선택 (first-match)"""
        item = OBBItem(id="test", original_dims=(100, 100, 50))
        truck_dims = TRUCK_PRESETS_CM["2.5ton"]
        truck_w, truck_d, truck_h = truck_dims["width"], truck_dims["depth"], truck_dims["height"]

        available_corners = [
            CornerPosition.LEFT_REAR,
            CornerPosition.RIGHT_REAR,
            CornerPosition.LEFT_FRONT,
            CornerPosition.RIGHT_FRONT,
        ]

        result = find_corner_placement(
            item, available_corners, [],
            truck_w, truck_d, truck_h
        )

        assert result is not None
        corner, _, _, _ = result
        assert corner == CornerPosition.LEFT_REAR  # 1순위

    def test_첫코너_불가시_다음_코너(self):
        """첫 코너 불가 시 다음 코너로 이동"""
        truck_dims = TRUCK_PRESETS_CM["2.5ton"]
        truck_w, truck_d, truck_h = truck_dims["width"], truck_dims["depth"], truck_dims["height"]

        # 뒤쪽 왼쪽에 박스 배치 (트럭 너비 180, 박스 70이면 양쪽 코너 사용 가능)
        existing_box = PlacedBox(
            item_id="existing",
            x=-truck_w/2 + 35,  # 70/2 = 35
            y=0,
            z=-truck_d/2 + 35,
            width=70, depth=70, height=50,
            orientation=0
        )

        item = OBBItem(id="test", original_dims=(70, 70, 50))
        available_corners = [
            CornerPosition.LEFT_REAR,
            CornerPosition.RIGHT_REAR,
        ]

        result = find_corner_placement(
            item, available_corners, [existing_box],
            truck_w, truck_d, truck_h
        )

        assert result is not None
        corner, _, _, _ = result
        assert corner == CornerPosition.RIGHT_REAR  # 2순위로 이동


class TestCornerFirstPacking:
    """Corner-first 배치 통합 테스트"""

    def test_4개_코너_순차_배치(self):
        """4개 아이템이 4개 코너에 순차 배치"""
        items = [
            OBBItem(id="item1", original_dims=(80, 80, 50)),
            OBBItem(id="item2", original_dims=(80, 80, 50)),
            OBBItem(id="item3", original_dims=(80, 80, 50)),
            OBBItem(id="item4", original_dims=(80, 80, 50)),
        ]
        truck_dims = TRUCK_PRESETS_CM["2.5ton"]

        result = extreme_points_pack(items, truck_dims, corner_first=True)

        assert result.success is True
        assert len(result.placed_items) == 4
        assert "코너 4개" in result.message

    def test_corner_first_비활성화(self):
        """corner_first=False 시 일반 EP 배치"""
        items = [
            OBBItem(id="item1", original_dims=(80, 80, 50)),
            OBBItem(id="item2", original_dims=(80, 80, 50)),
        ]
        truck_dims = TRUCK_PRESETS_CM["2.5ton"]

        result = extreme_points_pack(items, truck_dims, corner_first=False)

        assert result.success is True
        assert "코너" not in result.message

    def test_5개_아이템_corner_first(self):
        """5개 아이템: 4개 코너 + 1개 EP 배치"""
        items = [
            OBBItem(id="item1", original_dims=(80, 80, 50)),
            OBBItem(id="item2", original_dims=(80, 80, 50)),
            OBBItem(id="item3", original_dims=(80, 80, 50)),
            OBBItem(id="item4", original_dims=(80, 80, 50)),
            OBBItem(id="item5", original_dims=(80, 80, 50)),
        ]
        truck_dims = TRUCK_PRESETS_CM["2.5ton"]

        result = extreme_points_pack(items, truck_dims, corner_first=True)

        assert result.success is True
        assert len(result.placed_items) == 5
        assert "코너 4개" in result.message

    def test_optimize_obb_corner_first(self):
        """optimize_obb에서 corner_first 파라미터 동작"""
        items = [
            {"id": "item1", "width": 0.8, "depth": 0.8, "height": 0.5},
            {"id": "item2", "width": 0.8, "depth": 0.8, "height": 0.5},
            {"id": "item3", "width": 0.8, "depth": 0.8, "height": 0.5},
            {"id": "item4", "width": 0.8, "depth": 0.8, "height": 0.5},
        ]

        result = optimize_obb(items, truck_type="2.5ton", unit="m", corner_first=True)

        assert result.success is True
        assert len(result.placed_items) == 4
        assert "코너 4개" in result.message

    def test_corner_first_큰_아이템_우선(self):
        """큰 아이템이 먼저 코너에 배치"""
        items = [
            OBBItem(id="small", original_dims=(50, 50, 30)),
            OBBItem(id="large", original_dims=(100, 100, 80)),
        ]
        truck_dims = TRUCK_PRESETS_CM["2.5ton"]

        result = extreme_points_pack(items, truck_dims, corner_first=True)

        assert result.success is True
        # large가 먼저 배치되어야 함 (부피 정렬)
        assert result.placed_items[0].item_id == "large"


class TestRealFurnitureData:
    """실제 가구 데이터 기반 테스트 (assets PLY 파일 기준)"""

    # 실제 가구 크기 (미터 단위, routes.py _get_sample_furniture 기준)
    REAL_FURNITURE = [
        {"id": "bed_single", "width": 1.0, "depth": 1.75, "height": 0.45},      # 싱글침대
        {"id": "bed_double", "width": 1.4, "depth": 2.0, "height": 0.5},        # 더블침대
        {"id": "bed_queen", "width": 1.5, "depth": 2.0, "height": 0.55},        # 퀸침대
        {"id": "sofa_1seat", "width": 0.8, "depth": 0.85, "height": 0.75},      # 1인소파
        {"id": "sofa_2seat", "width": 1.8, "depth": 0.9, "height": 0.8},        # 2인소파
        {"id": "table_living", "width": 0.8, "depth": 0.5, "height": 0.45},     # 거실테이블
        {"id": "table_small", "width": 0.5, "depth": 0.4, "height": 0.4},       # 소형테이블
        {"id": "chair_stool", "width": 0.4, "depth": 0.4, "height": 0.45},      # 스툴
        {"id": "cabinet", "width": 0.6, "depth": 0.4, "height": 1.2},           # 수납장
        {"id": "nightstand_1", "width": 0.4, "depth": 0.4, "height": 0.55},     # 협탁1
        {"id": "nightstand_2", "width": 0.35, "depth": 0.45, "height": 0.6},    # 협탁2
        {"id": "tv_55inch", "width": 1.22, "depth": 0.07, "height": 0.7},       # 55인치TV
        {"id": "plant_large", "width": 0.35, "depth": 0.35, "height": 0.9},     # 대형화분
        {"id": "plant_medium", "width": 0.3, "depth": 0.3, "height": 0.55},     # 중형화분
    ]

    def test_전체_가구_1톤_배치(self):
        """모든 샘플 가구를 1톤 트럭에 배치 시도"""
        result = optimize_obb(
            items=self.REAL_FURNITURE,
            truck_type="1ton",
            unit="m",
            corner_first=True
        )

        # 1톤 트럭은 작아서 일부 미배치 예상
        assert len(result.placed_items) > 0
        assert result.truck_type == "1ton"

    def test_전체_가구_2_5톤_배치(self):
        """모든 샘플 가구를 2.5톤 트럭에 배치"""
        result = optimize_obb(
            items=self.REAL_FURNITURE,
            truck_type="2.5ton",
            unit="m",
            corner_first=True
        )

        # 2.5톤이면 대부분 배치 가능
        assert len(result.placed_items) >= 10
        assert result.volume_utilization > 20  # 최소 20% 이상 적재

    def test_전체_가구_자동_트럭_선택(self):
        """모든 가구 배치에 적합한 트럭 자동 선택"""
        result = optimize_obb(
            items=self.REAL_FURNITURE,
            truck_type=None,  # 자동 선택
            unit="m",
            corner_first=True
        )

        assert result.truck_type in ["1ton", "2.5ton", "5ton"]
        # 자동 선택시 모두 배치되어야 함
        if result.success:
            assert len(result.unplaced_items) == 0

    def test_침대_3개_배치(self):
        """침대 3개 (싱글, 더블, 퀸) 배치"""
        beds = [
            {"id": "bed_single", "width": 1.0, "depth": 1.75, "height": 0.45},
            {"id": "bed_double", "width": 1.4, "depth": 2.0, "height": 0.5},
            {"id": "bed_queen", "width": 1.5, "depth": 2.0, "height": 0.55},
        ]
        result = optimize_obb(beds, truck_type="2.5ton", unit="m", corner_first=True)

        assert result.success is True
        assert len(result.placed_items) == 3
        # 트럭 깊이(4.2m) 대비 침대 깊이(2.0m)로 코너 2개만 사용, 나머지는 스택
        assert "코너" in result.message

    def test_소파_테이블_조합(self):
        """소파 2개 + 테이블 2개 배치"""
        items = [
            {"id": "sofa_1seat", "width": 0.8, "depth": 0.85, "height": 0.75},
            {"id": "sofa_2seat", "width": 1.8, "depth": 0.9, "height": 0.8},
            {"id": "table_living", "width": 0.8, "depth": 0.5, "height": 0.45},
            {"id": "table_small", "width": 0.5, "depth": 0.4, "height": 0.4},
        ]
        result = optimize_obb(items, truck_type="1ton", unit="m", corner_first=True)

        assert result.success is True
        assert len(result.placed_items) == 4
        assert "코너 4개" in result.message

    def test_소형_가구_다수_배치(self):
        """소형 가구 다수 (협탁, 스툴, 화분) 배치"""
        small_items = [
            {"id": "nightstand_1", "width": 0.4, "depth": 0.4, "height": 0.55},
            {"id": "nightstand_2", "width": 0.35, "depth": 0.45, "height": 0.6},
            {"id": "chair_stool", "width": 0.4, "depth": 0.4, "height": 0.45},
            {"id": "plant_large", "width": 0.35, "depth": 0.35, "height": 0.9},
            {"id": "plant_medium", "width": 0.3, "depth": 0.3, "height": 0.55},
        ]
        result = optimize_obb(small_items, truck_type="1ton", unit="m", corner_first=True)

        assert result.success is True
        assert len(result.placed_items) == 5

    def test_TV_수직_배치(self):
        """얇은 TV의 배치 (depth=0.07m)"""
        items = [
            {"id": "tv_55inch", "width": 1.22, "depth": 0.07, "height": 0.7},
            {"id": "cabinet", "width": 0.6, "depth": 0.4, "height": 1.2},
        ]
        result = optimize_obb(items, truck_type="1ton", unit="m", corner_first=True)

        assert result.success is True
        # TV가 배치되어야 함
        tv_placed = any(p.item_id == "tv_55inch" for p in result.placed_items)
        assert tv_placed is True

    def test_corner_first_vs_ep_only_비교(self):
        """Corner-first vs EP-only 배치 비교"""
        items = [
            {"id": "bed_double", "width": 1.4, "depth": 2.0, "height": 0.5},
            {"id": "sofa_2seat", "width": 1.8, "depth": 0.9, "height": 0.8},
            {"id": "table_living", "width": 0.8, "depth": 0.5, "height": 0.45},
            {"id": "cabinet", "width": 0.6, "depth": 0.4, "height": 1.2},
        ]

        result_corner = optimize_obb(items, truck_type="2.5ton", unit="m", corner_first=True)
        result_ep = optimize_obb(items, truck_type="2.5ton", unit="m", corner_first=False)

        # 둘 다 성공해야 함
        assert result_corner.success is True
        assert result_ep.success is True

        # Corner-first는 코너 정보가 메시지에 포함
        assert "코너" in result_corner.message
        assert "코너" not in result_ep.message
