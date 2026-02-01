"""
Absolute Volume Calculator Tests

AbsoluteVolumeCalculator 클래스의 단위 테스트.
Backend의 FurnitureDimensionConverter.java 로직과 동일한 결과를 보장합니다.
"""

import pytest

from ai.processors import AbsoluteVolumeCalculator, AbsoluteVolumeResult
from ai.data.furniture_dimensions import (
    FURNITURE_TYPES,
    FURNITURE_LABELS,
    get_furniture_type,
    get_subtypes_for_label,
)


class TestFurnitureDimensionsData:
    """furniture_dimensions.py 데이터 검증 테스트"""

    def test_furniture_types_count(self):
        """52개 가구 타입이 정의되어 있어야 함"""
        assert len(FURNITURE_TYPES) == 52

    def test_furniture_labels_count(self):
        """29개 라벨이 정의되어 있어야 함"""
        assert len(FURNITURE_LABELS) == 29

    def test_all_label_subtypes_exist_in_types(self):
        """모든 라벨의 서브타입이 FURNITURE_TYPES에 존재해야 함"""
        for label, subtypes in FURNITURE_LABELS.items():
            for subtype in subtypes:
                assert subtype in FURNITURE_TYPES, f"{subtype} not found for label {label}"

    def test_bed_types_have_variable_height(self):
        """침대 타입은 높이가 가변(-1)이어야 함"""
        bed_subtypes = get_subtypes_for_label("BED")
        for subtype_name in bed_subtypes:
            furniture_type = get_furniture_type(subtype_name)
            assert furniture_type is not None
            assert furniture_type.height == -1, f"{subtype_name} should have variable height"

    def test_dining_table_fully_variable(self):
        """식탁은 모든 치수가 가변(-1,-1,-1)이어야 함"""
        dining_table = get_furniture_type("DEFAULT_DINING_TABLE")
        assert dining_table is not None
        assert dining_table.is_fully_variable

    def test_sofa_types_fixed_height(self):
        """소파 타입은 고정 높이(900mm)를 가져야 함"""
        sofa_subtypes = get_subtypes_for_label("SOFA")
        for subtype_name in sofa_subtypes:
            furniture_type = get_furniture_type(subtype_name)
            assert furniture_type is not None
            assert furniture_type.height == 900, f"{subtype_name} should have height 900mm"

    def test_dimension_ratio_calculation(self):
        """치수 비율 계산이 정확해야 함"""
        # THREE_SEATER_SOFA: 3000 x 1000 x 900
        sofa = get_furniture_type("THREE_SEATER_SOFA")
        assert sofa is not None
        # ratio = min(3000, 1000) / max(3000, 1000) = 1000 / 3000 = 0.333...
        expected_ratio = 1000 / 3000
        assert abs(sofa.get_dimension_ratio() - expected_ratio) < 0.001


class TestFindBestMatch:
    """find_best_match() 메서드 테스트"""

    @pytest.fixture
    def calculator(self):
        return AbsoluteVolumeCalculator()

    def test_single_subtype_label(self, calculator):
        """서브타입이 하나인 라벨은 해당 타입 반환"""
        result = calculator.find_best_match("COFFEE_TABLE", 1.0, 0.5, 0.3)
        assert result == "DEFAULT_COFFEE_TABLE"

    def test_sofa_single_seater_ratio(self, calculator):
        """1:1 비율의 소파는 SINGLE_SOFA 매칭"""
        # SINGLE_SOFA: 1000 x 1000 (ratio = 1.0)
        result = calculator.find_best_match("SOFA", 1.0, 1.0, 0.9)
        assert result == "SINGLE_SOFA"

    def test_sofa_three_seater_ratio(self, calculator):
        """3:1 비율의 소파는 THREE_SEATER_SOFA 매칭"""
        # THREE_SEATER_SOFA: 3000 x 1000 (ratio = 0.333)
        result = calculator.find_best_match("SOFA", 3.0, 1.0, 0.9)
        assert result == "THREE_SEATER_SOFA"

    def test_sofa_l_shaped_ratio(self, calculator):
        """L자형 비율의 소파는 L_SHAPED_SOFA 매칭"""
        # L_SHAPED_SOFA: 3200 x 1800 (ratio = 0.5625)
        result = calculator.find_best_match("SOFA", 3.2, 1.8, 0.9)
        assert result == "L_SHAPED_SOFA"

    def test_bed_single_ratio(self, calculator):
        """1000x2000 비율의 침대는 SINGLE_BED 매칭"""
        # SINGLE_BED: 1000 x 2000 (ratio = 0.5)
        result = calculator.find_best_match("BED", 1.0, 2.0, 0.5)
        assert result == "SINGLE_BED"

    def test_bed_king_ratio(self, calculator):
        """1600x2000 비율의 침대는 KING_SIZE_BED에 가까움"""
        # KING_SIZE_BED: 1600 x 2000 (ratio = 0.8)
        result = calculator.find_best_match("BED", 1.6, 2.0, 0.5)
        assert result == "KING_SIZE_BED"

    def test_refrigerator_small_ratio(self, calculator):
        """작은 냉장고 비율은 TOP_BOTTOM_REFRIGERATOR 매칭"""
        # TOP_BOTTOM_REFRIGERATOR: 500 x 600 (ratio = 0.833)
        result = calculator.find_best_match("REFRIGERATOR", 0.5, 0.6, 1.25)
        assert result == "TOP_BOTTOM_REFRIGERATOR"

    def test_unknown_label_returns_label(self, calculator):
        """알 수 없는 라벨은 라벨 자체를 반환"""
        result = calculator.find_best_match("UNKNOWN_LABEL", 1.0, 1.0, 1.0)
        assert result == "UNKNOWN_LABEL"


class TestCalculateAbsoluteVolumeFixedHeight:
    """고정 높이 타입의 절대 부피 계산 테스트"""

    @pytest.fixture
    def calculator(self):
        return AbsoluteVolumeCalculator()

    def test_three_seater_sofa_volume(self, calculator):
        """THREE_SEATER_SOFA 부피 계산 (고정 높이)"""
        # 표준 치수: 3000 x 1000 x 900 mm
        # 부피 = 3000 * 1000 * 900 * 1e-9 = 2.7 m³
        result = calculator.calculate_absolute_volume(
            label="SOFA",
            type_name="THREE_SEATER_SOFA",
            rel_width=3.0,
            rel_depth=1.0,
            rel_height=0.9
        )

        assert result.matched_type == "THREE_SEATER_SOFA"
        assert result.volume_m3 == pytest.approx(2.7, rel=0.01)
        # 표준 치수 반환 (short=1000, long=3000, height=900)
        assert result.width_mm == 1000.0
        assert result.depth_mm == 3000.0
        assert result.height_mm == 900.0

    def test_single_sofa_volume(self, calculator):
        """SINGLE_SOFA 부피 계산"""
        # 표준 치수: 1000 x 1000 x 900 mm
        # 부피 = 1000 * 1000 * 900 * 1e-9 = 0.9 m³
        result = calculator.calculate_absolute_volume(
            label="SOFA",
            type_name="SINGLE_SOFA",
            rel_width=1.0,
            rel_depth=1.0,
            rel_height=0.9
        )

        assert result.matched_type == "SINGLE_SOFA"
        assert result.volume_m3 == pytest.approx(0.9, rel=0.01)

    def test_refrigerator_volume(self, calculator):
        """SIDE_BY_SIDE_REFRIGERATOR 부피 계산"""
        # 표준 치수: 920 x 930 x 1800 mm
        # 부피 = 920 * 930 * 1800 * 1e-9 ≈ 1.54 m³
        result = calculator.calculate_absolute_volume(
            label="REFRIGERATOR",
            type_name="SIDE_BY_SIDE_REFRIGERATOR",
            rel_width=0.92,
            rel_depth=0.93,
            rel_height=1.8
        )

        assert result.matched_type == "SIDE_BY_SIDE_REFRIGERATOR"
        expected_volume = 920 * 930 * 1800 * 1e-9
        assert result.volume_m3 == pytest.approx(expected_volume, rel=0.01)


class TestCalculateAbsoluteVolumeVariableHeight:
    """가변 높이 타입의 절대 부피 계산 테스트"""

    @pytest.fixture
    def calculator(self):
        return AbsoluteVolumeCalculator()

    def test_single_bed_variable_height(self, calculator):
        """SINGLE_BED 부피 계산 (가변 높이)"""
        # 표준 치수: 1000 x 2000 x -1 mm
        # 상대 치수: (1.0, 2.0, 0.5)
        # 정렬 후: l1=0.5, l2=1.0, l3=2.0
        # scaleFactor = max(1000, 2000) / l3 = 2000 / 2.0 = 1000
        # actualHeight = l1 * scaleFactor = 0.5 * 1000 = 500
        # 부피 = 1000 * 2000 * 500 * 1e-9 = 1.0 m³
        result = calculator.calculate_absolute_volume(
            label="BED",
            type_name="SINGLE_BED",
            rel_width=1.0,
            rel_depth=2.0,
            rel_height=0.5
        )

        assert result.matched_type == "SINGLE_BED"
        assert result.height_mm == pytest.approx(500.0, rel=0.01)
        expected_volume = 1000 * 2000 * 500 * 1e-9
        assert result.volume_m3 == pytest.approx(expected_volume, rel=0.01)

    def test_queen_bed_variable_height(self, calculator):
        """QUEEN_SIZE_BED 부피 계산 (가변 높이)"""
        # 표준 치수: 1500 x 2000 x -1 mm
        # 상대 치수: (1.5, 2.0, 0.4)
        # 정렬 후: l1=0.4, l2=1.5, l3=2.0
        # scaleFactor = 2000 / 2.0 = 1000
        # actualHeight = 0.4 * 1000 = 400
        result = calculator.calculate_absolute_volume(
            label="BED",
            type_name="QUEEN_SIZE_BED",
            rel_width=1.5,
            rel_depth=2.0,
            rel_height=0.4
        )

        assert result.matched_type == "QUEEN_SIZE_BED"
        assert result.height_mm == pytest.approx(400.0, rel=0.01)
        expected_volume = 1500 * 2000 * 400 * 1e-9
        assert result.volume_m3 == pytest.approx(expected_volume, rel=0.01)

    def test_bed_with_high_mattress(self, calculator):
        """높은 매트리스가 있는 침대 계산"""
        # 상대 치수: (1.0, 2.0, 0.8) - 높은 매트리스
        # 정렬 후: l1=0.8, l2=1.0, l3=2.0
        # scaleFactor = 2000 / 2.0 = 1000
        # actualHeight = 0.8 * 1000 = 800
        result = calculator.calculate_absolute_volume(
            label="BED",
            type_name="SINGLE_BED",
            rel_width=1.0,
            rel_depth=2.0,
            rel_height=0.8
        )

        assert result.height_mm == pytest.approx(800.0, rel=0.01)


class TestCalculateAbsoluteVolumeFullyVariable:
    """모든 치수가 가변인 타입 테스트 (DEFAULT_DINING_TABLE)"""

    @pytest.fixture
    def calculator(self):
        return AbsoluteVolumeCalculator()

    def test_dining_table_calculation(self, calculator):
        """DEFAULT_DINING_TABLE은 기준 치수로 스케일링됨"""
        result = calculator.calculate_absolute_volume(
            label="DINING_TABLE",
            type_name="DEFAULT_DINING_TABLE",
            rel_width=1.2,
            rel_depth=0.8,
            rel_height=0.75
        )

        assert result.matched_type == "DEFAULT_DINING_TABLE"
        # 부피는 양수여야 함
        assert result.volume_m3 > 0
        # 치수가 합리적인 범위인지 확인
        assert 600 <= result.width_mm <= 2000
        assert 800 <= result.depth_mm <= 3000
        assert 700 <= result.height_mm <= 800


class TestCalculateAbsoluteVolumeFallback:
    """타입이 없거나 알 수 없는 경우의 fallback 테스트"""

    @pytest.fixture
    def calculator(self):
        return AbsoluteVolumeCalculator()

    def test_unknown_type_fallback(self, calculator):
        """알 수 없는 타입은 fallback 계산 사용"""
        result = calculator.calculate_absolute_volume(
            label="UNKNOWN",
            type_name="UNKNOWN_TYPE",
            rel_width=1.0,
            rel_depth=1.0,
            rel_height=1.0
        )

        assert result.matched_type == "UNKNOWN_TYPE"
        # fallback은 상대 치수 * 100 (cm → mm 가정)
        assert result.width_mm == pytest.approx(100.0, rel=0.01)
        assert result.depth_mm == pytest.approx(100.0, rel=0.01)
        assert result.height_mm == pytest.approx(100.0, rel=0.01)

    def test_none_type_uses_best_match(self, calculator):
        """type_name이 None이면 find_best_match 사용"""
        result = calculator.calculate_absolute_volume(
            label="SOFA",
            type_name=None,
            rel_width=3.0,
            rel_depth=1.0,
            rel_height=0.9
        )

        # 3:1 비율은 THREE_SEATER_SOFA에 매칭됨
        assert result.matched_type == "THREE_SEATER_SOFA"


class TestEdgeCases:
    """엣지 케이스 테스트"""

    @pytest.fixture
    def calculator(self):
        return AbsoluteVolumeCalculator()

    def test_zero_relative_dimension(self, calculator):
        """상대 치수에 0이 있는 경우"""
        result = calculator.calculate_absolute_volume(
            label="SOFA",
            type_name="SINGLE_SOFA",
            rel_width=1.0,
            rel_depth=0.0,  # 0
            rel_height=0.9
        )

        # 0으로 나누기 방지 확인
        assert result.volume_m3 >= 0

    def test_very_small_relative_dimensions(self, calculator):
        """매우 작은 상대 치수"""
        result = calculator.calculate_absolute_volume(
            label="SOFA",
            type_name="SINGLE_SOFA",
            rel_width=0.001,
            rel_depth=0.001,
            rel_height=0.001
        )

        assert result.matched_type == "SINGLE_SOFA"
        # 표준 치수를 사용하므로 부피는 고정
        assert result.volume_m3 > 0

    def test_result_dataclass_fields(self, calculator):
        """AbsoluteVolumeResult의 모든 필드가 올바르게 설정됨"""
        result = calculator.calculate_absolute_volume(
            label="SOFA",
            type_name="THREE_SEATER_SOFA",
            rel_width=3.0,
            rel_depth=1.0,
            rel_height=0.9
        )

        assert isinstance(result, AbsoluteVolumeResult)
        assert isinstance(result.matched_type, str)
        assert isinstance(result.width_mm, float)
        assert isinstance(result.depth_mm, float)
        assert isinstance(result.height_mm, float)
        assert isinstance(result.volume_m3, float)

    def test_rounding_precision(self, calculator):
        """결과값 반올림 정밀도 확인"""
        result = calculator.calculate_absolute_volume(
            label="SOFA",
            type_name="THREE_SEATER_SOFA",
            rel_width=3.0,
            rel_depth=1.0,
            rel_height=0.9
        )

        # 치수는 소수점 1자리
        assert result.width_mm == round(result.width_mm, 1)
        assert result.depth_mm == round(result.depth_mm, 1)
        assert result.height_mm == round(result.height_mm, 1)
        # 부피는 소수점 6자리
        assert result.volume_m3 == round(result.volume_m3, 6)


class TestPottedPlantSpecialCase:
    """DEFAULT_POTTED_PLANT 특수 케이스 (1x1x1)"""

    @pytest.fixture
    def calculator(self):
        return AbsoluteVolumeCalculator()

    def test_potted_plant_minimal_dimensions(self, calculator):
        """화분은 1x1x1 mm로 최소 부피"""
        result = calculator.calculate_absolute_volume(
            label="POTTED_PLANT",
            type_name="DEFAULT_POTTED_PLANT",
            rel_width=0.3,
            rel_depth=0.3,
            rel_height=0.5
        )

        assert result.matched_type == "DEFAULT_POTTED_PLANT"
        # 1 * 1 * height * 1e-9 → 매우 작은 부피
        assert result.volume_m3 < 0.001


class TestIntegrationWithPipeline:
    """파이프라인 통합 테스트"""

    @pytest.fixture
    def calculator(self):
        return AbsoluteVolumeCalculator()

    def test_realistic_sofa_detection(self, calculator):
        """실제 소파 탐지 결과와 유사한 입력"""
        # 실제 3D 메시에서 나온 것 같은 상대 치수
        result = calculator.calculate_absolute_volume(
            label="SOFA",
            type_name=None,  # 타입 모름
            rel_width=2.8,
            rel_depth=0.95,
            rel_height=0.85
        )

        # 비율로 THREE_SEATER_SOFA에 매칭되어야 함
        assert result.matched_type == "THREE_SEATER_SOFA"
        # 부피가 합리적인 범위
        assert 1.0 < result.volume_m3 < 5.0

    def test_realistic_bed_detection(self, calculator):
        """실제 침대 탐지 결과와 유사한 입력"""
        result = calculator.calculate_absolute_volume(
            label="BED",
            type_name=None,
            rel_width=1.45,
            rel_depth=1.95,
            rel_height=0.55
        )

        # 퀸 사이즈에 가까움
        assert result.matched_type in ["QUEEN_SIZE_BED", "DOUBLE_BED"]
        # 부피가 합리적인 범위
        assert 0.5 < result.volume_m3 < 3.0
