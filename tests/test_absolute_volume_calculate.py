"""
Absolute Volume Calculator 테스트

ai/processors/8_absolute_volume_calculate.py 커버리지 향상을 위한 단위 테스트
"""

import importlib
import pytest
from unittest.mock import patch, MagicMock

# 숫자로 시작하는 모듈명 import
absolute_volume_module = importlib.import_module("ai.processors.8_absolute_volume_calculate")
AbsoluteVolumeCalculator = absolute_volume_module.AbsoluteVolumeCalculator
AbsoluteVolumeResult = absolute_volume_module.AbsoluteVolumeResult
get_calculator = absolute_volume_module.get_calculator

from ai.data.furniture_dimensions import FurnitureTypeDimension


class TestAbsoluteVolumeResult:
    """AbsoluteVolumeResult 데이터클래스 테스트"""

    def test_create_result(self):
        """결과 객체 생성"""
        result = AbsoluteVolumeResult(
            matched_type="THREE_SEATER_SOFA",
            width_mm=1000.0,
            depth_mm=3000.0,
            height_mm=900.0,
            volume_m3=2.7
        )
        assert result.matched_type == "THREE_SEATER_SOFA"
        assert result.width_mm == 1000.0
        assert result.depth_mm == 3000.0
        assert result.height_mm == 900.0
        assert result.volume_m3 == 2.7

    def test_create_result_none_type(self):
        """type이 None인 결과"""
        result = AbsoluteVolumeResult(
            matched_type=None,
            width_mm=500.0,
            depth_mm=500.0,
            height_mm=500.0,
            volume_m3=0.125
        )
        assert result.matched_type is None


class TestAbsoluteVolumeCalculatorFindBestMatch:
    """find_best_match 메서드 테스트"""

    def test_find_best_match_single_subtype(self):
        """서브타입이 하나인 경우"""
        calculator = AbsoluteVolumeCalculator()

        with patch.object(absolute_volume_module, 'get_subtypes_for_label') as mock_subtypes:
            mock_subtypes.return_value = ["SINGLE_SOFA"]
            result = calculator.find_best_match("SOFA", 1.0, 2.0, 3.0)

        assert result == "SINGLE_SOFA"

    def test_find_best_match_no_subtypes(self):
        """서브타입이 없는 경우"""
        calculator = AbsoluteVolumeCalculator()

        with patch.object(absolute_volume_module, 'get_subtypes_for_label') as mock_subtypes:
            mock_subtypes.return_value = []
            result = calculator.find_best_match("UNKNOWN", 1.0, 2.0, 3.0)

        assert result == "UNKNOWN"

    def test_find_best_match_multiple_subtypes(self):
        """여러 서브타입 중 가장 유사한 것 선택"""
        calculator = AbsoluteVolumeCalculator()

        mock_type1 = MagicMock()
        mock_type1.get_dimension_ratio.return_value = 0.5  # 비율 0.5

        mock_type2 = MagicMock()
        mock_type2.get_dimension_ratio.return_value = 0.7  # 비율 0.7 (0.67에 더 가까움)

        with patch.object(absolute_volume_module, 'get_subtypes_for_label') as mock_subtypes:
            mock_subtypes.return_value = ["TYPE_A", "TYPE_B"]
            with patch.object(absolute_volume_module, 'get_furniture_type') as mock_get_type:
                def get_type_side_effect(name):
                    if name == "TYPE_A":
                        return mock_type1
                    return mock_type2

                mock_get_type.side_effect = get_type_side_effect

                # l2/l3 = 2/3 = 0.67, TYPE_A(0.5)와 TYPE_B(0.7) 중 TYPE_B가 더 가까움
                result = calculator.find_best_match("SOFA", 1.0, 2.0, 3.0)

        assert result == "TYPE_B"

    def test_find_best_match_zero_largest_dimension(self):
        """가장 큰 치수가 0인 경우"""
        calculator = AbsoluteVolumeCalculator()

        with patch.object(absolute_volume_module, 'get_subtypes_for_label') as mock_subtypes:
            mock_subtypes.return_value = ["TYPE_A"]
            result = calculator.find_best_match("SOFA", 0.0, 0.0, 0.0)

        # l3 = 0이면 detected_ratio = 0
        assert result == "TYPE_A"


class TestAbsoluteVolumeCalculatorCalculate:
    """calculate_absolute_volume 메서드 테스트"""

    def test_calculate_with_type_name(self):
        """타입 이름이 주어진 경우"""
        calculator = AbsoluteVolumeCalculator()

        mock_type = MagicMock(spec=FurnitureTypeDimension)
        mock_type.width = 1000
        mock_type.depth = 800
        mock_type.height = 400
        mock_type.is_fully_variable = False

        with patch.object(absolute_volume_module, 'get_furniture_type') as mock_get_type:
            mock_get_type.return_value = mock_type
            with patch.object(absolute_volume_module, 'get_valid_type_or_none') as mock_valid:
                mock_valid.return_value = "THREE_SEATER_SOFA"

                result = calculator.calculate_absolute_volume(
                    label="SOFA",
                    type_name="THREE_SEATER_SOFA",
                    rel_width=1.0,
                    rel_depth=2.0,
                    rel_height=3.0
                )

        assert result.matched_type == "THREE_SEATER_SOFA"
        assert result.volume_m3 > 0

    def test_calculate_without_type_name_uses_best_match(self):
        """타입 이름이 없으면 best match 사용"""
        calculator = AbsoluteVolumeCalculator()

        mock_type = MagicMock(spec=FurnitureTypeDimension)
        mock_type.width = 1000
        mock_type.depth = 800
        mock_type.height = 400
        mock_type.is_fully_variable = False

        with patch.object(calculator, 'find_best_match') as mock_find:
            mock_find.return_value = "FOUND_TYPE"
            with patch.object(absolute_volume_module, 'get_furniture_type') as mock_get_type:
                mock_get_type.return_value = mock_type
                with patch.object(absolute_volume_module, 'get_valid_type_or_none') as mock_valid:
                    mock_valid.return_value = "FOUND_TYPE"

                    result = calculator.calculate_absolute_volume(
                        label="SOFA",
                        type_name=None,
                        rel_width=1.0,
                        rel_depth=2.0,
                        rel_height=3.0
                    )

        mock_find.assert_called_once_with("SOFA", 1.0, 2.0, 3.0)
        assert result.matched_type == "FOUND_TYPE"

    def test_calculate_type_not_found_uses_fallback(self):
        """타입을 찾을 수 없으면 fallback 사용"""
        calculator = AbsoluteVolumeCalculator()

        with patch.object(absolute_volume_module, 'get_furniture_type') as mock_get_type:
            mock_get_type.return_value = None
            with patch.object(calculator, '_calculate_fallback') as mock_fallback:
                mock_fallback.return_value = AbsoluteVolumeResult(
                    matched_type=None,
                    width_mm=100.0,
                    depth_mm=200.0,
                    height_mm=300.0,
                    volume_m3=0.006
                )

                result = calculator.calculate_absolute_volume(
                    label="UNKNOWN",
                    type_name="UNKNOWN_TYPE",
                    rel_width=1.0,
                    rel_depth=2.0,
                    rel_height=3.0
                )

        mock_fallback.assert_called_once()
        assert result.matched_type is None

    def test_calculate_fully_variable_type(self):
        """완전 가변 타입 처리"""
        calculator = AbsoluteVolumeCalculator()

        mock_type = MagicMock(spec=FurnitureTypeDimension)
        mock_type.is_fully_variable = True

        with patch.object(absolute_volume_module, 'get_furniture_type') as mock_get_type:
            mock_get_type.return_value = mock_type
            with patch.object(calculator, '_calculate_fully_variable') as mock_variable:
                mock_variable.return_value = AbsoluteVolumeResult(
                    matched_type=None,
                    width_mm=800.0,
                    depth_mm=1200.0,
                    height_mm=750.0,
                    volume_m3=0.72
                )

                result = calculator.calculate_absolute_volume(
                    label="TABLE",
                    type_name="DEFAULT_DINING_TABLE",
                    rel_width=1.0,
                    rel_depth=2.0,
                    rel_height=3.0
                )

        mock_variable.assert_called_once()


class TestAbsoluteVolumeCalculatorPrivateMethods:
    """Private 메서드 테스트"""

    def test_calculate_absolute_dimensions_fixed_height(self):
        """고정 높이 계산"""
        calculator = AbsoluteVolumeCalculator()

        mock_type = MagicMock(spec=FurnitureTypeDimension)
        mock_type.width = 1000  # 장변
        mock_type.depth = 800   # 단변
        mock_type.height = 450  # 고정 높이

        width, depth, height = calculator._calculate_absolute_dimensions(
            mock_type, 1.0, 2.0, 3.0
        )

        assert height == 450  # 고정 높이 사용

    def test_calculate_absolute_dimensions_variable_height(self):
        """가변 높이 계산"""
        calculator = AbsoluteVolumeCalculator()

        mock_type = MagicMock(spec=FurnitureTypeDimension)
        mock_type.width = 1000
        mock_type.depth = 800
        mock_type.height = -1  # 가변 높이

        width, depth, height = calculator._calculate_absolute_dimensions(
            mock_type, 1.0, 2.0, 3.0
        )

        # 높이는 스케일 팩터로 계산됨
        assert height != -1
        assert height > 0

    def test_calculate_absolute_dimensions_zero_l3(self):
        """l3가 0인 경우 scale factor = 1"""
        calculator = AbsoluteVolumeCalculator()

        mock_type = MagicMock(spec=FurnitureTypeDimension)
        mock_type.width = 1000
        mock_type.depth = 800
        mock_type.height = -1

        # 모든 상대 치수가 0
        width, depth, height = calculator._calculate_absolute_dimensions(
            mock_type, 0.0, 0.0, 0.0
        )

        assert height == 0.0  # l1 * 1.0 = 0

    def test_calculate_fallback(self):
        """Fallback 계산"""
        calculator = AbsoluteVolumeCalculator()

        with patch.object(absolute_volume_module, 'get_valid_type_or_none') as mock_valid:
            mock_valid.return_value = None

            result = calculator._calculate_fallback(
                "UNKNOWN_TYPE", 1.0, 2.0, 3.0
            )

        # scale = 100
        assert result.width_mm == 100.0
        assert result.depth_mm == 200.0
        assert result.height_mm == 300.0
        assert result.matched_type is None

    def test_calculate_fully_variable(self):
        """완전 가변 계산"""
        calculator = AbsoluteVolumeCalculator()

        with patch.object(absolute_volume_module, 'get_valid_type_or_none') as mock_valid:
            mock_valid.return_value = None

            result = calculator._calculate_fully_variable(
                "DEFAULT_DINING_TABLE", 1.0, 2.0, 3.0
            )

        # 치수가 합리적인 범위 내에 있어야 함
        assert 600 <= result.width_mm <= 2000
        assert 800 <= result.depth_mm <= 3000
        assert 700 <= result.height_mm <= 800
        assert result.volume_m3 > 0

    def test_calculate_fully_variable_clamping(self):
        """완전 가변 계산 - 클램핑 테스트"""
        calculator = AbsoluteVolumeCalculator()

        with patch.object(absolute_volume_module, 'get_valid_type_or_none') as mock_valid:
            mock_valid.return_value = None

            # 극단적으로 큰 상대 치수
            result = calculator._calculate_fully_variable(
                "DEFAULT_DINING_TABLE", 100.0, 200.0, 300.0
            )

        # 클램핑 확인
        assert result.width_mm <= 2000
        assert result.depth_mm <= 3000
        assert result.height_mm <= 800

    def test_calculate_fully_variable_zero_l3(self):
        """완전 가변 계산 - l3가 0인 경우"""
        calculator = AbsoluteVolumeCalculator()

        with patch.object(absolute_volume_module, 'get_valid_type_or_none') as mock_valid:
            mock_valid.return_value = None

            result = calculator._calculate_fully_variable(
                "DEFAULT_DINING_TABLE", 0.0, 0.0, 0.0
            )

        # 최소 클램핑 값이 적용됨
        assert result.width_mm >= 600
        assert result.depth_mm >= 800
        assert result.height_mm >= 700


class TestGetCalculator:
    """get_calculator 싱글톤 함수 테스트"""

    def test_get_calculator_singleton(self):
        """싱글톤 패턴 확인"""
        # Reset singleton
        absolute_volume_module._calculator = None

        calc1 = get_calculator()
        calc2 = get_calculator()

        assert calc1 is calc2
        assert isinstance(calc1, AbsoluteVolumeCalculator)

    def test_get_calculator_reuses_instance(self):
        """인스턴스 재사용 확인"""
        # Set existing calculator
        existing = AbsoluteVolumeCalculator()
        absolute_volume_module._calculator = existing

        result = get_calculator()

        assert result is existing


class TestAbsoluteVolumeCalculatorIntegration:
    """통합 테스트"""

    def test_sofa_volume_calculation(self):
        """소파 부피 계산 통합 테스트"""
        calculator = AbsoluteVolumeCalculator()

        # 실제 데이터 기반 테스트
        result = calculator.calculate_absolute_volume(
            label="SOFA",
            type_name="THREE_SEATER_SOFA",
            rel_width=1.0,
            rel_depth=2.5,
            rel_height=0.8
        )

        # 부피가 합리적인 범위 (0.5m³ ~ 5m³)
        assert 0.1 < result.volume_m3 < 10.0

    def test_table_volume_calculation(self):
        """테이블 부피 계산 통합 테스트"""
        calculator = AbsoluteVolumeCalculator()

        result = calculator.calculate_absolute_volume(
            label="TABLE",
            type_name="DINING_TABLE_4",
            rel_width=1.0,
            rel_depth=1.5,
            rel_height=0.8
        )

        # 치수가 양수
        assert result.width_mm > 0
        assert result.depth_mm > 0
        assert result.height_mm > 0
        assert result.volume_m3 > 0

    def test_chair_volume_calculation(self):
        """의자 부피 계산 통합 테스트"""
        calculator = AbsoluteVolumeCalculator()

        result = calculator.calculate_absolute_volume(
            label="CHAIR",
            type_name="DINING_CHAIR",
            rel_width=0.5,
            rel_depth=0.5,
            rel_height=1.0
        )

        # 부피가 양수면 됨 (타입 없으면 fallback)
        assert result.volume_m3 > 0
        assert result.width_mm > 0
        assert result.depth_mm > 0
        assert result.height_mm > 0
