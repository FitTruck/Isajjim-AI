"""
Absolute Volume Calculator

Backend의 FurnitureDimensionConverter.java 로직을 Python으로 포팅.
상대 치수(OBB 기반)를 절대 치수(mm)와 부피(m³)로 변환합니다.

Ported from:
- Isajjim-Backend/src/main/java/kr/co/isajjim/domains/furniture/domain/service/FurnitureDimensionConverter.java
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from ai.data.furniture_dimensions import (
    FurnitureTypeDimension,
    get_furniture_type,
    get_subtypes_for_label,
    get_valid_type_or_none,
)

logger = logging.getLogger(__name__)


@dataclass
class AbsoluteVolumeResult:
    """
    절대 부피 계산 결과

    Attributes:
        matched_type: 매칭된 가구 타입 이름 (예: "THREE_SEATER_SOFA")
                      백엔드에 없는 타입(DEFAULT_* 등)은 None
        width_mm: 절대 가로 길이 (mm)
        depth_mm: 절대 세로 길이 (mm)
        height_mm: 절대 높이 (mm)
        volume_m3: 절대 부피 (m³)
    """
    matched_type: Optional[str]
    width_mm: float
    depth_mm: float
    height_mm: float
    volume_m3: float


class AbsoluteVolumeCalculator:
    """
    절대 부피 계산기

    상대 치수(3D 메시 OBB 기반)를 표준 가구 치수와 비교하여
    절대 치수와 부피를 계산합니다.

    알고리즘:
    1. 상대 치수를 정렬 (l1 < l2 < l3)
    2. 표준 치수에서 장단변 추출
    3. height == -1 (가변 높이)이면:
       - scaleFactor = standardLong / l3
       - actualHeight = l1 * scaleFactor
    4. volume = short * long * height * 1e-9 (mm³ → m³)
    """

    def find_best_match(
        self,
        label: str,
        rel_width: float,
        rel_depth: float,
        rel_height: float
    ) -> str:
        """
        비율이 가장 유사한 서브타입을 찾습니다.

        Args:
            label: 가구 라벨 (예: "SOFA")
            rel_width: 상대 가로 길이
            rel_depth: 상대 세로 길이
            rel_height: 상대 높이

        Returns:
            가장 유사한 서브타입 이름

        Java 원본:
        ```java
        List<Double> sorted = Stream.of(info.width(), info.depth(), info.height()).sorted().toList();
        double detectedRatio = sorted.get(1) / sorted.get(2);

        return candidates.stream()
            .min(Comparator.comparingDouble(type ->
                Math.abs(type.getDimensionRatio() - detectedRatio)))
            .orElse(null);
        ```
        """
        subtypes = get_subtypes_for_label(label)

        if not subtypes:
            logger.warning(f"No subtypes found for label: {label}")
            return label

        if len(subtypes) == 1:
            return subtypes[0]

        # 상대 치수 정렬 (l1 < l2 < l3)
        sorted_dims = sorted([rel_width, rel_depth, rel_height])
        l2, l3 = sorted_dims[1], sorted_dims[2]

        # 탐지된 비율 계산 (middle / largest)
        detected_ratio = l2 / l3 if l3 > 0 else 0

        # 가장 유사한 비율의 서브타입 찾기
        best_match = subtypes[0]
        min_diff = float('inf')

        for subtype_name in subtypes:
            furniture_type = get_furniture_type(subtype_name)
            if furniture_type is None:
                continue

            type_ratio = furniture_type.get_dimension_ratio()
            diff = abs(type_ratio - detected_ratio)

            if diff < min_diff:
                min_diff = diff
                best_match = subtype_name

        logger.debug(
            f"find_best_match: label={label}, detected_ratio={detected_ratio:.3f}, "
            f"best_match={best_match}, diff={min_diff:.3f}"
        )

        return best_match

    def calculate_absolute_volume(
        self,
        label: str,
        type_name: Optional[str],
        rel_width: float,
        rel_depth: float,
        rel_height: float
    ) -> AbsoluteVolumeResult:
        """
        절대 치수와 부피를 계산합니다.

        Args:
            label: 가구 라벨 (예: "SOFA")
            type_name: 가구 타입 이름 (예: "THREE_SEATER_SOFA"), None이면 find_best_match 사용
            rel_width: 상대 가로 길이
            rel_depth: 상대 세로 길이
            rel_height: 상대 높이

        Returns:
            AbsoluteVolumeResult: 절대 치수와 부피

        Java 원본:
        ```java
        public Double calculateAbsoluteVolume(AIAnalysisResponse.FurnitureInfo info, FurnitureType matchedType) {
            List<Double> sortedRelative = Stream.of(info.width(), info.depth(), info.height())
                    .sorted()
                    .toList();

            double l1 = sortedRelative.get(0);
            double l2 = sortedRelative.get(1);
            double l3 = sortedRelative.get(2);

            double standardLongSide = Math.max(matchedType.getWidth(), matchedType.getDepth());
            double standardShortSide = Math.min(matchedType.getWidth(), matchedType.getDepth());

            double actualHeight;
            if (matchedType.getHeight() != -1) {
                actualHeight = matchedType.getHeight();
            } else {
                double scaleFactor = standardLongSide / l3;
                actualHeight = l1 * scaleFactor;
            }

            double volumeM3 = standardShortSide * standardLongSide * actualHeight;
            return volumeM3 * 1e-9;
        }
        ```
        """
        # 타입이 없으면 best match 찾기
        matched_type = type_name
        if not matched_type:
            matched_type = self.find_best_match(label, rel_width, rel_depth, rel_height)

        # 가구 타입 정보 가져오기
        furniture_type = get_furniture_type(matched_type)

        # 타입을 찾을 수 없는 경우 fallback
        if furniture_type is None:
            logger.warning(f"Furniture type not found: {matched_type}, using default calculation")
            return self._calculate_fallback(matched_type, rel_width, rel_depth, rel_height)

        # 모든 치수가 가변인 경우 (DEFAULT_DINING_TABLE 등)
        if furniture_type.is_fully_variable:
            return self._calculate_fully_variable(matched_type, rel_width, rel_depth, rel_height)

        # 절대 치수 계산
        width_mm, depth_mm, height_mm = self._calculate_absolute_dimensions(
            furniture_type, rel_width, rel_depth, rel_height
        )

        # 부피 계산 (mm³ → m³)
        volume_m3 = width_mm * depth_mm * height_mm * 1e-9

        # 백엔드에 없는 타입은 None으로 변환
        valid_type = get_valid_type_or_none(matched_type)

        logger.debug(
            f"calculate_absolute_volume: type={matched_type} -> valid_type={valid_type}, "
            f"dims=({width_mm:.0f}, {depth_mm:.0f}, {height_mm:.0f})mm, "
            f"volume={volume_m3:.3f}m³"
        )

        return AbsoluteVolumeResult(
            matched_type=valid_type,
            width_mm=round(width_mm, 1),
            depth_mm=round(depth_mm, 1),
            height_mm=round(height_mm, 1),
            volume_m3=round(volume_m3, 6)
        )

    def _calculate_absolute_dimensions(
        self,
        furniture_type: FurnitureTypeDimension,
        rel_width: float,
        rel_depth: float,
        rel_height: float
    ) -> Tuple[float, float, float]:
        """
        표준 치수를 기반으로 절대 치수를 계산합니다.

        Returns:
            (width_mm, depth_mm, height_mm) 튜플

        Note:
            표준 치수(width, depth, height)를 그대로 유지합니다.
            높이가 가변(-1)인 경우에만 상대 치수 비율로 계산합니다.
            이렇게 하면 TV처럼 납작한 가구도 올바른 치수로 반환됩니다.
        """
        # 표준 치수 사용
        actual_width = furniture_type.width
        actual_depth = furniture_type.depth

        # 높이 계산
        if furniture_type.height != -1:
            # 고정 높이 사용
            actual_height = furniture_type.height
        else:
            # 가변 높이: 상대 치수 비율로 스케일 계산
            # 상대 치수 정렬 (l1 < l2 < l3)
            sorted_relative = sorted([rel_width, rel_depth, rel_height])
            l1 = sorted_relative[0]  # 가장 작은 값
            l3 = sorted_relative[2]  # 가장 큰 값

            # 표준 장변 기준 스케일 팩터
            standard_long_side = max(furniture_type.width, furniture_type.depth)
            scale_factor = standard_long_side / l3 if l3 > 0 else 1.0
            actual_height = l1 * scale_factor

        return actual_width, actual_depth, actual_height

    def _calculate_fallback(
        self,
        type_name: str,
        rel_width: float,
        rel_depth: float,
        rel_height: float
    ) -> AbsoluteVolumeResult:
        """
        타입 정보가 없을 때 fallback 계산

        상대 치수를 그대로 사용하고, 단위를 mm로 가정합니다.
        이 경우 부피가 매우 작을 수 있으므로 경고 로그를 남깁니다.
        """
        # 상대 치수를 100 배율로 스케일 (cm → mm 가정)
        scale = 100.0
        width_mm = rel_width * scale
        depth_mm = rel_depth * scale
        height_mm = rel_height * scale

        volume_m3 = width_mm * depth_mm * height_mm * 1e-9

        # 백엔드에 없는 타입은 None으로 변환
        valid_type = get_valid_type_or_none(type_name)

        logger.warning(
            f"Using fallback calculation for {type_name} -> valid_type={valid_type}: "
            f"dims=({width_mm:.0f}, {depth_mm:.0f}, {height_mm:.0f})mm, "
            f"volume={volume_m3:.6f}m³"
        )

        return AbsoluteVolumeResult(
            matched_type=valid_type,
            width_mm=round(width_mm, 1),
            depth_mm=round(depth_mm, 1),
            height_mm=round(height_mm, 1),
            volume_m3=round(volume_m3, 6)
        )

    def _calculate_fully_variable(
        self,
        type_name: str,
        rel_width: float,
        rel_depth: float,
        rel_height: float
    ) -> AbsoluteVolumeResult:
        """
        모든 치수가 가변인 경우 계산 (DEFAULT_DINING_TABLE 등)

        일반적인 식탁 기준 치수를 사용하여 스케일 팩터를 계산합니다.
        기준: 1200mm x 800mm x 750mm (4인용 식탁)
        """
        # 기준 치수 (4인용 식탁: 1200mm x 800mm x 750mm)
        reference_long = 1200.0

        # 상대 치수 정렬
        sorted_relative = sorted([rel_width, rel_depth, rel_height])
        l1 = sorted_relative[0]
        l2 = sorted_relative[1]
        l3 = sorted_relative[2]

        # 스케일 팩터 계산 (가장 긴 변 기준)
        scale_factor = reference_long / l3 if l3 > 0 else 1.0

        # 절대 치수 계산
        width_mm = l2 * scale_factor  # 중간값 → 단변
        depth_mm = l3 * scale_factor  # 최대값 → 장변
        height_mm = l1 * scale_factor  # 최소값 → 높이

        # 합리적인 범위로 클램핑
        width_mm = max(600, min(width_mm, 2000))
        depth_mm = max(800, min(depth_mm, 3000))
        height_mm = max(700, min(height_mm, 800))

        volume_m3 = width_mm * depth_mm * height_mm * 1e-9

        # 백엔드에 없는 타입은 None으로 변환
        valid_type = get_valid_type_or_none(type_name)

        logger.debug(
            f"Fully variable calculation for {type_name} -> valid_type={valid_type}: "
            f"dims=({width_mm:.0f}, {depth_mm:.0f}, {height_mm:.0f})mm, "
            f"volume={volume_m3:.3f}m³"
        )

        return AbsoluteVolumeResult(
            matched_type=valid_type,
            width_mm=round(width_mm, 1),
            depth_mm=round(depth_mm, 1),
            height_mm=round(height_mm, 1),
            volume_m3=round(volume_m3, 6)
        )


# 편의를 위한 모듈 레벨 인스턴스
_calculator: Optional[AbsoluteVolumeCalculator] = None


def get_calculator() -> AbsoluteVolumeCalculator:
    """
    싱글톤 패턴으로 AbsoluteVolumeCalculator 인스턴스 반환
    """
    global _calculator
    if _calculator is None:
        _calculator = AbsoluteVolumeCalculator()
    return _calculator
