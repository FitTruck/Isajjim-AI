"""
Dimension Strategy

절대 치수 결정 전략: Boxer 우선, 규칙기반 fallback.

벤치마크 비교를 위해 3가지 모드를 지원합니다:
- Mode A: Boxer 우선 + 규칙기반 fallback
- Mode B: Boxer only (실패 시 None)
- Mode C: 규칙기반 only (baseline, 기존 V2.5 동일)
"""

import importlib
import logging
from enum import Enum
from typing import Optional

from ai.config import Config

# 숫자로 시작하는 모듈명은 importlib 사용
_stage8 = importlib.import_module('.8_absolute_volume_calculate', package='ai.processors')
_stage9 = importlib.import_module('.9_boxer_dimension', package='ai.processors')

AbsoluteVolumeCalculator = _stage8.AbsoluteVolumeCalculator
AbsoluteVolumeResult = _stage8.AbsoluteVolumeResult
BoxerResult = _stage9.BoxerResult

logger = logging.getLogger(__name__)


class DimensionMode(Enum):
    """절대 치수 결정 모드"""
    BOXER_WITH_FALLBACK = "A"   # Boxer 우선 + 규칙기반 fallback
    BOXER_ONLY = "B"            # Boxer only
    RULE_BASED_ONLY = "C"       # 규칙기반 only (baseline)


class DimensionStrategy:
    """
    절대 치수 결정 전략.

    Boxer 추론 결과를 검증하고, 통과하면 사용합니다.
    실패/저신뢰도 시 모드에 따라 fallback 또는 None 반환.

    Usage:
        strategy = DimensionStrategy(mode=DimensionMode.BOXER_WITH_FALLBACK)
        result = strategy.resolve(
            boxer_result=boxer_res,
            label="SOFA",
            type_name="THREE_SEATER_SOFA",
            rel_width=1.2, rel_depth=2.5, rel_height=0.85
        )
    """

    def __init__(
        self,
        mode: DimensionMode = DimensionMode.BOXER_WITH_FALLBACK,
        rule_based: Optional[AbsoluteVolumeCalculator] = None,
    ):
        self.mode = mode
        self.rule_based = rule_based or AbsoluteVolumeCalculator()

        # 통계 (벤치마크용)
        self.stats = {
            "total": 0,
            "boxer_used": 0,
            "fallback_used": 0,
            "boxer_rejected_low_conf": 0,
            "boxer_rejected_invalid": 0,
            "boxer_failed": 0,
        }

    def resolve(
        self,
        boxer_result: Optional[BoxerResult],
        label: str,
        type_name: Optional[str],
        rel_width: float = 0.0,
        rel_depth: float = 0.0,
        rel_height: float = 0.0,
    ) -> Optional[AbsoluteVolumeResult]:
        """
        Boxer 결과와 규칙기반을 조합하여 최종 절대 치수를 결정합니다.

        Args:
            boxer_result: Boxer 추론 결과 (None이면 실패)
            label: 가구 라벨 (예: "SOFA")
            type_name: 서브타입 (예: "THREE_SEATER_SOFA")
            rel_width, rel_depth, rel_height: 상대 치수 (fallback용)

        Returns:
            AbsoluteVolumeResult 또는 None (Mode B에서 Boxer 실패 시)
        """
        self.stats["total"] += 1

        # Mode C: 규칙기반 only
        if self.mode == DimensionMode.RULE_BASED_ONLY:
            self.stats["fallback_used"] += 1
            return self._rule_based(label, type_name, rel_width, rel_depth, rel_height)

        # Mode A, B: Boxer 시도
        validated = self._validate_boxer(boxer_result, type_name)

        if validated is not None:
            # Boxer 성공
            self.stats["boxer_used"] += 1
            return validated

        # Boxer 실패 → 모드별 처리
        if self.mode == DimensionMode.BOXER_WITH_FALLBACK:
            self.stats["fallback_used"] += 1
            return self._rule_based(label, type_name, rel_width, rel_depth, rel_height)

        # Mode B: Boxer only → None 반환
        return None

    def _validate_boxer(
        self,
        boxer_result: Optional[BoxerResult],
        type_name: Optional[str] = None,
    ) -> Optional[AbsoluteVolumeResult]:
        """
        Boxer 결과를 검증하고 AbsoluteVolumeResult로 변환합니다.

        검증 기준:
        1. boxer_result가 None이 아님
        2. confidence >= threshold
        3. 모든 치수가 물리적 타당 범위 내

        Args:
            boxer_result: Boxer 추론 결과
            type_name: 가구 서브타입 (API 응답 호환용)

        Returns:
            검증 통과 시 AbsoluteVolumeResult, 실패 시 None
        """
        if boxer_result is None:
            self.stats["boxer_failed"] += 1
            return None

        # 신뢰도 검증
        threshold = Config.BOXER_CONFIDENCE_THRESHOLD
        if boxer_result.confidence < threshold:
            logger.info(
                f"[DimensionStrategy] Boxer confidence too low: "
                f"{boxer_result.confidence:.2f} < {threshold}"
            )
            self.stats["boxer_rejected_low_conf"] += 1
            return None

        # 물리적 타당성 검증 (mm 변환 후)
        dims_mm = [
            boxer_result.width_m * 1000,
            boxer_result.depth_m * 1000,
            boxer_result.height_m * 1000,
        ]
        min_dim = Config.BOXER_MIN_DIMENSION_MM
        max_dim = Config.BOXER_MAX_DIMENSION_MM

        for dim_mm in dims_mm:
            if dim_mm < min_dim or dim_mm > max_dim:
                logger.info(
                    f"[DimensionStrategy] Boxer dimension out of range: "
                    f"{dims_mm} (valid: {min_dim}-{max_dim}mm)"
                )
                self.stats["boxer_rejected_invalid"] += 1
                return None

        # AbsoluteVolumeResult로 변환
        width_mm = round(dims_mm[0], 1)
        depth_mm = round(dims_mm[1], 1)
        height_mm = round(dims_mm[2], 1)
        volume_m3 = round(boxer_result.volume_m3, 6)

        # matched_type: Boxer는 class-agnostic이므로 전달받은 type_name 사용
        # → backend API 응답 호환성 유지
        from ai.data.furniture_dimensions import get_valid_type_or_none
        valid_type = get_valid_type_or_none(type_name)

        return AbsoluteVolumeResult(
            matched_type=valid_type,
            width_mm=width_mm,
            depth_mm=depth_mm,
            height_mm=height_mm,
            volume_m3=volume_m3,
        )

    def _rule_based(
        self,
        label: str,
        type_name: Optional[str],
        rel_width: float,
        rel_depth: float,
        rel_height: float,
    ) -> AbsoluteVolumeResult:
        """기존 규칙기반 절대 치수 계산 (fallback)"""
        return self.rule_based.calculate_absolute_volume(
            label=label,
            type_name=type_name,
            rel_width=rel_width,
            rel_depth=rel_depth,
            rel_height=rel_height,
        )

    def get_stats_summary(self) -> str:
        """벤치마크용 통계 요약"""
        s = self.stats
        total = max(s["total"], 1)
        return (
            f"Mode={self.mode.value} | "
            f"Total={s['total']} | "
            f"Boxer={s['boxer_used']} ({s['boxer_used']*100//total}%) | "
            f"Fallback={s['fallback_used']} ({s['fallback_used']*100//total}%) | "
            f"Rejected(conf)={s['boxer_rejected_low_conf']} | "
            f"Rejected(invalid)={s['boxer_rejected_invalid']} | "
            f"Failed={s['boxer_failed']}"
        )

    def reset_stats(self):
        """통계 초기화"""
        for key in self.stats:
            self.stats[key] = 0
