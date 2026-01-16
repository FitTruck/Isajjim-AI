"""
Stage 4: DB 대조 및 이동 가능 여부 판단

YOLO 탐지 결과를 Knowledge Base(DB)와 대조하여
is_movable(이사 시 이동 가능 여부)을 결정합니다.

CLIP 제거 - YOLO 클래스로 직접 DB 매칭
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

# AI module import
from ai.data.knowledge_base import (
    FURNITURE_DB,
    get_db_key_from_label,
    get_dimensions,
    get_base_name,
    is_movable as db_is_movable,
    get_dimensions_for_subtype,
    estimate_size_variant
)


@dataclass
class MovabilityResult:
    """이동 가능 여부 판단 결과"""
    db_key: str                    # DB 키 (예: "bed", "sofa")
    label: str                     # 한글 라벨 (예: "침대", "소파")
    subtype_name: Optional[str]    # 세부 유형명 (CLIP 제거로 항상 None)
    is_movable: bool               # 이동 가능 여부
    confidence: float              # 탐지 신뢰도
    dimensions: Optional[Dict]     # DB 참조 치수 (mm)
    reason: str                    # 판단 사유


class MovabilityChecker:
    """
    이동 가능 여부 판단기

    AI Logic Step 4: DB 대조 → is_movable 결정

    Knowledge Base에서 가구 정보를 조회하여
    이사 시 이동 가능한 물품인지 판단합니다.

    CLIP 제거 - YOLO 클래스로 직접 DB 매칭
    """

    def __init__(self):
        """Knowledge Base 로드"""
        self.db = FURNITURE_DB
        self.class_map = self._build_class_map()
        print(f"[MovabilityChecker] Loaded {len(self.db)} furniture categories")

    def _build_class_map(self) -> Dict[str, str]:
        """
        동의어 → DB 키 매핑 생성

        Returns:
            {"bed": "bed", "sofa": "sofa", "couch": "sofa", ...}
        """
        class_map = {}
        for key, info in self.db.items():
            for syn in info.get('synonyms', []):
                class_map[syn.lower()] = key
        return class_map

    def get_search_classes(self) -> List[str]:
        """
        YOLO 탐지 대상 클래스 목록 반환
        (참고용 - YOLOE는 고정 클래스 사용)

        Returns:
            동의어 리스트
        """
        classes = []
        for key, info in self.db.items():
            classes.extend(info.get('synonyms', []))
        return classes

    def get_db_key(self, detected_label: str) -> Optional[str]:
        """
        탐지된 라벨에서 DB 키를 찾습니다.

        Args:
            detected_label: YOLO가 탐지한 라벨

        Returns:
            DB 키 또는 None
        """
        return self.class_map.get(detected_label.lower())

    def check(
        self,
        db_key: str,
        confidence: float = 1.0
    ) -> MovabilityResult:
        """
        이동 가능 여부를 판단합니다.

        Args:
            db_key: DB 키 (예: "bed", "sofa")
            confidence: YOLO 탐지 신뢰도

        Returns:
            MovabilityResult
        """
        if db_key not in self.db:
            return MovabilityResult(
                db_key=db_key,
                label=db_key,
                subtype_name=None,
                is_movable=True,  # 기본값: 이동 가능
                confidence=confidence,
                dimensions=None,
                reason="DB에 정보 없음 - 기본값 적용"
            )

        db_info = self.db[db_key]
        is_mov = db_info.get('is_movable', True)
        base_name = db_info.get('base_name', db_key)
        dimensions = db_info.get('dimensions')

        return MovabilityResult(
            db_key=db_key,
            label=base_name,
            subtype_name=None,  # CLIP 제거로 서브타입 없음
            is_movable=is_mov,
            confidence=confidence,
            dimensions=dimensions,
            reason="DB 기본값"
        )

    def check_from_label(
        self,
        detected_label: str,
        confidence: float = 1.0
    ) -> MovabilityResult:
        """
        YOLO 탐지 라벨로 이동 가능 여부를 판단합니다.

        Args:
            detected_label: YOLO가 탐지한 라벨 (예: "Bed", "Sofa")
            confidence: YOLO 탐지 신뢰도

        Returns:
            MovabilityResult
        """
        db_key = self.get_db_key(detected_label)

        if db_key is None:
            return MovabilityResult(
                db_key=detected_label.lower(),
                label=detected_label,
                subtype_name=None,
                is_movable=True,
                confidence=confidence,
                dimensions=None,
                reason="DB에 없는 클래스 - 기본값 적용"
            )

        return self.check(db_key, confidence)

    def get_reference_dimensions(
        self,
        db_key: str,
        subtype_name: Optional[str] = None,
        aspect_ratio: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        DB에서 참조 치수를 가져옵니다.

        SAM-3D로 계산된 상대적 비율과 대조하여
        실제 치수를 추정할 때 사용합니다.

        Args:
            db_key: DB 키
            subtype_name: 무시됨 (하위 호환성용)
            aspect_ratio: SAM-3D에서 계산된 비율 (무시됨)

        Returns:
            {"width": 1500, "depth": 2000, "height": 450} (mm)
        """
        return get_dimensions(db_key)

    def get_furniture_info(self, db_key: str) -> Optional[Dict]:
        """
        DB에서 가구 전체 정보를 가져옵니다.

        Args:
            db_key: DB 키

        Returns:
            가구 정보 딕셔너리
        """
        return self.db.get(db_key)

    def get_subtypes(self, db_key: str) -> List[Dict]:
        """
        특정 가구의 서브타입 목록을 가져옵니다.
        (CLIP 제거로 항상 빈 리스트 반환)

        Args:
            db_key: DB 키

        Returns:
            빈 리스트
        """
        return []  # CLIP 제거로 서브타입 분류 없음

    # =========================================================================
    # 하위 호환성 메서드 (CLIP 제거 후)
    # =========================================================================

    def check_with_classification(
        self,
        db_key: str,
        classification_result: Dict = None
    ) -> MovabilityResult:
        """
        CLIP 분류 결과를 사용하여 이동 가능 여부를 판단합니다.
        (CLIP 제거로 classification_result 무시)

        Args:
            db_key: DB 키
            classification_result: 무시됨 (하위 호환성용)

        Returns:
            MovabilityResult
        """
        # CLIP 제거로 분류 결과 무시, DB 기본값만 사용
        confidence = 1.0
        if classification_result:
            confidence = classification_result.get('score', 1.0)

        return self.check(db_key, confidence)
