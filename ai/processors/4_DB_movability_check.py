"""
Stage 4: DB 대조 및 한국어 라벨 매핑

YOLO 탐지 결과를 Knowledge Base(DB)와 대조하여
한국어 라벨을 반환합니다.

V2 파이프라인에서 단순화:
- is_movable 제거 (모든 탐지 객체는 이동 대상)
- dimensions 제거 (절대 부피는 백엔드에서 계산)
- CLIP 제거 - YOLO 클래스로 직접 DB 매칭
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

# AI module import
from ai.data.knowledge_base import FURNITURE_DB


@dataclass
class LabelMappingResult:
    """라벨 매핑 결과"""
    db_key: str                    # DB 키 (예: "bed", "sofa")
    label: str                     # 한글 라벨 (예: "침대", "소파")
    confidence: float              # 탐지 신뢰도
    reason: str                    # 판단 사유


# 하위 호환성을 위한 별칭
MovabilityResult = LabelMappingResult

class MovabilityChecker:
    """
    라벨 매핑기 (구 이동 가능 여부 판단기)

    AI Logic Step 4: DB 대조 → 한국어 라벨 반환

    Knowledge Base에서 가구 정보를 조회하여
    한국어 라벨을 반환합니다.

    V2 파이프라인에서 단순화:
    - is_movable 제거 (모든 탐지 객체는 이동 대상)
    - dimensions 제거 (절대 부피는 백엔드에서 계산)
    - CLIP 제거 - YOLO 클래스로 직접 DB 매칭
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
    ) -> LabelMappingResult:
        """
        DB에서 한국어 라벨을 가져옵니다.

        Args:
            db_key: DB 키 (예: "bed", "sofa")
            confidence: YOLO 탐지 신뢰도

        Returns:
            LabelMappingResult
        """
        if db_key not in self.db:
            return LabelMappingResult(
                db_key=db_key,
                label=db_key,
                confidence=confidence,
                reason="DB에 정보 없음 - 기본값 적용"
            )

        db_info = self.db[db_key]
        base_name = db_info.get('base_name', db_key)

        return LabelMappingResult(
            db_key=db_key,
            label=base_name,
            confidence=confidence,
            reason="DB 매칭 성공"
        )

    def check_from_label(
        self,
        detected_label: str,
        confidence: float = 1.0
    ) -> LabelMappingResult:
        """
        YOLO 탐지 라벨로 한국어 라벨을 가져옵니다.

        Args:
            detected_label: YOLO가 탐지한 라벨 (예: "Bed", "Sofa")
            confidence: YOLO 탐지 신뢰도

        Returns:
            LabelMappingResult
        """
        db_key = self.get_db_key(detected_label)

        if db_key is None:
            return LabelMappingResult(
                db_key=detected_label.lower(),
                label=detected_label,
                confidence=confidence,
                reason="DB에 없는 클래스 - 기본값 적용"
            )

        return self.check(db_key, confidence)

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
    # 하위 호환성 메서드 (V2 파이프라인에서 단순화)
    # =========================================================================

    def check_with_classification(
        self,
        db_key: str,
        classification_result: Dict = None
    ) -> LabelMappingResult:
        """
        [DEPRECATED] CLIP 분류 결과를 사용하여 라벨을 가져옵니다.
        (CLIP 제거로 classification_result 무시)

        Args:
            db_key: DB 키
            classification_result: 무시됨 (하위 호환성용)

        Returns:
            LabelMappingResult
        """
        # CLIP 제거로 분류 결과 무시, DB 기본값만 사용
        confidence = 1.0
        if classification_result:
            confidence = classification_result.get('score', 1.0)

        return self.check(db_key, confidence)

    def get_reference_dimensions(
        self,
        db_key: str,
        subtype_name: Optional[str] = None,
        aspect_ratio: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        [DEPRECATED] DB에서 참조 치수를 가져옵니다.

        V2 파이프라인에서 dimensions는 제거되었습니다.
        절대 부피는 백엔드에서 계산합니다.

        Returns:
            항상 None
        """
        _ = db_key, subtype_name, aspect_ratio  # 하위 호환성 시그니처 유지
        return None


# 하위 호환성을 위한 별칭
LabelMapper = MovabilityChecker
