"""
Stage 4: DB 대조 및 이동 가능 여부 판단

CLIP에서 도출된 세부 유형을 Knowledge Base(DB)와 대조하여
is_movable(이사 시 이동 가능 여부)을 결정합니다.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# AI module import
from ai.data.knowledge_base import (
    FURNITURE_DB,
    get_dimensions_for_subtype,
    estimate_size_variant
)


@dataclass
class MovabilityResult:
    """이동 가능 여부 판단 결과"""
    db_key: str                    # DB 키 (예: "bed", "sofa")
    label: str                     # 한글 라벨 (예: "퀸 사이즈 침대")
    subtype_name: Optional[str]    # 세부 유형명
    is_movable: bool               # 이동 가능 여부
    confidence: float              # 판단 신뢰도
    dimensions: Optional[Dict]     # DB 참조 치수 (mm)
    reason: str                    # 판단 사유


class MovabilityChecker:
    """
    이동 가능 여부 판단기

    AI Logic Step 4: DB 대조 → is_movable 결정

    Knowledge Base에서 가구 정보를 조회하여
    이사 시 이동 가능한 물품인지 판단합니다.
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
            {"소파": "sofa", "쇼파": "sofa", "couch": "sofa", ...}
        """
        class_map = {}
        for key, info in self.db.items():
            for syn in info.get('synonyms', []):
                class_map[syn.lower()] = key
        return class_map

    def get_search_classes(self) -> List[str]:
        """
        YOLO-World에 설정할 검색 클래스 목록 반환

        Returns:
            ["소파", "침대", "책상", ...] 동의어 포함 리스트
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
        subtype_name: Optional[str] = None,
        confidence: float = 1.0
    ) -> MovabilityResult:
        """
        이동 가능 여부를 판단합니다.

        Args:
            db_key: DB 키 (예: "bed", "sofa")
            subtype_name: 세부 유형명 (예: "퀸 사이즈 침대")
            confidence: CLIP 분류 신뢰도

        Returns:
            MovabilityResult
        """
        if db_key not in self.db:
            return MovabilityResult(
                db_key=db_key,
                label=db_key,
                subtype_name=subtype_name,
                is_movable=True,  # 기본값: 이동 가능
                confidence=confidence,
                dimensions=None,
                reason="DB에 정보 없음 - 기본값 적용"
            )

        db_info = self.db[db_key]
        base_is_movable = db_info.get('is_movable', True)
        base_name = db_info.get('base_name', db_key)

        # 서브타입별 is_movable 확인
        final_is_movable = base_is_movable
        final_label = base_name
        dimensions = None
        reason = "DB 기본값"

        if subtype_name and 'subtypes' in db_info:
            for subtype in db_info['subtypes']:
                if subtype.get('name') == subtype_name:
                    # 서브타입에 is_movable이 명시되어 있으면 사용
                    if 'is_movable' in subtype:
                        final_is_movable = subtype['is_movable']
                        reason = f"서브타입 '{subtype_name}' 설정"
                    final_label = subtype_name
                    dimensions = subtype.get('dimensions')
                    break

        # dimensions가 없으면 기본값에서 가져오기
        if dimensions is None:
            dimensions = db_info.get('default_dimensions')

        return MovabilityResult(
            db_key=db_key,
            label=final_label,
            subtype_name=subtype_name,
            is_movable=final_is_movable,
            confidence=confidence,
            dimensions=dimensions,
            reason=reason
        )

    def check_with_classification(
        self,
        db_key: str,
        classification_result: Dict
    ) -> MovabilityResult:
        """
        CLIP 분류 결과를 사용하여 이동 가능 여부를 판단합니다.

        Args:
            db_key: DB 키
            classification_result: CLIP 분류 결과
                {"name": "...", "score": 0.85, "is_movable": True, ...}

        Returns:
            MovabilityResult
        """
        subtype_name = classification_result.get('name')
        confidence = classification_result.get('score', 1.0)

        # 분류 결과에 is_movable이 이미 있으면 사용
        if 'is_movable' in classification_result:
            db_info = self.db.get(db_key, {})
            dimensions = classification_result.get('dimensions')

            if dimensions is None:
                dimensions = get_dimensions_for_subtype(db_key, subtype_name)

            return MovabilityResult(
                db_key=db_key,
                label=subtype_name or db_info.get('base_name', db_key),
                subtype_name=subtype_name,
                is_movable=classification_result['is_movable'],
                confidence=confidence,
                dimensions=dimensions,
                reason="CLIP 분류 결과"
            )

        return self.check(db_key, subtype_name, confidence)

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
            subtype_name: 세부 유형명
            aspect_ratio: SAM-3D에서 계산된 비율 {"w": 1.0, "h": 0.5, "d": 0.66}

        Returns:
            {"width": 1500, "depth": 2000, "height": 450} (mm)
        """
        # 기본 치수 가져오기
        dimensions = get_dimensions_for_subtype(db_key, subtype_name)

        # 비율로 사이즈 변형 추정
        if aspect_ratio and dimensions:
            variant_name, matched_dims = estimate_size_variant(
                db_key, subtype_name, aspect_ratio
            )
            if matched_dims:
                dimensions = matched_dims

        return dimensions

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

        CLIP 분류 시 후보로 사용합니다.

        Args:
            db_key: DB 키

        Returns:
            서브타입 딕셔너리 리스트
        """
        info = self.db.get(db_key, {})
        return info.get('subtypes', [])
