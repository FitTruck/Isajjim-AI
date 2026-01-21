"""
Tests for ai/data/knowledge_base.py

Knowledge Base 모듈의 단위 테스트:
- FURNITURE_DB 구조 검증
- 헬퍼 함수 테스트
- Deprecated 함수 하위 호환성 테스트
"""

import pytest
from ai.data.knowledge_base import (
    FURNITURE_DB,
    get_db_key_from_label,
    get_base_name,
    get_subtypes,
    get_content_labels,
    is_movable,
    get_dimensions,
    get_dimensions_for_subtype,
    estimate_size_variant,
)


class TestFurnitureDB:
    """FURNITURE_DB 구조 검증 테스트"""

    def test_furniture_db_is_dict(self):
        """FURNITURE_DB가 딕셔너리인지 확인"""
        assert isinstance(FURNITURE_DB, dict)

    def test_furniture_db_not_empty(self):
        """FURNITURE_DB가 비어있지 않은지 확인"""
        assert len(FURNITURE_DB) > 0

    def test_all_entries_have_synonyms(self):
        """모든 항목에 synonyms가 있는지 확인"""
        for key, info in FURNITURE_DB.items():
            assert "synonyms" in info, f"'{key}'에 synonyms가 없습니다"
            assert isinstance(info["synonyms"], list)
            assert len(info["synonyms"]) > 0

    def test_all_entries_have_base_name(self):
        """모든 항목에 base_name(한국어 라벨)이 있는지 확인"""
        for key, info in FURNITURE_DB.items():
            assert "base_name" in info, f"'{key}'에 base_name이 없습니다"
            assert isinstance(info["base_name"], str)
            assert len(info["base_name"]) > 0

    def test_known_furniture_types_exist(self):
        """알려진 가구 타입이 존재하는지 확인"""
        expected_keys = ["bed", "sofa", "desk", "chair", "refrigerator", "tv"]
        for key in expected_keys:
            assert key in FURNITURE_DB, f"'{key}'가 FURNITURE_DB에 없습니다"

    def test_subtypes_structure(self):
        """subtypes 구조가 올바른지 확인"""
        for key, info in FURNITURE_DB.items():
            if "subtypes" in info:
                assert isinstance(info["subtypes"], list)
                for subtype in info["subtypes"]:
                    assert "name" in subtype, f"'{key}'의 subtype에 name이 없습니다"
                    assert "prompt" in subtype, f"'{key}'의 subtype에 prompt가 없습니다"


class TestGetDbKeyFromLabel:
    """get_db_key_from_label 함수 테스트"""

    def test_exact_match(self):
        """정확한 매칭 테스트"""
        assert get_db_key_from_label("bed") == "bed"
        assert get_db_key_from_label("sofa") == "sofa"

    def test_case_insensitive(self):
        """대소문자 구분 없이 매칭"""
        assert get_db_key_from_label("Bed") == "bed"
        assert get_db_key_from_label("BED") == "bed"
        assert get_db_key_from_label("Sofa") == "sofa"

    def test_synonym_match(self):
        """동의어 매칭 테스트"""
        assert get_db_key_from_label("couch") == "sofa"  # sofa의 동의어
        assert get_db_key_from_label("fridge") == "refrigerator"  # refrigerator의 동의어
        assert get_db_key_from_label("armchair") == "chair"  # chair의 동의어

    def test_unknown_label_returns_none(self):
        """알 수 없는 라벨은 None 반환"""
        assert get_db_key_from_label("unknown_item") is None
        assert get_db_key_from_label("xyz123") is None
        assert get_db_key_from_label("") is None


class TestGetBaseName:
    """get_base_name 함수 테스트"""

    def test_returns_korean_label(self):
        """한국어 라벨 반환 테스트"""
        assert get_base_name("bed") == "침대"
        assert get_base_name("sofa") == "소파"
        assert get_base_name("desk") == "책상"
        assert get_base_name("chair") == "의자/스툴"
        assert get_base_name("refrigerator") == "냉장고"

    def test_unknown_key_returns_key(self):
        """알 수 없는 키는 키 자체를 반환"""
        assert get_base_name("unknown_item") == "unknown_item"
        assert get_base_name("xyz123") == "xyz123"


class TestGetSubtypes:
    """get_subtypes 함수 테스트"""

    def test_returns_subtypes_list(self):
        """서브타입 리스트 반환 테스트"""
        subtypes = get_subtypes("bed")
        assert isinstance(subtypes, list)
        assert len(subtypes) > 0
        # 침대 서브타입 확인
        names = [s["name"] for s in subtypes]
        assert "싱글 침대" in names
        assert "퀸 사이즈 침대" in names

    def test_sofa_subtypes(self):
        """소파 서브타입 테스트"""
        subtypes = get_subtypes("sofa")
        names = [s["name"] for s in subtypes]
        assert "1인용 소파" in names
        assert "3인용 소파" in names
        assert "L자형 소파" in names

    def test_unknown_key_returns_empty_list(self):
        """알 수 없는 키는 빈 리스트 반환"""
        assert get_subtypes("unknown_item") == []

    def test_no_subtypes_returns_empty_list(self):
        """서브타입이 없는 가구는 빈 리스트 반환"""
        # dryer는 서브타입이 없음
        assert get_subtypes("dryer") == []


class TestGetContentLabels:
    """get_content_labels 함수 테스트"""

    def test_kitchen_cabinet_content_labels(self):
        """주방 캐비닛 내용물 라벨 테스트"""
        labels = get_content_labels("kitchen cabinet")
        assert isinstance(labels, list)
        assert "dish" in labels
        assert "plate" in labels
        assert "cup" in labels

    def test_bookshelf_content_labels(self):
        """책장 내용물 라벨 테스트"""
        labels = get_content_labels("bookshelf")
        assert "book" in labels

    def test_no_content_labels_returns_empty_list(self):
        """내용물 라벨이 없는 가구는 빈 리스트 반환"""
        assert get_content_labels("bed") == []
        assert get_content_labels("sofa") == []

    def test_unknown_key_returns_empty_list(self):
        """알 수 없는 키는 빈 리스트 반환"""
        assert get_content_labels("unknown_item") == []


class TestDeprecatedFunctions:
    """Deprecated 함수 하위 호환성 테스트"""

    def test_is_movable_always_returns_true(self):
        """is_movable은 항상 True 반환 (V2에서 deprecated)"""
        assert is_movable("bed") is True
        assert is_movable("sofa") is True
        assert is_movable("unknown_item") is True

    def test_get_dimensions_always_returns_none(self):
        """get_dimensions는 항상 None 반환 (V2에서 deprecated)"""
        assert get_dimensions("bed") is None
        assert get_dimensions("sofa") is None
        assert get_dimensions("unknown_item") is None

    def test_get_dimensions_for_subtype_always_returns_none(self):
        """get_dimensions_for_subtype은 항상 None 반환 (V2에서 deprecated)"""
        assert get_dimensions_for_subtype("bed", "싱글 침대") is None
        assert get_dimensions_for_subtype("sofa", "3인용 소파") is None

    def test_estimate_size_variant_always_returns_medium(self):
        """estimate_size_variant는 항상 'medium' 반환 (V2에서 deprecated)"""
        assert estimate_size_variant("bed", {"w": 1, "h": 1}) == "medium"
        assert estimate_size_variant("sofa", {"w": 2, "h": 1}) == "medium"
