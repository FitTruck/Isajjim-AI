"""
Tests for ai/processors/4_DB_movability_check.py

MovabilityChecker (라벨 매핑기) 단위 테스트:
- 클래스 초기화
- DB 키 조회
- 한국어 라벨 매핑
- 하위 호환성 메서드
"""

from ai.processors import MovabilityChecker, LabelMappingResult, MovabilityResult
# LabelMapper is defined in 4_DB_movability_check.py but not exported from __init__.py
import importlib
_stage4 = importlib.import_module('.4_DB_movability_check', package='ai.processors')
LabelMapper = _stage4.LabelMapper


class TestMovabilityCheckerInit:
    """MovabilityChecker 초기화 테스트"""

    def test_init_loads_db(self):
        """DB가 로드되는지 확인"""
        checker = MovabilityChecker()
        assert checker.db is not None
        assert len(checker.db) > 0

    def test_init_builds_class_map(self):
        """클래스 맵이 생성되는지 확인"""
        checker = MovabilityChecker()
        assert checker.class_map is not None
        assert isinstance(checker.class_map, dict)
        assert len(checker.class_map) > 0

    def test_class_map_contains_synonyms(self):
        """클래스 맵에 동의어가 포함되어 있는지 확인"""
        checker = MovabilityChecker()
        # 'couch'는 'sofa'의 동의어
        assert 'couch' in checker.class_map
        assert checker.class_map['couch'] == 'sofa'


class TestMovabilityCheckerGetDbKey:
    """get_db_key 함수 테스트"""

    def test_exact_match(self):
        """정확한 매칭"""
        checker = MovabilityChecker()
        assert checker.get_db_key('bed') == 'bed'
        assert checker.get_db_key('sofa') == 'sofa'

    def test_case_insensitive(self):
        """대소문자 구분 없이 매칭"""
        checker = MovabilityChecker()
        assert checker.get_db_key('Bed') == 'bed'
        assert checker.get_db_key('BED') == 'bed'
        assert checker.get_db_key('Sofa') == 'sofa'

    def test_synonym_match(self):
        """동의어 매칭"""
        checker = MovabilityChecker()
        assert checker.get_db_key('couch') == 'sofa'
        assert checker.get_db_key('fridge') == 'refrigerator'

    def test_unknown_label_returns_none(self):
        """알 수 없는 라벨은 None 반환"""
        checker = MovabilityChecker()
        assert checker.get_db_key('unknown_item') is None
        assert checker.get_db_key('xyz123') is None


class TestMovabilityCheckerCheck:
    """check 함수 테스트"""

    def test_known_db_key(self):
        """알려진 DB 키로 조회"""
        checker = MovabilityChecker()
        result = checker.check('bed')

        assert isinstance(result, LabelMappingResult)
        assert result.db_key == 'bed'
        assert result.label == '침대'
        assert result.reason == 'DB 매칭 성공'

    def test_sofa_korean_label(self):
        """소파 한국어 라벨"""
        checker = MovabilityChecker()
        result = checker.check('sofa')

        assert result.label == '소파'

    def test_confidence_passed_through(self):
        """신뢰도가 전달되는지 확인"""
        checker = MovabilityChecker()
        result = checker.check('bed', confidence=0.85)

        assert result.confidence == 0.85

    def test_unknown_db_key_returns_default(self):
        """알 수 없는 DB 키는 기본값 반환"""
        checker = MovabilityChecker()
        result = checker.check('unknown_item')

        assert result.db_key == 'unknown_item'
        assert result.label == 'unknown_item'
        assert 'DB에 정보 없음' in result.reason


class TestMovabilityCheckerCheckFromLabel:
    """check_from_label 함수 테스트"""

    def test_yolo_label_to_korean(self):
        """YOLO 라벨 → 한국어 라벨"""
        checker = MovabilityChecker()
        result = checker.check_from_label('Bed')

        assert result.db_key == 'bed'
        assert result.label == '침대'

    def test_synonym_to_korean(self):
        """동의어 → 한국어 라벨"""
        checker = MovabilityChecker()
        result = checker.check_from_label('couch')  # sofa의 동의어

        assert result.db_key == 'sofa'
        assert result.label == '소파'

    def test_unknown_label_returns_default(self):
        """알 수 없는 라벨은 기본값 반환"""
        checker = MovabilityChecker()
        result = checker.check_from_label('UnknownItem', confidence=0.5)

        assert result.db_key == 'unknownitem'  # 소문자 변환
        assert result.label == 'UnknownItem'   # 원본 라벨 유지
        assert result.confidence == 0.5
        assert 'DB에 없는 클래스' in result.reason


class TestMovabilityCheckerGetFurnitureInfo:
    """get_furniture_info 함수 테스트"""

    def test_returns_furniture_info(self):
        """가구 정보 반환"""
        checker = MovabilityChecker()
        info = checker.get_furniture_info('bed')

        assert info is not None
        assert 'synonyms' in info
        assert 'base_name' in info

    def test_unknown_key_returns_none(self):
        """알 수 없는 키는 None 반환"""
        checker = MovabilityChecker()
        info = checker.get_furniture_info('unknown_item')

        assert info is None


class TestMovabilityCheckerGetSearchClasses:
    """get_search_classes 함수 테스트"""

    def test_returns_list(self):
        """리스트 반환"""
        checker = MovabilityChecker()
        classes = checker.get_search_classes()

        assert isinstance(classes, list)
        assert len(classes) > 0

    def test_contains_synonyms(self):
        """동의어가 포함됨"""
        checker = MovabilityChecker()
        classes = checker.get_search_classes()

        assert 'bed' in classes
        assert 'sofa' in classes
        assert 'couch' in classes


class TestMovabilityCheckerGetSubtypes:
    """get_subtypes 함수 테스트 (V2에서 항상 빈 리스트)"""

    def test_always_returns_empty_list(self):
        """V2에서 CLIP 제거로 항상 빈 리스트 반환"""
        checker = MovabilityChecker()

        assert checker.get_subtypes('bed') == []
        assert checker.get_subtypes('sofa') == []
        assert checker.get_subtypes('unknown') == []


class TestMovabilityCheckerDeprecatedMethods:
    """하위 호환성 메서드 테스트"""

    def test_check_with_classification_ignores_clip(self):
        """check_with_classification은 CLIP 결과를 무시"""
        checker = MovabilityChecker()

        # 가짜 CLIP 분류 결과
        classification_result = {
            'predicted_subtype': '싱글 침대',
            'score': 0.9
        }

        result = checker.check_with_classification('bed', classification_result)

        assert result.label == '침대'  # 기본 라벨 사용
        assert result.confidence == 0.9  # score만 사용

    def test_check_with_classification_no_result(self):
        """classification_result가 None일 때"""
        checker = MovabilityChecker()
        result = checker.check_with_classification('bed', None)

        assert result.label == '침대'
        assert result.confidence == 1.0

    def test_get_reference_dimensions_always_returns_none(self):
        """get_reference_dimensions는 항상 None 반환"""
        checker = MovabilityChecker()

        assert checker.get_reference_dimensions('bed') is None
        assert checker.get_reference_dimensions('bed', '싱글 침대') is None
        assert checker.get_reference_dimensions('bed', None, {'w': 1, 'h': 2}) is None


class TestLabelMappingResult:
    """LabelMappingResult 데이터클래스 테스트"""

    def test_create_result(self):
        """결과 생성"""
        result = LabelMappingResult(
            db_key='bed',
            label='침대',
            confidence=0.95,
            reason='테스트'
        )

        assert result.db_key == 'bed'
        assert result.label == '침대'
        assert result.confidence == 0.95
        assert result.reason == '테스트'


class TestAliases:
    """별칭 테스트"""

    def test_movability_result_alias(self):
        """MovabilityResult는 LabelMappingResult의 별칭"""
        assert MovabilityResult is LabelMappingResult

    def test_label_mapper_alias(self):
        """LabelMapper는 MovabilityChecker의 별칭"""
        assert LabelMapper is MovabilityChecker
