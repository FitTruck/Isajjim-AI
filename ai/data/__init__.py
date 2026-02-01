# AI Data Module
from .knowledge_base import (
    FURNITURE_DB,
    # Core functions
    get_db_key_from_label,
    get_base_name,
    get_subtypes,
    get_min_confidence,
    get_all_synonyms,
    # Deprecated functions (하위 호환성)
    get_content_labels,
    is_movable,
    get_dimensions,
    get_dimensions_for_subtype,
    estimate_size_variant,
    get_excluded_base_names,
    get_excluded_subtype_names
)

from .furniture_dimensions import (
    FurnitureTypeDimension,
    FURNITURE_TYPES,
    FURNITURE_LABELS,
    get_furniture_type,
    get_subtypes_for_label,
    get_first_subtype_for_label,
    get_dimension_for_label,
)

__all__ = [
    'FURNITURE_DB',
    # Core functions
    'get_db_key_from_label',
    'get_base_name',
    'get_subtypes',
    'get_min_confidence',
    'get_all_synonyms',
    # Deprecated functions (하위 호환성)
    'get_content_labels',
    'is_movable',
    'get_dimensions',
    'get_dimensions_for_subtype',
    'estimate_size_variant',
    'get_excluded_base_names',
    'get_excluded_subtype_names',
    # Furniture Dimensions (절대 부피 계산용)
    'FurnitureTypeDimension',
    'FURNITURE_TYPES',
    'FURNITURE_LABELS',
    'get_furniture_type',
    'get_subtypes_for_label',
    'get_first_subtype_for_label',
    'get_dimension_for_label',
]
