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
    estimate_size_variant
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
    'estimate_size_variant'
]
