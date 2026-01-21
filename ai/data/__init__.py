# AI Data Module
from .knowledge_base import (
    FURNITURE_DB,
    get_db_key_from_label,
    get_base_name,
    get_subtypes,
    get_content_labels,
    # Deprecated functions (하위 호환성)
    is_movable,
    get_dimensions,
    get_dimensions_for_subtype,
    estimate_size_variant
)

__all__ = [
    'FURNITURE_DB',
    'get_db_key_from_label',
    'get_base_name',
    'get_subtypes',
    'get_content_labels',
    # Deprecated functions (하위 호환성)
    'is_movable',
    'get_dimensions',
    'get_dimensions_for_subtype',
    'estimate_size_variant'
]
