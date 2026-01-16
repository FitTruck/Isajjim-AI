# DeCl Data Module
from .knowledge_base import (
    FURNITURE_DB,
    get_dimensions_for_subtype,
    estimate_size_variant,
    get_db_key_from_label,
    get_furniture_info,
    get_dimensions,
    is_movable,
    get_base_name,
    get_all_synonyms
)

__all__ = [
    'FURNITURE_DB',
    'get_dimensions_for_subtype',
    'estimate_size_variant',
    'get_db_key_from_label',
    'get_furniture_info',
    'get_dimensions',
    'is_movable',
    'get_base_name',
    'get_all_synonyms'
]
