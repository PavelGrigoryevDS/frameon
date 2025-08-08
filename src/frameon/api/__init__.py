"""
API module providing utility functions for data analysis operations.

This module contains functions for:
- Analyzing join keys between datasets
- Finding inconsistent data mappings
- Geographical distance calculations
"""

from .utils import analyze_join_keys, find_inconsistent_mappings, haversine_vectorized

__all__ = [
    "analyze_join_keys",
    "find_inconsistent_mappings",
    "haversine_vectorized",
]
