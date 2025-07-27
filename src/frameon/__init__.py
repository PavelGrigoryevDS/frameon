"""
frameon - A pandas extension package for enhanced data analysis.

This package extends pandas DataFrames and Series with additional methods for:
- Data quality analysis
- Anomaly detection
- Preprocessing
- Visualization
- Statistical analysis
"""
from .core.base import FrameOn, SeriesOn

from .api.utils import (
    analyze_join_keys,
    find_inconsistent_mappings,
    haversine_vectorized,
)
from frameon.utils.plotting.custom_figure import CustomFigure
from .datasets import load_dataset

from importlib.metadata import version

__version__ = version("frameon") 

__all__ = [
    'FrameOn',
    'analyze_join_keys',
    'find_inconsistent_mappings',
    'haversine_vectorized',
    'load_dataset',
    'CustomFigure'
]
