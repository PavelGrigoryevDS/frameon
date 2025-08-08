"""
DataFrame exploration methods for initial data examination.

Provides functionality for:
- Basic data information
- Anomaly detection
"""

from typing import TYPE_CHECKING

from frameon.dataframe.explore.anomalies import FrameOnAnomaly
from frameon.dataframe.explore.info import FrameOnInfo

if TYPE_CHECKING:
    from frameon.core.base import FrameOn


class FrameOnExplore(FrameOnInfo, FrameOnAnomaly):
    def __init__(self, parent_df: "FrameOn"):
        FrameOnInfo.__init__(self, parent_df)
        FrameOnAnomaly.__init__(self, parent_df)
