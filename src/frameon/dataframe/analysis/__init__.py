"""
DataFrame analysis methods for advanced data examination.

Provides functionality for:
- Cohort analysis
- Text analysis
- Correlation analysis
- RFM analysis
- Segment analysis
"""

from typing import TYPE_CHECKING

from frameon.dataframe.analysis.cohort import CohortAnalyzer
from frameon.dataframe.analysis.correlation import CorrelationAnalyzer
from frameon.dataframe.analysis.rfm import RFMAnalyzer
from frameon.dataframe.analysis.segment_analysis import SegmentAnalyzer
from frameon.dataframe.analysis.text_analysis import TextAnalyzer

if TYPE_CHECKING:
    from frameon.core.base import FrameOn


class FrameOnAnalysis(
    CohortAnalyzer, TextAnalyzer, CorrelationAnalyzer, RFMAnalyzer, SegmentAnalyzer
):
    def __init__(self, parent_df: "FrameOn"):
        CohortAnalyzer.__init__(self, parent_df)
        TextAnalyzer.__init__(self, parent_df)
        CorrelationAnalyzer.__init__(self, parent_df)
        RFMAnalyzer.__init__(self, parent_df)
        SegmentAnalyzer.__init__(self, parent_df)
