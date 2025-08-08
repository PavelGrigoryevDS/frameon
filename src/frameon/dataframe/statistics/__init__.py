"""
DataFrame statistical methods for hypothesis testing.

Provides functionality for performing various statistical tests
and analyses on data.
"""

from typing import TYPE_CHECKING

from frameon.dataframe.statistics.stat_tests import StatisticalTests

if TYPE_CHECKING:
    from frameon.core.base import FrameOn


class FrameOnStats(StatisticalTests):
    def __init__(self, parent_df: "FrameOn"):
        StatisticalTests.__init__(self, parent_df)
