from typing import TYPE_CHECKING
from frameon.dataframe.statistics.stat_tests import StatisticalTests
if TYPE_CHECKING:
    from frameon.core.base import FrameOn

class FrameOnStats(StatisticalTests):

    def __init__(self, parent_df: "FrameOn"):
        StatisticalTests.__init__(self, parent_df)
        