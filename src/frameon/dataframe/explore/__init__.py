from frameon.dataframe.explore.info import FrameOnInfo
from typing import TYPE_CHECKING
from frameon.dataframe.explore.anomalies import FrameOnAnomaly
if TYPE_CHECKING:
    from frameon.core.base import FrameOn

class FrameOnExplore(FrameOnInfo, FrameOnAnomaly):

    def __init__(self, parent_df: "FrameOn"):
        FrameOnInfo.__init__(self, parent_df)
        FrameOnAnomaly.__init__(self, parent_df)
