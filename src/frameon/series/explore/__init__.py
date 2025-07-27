from typing import TYPE_CHECKING
from frameon.series.explore.info import SeriesOnInfo
from frameon.series.explore.anomalies import SeriesOnAnomaly
if TYPE_CHECKING:
    from frameon.core.base import SeriesOn

class SeriesOnExplore(SeriesOnInfo, SeriesOnAnomaly):

    def __init__(self, parent_series: "SeriesOn"):
        SeriesOnInfo.__init__(self, parent_series)
        SeriesOnAnomaly.__init__(self, parent_series)