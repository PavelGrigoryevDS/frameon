from typing import TYPE_CHECKING
from frameon.dataframe.visualization.plot import FrameOnPlots
if TYPE_CHECKING:
    from frameon.core.base import FrameOn

class FrameOnViz(FrameOnPlots):

    def __init__(self, parent_df: "FrameOn"):
        super().__init__(parent_df)
        self._plotly_settings = {}
        
    @property
    def plotly_settings(self) -> dict:
        """Get current Plotly visualization settings."""
        return self._plotly_settings

    def update_plotly_settings(self, **kwargs) -> None:
        """
        Update Plotly visualization settings.
        
        Args:
            **kwargs: Plotly settings like labels, category_orders, etc.
        """
        self._plotly_settings.update(kwargs)

    def reset_plotly_settings(self) -> None:
        """Reset all Plotly visualization settings to defaults."""
        self._plotly_settings = {}