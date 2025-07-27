import pytest
import plotly.graph_objects as go
from frameon.utils.plotting.custom_figure import CustomFigure

@pytest.fixture
def basic_figure():
    """Fixture with basic figure for testing"""
    fig = CustomFigure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    return fig

class TestCustomFigureUpdate:
    """Tests for CustomFigure update method"""
    
    def test_basic_updates(self, basic_figure):
        """Test basic figure updates"""
        updated = basic_figure.update(
            title_text="Test Title",
            height=500,
            width=700
        )
        assert updated.layout.title.text == "Test Title"
        assert updated.layout.height == 500
        assert updated.layout.width == 700
        
    def test_axis_updates(self, basic_figure):
        """Test axis configuration updates"""
        updated = basic_figure.update(
            xaxis_title_text="X Axis",
            yaxis_title_text="Y Axis",
            xaxis_showgrid=False,
            yaxis_showgrid=True,
            xaxis_range=[0, 5],
            yaxis_range=[0, 10]
        )
        assert updated.layout.xaxis.title.text == "X Axis"
        assert updated.layout.yaxis.title.text == "Y Axis"
        assert updated.layout.xaxis.showgrid is False
        assert updated.layout.yaxis.showgrid is True
        
    def test_legend_updates(self, basic_figure):
        """Test legend configuration updates"""
        updated = basic_figure.update(
            legend_title_text="Legend",
            legend_position="top",
            legend_orientation="h",
            legend_x=0.5,
            legend_y=1.1,
            showlegend=True,
        )
        assert updated.layout.legend.title.text == "Legend"
        assert updated.layout.legend.x == 0.5
        assert updated.layout.legend.y == 1.1
        assert updated.layout.legend.orientation == "h"
        assert updated.layout.showlegend is True
        
    def test_plot_styling(self, basic_figure):
        """Test plot styling updates"""
        updated = basic_figure.update(
            bargap=0.2,
            plot_bgcolor="lightgray"
        )
        assert updated.layout.bargap == 0.2
        assert updated.layout.plot_bgcolor == "lightgray"
        
    def test_text_and_hover(self, basic_figure):
        """Test text and hover template updates"""
        updated = basic_figure.update(
            texttemplate="%{y}",
            hovertemplate="Value: %{y}<extra></extra>",
            hovermode="x unified"
        )
        assert updated.data[0].texttemplate == "%{y}"
        assert updated.data[0].hovertemplate == "Value: %{y}<extra></extra>"
        assert updated.layout.hovermode == "x unified"
        
    def test_edge_cases(self, basic_figure):
        """Test edge cases and invalid inputs"""
        # Empty update should work
        updated = basic_figure.update()
        assert updated is not None
        
        # Invalid legend position should raise error
        with pytest.raises(ValueError):
            basic_figure.update(legend_position="invalid")
            
        # Invalid barmode should raise error
        with pytest.raises(ValueError):
            basic_figure.update(barmode="invalid")