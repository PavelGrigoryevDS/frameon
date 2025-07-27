import pytest
import pandas as pd
import numpy as np
from frameon.dataframe.visualization import FrameOnViz
from frameon.core.base import FrameOn

@pytest.fixture
def sample_data():
    """Fixture with sample data for visualization tests"""
    return FrameOn({
        'category': ['A']*20 + ['B']*20 + ['C']*20,
        'category2': ['A']*20 + ['B']*20 + ['C']*20,
        'value': np.concatenate([
            np.random.normal(0, 1, 20),
            np.random.normal(1, 1, 20),
            np.random.normal(2, 1, 20)
        ]),
        'value2': np.concatenate([
            np.random.normal(0, 1, 20),
            np.random.normal(1, 1, 20),
            np.random.normal(2, 1, 20)
        ]),
        'date': pd.date_range('2023-01-01', periods=60),
        'text': ['some text'] * 60
    })

class TestFrameOnViz:
    """Tests for FrameOnViz class"""
    
    # Basic Plots
    def test_bar(self, sample_data):
        """Test bar plot"""
        plots = FrameOnViz(sample_data)
        fig = plots.bar(x='category', y='value')
        assert fig is not None
        
    def test_line(self, sample_data):
        """Test line plot"""
        plots = FrameOnViz(sample_data)
        fig = plots.line(x='date', y='value', color='category')
        assert fig is not None
        
    def test_area(self, sample_data):
        """Test area plot"""
        plots = FrameOnViz(sample_data)
        fig = plots.area(x='date', y='value', color='category')
        assert fig is not None        
        
    def test_box(self, sample_data):
        """Test box plot"""
        plots = FrameOnViz(sample_data)
        fig = plots.box(x='category', y='value')
        assert fig is not None
        
    def test_violin(self, sample_data):
        """Test violin plot"""
        plots = FrameOnViz(sample_data)
        fig = plots.violin(x='category', y='value')
        assert fig is not None        
        
    def test_heatmap(self, sample_data):
        """Test heatmap"""
        plots = FrameOnViz(sample_data)
        fig = plots.heatmap(x='category', y='category2', z='value', do_pivot=True, agg_func='mean')
        assert fig is not None

    def test_plot_ci(self, sample_data):
        """Test plot_ci plot"""
        plots = FrameOnViz(sample_data)
        fig = plots.plot_ci(cat_col='category', num_col='value')
        assert fig is not None        
        
    def test_pie_bar(self, sample_data):
        """Test pie_bar plot"""
        plots = FrameOnViz(sample_data)
        fig = plots.pie_bar(x='category', y='value')
        assert fig is not None
        
    # Statistical Plots
    def test_histogram(self, sample_data):
        """Test histogram"""
        plots = FrameOnViz(sample_data)
        fig = plots.histogram(x='value')
        assert fig is not None
        
    def test_qqplot(self, sample_data):
        """Test qqplot"""
        plots = FrameOnViz(sample_data)
        fig = plots.qqplot(x='value', renderer=None)
        assert fig is not None
        
    def test_pairplot(self, sample_data):
        """Test pairplot"""
        plots = FrameOnViz(sample_data)
        fig = plots.pairplot(pairs=['value', 'value2'], renderer=None)
        assert fig is not None
        
    # Specialized Plots
    def test_wordcloud(self, sample_data):
        """Test wordcloud"""
        plots = FrameOnViz(sample_data)
        fig = plots.wordcloud(text_column='text', return_fig=True)
        assert fig is not None

    def test_cat_compare(self, sample_data):
        """Test cat_compare"""
        plots = FrameOnViz(sample_data)
        fig = plots.cat_compare(cat1='category', cat2='category2', return_figs=True)
        assert fig is not None
        
    def test_parallel_categories(self, sample_data):
        """Test parallel categories plot"""
        plots = FrameOnViz(sample_data)
        fig = plots.parallel_categories(dimensions=['category', 'value'])
        assert fig is not None
        
    # Input Validation
    def test_input_validation(self, sample_data):
        """Test input validation"""
        plots = FrameOnViz(sample_data)
        
        # Test invalid x column
        with pytest.raises(ValueError):
            plots.bar(x='invalid', y='value')
            
        # Test invalid color column
        with pytest.raises(ValueError):
            plots.bar(x='category', y='value', color='invalid')
            
    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty DataFrame
        plots = FrameOnViz(FrameOn())
        with pytest.raises(ValueError):
            plots.bar(x='category', y='value')
            
        # Test constant data
        df = FrameOn({
            'value': [1]*20,
            'category': ['A']*10 + ['B']*10
        })
        plots = FrameOnViz(df)
        fig = plots.bar(x='category', y='value')
        assert fig is not None
        
    @pytest.mark.parametrize("agg_func", ['mean', 'median', 'sum', 'count', 'nunique'])
    def test_bar_agg_funcs(self, sample_data, agg_func):
        """Test bar plot with different aggregation functions"""
        plots = FrameOnViz(sample_data)
        plots.bar(x='category', y='value', agg_func=agg_func)

    @pytest.mark.parametrize("orientation", ['v', 'h'])
    def test_bar_orientations(self, sample_data, orientation):
        """Test bar plot with different orientations"""
        plots = FrameOnViz(sample_data)
        if orientation == 'h':
            plots.bar(y='category', x='value', orientation=orientation)
        else:
            plots.bar(x='category', y='value', orientation=orientation)

    @pytest.mark.parametrize("barmode", ['group', 'stack', 'relative', 'overlay'])
    def test_bar_modes(self, sample_data, barmode):
        """Test bar plot with different barmodes"""
        plots = FrameOnViz(sample_data)
        plots.bar(x='category', y='value', color='category2', barmode=barmode)

    @pytest.mark.parametrize("freq", ['D', 'W', 'M', 'Q', 'Y'])
    def test_line_frequencies(self, sample_data, freq):
        """Test line plot with different time frequencies"""
        plots = FrameOnViz(sample_data)
        plots.line(x='date', y='value', color='category', freq=freq)

    @pytest.mark.parametrize("histnorm", [None, 'percent', 'probability', 'density'])
    def test_histogram_norms(self, sample_data, histnorm):
        """Test histogram with different normalization methods"""
        plots = FrameOnViz(sample_data)
        plots.histogram(x='value', histnorm=histnorm)

    @pytest.mark.parametrize("reference_line", ['45', 's', 'r', None])
    def test_qqplot_reference_lines(self, sample_data, reference_line):
        """Test qqplot with different reference lines"""
        plots = FrameOnViz(sample_data)
        plots.qqplot(x='value', reference_line=reference_line, renderer=None)

    @pytest.mark.parametrize("color_mode", ['count', 'kde', 'category'])
    def test_pairplot_color_modes(self, sample_data, color_mode):
        """Test pairplot with different color modes"""
        plots = FrameOnViz(sample_data)
        plots.pairplot(
            pairs=['value', 'value2'], 
            color_mode=color_mode,
            color_column='category' if color_mode == 'category' else None,
            renderer=None
        )

    @pytest.mark.parametrize("period", ['mom', 'wow', 'dod', 'yoy'])
    def test_period_change_periods(self, sample_data, period):
        """Test period_change with different periods"""
        plots = FrameOnViz(sample_data)
        plots.period_change(
            metric_col='value',
            date_col='date',
            period=period
        )

    @pytest.mark.parametrize("display_mode", ['scatter', 'density_contour'])
    def test_pairplot_display_modes(self, sample_data, display_mode):
        """Test pairplot with different display modes"""
        plots = FrameOnViz(sample_data)
        plots.pairplot(
            pairs=['value', 'value2'],
            display_mode=display_mode,
            renderer=None
        )

    @pytest.mark.parametrize("mode", ['base', 'dual_hist_trim', 'dual_box_trim', 'dual_hist_qq'])
    def test_histogram_modes(self, sample_data, mode):
        """Test histogram with different modes"""
        plots = FrameOnViz(sample_data)
        plots.histogram(
            x='value',
            mode=mode,
            lower_quantile=0.1 if mode in ['dual_hist_trim', 'dual_box_trim'] else None,
            upper_quantile=0.9 if mode in ['dual_hist_trim', 'dual_box_trim'] else None
        )

    @pytest.mark.parametrize("trendline", ['ols', 'lowess', None])
    def test_pairplot_trendlines(self, sample_data, trendline):
        """Test pairplot with different trendlines"""
        plots = FrameOnViz(sample_data)
        plots.pairplot(
            pairs=[('value', 'value2')],
            trendline=trendline,
            renderer=None
        )        