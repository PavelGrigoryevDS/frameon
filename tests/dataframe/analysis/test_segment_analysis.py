import pytest
import pandas as pd
import numpy as np
from frameon.dataframe.analysis.segment_analysis import SegmentAnalyzer
from frameon.core.base import FrameOn

@pytest.fixture
def sample_segment_data():
    """Fixture with sample segment data for testing"""
    return FrameOn({
        'segment': np.random.choice(['A', 'B', 'C'], size=100),
        'metric1': np.random.normal(0, 1, 100),
        'metric2': np.random.uniform(0, 100, 100),
        'count': np.ones(100)
    })

class TestSegmentAnalyzer:
    """Tests for SegmentAnalyzer class"""
    
    def test_segment_polar(self, sample_segment_data):
        """Test polar segment visualization"""
        analyzer = SegmentAnalyzer(sample_segment_data)
        result = analyzer.segment_polar(
            metrics=['metric1', 'metric2'],
            dimension='segment',
            count_column='count'
        )
        assert result is not None
        
    def test_segment_table(self, sample_segment_data):
        """Test segment table generation"""
        analyzer = SegmentAnalyzer(sample_segment_data)
        analyzer.segment_table(
            metrics=['metric1', 'metric2'],
            dimension='segment',
            count_column='count'
        )
        # Just verify it runs without errors
        
    def test_metric_by_dimensions(self, sample_segment_data):
        """Test metric analysis across dimensions"""
        analyzer = SegmentAnalyzer(sample_segment_data)
        # Test table output
        table_result = analyzer.metric_by_dimensions_table(
            metric='metric1',
            dimensions=['segment']
        )
        assert table_result is None
        
        # Test plot output
        plot_result = analyzer.metric_by_dimensions_plot(
            metric='metric1',
            dimensions=['segment']
        )
        assert plot_result is not None
        
    def test_input_validation(self, sample_segment_data):
        """Test input validation"""
        analyzer = SegmentAnalyzer(sample_segment_data)
        
        # Test missing required columns
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=['invalid'],
                dimension='segment',
                count_column='count'
            )
            
        # Test empty metrics list
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=[],
                dimension='segment',
                count_column='count'
            )

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty DataFrame
        analyzer = SegmentAnalyzer(FrameOn())
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=['metric'],
                dimension='segment',
                count_column='count'
            )
            
        # Test single segment
        df = FrameOn({
            'segment': ['A'],
            'metric1': [1],
            'count': [1]
        })
        analyzer = SegmentAnalyzer(df)
        result = analyzer.segment_polar(
            metrics=['metric1'],
            dimension='segment',
            count_column='count'
        )
        assert result is not None
        
    def test_segment_polar(self, sample_segment_data):
        """Test polar segment visualization"""
        analyzer = SegmentAnalyzer(sample_segment_data)
        result = analyzer.segment_polar(
            metrics=['metric1', 'metric2'],
            dimension='segment',
            count_column='count'
        )
        assert result is not None

    @pytest.mark.parametrize("agg_func", ['median', 'mean', 'sum', 'min', 'max', 'p25', 'p75'])
    def test_agg_func(self, sample_segment_data, agg_func):
        """Test agg_func parameter"""
        analyzer = SegmentAnalyzer(sample_segment_data)
        analyzer.segment_table(
            metrics=['metric1', 'metric2'],
            dimension='segment',
            count_column='count',
            agg_func=agg_func
        )

    def test_segment_table(self, sample_segment_data):
        """Test segment table generation"""
        analyzer = SegmentAnalyzer(sample_segment_data)
        analyzer.segment_table(
            metrics=['metric1', 'metric2'],
            dimension='segment',
            count_column='count'
        )
        # Just verify it runs without errors

    def test_metric_by_dimensions(self, sample_segment_data):
        """Test metric analysis across dimensions"""
        analyzer = SegmentAnalyzer(sample_segment_data)
        # Test table output
        analyzer.metric_by_dimensions_table(
            metric='metric1',
            dimensions=['segment']
        )

        # Test plot output
        result = analyzer.metric_by_dimensions_plot(
            metric='metric1',
            dimensions=['segment']
        )
        assert result is not None

    def test_input_validation(self, sample_segment_data):
        """Test input validation"""
        analyzer = SegmentAnalyzer(sample_segment_data)

        # Test missing required columns
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=['invalid'],
                dimension='segment',
                count_column='count'
            )

        # Test empty metrics list
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=[],
                dimension='segment',
                count_column='count'
            )

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty DataFrame
        analyzer = SegmentAnalyzer(FrameOn())
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=['metric'],
                dimension='segment',
                count_column='count'
            )

        # Test single segment
        df = FrameOn({
            'segment': ['A'],
            'metric1': [1],
            'count': [1]
        })
        analyzer = SegmentAnalyzer(df)
        result = analyzer.segment_polar(
            metrics=['metric1'],
            dimension='segment',
            count_column='count'
        )
        assert result is not None        