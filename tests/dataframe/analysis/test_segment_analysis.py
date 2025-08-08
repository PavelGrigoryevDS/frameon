import pytest
import pandas as pd
import numpy as np
from frameon.dataframe.analysis.segment_analysis import SegmentAnalyzer
from frameon.core.base import FrameOn


@pytest.fixture
def sample_segment_data():
    """Fixture with sample segment data for testing"""
    return FrameOn(
        {
            "segment": np.random.choice(["A", "B", "C"], size=100),
            "metric1": np.random.normal(0, 1, 100),
            "metric2": np.random.uniform(0, 100, 100),
            "count": np.ones(100),
        }
    )


class TestSegmentAnalyzer:
    """Tests for SegmentAnalyzer class"""

    def test_segment_polar(self, sample_segment_data):
        """Test polar segment visualization"""
        analyzer = SegmentAnalyzer(sample_segment_data)
        result = analyzer.segment_polar(
            metrics=["metric1", "metric2"], dimension="segment", count_column="count"
        )
        assert result is not None

    def test_segment_table(self, sample_segment_data):
        """Test segment table generation"""
        analyzer = SegmentAnalyzer(sample_segment_data)
        analyzer.segment_table(
            metrics=["metric1", "metric2"], dimension="segment", count_column="count"
        )

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty DataFrame
        analyzer = SegmentAnalyzer(FrameOn())
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=["metric"], dimension="segment", count_column="count"
            )

        # Test include/exclude segments mutual exclusivity
        df = FrameOn(
            {"segment": ["A", "B", "C"], "metric1": [1, 2, 3], "count": [1, 1, 1]}
        )
        analyzer = SegmentAnalyzer(df)
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=["metric1"],
                dimension="segment",
                count_column="count",
                include_segments=["A"],
                exclude_segments=["B"],
            )

        # Test single segment
        df = FrameOn({"segment": ["A"], "metric1": [1], "count": [1]})
        analyzer = SegmentAnalyzer(df)
        result = analyzer.segment_polar(
            metrics=["metric1"], dimension="segment", count_column="count"
        )
        assert result is not None

    def test_normalization_options(self, sample_segment_data):
        """Test different normalization options"""
        analyzer = SegmentAnalyzer(sample_segment_data)

        # Test with metric normalization only
        result1 = analyzer.segment_polar(
            metrics=["metric1", "metric2"],
            dimension="segment",
            count_column="count",
            normalize_metric=True,
            normalize_counts=False,
        )
        assert result1 is not None

        # Test with count normalization only
        result2 = analyzer.segment_polar(
            metrics=["metric1", "metric2"],
            dimension="segment",
            count_column="count",
            normalize_metric=False,
            normalize_counts=True,
        )
        assert result2 is not None

        # Test with both normalizations
        result3 = analyzer.segment_polar(
            metrics=["metric1", "metric2"],
            dimension="segment",
            count_column="count",
            normalize_metric=True,
            normalize_counts=True,
        )
        assert result3 is not None

    def test_visualization_outputs(self, sample_segment_data):
        """Verify visualization output formats"""
        analyzer = SegmentAnalyzer(sample_segment_data)

        # Test polar plot output
        polar = analyzer.segment_polar(
            metrics=["metric1", "metric2"], dimension="segment", count_column="count"
        )
        assert hasattr(polar, "show")  # Verify it's a plotly figure

        # Test metric plot output
        metric_plot = analyzer.metric_by_dimensions_plot(
            metric="metric1", dimensions=["segment"]
        )
        assert hasattr(metric_plot, "show")  # Verify it's a plotly figure

    def test_segment_filtering(self, sample_segment_data):
        """Test segment filtering options"""
        analyzer = SegmentAnalyzer(sample_segment_data)

        # Test include_segments
        result1 = analyzer.segment_polar(
            metrics=["metric1", "metric2"],
            dimension="segment",
            count_column="count",
            include_segments=["A", "B"],
        )
        assert result1 is not None

        # Test exclude_segments
        result2 = analyzer.segment_polar(
            metrics=["metric1", "metric2"],
            dimension="segment",
            count_column="count",
            exclude_segments=["C"],
        )
        assert result2 is not None

        # Test max_segments
        result3 = analyzer.segment_polar(
            metrics=["metric1", "metric2"],
            dimension="segment",
            count_column="count",
            max_segments=2,
        )
        assert result3 is not None

    def test_comprehensive_parameters(self, sample_segment_data):
        """Test comprehensive parameter combinations"""
        analyzer = SegmentAnalyzer(sample_segment_data)

        result = analyzer.segment_polar(
            metrics=["metric1", "metric2"],
            dimension="segment",
            count_column="count",
            normalize_metric=True,
            normalize_counts=True,
            text_auto=".2f",
            labels={"segment": "Customer Segment"},
            title="Segment Analysis",
            agg_func="mean",
            exclude_segments=["D"],
            max_segments=5,
            width=1200,
            height=600,
            horizontal_spacing=0.2,
        )
        assert result is not None

    def test_nan_handling(self):
        """Test handling of NaN values"""
        df = FrameOn(
            {
                "segment": ["A", "B", "A", np.nan],
                "metric1": [1, 2, np.nan, 4],
                "count": [1, 1, 1, 1],
            }
        )
        analyzer = SegmentAnalyzer(df)

        # Should handle NaN in dimension
        result1 = analyzer.segment_polar(
            metrics=["metric1"], dimension="segment", count_column="count"
        )
        assert result1 is not None

        # Should handle NaN in metrics
        result2 = analyzer.segment_table(
            metrics=["metric1"], dimension="segment", count_column="count"
        )
        assert result2 is None  # Table returns None

    @pytest.mark.parametrize(
        "agg_func", ["median", "mean", "sum", "min", "max", "p25", "p75"]
    )
    def test_agg_func(self, sample_segment_data, agg_func):
        """Test agg_func parameter"""
        analyzer = SegmentAnalyzer(sample_segment_data)
        analyzer.segment_table(
            metrics=["metric1", "metric2"],
            dimension="segment",
            count_column="count",
            agg_func=agg_func,
        )

    def test_metric_by_dimensions(self, sample_segment_data):
        """Test metric analysis across dimensions"""
        analyzer = SegmentAnalyzer(sample_segment_data)
        # Test table output
        analyzer.metric_by_dimensions_table(metric="metric1", dimensions=["segment"])

        # Test plot output
        result = analyzer.metric_by_dimensions_plot(
            metric="metric1", dimensions=["segment"]
        )
        assert result is not None

        # Test with color parameter
        df = sample_segment_data.copy()
        df["group"] = np.random.choice(["X", "Y"], size=100)
        analyzer = SegmentAnalyzer(df)
        color_result = analyzer.metric_by_dimensions_plot(
            metric="metric1", dimensions=["segment"], color="group"
        )
        assert color_result is not None
        assert hasattr(color_result, "show")

        # Test sorting with color parameter
        sorted_result = analyzer.metric_by_dimensions_plot(
            metric="metric1", dimensions=["segment"], color="group", sort_bars=True
        )
        assert sorted_result is not None

    def test_input_validation(self, sample_segment_data):
        """Test input validation"""
        analyzer = SegmentAnalyzer(sample_segment_data)

        # Test missing required columns
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=["invalid"], dimension="segment", count_column="count"
            )

        # Test empty metrics list
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=[], dimension="segment", count_column="count"
            )

        # Test horizontal_spacing validation
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=["metric1"],
                dimension="segment",
                count_column="count",
                horizontal_spacing=1.1,  # Invalid - must be <= 1
            )

        # Test max_segments validation
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=["metric1"],
                dimension="segment",
                count_column="count",
                max_segments=0,  # Invalid - must be positive
            )

        # Test empty dimension column
        empty_df = FrameOn({"segment": [], "metric1": [], "count": []})
        analyzer = SegmentAnalyzer(empty_df)
        with pytest.raises(ValueError):
            analyzer.segment_polar(
                metrics=["metric1"], dimension="segment", count_column="count"
            )

        # Test percentile format validation
        with pytest.raises(ValueError):
            analyzer.segment_table(
                metrics=["metric1"],
                dimension="segment",
                count_column="count",
                agg_func="pXX",  # Invalid format
            )

    def test_gradient_row_basic(self):
        """Test basic gradient row styling"""
        analyzer = SegmentAnalyzer(FrameOn())

        row = pd.Series(["10%", "20%", "30%"])
        result = analyzer._gradient_row(row)
        assert len(result) == 3
        assert all("background-color" in style for style in result)
        assert all("color" in style for style in result)

    def test_gradient_row_with_nan(self):
        """Test gradient row with NaN values"""
        analyzer = SegmentAnalyzer(FrameOn())

        row = pd.Series(["10%", "NaN", "30%"])
        result = analyzer._gradient_row(row)
        assert len(result) == 3
        assert "background-color: white" in result[1]

    def test_gradient_row_empty(self):
        """Test gradient row with empty/zero values"""
        analyzer = SegmentAnalyzer(FrameOn())

        row = pd.Series(["0%", "0%", "0%"])
        result = analyzer._gradient_row(row)
        assert len(result) == 3
        assert all("background-color" in style for style in result)

    def test_gradient_row_non_percentage(self):
        """Test gradient row with non-percentage numbers"""
        analyzer = SegmentAnalyzer(FrameOn())

        row = pd.Series(["10", "20", "30"])
        result = analyzer._gradient_row(row)
        assert len(result) == 3
        assert all("background-color" in style for style in result)

    def test_gradient_row_single_value(self):
        """Test gradient row with single value"""
        analyzer = SegmentAnalyzer(FrameOn())

        row = pd.Series(["100%"])
        result = analyzer._gradient_row(row)
        assert len(result) == 1
        assert "background-color" in result[0]

    def test_gradient_row_negative_values(self):
        """Test gradient row with negative values"""
        analyzer = SegmentAnalyzer(FrameOn())

        row = pd.Series(["-10%", "0%", "10%"])
        result = analyzer._gradient_row(row)
        assert len(result) == 3
        assert all("background-color" in style for style in result)

    def test_gradient_row_invalid_values(self):
        """Test gradient row with invalid string values"""
        analyzer = SegmentAnalyzer(FrameOn())

        row = pd.Series(["text", "20%", "30%"])
        result = analyzer._gradient_row(row)
        assert len(result) == 3
        assert "background-color: white" in result[0]
