import pytest
import pandas as pd
import numpy as np
from frameon.series.explore.info import SeriesOnInfo
from frameon.core.base import SeriesOn


@pytest.fixture
def numeric_series():
    """Fixture with numeric series"""
    return SeriesOn(np.random.normal(0, 1, 100))


@pytest.fixture
def datetime_series():
    """Fixture with datetime series"""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    return SeriesOn(dates)


@pytest.fixture
def categorical_series():
    """Fixture with categorical series"""
    categories = ["A"] * 40 + ["B"] * 30 + ["C"] * 20 + ["D"] * 10
    return SeriesOn(categories)


class TestSeriesOnInfo:
    """Tests for SeriesOnInfo class"""

    # Core Info Tests
    def test_info_numeric(self, numeric_series):
        """Test core info method for numeric series"""
        analyzer = SeriesOnInfo(numeric_series)
        result = analyzer.info()
        assert result is not None

    def test_info_datetime(self, datetime_series):
        """Test core info method for datetime series"""
        analyzer = SeriesOnInfo(datetime_series)
        analyzer.info()

    def test_info_categorical(self, categorical_series):
        """Test core info method for categorical series"""
        analyzer = SeriesOnInfo(categorical_series)
        result = analyzer.info()
        assert result is not None

    # Datetime Analysis Tests
    def test_datetime_stats(self, datetime_series):
        """Test datetime analysis methods"""
        analyzer = SeriesOnInfo(datetime_series)

        basic_stats = analyzer._generate_basic_stats_datetime()
        assert isinstance(basic_stats, pd.DataFrame)

        quality_stats = analyzer._generate_data_quality_stats_datetime()
        assert isinstance(quality_stats, pd.DataFrame)

    # Numeric Analysis Tests
    def test_numeric_stats(self, numeric_series):
        """Test numeric analysis methods"""
        analyzer = SeriesOnInfo(numeric_series)

        summary = analyzer._generate_summary_for_numeric()
        assert isinstance(summary, pd.DataFrame)

        percentiles = analyzer._generate_percentiles_for_numeric()
        assert isinstance(percentiles, pd.DataFrame)

    # Input Validation
    def test_input_validation(self, numeric_series):
        """Test input validation"""
        analyzer = SeriesOnInfo(numeric_series)

        # Test invalid column type
        with pytest.raises(ValueError):
            analyzer.info(column_type="invalid")

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty series
        analyzer = SeriesOnInfo(SeriesOn([]))
        with pytest.raises(ValueError):
            analyzer.info()

        # Test constant numeric series
        analyzer = SeriesOnInfo(SeriesOn([1] * 100))
        result = analyzer.info()
        assert result is not None

    @pytest.mark.parametrize("hist_mode", ["base", "dual_hist_trim", "dual_hist_qq"])
    def test_info_numeric_hist_mode(self, numeric_series, hist_mode):
        """Test info method with different hist_mode values for numeric series"""
        analyzer = SeriesOnInfo(numeric_series)
        result = analyzer.info(hist_mode=hist_mode, upper_quantile=0.95)
        if hist_mode != "dual_hist_qq":
            assert result is not None
