import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from frameon.series.explore.anomalies import SeriesOnAnomaly, OutlierMethod
from frameon.core.base import SeriesOn, FrameOn


@pytest.fixture
def normal_series():
    """Fixture with normal series containing some outliers"""
    data = np.random.normal(0, 1, 100)
    # Add some outliers
    data[10] = 10
    data[20] = -10
    data[30] = np.nan  # Missing value
    return SeriesOn(data)


@pytest.fixture
def df_with_time_column():
    """Fixture with time series data"""
    dates = pd.date_range(datetime.now() - timedelta(days=29), datetime.now(), freq="D")
    df = FrameOn(
        {
            "date": dates,
            "value": [100] * 25 + [200] * 5,  # Outliers
            "text": ["A"] * 20 + ["B"] * 5 + ["C"] * 5,
            "missing1": [1, 2, 3] + [np.nan] * 27,
            "missing2": [np.nan] * 27 + [1, 2, 3],
            "zeros1": [0] * 10 + [1] * 20,
            "zeros2": [1] * 20 + [0] * 10,
            "negatives1": [-1] * 5 + [1] * 25,
            "negatives2": [1] * 25 + [-1] * 5,
        }
    )

    # Add two identical lines (duplicates)
    duplicate_row = df.iloc[0].copy()  # We take the first line
    df = pd.concat(
        [df, pd.DataFrame([duplicate_row, duplicate_row])], ignore_index=True
    )

    return df


class TestSeriesOnAnomaly:
    """Tests for SeriesOnAnomaly class"""

    # Core Detection Tests
    def test_detect_anomalies(self, normal_series):
        """Test basic anomaly detection"""
        detector = SeriesOnAnomaly(normal_series)
        anomalies = detector.detect_anomalies()
        assert isinstance(anomalies, pd.Series)
        assert anomalies.sum() > 0  # Should detect our injected anomalies

    def test_detect_outliers(self, normal_series):
        """Test outlier detection"""
        detector = SeriesOnAnomaly(normal_series)
        result = detector.detect_outliers(method="iqr", return_outliers=True)
        assert isinstance(result, pd.DataFrame)

    # Method-Specific Tests
    def test_iqr_method(self, normal_series):
        """Test IQR outlier detection"""
        detector = SeriesOnAnomaly(normal_series)
        result = detector.detect_outliers(
            method=OutlierMethod.IQR, return_outliers=True
        )
        assert not result.empty

    def test_zscore_method(self, normal_series):
        """Test z-score outlier detection"""
        detector = SeriesOnAnomaly(normal_series)
        result = detector.detect_outliers(
            method=OutlierMethod.ZSCORE, return_outliers=True
        )
        assert not result.empty

    # Window Detection Tests
    def test_window_outliers(self, df_with_time_column):
        """Test window-based outlier detection"""
        result = df_with_time_column["missing1"].explore.detect_window_outliers(
            time_column="date", window=7
        )
        assert result is not None

    # Anomaly Type Tests
    def test_missing_values(self, normal_series):
        """Test missing value detection"""
        detector = SeriesOnAnomaly(normal_series)
        missing = detector.detect_anomalies(anomaly_type="missing")
        assert missing.sum() == 1  # Should detect our single NaN

    # Input Validation
    def test_input_validation(self, normal_series):
        """Test input validation"""
        detector = SeriesOnAnomaly(normal_series)

        # Test invalid method
        with pytest.raises(ValueError):
            detector.detect_outliers(method="invalid_method")

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty series
        detector = SeriesOnAnomaly(SeriesOn([]))
        with pytest.raises(ValueError):
            detector.detect_anomalies()

        # Test constant series
        detector = SeriesOnAnomaly(SeriesOn([1] * 100))
        result = detector.detect_outliers(
            method=OutlierMethod.IQR, return_outliers=True
        )
        assert result.empty  # No outliers in constant data

    @pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
    def test_merge_methods(self, how):
        """Test different merge methods"""
        df1 = pd.DataFrame({"A": ["A0", "A1", "A2"], "B": ["B0", "B1", "B2"]})
        df2 = pd.DataFrame({"A": ["A0", "A3", "A4"], "C": ["C0", "C3", "C4"]})
        merged_df = df1.merge(df2, on="A", how=how)
        assert not merged_df.empty

    @pytest.mark.parametrize(
        "method",
        [
            "iqr",
            "zscore",
            "quantile",
            "mad",
            "tukey",
            "isolation_forest",
            "lof",
            "one_class_svm",
        ],
    )
    def test_detect_outliers_methods(self, method, normal_series):
        """Test different outlier detection methods"""
        detector = SeriesOnAnomaly(normal_series)
        result = detector.detect_outliers(method=method, return_outliers=True)
        assert isinstance(result, pd.DataFrame)

    def test_detect_anomalies_missing_values(self, df_with_time_column):
        """Test missing value detection"""
        detector = SeriesOnAnomaly(df_with_time_column["missing1"])
        missing = detector.detect_anomalies(anomaly_type="missing")
        assert missing.sum() == 27  # Should detect 27 NaN values

    def test_detect_anomalies_zero_values(self, df_with_time_column):
        """Test zero value detection"""
        detector = SeriesOnAnomaly(df_with_time_column["zeros1"])
        zeros = detector.detect_anomalies(anomaly_type="zero")
        assert zeros.sum() == 12  # Should detect 12 zeros

    def test_detect_anomalies_negative_values(self, df_with_time_column):
        """Test negative value detection"""
        detector = SeriesOnAnomaly(df_with_time_column["negatives1"])
        negatives = detector.detect_anomalies(anomaly_type="negative")
        assert negatives.sum() == 7  # Should detect 5 negative values

    def test_detect_anomalies_duplicates(self, df_with_time_column):
        """Test duplicate value detection"""
        detector = SeriesOnAnomaly(df_with_time_column["value"])
        duplicates = detector.detect_anomalies(anomaly_type="duplicate")
        assert duplicates.sum() == 30  # Should detect 2 duplicates

    def test_detect_anomalies_outliers(self, df_with_time_column):
        """Test outlier value detection"""
        detector = SeriesOnAnomaly(df_with_time_column["value"])
        outliers = detector.detect_anomalies(anomaly_type="outlier", method="iqr")
        assert outliers.sum() > 0  # Should detect some outliers

    @pytest.mark.parametrize(
        "method",
        [
            "confidence",
            "iqr",
            "zscore",
            "quantile",
            "mad",
            "tukey",
            "isolation_forest",
            "lof",
            "one_class_svm",
        ],
    )
    def test_window_outliers_methods(self, method, df_with_time_column):
        """Test different window-based outlier detection methods"""
        result = df_with_time_column["value"].explore.detect_window_outliers(
            time_column="date", window=7, method=method
        )
        assert result is not None

    def test_detect_anomalies_empty_series(self):
        """Test anomaly detection on an empty series"""
        detector = SeriesOnAnomaly(SeriesOn([]))
        with pytest.raises(ValueError):
            detector.detect_anomalies()

    def test_detect_outliers_constant_series(self):
        """Test outlier detection on a constant series"""
        detector = SeriesOnAnomaly(SeriesOn([1] * 100))
        result = detector.detect_outliers(
            method=OutlierMethod.IQR, return_outliers=True
        )
        assert result.empty  # No outliers in constant data

    # Confidence interval method tests
    def test_confidence_method(self, normal_series):
        """Test confidence interval outlier detection"""
        detector = SeriesOnAnomaly(normal_series)
        result = detector.detect_outliers(
            method=OutlierMethod.CONFIDENCE, threshold=0.05, return_outliers=True
        )
        assert not result.empty

    def test_confidence_method_edge_cases(self):
        """Test confidence interval method edge cases"""
        # Small sample size
        detector = SeriesOnAnomaly(SeriesOn([1, 2, 3, 4, 100]))
        detector.detect_outliers(
            method=OutlierMethod.CONFIDENCE, threshold=0.1, return_outliers=True
        )

    # anomalies_by_categories tests
    def test_anomalies_by_categories(self, df_with_time_column):
        """Test anomalies_by_categories method"""
        series = SeriesOn(df_with_time_column["value"], _parent_df=df_with_time_column)
        detector = SeriesOnAnomaly(series)

        # Test with outlier detection
        detector.anomalies_by_categories(anomaly_type="outlier", method="iqr")

        # Test with missing value detection
        detector.anomalies_by_categories(anomaly_type="missing")

        # Test with zero detection
        detector.anomalies_by_categories(anomaly_type="zero")

    # anomalies_over_time tests
    def test_anomalies_over_time(self, df_with_time_column):
        """Test anomalies_over_time method"""
        series = SeriesOn(
            df_with_time_column["missing1"], _parent_df=df_with_time_column
        )
        detector = SeriesOnAnomaly(series)

        # Test with daily frequency
        result = detector.anomalies_over_time(time_column="date", freq="D")
        assert result is not None

        # Test with weekly frequency
        result = detector.anomalies_over_time(time_column="date", freq="W")
        assert result is not None

    # plot_rolling_anomaly_rate tests
    def test_plot_rolling_anomaly_rate(self, df_with_time_column):
        """Test plot_rolling_anomaly_rate method"""
        series = SeriesOn(
            df_with_time_column["missing1"], _parent_df=df_with_time_column
        )
        detector = SeriesOnAnomaly(series)

        # Test with weekly window
        result = detector.plot_rolling_anomaly_rate(
            time_column="date",
            window=7,
        )
        assert result is not None

        # Test with monthly window
        result = detector.plot_rolling_anomaly_rate(
            time_column="date",
            window=30,
        )
        assert result is not None
