import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from frameon.dataframe.explore.anomalies import FrameOnAnomaly
from frameon.core.base import FrameOn

@pytest.fixture
def sample_data_with_anomalies():
    """Fixture with sample data containing various anomalies"""
    dates = pd.date_range(datetime.now() - timedelta(days=29), datetime.now(), freq='D')
    df = FrameOn({
        'date': dates,
        'value': [100] * 25 + [200] * 5,  # Outliers
        'text': ['A'] * 20 + ['B'] * 5 + ['C'] * 5,
        'missing1': [1, 2, 3] + [np.nan] * 27,
        'missing2': [np.nan] * 27 + [1, 2, 3],
        'zeros1': [0] * 10 + [1] * 20,
        'zeros2': [1] * 20 + [0] * 10,
        'negatives1': [-1] * 5 + [1] * 25,
        'negatives2': [1] * 25 + [-1] * 5,
    })
    
    # Add two identical lines (duplicates)
    duplicate_row = df.iloc[0].copy()  # We take the first line
    df = pd.concat([df, pd.DataFrame([duplicate_row, duplicate_row])], ignore_index=True)
    
    return df

class TestFrameOnAnomaly:
    """Tests for FrameOnAnomaly class"""
    
    def test_anomaly_types(self, sample_data_with_anomalies):
        """Test detection of different anomaly types"""
        analyzer = FrameOnAnomaly(sample_data_with_anomalies)
        
        # Test missing values
        missing_result = analyzer.anomalies_report(anomaly_type='missing')
        assert missing_result is not None
        
        # Test duplicates
        duplicate_result = analyzer.anomalies_report(anomaly_type='duplicate')
        assert duplicate_result is not None
        
        # Test outliers
        outlier_result = analyzer.anomalies_report(anomaly_type='outlier')
        assert outlier_result is not None
        
        # Test zeros
        zero_result = analyzer.anomalies_report(anomaly_type='zero')
        assert zero_result is not None
        
        # Test negatives
        negative_result = analyzer.anomalies_report(anomaly_type='negative')
        assert negative_result is not None
        
    def test_anomaly_correlation(self, sample_data_with_anomalies):
        """Test anomaly correlation analysis"""
        analyzer = FrameOnAnomaly(sample_data_with_anomalies)
        
        # Test missing values correlation
        corr_result = analyzer.anomalies_corr_matrix(anomaly_type='missing')
        assert corr_result is not None
        
    def test_time_series_anomalies(self, sample_data_with_anomalies):
        """Test time series anomaly detection"""
        analyzer = FrameOnAnomaly(sample_data_with_anomalies)
        time_result = analyzer.anomalies_over_time(time_column='date')
        assert time_result is not None
        
    def test_input_validation(self, sample_data_with_anomalies):
        """Test input validation"""
        analyzer = FrameOnAnomaly(sample_data_with_anomalies)
        
        # Test invalid anomaly type
        with pytest.raises(ValueError):
            analyzer.anomalies_report(anomaly_type='invalid')
            
        # Test missing date column
        with pytest.raises(ValueError):
            analyzer.anomalies_over_time(time_column='invalid_column')
            
    @pytest.mark.parametrize("anomaly_type", ['missing', 'duplicate', 'outlier', 'zero', 'negative'])
    def test_all_anomaly_types(self, sample_data_with_anomalies, anomaly_type):
        """Test all anomaly types without assertions"""
        analyzer = FrameOnAnomaly(sample_data_with_anomalies)
        analyzer.anomalies_report(anomaly_type=anomaly_type)

    @pytest.mark.parametrize("method", ['iqr', 'zscore', 'quantile'])
    def test_all_outlier_methods(self, sample_data_with_anomalies, method):
        """Test all outlier detection methods"""
        analyzer = FrameOnAnomaly(sample_data_with_anomalies)
        analyzer.anomalies_report(anomaly_type='outlier', method=method)

    @pytest.mark.parametrize("exact", [True, False])
    def test_duplicate_exact_options(self, sample_data_with_anomalies, exact):
        """Test both exact and fuzzy duplicate detection"""
        analyzer = FrameOnAnomaly(sample_data_with_anomalies)
        analyzer.anomalies_report(anomaly_type='duplicate', exact=exact)

    @pytest.mark.parametrize("show_option", [True, False])
    def test_show_options(self, sample_data_with_anomalies, show_option):
        """Test all show/hide options"""
        analyzer = FrameOnAnomaly(sample_data_with_anomalies)
        analyzer.anomalies_report(
            show_combinations=show_option,
            show_sample=show_option,
            show_by_categories=show_option,
            show_correlation_matrix=show_option
        )

    @pytest.mark.parametrize("freq", ['D', 'W', 'M'])
    def test_time_frequencies(self, sample_data_with_anomalies, freq):
        """Test different time frequencies"""
        analyzer = FrameOnAnomaly(sample_data_with_anomalies)
        analyzer.anomalies_over_time(time_column='date', freq=freq)

    @pytest.mark.parametrize("return_mode", [False, 'combined', 'by_column'])
    def test_return_modes(self, sample_data_with_anomalies, return_mode):
        """Test all return modes for detect_anomalies"""
        analyzer = FrameOnAnomaly(sample_data_with_anomalies)
        analyzer.detect_anomalies(return_mode=return_mode)

    @pytest.mark.parametrize("n", [2, 3])
    def test_combination_sizes(self, sample_data_with_anomalies, n):
        """Test different combination sizes"""
        analyzer = FrameOnAnomaly(sample_data_with_anomalies)
        analyzer.anomalies_combinations(n=n)
        
    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty DataFrame
        analyzer = FrameOnAnomaly(FrameOn())
        with pytest.raises(ValueError):
            analyzer.anomalies_report(anomaly_type='missing')
            
        # Test no anomalies
        df = FrameOn({
            'value': [1, 2, 3, 4, 5],
            'text': ['A', 'B', 'C', 'D', 'E']
        })
        analyzer = FrameOnAnomaly(df)
        result = analyzer.anomalies_report(anomaly_type='missing')
        assert result is None
        
    def test_no_numeric_columns(self):
        """Test with DataFrame containing no numeric columns"""
        df = FrameOn({
            'text': ['A', 'B', 'C'],
            'category': ['X', 'Y', 'Z']
        })
        analyzer = FrameOnAnomaly(df)
        # These should not raise errors
        analyzer.anomalies_report(anomaly_type='outlier')
        analyzer.anomalies_report(anomaly_type='zero')
        analyzer.anomalies_report(anomaly_type='negative')

    def test_single_column(self):
        """Test with single-column DataFrame"""
        df = FrameOn({'value': [1, 2, 3]})
        analyzer = FrameOnAnomaly(df)
        analyzer.anomalies_report()
        analyzer.anomalies_combinations(n=2)  # Should handle gracefully

    def test_all_identical_rows(self):
        """Test with DataFrame where all rows are identical"""
        df = FrameOn({'value': [1, 1, 1]})
        analyzer = FrameOnAnomaly(df)
        analyzer.anomalies_report(anomaly_type='duplicate')        