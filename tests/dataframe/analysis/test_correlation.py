import pytest
import numpy as np
from frameon.dataframe.analysis.correlation import CorrelationAnalyzer
from frameon.core.base import FrameOn


@pytest.fixture
def sample_numeric_data():
    """Fixture with sample numeric data for testing"""
    return FrameOn(
        {
            "col1": np.random.normal(0, 1, 100),
            "col2": np.random.normal(0, 1, 100),
            "col3": np.random.normal(0, 1, 100),
            "category": np.random.choice(["A", "B"], size=100),
        }
    )


@pytest.fixture
def sample_mixed_data():
    """Fixture with mixed numeric and non-numeric data"""
    return FrameOn(
        {
            "numeric1": [1, 2, 3, 4, 5],
            "numeric2": [5, 4, 3, 2, 1],
            "text": ["a", "b", "c", "d", "e"],
        }
    )


class TestCorrelationAnalyzer:
    """Tests for CorrelationAnalyzer class"""

    def test_basic_correlation(self, sample_numeric_data):
        """Test basic correlation matrix"""
        analyzer = CorrelationAnalyzer(sample_numeric_data)
        result = analyzer.corr_matrix()
        assert result is not None

    def test_input_validation(self, sample_mixed_data):
        """Test input validation"""
        analyzer = CorrelationAnalyzer(sample_mixed_data)

        # Test invalid method
        with pytest.raises(ValueError):
            analyzer.corr_matrix(method="invalid_method")

        # Test with insufficient numeric columns
        with pytest.raises(ValueError):
            analyzer = CorrelationAnalyzer(FrameOn({"text": ["a", "b", "c"]}))
            analyzer.corr_matrix()

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty DataFrame
        analyzer = CorrelationAnalyzer(FrameOn())
        with pytest.raises(ValueError) as excinfo:
            analyzer.corr_matrix()
        assert (
            "Not enough numeric columns for correlation analysis (need at least 2)"
            in str(excinfo.value)
        )

        # Test single numeric column
        analyzer = CorrelationAnalyzer(FrameOn({"col1": [1, 2, 3]}))
        with pytest.raises(ValueError) as excinfo:
            analyzer.corr_matrix()
        assert (
            "Not enough numeric columns for correlation analysis (need at least 2)"
            in str(excinfo.value)
        )

    def test_custom_labels(self, sample_numeric_data):
        """Test custom column labels"""
        analyzer = CorrelationAnalyzer(sample_numeric_data)
        column_labels = {"col1": "Column 1", "col2": "Column 2"}
        result = analyzer.corr_matrix(column_labels=column_labels)
        assert result is not None

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_different_methods(self, sample_numeric_data, method):
        """Test different correlation methods"""
        analyzer = CorrelationAnalyzer(sample_numeric_data)
        result = analyzer.corr_matrix(method=method)
        assert result is not None

    @pytest.mark.parametrize("significance_level", [0.01, 0.05, 0.1])
    def test_significance_level(self, sample_numeric_data, significance_level):
        """Test significance level parameter"""
        analyzer = CorrelationAnalyzer(sample_numeric_data)
        result = analyzer.corr_matrix(significance_level=significance_level)
        assert result is not None
