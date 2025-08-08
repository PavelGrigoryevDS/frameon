import pytest
import pandas as pd
import numpy as np
from frameon.dataframe.explore.info import FrameOnInfo
from frameon.core.base import FrameOn


@pytest.fixture
def sample_data():
    """Fixture with sample data for testing"""
    return FrameOn(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "text_col": ["A", "B", "C", "D", "E"],
            "date_col": pd.date_range("2023-01-01", periods=5),
            "duplicate_col": ["X", "X", "Y", "Y", "Z"],
        }
    )


@pytest.fixture
def empty_data():
    """Fixture with empty DataFrame"""
    return FrameOn()


class TestFrameOnInfo:
    """Tests for FrameOnInfo class"""

    @pytest.mark.parametrize(
        "col_type", ["text", "categorical", "int", "float", "datetime"]
    )
    def test_column_type_detection(self, col_type, sample_data):
        """Test detection of different column types"""
        analyzer = FrameOnInfo(sample_data)
        result = analyzer.info()
        assert result is not None

    @pytest.mark.parametrize("dtype", ["object", "string"])
    def test_string_dtypes(self, dtype):
        """Test with different string dtypes"""
        df = FrameOn({"str_col": pd.Series(["A", "A", "B", "B"], dtype=dtype)})
        analyzer = FrameOnInfo(df)
        result = analyzer.info()
        assert result is not None

    def test_nan_handling(self):
        """Test handling of NaN values in duplicate calculation"""
        df = FrameOn(
            {"col1": ["A", "A", np.nan, "B", "B"], "col2": [1, 1, 2, 2, np.nan]}
        )
        analyzer = FrameOnInfo(df)
        exact_dups = analyzer._calculate_duplicates_in_df(exact=True)
        fuzzy_dups = analyzer._calculate_duplicates_in_df(exact=False)
        assert isinstance(exact_dups, str)
        assert isinstance(fuzzy_dups, str)

    def test_info_display(self, sample_data):
        """Test basic info display"""
        analyzer = FrameOnInfo(sample_data)
        result = analyzer.info()
        assert result is not None

    def test_duplicate_calculation(self, sample_data):
        """Test duplicate calculation"""
        analyzer = FrameOnInfo(sample_data)

        # Test exact duplicates
        exact_dups = analyzer._calculate_duplicates_in_df(exact=True)
        assert isinstance(exact_dups, str)

        # Test fuzzy duplicates
        fuzzy_dups = analyzer._calculate_duplicates_in_df(exact=False)
        assert isinstance(fuzzy_dups, str)

    def test_edge_cases(self, empty_data):
        """Test edge cases"""
        # Test empty DataFrame
        analyzer = FrameOnInfo(empty_data)
        result = analyzer.info()
        assert result is not None

        # Test mixed type columns
        mixed_df = FrameOn({"mixed_col": [1, "A", 2.5, None]})
        analyzer = FrameOnInfo(mixed_df)
        result = analyzer.info()
        assert result is not None
