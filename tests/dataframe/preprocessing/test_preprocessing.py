import pytest
import pandas as pd
import numpy as np
from frameon.dataframe.preprocessing.preprocessing import FrameOnPreproc
from frameon.core.base import FrameOn


@pytest.fixture
def sample_data_with_missing():
    """Fixture with sample data containing missing values"""
    dates = pd.date_range("2023-01-01", periods=5)
    return FrameOn(
        {
            "date": dates,
            "value": [1, 2, np.nan, 4, 5],
            "category": ["A", "B", "C", np.nan, "E"],
            "group": ["X", "X", "Y", "Y", "Z"],
        }
    )


@pytest.fixture
def time_series_data():
    """Fixture with time series data with missing dates"""
    dates = pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-05"])
    return FrameOn({"date": dates, "value": [1, 2, 3], "group": ["A", "A", "B"]})


class TestFrameOnPreproc:
    """Tests for FrameOnPreproc class"""

    def test_impute_missing(self, sample_data_with_missing):
        """Test missing value imputation"""
        preproc = FrameOnPreproc(sample_data_with_missing)

        # Test simple imputation
        simple_result = preproc.impute_missing(
            target_cols="value", auxiliary_cols="all", method="simple", strategy="mean"
        )
        assert not simple_result["value"].isna().any()

        # Test knn imputation
        knn_result = preproc.impute_missing(
            target_cols="value",
            auxiliary_cols=["category"],
            method="knn",
            n_neighbors=2,
        )
        assert not knn_result["value"].isna().any()

    def test_restore_full_index(self, time_series_data):
        """Test index restoration for time series"""
        preproc = FrameOnPreproc(time_series_data)
        result = preproc.restore_full_index(
            date_cols="date", group_cols="group", freq="D", fill_value=0
        )
        assert (
            len(result) == 10
        )  # Should have all dates from 1st to 5th for groups A and B

    def test_input_validation(self, sample_data_with_missing):
        """Test input validation"""
        preproc = FrameOnPreproc(sample_data_with_missing)

        # Test invalid method
        with pytest.raises(ValueError):
            preproc.impute_missing(
                target_cols="value", auxiliary_cols="all", method="invalid"
            )

        # Test missing date column
        with pytest.raises(ValueError):
            preproc.restore_full_index(date_cols="invalid", group_cols="group")

    @pytest.mark.parametrize("method", ["simple", "knn", "iterative"])
    def test_all_imputation_methods(self, sample_data_with_missing, method):
        """Test all imputation methods without assertions"""
        preproc = FrameOnPreproc(sample_data_with_missing)
        preproc.impute_missing(target_cols="value", auxiliary_cols="all", method=method)

    @pytest.mark.parametrize(
        "strategy", ["mean", "median", "most_frequent", "constant"]
    )
    def test_all_imputation_strategies(self, sample_data_with_missing, strategy):
        """Test all simple imputer strategies"""
        preproc = FrameOnPreproc(sample_data_with_missing)
        preproc.impute_missing(
            target_cols="value",
            auxiliary_cols="all",
            method="simple",
            strategy=strategy,
        )

    @pytest.mark.parametrize("inplace", [True, False])
    def test_inplace_options(self, sample_data_with_missing, inplace):
        """Test both inplace options"""
        preproc = FrameOnPreproc(sample_data_with_missing)
        preproc.impute_missing(
            target_cols="value", auxiliary_cols="all", method="simple", inplace=inplace
        )

    @pytest.mark.parametrize("freq", ["D", "W", "ME", "YE"])
    def test_time_frequencies(self, time_series_data, freq):
        """Test different time frequencies for index restoration"""
        preproc = FrameOnPreproc(time_series_data)
        preproc.restore_full_index(date_cols="date", group_cols="group", freq=freq)

    def test_find_optimal_k(self, sample_data_with_missing):
        """Test KNN optimization"""
        preproc = FrameOnPreproc(sample_data_with_missing)
        result = preproc.find_optimal_k_for_knn_imputer(
            target_cols="value", auxiliary_cols="all", max_k=5
        )
        assert "optimal_k" in result

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty DataFrame
        preproc = FrameOnPreproc(FrameOn())
        with pytest.raises(ValueError):
            preproc.impute_missing(target_cols="value", auxiliary_cols="all")

        # Test all missing values
        df = FrameOn({"value": [np.nan, np.nan, np.nan], "group": ["A", "B", "C"]})
        preproc = FrameOnPreproc(df)
        with pytest.raises(ValueError):
            preproc.impute_missing(
                target_cols="value",
                auxiliary_cols="group",
                method="simple",
                strategy="mean",
            )

    def test_all_categorical_auxiliary(self):
        """Test with only categorical auxiliary columns"""
        df = FrameOn(
            {
                "target": [1, 2, np.nan, 4],
                "cat1": ["A", "B", "A", "B"],
                "cat2": ["X", "Y", "X", "Y"],
            }
        )
        preproc = FrameOnPreproc(df)
        result = preproc.impute_missing(
            target_cols="target", auxiliary_cols=["cat1", "cat2"], method="knn"
        )
        assert not result["target"].isna().any()

    def test_single_column_imputation(self):
        """Test imputation with single target and auxiliary column"""
        df = FrameOn({"target": [1, 2, np.nan, 4], "aux": [10, 20, 30, 40]})
        preproc = FrameOnPreproc(df)
        result = preproc.impute_missing(
            target_cols="target", auxiliary_cols="aux", method="iterative"
        )
        assert not result["target"].isna().any()

    def test_duplicate_index_restoration(self):
        """Test index restoration with duplicate index values"""
        df = FrameOn(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "group": ["A", "A"],
                "value": [1, 2],
            }
        )
        preproc = FrameOnPreproc(df)
        with pytest.raises(ValueError):
            preproc.restore_full_index(date_cols="date", group_cols="group")

    @pytest.mark.parametrize("metric", ["nan_euclidean", "nan_manhattan"])
    def test_knn_metrics(self, sample_data_with_missing, metric):
        """Test different distance metrics for KNN"""
        preproc = FrameOnPreproc(sample_data_with_missing)
        result = preproc.find_optimal_k_for_knn_imputer(
            target_cols="value", auxiliary_cols="all", metric=metric, max_k=3
        )
        assert isinstance(result, dict)

    @pytest.mark.parametrize("standardize", [True, False])
    def test_standardization(self, sample_data_with_missing, standardize):
        """Test standardization option"""
        preproc = FrameOnPreproc(sample_data_with_missing)
        preproc.impute_missing(
            target_cols="value",
            auxiliary_cols="all",
            method="knn",
            standardize=standardize,
        )

    def test_custom_imputer_params(self, sample_data_with_missing):
        """Test custom parameters for iterative imputer"""
        preproc = FrameOnPreproc(sample_data_with_missing)
        preproc.impute_missing(
            target_cols="value",
            auxiliary_cols="all",
            method="iterative",
            imputer_params={"max_iter": 5},
        )
