import pytest
import pandas as pd
import numpy as np
from frameon.series.preprocessing.preprocessing import SeriesOnPreproc
from frameon.core.base import SeriesOn, FrameOn
from datetime import datetime, timedelta


@pytest.fixture
def numeric_series():
    """Fixture with numeric series"""
    data = np.random.normal(0, 1, 100)
    # Add some missing values
    data[10] = np.nan
    data[20] = np.nan
    return SeriesOn(data)


@pytest.fixture
def time_series():
    """Fixture with time series data"""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    values = np.random.normal(0, 1, 100)
    return SeriesOn(values, index=dates)


@pytest.fixture
def categorical_series():
    """Fixture with categorical series"""
    categories = ["A"] * 40 + ["B"] * 30 + ["C"] * 20 + ["D"] * 10
    return SeriesOn(categories)


@pytest.fixture
def text_series():
    """Fixture with text series"""
    texts = ["hello world", "test text", "another example"] * 10
    return SeriesOn(texts)


@pytest.fixture
def categorical_data():
    """Fixture with categorical data"""
    df = FrameOn(
        {
            "category": ["A"] * 5 + ["B"] * 5,
            "category2": ["C"] * 5 + ["D"] * 5,
            "value": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
            "missing": [1, 2, np.nan, 4, 5, 10, np.nan, 30, 40, 50],
        }
    )
    return SeriesOn(df["missing"], _parent_df=df)


@pytest.fixture
def categorical_data_category_column():
    """Fixture with categorical data"""
    df = FrameOn(
        {
            "category": ["A"] * 5 + ["B"] * 5,
            "category2": ["C"] * 5 + ["D"] * 5,
            "value": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
            "missing": [1, 2, np.nan, 4, 5, 10, np.nan, 30, 40, 50],
        }
    )
    return SeriesOn(df["category"], _parent_df=df)


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


class TestSeriesOnPreproc:
    """Tests for SeriesOnPreproc class"""

    # Categorical Conversion Tests
    def test_to_categorical(self, numeric_series):
        """Test numeric to categorical conversion"""
        preproc = SeriesOnPreproc(numeric_series)
        result = preproc.to_categorical(method="equal_intervals", n_categories=3)
        assert result.nunique() == 3

    # Time Series Tests
    def test_smooth_time_series(self, time_series):
        """Test time series smoothing"""
        preproc = SeriesOnPreproc(time_series)
        result = preproc.smooth_time_series(method="exponential", alpha=0.5)
        assert len(result) == len(time_series)

    # Numeric Transformation Tests
    def test_transform_numeric(self, numeric_series):
        """Test numeric transformations"""
        preproc = SeriesOnPreproc(numeric_series)
        result = preproc.transform_numeric(method="log")
        assert not result.isna().all()

    # String Normalization Tests
    def test_normalize_string_series(self, text_series):
        """Test string normalization"""
        preproc = SeriesOnPreproc(text_series)
        result = preproc.normalize_string_series(case_format="lower")
        assert result.str.islower().all()

    # Missing Value Tests
    def test_impute_missing(self, df_with_time_column):
        """Test missing value imputation"""
        result = df_with_time_column["missing1"].preproc.impute_missing(strategy="mean")
        assert not result.isna().any()

    # Group Analysis Tests
    def test_check_group_counts(self, df_with_time_column):
        """Test group count analysis"""
        result = df_with_time_column["missing1"].preproc.check_group_counts(
            category_columns=["text"], return_report=True
        )
        assert isinstance(result, dict)

    # Input Validation
    def test_input_validation(self, numeric_series):
        """Test input validation"""
        preproc = SeriesOnPreproc(numeric_series)

        # Test invalid method
        with pytest.raises(ValueError):
            preproc.to_categorical(method="invalid_method")

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty series
        preproc = SeriesOnPreproc(SeriesOn([]))
        with pytest.raises(ValueError):
            preproc.to_categorical()

        # Test all missing values
        df = FrameOn({"value": [np.nan] * 10 + [0]})
        result = df["value"].preproc.impute_missing(
            strategy="constant", imputer_params={"fill_value": 0}
        )
        assert (result == 0).all()

    # Parametrized tests for method selection parameters
    @pytest.mark.parametrize(
        "method", ["equal_intervals", "quantiles", "custom_bins", "clustering", "rules"]
    )
    def test_to_categorical_methods(self, numeric_series, method):
        """Test all categorical conversion methods don't raise exceptions"""
        preproc = SeriesOnPreproc(numeric_series)
        if method == "custom_bins":
            result = preproc.to_categorical(method=method, bins=[-2, 0, 2])
        elif method == "rules":
            result = preproc.to_categorical(
                method=method,
                rules={
                    "High": lambda x: x > 1,
                    "Low": lambda x: x < -1,
                    "Mid": "default",
                },
            )
        else:
            result = preproc.to_categorical(method=method)
        assert result is not None
        if method == "quantiles":
            result = preproc.to_categorical(method=method, quantiles=None)
            assert result is not None

    @pytest.mark.parametrize(
        "method", ["exponential", "moving_avg", "double", "triple", "median"]
    )
    def test_smooth_time_series_methods(self, time_series, method):
        """Test all smoothing methods don't raise exceptions"""
        preproc = SeriesOnPreproc(time_series)
        result = preproc.smooth_time_series(method=method)
        assert len(result) == len(time_series)

    @pytest.mark.parametrize(
        "method",
        [
            "log",
            "boxcox",
            "yeojohnson",
            "sqrt",
            "reciprocal",
            "zscore",
            "robust",
            "quantile",
        ],
    )
    def test_transform_numeric_methods(self, numeric_series, method):
        """Test all numeric transform methods don't raise exceptions"""
        if method in ["boxcox", "yeojohnson"]:
            numeric_series = numeric_series.dropna()
        preproc = SeriesOnPreproc(numeric_series)
        result = preproc.transform_numeric(method=method)
        assert not result.isna().all()

    @pytest.mark.parametrize(
        "case_format", ["title", "lower", "upper", "sentence", "none"]
    )
    def test_normalize_string_case_formats(self, text_series, case_format):
        """Test all case formats don't raise exceptions"""
        preproc = SeriesOnPreproc(text_series)
        result = preproc.normalize_string_series(case_format=case_format)
        assert len(result) == len(text_series)

    def test_to_categorical_with_empty_rules(self, numeric_series):
        """Test rules method with empty rules"""
        preproc = SeriesOnPreproc(numeric_series)
        with pytest.raises(ValueError):
            preproc.to_categorical(method="rules", rules={})

    def test_smooth_time_series_with_short_series(self):
        """Test smoothing with very short series"""
        short_series = SeriesOn([1, 2, 3])
        preproc = SeriesOnPreproc(short_series)
        result = preproc.smooth_time_series(method="exponential")
        assert len(result) == 3

    def test_fill_missing_by_category_with_missing_categories(
        self, df_with_time_column
    ):
        """Test fill_missing_by_category with missing categories"""
        with pytest.raises(ValueError, match="Missing values in categorical columns"):
            df_with_time_column["missing1"].preproc.fill_missing_by_category(
                category_columns=["missing2"]  # This column has missing values
            )

    def test_impute_missing_with_invalid_aux_cols(self, df_with_time_column):
        """Test impute_missing with invalid auxiliary columns"""
        with pytest.raises(ValueError):
            df_with_time_column["missing1"].preproc.impute_missing(
                auxiliary_cols=["nonexistent_column"]
            )

    def test_to_categorical_quantiles(self, numeric_series):
        """Test quantiles method"""
        preproc = SeriesOnPreproc(numeric_series)
        result = preproc.to_categorical(
            method="quantiles", quantiles=[0, 0.5, 1], labels=["low", "high"]
        )
        assert result.nunique() == 2
        assert set(result.unique()) == {np.nan, "low", "high"}

    def test_to_categorical_quantiles_none(self, numeric_series):
        """Test quantiles method with quantiles=None"""
        preproc = SeriesOnPreproc(numeric_series)
        result = preproc.to_categorical(
            method="quantiles", quantiles=None, n_categories=3
        )
        assert result.nunique() == 3

    def test_to_categorical_clustering(self, numeric_series):
        """Test clustering method"""
        preproc = SeriesOnPreproc(numeric_series)
        result = preproc.to_categorical(
            method="clustering", n_categories=2, as_category=True
        )
        assert result.dtype.name == "category"
        assert len(result.cat.categories) == 2

    def test_to_categorical_as_category(self, numeric_series):
        """Test as_category parameter"""
        preproc = SeriesOnPreproc(numeric_series)
        result = preproc.to_categorical(method="equal_intervals", as_category=True)
        assert result.dtype.name == "category"

    # smooth_time_series tests
    def test_smooth_time_series_inplace(self, time_series):
        """Test inplace smoothing"""
        preproc = SeriesOnPreproc(time_series)
        result = preproc.smooth_time_series(inplace=True)
        assert result is None

    def test_smooth_time_series_seasonality(self, time_series):
        """Test seasonality detection"""
        preproc = SeriesOnPreproc(time_series)
        result = preproc.smooth_time_series(
            adjust_for_seasonality=True, seasonality_period=None
        )
        assert isinstance(result, pd.Series)

    # transform_numeric tests
    def test_transform_numeric_inplace(self, numeric_series):
        """Test inplace transformation"""
        for_preproc = numeric_series.copy()
        preproc = SeriesOnPreproc(for_preproc)
        result = preproc.transform_numeric(inplace=True)
        assert result is None

    # normalize_string_series tests
    def test_normalize_string_series_inplace(self, categorical_data):
        """Test inplace string normalization"""
        s = SeriesOn(pd.Series(["Test  ", "  TEST", "test"]))
        preproc = SeriesOnPreproc(s)
        result = preproc.normalize_string_series(inplace=True)
        assert result is None

    # fill_missing_by_category tests
    @pytest.mark.parametrize("strategy", ["simple", "hierarchical"])
    def test_fill_missing_by_category(self, categorical_data, strategy):
        """Test different filling strategies"""
        preproc = SeriesOnPreproc(categorical_data)
        result = preproc.fill_missing_by_category(
            category_columns="category", strategy=strategy
        )
        assert not result.isna().any()

    # calc_target_category_share tests
    def test_calc_target_category_share(self, categorical_data_category_column):
        """Test category share calculation"""
        preproc = SeriesOnPreproc(categorical_data_category_column)
        result = preproc.calc_target_category_share(
            target_category="A", group_columns=["category2"]
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    # Edge cases
    def test_empty_series(self):
        """Test with empty series"""
        preproc = SeriesOnPreproc(SeriesOn([]))
        with pytest.raises(ValueError):
            preproc.to_categorical()

    def test_single_value_series(self):
        """Test with single value"""
        preproc = SeriesOnPreproc(SeriesOn([1]))
        result = preproc.transform_numeric()
        assert isinstance(result, pd.Series)
