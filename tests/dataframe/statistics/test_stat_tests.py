import pytest
import pandas as pd
import numpy as np
from frameon.dataframe.statistics.stat_tests import StatisticalTests
from frameon.core.base import FrameOn
import statsmodels.api as sm
from unittest.mock import patch
import matplotlib as mpl

mpl.use("agg")


@pytest.fixture
def sample_data_2_group():
    """Fixture with sample data for statistical tests"""
    return FrameOn(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "value": np.concatenate(
                [
                    np.random.normal(0, 1, 20),
                    np.random.normal(1, 1, 20),
                ]
            ),
            "binary": np.random.choice([0, 1], size=40),
            "category": np.random.choice(["X", "Y"], size=40),
        }
    )


@pytest.fixture
def sample_data():
    """Fixture with sample data for statistical tests"""
    return FrameOn(
        {
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "value": np.concatenate(
                [
                    np.random.normal(0, 1, 20),
                    np.random.normal(1, 1, 20),
                    np.random.normal(2, 1, 20),
                ]
            ),
            "binary": np.random.choice([0, 1], size=60),
            "category": np.random.choice(["X", "Y", "Z"], size=60),
        }
    )


@pytest.fixture
def regression_data():
    """Fixture with data for regression tests"""
    x = np.random.normal(0, 1, 100)
    return FrameOn(
        {
            "x": x,
            "y": 2 * x + np.random.normal(0, 0.5, 100),
            "group": np.random.choice(["A", "B"], size=100),
        }
    )


class TestStatisticalTests:
    """Tests for StatisticalTests class"""

    # Normality Tests
    def test_normality(self, sample_data):
        """Test normality tests"""
        stats = StatisticalTests(sample_data)
        stats.normality(dv="value", between="group")

    def test_levene(self, sample_data):
        """Test Levene's test"""
        stats = StatisticalTests(sample_data)
        stats.levene(dv="value", between="group")

    # Hypothesis Tests
    def test_ttest_many_group(self, sample_data):
        """Test t-test 2+ group"""
        with pytest.raises(ValueError):
            stats = StatisticalTests(sample_data)
            stats.ttest(dv="value", between="group")

    def test_mwu_many_group(self, sample_data):
        """Test Mann-Whitney U test 2+group"""
        with pytest.raises(ValueError):
            stats = StatisticalTests(sample_data)
            stats.mwu(dv="value", between="group")

    def test_ttest(self, sample_data_2_group):
        """Test t-test"""
        stats = StatisticalTests(sample_data_2_group)
        stats.ttest(dv="value", between="group")

    def test_mwu(self, sample_data_2_group):
        """Test Mann-Whitney U test"""
        stats = StatisticalTests(sample_data_2_group)
        stats.mwu(dv="value", between="group")

    def test_anova(self, sample_data):
        """Test ANOVA"""
        stats = StatisticalTests(sample_data)
        stats.anova(dv="value", between="group")

    def test_kruskal(self, sample_data):
        """Test Kruskal-Wallis test"""
        stats = StatisticalTests(sample_data)
        stats.kruskal(dv="value", between="group")

    def test_chi2_independence(self, sample_data):
        """Test chi-square test of independence"""
        stats = StatisticalTests(sample_data)
        stats.chi2_independence(x="group", y="category")

    # Regression Tests
    def test_ols(self, regression_data):
        """Test OLS regression"""
        stats = StatisticalTests(regression_data)
        stats.ols(formula="y ~ x")

    def test_rlm(self, regression_data):
        """Test robust linear regression"""
        stats = StatisticalTests(regression_data)
        stats.rlm(formula="y ~ x")

    def test_glm(self, regression_data):
        """Test generalized linear regression"""
        stats = StatisticalTests(regression_data)
        stats.glm(formula="y ~ x", family=sm.families.Gaussian())

    def test_mixedlm(self, regression_data):
        """Test mixed effects model"""
        stats = StatisticalTests(regression_data)
        stats.mixedlm(formula="y ~ x", groups="group")

    # Other Tests
    def test_bootstrap_strings_statistic(self, sample_data_2_group):
        """Test bootstrap method string statistic"""
        stats = StatisticalTests(sample_data_2_group)
        stats.bootstrap(
            dv="value", between="group", reference_group="A", statistic="mean_diff"
        )

    def test_bootstrap_callable_statistic(self, sample_data_2_group):
        """Test bootstrap method"""
        stats = StatisticalTests(sample_data_2_group)

        def pct_95_diff(x, y, axis=-1):
            return np.percentile(x, 95, axis=axis) - np.percentile(y, 95, axis=axis)

        stats.bootstrap(
            dv="value", between="group", reference_group="A", statistic=pct_95_diff
        )

    def test_feature_importance(self, regression_data):
        """Test feature importance analysis"""
        stats = StatisticalTests(regression_data)
        stats.feature_importance_analysis(
            target_column="y", feature_columns=["x", "group"]
        )

    # Input Validation
    def test_input_validation(self, sample_data):
        """Test input validation"""
        stats = StatisticalTests(sample_data)

        # Test invalid DV
        with pytest.raises(ValueError):
            stats.ttest(dv="invalid", between="group")

        # Test invalid between
        with pytest.raises(ValueError):
            stats.ttest(dv="value", between="invalid")

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty DataFrame
        with pytest.raises(ValueError):
            stats = StatisticalTests(FrameOn())
            stats.ttest(dv="value", between="group")

        # Test constant data
        df = FrameOn({"value": [1] * 20, "group": ["A"] * 10 + ["B"] * 10})
        stats = StatisticalTests(df)
        stats.ttest(dv="value", between="group")

    @pytest.mark.parametrize("method", ["shapiro", "normaltest", "jarque_bera"])
    def test_normality_methods(self, sample_data, method):
        """Test normality with different methods"""
        stats = StatisticalTests(sample_data)
        stats.normality(dv="value", between="group", method=method)

    @pytest.mark.parametrize(
        "alternative", ["two-sided", "2s", "greater", "g", "less", "l"]
    )
    def test_ttest_alternatives(self, sample_data_2_group, alternative):
        """Test t-test with different alternative hypotheses"""
        stats = StatisticalTests(sample_data_2_group)
        stats.ttest(
            dv="value", between="group", alternative=alternative, reference_group="A"
        )

    @pytest.mark.parametrize(
        "alternative", ["two-sided", "2s", "greater", "g", "less", "l"]
    )
    def test_mwu_alternatives(self, sample_data_2_group, alternative):
        """Test Mann-Whitney with different alternative hypotheses"""
        stats = StatisticalTests(sample_data_2_group)
        stats.mwu(
            dv="value", between="group", alternative=alternative, reference_group="A"
        )

    @pytest.mark.parametrize("effsize", ["np2", "n2"])
    def test_anova_effsize(self, sample_data, effsize):
        """Test ANOVA with different effect sizes"""
        stats = StatisticalTests(sample_data)
        stats.anova(dv="value", between="group", effsize=effsize)

    @pytest.mark.parametrize(
        "test",
        [
            "pearson",
            "cressie-read",
            "log-likelihood",
            "freeman-tukey",
            "mod-log-likelihood",
            "neyman",
        ],
    )
    def test_chi2_tests(self, sample_data, test):
        """Test chi-square with different test types"""
        stats = StatisticalTests(sample_data)
        stats.chi2_independence(x="group", y="category", test=test)

    @pytest.mark.parametrize("cov_type", ["nonrobust", "HC0", "HC1", "HC2", "HC3"])
    def test_ols_cov_types(self, regression_data, cov_type):
        """Test OLS with different covariance types"""
        stats = StatisticalTests(regression_data)
        stats.ols(formula="y ~ x", cov_type=cov_type)

    @pytest.mark.parametrize("method", ["percentile", "basic", "BCa"])
    def test_bootstrap_methods(self, sample_data_2_group, method):
        """Test bootstrap with different methods"""
        stats = StatisticalTests(sample_data_2_group)
        stats.bootstrap(dv="value", between="group", reference_group="A", method=method)

    def test_normality_qqplot(self, sample_data):
        """Test normality with QQ plot"""
        stats = StatisticalTests(sample_data)
        stats.normality(dv="value", between="group", show_qqplot=True)

    def test_na_handling(self):
        """Test NA handling logic"""
        df = FrameOn(
            {
                "group": ["A", "A", "B", "B", np.nan],
                "value": [1, 2, np.nan, 4, 5],
                "na_col": [np.nan] * 5,
            }
        )
        stats = StatisticalTests(df)
        stats.normality(dv="value", between="group")

    def test_ttest_correction_none(self, sample_data_2_group):
        """Test ttest with correction=None"""
        stats = StatisticalTests(sample_data_2_group)
        stats.ttest(dv="value", between="group", correction=None)

    def test_anova_non_oneway(self, sample_data):
        """Test non-oneway ANOVA"""
        df = sample_data.copy()
        df["group2"] = np.random.choice(["X", "Y"], size=len(df))
        stats = StatisticalTests(df)
        stats.anova(dv="value", between=["group", "group2"])

    def test_bootstrap_parallel(self, sample_data_2_group):
        """Test bootstrap with parallel processing"""
        stats = StatisticalTests(sample_data_2_group)
        stats.bootstrap(dv="value", between="group", reference_group="A", parallel=True)

    def test_bootstrap_plot(self, sample_data_2_group):
        """Test bootstrap plotting"""
        result = StatisticalTests(sample_data_2_group).bootstrap(
            dv="value",
            between="group",
            reference_group="A",
            return_results=True,
            plot=True,
        )
        assert hasattr(result.plot, "show")  # Verify plotly figure

    def test_regression_options(self, regression_data):
        """Test regression with show_plots and return_results"""
        stats = StatisticalTests(regression_data)
        # Test show_plots
        stats.ols(formula="y ~ x", show_plots=True)
        # Test return_results
        result = stats.ols(formula="y ~ x", return_results=True)
        assert hasattr(result, "summary")

    def test_glm_padjust(self, regression_data):
        """Test GLM with p_adjust"""
        stats = StatisticalTests(regression_data)
        stats.glm(formula="y ~ x", family=sm.families.Gaussian(), p_adjust="fdr_bh")

    def test_quantreg(self, regression_data):
        """Test quantile regression"""
        stats = StatisticalTests(regression_data)
        stats.quantreg(formula="y ~ x", q=0.5)

    def test_ordered_model(self):
        """Test ordered model"""
        data = {
            "y": [
                "bad",
                "medium",
                "good",
                "medium",
                "good",
                "bad",
                "medium",
                "good",
                "good",
                "medium",
            ],
            "x1": [0.5, 1.2, 2.3, 1.0, 2.1, 0.8, 1.5, 2.4, 2.2, 1.3],
            "x2": [10, 15, 20, 12, 18, 11, 16, 22, 19, 14],
        }
        df = pd.DataFrame(data)

        df["y"] = pd.Categorical(
            df["y"], categories=["bad", "medium", "good"], ordered=True
        )
        stats = StatisticalTests(df)
        stats.ordered_model(formula="y ~ x1 + x2")

    def test_feature_importance_classification(self, regression_data):
        """Test feature importance for classification"""
        df = regression_data.copy()
        df["y"] = (df["y"] > df["y"].median()).astype(int)
        stats = StatisticalTests(df)
        stats.feature_importance_analysis(
            target_column="y",
            feature_columns=["x", "group"],
            problem_type="classification",
        )

    def test_validation_errors(self):
        """Test validation error cases"""
        stats = StatisticalTests(FrameOn({"a": [1, 2, 3]}))

        # Test _validate_dataframe
        with pytest.raises(ValueError):
            stats._validate_dataframe(None)
        with pytest.raises(ValueError):
            stats._validate_dataframe("not a dataframe")
        with pytest.raises(ValueError):
            stats._validate_dataframe(FrameOn())

    def test_bootstrap_methods_validation(self, sample_data_2_group):
        """Test bootstrap method validation"""
        stats = StatisticalTests(sample_data_2_group)

        # Test invalid method
        with pytest.raises(ValueError):
            stats.bootstrap(dv="value", between="group", method="invalid")

        # Test invalid n_jobs
        with pytest.raises(ValueError):
            stats.bootstrap(dv="value", between="group", n_jobs=-2)

        # Test invalid confidence_level
        with pytest.raises(ValueError):
            stats.bootstrap(dv="value", between="group", confidence_level=1.1)

    def test_bootstrap_edge_cases(self):
        """Test bootstrap edge cases"""
        # Test single observation
        df = FrameOn({"value": [1], "group": ["A"]})
        stats = StatisticalTests(df)
        with pytest.raises(ValueError):
            stats.bootstrap(dv="value", between="group", method="BCa")

        # Test one-sample bootstrap
        df = FrameOn({"value": [1, 2]})
        stats = StatisticalTests(df)
        stats.bootstrap(dv="value", between=None, statistic="mean")

    def test_statistical_test_validations(self, sample_data):
        """Test statistical test validations"""
        stats = StatisticalTests(sample_data)

        # Test invalid test type
        with pytest.raises(ValueError):
            stats.chi2_independence(x="group", y="category", test="invalid")

        # Test invalid alpha
        with pytest.raises(ValueError):
            stats.ttest(dv="value", between="group", alpha=1.1)

        # Test single group
        df = FrameOn({"value": [1] * 10, "group": ["A"] * 10})
        stats = StatisticalTests(df)
        with pytest.raises(ValueError):
            stats.ttest(dv="value", between="group")

    def test_regression_validations(self, regression_data):
        """Test regression validations"""
        stats = StatisticalTests(regression_data)

        # Test invalid formula
        with pytest.raises(ValueError):
            stats.ols(formula=123)

        # Test invalid family
        with pytest.raises(AttributeError):
            stats.glm(formula="y ~ x", family="invalid")

        # Test invalid q
        with pytest.raises(ValueError):
            stats.quantreg(formula="y ~ x", q=1.1)

    def test_feature_importance_validations(self, regression_data):
        """Test feature importance validations"""
        stats = StatisticalTests(regression_data)

        # Test missing columns
        with pytest.raises(ValueError):
            stats.feature_importance_analysis(
                target_column="missing", feature_columns=["x"]
            )

        # Test single feature
        with pytest.raises(ValueError):
            stats.feature_importance_analysis(target_column="y", feature_columns=["x"])

        # Test single class
        df = regression_data.copy()
        df["y"] = 1
        stats = StatisticalTests(df)
        with pytest.raises(ValueError):
            stats.feature_importance_analysis(
                target_column="y",
                feature_columns=["x", "group"],
                problem_type="classification",
            )

    def test_parallel_processing(self, sample_data_2_group):
        """Test parallel processing paths"""
        stats = StatisticalTests(sample_data_2_group)

        # Test parallel bootstrap
        result = stats.bootstrap(
            dv="value",
            between="group",
            reference_group="A",
            parallel=True,
            n_jobs=2,
            return_results=True,
        )
        assert result is not None

    def test_exception_handling(self, sample_data):
        """Test exception handling paths"""
        stats = StatisticalTests(sample_data)

        # Test failed metric calculation
        with patch(
            "sklearn.metrics.accuracy_score", side_effect=Exception("Test error")
        ):
            stats.feature_importance_analysis(
                target_column="binary",
                feature_columns=["value", "group"],
                problem_type="classification",
            )
