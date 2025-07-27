import pytest
import pandas as pd
import numpy as np
from frameon.dataframe.statistics.stat_tests import StatisticalTests
from frameon.core.base import FrameOn
import statsmodels.api as sm

@pytest.fixture
def sample_data_2_group():
    """Fixture with sample data for statistical tests"""
    return FrameOn({
        'group': ['A']*20 + ['B']*20,
        'value': np.concatenate([
            np.random.normal(0, 1, 20),
            np.random.normal(1, 1, 20),
        ]),
        'binary': np.random.choice([0, 1], size=40),
        'category': np.random.choice(['X', 'Y'], size=40)
    })
    
@pytest.fixture
def sample_data():
    """Fixture with sample data for statistical tests"""
    return FrameOn({
        'group': ['A']*20 + ['B']*20 + ['C']*20,
        'value': np.concatenate([
            np.random.normal(0, 1, 20),
            np.random.normal(1, 1, 20),
            np.random.normal(2, 1, 20)
        ]),
        'binary': np.random.choice([0, 1], size=60),
        'category': np.random.choice(['X', 'Y', 'Z'], size=60)
    })

@pytest.fixture
def regression_data():
    """Fixture with data for regression tests"""
    x = np.random.normal(0, 1, 100)
    return FrameOn({
        'x': x,
        'y': 2*x + np.random.normal(0, 0.5, 100),
        'group': np.random.choice(['A', 'B'], size=100)
    })

class TestStatisticalTests:
    """Tests for StatisticalTests class"""
    
    # Normality Tests
    def test_normality(self, sample_data):
        """Test normality tests"""
        stats = StatisticalTests(sample_data)
        stats.normality(dv='value', between='group')
        
    def test_levene(self, sample_data):
        """Test Levene's test"""
        stats = StatisticalTests(sample_data)
        stats.levene(dv='value', between='group')
        
    # Hypothesis Tests
    def test_ttest_many_group(self, sample_data):
        """Test t-test 2+ group"""
        with pytest.raises(ValueError):
            stats = StatisticalTests(sample_data)
            stats.ttest(dv='value', between='group')
        
    def test_mwu_many_group(self, sample_data):
        """Test Mann-Whitney U test 2+group"""
        with pytest.raises(ValueError):
            stats = StatisticalTests(sample_data)
            stats.mwu(dv='value', between='group')

    def test_ttest(self, sample_data_2_group):
        """Test t-test """
        stats = StatisticalTests(sample_data_2_group)
        stats.ttest(dv='value', between='group')
        
    def test_mwu(self, sample_data_2_group):
        """Test Mann-Whitney U test"""
        stats = StatisticalTests(sample_data_2_group)
        stats.mwu(dv='value', between='group')
        
    def test_anova(self, sample_data):
        """Test ANOVA"""
        stats = StatisticalTests(sample_data)
        stats.anova(dv='value', between='group')
        
    def test_kruskal(self, sample_data):
        """Test Kruskal-Wallis test"""
        stats = StatisticalTests(sample_data)
        stats.kruskal(dv='value', between='group')
        
    def test_chi2_independence(self, sample_data):
        """Test chi-square test of independence"""
        stats = StatisticalTests(sample_data)
        stats.chi2_independence(x='group', y='category')
        
    # Regression Tests
    def test_ols(self, regression_data):
        """Test OLS regression"""
        stats = StatisticalTests(regression_data)
        stats.ols(formula='y ~ x')
        
    def test_rlm(self, regression_data):
        """Test robust linear regression"""
        stats = StatisticalTests(regression_data)
        stats.rlm(formula='y ~ x')
        
    def test_glm(self, regression_data):
        """Test generalized linear regression"""
        stats = StatisticalTests(regression_data)
        stats.glm(formula='y ~ x', family=sm.families.Gaussian())
        
    def test_mixedlm(self, regression_data):
        """Test mixed effects model"""
        stats = StatisticalTests(regression_data)
        stats.mixedlm(formula='y ~ x', groups='group')
        
    # Other Tests
    def test_bootstrap_strings_statistic(self, sample_data_2_group):
        """Test bootstrap method string statistic"""
        stats = StatisticalTests(sample_data_2_group)
        stats.bootstrap(dv='value', between='group', reference_group='A', statistic='mean_diff')

    def test_bootstrap_callable_statistic(self, sample_data_2_group):
        """Test bootstrap method"""
        stats = StatisticalTests(sample_data_2_group)
        def pct_95_diff(x, y, axis=-1):
            return np.percentile(x, 95, axis=axis) - np.percentile(y, 95, axis=axis)
        stats.bootstrap(dv='value', between='group', reference_group='A', statistic=pct_95_diff)
        
    def test_feature_importance(self, regression_data):
        """Test feature importance analysis"""
        stats = StatisticalTests(regression_data)
        stats.feature_importance_analysis(target_column='y', feature_columns=['x', 'group'])
        
    # Input Validation
    def test_input_validation(self, sample_data):
        """Test input validation"""
        stats = StatisticalTests(sample_data)
        
        # Test invalid DV
        with pytest.raises(ValueError):
            stats.ttest(dv='invalid', between='group')
            
        # Test invalid between
        with pytest.raises(ValueError):
            stats.ttest(dv='value', between='invalid')
            
    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty DataFrame
        with pytest.raises(ValueError):
            stats = StatisticalTests(FrameOn())
            stats.ttest(dv='value', between='group')
            
        # Test constant data
        df = FrameOn({
            'value': [1]*20,
            'group': ['A']*10 + ['B']*10
        })
        stats = StatisticalTests(df)
        stats.ttest(dv='value', between='group')
        
    @pytest.mark.parametrize("method", ['shapiro', 'normaltest', 'jarque_bera'])
    def test_normality_methods(self, sample_data, method):
        """Test normality with different methods"""
        stats = StatisticalTests(sample_data)
        stats.normality(dv='value', between='group', method=method)

    @pytest.mark.parametrize("alternative", ['two-sided', '2s', 'greater', 'g', 'less', 'l'])
    def test_ttest_alternatives(self, sample_data_2_group, alternative):
        """Test t-test with different alternative hypotheses"""
        stats = StatisticalTests(sample_data_2_group)
        stats.ttest(dv='value', between='group', alternative=alternative, reference_group='A')

    @pytest.mark.parametrize("alternative", ['two-sided', '2s', 'greater', 'g', 'less', 'l'])
    def test_mwu_alternatives(self, sample_data_2_group, alternative):
        """Test Mann-Whitney with different alternative hypotheses"""
        stats = StatisticalTests(sample_data_2_group)
        stats.mwu(dv='value', between='group', alternative=alternative, reference_group='A')

    @pytest.mark.parametrize("effsize", ['np2', 'n2'])
    def test_anova_effsize(self, sample_data, effsize):
        """Test ANOVA with different effect sizes"""
        stats = StatisticalTests(sample_data)
        stats.anova(dv='value', between='group', effsize=effsize)

    @pytest.mark.parametrize("test", ['pearson', 'cressie-read', 'log-likelihood', 'freeman-tukey', 'mod-log-likelihood', 'neyman'])
    def test_chi2_tests(self, sample_data, test):
        """Test chi-square with different test types"""
        stats = StatisticalTests(sample_data)
        stats.chi2_independence(x='group', y='category', test=test)

    @pytest.mark.parametrize("cov_type", ['nonrobust', 'HC0', 'HC1', 'HC2', 'HC3'])
    def test_ols_cov_types(self, regression_data, cov_type):
        """Test OLS with different covariance types"""
        stats = StatisticalTests(regression_data)
        stats.ols(formula='y ~ x', cov_type=cov_type)

    @pytest.mark.parametrize("method", ['percentile', 'basic', 'BCa'])
    def test_bootstrap_methods(self, sample_data_2_group, method):
        """Test bootstrap with different methods"""
        stats = StatisticalTests(sample_data_2_group)
        stats.bootstrap(dv='value', between='group', reference_group='A', method=method)        