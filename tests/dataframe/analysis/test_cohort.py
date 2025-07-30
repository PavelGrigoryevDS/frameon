import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from frameon.dataframe.analysis.cohort import CohortAnalyzer, CohortConfig
from frameon.core.base import FrameOn

@pytest.fixture
def sample_transactions():
    """Fixture with sample transaction data for testing"""
    dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
    users = [f'user_{i}' for i in range(1, 101)]
    return FrameOn({
        'user_id': np.random.choice(users, size=500),
        'date': np.random.choice(dates, size=500),
        'revenue': np.random.uniform(10, 100, size=500).round(2),
        'order_id': [f'order_{i}' for i in range(500)]
    })

@pytest.fixture
def sample_marketing_costs():
    """Fixture with sample marketing costs data"""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-03-31', freq='ME'),
        'cost': [1000, 1500, 2000]
    })

class TestCohortConfig:
    """Tests for CohortConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CohortConfig(user_id_col='user_id', date_col='date')
        assert config.user_id_col == 'user_id'
        assert config.date_col == 'date'
        assert config.mode == 'retention'
        assert config.granularity == 'month'

class TestCohortAnalyzer:
    """Tests for CohortAnalyzer class"""
    
    def test_basic_retention(self, sample_transactions):
        """Test basic retention analysis"""
        analyzer = CohortAnalyzer(sample_transactions)
        result = analyzer.cohort(
            user_id_col='user_id',
            date_col='date',
            mode='retention'
        )
        assert result is not None
        
    def test_revenue_metrics(self, sample_transactions):
        """Test revenue-related metrics"""
        analyzer = CohortAnalyzer(sample_transactions)
        for metric in ['revenue', 'arpu', 'aov']:
            result = analyzer.cohort(
                user_id_col='user_id',
                order_id_col='order_id',
                date_col='date',
                revenue_col='revenue',
                mode=metric
            )
            assert result is not None
            
    def test_marketing_metrics(self, sample_transactions, sample_marketing_costs):
        """Test marketing-related metrics"""
        analyzer = CohortAnalyzer(sample_transactions)
        for metric in ['romi', 'ltv_cac_ratio']:
            result = analyzer.cohort(
                user_id_col='user_id',
                date_col='date',
                revenue_col='revenue',
                marketing_costs_df=sample_marketing_costs,
                marketing_costs_date_col='date',
                marketing_costs_value_col='cost',
                mode=metric
            )
            assert result is not None
            
    def test_input_validation(self, sample_transactions):
        """Test parameter validation"""
        analyzer = CohortAnalyzer(sample_transactions)
        
        # Test invalid margin
        with pytest.raises(ValueError):
            analyzer.cohort(
                user_id_col='user_id',
                date_col='date',
                margin=1.5  # Should be between 0 and 1
            )
            
        # Test missing required columns
        with pytest.raises(ValueError):
            analyzer.cohort(
                user_id_col='invalid_column',
                date_col='date'
            )

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test empty DataFrame
        analyzer = CohortAnalyzer(FrameOn())
        with pytest.raises(ValueError):
            analyzer.cohort(user_id_col='user_id', date_col='date')
            
        # Test missing revenue column when required
        df = FrameOn({
            'user_id': ['user1'],
            'date': [datetime(2023,1,1)]
        })
        analyzer = CohortAnalyzer(df)
        with pytest.raises(ValueError):
            analyzer.cohort(
                user_id_col='user_id',
                date_col='date',
                mode='revenue'
            )

    def test_different_granularities(self, sample_transactions):
        """Test different time granularities"""
        analyzer = CohortAnalyzer(sample_transactions)
        for granularity in ['day', 'week', 'month', 'quarter']:
            result = analyzer.cohort(
                user_id_col='user_id',
                date_col='date',
                granularity=granularity
            )
            assert result is not None
            
    @pytest.mark.parametrize("mode", [
        'retention', 'users', 'buyers', 'orders', 'sales',
        'revenue', 'revenue_cumsum', 'arpu', 'arppu',
        'apc', 'aov', 'ltv', 'romi', 'ltv_cac_ratio', 'churn_rate'
    ])
    def test_all_modes(self, sample_transactions, sample_marketing_costs, mode):
        """Test all available analysis modes"""
        analyzer = CohortAnalyzer(sample_transactions)
        kwargs = {
            'user_id_col': 'user_id',
            'revenue_col': 'revenue',
            'date_col': 'date',
            'mode': mode
        }
        
        # Add required columns for specific modes
        if mode in ['revenue', 'arpu', 'arppu', 'ltv']:
            kwargs['revenue_col'] = 'revenue'
        if mode in ['orders', 'sales', 'apc', 'aov']:
            kwargs['order_id_col'] = 'order_id'
        if mode in ['romi', 'ltv_cac_ratio']:
            kwargs.update({
                'revenue_col': 'revenue',
                'marketing_costs_df': sample_marketing_costs,
                'marketing_costs_date_col': 'date',
                'marketing_costs_value_col': 'cost'
            })
        
        result = analyzer.cohort(**kwargs)
        assert result is not None

    @pytest.mark.parametrize("granularity", ['day', 'week', 'month', 'quarter'])
    def test_granularities(self, sample_transactions, granularity):
        """Test different time granularities"""
        analyzer = CohortAnalyzer(sample_transactions)
        result = analyzer.cohort(
            user_id_col='user_id',
            date_col='date',
            granularity=granularity
        )
        assert result is not None

    @pytest.mark.parametrize("month_lifetime_method", ['calendar', '30days'])
    def test_month_lifetime_methods(self, sample_transactions, month_lifetime_method):
        """Test different month lifetime calculation methods"""
        analyzer = CohortAnalyzer(sample_transactions)
        result = analyzer.cohort(
            user_id_col='user_id',
            date_col='date',
            granularity='month',
            month_lifetime_method=month_lifetime_method
        )
        assert result is not None

    @pytest.mark.parametrize("display_mode", ['matrix', 'summary'])
    def test_display_modes(self, sample_transactions, display_mode):
        """Test different display modes"""
        analyzer = CohortAnalyzer(sample_transactions)
        result = analyzer.cohort(
            user_id_col='user_id',
            date_col='date',
            display_mode=display_mode,
            summary_stat='mean' if display_mode == 'summary' else None
        )
        assert result is not None

    @pytest.mark.parametrize("summary_stat", ['mean', 'weighted_mean', 'median', 'min', 'max'])
    def test_summary_stats(self, sample_transactions, summary_stat):
        """Test different summary statistics"""
        analyzer = CohortAnalyzer(sample_transactions)
        result = analyzer.cohort(
            user_id_col='user_id',
            date_col='date',
            display_mode='summary',
            summary_stat=summary_stat
        )
        assert result is not None

    def test_helper_methods(self, sample_transactions):
        """Test helper methods directly"""
        analyzer = CohortAnalyzer(sample_transactions)
        
        # Test _validate_parameters
        with pytest.raises(ValueError):
            analyzer._validate_parameters({'margin': 1.5})  # Invalid margin
        
        # Test _validate_data
        analyzer.config_cohort.user_id_col = 'user_id'
        analyzer.config_cohort.date_col = 'date'
        analyzer._validate_data()  # Should pass with valid data
        
        # Test _check_duplicate_user_dates
        analyzer._check_duplicate_user_dates()  # Should pass with test data
        
        # Test _check_negative_revenue
        analyzer.config_cohort.revenue_col = 'revenue'
        analyzer._check_negative_revenue()  # Should pass with test data

    def test_edge_cases_extended(self):
        """Test additional edge cases"""
        # Test all NaN values
        df = FrameOn({
            'user_id': [np.nan, np.nan],
            'date': [np.nan, np.nan],
            'revenue': [np.nan, np.nan]
        })
        analyzer = CohortAnalyzer(df)
        with pytest.raises(ValueError):
            analyzer.cohort(user_id_col='user_id', date_col='date')
        
        # Test negative revenue
        df = FrameOn({
            'user_id': ['user1', 'user2'],
            'date': [datetime(2023,1,1), datetime(2023,1,2)],
            'revenue': [-100, 200]
        })
        analyzer = CohortAnalyzer(df)
        analyzer.cohort(
            user_id_col='user_id',
            date_col='date',
            revenue_col='revenue',
            mode='revenue',
            show_warnings=True
        )
        
        # Test single cohort period
        df = FrameOn({
            'user_id': ['user1', 'user2'],
            'date': [datetime(2023,1,1), datetime(2023,1,2)]
        })
        analyzer = CohortAnalyzer(df)
        result = analyzer.cohort(user_id_col='user_id', date_col='date')
        assert result is not None

    def test_visualization_outputs(self, sample_transactions):
        """Verify visualization output formats"""
        analyzer = CohortAnalyzer(sample_transactions)
        
        # Test heatmap output
        heatmap = analyzer.cohort(
            user_id_col='user_id',
            date_col='date',
            display_mode='matrix'
        )
        assert hasattr(heatmap, 'show')  # Verify it's a plotly figure
        
        # Test summary output
        summary = analyzer.cohort(
            user_id_col='user_id',
            date_col='date',
            display_mode='summary',
            summary_stat='mean'
        )
        assert hasattr(summary, 'show')  # Verify it's a plotly figure

    def test_comprehensive_parameters(self, sample_transactions):
        """Test comprehensive parameter combinations"""
        analyzer = CohortAnalyzer(sample_transactions)
        
        # Test with all optional parameters
        result = analyzer.cohort(
            user_id_col='user_id',
            date_col='date',
            revenue_col='revenue',
            order_id_col='order_id',
            mode='arpu',
            display_mode='summary',
            granularity='week',
            min_cohort_size=5,
            max_cohort_size=100,
            margin=0.5,
            include_period0=False,
            month_lifetime_method='calendar',
            cumulative=True,
            summary_stat='weighted_mean',
            text_auto='.2f',
            color_continuous_scale='Blues',
            title='Test Cohort',
            height=800,
            width=1200,
            show_colorbar=False,
            xgap=2,
            ygap=2,
            row_heights=[20, 1],
            vertical_spacing=0.1,
            xaxis_title='Period',
            yaxis_title='Cohort',
            show_warnings=True
        )
        assert result is not None