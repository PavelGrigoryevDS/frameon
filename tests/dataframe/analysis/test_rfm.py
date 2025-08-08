import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from frameon.dataframe.analysis.rfm import RFMAnalyzer, RFMConfig
from frameon.core.base import FrameOn


@pytest.fixture
def sample_transactions():
    """Fixture with sample transaction data for testing"""
    dates = pd.date_range(datetime.now() - timedelta(days=90), datetime.now(), freq="D")
    users = [f"user_{i}" for i in range(1, 101)]
    return FrameOn(
        {
            "user_id": np.random.choice(users, size=500),
            "date": np.random.choice(dates, size=500),
            "revenue": np.random.uniform(10, 100, size=500).round(2),
            "order_id": [f"order_{i}" for i in range(500)],
        }
    )


class TestRFMConfig:
    """Tests for RFMConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = RFMConfig(
            user_id_col="user_id", date_col="date", revenue_col="revenue"
        )
        assert config.user_id_col == "user_id"
        assert config.date_col == "date"
        assert config.revenue_col == "revenue"
        assert config.score_bins == 3
        assert config.color_continuous_scale == "Greens"


class TestRFMAnalyzer:
    """Tests for RFMAnalyzer class"""

    def test_basic_rfm(self, sample_transactions):
        """Test basic RFM analysis"""
        analyzer = RFMAnalyzer(sample_transactions)
        result = analyzer.rfm(
            user_id_col="user_id", date_col="date", revenue_col="revenue"
        )
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_input_validation(self, sample_transactions):
        """Test input validation"""
        analyzer = RFMAnalyzer(sample_transactions)

        # Test missing required columns
        with pytest.raises(ValueError):
            analyzer.rfm(
                user_id_col="invalid_col", date_col="date", revenue_col="revenue"
            )

        # Test invalid score bins
        with pytest.raises(ValueError):
            analyzer.rfm(
                user_id_col="user_id",
                date_col="date",
                revenue_col="revenue",
                score_bins=4,  # Only 3 or 5 allowed
            )

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty DataFrame
        analyzer = RFMAnalyzer(FrameOn())
        with pytest.raises(ValueError):
            analyzer.rfm(user_id_col="user_id", date_col="date", revenue_col="revenue")

        # Test single user
        df = FrameOn(
            {
                "user_id": ["user1", "user1"],
                "date": [datetime.now(), datetime.now() - timedelta(days=1)],
                "revenue": [100, 200],
            }
        )
        analyzer = RFMAnalyzer(df)
        result = analyzer.rfm(
            user_id_col="user_id", date_col="date", revenue_col="revenue"
        )
        assert isinstance(result, dict)

    def test_return_rfm(self, sample_transactions):
        """Test return_rfm parameter"""
        analyzer = RFMAnalyzer(sample_transactions)
        result = analyzer.rfm(
            user_id_col="user_id",
            date_col="date",
            revenue_col="revenue",
            return_rfm=True,
        )
        assert isinstance(result, dict)
        assert isinstance(result["df_rfm"], FrameOn)

    @pytest.mark.parametrize("score_bins", [3, 5])
    def test_different_score_bins(self, sample_transactions, score_bins):
        """Test different score bin configurations"""
        analyzer = RFMAnalyzer(sample_transactions)
        result = analyzer.rfm(
            user_id_col="user_id",
            date_col="date",
            revenue_col="revenue",
            score_bins=score_bins,
        )
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "plots",
        [
            ["hist"],
            ["heat"],
            ["heat_pairs"],
            ["heat_sliced"],
            ["bar_sliced"],
            ["scatter_sliced"],
            ["distr_grid"],
            ["seg_bar"],
            ["seg_tree"],
        ],
    )
    def test_visualization_options(self, sample_transactions, plots):
        """Test different visualization options"""
        analyzer = RFMAnalyzer(sample_transactions)
        result = analyzer.rfm(
            user_id_col="user_id", date_col="date", revenue_col="revenue", plots=plots
        )
        assert isinstance(result, dict)
        assert len(result) == len(plots)

    @pytest.mark.parametrize("distr_plot_type", ["box", "violin"])
    def test_distr_plot_type(self, sample_transactions, distr_plot_type):
        """Test distr_plot_type parameter"""
        analyzer = RFMAnalyzer(sample_transactions)
        result = analyzer.rfm(
            user_id_col="user_id",
            date_col="date",
            revenue_col="revenue",
            distr_plot_type=distr_plot_type,
        )
        assert isinstance(result, dict)

    @pytest.mark.parametrize("pair_for_slice", ["rf", "fm", "rm"])
    def test_pair_for_slice(self, sample_transactions, pair_for_slice):
        """Test pair_for_slice parameter"""
        analyzer = RFMAnalyzer(sample_transactions)
        result = analyzer.rfm(
            user_id_col="user_id",
            date_col="date",
            revenue_col="revenue",
            pair_for_slice=pair_for_slice,
        )
        assert isinstance(result, dict)
