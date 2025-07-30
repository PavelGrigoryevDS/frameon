import pandas as pd
import numpy as np
import plotly.express as px
from dataclasses import dataclass
from typing import Union, Dict, Optional, Any, List, Literal, TYPE_CHECKING
from typing import get_type_hints
from plotly.subplots import make_subplots
from dataclasses import fields, field
from frameon.utils.plotting import CustomFigure
if TYPE_CHECKING: # pragma: no cover
    from frameon.core.base import SmartDataFrame

__all__ = ['CohortAnalyzer']

@dataclass
class CohortConfig:
    user_id_col: str
    date_col: str
    revenue_col: Optional[str] = None
    order_id_col: Optional[str] = None
    marketing_costs_df: Optional[pd.DataFrame] = None
    marketing_costs_date_col: Optional[str] = None
    marketing_costs_value_col: Optional[str] = None
    mode: Literal[
        'retention', 'users', 'buyers', 'orders', 'sales',
        'revenue', 'revenue_cumsum', 'arpu', 'arppu',
        'apc', 'aov', 'ltv', 'romi',
        'ltv_cac_ratio', 'churn_rate'
    ] = 'retention'
    display_mode: Literal['matrix', 'summary'] = 'matrix'
    granularity: Literal['day', 'week', 'month', 'quarter'] = 'month'
    min_cohort_size: Optional[int] = None
    max_cohort_size: Optional[int] = None
    margin: Union[int, float] = 1
    include_period0: bool = True
    month_lifetime_method: Literal['calendar', '30days'] = '30days'
    cumulative: bool = False
    summary_stat: Optional[List[Literal['mean', 'weighted_mean', 'median', 'min', 'max']]] = 'median'
    date_format: Optional[str] = None
    text_auto: Union[bool, str] = True
    color_continuous_scale: str = 'Greens'
    title: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    show_colorbar: bool = True
    xgap: int = 1
    ygap: int = 1
    row_heights: List[int] = field(default_factory=lambda: [10, 1])
    vertical_spacing: float = 0.05
    xaxis_title: Optional[str] = None
    yaxis_title: Optional[str] = None
    show_warnings: bool = False

class CohortAnalyzer:
    def __init__(self, df: "SmartDataFrame"):
        """Initialize with the main dataframe."""
        self.df = df
        self.config_cohort = CohortConfig(user_id_col='', date_col='')
        self.cohort_sizes = None
        self.max_lifetime = None
        self.result = None
        self.summary_stat_result = None
        self.freq_map = {'day': 'D', 'week': 'W', 'month': "M", 'quarter': "Q"}

    def cohort(self,
        user_id_col: str,
        date_col: str,
        revenue_col: Optional[str] = None,
        order_id_col: Optional[str] = None,
        marketing_costs_df: Optional[pd.DataFrame] = None,
        marketing_costs_date_col: Optional[str] = None,
        marketing_costs_value_col: Optional[str] = None,
        mode: Literal[
            'retention', 'users', 'buyers', 'orders', 'sales',
            'revenue', 'revenue_cumsum', 'arpu', 'arppu',
            'apc', 'aov', 'ltv', 'romi',
            'ltv_cac_ratio', 'churn_rate'
        ] = 'retention',
        display_mode: Literal['matrix', 'summary'] = 'matrix',
        granularity: Literal['day', 'week', 'month', 'quarter'] = 'month',
        min_cohort_size: Optional[int] = None,
        max_cohort_size: Optional[int] = None,
        margin: Union[int, float] = 1,
        include_period0: bool = True,
        month_lifetime_method: Literal['calendar', '30days'] = '30days',
        cumulative: bool = False,
        summary_stat: Optional[List[Literal['mean', 'weighted_mean', 'median', 'min', 'max']]] = 'median',
        date_format: Optional[str] = None,
        text_auto: Union[bool, str] = True,
        color_continuous_scale: str = 'Greens',
        title: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        show_colorbar: bool = True,
        xgap: int = 1,
        ygap: int = 1,
        row_heights: List[int] = [15, 1],
        vertical_spacing: float = 0.05,
        xaxis_title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        show_warnings: bool = False
        ) -> CustomFigure:
        """
        Enhanced cohort analysis with multiple metrics and professional features.

        Parameters
        ----------
        user_id_col : str
            Column containing user/account identifiers. Interpretation depends on mode:
            
            - For 'users', 'arpu', 'cac' - identifies unique users
            - For 'buyers', 'arppu', 'cpa' - identifies paying accounts
            - For 'orders', 'aov' - identifies order makers

        date_col : str
            Column with event dates (datetime or convertible).
            
            Note: For revenue metrics ('arpu', 'arppu', 'ltv'), ensure the date aligns with the analysis goal:
            
            - If cohorting *all users* (e.g., ARPU), use *first visit date*
            - If cohorting *paying users* (e.g., ARPPU/LTV), use *first purchase date*

        revenue_col : str, optional
            Column with transaction amounts. Required for revenue metrics.
        order_id_col : str, optional
            Column with order/purchase identifiers. Required for order-based metrics.
        marketing_costs_df : ndarray, optional
            Array with marketing costs data. Required for ROMI and LTV/CAC ratio calculations.
            Should contain date and cost columns.
        marketing_costs_date_col : str, optional
            Column name in marketing_costs_df containing dates.
            Must align with main data date ranges.
        marketing_costs_value_col : str, optional
            Column name in marketing_costs_df containing cost values.
            Must be numeric values.
        mode : str, optional
            Analysis mode. Defaults to 'retention'. Available options:
            
            - Basic metrics: 'users', 'buyers', 'orders', 'sales'
            - Revenue metrics: 'revenue', 'revenue_cumsum', 'arpu', 'arppu'
            - Order metrics: 'apc', 'aov'
            - Advanced metrics: 'ltv', 'romi', 'ltv_cac_ratio'

        display_mode : {'matrix', 'summary'}, optional
            Visualization mode to display (Default: 'matrix'):
            
            - 'matrix': Shows the full cohort matrix (heatmap)
            - 'summary': Shows only the summary line plot (requires summary_stat)

        granularity : str, optional
            Time period granularity. Options: 'day', 'week', 'month'. Default 'day'.
        min_cohort_size : int, optional
            Minimum users required in cohort to be included.
        max_cohort_size : int, optional
            Maximum users allowed in cohort.
        margin : float, optional
            Profit margin coefficient for LTV calculation (0-1). Default 1.0.
        include_period0 : bool, optional
            Whether to include period 0 (acquisition period). Default True.
        month_lifetime_method: Literal['calendar', '30days'] = '30days'
            Lifetime calculation method for month granularity:
            
            - '30days' - every 30 days after the first activity
            - 'calendar' - strictly by calendar months

        summary_stat : str, optional
            Summary statistic to calculate across cohorts:
            
                - 'mean' - arithmetic mean
                - 'weighted_mean' - weighted by cohort size
                - 'median' - median value
                - 'min' - minimum value
                - 'max' - maximum value

        cumulative : bool, optional
            Whether to show cumulative values across periods. Default False.
        show_warnings : bool, optional
            Whether to show warnings.
        date_format : str, optional
            Format string for datetime display using strftime syntax.
        text_auto : bool or str, optional
            Controls display of values in heatmap cells. Default False.
        color_continuous_scale : str, optional
            Color scale for heatmap. Default 'Greens'.
        title : str, optional
            Custom title for the plot.
        height : int, optional
            Plot height in pixels.
        width : int, optional
            Plot width in pixels.
        show_colorbar : bool, optional
            Whether to display color scale bar. Default True.
        xgap : int, optional
            Horizontal gap between heatmap cells in pixels.
        ygap : int, optional
            Vertical gap between heatmap cells in pixels.
        row_heights : list, optional
            Relative heights of subplot rows.
        vertical_spacing : float, optional
            Space between subplot rows (0-1 normalized).
        xaxis_title : str, optional
            Custom label for x-axis.
        yaxis_title : str, optional
            Custom label for y-axis.

        Returns
        -------
        CustomFigure
            Interactive Plotly figure object containing:
            
            - Heatmap visualization of cohort analysis results
            - Configurable color scale and text display          
        """
        try:
            # Validate all parameters before processing
            self._validate_parameters(locals())

            # Separate parameters into config
            self._separate_params(locals())

            # Run all data validations
            self._validate_data()

            # Check for missing periods
            self._check_missing_periods()

            # Calculate cohorts and lifetimes
            self._calculate_cohorts()

            # Filter cohorts and prepare base data
            self._prepare()

            # Calculate the requested metric
            self._calculate_metric()

            # Post-processing and formatting
            self._postprocess_results()

            # Calculate weighted mean
            if self.config_cohort.summary_stat and self.result is not None:
                self._calculate_summary_stat()

            # display(self.result)
            self._filter()


            # Create figure based on display mode
            if self.config_cohort.display_mode == 'summary':
                fig = self._plot_summary()
            else:
                fig = self._plot_heatmap()
        except:
            if self.config_cohort.marketing_costs_df is not None and '_cohort_' in self.config_cohort.marketing_costs_df.columns:
                self.config_cohort.marketing_costs_df.drop(columns='_cohort_', inplace=True, errors='ignore')
            self.df.drop(columns=['_first_activity_', '_cohort_', '_lifetime_'], inplace=True, errors='ignore')
            raise
        finally:
            if self.config_cohort.marketing_costs_df is not None and '_cohort_' in self.config_cohort.marketing_costs_df.columns:
                self.config_cohort.marketing_costs_df.drop(columns='_cohort_', inplace=True, errors='ignore')
            self.df.drop(columns=['_first_activity_', '_cohort_', '_lifetime_'], inplace=True, errors='ignore')
        return fig

    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate all input parameters before processing."""
        # Validate margin is between 0 and 1
        if 'margin' in params and not 0 <= params['margin'] <= 1:
            raise ValueError("Margin must be between 0 and 1")

        # Validate min/max cohort sizes
        if 'min_cohort_size' in params and params['min_cohort_size'] is not None:
            if params['min_cohort_size'] < 1:
                raise ValueError("min_cohort_size must be positive")

        if 'max_cohort_size' in params and params['max_cohort_size'] is not None:
            if params['max_cohort_size'] < 1:
                raise ValueError("max_cohort_size must be positive")

        if ('min_cohort_size' in params and 'max_cohort_size' in params and
            params['min_cohort_size'] is not None and params['max_cohort_size'] is not None and
            params['min_cohort_size'] > params['max_cohort_size']):
            raise ValueError("min_cohort_size cannot be greater than max_cohort_size")

    def _validate_data(self) -> None:
        """Run all data validation checks."""
        if self.df.empty:
            raise ValueError("Dataframe is empty.")
        required_cols = self._get_required_columns()
        if self.config_cohort.show_warnings:
            self._check_missing_values(required_cols)
            self._check_duplicate_user_dates()
            self._check_negative_revenue()
        if self.config_cohort.marketing_costs_df is not None and self.config_cohort.mode in ['romi', 'ltv_cac_ratio']:
            self._validate_marketing_data()
            self._validate_marketing_period_alignment()

    def _check_duplicate_user_dates(self) -> None:
        """Check for duplicate user-date pairs which could affect metrics."""
        config = self.config_cohort
        duplicates = self.df.duplicated(subset=[config.user_id_col, config.date_col], keep=False)
        if duplicates.any():
            pct = duplicates.mean() * 100
            print(f"Warning: Found {duplicates.sum()} duplicate user-date pairs ({pct:.2f}% of data)")

    def _check_negative_revenue(self) -> None:
        """Check for negative revenue values if revenue column exists."""
        config = self.config_cohort
        if config.revenue_col and config.revenue_col in self.df.columns:
            negative = self.df[config.revenue_col] < 0
            if negative.any():
                pct = negative.mean() * 100
                print(f"Warning: Found {negative.sum()} negative revenue values ({pct:.2f}% of data)")

    def _validate_marketing_period_alignment(self) -> None:
        """Validate that marketing costs period aligns with cohort periods."""
        config = self.config_cohort
        if config.marketing_costs_df is None:
            return

        # Get min/max dates from both datasets
        min_main_date = self.df[config.date_col].min()
        max_main_date = self.df[config.date_col].max()
        min_marketing_date = config.marketing_costs_df[config.marketing_costs_date_col].min()
        max_marketing_date = config.marketing_costs_df[config.marketing_costs_date_col].max()

        # Check if marketing data covers the same period as main data
        if min_marketing_date > min_main_date:
            print(f"Warning: Marketing data starts {min_marketing_date} while main data starts {min_main_date}")
        if max_marketing_date < max_main_date:
            print(f"Warning: Marketing data ends {max_marketing_date} while main data ends {max_main_date}")

    def _check_missing_periods(self) -> None:
        """Check for missing periods in main and marketing data."""
        config = self.config_cohort

        # Check main dataframe
        self._check_df_missing_periods(self.df, config.date_col, "Main dataframe")

        # Check marketing data if exists
        if config.marketing_costs_df is not None and config.mode in ['romi', 'ltv_cac_ratio']:
            self._check_df_missing_periods(
                config.marketing_costs_df,
                config.marketing_costs_date_col,
                "Marketing costs data"
            )

    def _check_df_missing_periods(self, df: pd.DataFrame, date_col: str, data_name: str) -> None:
        """
        Check for missing periods in a specific dataframe and print count of missing periods.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to check
        date_col : str
            Name of the date column
        data_name : str
            Name of the dataset for warning message
        """
        config = self.config_cohort

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])

        valid_dates = df[date_col].dropna()
        if valid_dates.empty:
            return

        date_range = pd.date_range(valid_dates.min(), valid_dates.max())
        missing_days = date_range.difference(valid_dates)

        # Calculate missing counts based on granularity
        if config.granularity == 'day':
            missing_count = len(missing_days)
            period_name = 'days'
        elif config.granularity == 'week':
            missing_count = len(set(zip(date_range.year, date_range.isocalendar().week)) -
                        set(zip(valid_dates.dt.year, valid_dates.dt.isocalendar().week)))
            period_name = 'weeks'
        elif config.granularity == 'month':
            missing_count = len(set(zip(date_range.year, date_range.month)) -
                        set(zip(valid_dates.dt.year, valid_dates.dt.month)))
            period_name = 'months'
        elif config.granularity == 'quarter':
            missing_count = len(set(zip(date_range.year, date_range.quarter)) -
                            set(zip(valid_dates.dt.year, valid_dates.dt.quarter)))
            period_name = 'quarters'
        if missing_count > 0:
            print(f"Warning: {data_name} has {missing_count} missing {period_name}")

    def _separate_params(self, kwargs: Dict[str, Any]) -> None:
        """Separate and validate parameters for config."""
        config_updates = {}
        config_fields = {f.name for f in fields(self.config_cohort)}
        config_types = get_type_hints(CohortConfig)
        for key, value in kwargs.items():
            if value is None:
                continue
            if not isinstance(key, str):
                raise TypeError(f"Parameter name must be string, got {type(key)}")
            if key in config_fields:
                expected_type = config_types[key]

                # Handle Literal types
                if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Literal:
                    if value not in expected_type.__args__:
                        raise ValueError(
                            f"Invalid value '{value}' for {key}. Must be one of: {expected_type.__args__}"
                        )

                # Handle regular types
                elif isinstance(expected_type, type) and not isinstance(value, expected_type):
                    raise TypeError(
                        f"Invalid type for '{key}'. Expected {expected_type}, got {type(value)}"
                    )

                config_updates[key] = value

        self.config_cohort.__dict__.update(config_updates)

        # Validate required columns are set
        if not self.config_cohort.user_id_col or not self.config_cohort.date_col:
            raise ValueError("user_id_col and date_col are required parameters")

    def _check_missing_values(self, required_cols: list) -> None:
        """Check for missing values in required columns and report them."""
        config = self.config_cohort
        # Check main dataframe
        missing_info = []
        for col in required_cols:
            if col in self.df.columns:
                missing_count = self.df[col].isna().sum()
                if missing_count > 0:
                    pct = missing_count / len(self.df) * 100
                    missing_info.append(
                        f"Main dataframe - {col}: {missing_count} ({pct:.1f}%)"
                    )

        # Check marketing costs dataframe if needed
        if config.marketing_costs_df is not None and config.mode in ['romi', 'ltv_cac_ratio']:
            marketing_cols = [
                config.marketing_costs_date_col,
                config.marketing_costs_value_col
            ]
            for col in marketing_cols:
                if col in config.marketing_costs_df.columns:
                    missing_count = config.marketing_costs_df[col].isna().sum()
                    if missing_count > 0:
                        pct = missing_count / len(config.marketing_costs_df) * 100
                        missing_info.append(
                            f"Marketing costs - {col}: {missing_count} ({pct:.1f}%)"
                        )

        if missing_info:
            print("Warning: Missing values detected:\n" + "\n".join(missing_info))

    def _validate_marketing_data(self) -> None:
        """Validate marketing costs dataframe and columns."""
        if self.config_cohort.marketing_costs_df is not None:
            df = self.config_cohort.marketing_costs_df
            required = [
                self.config_cohort.marketing_costs_date_col,
                self.config_cohort.marketing_costs_value_col
            ]
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in marketing costs data: {missing}")

            if not pd.api.types.is_datetime64_any_dtype(df[self.config_cohort.marketing_costs_date_col]):
                raise ValueError("Marketing costs date column must be datetime type")

            if not pd.api.types.is_numeric_dtype(df[self.config_cohort.marketing_costs_value_col]):
                raise ValueError("Marketing costs value column must be numeric")

            if (df[self.config_cohort.marketing_costs_value_col] < 0).any():
                raise ValueError("Marketing costs cannot be negative")

    def _get_required_columns(self) -> List[str]:
        """Get required columns based on analysis mode."""
        config = self.config_cohort
        mode = config.mode
        requirements = {
            # User metrics
            'users': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col')
            ],
            'buyers': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col')
            ],
            # Retention metrics
            'retention': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col')
            ],
            'churn_rate': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col')
            ],
            # Order metrics
            'orders': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col'),
                (config.order_id_col, 'order_id_col')
            ],
            'sales': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col'),
                (config.order_id_col, 'order_id_col')
            ],
            'apc': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col'),
                (config.order_id_col, 'order_id_col')
            ],
            'aov': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col'),
                (config.order_id_col, 'order_id_col'),
                (config.revenue_col, 'revenue_col')
            ],

            # Revenue metrics
            'revenue': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col'),
                (config.revenue_col, 'revenue_col')
            ],
            'revenue_cumsum': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col'),
                (config.revenue_col, 'revenue_col')
            ],
            'arpu': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col'),
                (config.revenue_col, 'revenue_col')
            ],
            'arppu': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col'),
                (config.revenue_col, 'revenue_col')
            ],
            # Advanced metrics
            'ltv': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col'),
                (config.revenue_col, 'revenue_col')
            ],
            'romi': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col'),
                (config.revenue_col, 'revenue_col'),
            ],
            'ltv_cac_ratio': [
                (config.user_id_col, 'user_id_col'),
                (config.date_col, 'date_col'),
                (config.revenue_col, 'revenue_col'),
            ]
        }.get(mode, [
            (config.user_id_col, 'user_id_col'),
            (config.date_col, 'date_col')
        ])

        # Check for missing parameters
        missing_params = [
            param_name for value, param_name in requirements
            if value is None
        ]

        if missing_params:
            raise ValueError(
                f"For mode '{mode}', these parameters must be specified: {', '.join(missing_params)}"
            )

        # Get only column names that exist in config (not None)
        required_cols = [
            value for value, _ in requirements
            if value is not None and
            not isinstance(value, pd.DataFrame)  # Exclude DataFrame objects
        ]

        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

        # Validate column types
        if not pd.api.types.is_datetime64_any_dtype(self.df[config.date_col]):
            raise ValueError(f"Date column '{config.date_col}' must be datetime type")

        if config.revenue_col and not pd.api.types.is_numeric_dtype(self.df[config.revenue_col]):
            raise ValueError(f"Revenue column '{config.revenue_col}' must be numeric")

        # Additional validation for marketing data if needed
        if mode in ['romi', 'ltv_cac_ratio']:
            if config.marketing_costs_df is None:
                raise ValueError("Marketing costs dataframe is required for this mode")
            if not config.marketing_costs_date_col or not config.marketing_costs_value_col:
                raise ValueError("Marketing costs date and value columns must be specified")
            if config.marketing_costs_date_col not in config.marketing_costs_df.columns:
                raise ValueError(f"Missing required columns in DataFrame: {config.marketing_costs_date_col}")
            if config.marketing_costs_value_col not in config.marketing_costs_df.columns:
                raise ValueError(f"Missing required columns in DataFrame: {config.marketing_costs_value_col}")

        return required_cols

    def _calculate_cohorts(self) -> None:
        """Calculate cohort assignments and lifetimes."""
        config = self.config_cohort
        df = self.df
        freq_map = self.freq_map
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[config.date_col]):
            raise ValueError('Date column in main DataFrame must be datetime type.')

        df['_first_activity_'] = df.groupby(config.user_id_col)[config.date_col].transform('min')

        if config.granularity == 'day':
            time_diff = df[config.date_col] - df['_first_activity_']
            days = time_diff.dt.total_seconds() / (3600 * 24)
            df['_cohort_'] = df['_first_activity_'].dt.to_period('D')
            df['_lifetime_'] = np.floor(days).astype(int)
            max_lifetime = df[config.date_col].dt.to_period('D').max() - df[config.date_col].dt.to_period('D').min()
            self.max_lifetime = max_lifetime.n

        elif config.granularity == 'week':
            time_diff = df[config.date_col] - df['_first_activity_']
            days = time_diff.dt.total_seconds() / (3600 * 24)
            df['_cohort_'] = df['_first_activity_'].dt.to_period('W')
            df['_lifetime_'] = np.floor(days / 7).astype(int)
            max_lifetime = df[config.date_col].dt.to_period('W').max() - df[config.date_col].dt.to_period('W').min()
            self.max_lifetime = max_lifetime.n

        elif config.granularity == 'month':
            first_activity_dt = df['_first_activity_']
            df['_cohort_'] = first_activity_dt.dt.to_period('M')
            activity_dt = df[config.date_col]
            if config.month_lifetime_method == '30days':
                time_diff = activity_dt - first_activity_dt
                df['_lifetime_'] = np.floor(time_diff.dt.total_seconds() / (3600 * 24 * 30)).astype(int)
                start_of_month = df[['_cohort_']].drop_duplicates()
                all_cohorts = pd.period_range(start=df[config.date_col].min(), end=df[config.date_col].max(), freq=freq_map[self.config_cohort.granularity])
                start_of_month = (
                    start_of_month.set_index('_cohort_')
                    .sort_index()
                    .reindex(index=all_cohorts, fill_value=np.nan)
                    .reset_index(names='_cohort_')
                )
                start_of_month['start_of_month_day'] = start_of_month['_cohort_'].dt.to_timestamp()
                start_of_month = start_of_month.set_index('_cohort_').sort_index()
                max_date = df[config.date_col].max()
                self.max_lifetime = np.floor(
                    (max_date - start_of_month['start_of_month_day']).dt.total_seconds() / (3600 * 24 * 30)
                ).astype(int)
            elif config.month_lifetime_method == 'calendar':
                # Calculate month difference (integer part)
                month_diff = (activity_dt.dt.year - first_activity_dt.dt.year) * 12 + \
                            (activity_dt.dt.month - first_activity_dt.dt.month)

                # Check if we've passed the "anniversary" day in current month
                current_month_day = activity_dt.dt.day
                first_activity_day = first_activity_dt.dt.day
                activity_last_day_of_month = activity_dt.dt.days_in_month
                effective_current_month_day = np.minimum(first_activity_day, activity_last_day_of_month)

                # Compare day and time components
                full_month = current_month_day >= effective_current_month_day

                # Adjust lifetime
                df['_lifetime_'] = month_diff - 1 + full_month.astype(int)
                max_lifetime = df[config.date_col].dt.to_period('M').max() - df[config.date_col].dt.to_period('M').min()
                self.max_lifetime = max_lifetime.n
            else:
                raise ValueError(f"Unsupported month_lifetime_method: {config.month_lifetime_method}. Can be one of calendar or 30days")

            df['_lifetime_'] = df['_lifetime_'].clip(lower=0)
        elif config.granularity == 'quarter':
            first_activity_dt = df['_first_activity_']
            df['_cohort_'] = first_activity_dt.dt.to_period('Q')
            activity_dt = df[config.date_col]
            
            # Calculate quarter difference
            quarter_diff = (activity_dt.dt.year - first_activity_dt.dt.year) * 4 + \
                        (activity_dt.dt.quarter - first_activity_dt.dt.quarter)
            
            # Calculate day within quarter for both dates
            current_quarter_day = (activity_dt - pd.to_datetime({
                'year': activity_dt.dt.year,
                'month': (activity_dt.dt.quarter - 1) * 3 + 1,
                'day': 1
            })).dt.days
            
            first_activity_quarter_day = (first_activity_dt - pd.to_datetime({
                'year': first_activity_dt.dt.year,
                'month': (first_activity_dt.dt.quarter - 1) * 3 + 1,
                'day': 1
            })).dt.days
            
            full_quarter = current_quarter_day >= first_activity_quarter_day
            df['_lifetime_'] = quarter_diff - 1 + full_quarter.astype(int)
            max_lifetime = df[config.date_col].dt.to_period('Q').max() - df[config.date_col].dt.to_period('Q').min()
            self.max_lifetime = max_lifetime.n
            df['_lifetime_'] = df['_lifetime_'].clip(lower=0)
        else:
            raise ValueError(f"Unsupported granularity: {config.granularity}")

        # Update marketing costs period if needed
        if config.marketing_costs_df is not None and config.marketing_costs_date_col:
            self._calculate_marketing_period()

    def _calculate_marketing_period(self) -> None:
        """Calculate period for grouping marketing costs using granularity."""
        config = self.config_cohort
        df = config.marketing_costs_df
        if df is None:
            raise ValueError('Marketing data is None.')

        if not pd.api.types.is_datetime64_any_dtype(df[config.marketing_costs_date_col]):
            raise ValueError('Date column in marketing costs data must be datetime type.')

        if config.granularity == 'day':
            df['_cohort_'] = df[config.marketing_costs_date_col].dt.to_period('D')
        elif config.granularity == 'week':
            df['_cohort_'] = df[config.marketing_costs_date_col].dt.to_period('W')
        elif config.granularity == 'month':
            df['_cohort_'] = df[config.marketing_costs_date_col].dt.to_period('M')
        elif config.granularity == 'quarter':
            df['_cohort_'] = df[config.marketing_costs_date_col].dt.to_period('Q')
                
    def _prepare(self) -> None:
        """Filter cohorts by size and prepare base data."""
        config = self.config_cohort
        # Get cohort sizes from period 0
        self.cohort_sizes = (
            self.df[self.df['_lifetime_'] == 0]
            .groupby('_cohort_', observed=False)[config.user_id_col]
            .nunique()
        )
        # Handle division by zero
        self.cohort_sizes = self.cohort_sizes.replace(0, np.nan)

    def _calculate_metric(self) -> None:
        """Calculate the requested cohort metric."""
        config = self.config_cohort

        # Split into smaller focused methods for each metric type
        if config.mode in ['retention', 'users', 'buyers', 'churn_rate']:
            self._calculate_user_metrics()
        elif config.mode in ['orders', 'sales', 'apc', 'aov']:
            self._calculate_order_metrics()
        elif config.mode in ['revenue', 'revenue_cumsum', 'arpu', 'arppu', 'ltv']:
            self._calculate_revenue_metrics()
        elif config.mode in ['romi', 'ltv_cac_ratio']:
            self._calculate_marketing_metrics()
        else:
            raise ValueError(f"Unknown mode: {config.mode}")

    def _calculate_user_metrics(self) -> None:
        """Calculate user-based metrics (retention, users, buyers, churn)."""
        config = self.config_cohort
        df = self.df

        if config.mode == 'retention':
            # Retention rate: percentage of users active in each period
            users = df.groupby(['_cohort_', '_lifetime_'], observed=False)[config.user_id_col].nunique().unstack(fill_value=0)
            result = users.div(self.cohort_sizes, axis=0)

        elif config.mode in ['users', 'buyers']:
            # Number of unique users per cohort per period
            result = df.groupby(['_cohort_', '_lifetime_'], observed=False)[config.user_id_col].nunique().unstack(fill_value=0)

        elif config.mode == 'churn_rate':
            # Churn rate: 1 - retention rate
            retention = (
                df.groupby(['_cohort_', '_lifetime_'], observed=False)[config.user_id_col]
                .nunique()
                .unstack(fill_value=0)
                .div(self.cohort_sizes, axis=0)
            )
            result = 1 - retention

        # Apply cumulative if needed
        if config.cumulative:
            result = result.cumsum(axis=1)

        self.result = result

    def _calculate_order_metrics(self) -> None:
        """Calculate order-based metrics (orders, sales, apc, aov)."""
        config = self.config_cohort
        df = self.df

        if config.mode in ['orders', 'sales']:
            # Number of orders per cohort per period
            result = df.groupby(['_cohort_', '_lifetime_'], observed=False)[config.order_id_col].nunique().unstack(fill_value=0)
        elif config.mode == 'apc':
            # Average payment count per user
            orders = (
                df.groupby(['_cohort_', '_lifetime_', config.user_id_col], observed=False)[config.order_id_col]
                .nunique()
            )
            result = orders.groupby(['_cohort_', '_lifetime_'], observed=False).mean().unstack(fill_value=0)

        elif config.mode == 'aov':
            # Average order value
            revenue = df.groupby(['_cohort_', '_lifetime_'], observed=False)[config.revenue_col].sum()
            orders = df.groupby(['_cohort_', '_lifetime_'], observed=False)[config.order_id_col].nunique()

            # Handle division by zero
            aov = np.divide(
                revenue,
                orders,
                out=np.zeros_like(revenue, dtype=float),
                where=(orders != 0)
            )
            result = aov.unstack(fill_value=0)

        # Apply cumulative if needed
        if config.cumulative:
            result = result.cumsum(axis=1)

        self.result = result

    def _calculate_revenue_metrics(self) -> None:
        """Calculate revenue-based metrics (revenue, arpu, arppu, ltv)."""
        config = self.config_cohort
        df = self.df

        if config.mode == 'revenue':
            # Total revenue per cohort per period
            result = df.groupby(['_cohort_', '_lifetime_'], observed=False)[config.revenue_col].sum().unstack(fill_value=0)

        elif config.mode == 'revenue_cumsum':
            # Cumulative revenue per cohort
            result = (
                df.groupby(['_cohort_', '_lifetime_'], observed=False)[config.revenue_col]
                .sum()
                .unstack(fill_value=0)
                .cumsum(axis=1)
            )

        elif config.mode in ['arpu', 'arppu']:
            # Average revenue per user
            revenue = df.groupby(['_cohort_', '_lifetime_'], observed=False)[config.revenue_col].sum()
            active_users = df.groupby(['_cohort_', '_lifetime_'], observed=False)[config.user_id_col].nunique()
            # Safe division to avoid divide by zero
            result = np.divide(
                revenue,
                active_users,
                out=np.zeros_like(revenue, dtype=float),
                where=(active_users != 0)
            ).unstack(fill_value=0)

        elif config.mode == 'ltv':
            # Lifetime value (cumulative)
            revenue = (
                df.groupby(['_cohort_', '_lifetime_'], observed=False)[config.revenue_col]
                .sum() * config.margin
            )
            result = revenue.unstack(fill_value=0).div(self.cohort_sizes, axis=0)
            result = result.cumsum(axis=1)

        # Apply cumulative if needed (for modes not already cumulative)
        if config.cumulative and config.mode not in ['revenue_cumsum', 'ltv']:
            result = result.cumsum(axis=1)

        self.result = result

    def _calculate_marketing_metrics(self) -> None:
        """Calculate marketing-based metrics (romi, ltv_cac_ratio)."""
        config = self.config_cohort

        if config.marketing_costs_df is None or config.marketing_costs_df.empty:
            raise ValueError("Marketing costs dataframe is empty or None")
        # 1. Calculate cumulative LTV components
        ltv_cumulative = (
            self.df.groupby(['_cohort_', '_lifetime_'], observed=False)[config.revenue_col]
            .sum()
            .groupby(level=0, observed=False).cumsum()
            .div(self.cohort_sizes)
            .mul(config.margin)
            .reset_index(name='ltv')
        )

        # 2. Calculate CAC (cost per cohort)
        cac = (
            config.marketing_costs_df
            .groupby('_cohort_', observed=False)[config.marketing_costs_value_col]
            .sum()
            .div(self.cohort_sizes)
            .reset_index(name='cac')
        )
        # 3. Merge and calculate metrics
        merged = pd.merge(
            ltv_cumulative,
            cac,
            on='_cohort_',
            how='left'
        )

        # 4. Calculate the required metric with safe division
        if config.mode == 'romi':
            merged['metric'] = np.divide(
                merged['ltv'] - merged['cac'],
                merged['cac'],
                out=np.zeros(len(merged)),
                where=merged['cac'].notna() & (merged['cac'] != 0)
            )
        else:  # ltv_cac_ratio
            merged['metric'] = np.divide(
                merged['ltv'],
                merged['cac'],
                out=np.zeros(len(merged)),
                where=merged['cac'].notna() & (merged['cac'] != 0)
            )

        # 5. Pivot to final format
        self.result = (
            merged.pivot(index='_cohort_', columns='_lifetime_', values='metric')
            .rename_axis(columns=None)
            .fillna(0)
        )

    def _postprocess_results(self) -> None:
        """Format and finalize results."""
        config = self.config_cohort
        result = self.result
        df = self.df
        freq_map = self.freq_map
        all_cohorts = pd.period_range(start=df[config.date_col].min(), end=df[config.date_col].max(), freq=freq_map[self.config_cohort.granularity])
        if config.granularity == 'month' and config.month_lifetime_method == '30days':
            # If we have a calculation of 30 days, then each cohort will have its own potentially possible last period
            all_periods = range(self.max_lifetime.max() + 1)
            result = result.reindex(index=all_cohorts, columns=all_periods, fill_value=0)
            self.cohort_sizes = self.cohort_sizes.reindex(index=all_cohorts, fill_value=0)
            mask = result.columns.values > self.max_lifetime.values[:, None]
        else:
            all_periods = range(self.max_lifetime + 1)
            result = result.reindex(index=all_cohorts, columns=all_periods, fill_value=0)
            self.cohort_sizes = self.cohort_sizes.reindex(index=all_cohorts, fill_value=0)
            # Create mask for upper triangle
            mask = np.zeros_like(result, dtype=bool)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if j > (mask.shape[1] - 1 - i):
                        mask[i, j] = True

        # # Apply mask to create NaN values
        result[mask] = np.nan
        # Calculate weighted mean if requested
        self.result = result

    def _filter(self) -> None:
        """Filter cohorts by size and prepare base data."""
        config = self.config_cohort
        result = self.result
        cohort_sizes = self.cohort_sizes
        # Filter by min and max cohort size
        if config.min_cohort_size or config.max_cohort_size:
            size_condition = True
            if config.min_cohort_size:
                size_condition &= (cohort_sizes >= config.min_cohort_size)
            if config.max_cohort_size:
                size_condition &= (cohort_sizes <= config.max_cohort_size)
            excluded_cohorts = result[~size_condition]

            if not excluded_cohorts.empty:
                pct = len(excluded_cohorts) / len(result) * 100
                size_range = []
                if config.min_cohort_size:
                    size_range.append(f"min {config.min_cohort_size}")
                if config.max_cohort_size:
                    size_range.append(f"max {config.max_cohort_size}")
                print(f"Excluded {len(excluded_cohorts)} cohorts ({pct:.1f}%) outside {', '.join(size_range)} size range")
            result = result[size_condition]

            # Cut the table if the first cohort has NA at the end
            while np.isnan(result.iloc[0, -1]):
                result = result.iloc[:, :-1]

        if not config.include_period0 and 0 in result.columns:
            result = result.drop(columns=[0])
            # Exclude the last cohort if it only has 0 lifetime"
            # if np.isnan(result.iloc[-1, 1]):
            result = result.iloc[:-1]
            if self.config_cohort.summary_stat and not self.summary_stat_result.empty and 0 in self.summary_stat_result.columns:
                self.summary_stat_result = self.summary_stat_result.drop(columns=[0])
        self.result = result

    def _calculate_summary_stat(self) -> None:
        """Calculate weighted mean"""
        config = self.config_cohort
        result = self.result
        cohort_sizes = self.cohort_sizes
        stat = self.config_cohort.summary_stat
        if stat == 'mean':
            self.summary_stat_result = self.result.mean().to_frame('Mean').T
        elif stat == 'weighted_mean':
            cohort_sizes_extend = pd.DataFrame(index=result.index, columns=result.columns)
            for col in cohort_sizes_extend.columns:
                cohort_sizes_extend[col] = cohort_sizes
            cohort_sizes_extend[result.isna()] = np.nan
            cohort_sizes_sum_by_lifetime = cohort_sizes_extend.sum()
            weighted_mean = result.mul(cohort_sizes_extend[0], axis=0).sum().astype(float)
            weighted_mean = weighted_mean / cohort_sizes_sum_by_lifetime
            self.summary_stat_result = weighted_mean.to_frame('Wtd Mean').T
        elif stat == 'median':
            self.summary_stat_result = self.result.median().to_frame('Median').T
        elif stat == 'min':
            self.summary_stat_result = self.result.min().to_frame('Min').T
        elif stat == 'max':
            self.summary_stat_result = self.result.max().to_frame('Max').T



    def _plot_heatmap(self) -> CustomFigure:
        """
        Create a triangular heatmap visualization of the cohort analysis results.
        """
        config = self.config_cohort
        result = self.result
        text_auto = config.text_auto
        color_continuous_scale = config.color_continuous_scale
        title = config.title
        height = config.height
        width = config.width
        show_colorbar = config.show_colorbar
        summary_stat = config.summary_stat
        xgap = self.config_cohort.xgap
        ygap = self.config_cohort.ygap
        row_heights = self.config_cohort.row_heights
        vertical_spacing = self.config_cohort.vertical_spacing
        xaxis_title_text = self.config_cohort.xaxis_title
        yaxis_title_text = self.config_cohort.yaxis_title
        if not height:
            height = 600
        if not width:
            width = 1100
        result.index = result.index.strftime(self.config_cohort.date_format)
        result.columns = list(map(str, result.columns))

        # Create default title if not provided
        if title is None:
            title = self._get_title()

        if text_auto == True:
            if self.config_cohort.mode == 'retention':
                text_auto='.1%'
            else:
                decimal_places = self._determine_decimal_places(result)
                # Prepare text for display
                text_auto = f'.{decimal_places}f'
        # Replace NaN with empty string for using text_auto parameter
        result = result.map(lambda x: '' if pd.isna(x) else f'{x}')
        # Create the main heatmap
        fig_main = px.imshow(
            result,
            text_auto=text_auto,
            color_continuous_scale=color_continuous_scale,
            aspect='auto',  # Makes heatmap use full width
        )
        fig_main.update_traces(xgap=xgap, ygap=ygap)
        # Customize hover text for main heatmap
        if config.mode == 'retention':
            hovertemplate="Cohort: %{y}<br>Lifetime: %{x}<br>Value: %{z:.2%}<extra></extra>"
        else:
            hovertemplate="Cohort: %{y}<br>Lifetime: %{x}<br>Value: %{z:" + text_auto + "}<extra></extra>"
        fig_main.update_traces(
            hovertemplate=hovertemplate
        )
        # Create the mini heatmap if needed
        if summary_stat and not self.summary_stat_result.empty:
            fig_mini = px.imshow(
                self.summary_stat_result,
                text_auto=text_auto,
                color_continuous_scale=color_continuous_scale,
                aspect='auto',  # Makes heatmap use full width
            )
            fig_mini.update_traces(xgap=xgap, ygap=ygap)
            # Hide colorbar for the mini heatmap
            fig_mini.update_coloraxes(showscale=False)

            # Customize hover text for mini heatmap if it exists
            fig_mini.update_traces(
                hovertemplate="Cohort: %{y}<br>Lifetime: %{x}<br>Value: %{z:" + text_auto + "}<extra></extra>"
            )

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=row_heights,
                vertical_spacing=vertical_spacing,  # Adjust spacing between subplots
                subplot_titles=(None, None)
            )

            # Add main heatmap to the first subplot
            for trace in fig_main.data:
                fig.add_trace(trace, row=1, col=1)

            # Add mini heatmap to the second subplot
            for trace in fig_mini.data:
                fig.add_trace(trace, row=2, col=1)

            # Customize x-axis for the mini heatmap
            fig.update_xaxes(
                showline=False,
                showticklabels=False,
                # ticks='',
                row=2, col=1
            )

            # Customize y-axis for the mini heatmap
            fig.update_yaxes(
                showline=False,
                # ticks='',
                # showticklabels=False,
                row=2, col=1
            )
        else:
            fig = fig_main

        # Customize axes for main heatmap
        xaxis_title_text = xaxis_title_text if xaxis_title_text else f'Lifetime ({self.config_cohort.granularity})'
        fig.update_xaxes(
            title_text=xaxis_title_text,
            title_standoff=5,
            side='top',
            showgrid=False,
            showline=True,
            linecolor='#99A3B5',
            ticks='outside',
            showticklabels=True,
            row=1, col=1,
        )
        yaxis_title_text = yaxis_title_text if yaxis_title_text else 'Cohort'
        fig.update_yaxes(
            title_text=yaxis_title_text,
            title_standoff=5,
            type='category',
            autorange='reversed',  # Newest cohorts at top
            showgrid=False,
            showline=True,
            linecolor='#99A3B5',
            ticks='outside',
            row=1, col=1,
        )
        # Add a single colorbar for the main heatmap
        if xaxis_title_text:
            fig.update_layout(title_y=0.97)
        fig.update_layout(
            coloraxis=dict(
                colorscale=color_continuous_scale,
                showscale=show_colorbar
            ),
            title_text=title,
            height=height,
            width=width,
            margin=dict(l=10, r=10, b=10, t=80)
        )
        if config.mode == 'retention':
            fig.update_coloraxes(
                colorbar_tickformat=".0%"
            )

        return CustomFigure(fig)

    def _plot_summary(self) -> CustomFigure:
        """Create line plot of summary statistics."""
        config = self.config_cohort
        summary_data = self.summary_stat_result.iloc[0]
        yaxis_title_text = self.config_cohort.yaxis_title
        height = config.height
        width = config.width
        if not height:
            height = 400
        if not width:
            width = 800
        # Create default title if not provided
        yaxis_label = f"{config.summary_stat.replace('_', ' ').title()} Value"
        if config.title is None:
            title = self._get_title()
            if not yaxis_title_text:
                yaxis_label = title.split('by Lifetime')[0]
        else:
            title = config.title
        hover_data = {}
        if self.config_cohort.mode == 'retention':
            hover_data = {'value': ':.1%'}
        else:
            hover_data = {'value': ':.2s'}
        # Create line plot
        fig = px.line(
            summary_data.to_frame('value').reset_index(),
            x='_lifetime_',
            y='value',
            labels={'_lifetime_': f'Lifetime ({self.config_cohort.granularity})', 'value': yaxis_label},
            hover_data=hover_data,
            title=title
        )

        # Customize layout
        fig.update_layout(
            height=height,
            width=width,
            margin=dict(l=10, r=10, b=10, t=50)
        )
        if config.mode == 'retention':
            fig.update_layout(
                yaxis=dict(
                    tickformat=".0%",
                ),
            )
        return CustomFigure(fig)

    def _determine_decimal_places(self, data: pd.DataFrame) -> None:
        """Determin decimal places for text_auto parameter plotly"""
        # Filter out NaN values
        valid_values = data[~pd.isna(data)]
        if valid_values.empty:
            return 0

        # Calculate the range of the valid values
        value_range = valid_values.max().max() - valid_values.min().min()

        # Determine the number of decimal places based on the range
        if value_range > 10:
            return 0
        elif value_range > 1:
            return 1
        elif value_range > 0.1:
            return 2
        elif value_range > 0.01:
            return 3
        elif value_range > 0.001:
            return 4
        elif value_range > 0.0001:
            return 5
        else:
            return 6

    def _get_title(self) -> str:
        """Get display name for current metric mode."""
        config = self.config_cohort
        metric_names = {
                'retention': 'Retention Rate',
                'users': 'Number of Active Users',
                'buyers': 'Number of Active Buyers',
                'orders': 'Number of Orders',
                'sales': 'Number of Sales',
                'revenue': 'Revenue',
                'revenue_cumsum': 'Cumulative Revenue',
                'arpu': 'Average Revenue Per User',
                'arppu': 'Average Revenue Per Paying User',
                'apc': 'Average Payment Count',
                'aov': 'Average Order Value',
                'ltv': 'Lifetime Value',
                'romi': 'Return on Marketing Investment',
                'ltv_cac_ratio': 'LTV to CAC Ratio',
                'churn_rate': 'Churn Rate'
            }
        if config.mode in ['aov', 'apc', 'arpu', 'arppu']:
            stat_names = {
                'mean': 'Mean of',
                'median': 'Median of',
                'weighted_mean': 'Weighted Mean of',
                'min': 'Minimum of',
                'max': 'Maximum of'
            }
        else:
            stat_names = {
                'mean': 'Mean',
                'weighted_mean': 'Weighted Mean',
                'median': 'Median',
                'min': 'Minimum',
                'max': 'Maximum'
            }
        if config.mode == 'ltv' and config.margin == 1:
            base_name = "LTV (Revenue-Based)"
        else:
            base_name = metric_names.get(config.mode, config.mode)
        stat_name = stat_names.get(config.summary_stat, config.summary_stat)

        if config.display_mode == 'summary':
            title = ''
            if config.cumulative and config.mode not in ['revenue_cumsum', 'ltv']:
                title += 'Cumulative'
            title = f"{stat_name} {base_name} by Lifetime"
        else:
            title = f"Cohort Analysis - {base_name}"
            if config.cumulative and config.mode not in ['revenue_cumsum', 'ltv']:
                title += " (Cumulative)"

        return title