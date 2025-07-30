import itertools
import warnings
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum, auto
from typing import (Callable, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING,
                    Union, overload)
from frameon.utils.plotting import CustomFigure
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots

from frameon.utils.miscellaneous import (analyze_anomalies_all_categories,
                                              style_dataframe)
from frameon.utils.plotting import plot_utils

if TYPE_CHECKING: # pragma: no cover
    from frameon.core.base import FrameOn, SeriesOn

__all__ = ['SeriesOnPreproc']

class SeriesOnPreproc:
    """
    Class containing methods for Series preprocessing.
    """

    def __init__(self, series: "SeriesOn"):
        self._series = series

    def to_categorical(
        self,
        method: Literal['equal_intervals', 'quantiles', 'custom_bins', 'clustering', 'rules'] = "rules",
        labels: Optional[List[str]] = None,
        n_categories: Optional[int] = None,
        bins: Optional[List[Union[float, int]]] = None,
        right: bool = True,
        fill_na_value: Optional[str] = None,
        quantiles: Optional[List[float]] = [0, 0.25, 0.75, 1],
        rules: Optional[Dict[str, Union[Callable[[pd.Series], pd.Series], Literal["default"]]]] = None,
        ordered: bool = False,
        as_category: bool = True,
        show_value_counts: bool = True,
        default_label: Optional[str] = None
    ) -> pd.Series:
        """
        Convert numerical series to categorical using specified method.

        Parameters:
        -----------
        method : str, optional (default="rules")
            Method for categorization. Options:
            
            - "equal_intervals": equal width intervals
            - "quantiles": equal frequency intervals
            - "custom_bins": use custom bin edges
            - "clustering": use clustering algorithm (k-means)
            - "rules": use custom rules defined by lambda functions

        labels : list of str, optional
            Labels for categories. Length should be (n_categories) for equal_intervals/quantiles,
            or (len(bins)-1) for custom_bins.

        n_categories : int, optional
            Number of categories to create (for equal_intervals/quantiles/clustering methods).

        bins : list of float/int, optional
            Bin edges for "custom_bins" method. Should be monotonically increasing.

        right : bool, optional (default=True)
            For interval-based methods, indicates whether bins include the right edge.

        fill_na_value : str, optional
            Explicit NA fill value (overrides automatic defaults if specified)
            If None and "default" exists in rules, will use that label

        quantiles : list of float, optional
            Specific quantiles to use for "quantiles" method (e.g., [0, 0.25, 0.5, 0.75, 1.0]).

        rules : dict, optional
            For "rules" method - dictionary where:
            
            - keys are category labels
            - values can be either:
            
                * lambda functions that take the series and return boolean Series
                * special string "default" to mark this category as default

            Advanced default handling:
            
            1. If a value is "default", this label will be used for:
            
                - default_label (if parameter not explicitly set)
                - fill_na_value (if parameter not explicitly set)
                
            2. Explicit parameters have higher priority than "default" in rules
            3. If no default specified anywhere, "Unknown" will be used

            Examples:
            
            Automatic default from rules
            
                {
                "High": lambda x: x > 90,
                "Low": lambda x: x < 10,
                "Other": "default"  # Auto-used for default_label and fill_na_value
                }

            Mixed with explicit parameters (explicit has priority)
            
                {
                "Valid": lambda x: x > 0,
                "AutoDefault": "default"  # Ignored due to explicit default_label
                }
                
            to_categorical(..., default_label="ManualDefault")

        ordered : bool, optional (default=False)
            Whether to create ordered categorical (respecting labels order)

        as_category : bool, optional (default=True)
            Whether to convert result to pandas.Categorical

        show_value_counts : bool, optional (default=True)
            Whether to display value counts of the resulting categories

        default_label : str, optional
            Explicit default label (overrides "default" in rules if specified)

        Returns:
        --------
        pd.Series
            Categorical series with the same index as input.
        """
        series = self._series
        if series.empty:
            raise ValueError("Series is empty")
        auto_default = None
        if method == "rules":
            if not rules:
                raise ValueError("For 'rules' method, not empty'rules' dictionary must be provided")
            auto_default = next(
                (label for label, rule in rules.items()
                if isinstance(rule, str) and rule == "default"),
                None
            )
        if method == "equal_intervals":
            if n_categories is None:
                n_categories = len(labels) if labels is not None else 5

            if bins is not None:
                warnings.warn("'bins' parameter is ignored for 'equal_intervals' method")
            min_val = series.min()
            max_val = series.max()
            epsilon = np.finfo(float).eps * 10
            if pd.api.types.is_integer_dtype(series):
                epsilon = 1
            bins = np.linspace(min_val - epsilon, max_val + epsilon, n_categories + 1)
            result = pd.cut(series, bins=bins, labels=labels, right=right)

        elif method == "quantiles":
            if quantiles is not None:
                quantiles = sorted(set(quantiles))
                if quantiles[0] != 0 or quantiles[-1] != 1:
                    raise ValueError("Quantiles must start with 0 and end with 1")
                result = pd.qcut(
                    series.rank(method='first', na_option='keep'),
                    q=quantiles,
                    labels=labels,
                    duplicates='drop'
                )
            else:
                if n_categories is None:
                    n_categories = len(labels) if labels is not None else 5
                if series.nunique() < n_categories:
                    warnings.warn(f"Number of unique values ({series.nunique()}) is less than n_categories ({n_categories})")
                result = pd.qcut(
                    series.rank(method='first', na_option='keep'),
                    q=n_categories,
                    labels=labels,
                    duplicates='drop'
                )

        elif method == "custom_bins":
            if bins is None:
                raise ValueError("For 'custom_bins' method, 'bins' parameter must be provided")
            if labels and len(labels) != len(bins) - 1:
                raise ValueError("Length of labels must be equal to len(bins) - 1")

            result = pd.cut(series, bins=bins, labels=labels, right=right)

        elif method == "clustering":
            if n_categories is None:
                n_categories = len(labels) if labels is not None else 5

            na_mask = series.isna()
            if na_mask.any():
                warnings.warn(f"Dropping {na_mask.sum()} NA values for clustering")
                clean_series = series.dropna()

                if labels:
                    if len(labels) != n_categories + 1:
                        raise ValueError(
                            f"With NA values, number of labels ({len(labels)}) "
                            f"must match n_categories + 1 ({n_categories + 1})"
                        )
            else:
                clean_series = series.copy()
                if labels and len(labels) != n_categories:
                    raise ValueError(f"Number of labels ({len(labels)}) must match n_categories ({n_categories})")

            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_categories, random_state=42)
            clusters = kmeans.fit_predict(clean_series.values.reshape(-1, 1))

            result = pd.Series(index=series.index, dtype='object')
            result[clean_series.index] = clusters

            if labels:
                label_map = {i: labels[i] for i in range(n_categories)}
                if na_mask.any():
                    label_map[-1] = labels[-1]
                result = result.map(label_map)

        elif method == "rules":
            effective_default = default_label if default_label is not None else auto_default
            effective_fill_na = fill_na_value if fill_na_value is not None else effective_default

            filtered_rules = {
                label: rule for label, rule in rules.items()
                if not (isinstance(rule, str) and rule == "default")
            }
            if not filtered_rules:
                raise ValueError("Must provide at least one non-default rule")
            processed_rules = {}
            for label, rule in filtered_rules.items():
                if isinstance(rule, pd.Series):
                    if not pd.api.types.is_bool_dtype(rule):
                        raise ValueError(f"Rule '{label}' should be boolean Series")
                    processed_rules[label] = rule
                elif callable(rule):
                    processed_rules[label] = rule(series)
                else:
                    raise ValueError(f"Rule for {label} should be a function, boolean Series or default")
            conditions = list(processed_rules.values())
            choices = list(processed_rules.keys())
            if fill_na_value is not None:
                na_mask = series.isna()
                conditions.append(na_mask)
                choices.append(fill_na_value)
            result = pd.Series(
                np.select(
                    condlist=conditions,
                    choicelist=choices,
                    default=effective_default if effective_default is not None else "Unknown"
                ),
                index=series.index
            )

        else:
            raise ValueError(f"Unknown method: {method}. Available methods: 'equal_intervals', 'quantiles', "
                          "'custom_bins', 'clustering', 'rules'")

        # Convert to categorical if requested
        if as_category:
            result = result.astype('category')
            # Set order if requested and labels are provided
            if ordered and labels is not None:
                result = result.cat.set_categories(labels, ordered=True)
            elif ordered and method == "rules" and rules is not None:
                result = result.cat.set_categories(list(rules.keys()), ordered=True)

        # Handle NA values
        effective_fill_na = fill_na_value if fill_na_value is not None else auto_default
        if effective_fill_na is not None and result.isna().any():
            if as_category:
                result = result.cat.add_categories([effective_fill_na]).fillna(effective_fill_na)
            else:
                result = result.fillna(effective_fill_na)

        # Show value counts if requested
        if show_value_counts:
            display(result.value_counts(dropna=False).to_frame("Count").rename_axis(None))

        return result

    def smooth_time_series(
        self,
        alpha: float = 0.3,
        method: Literal['exponential', 'moving_avg', 'double', 'triple', 'median'] = 'exponential',
        window: Optional[int] = None,
        iterations: int = 1,
        inplace: bool = False,
        adjust_for_seasonality: bool = False,
        seasonality_period: Optional[int] = None,
        robust: bool = False,
        min_periods: int = 1
    ) -> Union[pd.Series, None]:
        """
        Advanced time series smoothing with multiple methodological approaches.

        Parameters:
        -----------
        alpha : float, optional (default=0.3)
            Smoothing factor between 0 and 1.
            Higher values preserve more original signal (less smoothing).
            Typical range: 0.05-0.5 for most applications.

        method : str, optional (default='exponential')
            Smoothing algorithm selection:
            
            - 'exponential': Basic exponential smoothing
            
                - Best for: General purpose smoothing, real-time applications
                - Formula: x[t] = alpha*x[t] + (1-alpha)*x[t-1]
                - Pros: Simple, efficient, maintains recent trends
                - Cons: Lags behind sudden changes
                - When to use: Default choice for most non-seasonal data
                
            - 'moving_avg': Hybrid moving average + exponential
            
                - Best for: Noisy data with stable underlying pattern
                - Pros: Reduces high-frequency noise effectively
                - Cons: Can oversmooth sudden changes
                - When to use: Sensor data, measurement smoothing
                
            - 'double': Second-order exponential smoothing
            
                - Best for: Data with trends but no seasonality
                - Pros: Captures trend direction better than basic
                - Cons: More sensitive to parameter tuning
                - When to use: Economic indicators, trend analysis
                
            - 'triple': Triple exponential (Holt-Winters) smoothing
            
                - Best for: Data with both trends and seasonality
                - Pros: Handles complex patterns well
                - Cons: Computationally heavier, needs more data
                - When to use: Sales forecasting, seasonal metrics
                
            - 'median': Robust median-based smoothing
            
                - Best for: Noisy data with outliers
                - Pros: Resistant to extreme values
                - Cons: Can create stair-step artifacts
                - When to use: Sensor data with spikes, anomaly detection

        window : int, optional
            Rolling window size for moving_avg/median methods.
            If None, auto-calculates as 5% of series length.

        iterations : int, optional (default=1)
            Number of smoothing passes (1-3 typically sufficient).

        inplace : bool, optional (default=False)
            Whether to modify the original series.

        adjust_for_seasonality : bool, optional (default=False)
            Auto-detect and adjust seasonal patterns.

        seasonality_period : int, optional
            Manual override for seasonal cycle length.

        robust : bool, optional (default=False)
            Use median instead of mean for noise resistance.

        min_periods : int, optional (default=1)
            Minimum observations required in each window.

        Returns:
        --------
        Union[pd.Series, None]
            Smoothed series with same index as input, or None if inplace=True
        """
        series = self._series.copy()

        # Auto-detect seasonality if requested
        if adjust_for_seasonality and seasonality_period is None:
            seasonality_period = self._detect_seasonality(series)

        # Choose smoothing method
        if method == 'exponential':
            result = self._exponential_smoothing(series, alpha, iterations)
        elif method == 'moving_avg':
            result = self._moving_avg_smoothing(series, alpha, window, iterations, robust, min_periods)
        elif method == 'double':
            result = self._double_exponential_smoothing(series, alpha, iterations)
        elif method == 'triple':
            result = self._triple_exponential_smoothing(series, alpha, seasonality_period, iterations)
        elif method == 'median':
            result = self._median_smoothing(series, window or len(series)//10)
        else:
            raise ValueError("Invalid method. Choose: 'exponential', 'moving_avg', 'double', 'triple', 'median'")

        if inplace:
            self._series = result
            # Update parent DataFrame if this series came from one
            if hasattr(self._series, '_parent_df') and self._series._parent_df is not None:
                parent = self._series._parent_df
                col_name = self._series.name
                parent[col_name] = result
            return
        return result

    def _exponential_smoothing(self, s: pd.Series, alpha: float, iterations: int) -> pd.Series:
        """Enhanced exponential smoothing with edge handling"""
        smoothed = s.copy()
        for _ in range(iterations):
            smoothed = alpha * smoothed + (1 - alpha) * smoothed.shift(1, fill_value=smoothed.median())
        return smoothed

    def _double_exponential_smoothing(self, s: pd.Series, alpha: float, iterations: int) -> pd.Series:
        """Second-order smoothing for trended data"""
        s1 = self._exponential_smoothing(s, alpha, iterations)
        s2 = self._exponential_smoothing(s1, alpha, iterations)
        return 2 * s1 - s2

    def _moving_avg_smoothing(self, s: pd.Series, alpha: float, window: int,
                            iterations: int, robust: bool, min_periods: int) -> pd.Series:
        """Improved moving average with robustness options"""
        if not window:
            window = max(3, len(s) // 20)

        if robust:
            ma = s.rolling(window=window, center=True, min_periods=min_periods).median()
        else:
            ma = s.rolling(window=window, center=True, min_periods=min_periods).mean()

        return self._exponential_smoothing(ma, alpha, iterations)

    def _triple_exponential_smoothing(self, s: pd.Series, alpha: float,
                                    seasonality: int, iterations: int) -> pd.Series:
        """Holt-Winters inspired seasonal smoothing"""
        if seasonality is None:
            seasonality = 1

        result = s.copy()
        for _ in range(iterations):
            # Level
            level = alpha * (result - result.shift(seasonality)) + (1 - alpha) * result
            # Trend
            trend = alpha * (level - level.shift(1)) + (1 - alpha) * level.diff().mean()
            # Seasonality
            seasonal = (s - level).rolling(seasonality, center=True).mean()

            result = level + trend + seasonal

        return result

    def _median_smoothing(self, s: pd.Series, window: int) -> pd.Series:
        """Robust median-based smoothing"""
        return s.rolling(window=window, center=True, min_periods=1).median()

    def _detect_seasonality(self, s: pd.Series, max_lag: int = 100) -> Optional[int]:
        """Auto-detect seasonality period using ACF"""
        from statsmodels.tsa.stattools import acf
        if len(s) < 10:
            return None

        max_lag = min(max_lag, len(s)//2)
        acf_values = acf(s.dropna(), nlags=max_lag)
        # Find peaks by comparing with neighbors
        peaks = []
        for i in range(1, len(acf_values)-1):
            if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                peaks.append(i)
        
        return peaks[0] if len(peaks) > 0 else None

    def transform_numeric(
        self,
        method: Literal['log', 'boxcox', 'yeojohnson', 'sqrt', 'reciprocal', 'zscore', 'robust', 'quantile'] = 'log',
        inplace: bool = False,
        show_dist: bool = True,
        **kwargs
    ) -> Union[pd.Series, None]:
        """
        Apply advanced numeric transformations with automatic visualization and skewness handling.

        Parameters:
        -----------
        method : str
            Transformation technique:
            
            - 'log': Natural logarithm (best for right-skewed data)
            
                * Reduces right tail, normalizes multiplicative relationships
                * Use when: Data spans several orders of magnitude
                * Formula: log(x + shift) where shift handles zeros
                
            - 'boxcox': Box-Cox power transformation (right-skewed)
            
                * More flexible than log for right-skewed data
                * Automatically finds optimal power parameter (λ)
                * Requires positive values (auto-shifts if needed)
                
            - 'yeojohnson': Extended Box-Cox (works with left/right skew)
            
                * Handles both positive and negative values
                * Good alternative when data has zeros or negatives
                
            - 'sqrt': Square root (mild right skew)
            
                * Less aggressive than log transform
                * Use for count data with moderate right skew
                
            - 'reciprocal': 1/x transform (left-skewed data)
            
                * Inverts distribution to handle left skew
                * Use when: Extreme left tail needs correction
                
            - 'zscore': Standardization (μ=0, σ=1)
            
                * Preserves original shape but centers/scales
                * Use for: Comparing features on same scale
                
            - 'robust': Robust scaling (median/IQR)
            
                * Resistant to outliers
                * Use when: Outliers distort z-score
                
            - 'quantile': Non-parametric normalization
            
                * Forces uniform/normal distribution
                * Use for: Non-normal data needing strict normality

        inplace : bool
            Modify series directly if True
        show_dist : bool
            Show before/after distributions (default True)
        kwargs
            Additional parameters:
            
            - shift: Value to add before log/sqrt (default 1 for log, 0 for sqrt)
            - lmbda: Box-Cox λ (None for auto)
            - eps: Small value to avoid zeros (default 1e-6)
            - dist: For quantile ('uniform' or 'normal')

        Returns:
        --------
        Union[pd.Series, None]
            Transformed series or None if inplace=True

        Skewness Guide:
        ---------------
        Right-Skewed (log, boxcox (λ<1), sqrt, x^2)
        
        Left-Skewed (reciprocal (1/x), yeojohnson (λ>1), x^3, exponential)
        """
        series = self._series
        original = series.copy()

        # Apply transformation
        result = self._apply_transform(series, method, **kwargs)

        # Visualization
        if show_dist:
            self._plot_transform_comparison(original, result, method)

        if inplace:
            self._series = result
            # Update parent DataFrame if this series came from one
            if hasattr(self._series, '_parent_df') and self._series._parent_df is not None:
                parent = self._series._parent_df
                col_name = self._series.name
                parent[col_name] = result
            return
        return result

    def _apply_transform(self, series: pd.Series, method: str, **kwargs) -> pd.Series:
        """Core transformation logic"""
        eps = kwargs.get('eps', 1e-6)
        if method in ['boxcox', 'yeojohnson'] and series.isna().any():
            raise ValueError(f"The series contains NaN values which are not allowed for the {method} transformation.")
        if method == 'log':
            shift = kwargs.get('shift', 1)
            if (series + shift <= 0).any():
                shift = abs(series.min()) + eps
            return np.log(series + shift)

        elif method == 'boxcox':
            from scipy.stats import boxcox
            shift = abs(series.min()) + eps if (series <= 0).any() else 0
            display(series + shift)
            transformed, _ = boxcox(series + shift, lmbda=kwargs.get('lmbda'))
            return pd.Series(transformed, index=series.index)

        elif method == 'yeojohnson':
            from scipy.stats import yeojohnson
            transformed, _ = yeojohnson(series)
            return pd.Series(transformed, index=series.index)

        elif method == 'sqrt':
            shift = kwargs.get('shift', 0)
            return np.sqrt(series + shift)

        elif method == 'reciprocal':
            return 1 / (series + eps)

        elif method == 'zscore':
            return (series - series.mean()) / series.std()

        elif method == 'robust':
            iqr = series.quantile(0.75) - series.quantile(0.25)
            return (series - series.median()) / (iqr + eps)

        elif method == 'quantile':
            from scipy.stats import norm, uniform
            ranks = series.rank(pct=True)
            dist = kwargs.get('dist', 'uniform')
            return pd.Series(
                norm.ppf(ranks) if dist == 'normal' else uniform.ppf(ranks),
                index=series.index
            )

        elif method == 'custom':
            if 'func' not in kwargs:
                raise ValueError("Must provide 'func' parameter for custom transform")
            return series.apply(kwargs['func'])

        else:
            raise ValueError(f"Unknown method: {method}")

    def _plot_transform_comparison(self, original: pd.Series, transformed: pd.Series, method: str):
        """Interactive before/after visualization using Plotly"""

        # Create figures with histograms and boxplots
        labels = dict(x='Value')
        fig_original = px.histogram(x=original, marginal='box', nbins=50, labels=labels)
        fig_transformed = px.histogram(x=transformed, marginal='box', nbins=50, labels=labels)

        # Create 2x2 subplot grid
        fig_new = make_subplots(rows=2, cols=2,
                            row_heights=[0.1, 0.9],
                            vertical_spacing=0.05,
                            horizontal_spacing=0.07,
                            subplot_titles=(
                                "Original Boxplot",
                                "Transformed Boxplot",
                                "Original Histogram",
                                "Transformed Histogram"
                            ))

        # Add original plot traces
        for trace in fig_original.data:
            if trace.type == 'box':
                fig_new.add_trace(trace, row=1, col=1)
                fig_new.update_xaxes(
                    showticklabels=False, showline=False,
                    ticks='', showgrid=True, row=1, col=1
                )
                fig_new.update_yaxes(visible=False, row=1, col=1)
            else:
                trace.bingroup = None
                fig_new.add_trace(trace, row=2, col=1)

        # Add transformed plot traces
        for trace in fig_transformed.data:
            if trace.type == 'box':
                fig_new.add_trace(trace, row=1, col=2)
                fig_new.update_xaxes(
                    showticklabels=False, showline=False,
                    ticks='', showgrid=True, row=1, col=2
                )
                fig_new.update_yaxes(visible=False, row=1, col=2)
            else:
                trace.bingroup = None
                fig_new.add_trace(trace, row=2, col=2)

        # Style adjustments
        fig_new.update_traces(
            marker_line_color='white',
            marker_line_width=0.3,
            selector=dict(type='histogram')
        )

        # Update layout with titles and labels
        fig_new.update_layout(
            title_text=f"Transformation: {method}",
            margin=dict(l=50, r=50, b=50, t=70),
            width=800,
            height=400,
            showlegend=False,
        )

        # Add axis labels
        fig_new.update_xaxes(title_text="Value", row=2, col=1)
        fig_new.update_xaxes(title_text="Value", row=2, col=2)
        fig_new.update_yaxes(title_text="Count", row=2, col=1)

        # Add skewness and kurtosis annotations
        annotations = [
            dict(
                x=0.25, y=1.01,
                xref="paper", yref="paper",
                text=f"Original: Skew = {original.skew():.2f}, Kurtosis = {original.kurtosis():.2f}",
                showarrow=False,
                font=dict(size=12)
            ),
            dict(
                x=0.75, y=1.01,
                xref="paper", yref="paper",
                text=f"Transformed: Skew = {transformed.skew():.2f}, Kurtosis = {transformed.kurtosis():.2f}",
                showarrow=False,
                font=dict(size=12)
            )
        ]
        fig_new.update_layout(annotations=annotations)
        CustomFigure(fig_new).show()

    def normalize_string_series(
        self,
        symbols: Optional[List[str]] = None,
        case_format: Literal['title', 'lower', 'upper', 'sentence', 'none'] = 'title',
        remove_accents: bool = True,
        replace_symbols_with: str = ' ',
        custom_replacements: Optional[Dict[str, str]] = None,
        inplace: bool = False,
    ) -> Union[pd.Series, None]:
        """
        Normalize a pandas Series of strings with comprehensive cleaning and standardization options.

        Performs multiple text normalization operations including:
        
        - Whitespace normalization (trimming, reducing multiple spaces)
        - Symbol removal/replacement
        - Case conversion
        - Accent/diacritic removal
        - Custom character replacements

        Parameters:
        -----------
        symbols : list of str, optional
            Symbols to remove/replace. Defaults to common punctuation.
        case_format : str, optional (default='title')
            Case conversion mode. Options:
            
            - 'title': Title Case (default)
            - 'lower': lowercase
            - 'upper': UPPERCASE
            - 'sentence': Sentence case
            - 'none': No case conversion
            
        remove_accents : bool, optional (default=True)
            Whether to remove diacritics/accents
        replace_symbols_with : str, optional (default=' ')
            What to replace symbols with
        custom_replacements : dict, optional
            Custom character replacements mapping
        inplace : bool, optional (default=False)
            Modify series directly if True

        Returns:
        --------
        pd.Series
            Normalized string Series with same index as input

        Raises:
        -------
        ValueError
            If input is not a pandas Series or contains non-string values
        ValueError
            If invalid case_format is specified
        """
        column = self._series
        # Input validation

        if not pd.api.types.is_string_dtype(column.dropna()):
            raise ValueError("Series must contain strings")

        # Default symbols if not provided
        if symbols is None:
            symbols = ['_', '.', ',', '«', '»', '(', ')', '"', "'", "`", "!", "?", "-", "—", "–"]

        # Preserve original categorical dtype if present
        is_column_category = isinstance(column.dtype, pd.CategoricalDtype)

        # Custom replacements
        if custom_replacements:
            for old, new in custom_replacements.items():
                column = column.str.replace(re.escape(old), new, regex=True)

        # Symbol replacement
        if symbols:
            symbols_pattern = '|'.join(map(re.escape, symbols))
            column = column.str.replace(symbols_pattern, replace_symbols_with, regex=True)

        # Whitespace normalization
        res = (
            column
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)  # Collapse multiple spaces
        )

        # Case conversion
        case_format = case_format.lower()
        if case_format == 'title':
            res = res.str.title()
        elif case_format == 'lower':
            res = res.str.lower()
        elif case_format == 'upper':
            res = res.str.upper()
        elif case_format == 'sentence':
            res = res.str.capitalize()
        elif case_format != 'none':
            raise ValueError(f"Invalid case_format: {case_format}. Must be 'title', 'lower', 'upper', 'sentence', or 'none'")
        # Remove accents/diacritics
        if remove_accents:
            res = (
                res
                .str.normalize('NFKD')
                .str.encode('ascii', errors='ignore')
                .str.decode('utf-8')
            )

        # Restore categorical dtype if original was categorical
        if is_column_category:
            res = res.astype('category')

        if inplace:
            self._series = res
            # Update parent DataFrame if this series came from one
            if hasattr(self._series, '_parent_df') and self._series._parent_df is not None:
                parent = self._series._parent_df
                col_name = self._series.name
                parent[col_name] = res
            return
        return res

    def fill_missing_by_category(
        self,
        category_columns: Union[str, List[str]],
        strategy: Literal['simple', 'hierarchical'] = "simple",
        func: Union[str, Callable] = "median",
        minimal_group_size: int = 5,
        fill_unfilled: Union[str, float, None] = "global",
        inplace: bool = False
    ) -> Optional[pd.Series]:
        """
        Fill missing values using category-based strategies

        Parameters:
        -----------
        category_columns : str or list
            Column name(s) to group by for calculating fill values
        strategy : {'simple', 'hierarchical'}, optional (default='simple')
            Filling strategy:
            
            - 'simple': Fill using exact category groups
            - 'hierarchical': Try broader category combinations
            
        func : str or callable, optional (default="median")
            Aggregation function for valid groups:
            
            - "median", "mean", "max", "min", "mode"
            - Custom function that reduces a Series
            
        minimal_group_size : int, optional (default=5)
            Minimum non-NA values required to use group statistic
        fill_unfilled : str, float or None, optional (default="global")
            Strategy for groups with insufficient data:
            
            - "global": Use overall statistic
            - numeric: Use specified constant value
            - None: Leave as NA
            
        inplace : bool, optional (default=False)
            Modify the series in-place instead of returning a copy

        Returns:
        --------
        pd.Series or None
            Filled series unless inplace=True
        """
        # Validate inputs
        self._validate_inputs(category_columns, strategy, fill_unfilled)
        
        # Get working copies
        df, target = self._get_data_objects(inplace)
        category_columns = self._normalize_categories(category_columns)
        
        # Check for missing categories
        self._check_missing_categories(df, category_columns)

        # Get aggregation function
        agg_func = self._resolve_agg_func(func)
        
        # Main filling logic
        if strategy == "simple":
            filled = self._simple_strategy(df, target, category_columns, agg_func, minimal_group_size)
        else:
            filled = self._hierarchical_strategy(df, target, category_columns, agg_func, minimal_group_size)

        # Handle remaining missing values
        filled = self._handle_remaining_nas(filled, df[target], fill_unfilled, agg_func)
        
        return self._return_result(filled, inplace)

    def _validate_inputs(self, categories, strategy, fill_unfilled):
        """Validate all input parameters"""
        valid_strategies = ["simple", "hierarchical"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Choose from {valid_strategies}")
            
        if fill_unfilled not in ["global", None] and not isinstance(fill_unfilled, (int, float)):
            raise TypeError("fill_unfilled must be 'global', None, or numeric value")

    def _get_data_objects(self, inplace: bool):
        """Get DataFrame and series name"""
        if self._series.parent_df is None:
            raise ValueError("Series must belong to a DataFrame")
            
        df = self._series.parent_df
        target = self._series.name
        return df, target

    def _normalize_categories(self, categories):
        """Convert to list if single column name"""
        return [categories] if isinstance(categories, str) else categories

    def _check_missing_categories(self, df, categories):
        """Check for NaN in categorical columns"""
        missing = df[categories].isna()
        if missing.any().any():
            bad_cols = missing.any()[missing.any()].index.tolist()
            raise ValueError(
                f"Missing values in categorical columns: {bad_cols}. "
                "Handle missing categories before filling."
            )

    def _resolve_agg_func(self, func):
        """Get appropriate aggregation function"""
        func_map = {
            "median": pd.Series.median,
            "mean": pd.Series.mean,
            "max": pd.Series.max,
            "min": pd.Series.min,
            "mode": lambda x: x.mode()[0] if not x.mode().empty else np.nan
        }
        
        if isinstance(func, str):
            if func not in func_map:
                raise ValueError(f"Invalid function: {func}. Choose from {list(func_map)}")
            return func_map[func]
            
        if callable(func):
            return func
            
        raise TypeError("func must be string or callable")

    def _simple_strategy(self, df, target, categories, agg_func, min_size):
        """Simple grouping strategy"""
        groups = df.groupby(categories, observed=True)[target]
        group_values = groups.transform(lambda x: agg_func(x) if x.count() >= min_size else np.nan)
        return df[target].fillna(group_values)

    def _hierarchical_strategy(self, df, target, categories, agg_func, min_size):
        """Hierarchical filling strategy"""
        filled = df[target].copy()
        remaining_na = filled.isna()
        display(filled)
        # Try different category combinations from specific to general
        for level in range(len(categories), 0, -1):
            for cols in itertools.combinations(categories, level):
                if not remaining_na.any():
                    return filled
                    
                # Fill with current combination
                temp_filled = self._simple_strategy(df, target, list(cols), agg_func, min_size)
                filled.update(temp_filled[remaining_na])
                print(level, cols)
                display(filled)
                remaining_na = filled.isna()
                
        return filled

    def _handle_remaining_nas(self, filled, original, fill_unfilled, agg_func):
        """Apply fill_unfilled strategy"""
        na_mask = filled.isna()
        
        if not na_mask.any() or fill_unfilled is None:
            return filled
            
        if fill_unfilled == "global":
            fill_value = agg_func(original.dropna())
        else:
            fill_value = fill_unfilled
            
        filled[na_mask] = fill_value
        return filled

    def _return_result(self, filled, inplace):
        """Handle in-place modification"""
        if inplace:
            self._series = filled
            # Update parent DataFrame if this series came from one
            if hasattr(self._series, '_parent_df') and self._series._parent_df is not None:
                parent = self._series._parent_df
                col_name = self._series.name
                parent[col_name] = filled
            return None
        return filled

    def impute_missing(
        self,
        auxiliary_cols: Union[str, List[str]] = 'all',
        method: Literal['simple', 'knn', 'iterative'] = 'simple',
        strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = 'median',
        n_neighbors: int = 5,
        sample_size: Optional[int] = None,
        random_state: int = 42,
        standardize: bool = False,
        imputer_params: Optional[Dict] = None,
        inplace: bool = False
    ) -> Optional[pd.Series]:
        """
        Perform missing value imputation on specified numerical columns.
        
        Parameters:
        -----------
        target_cols : str or list
            Numerical columns to impute (must contain missing values)
            
        auxiliary_cols : str or list, default='all'
            Columns to use as features for imputation. Can include:
            
            - Numerical columns (used directly)
            - Categorical columns (one-hot encoded)
            - Datetime columns (feature engineered)
            
            Does not include text columns.
            
        method : {'simple', 'knn', 'iterative'}, default='simple'
            Imputation strategy:
            
            - simple: Fast univariate imputation
            - knn: Nearest neighbors-based imputation
            - iterative: Multivariate imputation using chained equations
            
        strategy : str, default='median'
            Strategy for SimpleImputer: ['mean', 'median', 'most_frequent', 'constant']
            
        n_neighbors : int, default=5
            Number of neighbors for KNNImputer
            
        sample_size : int, optional
            Subsample size for large datasets optimization
            
        random_state : int, default=42
            Random seed for reproducibility
            
        standardize : bool, default=False
            Whether to standardize features before imputation.
            Recommended for knn and iterative methods.
            
        imputer_params : dict, optional
            Additional parameters for IterativeImputer:
            
            - estimator: sklearn estimator (default=BayesianRidge())
            - max_iter: int (default=10)
            - tol: float (default=1e-3)
            
        inplace : bool, default=False
            Whether to modify the original DataFrame
            
        Returns:
        --------
        pd.DataFrame or None
            DataFrame with imputed values or None if inplace=True
        """
        series_name = self._series.name
        if self._series.parent_df is None:
            raise ValueError("Series must have a parent DataFrame")
        filled = self._series.parent_df.preproc.impute_missing(
            target_cols=series_name,
            auxiliary_cols=auxiliary_cols,
            method=method,
            strategy=strategy,
            n_neighbors=n_neighbors,
            sample_size=sample_size,
            random_state=random_state,
            standardize=standardize,
            imputer_params=imputer_params,
            inplace = inplace
        )
        # Handle in-place modification
        if inplace:
            self._series = filled[series_name]
            return None
        return filled[series_name]

    def calc_target_category_share(
        self,
        target_category: Union[str, int, float],
        group_columns: List[str],
        resample_freq: str = 'ME',
        fill_missing_periods: bool = True,
        min_group_size: int = 1
    ) -> pd.DataFrame:
        """
        Calculate the proportional share of a target category within grouped data,
        with support for time-based resampling and comprehensive data validation.

        This function:
        
        1. Validates input data and parameters
        2. Calculates the percentage share of a specified category
        3. Supports both regular grouping and time-based resampling
        4. Handles edge cases and provides meaningful error messages

        Parameters:
        -----------
        target_category : str, int, or float
            The specific category value to calculate the share for

        group_columns : List[str]
            List of columns to group by

        resample_freq : str, optional
            Pandas frequency string for time resampling (default 'ME' for month-end)
            Only used if a datetime column is present in group_columns
            Common options: 'D' (daily), 'W' (weekly), 'ME' (monthly), 'QE' (quarterly)

        fill_missing_periods : bool, optional
            Whether to fill missing time periods with 0 values (default True)
            Only applies when using time-based grouping

        min_group_size : int, optional
            Minimum number of observations required per group (default 1)
            Groups with fewer observations will be assigned NaN

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the calculated shares with columns:
            
            - All grouping columns
            - 'target_share': The percentage share of the target category (0-1)
            - 'total_count': The total observations per group (optional)

        Raises:
        -------
        ValueError
            If input validation fails (missing columns, invalid types, etc.)

        """
        # ======================
        # Input Validation
        # ======================
        series = self._series
        series_name = series.name
        df = series.parent_df
        if not group_columns:
            raise ValueError('group_columns must be define')     
        group_columns = [group_columns] if isinstance(group_columns, str) else group_columns   
        if series_name in group_columns:
            raise ValueError('Current column should not be in group_columns')           
        if len(series) == 0:
            raise ValueError("Series is empty")
        
        if df is None:
            raise ValueError("Series must belong to a DataFrame")

        # Validate category column
        if series.nunique() == 0:
            raise ValueError(f"Current column has no unique values")

        # Validate target category exists
        if target_category not in series.unique():
            raise ValueError(f"Target category '{target_category}' not found in current column")

        # Validate group columns
        missing_group_cols = [col for col in group_columns if col not in df.columns]
        if missing_group_cols:
            raise ValueError(f"Group columns not found in DataFrame: {missing_group_cols}")

        # Check for datetime columns
        datetime_cols = [col for col in group_columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if len(datetime_cols) > 1:
            raise ValueError("Only one datetime column allowed in group_columns")
        time_column = datetime_cols[0] if datetime_cols else None

        # Check for missing values
        cols_to_check = [series_name] + group_columns
        for col in cols_to_check:
            if df[col].isna().any():
                raise ValueError(f"Missing values found in column: '{col}'")

        # ======================
        # Data Preparation
        # ======================

        # Create working copy
        df_work = df[cols_to_check].copy()

        # Create target indicator
        df_work['is_target'] = (df_work[series_name] == target_category).astype(int)

        # ======================
        # Grouping Logic
        # ======================

        # Prepare grouping columns
        regular_group_cols = [col for col in group_columns if col != time_column]
        
        # Handle time-based grouping
        if time_column:
            grouper = [pd.Grouper(key=time_column, freq=resample_freq)] + regular_group_cols
        else:
            grouper = regular_group_cols

        # Calculate shares
        result = (
            df_work.groupby(grouper, observed=True, as_index=False)
            .agg(
                target_share=('is_target', 'mean'),
                total_count=('is_target', 'count')
            )
        )

        # Filter small groups
        if min_group_size > 1:
            result.loc[result['total_count'] < min_group_size, 'target_share'] = np.nan

        # Fill missing time periods if requested
        if time_column and fill_missing_periods and regular_group_cols:
            # Create complete date range
            date_range = pd.date_range(
                start=result[time_column].min(),
                end=result[time_column].max(),
                freq=resample_freq
            )
            
            # Create full multi-index
            full_index = pd.MultiIndex.from_product(
                [date_range, result[regular_group_cols[0]].unique()],
                names=[time_column, regular_group_cols[0]]
            )
            
            # Reindex and fill missing
            result = (
                result.set_index([time_column, regular_group_cols[0]])
                .reindex(full_index, fill_value=np.nan)
                .reset_index()
            )

        return result.drop('total_count', axis=1)
    
    def check_group_counts(
        self,
        category_columns: Union[str, List[str]],
        threshold_counts: List[int] = [5, 10, 20, 30, 40, 50],
        return_report: bool = False
    ) -> Union[dict, None]:
        """
        Analyze group statistics to assess viability for missing value imputation.
        
        Provides detailed metrics about group sizes and missing value distribution
        to help determine appropriate imputation strategy parameters.

        Parameters:
        -----------
        category_columns : str or list
            Column name(s) used for grouping
        threshold_counts : list of int, optional (default=[5, 10, 20, 30, 40, 50])
            List of thresholds to evaluate group sizes against
        return_report : bool, optional (default=False)
            Whether to return metrics as a dictionary
            If False, prints summary to stdout

        Returns:
        --------
        Union[dict, None]
            Dictionary with metrics if return_report=True, otherwise None
        """
        # Validate inputs
        series = self._series
        value_column = series.name
        df = series.parent_df
        if df is None:
            raise ValueError("Series must belong to a DataFrame")
            
        if isinstance(category_columns, str):
            category_columns = [category_columns]
            
        missing_cols = [col for col in category_columns + [value_column] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        # Calculate basic group statistics
        group_stats = (
            df.groupby(category_columns, observed=False)[value_column]
            .agg(
                count = 'count',
                missing_count = lambda x: x.isna().sum()
            )
        )
        # Calculate metrics
        metrics = {
            'groups_total': len(group_stats),
            'groups_with_missing': (group_stats['missing_count'] > 0).mean(),
            'missing_values_total': group_stats['missing_count'].sum(),
            'groups_all_missing': (group_stats['count'] == 0).mean(),
            'missing_in_complete_groups': group_stats.loc[group_stats['count'] > 0, 'missing_count'].sum(),
            'group_size_stats': {
                'mean': group_stats['count'].mean(),
                'median': group_stats['count'].median(),
                'min': group_stats['count'].min(),
                'max': group_stats['count'].max(),
                'std': group_stats['count'].std()
            },
            'missing_distribution': {
                'groups_with_1_missing': (group_stats['missing_count'] == 1).mean(),
                'groups_with_2-5_missing': ((group_stats['missing_count'] >= 2) & 
                                        (group_stats['missing_count'] <= 5)).mean(),
                'groups_with_5+_missing': (group_stats['missing_count'] > 5).mean()
            },
            'threshold_stats': {}
        }
        
        # Calculate threshold statistics
        for threshold in sorted(threshold_counts):
            valid_groups = group_stats[group_stats['missing_count'] > 0]
            threshold_pct = (valid_groups['count'] >= threshold).mean()
            metrics['threshold_stats'][f'{threshold}'] = threshold_pct
        
        # Generate report
        self._print_group_report(metrics, category_columns, value_column)
        if return_report:
            return metrics

    def _print_group_report(self, metrics: dict, category_columns: List[str], value_column: str) -> None:
        """Print formatted group analysis report"""
        print(f"\n{' Group Analysis Report ':=^80}")
        print(f"Grouping columns: {', '.join(category_columns)}")
        print(f"Value column: {value_column}\n")
        
        print(f"{'Total groups:':<40} {metrics['groups_total']:,}")
        print(f"{'Groups with missing values:':<40} {metrics['groups_with_missing']:.1%}")
        print(f"{'Groups with ALL values missing:':<40} {metrics['groups_all_missing']:.1%}")
        print(f"{'Total missing values:':<40} {metrics['missing_values_total']:,}")
        print(f"{'Missing in non-empty groups:':<40} {metrics['missing_in_complete_groups']:,}\n")
        
        print(f"{' Group Size Statistics ':-^80}")
        stats = metrics['group_size_stats']
        print(f"{'Mean group size:':<30} {stats['mean']:.1f}")
        print(f"{'Median group size:':<30} {stats['median']:.1f}")
        print(f"{'Minimum group size:':<30} {stats['min']:,}")
        print(f"{'Maximum group size:':<30} {stats['max']:,}")
        print(f"{'Standard deviation:':<30} {stats['std']:.1f}\n")
        
        print(f"{' Missing Value Distribution ':-^80}")
        dist = metrics['missing_distribution']
        print(f"{'Groups with 1 missing value:':<30} {dist['groups_with_1_missing']:.1%}")
        print(f"{'Groups with 2-5 missing values:':<30} {dist['groups_with_2-5_missing']:.1%}")
        print(f"{'Groups with 5+ missing values:':<30} {dist['groups_with_5+_missing']:.1%}\n")
        
        print(f"{' Threshold Analysis (ontly groups with missings)':-^80}")
        for threshold, pct in metrics['threshold_stats'].items():
            print(f"{f'Groups with {threshold}+ elements:':<30} {pct:.1%}")
        print("=" * 80)    
