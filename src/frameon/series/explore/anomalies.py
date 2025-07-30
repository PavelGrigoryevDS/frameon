import dis
import itertools
from enum import Enum, auto
from typing import List, Literal, Optional, Tuple, Callable, TYPE_CHECKING, Union, overload

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import t
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from statsmodels.robust import mad
import plotly.io as pio
from frameon.utils.plotting import BarLineAreaBuilder, CustomFigure
from frameon.utils.plotting.base_histogram import HistogramBuilder
from frameon.utils.miscellaneous import (
    analyze_anomalies_all_categories,
    style_dataframe
)
from frameon.utils.plotting import plot_utils

if TYPE_CHECKING: # pragma: no cover
    from frameon.core.base import FrameOn, SeriesOn

__all__ = ['SeriesOnAnomaly']

class OutlierMethod(Enum):
    """Enumeration of outlier detection methods."""
    CONFIDENCE = 'confidence'      # Confidence interval
    IQR = 'iqr'             # Interquartile range
    ZSCORE = 'zscore'       # Z-score
    QUANTILE = 'quantile'      # quantile
    ISOLATION_FOREST = 'isolation_forest'  # Isolation Forest
    LOF = 'lof'             # Local Outlier Factor
    ONE_CLASS_SVM = 'one_class_svm'   # One-Class SVM
    MAD = 'mad'             # Median Absolute Deviation
    TUKEY = 'tukey'           # Tukey's method

class AggregationMethod(Enum):
    """Enumeration of data aggregation methods."""
    MEAN = auto()
    MEDIAN = auto()
    SUM = auto()
    MIN = auto()
    MAX = auto()
    FIRST = auto()
    LAST = auto()

class OutlierDetector:
    """Core outlier detection functionality shared between methods."""
    
    @staticmethod
    def detect(
        values: np.ndarray,
        method: OutlierMethod,
        threshold: float = 0.05,
        contamination: float = 0.05,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
        """
        Core outlier detection logic shared between window and non-window methods.
        
        Args:
            values: Input values (1D array)
            method: Detection method to use
            threshold: Sensitivity parameter
            contamination: Expected outlier fraction (for ML methods)
            kwargs: Method-specific parameters
            
        Returns:
            Tuple of (outlier_mask, lower_bound, upper_bound)
        """
        values = values.reshape(-1, 1)
        lower, upper = None, None
        
        if method == OutlierMethod.IQR:
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower = float(q1 - threshold * iqr)
            upper = float(q3 + threshold * iqr)
            mask = (values < lower) | (values > upper)
        elif method == OutlierMethod.CONFIDENCE:
            mean = np.mean(values)
            std = np.std(values)
            n = len(values)
            se = std / np.sqrt(n)
            ci = se * stats.t.ppf(1 - threshold/2, n-1)
            lower = float(mean - ci)
            upper = float(mean + ci)
            mask = (values < lower) | (values > upper)            
        elif method == OutlierMethod.ZSCORE:
            z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
            mask = z_scores > threshold
            lower = float(np.mean(values) - threshold * np.std(values))
            upper = float(np.mean(values) + threshold * np.std(values))

        elif method == OutlierMethod.MAD:
            median = np.median(values)
            mad = stats.median_abs_deviation(values, nan_policy='omit')
            lower = float(median - threshold * mad)
            upper = float(median + threshold * mad)
            mask = (values < lower) | (values > upper)
            
        elif method == OutlierMethod.TUKEY:
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower = float(q1 - 1.5 * iqr)
            upper = float(q3 + 1.5 * iqr)
            mask = (values < lower) | (values > upper)
            
        elif method == OutlierMethod.QUANTILE:          
            lower = float(np.quantile(values, threshold))
            upper = float(np.quantile(values, 1 - threshold))
            mask = (values < lower) | (values > upper)
        elif method == OutlierMethod.ISOLATION_FOREST:
            model = IsolationForest(
                contamination=contamination,
                **kwargs.get('iforest_params', {}))
            mask = model.fit_predict(values) == -1
            
        elif method == OutlierMethod.LOF:
            model = LocalOutlierFactor(
                n_neighbors=kwargs.get('n_neighbors', min(20, len(values))),
                contamination=contamination,
                **kwargs.get('lof_params', {}))
            mask = model.fit_predict(values) == -1
            
        elif method == OutlierMethod.ONE_CLASS_SVM:
            model = OneClassSVM(
                nu=contamination,
                **kwargs.get('svm_params', {}))
            mask = model.fit_predict(values) == -1
            
        return mask.flatten(), lower, upper

    @staticmethod
    def prepare_data(
        series: pd.Series,
        time_series: pd.Series,
        resample_freq: Optional[str] = None,
        agg_func: Optional[Union[AggregationMethod, str]] = None
    ) -> pd.DataFrame:
        """Prepares data with optional resampling."""
        df = pd.DataFrame({
            'time': time_series,
            'value': series.values
        }).sort_values('time').set_index('time')
        
        if resample_freq is not None:
            if isinstance(agg_func, str):
                agg_func = AggregationMethod[agg_func.upper()]
            
            agg_mapping = {
                AggregationMethod.MEAN: 'mean',
                AggregationMethod.MEDIAN: 'median',
                AggregationMethod.SUM: 'sum',
                AggregationMethod.MIN: 'min',
                AggregationMethod.MAX: 'max',
                AggregationMethod.FIRST: 'first',
                AggregationMethod.LAST: 'last'
            }
            df = df.resample(resample_freq).agg({'value': agg_mapping[agg_func]}).ffill()
            
        return df.dropna()

    @staticmethod
    def generate_report(
        values: pd.Series,
        mask: np.ndarray,
        method: OutlierMethod,
        threshold: float,
        bounds: Tuple[Optional[float], Optional[float]],
    ) -> dict:
        """Generates standardized outlier report."""
        if method in [OutlierMethod.ISOLATION_FOREST, OutlierMethod.LOF, OutlierMethod.ONE_CLASS_SVM]:
            report = {
                'Method': method.name,
                'Total Points': len(values),
                'Outliers Count': mask.sum(),
                'Outliers Percentage': (mask.mean() * 100) if len(values) > 0 else 0,
            }
        else:
            lower, upper = bounds
            report = {
                'Method': method.name,
                'Threshold': threshold,
                'Total Points': len(values),
                'Outliers Count': mask.sum(),
                'Outliers Percentage': (mask.mean() * 100) if len(values) > 0 else 0,
                'Bounds': f"[{lower:.3g}, {upper:.3g}]",
            }
        
        return report

class SeriesOnAnomaly:
    """
    Class containing methods for Series anomaly detection.
    
    Attributes
    ----------
    _series : SeriesOn
        The series to analyze for anomalies.
    """
    
    def __init__(self, series: "SeriesOn"):
        self._series = series
    
    # ====================== BASE METHODS ======================
    def anomalies_by_categories(
        self,
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'] = 'missing',
        custom_mask: Optional[Union[pd.Series, np.ndarray, list]] = None,
        method: Literal['iqr', 'zscore', 'quantile'] = 'quantile',
        threshold: float = 0.05,
        pct_diff_threshold: float = -100,
        include_columns: Optional[Union[str, List[str]]] = None,
        exclude_columns: Optional[Union[str, List[str]]] = None,          
    ) -> None:
        """
        Analyze anomaly distribution across all categorical columns in parent DataFrame.

        Parameters:
        -----------
        anomaly_type : str, default 'missing'
            Type of anomaly to analyze
        custom_mask : Union[pd.Series, np.ndarray, list], optional
            Boolean custom mask to detect anomalies.
            If provided, overrides `anomaly_type`.
        method : str, default 'quantile'
            For outliers: detection method
        threshold : float
            For outliers: detection threshold. Interpretation depends on method:
            
            - 'iqr': multiplier for IQR (typical values 1.5-3.0)
            - 'zscore': cutoff value in standard deviations (typical values 2.0-3.0)
            - 'quantile': probability threshold (0 < threshold < 1, typical values 0.05-0.01)
            
        pct_diff_threshold : float, default 1.0
            Minimum % difference to include in results (from -100 to 100)
        include_columns : str or List[str], optional
            Specific categorical columns to include (None for all)
        exclude_columns : str or List[str], optional
            Categorical columns to exclude from analysis     
            
        Returns:
        --------
            None                                             
        """
        if self._series.parent_df is None:
            raise ValueError("This series doesn't belong to a DataFrame")
        if custom_mask is not None:
            mask = pd.Series(custom_mask, index=self._series.index, dtype=bool)
            anomaly_type = "custom"
        else:
            # Validate threshold based on method
            if anomaly_type == 'outlier':
                if method == 'quantile' and not (0 < threshold < 1):
                    raise ValueError("For 'quantile' method, threshold must be between 0 and 1")
                elif method in ('iqr', 'zscore') and threshold <= 0:
                    raise ValueError(f"For '{method}' method, threshold must be positive")
            # Get anomaly mask
            if anomaly_type == 'missing':
                mask = self._series.isna()
            elif anomaly_type == 'duplicate':
                mask = self._series.duplicated()
            elif anomaly_type == 'outlier':
                if not pd.api.types.is_numeric_dtype(self._series):
                    raise ValueError("Outlier detection requires numeric series")
                mask = self._get_anomaly_mask('outlier', method, threshold)
            elif anomaly_type == 'zero':
                if not pd.api.types.is_numeric_dtype(self._series):
                    raise ValueError("Zero detection requires numeric series")
                mask = (self._series == 0)
            elif anomaly_type == 'negative':
                if not pd.api.types.is_numeric_dtype(self._series):
                    raise ValueError("Negative value detection requires numeric series")
                mask = (self._series < 0)
            else:
                raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        # Create temporary DataFrame for analysis
        # temp_df = pd.DataFrame({
        #     'value': self._series,
        #     **{col: self._series.parent_df[col] for col in self._series.parent_df.columns}
        # })
        df = self._series.parent_df
        if df is None or df.empty:
            print('Dataframe is None or empty')
            return
        results = analyze_anomalies_all_categories(
            df=df,
            anomaly_df=df[mask],
            pct_diff_threshold=pct_diff_threshold,
            include_columns=include_columns,
            exclude_columns=exclude_columns,
        )
        if anomaly_type == 'custom':
            caption = f"Custom anomalies distribution across categories"
        else:
            caption = f"{anomaly_type.capitalize()}s distribution across categories"
        if anomaly_type == 'outlier':
            caption += f" (method: {method}, threshold: {threshold})"
        
        display(style_dataframe(
            results,
            caption=caption,
            hide_columns=False,
            formatters={'% Diff': '{:.1f}%'}
        ))
    
    def anomalies_over_time(
        self,
        time_column: str,
        freq: str = 'D',
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'] = 'missing',
        custom_mask: Optional[Union[pd.Series, np.ndarray, list]] = None,
        method: Literal['iqr', 'zscore', 'quantile'] = 'quantile',
        threshold: float = 0.05,
        title: Optional[str] = None
    ) -> CustomFigure:
        """
        Plot anomalies over time using resampling.
        
        Parameters:
        -----------
        time_column : str
            Name of the datetime column in the parent DataFrame
        freq : str, default 'D'
            Resampling frequency (e.g., 'D', 'W', 'M', 'Y')
        anomaly_type : str, default 'missing'
            Type of anomaly to count ('missing', 'duplicate', 'outlier', 'zero', 'negative')
        custom_mask : Union[pd.Series, np.ndarray, list], optional
            Boolean custom mask to detect anomalies.
            If provided, overrides `anomaly_type`.
        method : str, default 'quantile'
            For outliers: detection method
        threshold : float
            For outliers: detection threshold. Interpretation depends on method:
            
            - 'iqr': multiplier for IQR (typical values 1.5-3.0)
            - 'zscore': cutoff value in standard deviations (typical values 2.0-3.0)
            - 'quantile': probability threshold (0 < threshold < 1, typical values 0.05-0.01)
            
        title : Optional[str]
            Custom plot title
            
        Returns:
        --------
            CustomFigure                 
        """
        # Get parent DataFrame
        parent_df = self._series._parent_df
        if anomaly_type == 'outlier':
            if method == 'quantile' and not (0 < threshold < 1):
                raise ValueError("For 'quantile' method, threshold must be between 0 and 1")
            elif method in ('iqr', 'zscore') and threshold <= 0:
                raise ValueError(f"For '{method}' method, threshold must be positive")
        if time_column not in parent_df.columns:
            raise ValueError(f"Time column '{time_column}' not found in parent DataFrame")
            
        if not pd.api.types.is_datetime64_any_dtype(parent_df[time_column]):
            raise ValueError(f"Column '{time_column}' must be datetime type")
        
        # Get anomaly mask
        if custom_mask is not None:
            mask = pd.Series(custom_mask, index=self._series.index, dtype=bool)
            anomaly_type = "custom"
        else:
            mask = self._get_anomaly_mask(anomaly_type, method, threshold)
        if not mask.any():
            print(f"No {anomaly_type} found in specified columns")
            return None
        # Create temporary DataFrame with time and anomalies
        temp_df = pd.DataFrame({
            'date': parent_df[time_column],
            'anomaly': mask.astype(int),
        })

        labels = pd.Series(dict(
            date = 'Date'
            , anomaly = f'{anomaly_type.capitalize()} Count'
        ))
        builder = BarLineAreaBuilder('line')
        if anomaly_type == 'custom':
            default_title = f"Custom Anomalies Over Time ({freq})"
        else:
            default_title = f"{anomaly_type.capitalize()} Anomalies Over Time ({freq})"
        if anomaly_type == 'outlier':
            default_title += f" (method: {method}, threshold: {threshold})"
        params = dict(
            data_frame = temp_df
            , x=labels.index[0]
            , y=labels.index[1]
            , agg_func='sum'
            , freq=freq
            , labels=labels
            , title=title or default_title
        )
        return builder.build(**params)
    
    # ====================== ADDITIONAL USEFUL METHODS ======================
    
    def detect_anomalies(
        self,
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'] = 'missing',
        method: Literal['iqr', 'zscore', 'quantile'] = 'quantile',
        threshold: float = 0.05
    ) -> pd.Series:
        """
        Detects anomalies in the series using the specified method.

        Parameters
        ----------
        anomaly_type : str
            Type of anomalies to detect. Options are:
            
            - 'missing': Missing values (NaN)
            - 'duplicate': Duplicate values
            - 'outlier': Statistical outliers
            - 'zero': Zero values
            - 'negative': Negative values

        method : str, optional
            Detection method for outliers. Options are:
            
            - 'iqr': Interquartile range method
            - 'zscore': Z-score method
            - 'quantile': Quantile-based method
            
            Default is 'quantile'.

        threshold : float, optional
            For outliers: detection threshold. Interpretation depends on method:
            
            - 'iqr': multiplier for IQR (typical values 1.5-3.0)
            - 'zscore': cutoff value in standard deviations (typical values 2.0-3.0)
            - 'quantile': probability threshold (0 < threshold < 1, typical values 0.05-0.01)
            
            Default is 1.5.

        Returns
        -------
        pd.Series
            Boolean mask where True indicates anomalies in the series.
        """
        # Validate not empty Series
        if self._series.empty:
            raise ValueError(
                "Series is empty."
            )   
        return self._get_anomaly_mask(anomaly_type, method, threshold)
    
    def plot_rolling_anomaly_rate(
        self,
        time_column: str,
        window: int = 30,
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'] = 'missing',
        method: Literal['iqr', 'zscore', 'quantile'] = 'quantile',
        threshold: float = 0.05,
        min_periods: int = 30,
        threshold_rate: Optional[float] = None,
        **plotly_kwargs
    ) -> Union[pd.Series, None]:
        """
        Calculate and visualize the rolling rate of specified anomalies in a time series.
        
        This method helps monitor data quality over time by showing how the frequency
        of different types of anomalies changes. It's particularly useful for:
        
        - Detecting periods of degraded data quality
        - Monitoring the impact of data cleaning processes
        - Identifying systematic issues in data collection    
            
        Parameters:
        -----------
        time_column : str
            Name of the datetime column in the parent DataFrame that will be used
            as the time axis. The column must contain datetime values.
            
        window : int, default: 30
            Size of the rolling window.
            
        anomaly_type : {'missing', 'duplicate', 'outlier', 'zero', 'negative'}, default: 'missing'
            Type of anomalies to detect:
            
            - 'missing': NaN or None values
            - 'duplicate': consecutive duplicate values
            - 'outlier': statistically unusual values
            - 'zero': zero values
            - 'negative': negative values (for numeric series only)
            
        method : {'iqr', 'zscore', 'quantile'}, default: 'quantile'
            Detection method for outliers (only used when anomaly_type='outlier'):
            
            - 'iqr': uses interquartile range (threshold is multiplier for IQR)
            - 'zscore': uses standard deviations from mean (threshold is z-score)
            - 'quantile': uses percentile thresholds (threshold is quantile, e.g., 0.05 for 5%)
            
        threshold : float, default: 0.05
            Sensitivity threshold for anomaly detection. Interpretation depends on method:
            
            - For 'iqr': typically 1.5 (mild) to 3.0 (extreme outliers)
            - For 'zscore': typically 2.0 to 3.0 standard deviations
            - For 'quantile': must be between 0 and 0.5 (e.g., 0.05 detects top/bottom 5%)
            
        threshold_rate : float, optional
            If provided, adds a horizontal line to mark an acceptable anomaly rate threshold.
            For example, 0.1 would mark 10% anomaly rate threshold.
            Value should be between 0 and 1.
            
        plotly_kwargs
            Additional keyword arguments passed to plotly.express.line().
            
            Common options include:
            
            - template: plotly template name (e.g., 'plotly_dark')
            - width: figure width in pixels
            - height: figure height in pixels
            - labels: dictionary for axis labels
            - color_discrete_sequence: color for the line
            
        Returns:
        --------
        pd.Series
            Series with datetime index containing the rolling anomaly rate (values between 0 and 1).
        """
        if self._series.parent_df is None:
            raise ValueError("Series must have a parent DataFrame")
        if not isinstance(window, int):
            raise ValueError("window must integer")
            
        if anomaly_type == 'outlier':
            if method == 'quantile' and not (0 < threshold < 1):
                raise ValueError("For 'quantile' method, threshold must be between 0 and 1")
            elif method in ('iqr', 'zscore') and threshold <= 0:
                raise ValueError(f"For '{method}' method, threshold must be positive")
            availabel_methods = ['iqr', 'zscore', 'quantile']
            if method not in availabel_methods:
                raise ValueError(f"method must be one of {availabel_methods}")
                
        if time_column not in self._series.parent_df.columns:
            raise ValueError(f"Time column '{time_column}' not found in parent DataFrame")
        
        if not pd.api.types.is_datetime64_any_dtype(self._series.parent_df[time_column]):
            raise ValueError(f"Column '{time_column}' must be datetime type")
        if len(self._series) == 0:
            print("Series is empty")
            return None
        mask = self._get_anomaly_mask(anomaly_type, method, threshold)
        if len(self._series) == 0:
            print("No anomalies detected")
            return None
        time_series = self._series.parent_df[time_column]
        
        temp_df = pd.DataFrame({
            'time': time_series,
            'anomaly': mask.astype(int)
        }).sort_values('time')
        temp_df['anomaly_rate'] = temp_df['anomaly'].rolling(
            window = window, 
        ).mean()
        anomaly_rate = temp_df[['time', 'anomaly_rate']].set_index('time')
        return self._make_rolling_anomaly_rate_fig(anomaly_rate, window=window, threshold_rate=threshold_rate, plotly_kwargs=plotly_kwargs)
    
    def _make_rolling_anomaly_rate_fig(
        self,
        anomaly_rate,
        plotly_kwargs: dict,
        window: int = 30,
        threshold_rate: Optional[float] = None,
    ):
        plotly_kwargs.setdefault('height', 350)
        plotly_kwargs.setdefault('width', 800)
        fig = px.line(
            anomaly_rate,
            title=f"Rolling Anomaly Rate (window = {window})",
            labels={'value': 'Anomaly Rate', 'time': 'Date'},
            **plotly_kwargs
        )
        
        if threshold_rate is not None:
            fig.add_hline(
                y=threshold_rate,
                line_dash="dot",
                annotation_text=f"Threshold ({threshold_rate:.0%})",
                line_color="red"
            )
        
        fig.update_layout(
            yaxis_tickformat='.0%',
            hovermode='x', 
            showlegend=False
        )
        return CustomFigure(fig)
    
    # ====================== HELPER METHODS ======================
    
    def _get_anomaly_mask(
        self,
        anomaly_type: str,
        method: Literal['iqr', 'zscore', 'quantile'] = 'iqr',
        threshold: float = 1.5
    ) -> pd.Series:
        """Helper method to get boolean mask for specified anomalies"""
        if anomaly_type == 'missing':
            return self._series.isna()
        elif anomaly_type == 'duplicate':
            return self._series.duplicated()
        elif anomaly_type == 'outlier':
            if not pd.api.types.is_numeric_dtype(self._series):
                raise ValueError("Outlier detection requires numeric series")
                
            if method == 'iqr':
                q1 = self._series.quantile(0.25)
                q3 = self._series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                return (self._series < lower) | (self._series > upper)
            elif method == 'zscore':
                zscore = np.abs((self._series - self._series.mean()) / self._series.std())
                return pd.Series(zscore > threshold, index=self._series.index)
            elif method == 'quantile':
                lower = self._series.quantile(threshold)
                upper = self._series.quantile(1 - threshold)
                return (self._series < lower) | (self._series > upper)
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
        elif anomaly_type == 'zero':
            if not pd.api.types.is_numeric_dtype(self._series):
                raise ValueError("Zero detection requires numeric series")
            return (self._series == 0)
        elif anomaly_type == 'negative':
            if not pd.api.types.is_numeric_dtype(self._series):
                raise ValueError("Negative value detection requires numeric series")
            return (self._series < 0)
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
    
    # Outlier
    def detect_window_outliers(
        self,
        time_column: str,
        window: int = 30,
        method: Union[OutlierMethod, str] = OutlierMethod.CONFIDENCE,
        threshold: float = 0.05,
        show_plot: bool = True,
        resample_freq: Optional[str] = 'D',
        agg_func: Optional[Union[AggregationMethod, str]] = AggregationMethod.MEAN,
        return_outliers: bool = False,
        show_report: bool = True,
        height: int = 400, 
        width: int = 800,  
        **kwargs
    ) -> Union[CustomFigure, Tuple[CustomFigure, pd.DataFrame], pd.DataFrame, None]:
        """
        Detect and analyze outliers in rolling windows of time series data.

        The method:
        
        1. First aggregates the data (if resample_freq specified) using the chosen function
        2. Then applies outlier detection to values within each rolling window
        3. Finally visualizes results and provides detailed statistics

        Key Features:
        
        - Rolling analysis is performed on AGGREGATED VALUES (not raw points)
        - For detailed investigation:
        
            * Use return_outliers=True to get detected anomalies
            * Select specific date ranges from the plot and analyze raw data separately
            * Supports both statistical and machine learning methods

        Parameters:
        -----------
        time_column : str
            Column containing timestamps. Must be convertible to datetime.
        window : int, default 30
            Rolling window size.
            
        method : str or OutlierMethod
            Outlier detection method to use (case-insensitive). Supported methods:
            
            - 'CONFIDENCE': Confidence interval method
            - 'IQR': Interquartile range method
            - 'ZSCORE': Z-score method
            - 'QUANTILE': Quantile-based method
            - 'MAD': Median Absolute Deviation
            - 'TUKEY': Tukey's fences method
            - 'ISOLATION_FOREST': Isolation Forest algorithm
            - 'LOF': Local Outlier Factor
            - 'ONE_CLASS_SVM': One-Class SVM
            
        threshold : float, optional
            Sensitivity parameter (interpretation varies by method):
            
            - CONFIDENCE: alpha level (e.g., 0.05 for 95% CI)
            - IQR: multiplier for IQR (default 1.5)
            - ZSCORE: number of standard deviations from mean (e.g., 3 for 3σ)
            - QUANTILE: quantile threshold (0 < threshold < 1, typical 0.01-0.05)
            - MAD: multiplier for MAD (similar to z-score)
            - TUKEY: parameter ignored (uses fixed 1.5*IQR)
            - ISOLATION_FOREST/LOF/SVM: not used (use 'contamination' instead)
            
        show_plot : bool, default True
            Whether to display interactive plot.
        resample_freq : str, optional
            Frequency for data resampling (e.g., '1H', '15T') or None for no resampling.
        agg_func : str, default 'mean'
            Aggregation function for resampling ('mean', 'median', 'sum', etc.).
        return_outliers : bool, default True
            Whether to return DataFrame with detected outliers.
        show_report : bool, default True
            Whether to show detection statistics report.
        height: int 
            Plotly figure height
        width: int
            Plotly figure width
        kwargs : dict, optional
            Additional method-specific parameters:
            
            - For ML methods: contamination, n_neighbors, eps, etc.
            - For Isolation Forest: n_estimators, max_samples, etc.

        Returns:
        --------
        CustomFigure or pd.DataFrame or tuple or None
            Depending on parameters:
            
            - CustomFigure: if show_plot=True (default)
            - Tuple[CustomFigure, pd.DataFrame]: if show_plot and return_outliers
            - pd.DataFrame: if return_outliers and not show_plot
            - None: if both False

            The returned DataFrame contains:
            
            - Original aggregated values
            - Outlier flags (boolean)
            - Method-specific columns (bounds, scores etc.)
            - Timestamps for period analysis

        Raises:
        -------
        ValueError
            If invalid parameters or data issues.
        TypeError
            If incorrect parameter types.
        RuntimeError
            If algorithm fails to converge (ML methods).

        Notes:
        ------
        Threshold behavior:
        
        - Statistical methods (CONFIDENCE, IQR, STD, QUANTILE, MAD):
        
            * Higher thresholds make detection less sensitive (fewer outliers)
            * Typical values:
            
                - 1.5-3.0 for IQR/MAD
                - 2-3 for STD
                - 0.01-0.05 for CONFIDENCE

        - ML methods (ISOLATION_FOREST, LOF, SVM):
        
            * Use 'contamination' parameter
            * Threshold parameter is ignored
        """
        # Validate and convert inputs
        method = self._validate_method(method)
        time_series = self._validate_time_column(time_column)
        
        # Prepare data
        df = OutlierDetector.prepare_data(
            self._series,
            time_series,
            resample_freq,
            agg_func
        )
        # Apply detection
        if method in [OutlierMethod.ISOLATION_FOREST, OutlierMethod.LOF, 
                     OutlierMethod.ONE_CLASS_SVM]:
            result = self._apply_ml_window(df, window, method, **kwargs)
        else:
            result = self._apply_statistical_window(df, window, method, threshold)
        
        # Generate outputs
        mask = result['is_outlier'].values
        bounds = (
            float(result['lower_bound'].mean()) if 'lower_bound' in result else None,
            float(result['upper_bound'].mean()) if 'upper_bound' in result else None
        )
        
        if show_report:
            report = OutlierDetector.generate_report(
                pd.Series(mask),  # Dummy series for count
                mask,
                method,
                threshold,
                bounds
            )   
            report = pd.DataFrame(report, index=[0])
            report = report.rename(columns={'Bounds': 'Bounds (mean)'}) 
            caption = f'Outliers in "{self._series.name}"'
            display(style_dataframe(
                report,
                caption=caption,
                hide_columns=False,
                formatters={
                    'Outliers Percentage': '{:.2f}%'
                    , 'Threshold': '{:.2f}'
                }
            ))
        if show_plot:
            # self._plot_window_results(result, method.name, bounds)
            plot = self._plot_window_results(result, method.name, threshold, height, width)
            
        if return_outliers:
            return (plot, result[mask])
        else:
            return plot
    
    def detect_outliers(
        self,
        method: Union[OutlierMethod, str] = OutlierMethod.QUANTILE,
        threshold: float = 0.05,
        contamination: float = 0.05,
        show_report: bool = True,
        show_plot: bool = True,
        return_outliers: bool = False,
        height: int = None, 
        width: int = None,  
        **kwargs
    ) -> Union[pd.DataFrame, None]:
        """
        Detect outliers in series using statistical and machine learning methods.

        Supports multiple detection algorithms with different threshold interpretations.

        Parameters:
        -----------
        method : str or OutlierMethod
            Outlier detection method to use (case-insensitive). Supported methods:
            
            - 'CONFIDENCE': Confidence interval method
            - 'IQR': Interquartile range method
            - 'ZSCORE': Z-score method
            - 'QUANTILE': Quantile-based method
            - 'MAD': Median Absolute Deviation
            - 'TUKEY': Tukey's fences method
            - 'ISOLATION_FOREST': Isolation Forest algorithm
            - 'LOF': Local Outlier Factor
            - 'ONE_CLASS_SVM': One-Class SVM

        threshold : float, optional
            Sensitivity parameter (interpretation varies by method):
            
            - CONFIDENCE: alpha level (e.g., 0.05 for 95% CI)
            - IQR: multiplier for IQR (default 1.5)
            - ZSCORE: number of standard deviations from mean (e.g., 3 for 3σ)
            - QUANTILE: quantile threshold (0 < threshold < 1, typical 0.01-0.05)
            - MAD: multiplier for MAD (similar to z-score)
            - TUKEY: parameter ignored (uses fixed 1.5*IQR)
            - ISOLATION_FOREST/LOF/SVM: not used (use 'contamination' instead)

        contamination : float, optional
            Expected outlier fraction for machine learning methods.
            Default is 0.05 (5%).

        show_report : bool, optional
            Whether to return a report with detection statistics.
            Default is True.

        return_outliers : bool, optional
            Whether to return DataFrame with detected outliers.
            Default is True.

        return_plot : bool, optional
            Whether to display interactive plot of results.
            Default is True.

        height : int, optional
            Plot height in pixels for correlation matrix.

        width : int, optional
            Plot width in pixels for correlation matrix.

        kwargs : dict, optional
            Additional method-specific parameters:
            
            - For ML methods: n_neighbors, eps, etc.
            - For Isolation Forest: n_estimators, max_samples, etc.

        Returns:
        --------
        pandas.DataFrame or None
            DataFrame with detected outliers if return_outliers=True, else None.
            Contains original data with outlier flags when applicable.

        Raises:
        -------
        ValueError
            If invalid parameters or data issues are encountered.

        TypeError
            If incorrect parameter types are provided.

        RuntimeError
            If algorithm fails to converge (ML methods).

        Notes:
        ------
        Threshold behavior:
        
        - Statistical methods (CONFIDENCE, IQR, STD, QUANTILE, MAD):
        
            * Higher thresholds make detection less sensitive (fewer outliers)
            * Typical values:
            
                - 1.5-3.0 for IQR/MAD
                - 2-3 for STD
                - 0.01-0.05 for CONFIDENCE

        - ML methods (ISOLATION_FOREST, LOF, SVM):
        
            * Use 'contamination' parameter
            * Threshold parameter is ignored
        """
        # Validate and convert inputs
        method = self._validate_method(method)
        clean_series = self._series.dropna()
        
        if len(clean_series) == 0:
            raise ValueError("Series contains only NA values after cleanup")
        
        # Apply detection
        mask, lower, upper = OutlierDetector.detect(
            clean_series.values,
            method,
            threshold,
            contamination,
            **kwargs
        )
        
        # Generate outputs
        outliers_df = pd.DataFrame({
            'value': clean_series[mask],
            'index': clean_series.index[mask]
        })

        if show_report:
            report = OutlierDetector.generate_report(
                clean_series,
                mask,
                method,
                threshold,
                (lower, upper)
            )        
            report = pd.DataFrame(report, index=[0])
            caption = f'Outliers in "{self._series.name}"'
            display(style_dataframe(
                report,
                caption=caption,
                hide_columns=False,
                formatters={
                    'Outliers Percentage': '{:.2f}%'
                    , 'Threshold': '{:.2f}'
                }
            ))
        if show_plot:
            plot = self._plot_non_window_results(clean_series, mask, method.name, (lower, upper), height, width)
            plot.show()
        if return_outliers:
            if self._series.parent_df is not None:
                return self._series.parent_df[mask]
            else:
                return outliers_df

    def _plot_window_results(
        self,
        df: pd.DataFrame,
        method: OutlierMethod,
        threshold: float,
        height: int = None, 
        width: int = None,  
    ) -> CustomFigure:
        """
        Visualizes outlier detection results with professional color schemes.
        
        Args:
            df: DataFrame with detection results
            method: Detection method used
            threshold: Threshold parameter used
        """
        # Define color schemes      
        colors = {
                'normal': '#4E79A7',      # Tableau blue
                'mean': '#59A14F',         # Tableau green
                'bounds': '#E15759',      # Tableau red
                'outliers': '#F28E2B',     # Tableau orange
                'fill': 'rgba(225,87,89,0.1)',
                'background': 'white',
                'grid': '#f0f0f0'
        }
        
        # Create figure with subplots for potential additional visual elements
        fig = go.Figure()
        # Normal values - more elegant line style
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['value'],
            name='Values',
            mode='lines',
            line=dict(
                color=colors['normal'],
                width=1.2,
                shape='spline',
                smoothing=1.3
            ),
            opacity=0.9,
            hovertemplate='Value: %{y:.2f}<extra></extra>'
        ))
        
        # Rolling mean with enhanced visibility
        if 'rolling_mean' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['rolling_mean'],
                name='Rolling Mean',
                line=dict(
                    color=colors['mean'],
                    width=2.5,
                ),
                hovertemplate='Mean: %{y:.2f}<extra></extra>'
            ))
        
        # Bounds with improved fill and dash pattern
        if 'lower_bound' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['upper_bound'],
                name='Upper Bound',
                line=dict(
                    color=colors['bounds'],
                    dash='dot',
                    width=1.5
                ),
                hovertemplate='Upper Bound: %{y:.2f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['lower_bound'],
                name='Lower Bound',
                line=dict(
                    color=colors['bounds'],
                    dash='dot',
                    width=1.5
                ),
                fill='tonexty',
                fillcolor=colors['fill'],
                hovertemplate='Lower Bound: %{y:.2f}<extra></extra>'
            ))
        
        # Outliers with more distinctive markers
        if 'is_outlier' in df.columns:
            outliers = df[df['is_outlier']]
            fig.add_trace(go.Scatter(
                x=outliers.index,
                y=outliers['value'],
                name='Outliers',
                mode='markers',
                marker=dict(
                    color=colors['outliers'],
                    size=9,
                    line=dict(
                        width=1,
                        color='white'
                    ),
                    opacity=0.9,
                    symbol='diamond'
                ),
                hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Outlier: %{y:.2f}<extra></extra>'
            ))
        
        # Method names for title
        method_names = {
            OutlierMethod.CONFIDENCE: f"{100*(1-threshold)}% CI",
            OutlierMethod.IQR: f"IQR (k={threshold})",
            OutlierMethod.QUANTILE: f"Quantile (q={threshold})",  
            OutlierMethod.MAD: f"MAD (k={threshold})",
            OutlierMethod.TUKEY: "Tukey's Fences",
            OutlierMethod.ISOLATION_FOREST: "Isolation Forest",
            OutlierMethod.LOF: "Local Outlier Factor",
            OutlierMethod.ONE_CLASS_SVM: "One-Class SVM",
        }
        
        # Layout configuration
        fig.update_layout(
            title=f"Outlier Detection: {method_names.get(method, str(method))}",
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.01,
                xanchor='center',
                x=0.5,
                # bordercolor='rgba(0,0,0,0.1)',
                # borderwidth=1,
            ),
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=70, b=20),
            height=height,
            width=width,
        )
        return CustomFigure(fig)

    def _plot_non_window_results(
        self,
        series: pd.Series,
        mask: np.ndarray,
        method_name: str,
        bounds: Tuple[Optional[float], Optional[float]],
        height: int = None, 
        width: int = None,  
    ) -> CustomFigure:
        """Visualizes non-windowed outlier detection results."""
        lower, upper = bounds
        # Format the boundaries (if any)
        bounds_str = (
            f" | Bounds: [{lower:.3g}, {upper:.3g}]" 
            if (lower is not None and upper is not None) 
            else ""
        )
        labels = {series.name: 'Outliers'}
        builder = HistogramBuilder()
        height = height if height else 350
        width = width if width else 450
        params = dict(
            x=series[mask]
            , labels=labels
            , title=f"Outlier Detection ({method_name}){bounds_str}" 
            , height=height
            , width=width
        )
        fig = builder.build(**params)        
        return fig

    def _validate_method(self, method: Union[OutlierMethod, str]) -> OutlierMethod:
        """Validates and converts method input."""
        if isinstance(method, str):
            try:
                return OutlierMethod[method.upper()]
            except KeyError:
                valid = [m.name.lower() for m in OutlierMethod]
                raise ValueError(
                    f"Invalid method '{method}'. Valid methods: {', '.join(valid)}"
                )
        return method

    def _validate_time_column(self, time_column: str) -> pd.Series:
        """Validates time column exists and is datetime."""
        if not hasattr(self._series, 'parent_df'):
            raise ValueError("Series must have parent DataFrame")
        if time_column not in self._series.parent_df.columns:
            raise ValueError(f"Time column '{time_column}' not found")
        
        ts = self._series.parent_df[time_column]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            raise ValueError(f"Column '{time_column}' must be datetime")
        return ts

    def _apply_ml_window(
        self,
        df: pd.DataFrame,
        window: str,
        method: OutlierMethod,
        **kwargs
    ) -> pd.DataFrame:
        """Applies ML detection in rolling window."""
        result = df.copy()
        window_size = window  
        
        for i in range(len(df)):
            window_data = df['value'].iloc[max(0, i-window_size+1):i+1].values.reshape(-1, 1)
                
            try:
                mask, _, _ = OutlierDetector.detect(
                    window_data,
                    method,
                    contamination=kwargs.get('contamination', 0.05),
                    **kwargs
                )
                result.loc[df.index[i], 'is_outlier'] = mask[-1]
            except Exception as e:
                print(f"Error in window {i}: {str(e)}")
                result.loc[df.index[i], 'is_outlier'] = False
                
        return result

    def _apply_statistical_window(
        self,
        df: pd.DataFrame,
        window: str,
        method: OutlierMethod,
        threshold: float,
    ) -> pd.DataFrame:
        """Applies statistical detection in rolling window."""
        df = (
            df.sort_index()
            .reset_index()
        )
        result = df.copy()
        rolling_mean = (
            df['value']
            .rolling(window)
            .mean()
        )
        if method == OutlierMethod.CONFIDENCE:
            rolling_std = df['value'].rolling(window).std()
            n = df['value'].rolling(window).count()
            se = rolling_std / np.sqrt(n)
            ci = se * t.ppf(1 - threshold/2, n-1)
            result['lower_bound'] = rolling_mean - ci
            result['upper_bound'] = rolling_mean + ci
            
        elif method in [OutlierMethod.IQR, OutlierMethod.TUKEY]:
            q1 = df['value'].rolling(window).quantile(0.25)
            q3 = df['value'].rolling(window).quantile(0.75)
            iqr = q3 - q1
            k = 1.5 if method == OutlierMethod.TUKEY else threshold
            result['lower_bound'] = q1 - k * iqr
            result['upper_bound'] = q3 + k * iqr
            
        elif method == OutlierMethod.ZSCORE:
            rolling_std = df['value'].rolling(window).std()
            result['lower_bound'] = rolling_mean - threshold * rolling_std
            result['upper_bound'] = rolling_mean + threshold * rolling_std

        elif method == OutlierMethod.QUANTILE:
            result['lower_bound'] = df['value'].rolling(window).quantile(threshold/100)
            result['upper_bound'] = df['value'].rolling(window).quantile(1 - threshold/100)
            
        elif method == OutlierMethod.MAD:
            rolling_median = df['value'].rolling(window).median()
            mad = df['value'].rolling(window).apply(
                lambda x: stats.median_abs_deviation(x, scale='normal'))
            result['lower_bound'] = rolling_median - threshold * mad
            result['upper_bound'] = rolling_median + threshold * mad
            
        result['is_outlier'] = (
            (df['value'] < result['lower_bound']) | 
            (df['value'] > result['upper_bound'])
        )
        return result.set_index('time')
