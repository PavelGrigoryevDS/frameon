from typing import (Any, Callable, Dict, Generator, List, Literal, Optional,
                    Tuple, TYPE_CHECKING, Union, get_type_hints)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import make_colorscale

from frameon.utils.plotting import *

from matplotlib.colors import Colormap

if TYPE_CHECKING: # pragma: no cover
    from frameon.core.base import FrameOn
    from frameon.utils.plotting.custom_figure import CustomFigure

__all__ = ['FrameOnPlots']

class FrameOnPlots:
    def __init__(self, df: "FrameOn"):
        """Initialize with the main dataframe."""
        self._df = df

    def _merge_plotly_settings(self, **kwargs) -> dict:
        """
        Merge default plotly settings with method-specific settings.
        Preserves default settings except for explicitly provided values.
        
        Special handling for:
        
        - labels: merge dictionaries with method values taking precedence
        - category_orders: merge dictionaries with method values taking precedence
        - Other settings: method values override defaults
        """
        # Get default settings from DataFrame
        default_settings = self.plotly_settings
        # Initialize merged settings with defaults
        merged_settings = default_settings.copy()
        labels = kwargs.pop('labels', {})
        if isinstance(labels, pd.Series):
            labels = labels.to_dict()
        if not labels:
            labels = {}
        category_orders = kwargs.pop('category_orders', {})
        if isinstance(category_orders, pd.Series):
            category_orders = category_orders.to_dict()
        if not category_orders:
            category_orders = {}
        # Handle labels merge (preserve defaults except for explicitly provided keys)
        merged_labels = default_settings.get('labels', {}).copy()
        merged_labels.update(labels)
        merged_settings['labels'] = merged_labels
        
        # Handle category_orders merge (preserve defaults except for explicitly provided keys)
        merged_orders = default_settings.get('category_orders', {}).copy()
        merged_orders.update(category_orders)
        merged_settings['category_orders'] = merged_orders
        # Update with remaining kwargs (method-specific settings take precedence)
        merged_settings.update(kwargs)
        return merged_settings

    def bar(
        self,
        x: Optional[Union[str, List, pd.Series]] = None,
        y: Optional[Union[str, List, pd.Series]] = None,
        color: Optional[Union[str, List, pd.Series]] = None,
        category_orders: Optional[Dict[str, Union[str, List[str]]]] = None,
        labels: Optional[Dict[str, str]] = None,
        agg_func: Optional[Union[str, Callable]] = None,
        freq: Optional[str] = None,
        agg_column: Optional[str] = None,
        norm_by: Optional[Union[str, Literal['all']]] = None,
        trim_top_n_x: Optional[int] = None,
        trim_top_n_y: Optional[int] = None,
        trim_top_n_color: Optional[int] = None,
        trim_top_n_facet_col: Optional[int] = None,
        trim_top_n_facet_row: Optional[int] = None,
        trim_top_n_animation_frame: Optional[int] = None,
        trim_top_n_direction: Literal['top', 'bottom'] = 'top',
        trim_top_n_agg_func: Literal['mean', 'median', 'sum', 'count', 'nunique'] = 'count',
        show_group_size: bool = True,
        min_group_size: Optional[int] = None,
        show_box: bool = False,
        show_count: bool = False,
        lower_quantile: Optional[Union[float, int]] = None,
        upper_quantile: Optional[Union[float, int]] = None,
        show_top_and_bottom_n: int = None,
        show_legend_title: bool = False,
        observed_for_groupby: bool = True,
        horizontal_spacing: Optional[float] = None,
        **plotly_kwargs
    ) -> CustomFigure:
        """
        Creates a bar chart using the Plotly Express library. This function is a wrapper around Plotly Express bar and accepts all the same parameters, allowing for additional customization and functionality.

        Parameters:
        -----------
        x : str, optional
            Column to use for the x-axis
        y : str, optional
            Column to use for the y-axis
        color : str, optional
            Column to use for color encoding
        agg_func : str or callable, optional
            Aggregation function. Can be a string representing a built-in aggregation function
            ('mean', 'median', 'sum', 'count', 'nunique', 'min', 'max', 'std', 'var', 'prod', 'first', 'last')
            or a callable function that can be used with pandas.groupby.
        agg_column : str, optional
            Column to aggregate
        freq : str, optional
            Resample frequency for resample. Options: 'ME', 'W', D' and others
        category_orders : dict, optional
            Specifies the order of categories for different dimensions.
            
            Keys should be column names, values can be either:
            
            - A list of category values in desired order
            - A string specifying sorting method:

            For non-aggregated data (agg_mode=None):
            
            - 'category ascending/descending': Alphabetical order
            - 'count ascending/descending': Sort by count of values
            - 'min ascending/descending': Sort by minimum value
            - 'max ascending/descending': Sort by maximum value
            - 'sum ascending/descending': Sort by sum of values
            - 'mean ascending/descending': Sort by mean value
            - 'median ascending/descending': Sort by median value

            For aggregated data (agg_mode='groupby' or 'resample'):
            
            - 'category ascending/descending': Alphabetical order
            - 'ascending/descending': Sort by aggregated value

            Examples:
            
                - {'city': 'count descending'}  # Sort cities by count of values
                - {'product': ['A', 'B', 'C']}  # Manual order
                - {'month': 'category ascending'}  # Alphabetical order
        
        labels : dict, optional
            A dictionary mapping column names to their display names.
        norm_by: str, optional
            The name of the column to normalize by. If set to 'all', normalization will be performed based on the sum of all values in the dataset.
        trim_top_n_x : int, optional
            Only for aggregation mode. The number of top categories x axis to include in the chart. For top using num column and agg_func.
        trim_top_n_y : int, optional
            Only for aggregation mode. The number of top categories y axis to include in the chart. For top using num column and agg_func
        trim_top_n_color : int, optional
            Only for aggregation mode. The number of top categories legend to include in the chart. For top using num column and agg_func
        trim_top_n_facet_col : int, optional
            Only for aggregation mode. The number of top categories in facet_col to include in the chart. For top using num column and agg_func
        trim_top_n_facet_row : int, optional
            Only for aggregation mode. The number of top categories in facet_row to include in the chart. For top using num column and agg_func
        trim_top_n_animation_frame : int, optional
            Only for aggregation mode. The number of top categories in animation_frame to include in the chart. For top using num column and agg_func
        trim_top_n_direction : str, optional
            Trim from bottom or from top. Default is 'top'
        trim_top_n_agg_func : str, optional
            Aggregation function for top_n_trim. Options: 'mean', 'median', 'sum', 'count', 'nunique'
            By default trim_top_n_agg_func = 'count'
        show_group_size : bool, optional
            Whether to show the group size (only for groupby mode). Default is False
        min_group_size : int, optional
            The minimum number of observations required in a category to include it in the calculation.
            Categories with fewer observations than this threshold will be excluded from the analysis.
            This ensures that the computed mean is based on a sufficiently large sample size,
            improving the reliability of the results. Default is None (no minimum size restriction).
        show_box : bool, optional
            Whether to show boxplot in subplots
        show_count : bool, optional
            Whether to show countplot in subplots
            
            - Uses column 'count_for_subplots' (default) to display group sizes. By default count_for_subplots has label Value.
            - Column name can be customized via labels parameter:
            
            labels={'count_for_subplots': 'Custom Name'}
            
            - Only available when agg_func is specified

        lower_quantile : float, optional
            The lower quantile for filtering the data. Value should be in the range [0, 1].
        upper_quantile : float, optional
            The upper quantile for filtering the data. Value should be in the range [0, 1].
        show_top_and_bottom_n : int, optional
            Whether to show only top or both top and bottom
        show_legend_title : bool, optional
            Whether to show legend title. Default is False
        observed_for_groupby : bool, optional
            This only applies if any of the groupers are Categoricals.
            If True: only show observed values for categorical groupers.
            If False: show all values for categorical groupers.
            default False
        horizontal_spacing: float
            Space between subplot columns in normalized plot coordinates. Must be a float between 0 and 1.
        plotly_kwargs
            Additional keyword arguments to pass to the Plotly Express function. Default is None

        Returns
        -------
        CustomFigure
            Interactive Plotly figure object
        """
        # Merge default and method-specific settings
        params = {k: v for k, v in locals().items() if k not in ['self', 'builder', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            category_orders=params.pop('category_orders'),
            labels=params.pop('labels'),
            **params.pop('plotly_kwargs')
        )
        builder = BarLineAreaBuilder('bar')
        params['data_frame'] = self._df
        # Update params with merged settings
        params['plotly_kwargs'] = merged_plotly_kwargs
        return builder.build(**params)

    def line(
        self,
        x: Optional[Union[str, List, pd.Series]] = None,
        y: Optional[Union[str, List, pd.Series]] = None,
        color: Optional[Union[str, List, pd.Series]] = None,
        category_orders: Optional[Dict[str, Union[str, List[str]]]] = None,
        labels: Optional[Dict[str, str]] = None,
        agg_func: Optional[Literal['mean', 'median', 'sum', 'count', 'nunique']] = None,
        freq: Optional[str] = None,
        agg_column: Optional[str] = None,
        norm_by: Optional[Union[str, Literal['all']]] = None,
        trim_top_n_x: Optional[int] = None,
        trim_top_n_y: Optional[int] = None,
        trim_top_n_color: Optional[int] = None,
        trim_top_n_facet_col: Optional[int] = None,
        trim_top_n_facet_row: Optional[int] = None,
        trim_top_n_animation_frame: Optional[int] = None,
        trim_top_n_direction: Literal['top', 'bottom'] = 'top',
        trim_top_n_agg_func: Literal['mean', 'median', 'sum', 'count', 'nunique'] = 'count',
        show_group_size: bool = True,
        min_group_size: Optional[int] = None,
        show_legend_title: bool = False,
        observed_for_groupby: bool = True,
        **plotly_kwargs
    ) -> CustomFigure:
        """
        Creates a line chart using the Plotly Express library. This function is a wrapper around Plotly Express line and accepts all the same parameters, allowing for additional customization and functionality.

        Parameters:
        -----------
        x : str, optional
            Column to use for the x-axis
        y : str, optional
            Column to use for the y-axis
        color : str, optional
            Column to use for color encoding
        agg_func : str, optional
            Aggregation function. Options: 'mean', 'median', 'sum', 'count', 'nunique'
        agg_column : str, optional
            Column to aggregate
        freq : str, optional
            Resample frequency for resample. Options: 'ME', 'W', D' and others
        category_orders : dict, optional
            Specifies the order of categories for different dimensions.
            
            Keys should be column names, values can be either:
            
            - A list of category values in desired order
            - A string specifying sorting method:

            For non-aggregated data (agg_mode=None):
            
            - 'category ascending/descending': Alphabetical order
            - 'count ascending/descending': Sort by count of values
            - 'min ascending/descending': Sort by minimum value
            - 'max ascending/descending': Sort by maximum value
            - 'sum ascending/descending': Sort by sum of values
            - 'mean ascending/descending': Sort by mean value
            - 'median ascending/descending': Sort by median value

            For aggregated data (agg_mode='groupby' or 'resample'):
            
            - 'category ascending/descending': Alphabetical order
            - 'ascending/descending': Sort by aggregated value

            Examples:
            
                - {'city': 'count descending'}  # Sort cities by count of values
                - {'product': ['A', 'B', 'C']}  # Manual order
                - {'month': 'category ascending'}  # Alphabetical order
                
        labels : dict, optional
            A dictionary mapping column names to their display names.
        norm_by: str, optional
            The name of the column to normalize by. If set to 'all', normalization will be performed based on the sum of all values in the dataset.
        trim_top_n_x : int, optional
            Only for aggregation mode. The number of top categories x axis to include in the chart. For top using num column and agg_func.
        trim_top_n_y : int, optional
            Only for aggregation mode. The number of top categories y axis to include in the chart. For top using num column and agg_func
        trim_top_n_color : int, optional
            Only for aggregation mode. The number of top categories legend to include in the chart. For top using num column and agg_func
        trim_top_n_facet_col : int, optional
            Only for aggregation mode. The number of top categories in facet_col to include in the chart. For top using num column and agg_func
        trim_top_n_facet_row : int, optional
            Only for aggregation mode. The number of top categories in facet_row to include in the chart. For top using num column and agg_func
        trim_top_n_animation_frame : int, optional
            Only for aggregation mode. The number of top categories in animation_frame to include in the chart. For top using num column and agg_func
        trim_top_n_direction : str, optional
            Trim from bottom or from top. Default is 'top'
        trim_top_n_agg_func : str, optional
            Aggregation function for top_n_trim. Options: 'mean', 'median', 'sum', 'count', 'nunique'
            By default trim_top_n_agg_func = 'count'
        show_group_size : bool, optional
            Whether to show the group size (only for groupby mode). Default is False
        min_group_size : int, optional
            The minimum number of observations required in a category to include it in the calculation.
            Categories with fewer observations than this threshold will be excluded from the analysis.
            This ensures that the computed mean is based on a sufficiently large sample size,
            improving the reliability of the results. Default is None (no minimum size restriction).
        show_legend_title : bool, optional
            Whether to show legend title. Default is False
        observed_for_groupby : bool, optional
            This only applies if any of the groupers are Categoricals.
            If True: only show observed values for categorical groupers.
            If False: show all values for categorical groupers.
            default False
        plotly_kwargs
            Additional keyword arguments to pass to the Plotly Express function. Default is None
            
        Returns
        -------
        CustomFigure
            Interactive Plotly figure object       
        """
        # Merge default and method-specific settings
        params = {k: v for k, v in locals().items() if k not in ['self', 'builder', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            category_orders=params.pop('category_orders'),
            labels=params.pop('labels'),
            **params.pop('plotly_kwargs')
        )
        builder = BarLineAreaBuilder('line')
        params['data_frame'] = self._df
        # Update params with merged settings
        params['plotly_kwargs'] = merged_plotly_kwargs
        return builder.build(**params)

    def area(
        self,
        x: Optional[Union[str, List, pd.Series]] = None,
        y: Optional[Union[str, List, pd.Series]] = None,
        color: Optional[Union[str, List, pd.Series]] = None,
        category_orders: Optional[Dict[str, Union[str, List[str]]]] = None,
        labels: Optional[Dict[str, str]] = None,
        agg_func: Optional[Literal['mean', 'median', 'sum', 'count', 'nunique']] = None,
        freq: Optional[str] = None,
        agg_column: Optional[str] = None,
        norm_by: Optional[Union[str, Literal['all']]] = None,
        trim_top_n_x: Optional[int] = None,
        trim_top_n_y: Optional[int] = None,
        trim_top_n_color: Optional[int] = None,
        trim_top_n_facet_col: Optional[int] = None,
        trim_top_n_facet_row: Optional[int] = None,
        trim_top_n_animation_frame: Optional[int] = None,
        trim_top_n_direction: Literal['top', 'bottom'] = 'top',
        trim_top_n_agg_func: Literal['mean', 'median', 'sum', 'count', 'nunique'] = 'count',
        show_group_size: bool = True,
        min_group_size: Optional[int] = None,
        show_legend_title: bool = False,
        observed_for_groupby: bool = True,
        **plotly_kwargs
    ) -> CustomFigure:
        """
        Creates an area chart using the Plotly Express library. This function is a wrapper around Plotly Express area and accepts all the same parameters, allowing for additional customization and functionality.

        Parameters:
        -----------
        x : str, optional
            Column to use for the x-axis
        y : str, optional
            Column to use for the y-axis
        color : str, optional
            Column to use for color encoding
        agg_func : str, optional
            Aggregation function. Options: 'mean', 'median', 'sum', 'count', 'nunique'
        agg_column : str, optional
            Column to aggregate
        freq : str, optional
            Resample frequency for resample. Options: 'ME', 'W', D' and others
        category_orders : dict, optional
            Specifies the order of categories for different dimensions.
            
            Keys should be column names, values can be either:
            
            - A list of category values in desired order
            - A string specifying sorting method:

            For non-aggregated data (agg_mode=None):
            
            - 'category ascending/descending': Alphabetical order
            - 'count ascending/descending': Sort by count of values
            - 'min ascending/descending': Sort by minimum value
            - 'max ascending/descending': Sort by maximum value
            - 'sum ascending/descending': Sort by sum of values
            - 'mean ascending/descending': Sort by mean value
            - 'median ascending/descending': Sort by median value

            For aggregated data (agg_mode='groupby' or 'resample'):
            
            - 'category ascending/descending': Alphabetical order
            - 'ascending/descending': Sort by aggregated value

            Examples:
                - {'city': 'count descending'}  # Sort cities by count of values
                - {'product': ['A', 'B', 'C']}  # Manual order
                - {'month': 'category ascending'}  # Alphabetical order
                
        labels : dict, optional
            A dictionary mapping column names to their display names.
        norm_by: str, optional
            The name of the column to normalize by. If set to 'all', normalization will be performed based on the sum of all values in the dataset.
        trim_top_n_x : int, optional
            Only for aggregation mode. The number of top categories x axis to include in the chart. For top using num column and agg_func.
        trim_top_n_y : int, optional
            Only for aggregation mode. The number of top categories y axis to include in the chart. For top using num column and agg_func
        trim_top_n_color : int, optional
            Only for aggregation mode. The number of top categories legend to include in the chart. For top using num column and agg_func
        trim_top_n_facet_col : int, optional
            Only for aggregation mode. The number of top categories in facet_col to include in the chart. For top using num column and agg_func
        trim_top_n_facet_row : int, optional
            Only for aggregation mode. The number of top categories in facet_row to include in the chart. For top using num column and agg_func
        trim_top_n_animation_frame : int, optional
            Only for aggregation mode. The number of top categories in animation_frame to include in the chart. For top using num column and agg_func
        trim_top_n_direction : str, optional
            Trim from bottom or from top. Default is 'top'
        trim_top_n_agg_func : str, optional
            Aggregation function for top_n_trim. Options: 'mean', 'median', 'sum', 'count', 'nunique'
            By default trim_top_n_agg_func = 'count'
        show_group_size : bool, optional
            Whether to show the group size (only for groupby mode). Default is False
        min_group_size : int, optional
            The minimum number of observations required in a category to include it in the calculation.
            Categories with fewer observations than this threshold will be excluded from the analysis.
            This ensures that the computed mean is based on a sufficiently large sample size,
            improving the reliability of the results. Default is None (no minimum size restriction).
        show_legend_title : bool, optional
            Whether to show legend title. Default is False
        observed_for_groupby : bool, optional
            This only applies if any of the groupers are Categoricals.
            If True: only show observed values for categorical groupers.
            If False: show all values for categorical groupers.
            default False
        plotly_kwargs
            Additional keyword arguments to pass to the Plotly Express function. Default is None

        Returns
        -------
        CustomFigure
            Interactive Plotly figure object         
        """
        # Merge default and method-specific settings
        params = {k: v for k, v in locals().items() if k not in ['self', 'builder', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            category_orders=params.pop('category_orders'),
            labels=params.pop('labels'),
            **params.pop('plotly_kwargs')
        )
        builder = BarLineAreaBuilder('area')
        params['data_frame'] = self._df
        # Update params with merged settings
        params['plotly_kwargs'] = merged_plotly_kwargs
        return builder.build(**params)

    def box(
        self,
        x: str= None,
        y: str = None,
        color: Optional[str] = None,
        lower_quantile: Optional[Union[float, int]] = None,
        upper_quantile: Optional[Union[float, int]] = None,
        category_orders: Optional[Dict[str, Union[str, List[str]]]] = None,
        labels: Optional[Dict[str, str]] = None,
        show_dual: bool = False,
        trim_top_n_x: Optional[int] = None,
        trim_top_n_y: Optional[int] = None,
        trim_top_n_color: Optional[int] = None,
        trim_top_n_facet_col: Optional[int] = None,
        trim_top_n_facet_row: Optional[int] = None,
        trim_top_n_animation_frame: Optional[int] = None,
        trim_top_n_direction: Literal['top', 'bottom'] = 'top',
        trim_top_n_agg_func: Literal['mean', 'median', 'sum', 'count', 'nunique'] = 'count',
        mode: Literal['base', 'time_series'] = 'base',
        freq: Optional[str] = None,
        show_legend_title: bool = False,
        **plotly_kwargs
    ) -> CustomFigure:
        """
        Creates a box plot using Plotly Express.  This function is a wrapper around Plotly Express box and accepts all the same parameters, allowing for additional customization and

        Parameters:
        -----------
        x : str
            Column name in `data_frame` used for positioning marks along the x-axis.
        y : str
            Column name in `data_frame` used for positioning marks along the y-axis.
        color : str
            Column name in `data_frame` used to assign colors to marks.
        lower_quantile : float, optional
            The lower quantile for filtering the data. Value should be in the range [0, 1].

        upper_quantile : float, optional
            The upper quantile for filtering the data. Value should be in the range [0, 1].

        category_orders : dict, optional
            Specifies the order of categories for different dimensions.
            
            Keys should be column names, values can be either:
            
            - A list of category values in desired order
            - A string specifying sorting method:

            Can be one of:
            
            - 'category ascending/descending': Alphabetical order
            - 'count ascending/descending': Sort by count of values
            - 'min ascending/descending': Sort by minimum value
            - 'max ascending/descending': Sort by maximum value
            - 'sum ascending/descending': Sort by sum of values
            - 'mean ascending/descending': Sort by mean value
            - 'median ascending/descending': Sort by median value


            Examples:
                - {'city': 'count descending'}  # Sort cities by count of values
                - {'product': ['A', 'B', 'C']}  # Manual order
                - {'month': 'category ascending'}  # Alphabetical order

        category_orders : dict, optional
            A dictionary specifying the order of categories for the x-axis.

        labels : dict, optional
            A dictionary mapping column names to their display names.

        title : str, optional
            The title of the plot.

        template : str, optional
            The template to use for the plot.
        show_dual: bool, optional
            Whether to show 2 boxplots, left origin, right trimmed by quantile
        trim_top_n_x : int, optional
            Only for aggregation mode. The number of top categories x axis to include in the chart.
        trim_top_n_y : int, optional
            Only for aggregation mode. The number of top categories y axis to include in the chart.
        trim_top_n_color : int, optional
            Only for aggregation mode. The number of top categories legend to include in the chart.
        trim_top_n_facet_col : int, optional
            Only for aggregation mode. The number of top categories in facet_col to include in the chart.
        trim_top_n_facet_row : int, optional
            Only for aggregation mode. The number of top categories in facet_row to include in the chart.
        trim_top_n_animation_frame : int, optional
            Only for aggregation mode. The number of top categories in animation_frame to include in the chart.
        trim_top_n_direction : str, optional
            Trim from bottom or from top. Default is 'top'
        trim_top_n_agg_func : str, optional
            Aggregation function for top_n_trim. Options: 'mean', 'median', 'sum', 'count', 'nunique'
            By default trim_top_n_agg_func = 'count'
        mode : str, optional
            The mode of the box plot construction. Available options are:   
            
                - 'base': Standard box plot mode. This is the default behavior, where the box plot is created
                
            directly from the provided data without any temporal aggregation.
            
                - 'time_series': Temporal box plot mode. This mode aggregates the data based on a specified time
                
            column and granularity (e.g., months). It ensures that all time intervals are included in the
            plot, even if there is no data for some intervals. Requires the `time_column` parameter to be
            specified.
            Default is 'base'.

        freq : str, optional
            The frequency for temporal aggregation in 'time_box' mode. This parameter is only used when
            `mode='time_box'`. It defines how the time data is grouped (e.g., by months, days, etc.).
            
            Common values include:
            
            - 'D' for days
            - 'W' for weeks
            - 'M' for months
            - 'Q' for quarters
            - 'Y' for years

        show_legend_title : bool, optional
            Whether to show legend title. Default is False
        plotly_kwargs
            Additional keyword arguments to pass to the Plotly Express function. Default is None

        Returns
        -------
        CustomFigure
            Interactive Plotly figure object         
        """
        # Merge default and method-specific settings
        params = {k: v for k, v in locals().items() if k not in ['self', 'builder', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            category_orders=params.pop('category_orders'),
            labels=params.pop('labels'),
            **params.pop('plotly_kwargs')
        )
        builder = DistributionPlotBuilder('box')
        params['data_frame'] = self._df
        # Update params with merged settings
        params['plotly_kwargs'] = merged_plotly_kwargs
        return builder.build(**params)

    def violin(
        self,
        x: str = None,
        y: str = None,
        color: str = None,
        lower_quantile: Optional[Union[float, int]] = None,
        upper_quantile: Optional[Union[float, int]] = None,
        category_orders: Optional[Dict[str, Union[str, List[str]]]] = None,
        labels: Optional[Dict[str, str]] = None,
        show_dual: bool = False,
        trim_top_n_x: Optional[int] = None,
        trim_top_n_y: Optional[int] = None,
        trim_top_n_color: Optional[int] = None,
        trim_top_n_facet_col: Optional[int] = None,
        trim_top_n_facet_row: Optional[int] = None,
        trim_top_n_animation_frame: Optional[int] = None,
        trim_top_n_direction: Literal['top', 'bottom'] = 'top',
        trim_top_n_agg_func: Literal['mean', 'median', 'sum', 'count', 'nunique'] = 'count',
        mode: Literal['base', 'time_series'] = 'base',
        freq: Optional[str] = None,
        show_legend_title: bool = False,
        **plotly_kwargs
    ) -> CustomFigure:
        """
        Creates a violin plot using Plotly Express.  This function is a wrapper around Plotly Express violin and accepts all the same parameters, allowing for additional customization and

        Parameters:
        -----------
        x : str
            Column name in `data_frame` used for positioning marks along the x-axis.
        y : str
            Column name in `data_frame` used for positioning marks along the y-axis.
        color : str
            Column name in `data_frame` used to assign colors to marks.
        lower_quantile : float, optional
            The lower quantile for filtering the data. Value should be in the range [0, 1].

        upper_quantile : float, optional
            The upper quantile for filtering the data. Value should be in the range [0, 1].

        category_orders : dict, optional
            Specifies the order of categories for different dimensions.
            
            Keys should be column names, values can be either:
            
            - A list of category values in desired order
            - A string specifying sorting method:

            Can be one of:
            
            - 'category ascending/descending': Alphabetical order
            - 'count ascending/descending': Sort by count of values
            - 'min ascending/descending': Sort by minimum value
            - 'max ascending/descending': Sort by maximum value
            - 'sum ascending/descending': Sort by sum of values
            - 'mean ascending/descending': Sort by mean value
            - 'median ascending/descending': Sort by median value


            Examples:
                - {'city': 'count descending'}  # Sort cities by count of values
                - {'product': ['A', 'B', 'C']}  # Manual order
                - {'month': 'category ascending'}  # Alphabetical order

        labels : dict, optional
            A dictionary mapping column names to their display names.
        show_dual: bool, optional
            Whether to show 2 boxplots, left origin, right trimmed by quantile
        trim_top_n_x : int, optional
            Only for aggregation mode. The number of top categories x axis to include in the chart.
        trim_top_n_y : int, optional
            Only for aggregation mode. The number of top categories y axis to include in the chart.
        trim_top_n_color : int, optional
            Only for aggregation mode. The number of top categories legend to include in the chart.
        trim_top_n_facet_col : int, optional
            Only for aggregation mode. The number of top categories in facet_col to include in the chart.
        trim_top_n_facet_row : int, optional
            Only for aggregation mode. The number of top categories in facet_row to include in the chart.
        trim_top_n_animation_frame : int, optional
            Only for aggregation mode. The number of top categories in animation_frame to include in the chart.
        trim_top_n_direction : str, optional
            Trim from bottom or from top. Default is 'top'
        trim_top_n_agg_func : str, optional
            Aggregation function for top_n_trim. Options: 'mean', 'median', 'sum', 'count', 'nunique'
            By default trim_top_n_agg_func = 'count'
        mode : str, optional
            The mode of the box plot construction. Available options are:
            
                - 'base': Standard box plot mode. This is the default behavior, where the box plot is created
                
            directly from the provided data without any temporal aggregation.
            
                - 'time_series': Temporal box plot mode. This mode aggregates the data based on a specified time
                
            column and granularity (e.g., months). It ensures that all time intervals are included in the
            plot, even if there is no data for some intervals. Requires the `time_column` parameter to be
            specified.
            Default is 'base'.

        freq : str, optional
            The frequency for temporal aggregation in 'time_box' mode. This parameter is only used when
            `mode='time_box'`. It defines how the time data is grouped (e.g., by months, days, etc.).
            
            Common values include:
            
            - 'D' for days
            - 'W' for weeks
            - 'M' for months
            - 'Q' for quarters
            - 'Y' for years

        show_legend_title : bool, optional
            Whether to show legend title. Default is False
        plotly_kwargs
            Additional keyword arguments to pass to the Plotly Express function. Default is None
            
        Returns
        -------
        CustomFigure
            Interactive Plotly figure object             
        """
        # Merge default and method-specific settings
        params = {k: v for k, v in locals().items() if k not in ['self', 'builder', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            category_orders=params.pop('category_orders'),
            labels=params.pop('labels'),
            **params.pop('plotly_kwargs')
        )
        builder = DistributionPlotBuilder('violin')
        params['data_frame'] = self._df
        # Update params with merged settings
        params['plotly_kwargs'] = merged_plotly_kwargs
        return builder.build(**params)

    def heatmap(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        z: Optional[str] = None,
        do_pivot: bool = False,
        agg_func: Optional[str] = None,
        hide_first_column: bool = False,
        trim_top_n_x: Optional[int] = None,
        trim_top_n_y: Optional[int] = None,
        trim_top_n_direction: Literal['top', 'bottom'] = 'top',
        trim_top_n_agg_func: Literal['mean', 'median', 'sum', 'count', 'nunique'] = 'count',
        category_orders: Optional[Dict[str, Union[str, List[str]]]] = None,
        labels: Optional[Dict[str, str]] = None,
        fill_value: Optional[Union[int, float]] = None,
        **plotly_kwargs
    ) -> CustomFigure:
        """
        Creates an enhanced heatmap visualization with support for data aggregation,
        filtering, and advanced customization.

        Parameters:
        -----------
        x : str, optional
            Column name for x-axis (required for pivot mode)
        y : str, optional
            Column name for y-axis (required for pivot mode)
        z : str, optional
            Column name for values (required for pivot mode)
        do_pivot : bool, default False
            Whether to create a pivot table before visualization
        agg_func : str, optional
            Aggregation function for pivot mode. Options:
            'mean', 'median', 'sum', 'count', 'nunique', 'min', 'max'
        hide_first_column : bool, default False
            Whether to skip first column in cohort analysis or other cases
        trim_top_n_x : int, optional
            Only for aggregation mode. The number of top categories x axis to include in the chart. For top using num column and agg_func.
        trim_top_n_y : int, optional
            Only for aggregation mode. The number of top categories y axis to include in the chart. For top using num column and agg_func
        trim_top_n_direction : str, optional
            Trim from bottom or from top. Default is 'top'
        trim_top_n_agg_func : str, default 'count'
            Aggregation function for top-N filtering
        category_orders : dict, optional
            Specifies order of categories on axes. Keys can be 'x' or 'y',
            
            values can be:
            
            - List of categories in desired order
            - String specifying sorting method:
            
                * 'category ascending/descending' - alphabetical
                * 'count ascending/descending' - by value count
                * 'sum/mean/median/min/max ascending/descending' - by aggregated value

        labels : dict with str keys and str values (default {})
            Sets names used in the figure for axis titles (keys x and y), colorbar title and hoverlabel (key color). The values should correspond to the desired label to be displayed. If img is an xarray, dimension names are used for axis titles, and long name for the colorbar title
            (unless overridden in labels). Possible keys are: x, y, and color.
        fill_value : int, float, default None
            fill_value for pd.pivot_table. Value to replace missing values with (in the resulting pivot table, after aggregation).
        plotly_kwargs
            Additional keyword arguments to pass to the Plotly Express function. Default is None

        Returns
        -------
        CustomFigure
            Interactive Plotly figure object    
        """
        # Merge default and method-specific settings
        params = {k: v for k, v in locals().items() if k not in ['self', 'builder', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            category_orders=params.pop('category_orders'),
            labels=params.pop('labels'),
            **params.pop('plotly_kwargs')
        )
        builder = HeatmapBuilder()
        params['data_frame'] = self._df
        # Update params with merged settings
        params['plotly_kwargs'] = merged_plotly_kwargs
        return builder.build(**params)

    def pairplot(
        self,
        pairs: Union[List[str], List[Tuple[str, str]]] = None,
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        color_mode: Optional[Literal['count', 'kde', 'category']] = None,
        color_column: Optional[str] = None,
        display_mode: str = 'scatter',
        correlation_method: str = 'pearson',
        transforms: Optional[Union[str, Dict[str, str]]] = None,
        show_correlation: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
        category_orders: Optional[Dict[str, List[str]]] = None,
        labels: Optional[Dict[str, str]] = None,
        bins: int = 20,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        horizontal_spacing: float = 0.07,
        vertical_spacing: float = 0.11,
        generator_mode: bool = False,
        plots_per_page: Optional[int] = None,
        trendline: Optional[str] = None,
        plot_bgcolor: Optional[str] = None,
        title: Optional[str] = None,
        show_legend_title: bool = False,
        color_continuous_scale: Union[str, List[str]] = 'viridis',
        color_discrete_sequence: Optional[List[str]] = None,
        renderer: str = 'jpg'
    ) -> Union[Union[None, CustomFigure], Union[Generator[CustomFigure, None, None], None]]:
        """
        Create an advanced pairplot of numerical variables with multiple display options.

        Parameters:
        -----------
        pairs: Union[List[str], List[Tuple[str, str]], optional
            Specifies which column pairs to plot. Can be one of:
            
            - None: plots all pairwise combinations of numeric columns
            - List of column names (all pairwise combinations will be plotted)
            
              Example: ['col1', 'col2', 'col3']
              
            - List of explicit pairs to plot
            
              Example: [('col1', 'col2'), ('col1', 'col3')]

        ranges : Dict[str, Tuple[Optional[float], Optional[float]]], optional
            Dictionary specifying value ranges for columns. Keys are column names,
            values are tuples (min, max) where either can be None for no bound.
            Example: {'col1': (0, 100), 'col2': (None, 50)}

        color_mode : Literal['count', 'kde', 'category'], optional
            Coloring mode for points:
            
            - 'count' - color by point density (bin counting)
            - 'kde' - color by kernel density estimation
            - 'category' - color by values in color_column

        color_column : str, optional
            Column name for categorical coloring (used only when color_mode='category')
        display_mode : str, optional
            Visualization mode: 'scatter' for points or 'density_contour' for contours
        correlation_method : str, optional
            Correlation method: 'pearson', 'spearman', or 'kendall'
        transforms : str or dict, optional
            If string: apply same transform to all numeric columns (e.g. 'log')
            Dictionary specifying transformations for columns. Keys are column names,
            values are transformation names: 'log', 'boxcox', 'yeojohnson', 'sqrt', 'reciprocal'
            Example: {'price': 'log', 'area': 'sqrt'}
        show_correlation : bool, optional
            Whether to display correlation coefficient and p-value on each plot.
        category_orders : dict, optional
            Specifies the order of categories for different dimensions.
        labels : dict, optional
            A dictionary mapping column names to their display names.
        width : int, optional
            Figure width in pixels
        height : int, optional
            Figure height in pixels
        bins : int, optional
            Number of bins for density calculation
        rows : int, optional
            Number of rows in subplot grid
        cols : int, optional
            Number of columns in subplot grid
        horizontal_spacing : float, optional
            Horizontal spacing between subplots (0-1)
        vertical_spacing : float, optional
            Vertical spacing between subplots (0-1)
        generator_mode : bool, optional
            If True, returns a generator that yields figures page by page
        plots_per_page : int, optional
            Number of plots per page when using generator mode
        trendline : str, optional
            One of 'ols', 'lowess', 'rolling', 'expanding' or 'ewm'.
            If 'ols', an Ordinary Least Squares regression line will be drawn for each discrete-color/symbol group.
            If 'lowess’, a Locally Weighted Scatterplot Smoothing line will be drawn for each discrete-color/symbol group.
            If 'rolling’, a Rolling (e.g. rolling average, rolling median) line will be drawn for each discrete-color/symbol group.
            If 'expanding’, an Expanding (e.g. expanding average, expanding sum) line will be drawn for each discrete-color/symbol group.
            If 'ewm’, an Exponentially Weighted Moment (e.g. exponentially-weighted moving average) line will be drawn for each discrete-color/symbol group.
        plot_bgcolor : str, optional
            Sets the background color of the plotting area in-between x and y axes.
        show_legend_title : bool, optional
            Whether to show legend title. Default is False

        color_continuous_scale: Union[str, List[str]] = 'viridis'
            Color scale to use for continuous coloring (when color_mode is 'count' or 'kde').
            
            Can be either:
            
            - Name of a Plotly continuous color scale
                ('viridis', 'plasma', 'inferno', 'magma', 'cividis')
            - List of colors defining a custom color scale
            
            Default: 'viridis'

        color_discrete_sequence: Optional[Union[str, List[str]]] = None
            Color sequence to use for categorical coloring (when color_mode='category').
            
            Can be either:
            
            - Name of a Plotly qualitative color sequence
                px.colors.qualitative.G10
            - List of colors defining a custom discrete sequence
            
            When None, uses default Plotly qualitative palette.
        renderer : str, optional
            The Plotly renderer to use for displaying the figure. Common options include:
            
            - 'browser': Opens plot in default web browser
            - 'notebook': Renders plot in Jupyter notebook output
            - 'png': Static PNG image
            - 'svg': Static SVG image
            - 'json': Returns figure as JSON string
            
            If None, uses Plotly's default renderer.
            See Plotly documentation for full list of available renderers.

        Returns:
        --------
        Union[CustomFigure, Generator[CustomFigure, None, None]]
            Either a single Plotly figure or a generator of Plotly figures
        """
        # Merge default and method-specific settings
        params = {k: v for k, v in locals().items() if k not in ['self', 'builder', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            category_orders=params.pop('category_orders'),
            labels=params.pop('labels')
        )
        builder = PairplotBuilder()
        params['data_frame'] = self._df
        # Update params with merged settings
        params['category_orders'] = merged_plotly_kwargs.pop('category_orders')
        params['labels'] = merged_plotly_kwargs.pop('labels')
        return builder.build(**params)

    def plot_ci(
        self,
        num_col: str,
        cat_col: str,
        color: Optional[str] = None,
        alpha: float = 0.05,
        orientation: str = 'v',
        legend_position: str = 'top',
        title: str = None,
        labels: Optional[Dict[str, str]] = None,
        category_orders: Optional[Dict[str, list]] = None,
        min_group_size: int = 30,
        show_summary: bool = False,
        show_annotations: bool = False,
        annotation_format: str = ".2f",
        marker_size: int = 8,
        error_bar_width: int = 10,
        group_spacing: float = 0.3,
        show_legend_title: bool = False,
        **plotly_kwargs
    ) -> CustomFigure:
        """
        Creates a plot with mean values and confidence intervals using t-statistics.
        Points are grouped when using color to avoid overlap.

        Parameters:
        -----------
        num_col : str
            Name of numerical variable to calculate statistics
        cat_col : str
            Name of primary categorical variable for grouping
        color : str, optional
            Name of secondary categorical variable for additional grouping
        alpha : float, optional
            The significance level for interval calculation (0 < alpha < 1).
        orientation : str, optional
            Plot orientation ('v' for vertical or 'h' for horizontal). Default is 'v'
        height : int, optional
            Plot height in pixels. Default is 600
        width : int, optional
            Plot width in pixels. Default is 800
        legend_position : str, optional
            Legend position ('top', 'right', 'bottom', 'left'). Default is 'top'
        title : str, optional
            Plot title. Default is None
        labels : dict, optional
            Dictionary for custom axis labels. Default is None
        category_orders : dict, optional
            Dictionary specifying order of categories. Default is None
        min_group_size : int, optional
            Minimum group size to calculate confidence intervals. Groups smaller than this will
            generate warnings. Default is 30
        show_summary : bool, optional
            Whether to print the summary dataframe. Default is False
        show_annotations : bool, optional
            Whether to show mean value annotations on the plot. Default is False
        annotation_format : str, optional
            Format string for annotations. Default is ".2f"
        marker_size : int, optional
            Size of markers in pixels. Default is 8
        error_bar_width : int, optional
            Width of error bar caps in pixels. Default is 10
        group_spacing : float, optional
            Spacing between groups when using color (0-1). Default is 0.2
        show_legend_title : bool, optional
            Whether to show legend title. Default is False
        horizontal_spacing: float
            Space between subplot columns in normalized plot coordinates. Must be a float between 0 and 1.
        plotly_kwargs
            Additional keyword arguments to pass to the Plotly Express function. Default is None

        Returns
        -------
        CustomFigure
            Interactive Plotly figure object    
        """
        # Merge default and method-specific settings
        params = {k: v for k, v in locals().items() if k not in ['self', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            category_orders=params.pop('category_orders'),
            labels=params.pop('labels')
        )
        # Update params with merged settings
        params['category_orders'] = merged_plotly_kwargs.pop('category_orders')
        params['labels'] = merged_plotly_kwargs.pop('labels')
        return create_plot_ci(data_frame=self._df, **params)
    
    def pie_bar(
        self,
        x: str,
        y: str,
        agg_func: Literal['mean', 'median', 'sum', 'count', 'nunique'] = 'count',
        agg_column: Optional[str] = None,
        category_orders: Optional[Dict[str, Union[str, List[str]]]] = None,
        labels: Optional[Dict[str, str]] = None,
        trim_top_n_x: Optional[int] = None,
        trim_top_n_y: Optional[int] = None,
        trim_top_n_direction: Literal['top', 'bottom'] = 'top',
        trim_top_n_agg_func: Literal['mean', 'median', 'sum', 'count', 'nunique'] = 'count',
        norm_by: Optional[Union[str, Literal['all']]] = None,
        sort_by: Optional[str] = None,
        show_group_size: bool = True,
        min_group_size: Optional[int] = None,
        observed_for_groupby: bool = True,
        hole: float = None,
        label_for_others_in_pie: str = 'others',
        pie_textinfo: str = 'percent',
        agg_func_for_pie_others: str = 'sum',
        pull: Optional[int] = None,
        horizontal_spacing: Optional[float] = 0.15,
        **plotly_kwargs
    ) -> CustomFigure:
        """
        Creates a combined pie and bar chart visualization.

        The pie chart shows the proportion of the largest category versus all others combined,
        while the bar chart displays the distribution of the remaining categories. Using grouping by x or y column.

        Parameters:
        -----------
        x : str
            Column to use for the x-axis
        y : str
            Column to use for the y-axis
        agg_func : str
            Aggregation function. Options: 'mean', 'median', 'sum', 'count', 'nunique'
        agg_column : str, optional
            Column to aggregate
        category_orders : dict, optional
            Specifies the order of categories for different dimensions.
            
            Keys should be column names, values can be either:
            
            - A list of category values in desired order
            - A string specifying sorting method:

            For non-aggregated data (agg_mode=None):
            
            - 'category ascending/descending': Alphabetical order
            - 'count ascending/descending': Sort by count of values
            - 'min ascending/descending': Sort by minimum value
            - 'max ascending/descending': Sort by maximum value
            - 'sum ascending/descending': Sort by sum of values
            - 'mean ascending/descending': Sort by mean value
            - 'median ascending/descending': Sort by median value

            For aggregated data (agg_mode='groupby' or 'resample'):
            
            - 'category ascending/descending': Alphabetical order
            - 'ascending/descending': Sort by aggregated value

            Examples:
                - {'city': 'count descending'}  # Sort cities by count of values
                - {'product': ['A', 'B', 'C']}  # Manual order
                - {'month': 'category ascending'}  # Alphabetical order
                
        labels : dict, optional
            A dictionary mapping column names to their display names.
        trim_top_n_x : int, optional
            Only for aggregation mode. The number of top categories x axis to include in the chart. For top using num column and agg_func.
        trim_top_n_y : int, optional
            Only for aggregation mode. The number of top categories y axis to include in the chart. For top using num column and agg_func
        norm_by: str, optional
            The name of the column to normalize by. If set to 'all', normalization will be performed based on the sum of all values in the dataset.
        sort_by : str, optional
            Specifies which column to use for sorting when both x and y are numeric and in category_orders use string parameter for sorting
        show_group_size : bool, optional
            Whether to show the group size (only for groupby mode). Default is False
        min_group_size : int, optional
            The minimum number of observations required in a category to include it in the calculation.
            Categories with fewer observations than this threshold will be excluded from the analysis.
            This ensures that the computed mean is based on a sufficiently large sample size,
            improving the reliability of the results. Default is None (no minimum size restriction).
        hole : float, optional
            Size of pie hole. May be from 0 to 1.
        label_for_others_in_pie : str, optional
            Label for others part in pie
        pie_textinfo : str, optional
            textinfo parametr for px.pie. Options: 'value', 'percent', 'label', and their combinations
        agg_func_for_pie_others: str, optional
            function for aggregate others part in pie chart. Default is sum
        pull:float, optional
            For a "pulled-out" or "exploded" layout of the pie chart.
        horizontal_spacing: float
            Space between subplot columns in normalized plot coordinates. Must be a float between 0 and 1.
        plotly_kwargs
            Additional keyword arguments to pass to the Plotly Express function. Default is None

        Returns
        -------
        CustomFigure
            Interactive Plotly figure object       
        """
        params = {k: v for k, v in locals().items() if k not in [None, 'plotly_kwargs', 'x', 'y', 'self', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            category_orders=params.pop('category_orders'),
            labels=params.pop('labels'),
            **plotly_kwargs
        )
        update_kwargs = {
            'x': x,
            'y': y,
        }
        update_kwargs = {k: v for k,v in update_kwargs.items() if v is not None}
        merged_plotly_kwargs.update(**update_kwargs)
        return create_pie_bar(data_frame=self._df, **params, plotly_kwargs=merged_plotly_kwargs)

    def qqplot(
        self,
        x: str,
        show_skew_curt: bool = True,
        point_color: str = 'rgba(40, 115, 168, 0.9)',
        line_color: str = 'rgba(226, 85, 89, 0.9)',
        point_size: int = 8,
        line_width: int = 2,
        title: str = 'Q-Q Plot',
        width: int = 500,
        height: int = 400,
        show_grid: bool = True,
        reference_line: str = '45',
        renderer: str = 'png'
    ) -> Union[CustomFigure, None]:
        """
        Create an interactive Q-Q plot using Plotly with enhanced functionality.

        A Q-Q (quantile-quantile) plot is a probability plot that compares two distributions
        by plotting their quantiles against each other. This implementation compares the
        sample data against a theoretical normal distribution.

        Parameters:
        -----------
        x : str
            Column to use for the x-axis
        show_skew_curt : bool, optional
            Whether to display skewness and kurtosis values (default True)
        point_color : str, optional
            Color of the data points (default 'rgba(40, 115, 168, 0.9)')
        line_color : str, optional
            Color of the reference line (default 'rgba(226, 85, 89, 0.9)')
        point_size : int, optional
            Size of the data points (default 8)
        line_width : int, optional
            Width of the reference line (default 2)
        title : str, optional
            Plot title (default 'Q-Q Plot')
        width : int, optional
            Plot width in pixels (default 500)
        height : int, optional
            Plot height in pixels (default 400)
        show_grid : bool, optional
            Whether to show grid lines (default True)
        reference_line : str, optional
            Type of reference line ('s' for standardized line, '45' for 45-degree line,
            'r' for regression line, or None for no line) (default '45')
        renderer : str, optional
            The Plotly renderer to use for displaying the figure. Common options include:
            
            - 'browser': Opens plot in default web browser
            - 'notebook': Renders plot in Jupyter notebook output
            - 'png': Static PNG image
            - 'svg': Static SVG image
            - 'json': Returns figure as JSON string
            
            If None, uses Plotly's default renderer.
            See Plotly documentation for full list of available renderers.

        Returns
        -------
        CustomFigure
            Interactive Plotly figure object

        Raises:
        -------
        ValueError
            If input data contains NaN/infinite values or has length < 2
        TypeError
            If input is not a valid array-like object
        """
        if x not in self._df.columns:
            raise ValueError(f'{x} not found in DataFrame')
        params = {k: v for k, v in locals().items() if k not in ['self', 'x']}
        return create_qqplot(x = self._df[x], **params)

    def histogram(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        category_orders: Optional[Dict[str, Union[str, List[str]]]] = None,
        labels: Optional[Dict[str, str]] = None,
        lower_quantile: Optional[Union[float, int]] = None,
        upper_quantile: Optional[Union[float, int]] = None,
        mode: Literal['base', 'dual_hist_trim', 'dual_box_trim', 'dual_hist_qq'] = 'base',
        show_kde: bool = False,
        show_hist: bool = True,
        show_box: bool = True,
        legend_position: str = 'top',
        show_legend_title: bool = False,
        renderer: str = None,
        xaxis_type: str = None,
        yaxis_type: str = None,        
        **plotly_kwargs
    ) -> CustomFigure:
        """
        Creates a histogram chart using the Plotly Express library. This function is a wrapper around Plotly Express histogram and accepts all the same parameters, allowing for additional customization and functionality.

        Parameters:
        -----------
        mode : str, optional
            Specifies the type of visualization to be displayed. Available options are:
            
            - 'base': Display simple histogram
            - 'dual_hist_trim': Displays a dual plot with the original histogram on the left
                                and a trimmed histogram (based on quantiles) on the right.
            - 'dual_box_trim': Displays a dual plot with a boxplot on the left and a trimmed
                            histogram (based on quantiles) on the right.
            - 'dual_hist_qq': Displays a dual plot with the original histogram on the left
                            and a QQ-plot on the right.
                            
            Default is 'base'.

        category_orders : dict, optional
            Specifies the order of categories for different dimensions.
        
        labels : dict, optional
            A dictionary mapping column names to their display names.

        lower_quantile : float, optional
            The lower quantile for data filtering (default is 0).
        upper_quantile : float, optional
            The upper quantile for data filtering (default is 1).
        x : str, optional
            The name of the column in `data_frame` to be used for the x-axis. If not provided, the function will attempt to use the first column.
        y : str, optional
            The name of the column in `data_frame` to be used for the y-axis. If not provided, the function will count occurrences.
        color : str, optional
            The name of the column in `data_frame` to be used for color encoding. This will create a separate histogram for each unique value in this column.
        barmode : str, optional
            The mode for the bars in the histogram. Options include 'group', 'overlay', and 'relative'. Default is 'overlay'.
        nbins : int, optional
            The number of bins to use for the histogram. If not specified, the function will automatically determine the number of bins.
        histnorm : str, optional
            Normalization method for the histogram. Options include 'percent', 'probability', 'density', and 'probability density'. Default is None (no normalization).
        barnorm : str, optional
            Specifies how to normalize the heights of the bars in the histogram. Possible values include:
            
            - 'fraction': normalizes the heights of the bars so that the sum of all heights equals 1 (fraction of the total count).
            - 'percent': normalizes the heights of the bars so that the sum of all heights equals 100 (percentage of the total count).
            - 'density': normalizes the heights of the bars so that the area under the histogram equals 1 (probability density).
            - None: by default, the heights of the bars are not normalized.

        labels : dict, optional
            A dictionary mapping column names to labels for the axes and legend.
        show_legend_title : bool, optional
            Whether to show legend title. Default is False
        renderer : str, optional
            The Plotly renderer to use for displaying the figure. Common options include:
            
            - 'browser': Opens plot in default web browser
            - 'notebook': Renders plot in Jupyter notebook output
            - 'png': Static PNG image
            - 'svg': Static SVG image
            - 'json': Returns figure as JSON string
            
            If None, uses Plotly's default renderer.
            See Plotly documentation for full list of available renderers.
            
        xaxis_type : str, optional
            Type of the X-axis ['-', 'linear', 'log', 'date', 'category', 'multicategory']
        yaxis_type : str, optional
            Type of the Y-axis ['-', 'linear', 'log', 'date', 'category', 'multicategory']         
        plotly_kwargs
            Additional keyword arguments to pass to the Plotly Express function. Default is None

        Returns
        -------
        CustomFigure
            Interactive Plotly figure object           
        """
        builder = HistogramBuilder()
        params = {k: v for k, v in locals().items() if k not in ['builder', 'self', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            category_orders=params.pop('category_orders'),
            labels=params.pop('labels'),
            **params.pop('plotly_kwargs')
        )        
        params['plotly_kwargs'] = merged_plotly_kwargs
        params['data_frame'] = self._df
        return builder.build(**params)

    def cat_compare(
        self,
        cat1: str,
        cat2: str,
        trim_top_n_cat1: int = None,
        trim_top_n_cat2: int = None,
        barmode: str = 'group',
        text_auto: bool | str = False,
        labels: dict = None,
        category_orders: dict = None,
        hover_name: str = None,
        hover_data: list | dict = None,
        return_figs: bool = False,
        legend_position: str = 'top',
        fig_layouts: list = None,
        heights: list = [400, 450, 450],
        width: int = 900,
        visible_graphs: list = [1, 2, 3],
        horizontal_spacing: Optional[float] = 0.1,
        bargroupgap: Optional[float] = None,
        **plotly_kwargs
    ) -> Union[List[CustomFigure], None]:
        """
        Compare two categorical variables in a DataFrame and visualize the results.

        Creates a multi-panel visualization showing:
        1. Base distributions of categories
        2. Category1-by-Category2 analysis
        3. Category2-by-Category1 analysis

        Parameters:
        -----------
        cat1 : str
            The first categorical variable to compare (column name).
        cat2 : str
            The second categorical variable to compare (column name).
        trim_top_n_cat1 : int, optional
            Number of top categories to display for cat1 (trims others).
            If None, shows all categories.
        trim_top_n_cat2 : int, optional
            Number of top categories to display for cat2 (trims others).
            If None, shows all categories.
        barmode : str, optional
            Mode for bar chart display ('group', 'stack', 'relative', etc.).
            Default is 'group'.
        text_auto : bool or str, optional
            Whether to automatically display text on bars. Can be:
            
            - False: no text
            - True: show values
            - '.2f' etc: format string
            
            Default is False.
        labels : dict, optional
            Dictionary for custom axis labels in format {column_name: label}.
        category_orders : dict, optional
            Specifies order of categories for different dimensions.
            
            Keys should be column names, values can be:
            
            - List of category values in desired order
            - String specifying sorting method:
            
                * For non-aggregated data:
                
                    - 'category ascending/descending' (alphabetical)
                    - 'count ascending/descending' (by value count)
                    - 'min/max/sum/mean/median ascending/descending'
                    
                * For aggregated data:
                
                    - 'ascending/descending' (by aggregated value)
                    - 'category ascending/descending' (alphabetical)
                    
        hover_name : str, optional
            Column name to use as hover title.
        hover_data : list or dict, optional
            Additional columns to show in hover tooltip.
        return_figs : bool, optional
            If True, returns list of figures instead of displaying them.
            Default is False.
        legend_position : str, optional
            Position of legend ('top', 'right', 'bottom', 'left').
            Default is 'top'.
        fig_layouts : list, optional
            List of layout dictionaries for each subplot customization.
        heights : list, optional
            Heights for each subplot in pixels.
            Default is [400, 450, 450].
        width : int, optional
            Total figure width in pixels.
            Default is 900.
        visible_graphs : list, optional
            Which comparison panels to display (1, 2, and/or 3):
            
            1 - Base distributions (cat1 and cat2 normalized to total count)
            2 - cat1 distribution by cat2 groups (row/column normalized)
            3 - cat2 distribution by cat1 groups (row/column normalized)
            
            Default is [1, 2, 3] (show all).
        horizontal_spacing : float, optional
            Space between subplots (ratio of width).
            Default is 0.1.
        bargroupgap : float, optional
            Space between groups of bars (ratio of bar width).
        plotly_kwargs
            Additional arguments passed to Plotly figure.update_layout().

        Returns:
        --------
        list of CustomFigure or None
            If return_figs is True, returns list of Plotly figure objects.
            Otherwise displays Plotly figures and returns None.       
        """
        params = {k: v for k, v in locals().items() if k not in ['builder', 'self', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            category_orders=params.pop('category_orders'),
            labels=params.pop('labels')
        )
        params['category_orders'] = merged_plotly_kwargs.pop('category_orders')
        params['labels'] = merged_plotly_kwargs.pop('labels')
        data_frame = self._df
        cat1 = params.pop('cat1')
        cat2 = params.pop('cat2')
        builder = CatCompareBuilder(data_frame, cat1, cat2)
        return builder.build(**params)
    
    def wordcloud(
        self,
        text_column: str,
        title: Optional[str] = None,
        max_words: int = 100,
        width: int = 800,
        height: int = 400,
        background_color: str = 'white',
        colormap: Union[str, Colormap] = 'viridis',
        margin: Optional[Union[int, Dict[str, int]]] = None,
        relative_scaling: float = 0.5,
        prefer_horizontal: float = 0.9,
        contour_width: int = 0,
        contour_color: str = 'black',
        random_state: Optional[int] = None,
        mask: Optional[np.ndarray] = None,
        stopwords: Optional[List[str]] = None,
        collocations: bool = True,
        normalize_plurals: bool = True,
        include_numbers: bool = False,
        min_word_length: int = 0,
        repeat: bool = False,
        scale: float = 1,
        min_font_size: int = 4,
        max_font_size: Optional[int] = None,
        font_path: Optional[str] = None,
        color_func: Optional[Callable] = None,
        regexp: Optional[str] = None,
        collocation_threshold: int = 30,
        return_fig: bool = False,
        scroll_zoom: bool = False,
    ) -> Union[go.Figure, None]:
        """
        Generate an interactive word cloud visualization using Plotly.

        Parameters:
        -----------
        text_column : str
            Column with text to generate the word cloud from.
        title : Optional[str]
            Title for the plot. If None, no title will be displayed.
        max_words : int
            Maximum number of words to include in the word cloud.
        width : int
            Width of the output image in pixels.
        height : int
            Height of the output image in pixels.
        background_color : str
            Background color for the word cloud.
        colormap : Union[str, Colormap]
            Matplotlib colormap name or object for word coloring.
        margin : Optional[Union[int, Dict[str, int]]]
            Margin around the word cloud. Can be an integer (applied to all sides)
            or a dict with keys 'l', 'r', 'b', 't' for specific margins.
        relative_scaling : float
            Importance of relative word frequencies (0-1).
        prefer_horizontal : float
            Preference for horizontal words (0-1).
        contour_width : int
            Width of word cloud contour.
        contour_color : str
            Color of the contour.
        random_state : Optional[int]
            Seed for random layout.
        mask : Optional[np.ndarray]
            Numpy array defining custom shape for word cloud.
        stopwords : Optional[List[str]]
            List of words to exclude.
        collocations : bool
            Whether to include collocations (bigrams).
        normalize_plurals : bool
            Whether to normalize plurals.
        include_numbers : bool
            Whether to include numbers.
        min_word_length : int
            Minimum word length to include.
        repeat : bool
            Whether to repeat words to fill space.
        scale : float
            Scaling between computation and drawing.
        min_font_size : int
            Smallest font size to use.
        max_font_size : Optional[int]
            Largest font size to use (None for automatic).
        font_path : Optional[str]
            Path to custom font file.
        color_func : Optional[callable]
            Custom function for word coloring.
        regexp : Optional[str]
            Regular expression for tokenization.
        collocation_threshold : int
            Score threshold for collocations.
        fig_kwargs : Optional[Dict[str, Any]]
            Additional arguments to pass to go.Figure.
        scroll_zoom: bool
            Only with return_fig = False. This option allows users to zoom in and out of figures using the scroll wheel on their mouse and/or a two-finger scroll.
        return_fig: bool
            Wheather to return plotly figure

        Returns:
        --------
        go.Figure
            Plotly Figure object containing the word cloud visualization.     
        """
        if text_column not in self._df.columns:
            raise ValueError(f'{text_column} not found in DataFrame')
        params = {k: v for k, v in locals().items() if k not in ['text_column', 'self']}
        params['text'] = ' '.join(self._df[text_column].dropna().astype(str))

        return create_wordcloud_plotly(**params)
    
    def parallel_categories(
        self,
        dimensions: list,
        color_mapping: dict = None,
        top_n_categories: dict = None,
        title: str = None,
        labels: dict = None,
        margin_l: int = 50,
        margin_r: int = 50,
        width: int = None,
        height: int = None,
        **plotly_kwargs
    ) -> go.Figure:
        """
        Create a parallel categories plot with customizable colors and category filtering.
        
        Parameters:
        -----------
        dimensions : list
            List of column names to use as dimensions in the plot
        color_mapping : dict, optional
            Dictionary mapping category names to colors for the first dimension
            Format: {category_name: hex_color}
        top_n_categories : dict, optional
            Dictionary specifying how many top categories to show for each dimension
            Format: {dimension_name: n_categories}
        title : str, optional
            Plot title
        labels : dict, optional
            Dictionary for axis labels (key: dimension name, value: display name)
        margin_l : int, optional
            Left margin in pixels. Controls space on left side of plot.
            If None, uses Plotly's default margin.
        margin_r : int, optional
            Right margin in pixels. Controls space on right of plot.
            If None, uses Plotly's default margin.    
        height int, optional 
            Height of each figure.
        width int, optional
            Width of each figure.        
        plotly_kwargs
            Additional arguments passed to px.parallel_categories()
        
        Returns:
        --------
        go.Figure
            Interactive Plotly figure object     
        """

        params = {k: v for k, v in locals().items() if k not in [None,'self', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            labels=params.pop('labels'),
            **params.pop('plotly_kwargs')
        )
        params['data_frame'] = self._df
        params['plotly_kwargs'] = merged_plotly_kwargs
        return parallel_categories(**params)
    
    def period_change(
        self,
        metric_col: str,
        date_col: str,
        period: str = 'mom',
        agg_func: Callable = 'sum',
        facet_col: Optional[str] = None,
        facet_col_wrap: Optional[int] = None,
        facet_row: Optional[str] = None,
        animation_frame: Optional[str] = None,
        fill_value: Union[int, float] = 0,
        color_dict: Dict[str, str] = None,
        labels: Optional[Dict[str, str]] = None,
        category_orders: Optional[Dict[str, List]] = None,
        **plotly_kwargs
    ) -> CustomFigure:
        """
        Plot period-over-period changes for a given metric using pd.Grouper with enhanced customization.
        
        Parameters:
        -----------
        metric_col : str
            Name of the metric column to analyze
        date_col : str
            Name of the datetime column
        period : str, optional
            Period for change calculation: 
            
                - 'mom' - month over month, 
                - 'wow' - week over week,
                - 'dod' - day over day,
                - 'yoy' - year over year (default: 'mom')
                
        agg_func : Callable, optional
            Aggregation function (default: 'sum')
        facet_col : str, optional
            Column name for faceting by columns
        facet_col_wrap : int, optional
            Number of facet columns before wrapping
        facet_row : str, optional
            Column name for faceting by rows
        animation_frame : str, optional
            Column name for animation frames
        fill_value : Union[int, float], optional
            Value to use for filling missing dates (default: 0)
        color_dict : Dict[str, str], optional
            Dictionary mapping positive/negative values to colors
            Format: {'positive': 'color1', 'negative': 'color2'}
        labels : Dict[str, str], optional
            Dictionary for relabeling axes and other elements
        category_orders : Dict[str, List], optional
            Custom ordering of categories
        plotly_kwargs : dict
            Additional keyword arguments passed to px.bar()
            
        Returns
        -------
        CustomFigure
            Interactive Plotly figure object                    
        """    
        # Merge default and method-specific settings
        params = {k: v for k, v in locals().items() if k not in ['self', 'merged_plotly_kwargs']}
        merged_plotly_kwargs = self._merge_plotly_settings(
            category_orders=params.pop('category_orders'),
            labels=params.pop('labels'),
            **params.pop('plotly_kwargs')
        )
        # Update params with merged settings
        params['plotly_kwargs'] = merged_plotly_kwargs      
        params['df'] = self._df
        return period_change(**params)          