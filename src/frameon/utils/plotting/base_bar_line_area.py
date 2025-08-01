from typing import Union, Optional, Literal, Dict, Callable, List, Any, get_type_hints
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from frameon.utils.miscellaneous import display, is_datetime_column
from .custom_figure import CustomFigure
from dataclasses import dataclass, fields
from frameon.utils.miscellaneous import is_categorical_column
from datetime import timedelta
from pandas.tseries.offsets import DateOffset
import warnings

FIGURE_CREATORS = {
    'bar': px.bar,
    'line': px.line,
    'area': px.area
}

colorway_for_line = [
    'rgb(127, 60, 141)',
    'rgb(17, 165, 121)', 
    'rgb(231, 63, 116)', 
    'rgb(3, 169, 244)',  
    'rgb(242, 183, 1)',   
    'rgb(139, 148, 103)', 
    'rgb(255, 160, 122)', 
    'rgb(0, 90, 91)',     
    'rgb(102, 204, 204)',  
    'rgb(182, 144, 196)'    
]

@dataclass
class BarLineAreaConfig:
    """Configuration for DataFrame preprocessing steps."""
    agg_func: Optional[Union[str, Callable]] = None
    agg_column: Optional[str] = None
    trim_top_n_x: Optional[int] = None
    trim_top_n_y: Optional[int] = None
    trim_top_n_color: Optional[int] = None
    trim_top_n_facet_col: Optional[int] = None
    trim_top_n_facet_row: Optional[int] = None
    trim_top_n_animation_frame: Optional[int] = None
    trim_top_n_direction: Literal['top', 'bottom'] = 'top'
    trim_top_n_agg_func: Literal['mean', 'median', 'sum', 'count', 'nunique'] = 'count'
    norm_by: Optional[Union[str, Literal['all']]] = None
    freq: Optional[str] = None
    lower_quantile: Optional[Union[float, int]] = None
    upper_quantile: Optional[Union[float, int]] = None
    min_group_size: Optional[int] = None
    observed_for_groupby: bool = True
    show_group_size: bool = True
    show_box: bool = False
    show_count: bool = False
    show_top_and_bottom_n: Optional[int] = None
    show_legend_title: bool = False
    horizontal_spacing: Optional[float] = None

class BarLineAreaBuilder:
    """Builds bar/line/area plots with enhanced preprocessing."""
    COUNT_COLUMN = 'count_for_subplots'
    VALID_AGG_FUNC = ['mean', 'median', 'sum', 'count', 'nunique', 'min', 'max', 'std', 'var', 'prod', 'first', 'last']
    def __init__(self, plot_type: Literal['bar', 'line', 'area']):
        if plot_type not in FIGURE_CREATORS:
            raise ValueError(f"Invalid plot_type. Must be one of: {list(FIGURE_CREATORS.keys())}")
        self.original_data_frame = None
        self.plot_type = plot_type
        self.config = BarLineAreaConfig()
        self.plotly_kwargs = {}
        self.orientation = None
        self.need_trim_category = False
        self.norm_factor = None
        self.figure = None

    def build(
        self,
        **kwargs
    ) -> CustomFigure:
        """Build and return the configured figure."""
        self._separate_params(kwargs)
        trim_fields = ['x', 'y', 'color', 'facet_col', 'facet_row', 'animation_frame']
        if any(getattr(self.config, f"trim_top_n_{field}") for field in trim_fields) or self.config.show_top_and_bottom_n:
            self.need_trim_category = True
        self._validate_params()
        self._prepare_plotly_kwargs()
        self._set_default_params_in_plotly_kwargs()
        if self.config.show_box or self.config.show_top_and_bottom_n:
            self.original_data_frame = self.plotly_kwargs['data_frame'].copy()       
        if self.config.show_top_and_bottom_n:
            self._change_category_orders_for_top_and_bottom()             
        # Preprocess data
        self._preprocess_data()

        # Build figure
        self._create_figure()
        # Add optional components
        if self.plot_type == 'bar':
            if self.config.show_top_and_bottom_n:
                self._add_top_bottom_view()
            if self.config.show_count:
                self._add_count_view()
            if self.config.show_box:
                self._add_box_view()

        # Final styling
        self._apply_final_styling()

        return self.figure

    def _separate_params(self, kwargs: Dict[str, Any]) -> None:
        """
        Separate parameters between config and plotly_kwargs.
        """
        config_updates = {}
        plotly_updates = kwargs.pop('plotly_kwargs', {})
        config_fields = {f.name for f in fields(self.config)}
        config_types = get_type_hints(BarLineAreaConfig)

        for key, value in kwargs.items():
            if value is None:
                continue
            if not isinstance(key, str):  # Not string keys protection
                raise TypeError(f"Parameter name must be string, got {type(key)}")
            if key in config_fields:
                expected_type = config_types[key]
                # Checking for union types
                if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                    if not isinstance(value, expected_type.__args__):
                        raise TypeError(f"Invalid type for '{key}'. Expected one of {expected_type.__args__}, got {type(value)}")

                # Checking for Literal types
                elif hasattr(expected_type, '__origin__') and expected_type.__origin__ is Literal:
                    if value not in expected_type.__args__:
                        raise ValueError(f"Invalid value '{value}' for {key}. Must be one of: {expected_type.__args__}")

                # Checking for ordinary types
                elif isinstance(expected_type, type) and not isinstance(value, expected_type):
                    raise TypeError(f"Invalid type for '{key}'. Expected {expected_type}, got {type(value)}")

                config_updates[key] = value
            else:
                plotly_updates[key] = value

        self.config.__dict__.update(config_updates)
        self.plotly_kwargs.update(plotly_updates)

    def _prepare_plotly_kwargs(self) -> None:
        """
        Prepare various input formats to a DataFrame suitable for plotting.
        Handles cases similar to plotly express histogram.

        Parameters
        ----------
        plotly_kwargs: dict
            Dictionary containing plotly express arguments.

        Returns
        -------
        pd.DataFrame
            DataFrame suitable for plotting
        """
        plotly_kwargs = self.plotly_kwargs
        data = plotly_kwargs.get('data_frame')
        x = plotly_kwargs.get('x')
        y = plotly_kwargs.get('y')
        color = plotly_kwargs.get('color')
        facet_row = plotly_kwargs.get('facet_row')
        facet_col = plotly_kwargs.get('facet_col')
        animation_frame = plotly_kwargs.get('animation_frame')
        # Check if data is already a DataFrame
        if data is not None:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, pd.Series):
                if data.name is None:
                    df = data.to_frame(name=0)
                else:
                    df = data.to_frame()
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
                df.columns = range(data.shape[1])
            elif isinstance(data, list):
                if all(isinstance(row, list) for row in data):
                    df = pd.DataFrame(data)
                    df.columns = range(len(data[0])) if data else []
                else:
                    df = pd.DataFrame({0: data})
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                raise ValueError("Unsupported type for 'data'. Must be one of DataFrame, Series, array-like, or dict.")
        else:
            # Collect data from x, y, color, facet_row, facet_col, animation_frame
            data_dict = {}
            # display(locals().items())
            for key, value in locals().items():
                if key in ['x', 'y', 'color', 'facet_row', 'facet_col', 'animation_frame']:
                    if value is not None:
                        if isinstance(value, str):
                            raise ValueError(f"Unsupported type for '{key}'. If 'data' is not provided, '{key}' must be Series, array-like, or list.")
                        elif isinstance(value, pd.Series):
                            if value.name:
                                data_dict[value.name] = value
                                plotly_kwargs[key] = value.name
                            else:
                                data_dict[key] = value
                                plotly_kwargs[key] = key
                        elif isinstance(value, np.ndarray):
                            if value.ndim != 1:
                                raise ValueError(f"Only 1-dimensional arrays are supported for {key}.")
                            data_dict[key] = value
                            plotly_kwargs[key] = key
                        elif isinstance(value, list):
                            if not all(not isinstance(i, list) for i in value):
                                raise ValueError(f"Only 1-dimensional lists are supported for {key}.")
                            data_dict[key] = value
                            plotly_kwargs[key] = key
                        else:
                            raise ValueError(f"Unsupported type for '{key}'. Must be Series, array-like, or list.")

            if not data_dict:
                raise ValueError("Either 'data' or at least one of 'x', 'y', 'color', 'facet_row', 'facet_col', 'animation_frame' must be provided.")

            # Check if all provided data have the same length
            lengths = {key: len(value) for key, value in data_dict.items()}
            unique_lengths = set(lengths.values())
            if len(unique_lengths) != 1:
                raise ValueError(f"All provided data must have the same length. Provided lengths: {lengths}")

            # Create DataFrame from collected data
            df = pd.DataFrame(data_dict)

        # Check if x, y, color, facet_row, facet_col, animation_frame are strings and exist in the DataFrame
        for key, value in locals().items():
            if key in ['x', 'y', 'color', 'facet_row', 'facet_col', 'animation_frame']:
                if value is not None and isinstance(value, str):
                    if value not in df.columns:
                        raise ValueError(f"Column '{value}' not found in the DataFrame. Available columns: {list(df.columns)}")

        self.plotly_kwargs['data_frame'] = df

    def _validate_params(self) -> None:
        """Validate all input parameters."""
        self._validate_show_params()
        self._validate_dimensions_requirements()
        if self.config.agg_func:
            self._validate_agg_params()
        if self.need_trim_category:
            self._validate_trim_category_params()
        if self.config.norm_by:
            self._validate_norm_params()
        if self.config.freq:
            self._validate_datetime_columns()
            self._validate_time_params()
        if self.plotly_kwargs.get('category_orders'):
            self._validate_category_orders()
        self._validate_quantiles()

    def _validate_show_params(self) -> None:
        """Validate parameters"""
        show_box = self.config.show_box
        show_count = self.config.show_count
        show_top_and_bottom_n = self.config.show_top_and_bottom_n
        agg_func = self.config.agg_func
        plotly_kwargs = self.plotly_kwargs
        is_show_top_and_bottom_n = show_top_and_bottom_n is not None
        if sum([show_box, show_count, is_show_top_and_bottom_n]) > 1:
            raise ValueError("show_box/show_count/show_top_and_bottom_n cannot be used together")
        if any([show_box, show_count, show_top_and_bottom_n]) and not agg_func:
            raise ValueError("show_box/show_count/show_top_and_bottom_n require agg_func")

        if any([show_box, show_count, show_top_and_bottom_n]) and \
            any([plotly_kwargs.get('facet_col'), plotly_kwargs.get('facet_row'), plotly_kwargs.get('animation_frame')]):
            raise ValueError("Cannot use show_box/show_count/show_top_and_bottom_n with facet_col/facet_row/animation_frame")

    def _validate_dimensions_requirements(self) -> None:
        """Checks that when using Color/Facet_col/Facet_row/Animation_frame, X or Y are set"""
        dimensions_present = []

        for dim in ['color', 'facet_col', 'facet_row', 'animation_frame']:
            if dim in self.plotly_kwargs and self.plotly_kwargs[dim] is not None:
                dimensions_present.append(dim)

        if dimensions_present and 'x' not in self.plotly_kwargs and 'y' not in self.plotly_kwargs:
            raise ValueError(f'When using {", ".join(dimensions_present)}, at least one of x or y must be specified. Current values: x={self.plotly_kwargs.get("x")}, y={self.plotly_kwargs.get("y")}')

    def _validate_agg_params(self) -> None:
        """Validate aggregation parameters."""
        # Try to determine agg_column if not specified
        agg_func = self.config.agg_func
        if not self.config.agg_column:
            self.config.agg_column = self._determine_agg_column()
        agg_column = self.config.agg_column
        df = self.plotly_kwargs['data_frame']
        if (isinstance(agg_func, str) and agg_func not in self.VALID_AGG_FUNC) or (not isinstance(agg_func, str)  and not callable(agg_func)):
            raise ValueError(f'Invalid aggregation function "{agg_func}". Must be callable or one of: {self.VALID_AGG_FUNC}')

        grouping_cols = self._get_grouping_columns()

        if len(grouping_cols) != len(set(grouping_cols)):
            raise ValueError('When using agg_func x, y, color, facet_col, facet_row and animation_frame must be unique.')

        if not grouping_cols:
            raise ValueError('When using agg_func, at least one of x, y, color, facet_col, facet_row, or animation_frame must be specified for grouping.')

        # has_categorical = False
        # for col in grouping_cols:
        #     if is_categorical_column(df[col]) or is_datetime_column(df[col]):
        #         has_categorical = True
        #         break

        # if not has_categorical:
        #     raise ValueError('When using agg_func, at least one of x, y, color, facet_col, facet_row, or animation_frame must be categorical.')

        # Validate the agg_column is appropriate for the agg_func
        if (agg_func not in ['count', 'nunique'] and
            not pd.api.types.is_numeric_dtype(df[agg_column])):
            raise ValueError(f'For non-numeric x and y columns, agg_func must be either "count" or "nunique", got "{agg_func}"')
        if self.config.show_top_and_bottom_n:
            if agg_column == self.plotly_kwargs.get('x'):
                self.config.trim_top_n_y = self.config.show_top_and_bottom_n
            if agg_column == self.plotly_kwargs.get('y'):
                self.config.trim_top_n_x = self.config.show_top_and_bottom_n

    def _get_grouping_columns(self) -> List[str]:
        """Get columns to group by based on plot parameters."""
        plotly_kwargs = self.plotly_kwargs
        return [
            col for col in [
                plotly_kwargs.get('x'), plotly_kwargs.get('y'), plotly_kwargs.get('color'),
                plotly_kwargs.get('facet_col'), plotly_kwargs.get('facet_row'),
                plotly_kwargs.get('animation_frame')
            ] if col is not None
        ]

    def _determine_agg_column(self) -> str:
        """Determine which column to aggregate automatically."""
        numeric_cols = []
        categorical_cols = []
        plotly_kwargs = self.plotly_kwargs
        df = plotly_kwargs['data_frame']
        x_col = plotly_kwargs.get('x')
        y_col = plotly_kwargs.get('y')
        if is_datetime_column(df[x_col]) and y_col:
            return y_col
        if is_datetime_column(df[y_col]) and x_col:
            return x_col
        # Classify all columns
        specified_cols = self._get_grouping_columns()
        for col in specified_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            elif is_categorical_column(df[col]):
                categorical_cols.append(col)
        # Case 1: Only one numeric column - use it
        if len(numeric_cols) == 1:
            if numeric_cols[0] not in [x_col, y_col]:
                raise ValueError('Only one numerical column and it is not x or y')
            return numeric_cols[0]
        # Case 2: No numeric columns but one categorical - use it with count/nunique
        if not numeric_cols:
            if len(categorical_cols) == 1 and len(specified_cols) == 1:
                return categorical_cols[0]
            if x_col is None and y_col is not None:
                return y_col
            if x_col is not None and y_col is None:
                return x_col

        # Case 3: Multiple options - require explicit specification
        raise ValueError(
            "Cannot automatically determine aggregation column. "
            "Please specify agg_column explicitly. Options:\n"
            f"Numeric columns: {numeric_cols}\n"
            f"Categorical columns: {categorical_cols}"
        )

    def _validate_datetime_columns(self) -> None:
        """Checks the correctness of work with temporary rows """
        df = self.plotly_kwargs['data_frame']
        x_col = self.plotly_kwargs.get('x')
        y_col = self.plotly_kwargs.get('y')
        # For global freq
        if isinstance(self.config.freq, (str, timedelta, DateOffset)):
            datetime_cols = []
            for dim in ['x', 'y']:
                if dim in self.plotly_kwargs and pd.api.types.is_datetime64_any_dtype(df[self.plotly_kwargs[dim]]):
                    datetime_cols.append(self.plotly_kwargs[dim])

            if not datetime_cols:
                raise ValueError(f"Column x/y (current: x={x_col}, y={y_col}) is not datetime type but freq parameter is specified")

        # For Freq in the form of a dictionary
        elif isinstance(self.config.freq, dict):
            for col, freq in self.config.freq.items():
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    raise ValueError(f'Column "{col}" is not datetime type but freq parameter is specified')

    def _validate_trim_category_params(self) -> None:
        """Validate trimming parameters."""
        df = self.plotly_kwargs['data_frame']
        show_top_and_bottom_n = self.config.show_top_and_bottom_n
        if show_top_and_bottom_n and (not isinstance(show_top_and_bottom_n, int) or show_top_and_bottom_n < 0):
            raise ValueError('show_top_and_bottom_n must be positive integer')
        trim_params = {
            'x': self.config.trim_top_n_x,
            'y': self.config.trim_top_n_y,
            'color': self.config.trim_top_n_color,
            'facet_col': self.config.trim_top_n_facet_col,
            'facet_row': self.config.trim_top_n_facet_row,
            'animation_frame': self.config.trim_top_n_animation_frame
        }

        if any(trim_params.values()) and not self.config.trim_top_n_agg_func:
            raise ValueError(f'Top-N trimming requires specifying trim_top_n_agg_func (current: None)')

        if self.config.trim_top_n_direction not in ['top', 'bottom']:
            raise ValueError(f'Invalid trim direction "{self.config.trim_top_n_direction}". Must be either "top" or "bottom"')

        for dim, n in trim_params.items():
            if n is not None:
                if not isinstance(n, int) or n <= 0:
                    raise ValueError('Top-N value must be positive integer, got {n}')

                if dim in self.plotly_kwargs and pd.api.types.is_numeric_dtype(df[self.plotly_kwargs[dim]]):
                    raise ValueError(f'Cannot apply top-N trimming to numeric column "{self.plotly_kwargs[dim]}"')

    def _validate_norm_params(self) -> None:
        """Validate normalization parameters."""
        if self.config.norm_by and self.config.norm_by not in ['all', 'x', 'y', 'color']:
            raise ValueError(f'Invalid normalization target "{self.config.norm_by}". Must be one of: all or name of x, y, color')

    def _validate_time_params(self) -> None:
        """Validate time-related parameters."""
        df = self.plotly_kwargs['data_frame']
        if self.config.freq is None:
            return
        test_data = {
                'date': pd.date_range(start='2023-01-01', periods=1, freq='D'),
                'value': 1
                }
        test_df = pd.DataFrame(test_data)
        if isinstance(self.config.freq, dict):
            for col, freq in self.config.freq.items():
                if col not in df.columns:
                    raise ValueError(f'Date column "{col}" not found for time-based operations')
                if not self._is_valid_freq(test_df, freq):
                    raise ValueError(f'Invalid frequency specification for column "{col}": {freq}. See: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases')
        else:
            if not self._is_valid_freq(test_df, self.config.freq):
                raise ValueError(f'Invalid frequency specification: {self.config.freq}. See: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases')

    def _is_valid_freq(self, test_df: pd.DataFrame, freq: Union[str, timedelta, DateOffset]) -> bool:
        """Check if frequency specification is valid."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                test_df.groupby(pd.Grouper(key='date', freq=freq))
                return True
            except ValueError:
                return False

    def _validate_category_orders(self) -> None:
        """Validate category orders."""
        # Check if we have a numeric column when needed
        has_string_orders = any(isinstance(order, str) for order in self.plotly_kwargs.get('category_orders', {}).values())
        if has_string_orders and not self._has_numeric_column():
            raise ValueError('No numeric column found for sorting operations. Required when using numeric sorting methods')

    def _has_numeric_column(self) -> bool:
        """Check if DataFrame has any numeric columns."""
        df = self.plotly_kwargs['data_frame']
        return any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)

    def _validate_quantiles(self) -> None:
        """Validate quantile parameters."""
        for q in [self.config.lower_quantile, self.config.upper_quantile]:
            if q is not None and not (0 <= q <= 1):
                raise ValueError(f'Quantile value must be between 0 and 1, got {q}')

    def _set_default_params_in_plotly_kwargs(self) -> None:
        """Set default parameters for plotly kwargs"""
        if self.plot_type in ['line', 'area']:
            if self.plotly_kwargs.get('color'):
                self.plotly_kwargs.setdefault('color_discrete_sequence', colorway_for_line)
            self.plotly_kwargs.setdefault('line_shape', 'spline')
        if self.config.agg_func in ['count', 'nunique']:
            self.plotly_kwargs.setdefault('hover_data', {}).update({self.config.agg_column: ':.0f'})
        if self.config.norm_by:
            self.plotly_kwargs.setdefault('hover_data', {}).update({self.config.agg_column: ':.2f'})
        if self.config.agg_column == self.plotly_kwargs.get('y'):
            self.orientation = 'v'
        else:
            self.orientation = 'h'

    def _preprocess_data(self) -> None:
        """Preprocess data"""
        # Calculate normalization factors before any trimming
        if self.config.norm_by:
            self._calculate_normalization_factors()
        
        # Apply trimming if specified
        if self.need_trim_category:
            self._trim_categories_by_top_n()

        # Apply aggregation if specified
        if self.config.agg_func:
            self._apply_aggregation()
        else:
            # For static plotly rendering better convert datetime to string.
            self._convert_datetime_columns()
        # Apply normalization if specified
        if self.config.norm_by:
            self._apply_normalization()

        # Process category orders
        if self.plotly_kwargs.get('category_orders'):
            self._process_category_orders()

    def _calculate_normalization_factors(self) -> Optional[pd.Series]:
        """Calculate normalization factors before trimming."""
        
        df = self.plotly_kwargs['data_frame']
        num_column = self._determine_column_for_operation(type='normalization')
        agg_func = self.config.agg_func
        
        if self.config.norm_by == 'all':
            # For 'all' we use total sum
            if agg_func:
                self.norm_factor = df[num_column].agg(agg_func)
            else:
                self.norm_factor = df[num_column].sum()
        else:
            norm_col = self.plotly_kwargs.get(self.config.norm_by)
            if not norm_col:
                raise ValueError(f"Cannot normalize by {self.config.norm_by} - {self.config.norm_by} not specified")
        
            # Calculate sum for each normalization group
            if agg_func:
                self.norm_factor = df.groupby([norm_col], observed=self.config.observed_for_groupby)[num_column].agg(agg_func)
            else:
                self.norm_factor = df.groupby([norm_col], observed=self.config.observed_for_groupby)[num_column].sum()

    def _change_category_orders_for_top_and_bottom(self) -> None:
        self.config.trim_top_n_direction = 'top'
        agg_column = self.config.agg_column
        x_col = self.plotly_kwargs.get('x')
        y_col = self.plotly_kwargs.get('y')
        if agg_column == x_col and y_col:
            self.plotly_kwargs.setdefault('category_orders', {})[y_col] = 'descending'
        if agg_column == y_col and x_col:
            self.plotly_kwargs.setdefault('category_orders', {})[x_col] = 'descending'

    def _trim_categories_by_top_n(self) -> None:
        """Apply top-N trimming to specified dimensions."""
        plotly_kwargs = self.plotly_kwargs
        df = plotly_kwargs['data_frame']
        mask = None
        column_for_trim = self._determine_column_for_operation(type='trimming')

        trim_params = {
            'x': self.config.trim_top_n_x,
            'y': self.config.trim_top_n_y,
            'color': self.config.trim_top_n_color,
            'facet_col': self.config.trim_top_n_facet_col,
            'facet_row': self.config.trim_top_n_facet_row,
            'animation_frame': self.config.trim_top_n_animation_frame
        }
        group_cols = self._get_grouping_columns()
        df_for_trim = df[group_cols]
        if len(group_cols) == 1 and column_for_trim == group_cols[0]:
            new_column_for_trim = column_for_trim + '_trim'
            df_for_trim[new_column_for_trim] = df_for_trim[column_for_trim]
            column_for_trim = new_column_for_trim
        for dim, n in trim_params.items():
            if n is None:
                continue

            if dim not in plotly_kwargs:
                raise ValueError(f"Cannot trim {dim} - not specified in plot parameters")

            col = plotly_kwargs[dim]
            if not is_categorical_column(df_for_trim[col]):
                raise ValueError(f'Top-N trimming cannot be applied to numeric column "{col}". Only categorical columns can be trimmed')

            ascending = self.config.trim_top_n_direction == 'bottom'
            top_values = (
                df_for_trim.groupby(col, observed=self.config.observed_for_groupby)[column_for_trim]
                .agg(self.config.trim_top_n_agg_func)
                .sort_values(ascending=ascending)[:n]
                .index
            )
            dim_mask = df_for_trim[col].isin(top_values)
            mask = dim_mask if mask is None else mask & dim_mask
        plotly_kwargs['data_frame'] = df[mask] if mask is not None else df

    def _determine_column_for_operation(self, type: str) -> str:
        """Determine which column to use for operations."""
        plotly_kwargs = self.plotly_kwargs
        df = plotly_kwargs['data_frame']
        if self.config.agg_column:
            if not pd.api.types.is_numeric_dtype(df[self.config.agg_column]) and self.config.agg_func not in ['count', 'nunique']:
                raise ValueError(f'For non-numeric x and y columns, agg_func must be either "count" or "nunique", got "{self.config.agg_func}"')
            return self.config.agg_column

        x_col = plotly_kwargs.get('x')
        y_col = plotly_kwargs.get('y')

        if x_col and y_col:
            x_num = pd.api.types.is_numeric_dtype(df[x_col])
            y_num = pd.api.types.is_numeric_dtype(df[y_col])

            if x_num and y_num:
                raise ValueError(f'For both numeric x and y columns, {type} is not supported')
            elif x_num:
                return x_col
            elif y_num:
                return y_col
            else:
                raise ValueError(f'For both non numeric x and y columns, {type} is not supported')
        elif x_col and pd.api.types.is_numeric_dtype(df[x_col]):
            return x_col
        elif y_col and pd.api.types.is_numeric_dtype(df[y_col]):
            return y_col
        else:
            raise ValueError(f'For non-numeric x and y columns, {type} is not supported')

    def _apply_aggregation(self) -> None:
        """Apply aggregation to the DataFrame."""
        plotly_kwargs = self.plotly_kwargs
        df = plotly_kwargs['data_frame']
        agg_column =  self.config.agg_column
        agg_func = self.config.agg_func
        group_cols = self._get_grouping_columns()
        # Handle datetime columns with frequency
        datetime_cols = self._handle_datetime_columns(group_cols)

        # Remove agg_column from group_cols if it's present
        group_cols = [col for col in group_cols if col != agg_column]
        # Hande case than ongly agg_column in group_cols
        if not group_cols:
            group_cols = [agg_column]
            new_agg_column= agg_func if isinstance(agg_func, str) else '_agg'

            # Ensure new_agg_column is unique
            if new_agg_column == agg_column:
                new_agg_column += '_new'

            # Create a new column with the aggregated values
            df[new_agg_column] = df[agg_column]
            if self.config.norm_by :
                new_agg_column_label = 'Probability'
            else:
                new_agg_column_label = new_agg_column.title()
            self.plotly_kwargs.setdefault('labels', {}).update({new_agg_column: new_agg_column_label})
            agg_column = new_agg_column
            self.config.agg_column = agg_column

            # Update plotly_kwargs to use the new_agg_column
            if plotly_kwargs.get('y') is not None:
                plotly_kwargs['x'] = new_agg_column
            elif plotly_kwargs.get('x') is not None:
                plotly_kwargs['y'] = new_agg_column
            else:
                raise ValueError('x and y not specified when attempt grouping one categorical column')

        # If we aggregate a categorical variable, then it ceases to be categorical and we need to remove it from category_orders
        if self.plotly_kwargs.get('category_orders', {}) and agg_column in self.plotly_kwargs.get('category_orders', {}):
            self.plotly_kwargs.get('category_orders', {}).pop(agg_column)

        # Apply minimum group size filter if specified
        if self.config.min_group_size:
            df = self._filter_min_group_size(group_cols, agg_column)

        # Perform aggregation
        agg_dict = {
            agg_column: agg_func,
            'count_for_show_group_size': 'sum'
        }
        # Initialize count column
        df['count_for_show_group_size'] = df[agg_column].notna().astype(int)
        if datetime_cols:
            # Use pd.Grouper for time-based grouping with individual frequencies
            groupers = []
            for col in datetime_cols:
                freq = self._get_freq_for_column(col)
                groupers.append(pd.Grouper(key=col, freq=freq))

            other_cols = [col for col in group_cols if col not in datetime_cols]
            df_agg = df.groupby(groupers + other_cols, observed=self.config.observed_for_groupby).agg(agg_dict).reset_index()
            # Restore full index for each datetime column
            df_agg = self._restore_full_index(
                df_agg,
                date_cols=col,
                group_cols=other_cols,
                freq=self._get_freq_for_column(col),
            )
        else:
            df_agg = df.groupby(group_cols, observed=self.config.observed_for_groupby).agg(agg_dict).reset_index()

            if (plotly_kwargs.get('facet_col') or plotly_kwargs.get('facet_row')) and plotly_kwargs.get('animation_frame'):
                other_cols = [col for col in group_cols if col not in datetime_cols]
                df_agg = self._restore_full_index(
                    df_agg,
                    date_cols=[],
                    group_cols=[col for col in group_cols if col != plotly_kwargs['animation_frame']],
                )
        # Format count column
        if self.config.show_count:
            df_agg['count_for_subplots'] = df_agg['count_for_show_group_size']
        if self.config.show_group_size:
            df_agg['count_for_show_group_size'] = np.where(
                df_agg['count_for_show_group_size'] > 1e3,
                '>1000',
                df_agg['count_for_show_group_size'].astype(str)
            )
            plotly_kwargs['custom_data'] = [df_agg['count_for_show_group_size']]
        # Change type for all categories to str for plotly
        if group_cols:
            df_agg[group_cols] = df_agg[group_cols].astype(str)
        self.plotly_kwargs['data_frame'] = df_agg

    def _convert_datetime_columns(self, format: str = '%Y-%m-%d %H:%M:%S') -> pd.DataFrame:
        """Convert all datetime columns in a DataFrame to formatted strings. For better static rendering"""
        plotly_kwargs = self.plotly_kwargs
        df = plotly_kwargs['data_frame'].copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime(format)
        
        self.plotly_kwargs['data_frame'] = df      
        
        
    def _handle_datetime_columns(self, group_cols: List[str]) -> List[str]:
        """Handle datetime columns with frequency specification."""
        df = self.plotly_kwargs['data_frame']
        return [col for col in group_cols if pd.api.types.is_datetime64_any_dtype(df[col])]

    def _filter_min_group_size(self, group_cols: List[str], num_column: str) -> pd.DataFrame:
        """Filter groups smaller than min_group_size."""
        df = self.plotly_kwargs['data_frame']
        return df.groupby(group_cols, observed=self.config.observed_for_groupby).filter(
            lambda x: x[num_column].count() >= self.config.min_group_size
        )

    def _get_freq_for_column(self, col: str) -> Union[str, timedelta, DateOffset]:
        """Get frequency for specific column."""
        if isinstance(self.config.freq, dict):
            return self.config.freq.get(col, self.config.freq.get('default', 'D'))
        return self.config.freq or 'D'

    def _restore_full_index(
        self,
        df: pd.DataFrame,
        date_cols: Union[str, List[str]],
        group_cols: Union[str, List[str]],
        freq: Optional[Union[str, timedelta, DateOffset, Dict[str, Union[str, timedelta, DateOffset]]]] = None,
        fill_value: Optional[Union[str, int, float]] = None
    ) -> pd.DataFrame:
        """
        Restores a full index for a DataFrame by filling in missing dates/categories.
        Handles single or multiple date columns.
        """
        # Convert to lists if single values passed
        date_cols = [date_cols] if isinstance(date_cols, str) else date_cols
        group_cols = [group_cols] if isinstance(group_cols, str) else group_cols

        # Validate all columns exist
        for col in date_cols + group_cols:
            if col not in df.columns:
                raise ValueError(f'Date column "{col}" not found for time-based operations')

        # Generate date ranges for each date column
        date_ranges = []
        for col in date_cols:
            current_freq = self._get_freq_for_column(col) if isinstance(freq, dict) else freq
            date_ranges.append(pd.date_range(df[col].min(), df[col].max(), freq=current_freq))

        # Create full index
        if group_cols:
            # Case with grouping columns
            full_index = pd.MultiIndex.from_product(
                date_ranges + [df[col].unique() for col in group_cols],
                names=date_cols + group_cols
            )
        else:
            # Case with only date columns
            if len(date_ranges) == 1:
                full_index = date_ranges[0]
            else:
                full_index = pd.MultiIndex.from_product(date_ranges, names=date_cols)
            if len(date_cols) == 1:
                full_index.name = date_cols[0]
        # Reindex to the full index
        return df.set_index(date_cols + group_cols).reindex(full_index, fill_value=fill_value).reset_index()

    def _apply_normalization(self) -> None:
        """Apply normalization to the DataFrame."""
        plotly_kwargs = self.plotly_kwargs
        df = plotly_kwargs['data_frame']
        num_column = self._determine_column_for_operation(type='normalization')
        norm_by = self.config.norm_by
        norm_factor = self.norm_factor
        if norm_by == 'all':
            df[num_column] = df[num_column] / norm_factor
            plotly_kwargs['data_frame'] = df
        else:
            norm_factor = norm_factor.rename('_norm_factor_')
            df = (
                df.merge(norm_factor, left_on=self.plotly_kwargs[norm_by], right_index=True)
            )
            df[num_column] = df[num_column] / df['_norm_factor_']
            plotly_kwargs['data_frame'] = df.drop('_norm_factor_', axis=1)

    def _get_facet_animation_cols(self) -> List[str]:
        """Get facet and animation columns for normalization."""
        plotly_kwargs = self.plotly_kwargs
        return [
            col for col in [
                plotly_kwargs.get('facet_col'), plotly_kwargs.get('facet_row'),
                plotly_kwargs.get('animation_frame')
            ] if col is not None
        ]

    def _process_category_orders(self) -> None:
        """Process category orders and update plotly_kwargs."""
        category_orders = self.plotly_kwargs.get('category_orders')
        if category_orders is None:
            return
        plotly_kwargs = self.plotly_kwargs
        df = self.plotly_kwargs['data_frame']
        processed = {}
        agg_mode = self.config.agg_func is not None
        # Define valid sorting methods based on mode
        if not agg_mode:
            valid_methods = {
                'category ascending', 'category descending',
                'count ascending', 'count descending', 'min ascending', 'min descending',
                'max ascending', 'max descending', 'sum ascending', 'sum descending',
                'mean ascending', 'mean descending', 'median ascending', 'median descending'
            }
        else:
            valid_methods = {
                'category ascending', 'category descending',
                'ascending', 'descending'
            }

        for col, order_spec in category_orders.items():
            if col not in df.columns:
                continue

            if isinstance(order_spec, list):
                processed[col] = order_spec
                continue

            if isinstance(order_spec, str):
                # Validate sorting method
                if order_spec not in valid_methods:
                    raise ValueError(f'Invalid sorting method "{order_spec}" for column "{col}". Valid methods for {"aggregated" if agg_mode else "non-aggregated"} mode: {sorted(valid_methods)}')

                # Handle different sorting methods
                if 'category' in order_spec:
                    ascending = 'ascending' in order_spec
                    processed[col] = df[col].astype(str).sort_values(ascending=ascending).unique().tolist()
                else:
                    if agg_mode:
                        column_for_sort = self.config.agg_column
                    else:
                        column_for_sort = self._determine_column_for_operation('category orders')
                    if column_for_sort is None:
                        raise ValueError(
                            f"Cannot use '{order_spec}' for column '{col}' - "
                            "no numeric column available for sorting"
                        )

                    if not agg_mode:
                        # For non-aggregated mode, extract aggregation function from order_spec
                        agg_func = order_spec.split()[0]  # 'mean ascending' -> 'mean'
                        ascending = 'ascending' in order_spec
                        # for y axis change direction
                        if col == self.plotly_kwargs.get('y'):
                            ascending = not ascending
                        sorted_categories = (
                            df.groupby(col, observed=self.config.observed_for_groupby)[column_for_sort]
                            .agg(agg_func)
                            .sort_values(ascending=ascending)
                            .index
                            .tolist()
                        )
                    else:
                        # For aggregated mode, just sort by the numeric column
                        ascending = 'ascending' in order_spec
                        # sorted_categories = (
                        #     df.set_index(col)[column_for_sort]
                        #     .sort_values(ascending=ascending)
                        #     .index
                        #     .unique()
                        #     .tolist()
                        # )
                        # sort by grouped values
                        sorted_categories = (
                            df.groupby(col, observed=self.config.observed_for_groupby)[column_for_sort]
                            .sum()
                            .sort_values(ascending=ascending)
                            .index
                            .unique()
                            .tolist()
                        )
                    processed[col] = sorted_categories

        if processed:
            self.plotly_kwargs['category_orders'] = processed

    def _create_figure(self) -> None:
        """Create base figure using plotly express."""
        creator = FIGURE_CREATORS[self.plot_type]
        self.figure = creator(**self.plotly_kwargs)
        # Add group size to hover if needed
        if (self.config.show_group_size and self.config.agg_func and
            (self.config.agg_func not in ['count', 'nunique'] or self.config.norm_by)):
            self._add_group_size_hover()

    def _add_group_size_hover(self) -> None:
        """Add group size information to hover template."""
        df = self.plotly_kwargs['data_frame']
        if df is not None and 'count_for_show_group_size' in df.columns:
            for trace in self.figure.data:
                trace.hovertemplate += '<br>Group size = %{customdata[0]}'

    def _add_top_bottom_view(self) -> None:
        """Add top and bottom subplots."""
        left_figure = self.figure
        left_axis_category_order = self._get_axis_category_order(left_figure)
        x = self.plotly_kwargs.get('x')
        y = self.plotly_kwargs.get('y')
        category_orders = self.plotly_kwargs.get('category_orders')
        fig_subplots = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=self.config.horizontal_spacing or 0.15
        )

        # For the right figure, set the order order as on the left
        if self.plotly_kwargs.get('color'):
            left_figure_color_order = self._get_color_category_order(left_figure)
            self.plotly_kwargs.setdefault('category_orders', {}).update({self.plotly_kwargs.get('color'): left_figure_color_order})

        # Change categories for bottom figure
        if self.orientation == 'h':
            self.plotly_kwargs.setdefault('category_orders', {})[self.plotly_kwargs.get('y')] = 'ascending'
        else:
            self.plotly_kwargs.setdefault('category_orders', {})[self.plotly_kwargs.get('x')] = 'ascending'

        # Create bottom plot
        self.config.trim_top_n_direction = 'bottom'
        self.plotly_kwargs['data_frame'] = self.original_data_frame.copy()
        self._preprocess_data()
        self._create_figure()
        right_figure = self.figure
        right_axis_category_order = self._get_axis_category_order(right_figure)
        # Combine plots
        self._combine_subplots(fig_subplots, left_figure, right_figure, left_axis_category_order, right_axis_category_order)
        self.figure = fig_subplots

    def _add_count_view(self) -> None:
        """Add count subplot."""
        left_figure = self.figure
        left_axis_category_order = self._get_axis_category_order(left_figure)
        fig_subplots = make_subplots(
            rows=1, cols=2,
            shared_yaxes=True if self.orientation=='h' else False,
            horizontal_spacing=0.05 if self.orientation=='h' else 0.1
        )

        # Create count plot
        kwargs_count = self.plotly_kwargs.copy()

        # For the right figure, set the order order as on the left
        if kwargs_count.get('color'):
            left_figure_color_order = self._get_color_category_order(left_figure)
            kwargs_count.setdefault('category_orders', {}).update({kwargs_count.get('color'): left_figure_color_order})
        if self.COUNT_COLUMN not in kwargs_count['data_frame']:
            raise ValueError(f"Count column '{self.COUNT_COLUMN}' not found in DataFrame")
        if self.orientation == 'v':
            kwargs_count['y'] = self.COUNT_COLUMN
        else:
            kwargs_count['x'] = self.COUNT_COLUMN
        if self.COUNT_COLUMN not in kwargs_count.get('labels', {}):
            kwargs_count['labels'] = kwargs_count.get('labels', {}).copy()
            kwargs_count['labels'][self.COUNT_COLUMN] = 'Count'
        right_figure = px.bar(**kwargs_count)
        # Combine plots
        self._combine_subplots(fig_subplots, left_figure, right_figure, left_axis_category_order, left_axis_category_order)
        self.figure = fig_subplots

    def _add_box_view(self) -> None:
        """Add boxplot subplot."""
        self.plotly_kwargs['data_frame'] = self.original_data_frame
        if self.need_trim_category:
            self._trim_categories_by_top_n()
        left_figure = self.figure
        left_axis_category_order = self._get_axis_category_order(left_figure)
        fig_subplots = make_subplots(
            rows=1, cols=2,
            shared_yaxes=True if self.orientation=='h' else False,
            horizontal_spacing=0.05 if self.orientation=='h' else 0.1
        )

        # For the right figure, set the order order as on the left
        if self.plotly_kwargs.get('color'):
            left_figure_color_order = self._get_color_category_order(left_figure)
            self.plotly_kwargs.setdefault('category_orders', {}).update({self.plotly_kwargs.get('color'): left_figure_color_order})

        # Create boxplot
        right_figure = px.box(
            data_frame=self.plotly_kwargs['data_frame'],
            x=self.plotly_kwargs['x'],
            y=self.plotly_kwargs['y'],
            color=self.plotly_kwargs.get('color'),
            orientation=self.orientation,
            labels=self.plotly_kwargs.get('labels'),
            category_orders=self.plotly_kwargs.get('category_orders')
        )
        # Combine plots
        self._combine_subplots(fig_subplots, left_figure, right_figure, left_axis_category_order, left_axis_category_order)

        # Cut the range of the axis for boxing, if necessary
        if self.config.lower_quantile or self.config.upper_quantile:
            range_config = self._get_quantile_range()
            fig_subplots.update_layout(**range_config)
        self.figure = fig_subplots

    def _get_quantile_range(self):
        """Calculation of the quantil range"""
        range_config = {}
        lower_q = self.config.lower_quantile or 0
        upper_q = self.config.upper_quantile or 1
        num_col = self.config.agg_column
        cat_col = self.plotly_kwargs['y'] if num_col == self.plotly_kwargs['x'] else self.plotly_kwargs['x']
        group_cols = [cat_col]
        if self.plotly_kwargs.get('color'):
            group_cols.append(self.plotly_kwargs['color'])

        quantiles = (
            self.original_data_frame.groupby(group_cols, observed=self.config.observed_for_groupby)[num_col]
            .quantile([lower_q, upper_q])
            .unstack()
        )
        lower_range = quantiles.iloc[:, 0].min()
        upper_range = quantiles.iloc[:, 1].max()
        padding = (upper_range - lower_range) * 0.05

        if self.orientation == 'h':
            range_config['xaxis2_range'] = [lower_range - padding, upper_range]
        else:
            range_config['yaxis2_range'] = [lower_range - padding, upper_range]
        return range_config

    def _get_axis_category_order(self, fig: go.Figure):
        """Get current category order from the figure."""
        axis_name = 'yaxis' if self.orientation == 'h' else 'xaxis'
        if axis_name not in fig.layout:
            raise AttributeError(f'{axis_name} not found in fig.layout in _get_axis_category_order')
        has_categoryorder = fig.layout[axis_name]['categoryorder']
        if has_categoryorder:
            if isinstance(fig.layout[axis_name]['categoryorder'], str):
                return fig.layout[axis_name]['categoryarray']
            else:
                return fig.layout[axis_name]['categoryorder']

        # Fallback - extract from data
        if axis_name == 'xaxis':
            return fig.data[0].x.tolist()
        else:
            return fig.data[0].y.tolist()

    def _get_color_category_order(self, fig: go.Figure):
        """Returns the order of traces in figure"""
        return list(dict.fromkeys(trace.name for trace in fig.data if trace.name))

    def _combine_subplots(self, subplots, fig_left, fig_right, left_axis_category_order=None, right_axis_category_order=None):
        """Helper to combine two figures into subplots."""
        # Determine the axis with categories for the right fig

        for trace in fig_left.data:
            subplots.add_trace(trace, row=1, col=1)
        for trace in fig_right.data:
            subplots.add_trace(trace, row=1, col=2)
        if self.orientation == 'h':
            # Categories on Y-Axis
            subplots.update_yaxes(categoryorder='array', categoryarray=left_axis_category_order, row=1, col=1)
            subplots.update_yaxes(categoryorder='array', categoryarray=right_axis_category_order, row=1, col=2)
        else:
            # Category on x-axis
            subplots.update_xaxes(categoryorder='array', categoryarray=left_axis_category_order, row=1, col=1)
            subplots.update_xaxes(categoryorder='array', categoryarray=right_axis_category_order, row=1, col=2)

        # Handle legend duplicates
        seen_names = set()
        for trace in subplots.data:
            if trace.name in seen_names:
                trace.showlegend = False
            else:
                seen_names.add(trace.name)
        xaxis_title_text_left = fig_left.layout.xaxis.title.text
        yaxis_title_text_left = fig_left.layout.yaxis.title.text
        xaxis_title_text_right = fig_right.layout.xaxis.title.text
        yaxis_title_text_right = fig_right.layout.yaxis.title.text
        # Update axis labels
        subplots.update_xaxes(
            title_text=xaxis_title_text_left,
            row=1, col=1
        )
        subplots.update_xaxes(
            title_text=xaxis_title_text_right,
            row=1, col=2
        )
        subplots.update_yaxes(
            title_text=yaxis_title_text_left,
            row=1, col=1
        )
        if self.orientation == 'v' and self.config.show_count:
            subplots.update_yaxes(
                title_text=yaxis_title_text_right,
                row=1, col=2
            )

    def _apply_final_styling(self) -> None:
        """Apply final styling to the figure."""
        df = self.plotly_kwargs.get('data_frame')
        if df is None:
            return
        update_config = {
            'barmode': self.plotly_kwargs.get('barmode', 'group'),
            # 'boxmode': self.plotly_kwargs.get('barmode', 'group'),
            'width': self._calculate_width(),
            'height': self.plotly_kwargs.get('height', 400),
            'title_text': self.plotly_kwargs.get('title')
        }

        if self.plotly_kwargs.get('color'):               
            update_config.update({
                'legend_position': 'top',
            })
            if self.plotly_kwargs.get('facet_col'):
                update_config.update({
                    'legend_y': 1.17,
                })
                self.figure.update_layout(
                    margin=dict(t=70)
                    , title_y=0.97
                )
            if not self.config.show_legend_title:
                update_config.update({
                    'legend_title_text': ''
                })
            if self.plot_type in ['line', 'area']:
                update_config['opacity'] = 0.7
        if self.plotly_kwargs.get('facet_col'):
            for annotation in self.figure.layout.annotations:
                annotation.y -= 0.03
        self.figure = CustomFigure(self.figure).update(**update_config)

    def _calculate_width(self) -> int:
        """Calculate appropriate figure width."""
        if self.plotly_kwargs.get('width') is not None:
            return self.plotly_kwargs.get('width')
                     
        if self.plot_type in ['line', 'area']:
            width = self.plotly_kwargs.get('width', 800)
        else:    
            width = self.plotly_kwargs.get('width', 600)
        conditions = [
            self.plotly_kwargs.get('color'),
            self.config.show_box,
            self.config.show_top_and_bottom_n,
            self.config.show_count,
            self.plotly_kwargs.get('facet_col') or self.plotly_kwargs.get('facet_row'),
            self.config.freq,
        ]
        new_widths = [800, 1000, 900, 900, 1200, 800]

        for cond, new_width in zip(conditions, new_widths):
            if cond:
                width = max(width, new_width)
        return width
    
    def _calculate_height(self) -> int:
        """Calculate appropriate figure width."""
        if not self.plotly_kwargs.get('facet_col'):
            return self.plotly_kwargs.get('height', 400)
        height = self.plotly_kwargs.get('height', 400)
        if self.plotly_kwargs.get('color'):
            height = max(height, 450)
    
        return height    