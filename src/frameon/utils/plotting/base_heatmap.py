from typing import Union, Optional, Literal, Dict, List, get_type_hints, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, fields
from .custom_figure import CustomFigure
from frameon.utils.miscellaneous import is_categorical_column

@dataclass
class HeatmapConfig:
    """Configuration for heatmap visualization."""
    x: Optional[Union[str, pd.Grouper]] = None
    y: Optional[Union[str, pd.Grouper]] = None
    z: Optional[str] = None
    do_pivot: bool = False
    agg_func: Optional[str] = None
    hide_first_column: bool = False
    trim_top_n_x: Optional[int] = None
    trim_top_n_y: Optional[int] = None
    trim_top_n_direction: Literal['top', 'bottom'] = 'top'
    trim_top_n_agg_func: Literal['mean', 'median', 'sum', 'count', 'nunique'] = 'count'
    fill_value: Optional[Union[int, float]] = None

class HeatmapBuilder:
    """Builds heatmap visualizations with configurable preprocessing."""

    def __init__(self):
        self.config = HeatmapConfig()
        self.plotly_kwargs = {}
        self.figure = None

    def build(
        self,
        **kwargs
    ) -> CustomFigure:
        """Build and return the configured heatmap figure."""
        self._separate_params(kwargs)
        self._validate_params()
        # Apply top-N filtering if specified
        if self._needs_filtering():
            self._apply_trimming_categories()
        # Prepare data
        self._prepare_data()
        # For better plotly static rendering convert datetime to string
        if pd.api.types.is_datetime64_any_dtype(self.plotly_kwargs['data_frame'].columns):
            self.plotly_kwargs['data_frame'].columns = self.plotly_kwargs['data_frame'].columns.strftime('%Y-%m-%d %H:%M:%S')
        # Process category orders
        if self.plotly_kwargs.get('category_orders') is not None:
            self._process_category_orders()
        # Process labels
        if self.config.do_pivot and self.plotly_kwargs.get('labels') is not None:
            self._process_labels()
        # Create and return figure
        self._create_figure()
        self._update_figure_layout()
        return self.figure

    def _separate_params(self, kwargs: Dict[str, Any]) -> None:
        """
        Separate parameters between config and plotly_kwargs.
        """
        config_updates = {}
        plotly_updates = kwargs.pop('plotly_kwargs', {})
        config_fields = {f.name for f in fields(self.config)}
        config_types = get_type_hints(HeatmapConfig)

        for key, value in kwargs.items():
            if value is None:
                continue

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

    def _validate_params(self) -> None:
        """Validate heatmap parameters with detailed error messages."""
        config = self.config
        plotly_kwargs = self.plotly_kwargs
        data_frame = plotly_kwargs.get('data_frame')
        x = config.x
        y = config.y
        z = config.z
        do_pivot = config.do_pivot
        agg_func = config.agg_func
        hide_first_column = config.hide_first_column
        trim_top_n_x = config.trim_top_n_x
        trim_top_n_y = config.trim_top_n_y
        trim_top_n_direction = config.trim_top_n_direction
        trim_top_n_agg_func = config.trim_top_n_agg_func
        fill_value = config.fill_value
        # Basic DataFrame validation
        if not isinstance(data_frame, pd.DataFrame):
            raise TypeError("data_frame must be a pandas DataFrame")
        if do_pivot and (data_frame.shape[0] < 2 or data_frame.shape[1] < 2):
            raise ValueError("DataFrame is too small (min 2x2 required)")
        # Pivot table requirements
        if any([x, y, z]) and not do_pivot:
            raise ValueError("x, y and z can be used only with pivot mode (do_pivot)")
        if fill_value and (not isinstance(fill_value, int) or not isinstance(fill_value, float)):
            raise ValueError('fill_value must be numeric')
        for col in ['color', 'animation_frame', 'facet_col', 'facet_row']:
            if col in plotly_kwargs:
                raise ValueError(f'{col} not supported')
        trim_fields = ['x', 'y']
        top_n_params = [getattr(config, f"trim_top_n_{field}") for field in trim_fields if getattr(config, f"trim_top_n_{field}")]

        if top_n_params:
            # Check if top_n is applied to numeric columns
            for param in top_n_params:
                dim = param.replace('trim_top_n_', '')
                col = plotly_kwargs.get(dim)
                if col and pd.api.types.is_numeric_dtype(data_frame[col]):
                    raise ValueError(f'Top-N trimming cannot be applied to numeric column "{col}". Only categorical columns can be trimmed')

        if top_n_params and not self.config.trim_top_n_agg_func:
            raise ValueError(f'Top-N trimming requires specifying trim_top_n_agg_func (current: None)')

        if self.config.trim_top_n_direction not in ['top', 'bottom']:
            raise ValueError(f'Invalid trim direction "{self.config.trim_top_n_direction}". Must be either "top" or "bottom"')

        if do_pivot:
            if data_frame.empty:
                raise ValueError("DataFrame is empty")
            if not all([x, y, z]):
                raise ValueError("For pivot mode, x, y and z must all be specified")
            is_x_string = isinstance(x, str)
            is_x_grouper = isinstance(x, pd.Grouper)
            if is_x_grouper:
                x = x.key
            is_y_grouper = isinstance(y, pd.Grouper)
            if is_y_grouper:
                y = y.key
            is_y_string = isinstance(y, str)
            is_z_string = isinstance(z, str)
            if not all([is_x_string or is_x_grouper, is_y_string or is_y_grouper, is_z_string]):
                raise ValueError('x, y and z must be string type')
            if x not in data_frame.columns:
                raise ValueError(f"Column '{x}' not found in DataFrame")
            if y not in data_frame.columns:
                raise ValueError(f"Column '{y}' not found in DataFrame")
            if z not in data_frame.columns:
                raise ValueError(f"Column '{z}' not found in DataFrame")
            if agg_func is None:
                raise ValueError("For pivot mode, agg_func must be defined")
            if agg_func not in ['mean', 'median', 'sum', 'count', 'nunique', 'min', 'max']:
                raise ValueError(
                    f"Invalid agg_func '{agg_func}'. Must be one of: "
                    "'mean', 'median', 'sum', 'count', 'nunique', 'min', 'max'"
                )
            if agg_func not in ['count', 'nunique'] and not pd.api.types.is_numeric_dtype(data_frame[z]):
                raise ValueError(f"z must be numeric dtype, not numeric allow for agg_func = count/nunique")
            self.is_z_integer = pd.api.types.is_integer_dtype(data_frame[config.z])

        # Top-N trim validation
        top_n_params = ['trim_top_n_x', 'trim_top_n_y']
        if any(getattr(config, p) for p in top_n_params) and not config.trim_top_n_agg_func:
            raise ValueError("trim_top_n_agg_func must be specified when using top_n_trim parameters")

    def _needs_filtering(self) -> bool:
        """Check if top-N filtering is needed."""
        return any(getattr(self.config, p) for p in ['trim_top_n_x', 'trim_top_n_y'])

    def _apply_trimming_categories(self) -> None:
        """Apply top-N trimming to specified dimensions."""
        plotly_kwargs = self.plotly_kwargs
        df = plotly_kwargs['data_frame']
        mask = None
        column_for_trim = self.config.z
        trim_params = {
            'x': self.config.trim_top_n_x,
            'y': self.config.trim_top_n_y,
        }
        for dim, n in trim_params.items():
            if n is not None and (not isinstance(n, int) or n <= 0):
                raise ValueError('Top-N value must be positive integer, got {n}')
        group_cols = [self.config.x, self.config.y]

        df_for_trim = df[group_cols]
        for dim, n in trim_params.items():
            if n is None:
                continue
            if not isinstance(n, int) or n <= 0:
                raise ValueError('Top-N value must be positive integer, got {n}')
            if dim not in plotly_kwargs:
                raise ValueError(f"Cannot trim {dim} - not specified in plot parameters")

            col = plotly_kwargs[dim]
            if not is_categorical_column(df_for_trim[col]):
                raise ValueError(f'Top-N trimming cannot be applied to numeric column "{col}". Only categorical columns can be trimmed')

            ascending = self.config.trim_top_n_direction == 'bottom'
            top_values = (
                df_for_trim.groupby(col, observed=False)[column_for_trim]
                .agg(self.config.trim_top_n_agg_func)
                .sort_values(ascending=ascending)[:n]
                .index
            )
            dim_mask = df_for_trim[col].isin(top_values)
            mask = dim_mask if mask is None else mask & dim_mask
        df = df[mask] if mask is not None else df
        if df.empty:
            raise ValueError("No data remaining after filtering")

        if self.config.do_pivot and (df.shape[0] < 2 or df.shape[1] < 2):
            raise ValueError("Resulting DataFrame after filtering is too small (min 2x2 required)")
        self.plotly_kwargs['data_frame'] = df

    def _prepare_data(self) -> None:
        """Prepare data for heatmap visualization."""
        if self.config.do_pivot:
            df_pivoted = pd.pivot_table(
                self.plotly_kwargs['data_frame'],
                index=self.config.y,
                columns=self.config.x,
                values=self.config.z,
                aggfunc=self.config.agg_func,
                observed=False,
                fill_value=self.config.fill_value
            )
            self.plotly_kwargs['data_frame'] = df_pivoted
        if self.config.hide_first_column:
            self.plotly_kwargs['data_frame']  = self.plotly_kwargs['data_frame'].iloc[:, 1:]

    def _process_category_orders(self) -> None:
        """
        Process and sort heatmap data according to category_orders.

        Parameters:
            df: Input DataFrame (pivot table for heatmap)
            config: Configuration dictionary
            kwargs: Additional arguments

        Returns:
            Sorted DataFrame according to category_orders
        """
        category_orders = self.plotly_kwargs.pop('category_orders', {})
        df = self.plotly_kwargs['data_frame']
        valid_methods = {
            'category ascending', 'category descending',
            'count ascending', 'count descending', 'min ascending', 'min descending',
            'max ascending', 'max descending', 'sum ascending', 'sum descending',
            'mean ascending', 'mean descending', 'median ascending', 'median descending'
        }

        def _process_axis(axis: str, axis_col: str):
            nonlocal df
            if axis_col not in category_orders:
                return

            order_spec = category_orders[axis_col]

            if isinstance(order_spec, list):
                # For list order - filter existing values and keep order
                axis_values = df.columns if axis == 'columns' else df.index
                existing = [val for val in order_spec if val in axis_values]
                remaining = [val for val in axis_values if val not in existing]
                ordered = existing + remaining
                if axis == 'columns':
                    df = df[ordered]
                else:
                    df = df.loc[ordered]

            elif isinstance(order_spec, str):
                if order_spec not in valid_methods:
                    raise ValueError(
                        f"Invalid sorting method '{order_spec}' for {axis_col}. "
                        f"Valid methods: {sorted(valid_methods)}"
                    )

                if 'category' in order_spec:
                    ascending = 'ascending' in order_spec
                    sorted_values = sorted(
                        df.columns if axis == 'columns' else df.index,
                        reverse=not ascending
                    )
                else:
                    # Sort by aggregated values
                    agg_func = order_spec.split()[0]
                    ascending = 'ascending' in order_spec
                    axis_values = df.columns if axis == 'columns' else df.index
                    agg_values = df.agg(agg_func, axis=0 if axis == 'columns' else 1)
                    sorted_values = agg_values.sort_values(ascending=ascending).index.tolist()

                if axis == 'columns':
                    df = df[sorted_values]
                else:
                    df = df.loc[sorted_values]

        # Process x-axis (columns) if specified in config
        if self.config.x:
            axis_col = self.config.x
        elif df.columns.name:
            axis_col = df.columns.name
        else:
            raise ValueError('For sorting by category_orders without do_pivot df.column.name must be defined')
        _process_axis('columns', axis_col)

        # Process y-axis (index) if specified in config
        if self.config.y:
            axis_col = self.config.y
        elif df.index.name:
            axis_col = df.index.name
        else:
            raise ValueError('For sorting by category_orders without do_pivot df.index.name must be defined')
        _process_axis('index', axis_col)

        self.plotly_kwargs['data_frame'] = df

    def _process_labels(self) -> None:
        labels = self.plotly_kwargs['labels']
        new_labels = {}
        label_map = {
            self.config.x: 'x'
            , self.config.y: 'y'
            , self.config.z: 'color'
        }
        for label in labels:
            if label in label_map.values():
                new_labels[label] = labels[label]
                continue
            if label in label_map:
                new_labels[label_map[label]] = labels[label]
        self.plotly_kwargs['labels'] = new_labels

    def _create_figure(self) -> None:
        """Create the final heatmap figure."""
        self.plotly_kwargs.setdefault('aspect', True)
        self.plotly_kwargs.setdefault('text_auto', True)
        self.plotly_kwargs.setdefault('width', 1000)
        self.plotly_kwargs.setdefault('color_continuous_scale', 'Greens')
        # print(self.plotly_kwargs)
        self.figure = px.imshow(
            img = self.plotly_kwargs.pop('data_frame')
            , **self.plotly_kwargs
        )

    def _update_figure_layout(self) -> None:
        """Update figure layout and styling."""
        self.figure = CustomFigure(self.figure)
        self.figure.update_xaxes(showgrid=False)
        self.figure.update_yaxes(showgrid=False)