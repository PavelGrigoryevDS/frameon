from dataclasses import dataclass, field, fields
from typing import Union, Optional, Literal, List, Dict, get_type_hints, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from frameon.utils.miscellaneous import display, is_categorical_column
from .custom_figure import CustomFigure

@dataclass
class DistributionPlotConfig:
    """Configuration for distribution plots."""
    mode: Literal['base', 'time_series'] = 'base'
    freq: Optional[str] = None
    lower_quantile: Union[int, float] = 0
    upper_quantile: Union[int, float] = 1
    show_dual: bool = False
    trim_top_n_x: Optional[int] = None
    trim_top_n_y: Optional[int] = None
    trim_top_n_color: Optional[int] = None
    trim_top_n_facet_col: Optional[int] = None
    trim_top_n_facet_row: Optional[int] = None
    trim_top_n_animation_frame: Optional[int] = None
    trim_top_n_direction: Literal['top', 'bottom'] = 'top'
    trim_top_n_agg_func: Literal['mean', 'median', 'sum', 'count', 'nunique'] = 'count'
    show_legend_title: bool = False
    orientation: Optional[str] = None

class DistributionPlotBuilder:
    def __init__(self, plot_type: Literal['box', 'violin']):
        self.plot_type = plot_type
        self.config = DistributionPlotConfig()
        self.plotly_kwargs = {}
        self.cat_column = None
        self.num_column = None
        self.datetime_column = None
        self.figure = None

    def build(
        self,
        **kwargs
    ) -> CustomFigure:
        """Main method to build the distribution plot."""
        self._separate_params(kwargs)   
        self._prepare_plotly_kwargs()
        self._validate_params()
        if self.config.mode == 'time_series':
            self._prepare_data()
        self._determine_cat_num_columns()

        # Apply filtering and category orders
        trim_fields = ['x', 'y', 'color', 'facet_col', 'facet_row', 'animation_frame']
        if any(getattr(self.config, f"trim_top_n_{field}") for field in trim_fields):
            self._apply_trimming_categories()
        self._process_category_orders()
        # Create plot
        self._create_plot()
        self._update_layout()
        return self.figure


    def _separate_params(self, kwargs: Dict[str, Any]) -> None:
        """
        Separate parameters between config and plotly_kwargs.
        """
        config_updates = {}
        plotly_updates = kwargs.pop('plotly_kwargs', {})
        config_fields = {f.name for f in fields(self.config)}
        config_types = get_type_hints(DistributionPlotConfig)

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
        """Validate input parameters."""
        self._validate_trim_params()
        plotly_kwargs = self.plotly_kwargs
        x_col = plotly_kwargs.get('x')
        y_col = plotly_kwargs.get('y')
        df = plotly_kwargs['data_frame']
        mode = self.config.mode
        trim_top_n_x = self.config.trim_top_n_x
        freq = self.config.freq
        show_dual = self.config.show_dual
        upper_quantile = self.config.upper_quantile
        lower_quantile = self.config.lower_quantile
        if not x_col and not y_col:
            raise ValueError("At least one of x or y must be specified")    
        # Check if columns exist in DataFrame
        for col in [x_col, y_col]:
            if col and col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")    

        # cols_with_invalid_type = []
        # for col in ['color', 'animation_frame', 'facet_col', 'facet_row']:
        #     if plotly_kwargs.get(col) and plotly_kwargs[col] in df.columns and not is_categorical_column(df[plotly_kwargs[col]]):
        #         cols_with_invalid_type.append(col)
        # if cols_with_invalid_type:
        #     raise ValueError(f'{cols_with_invalid_type} must be categorical')
        
        if mode == 'time_series':
            if trim_top_n_x:
                raise ValueError("For time_series mode, trim_top_n_x can not be used")
            if not freq:
                raise ValueError("For time_series mode, freq must be specified")
            if not x_col or not pd.api.types.is_datetime64_any_dtype(df[x_col]):
                raise ValueError("For time_series mode, x must be a datetime column")
            if not y_col or not pd.api.types.is_numeric_dtype(df[y_col]):
                raise ValueError("For time_series mode, y must be numeric dtype")            

        # Checking Dual parameters
        if show_dual:
            if upper_quantile == 1 and lower_quantile == 0:
                raise ValueError("For show_dual mode, lower_quantile or upper_quantile must be defined")
            
            conflicting_options = [opt for opt in ['animation_frame', 'facet_col', 'facet_row'] 
                                if opt in plotly_kwargs]
            if conflicting_options:
                raise ValueError(
                    f"Cannot use show_dual mode with: {', '.join(conflicting_options)}. "
                    "Please remove these parameters."
                )

        # Quantiles check
        if not (0 <= lower_quantile <= 1 and 0 <= upper_quantile <= 1):
            raise ValueError("Quantiles must be between 0 and 1")
        
        if lower_quantile > upper_quantile:
            raise ValueError("lower_quantile must be <= upper_quantile")

    def _validate_trim_params(self) -> None:
        """Validate trimming parameters."""
        df = self.plotly_kwargs['data_frame']
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

    def _prepare_data(self) -> None:
        """Prepare dataframe based on mode."""
        df = self.plotly_kwargs['data_frame'].copy()
        x_col = self.plotly_kwargs.get('x')
        df[x_col] = df[x_col].dt.to_period(self.config.freq).dt.end_time
        all_periods = pd.period_range(
            start=df[x_col].min(),
            end=df[x_col].max(),
            freq=self.config.freq
        ).end_time
        df = pd.DataFrame({x_col: all_periods}).merge(df, on=x_col, how='left')
        if 'color' in self.plotly_kwargs:
            df[self.plotly_kwargs.get('color')] = df[self.plotly_kwargs.get('color')].astype(str)
        # Convert datetime to string for better plotly static rendering
        df[x_col] = df[x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
        self.plotly_kwargs['data_frame'] = df

    def _determine_cat_num_columns(self) -> None:
        """Determine plot orientation and column roles."""
        plotly_kwargs = self.plotly_kwargs
        if self.config.mode == 'time_series':
            self.num_column = self.plotly_kwargs.get('y')
            self.datetime_column = self.plotly_kwargs.get('x')
            return
        x_col = plotly_kwargs.get('x')
        y_col = plotly_kwargs.get('y')
        df = plotly_kwargs['data_frame']
        if x_col and y_col:
            x_is_num = pd.api.types.is_numeric_dtype(df[x_col])
            y_is_num = pd.api.types.is_numeric_dtype(df[y_col])
            
            if not x_is_num and not y_is_num:
                raise ValueError("At least one of x or y must be numeric")        
            is_x_cat = is_categorical_column(df[x_col])
            is_y_cat = is_categorical_column(df[y_col])          
            if not is_x_cat and not is_y_cat:
                raise ValueError("For base mode both x and y cannot be numeric - one must be categorical.")
            
            if is_x_cat:
                self.cat_column = x_col
                self.num_column = y_col
            else:
                self.cat_column = y_col
                self.num_column = x_col
        else:
            if x_col:
                self.cat_column = None
                self.num_column = x_col
            if y_col:
                self.cat_column = None
                self.num_column = y_col         

    def _apply_trimming_categories(self) -> None:
        """Apply top-N trimming to specified dimensions."""
        plotly_kwargs = self.plotly_kwargs
        df = plotly_kwargs['data_frame']
        mask = None
        column_for_trim = self.num_column
        trim_params = {
            'x': self.config.trim_top_n_x,
            'y': self.config.trim_top_n_y,
            'color': self.config.trim_top_n_color,
            'facet_col': self.config.trim_top_n_facet_col,
            'facet_row': self.config.trim_top_n_facet_row,
            'animation_frame': self.config.trim_top_n_animation_frame
        }
        group_cols = self._get_grouping_columns()
        if self.config.mode == 'time_series' and self.datetime_column:
            group_cols.pop(self.datetime_column)
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
        self.plotly_kwargs['data_frame'] = df
        
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

    def _process_category_orders(self) -> None:
        """Process category orders."""
        if not self.plotly_kwargs.get('category_orders'):
            return
        category_orders = self.plotly_kwargs['category_orders']
        df = self.plotly_kwargs['data_frame']
        num_column = self.num_column
        processed = {}
        valid_methods = {
            'category ascending', 'category descending',
            'count ascending', 'count descending', 'min ascending', 'min descending',
            'max ascending', 'max descending', 'sum ascending', 'sum descending',
            'mean ascending', 'mean descending', 'median ascending', 'median descending'
        }

        for col, order_spec in category_orders.items():
            if col not in df.columns:
                continue

            if isinstance(order_spec, list):
                processed[col] = order_spec
            elif isinstance(order_spec, str):
                if order_spec not in valid_methods:
                    raise ValueError(f"Invalid sorting method '{order_spec}' for column '{col}'. Valid methods: {sorted(valid_methods)}")

                if 'category' in order_spec:
                    ascending = 'ascending' in order_spec
                    # for y axis change direction
                    if col == self.plotly_kwargs.get('y'):
                        ascending = not ascending
                    processed[col] = (
                        df[col].astype(str)
                        .sort_values(ascending=ascending)
                        .unique()
                        .tolist()
                    )
                else:
                    agg_func = order_spec.split()[0]
                    ascending = 'ascending' in order_spec
                    # for y axis change direction
                    if col == self.plotly_kwargs.get('y'):
                        ascending = not ascending
                    processed[col] = (
                        df.groupby(col, observed=False)[num_column]
                        .agg(agg_func)
                        .sort_values(ascending=ascending)
                        .index
                        .tolist()
                    )

        if processed:
            self.plotly_kwargs.setdefault('category_orders', {}).update(processed)

    def _create_plot(self) -> None:
        """Create the main plot or dual plot."""
        plot_func = px.box if self.plot_type == 'box' else px.violin
        plotly_kwargs = self.plotly_kwargs    
        config = self.config
        if config.show_dual:
            fig_left = plot_func(**plotly_kwargs)
            self._trim_data()
            fig_right = plot_func(**plotly_kwargs)
            self.figure = self._create_dual_plot(fig_left, fig_right)
        else:
            if self.config.lower_quantile != 0 or self.config.upper_quantile != 1:
                self._trim_data()
            self.figure = plot_func(**self.plotly_kwargs)

    def _trim_data(self) -> None:
        """Trim data based on quantiles."""
        group_cols = []
        plotly_kwargs = self.plotly_kwargs
        df = plotly_kwargs['data_frame']    
        num_column = self.num_column 
        cat_column = self.cat_column
        if self.cat_column:
            group_cols.append(cat_column)
        for col in ['color', 'facet_col', 'facet_row', 'animation_frame']:
            if self.plotly_kwargs.get(col):
                group_cols.append(self.plotly_kwargs[col])

        if group_cols:
            def trim_group(group):
                lower = group.quantile(self.config.lower_quantile)
                upper = group.quantile(self.config.upper_quantile)
                return group.between(lower, upper)

            mask = df.groupby(group_cols, observed=False)[num_column].transform(trim_group)
            mask = mask.astype('boolean').fillna(False)
            self.plotly_kwargs['data_frame'] = df[mask]

        mask = df[self.num_column].between(
            df[self.num_column].quantile(self.config.lower_quantile),
            df[self.num_column].quantile(self.config.upper_quantile)
        )
        mask = mask.astype('boolean').fillna(False)
        self.plotly_kwargs['data_frame'] = df[mask]

    def _create_dual_plot(self, fig_left: go.Figure, fig_right: go.Figure) -> go.Figure:
        """Create dual plot layout."""
        num_column = self.num_column 
        cat_column = self.cat_column
        x_col = self.plotly_kwargs.get('x')
        shared_yaxes = num_column == x_col
        category_orders = self.plotly_kwargs.get('category_orders')
        fig = make_subplots(rows=1, cols=2, shared_yaxes=shared_yaxes, horizontal_spacing=0.05)

        for trace in fig_left.data:
            trace.update(offsetgroup=trace.legendgroup, alignmentgroup='category')
            fig.add_trace(trace, row=1, col=1)

        for trace in fig_right.data:
            if 'color' in self.plotly_kwargs:
                trace.showlegend = False
            trace.update(offsetgroup=trace.legendgroup, alignmentgroup='category')
            fig.add_trace(trace, row=1, col=2)

        # Handle category ordering
        if category_orders and cat_column in category_orders:
            order = category_orders[cat_column]
            if cat_column == x_col:
                fig.update_xaxes(categoryorder='array', categoryarray=order)
            else:
                fig.update_yaxes(categoryorder='array', categoryarray=order)
        fig.update_xaxes(title_text=fig_left.layout['xaxis']['title']['text'])
        fig.update_yaxes(title_text=fig_left.layout['yaxis']['title']['text'], row=1, col=1)
        # Update layout
        fig.update_layout(
            title_text=self.plotly_kwargs.get('title'),
        )
        if self.plot_type == 'box':
            fig.update_layout(boxmode='group')
        else:
            fig.update_layout(violinmode='group')
        return fig

    def _update_layout(self) -> None:
        """Apply final layout updates."""
        fig = self.figure
        num_column = self.num_column 
        x_col = self.plotly_kwargs.get('x')        
        is_x_numeric = num_column == x_col
        mode = self.config.mode
        show_dual = self.config.show_dual
        color = self.plotly_kwargs.get('color')
        # Grid configuration
        fig.update_xaxes(showgrid=is_x_numeric)
        fig.update_yaxes(showgrid=not is_x_numeric)

        # Format hover labels
        for trace in fig.data:
            if is_x_numeric:
                trace.hovertemplate = trace.hovertemplate.replace('{x}', '{x:.2f}')
            else:
                trace.hovertemplate = trace.hovertemplate.replace('{y}', '{y:.2f}')

        # Default dimensions
        layout_updates = {}
        if not self.plotly_kwargs.get('width'):
            layout_updates['width'] = 800 if mode == 'time_series' else (900 if show_dual else 700)
        if not self.plotly_kwargs.get('height'):
            layout_updates['height'] = 400

        if color:
            layout_updates.update({
                'legend_position': 'top',
                'legend_y': 1.08
            })
            if not self.config.show_legend_title:
                legend_title_text = ''
            else:
                legend_title_text = self.plotly_kwargs.get('labels', {}).get(color, color)
            layout_updates.update({
                'legend_title_text': legend_title_text
            })  
        self.figure = CustomFigure(fig).update(**layout_updates)