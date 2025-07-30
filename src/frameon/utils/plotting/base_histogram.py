import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import numpy as np
from typing import Optional, Dict, Any, Union, List, Literal, Tuple, get_type_hints
from frameon.utils.miscellaneous import convert_to_dataframe, is_datetime_column
from IPython.display import display
import warnings
from .custom_figure import CustomFigure
from frameon.utils.miscellaneous import is_categorical_column
from dataclasses import dataclass, fields
from frameon.utils.plotting.base_qqplot import create_qqplot

@dataclass
class HistogramConfig:
    """Configuration for histogram visualization."""
    lower_quantile: Optional[Union[float, int]] = 0
    upper_quantile: Optional[Union[float, int]] = 1
    mode: Literal['base', 'dual_hist_trim', 'dual_box_trim', 'dual_hist_qq'] = 'base'
    show_kde: bool = False
    show_hist: bool = True
    show_box: bool = True
    legend_position: str = 'top'
    colorway: Optional[List[str]] = None
    row_heights_box: float = 0.2  # Default boxplot height ratio
    row_heights_hist: float = 0.8  # Default histogram height ratio
    show_legend_title: bool = False
    renderer: str = None
    xaxis_type: str = None
    yaxis_type: str = None  

class HistogramBuilder:
    """Builds histogram plots with enhanced preprocessing and visualization options."""
    VALID_MODES = ['base', 'dual_hist_trim', 'dual_box_trim', 'dual_hist_qq']
    HISTNORM_MAP = {
        None: 'Count',
        '': 'Count',
        'percent': 'Percent',
        'probability': 'Probability',
        'density': 'Density',
        'probability density': 'Density',
    }
    def __init__(self):
        self.original_data = None
        self.config = HistogramConfig()
        self.plotly_kwargs = {}
        self.figure = None
        self.only_plotly_mode = False
        self.xaxis_title = None
        self.yaxis_title = None
        self.figure_width = None
        self.figure_height = None

    def build(
        self,
        **kwargs
    ) -> Union[None, CustomFigure]:
        """Main method to build and configure the histogram plot."""
        self._separate_params(kwargs)

        self._prepare_plotly_kwargs()

        self._validate_parameters()

        self._set_default_params_in_plotly_kwargs()

        self.num_col = self._determine_numeric_column()
        
        if self.config.mode in ['dual_hist_trim', 'dual_box_trim']:
            self.original_data = self.plotly_kwargs['data_frame'].copy()
        # Data processing pipeline
        self._preprocess_data()

        # If x and y are numical just send kwargs to plotly
        if self.only_plotly_mode:
            fig = px.histogram(**self.plotly_kwargs)
            return CustomFigure(fig)
        # Apply views based on mode
        if self.config.mode == 'base':
            self._create_base_figure()
        elif self.config.mode == 'dual_hist_trim':
            self._create_dual_histogram_view()
        elif self.config.mode == 'dual_box_trim':
            self._create_box_histogram_view()
        elif self.config.mode == 'dual_hist_qq':
            self._create_qq_histogram_view()
        self._apply_final_styling()

        if self.config.renderer is not None:
            self.figure.show(config=dict(dpi=200), renderer=self.config.renderer, height=self.figure_height, width=self.figure_width)
        else:
            return self.figure

    def _separate_params(self, kwargs: Dict[str, Any]) -> None:
        """
        Separate parameters between config and plotly_kwargs.
        """
        config_updates = {}
        plotly_updates = kwargs.pop('plotly_kwargs', {})
        config_fields = {f.name for f in fields(self.config)}
        config_types = get_type_hints(HistogramConfig)

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

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if self.config.mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode. Must be one of: {self.VALID_MODES}")

        if self.config.mode != 'base' and any(self.plotly_kwargs.get(k) for k in ['facet_col', 'facet_row', 'animation_frame']):
            raise ValueError("Facet and animation features are only supported in 'base' mode")

        for q in [self.config.lower_quantile, self.config.upper_quantile]:
            if q is not None and not (0 <= q <= 1):
                raise ValueError(f'Quantile value must be between 0 and 1, got {q}')
        if self.config.mode in ['dual_hist_trim', 'dual_box_trim'] and self.config.upper_quantile == 1 and self.config.lower_quantile == 0:
            raise ValueError(f'At least one quantile must differ from 0 (lower) or 1 (upper) for {self.config.mode}.')
        if self.config.mode == 'dual_hist_qq' and self.plotly_kwargs.get('color'):
            raise ValueError('For dual_hist_qq mode, color not supported')
        if self.plotly_kwargs.get('marginal') is not None:
            raise ValueError("marginal can not be used, use show_box instead")

    def _set_default_params_in_plotly_kwargs(self) -> None:
        """Setup plotly express parameters."""
        # Set defaults
        if self.plotly_kwargs.get('color') is not None:
            self.plotly_kwargs.setdefault('barmode', 'overlay')
        self.plotly_kwargs.setdefault('nbins', 50)

        if self.config.show_kde:
            self.plotly_kwargs.setdefault('histnorm', 'probability density')
        else:
            self.plotly_kwargs.setdefault('histnorm', 'probability')

    def _determine_numeric_column(self) -> Union[str, None]:
        """Determine which column contains numeric data for histogram."""
        if self.plotly_kwargs.get('x') and self.plotly_kwargs.get('y'):
            return None
        df = self.plotly_kwargs['data_frame']
        # Check if x or y is numeric
        for axis in ['x', 'y']:
            col = self.plotly_kwargs.get(axis)
            if col is None or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            return col
        return None

    def _preprocess_data(self) -> None:
        """Preprocess data based on quantile filtering."""
        num_col = self.num_col
        if num_col is None:
            self.only_plotly_mode = True
            return
        
        if self.config.upper_quantile == 1 and self.config.lower_quantile == 0:
            return
        
        group_cols = self._get_grouping_columns()
        df = self.plotly_kwargs['data_frame']
        if group_cols:
            # Group-aware quantile filtering
            def filter_group(group):
                lower = group.quantile(self.config.lower_quantile)
                upper = group.quantile(self.config.upper_quantile)
                return group.between(lower, upper)

            mask = df.groupby(group_cols, observed=False)[num_col].transform(filter_group)
            mask = mask.astype('boolean').fillna(False)
            df = df[mask]
        else:
            # Simple quantile filtering
            lower = df[num_col].quantile(self.config.lower_quantile)
            upper = df[num_col].quantile(self.config.upper_quantile)
            df = df[df[num_col].between(lower, upper)]
        if df.empty:
            warnings.warn("Filtered data is empty. Check your quantile parameters.")
        self.plotly_kwargs['data_frame'] = df

    def _get_grouping_columns(self) -> List[str]:
        """Get columns used for grouping (color/facet/animation)."""
        group_cols = []
        for dim in ['color', 'facet_col', 'facet_row', 'animation_frame']:
            if dim in self.plotly_kwargs and self.plotly_kwargs[dim] is not None:
                group_cols.append(self.plotly_kwargs[dim])
        return group_cols

    def _create_base_figure(self) -> None:
        """Create the base histogram figure."""
        if not self.config.show_hist and not self.config.show_kde and not self.config.show_box:
            self.figure = go.Figure()
            return

        fig = px.histogram(**self.plotly_kwargs)
        self._set_axis_titles(fig)
        rows = len(fig._grid_ref) if hasattr(fig, '_grid_ref') else 1
        cols = len(fig._grid_ref[0]) if hasattr(fig, '_grid_ref') and len(fig._grid_ref) > 0 else 1
        # Create subplots figure
        fig_new = self._make_subplots_figure(fig.data, rows, cols)
        fig_frames = fig.frames
        if fig_frames:
            fig_new.frames = fig_frames
            for i, frame in enumerate(fig_frames):
                fig_for_frame = self._make_subplots_figure(frame.data, rows, cols)
                fig_new.frames[i].data = fig_for_frame.data
        fig_new.update_layout(
            annotations = fig.layout.annotations
        )
        fig_new.layout.updatemenus = fig.layout.updatemenus
        fig_new.layout.sliders = fig.layout.sliders
        for layout_key in fig.layout:
            if layout_key.startswith('xaxis'):
                fig_new.layout[layout_key].domain = fig.layout[layout_key].domain
        fig_new.update_layout(annotations=fig.layout.annotations)
        if self.plotly_kwargs.get('color'):
            fig_new.update_layout(legend=fig.layout.legend)

        self.figure = fig_new

    def _set_axis_titles(self, fig) -> None:
        """Get axis titles and set them in config"""
        if self.num_col == self.plotly_kwargs.get('y'):
            self.yaxis_title = fig.layout.yaxis.title
            self.xaxis_title = self.HISTNORM_MAP[self.plotly_kwargs.get('histnorm')]
        if self.num_col == self.plotly_kwargs.get('x'):
            self.xaxis_title = fig.layout.xaxis.title
            self.yaxis_title = self.HISTNORM_MAP[self.plotly_kwargs.get('histnorm')]

    def _create_dual_histogram_view(self) -> None:
        """Create dual view with original and trimmed histograms."""
        fig_trimmed = px.histogram(**self.plotly_kwargs)
        self.plotly_kwargs['data_frame'] = self.original_data
        fig_origin = px.histogram(**self.plotly_kwargs)
        self._set_axis_titles(fig_origin)
        subplots = make_subplots(rows=1, cols=2, horizontal_spacing=0.07)
        # Add original histogram (left)
        for trace in fig_origin.data:
            trace.bingroup = None
            subplots.add_trace(trace, row=1, col=1)
        # Add trimmed histogram (right)
        for trace in fig_trimmed.data:
            trace.bingroup = None
            trace.showlegend = False
            subplots.add_trace(trace, row=1, col=2)

        rows = len(subplots._grid_ref) if hasattr(subplots, '_grid_ref') else 1
        cols = len(subplots._grid_ref[0]) if hasattr(subplots, '_grid_ref') and len(subplots._grid_ref) > 0 else 1
        # Create subplots figure
        self.figure = self._make_subplots_figure(subplots.data, rows, cols)
        if self.plotly_kwargs.get('color'):
            self.figure.update_layout(legend=fig_origin.layout.legend)
        if not self.config.show_hist:
            self.figure.update_traces(visible=False, selector={'type': 'histogram'})

    def _create_box_histogram_view(self) -> None:
        """Create dual view with boxplot and trimmed histogram."""
        fig_hist = px.histogram(**self.plotly_kwargs)
        self._set_axis_titles(fig_hist)
        rows = len(fig_hist._grid_ref) if hasattr(fig_hist, '_grid_ref') else 1
        cols = len(fig_hist._grid_ref[0]) if hasattr(fig_hist, '_grid_ref') and len(fig_hist._grid_ref) > 0 else 1
        # Create subplots figure
        right_fig = self._make_subplots_figure(fig_hist.data, rows, cols)
        if not self.config.show_hist:
            right_fig.update_traces(visible=False, selector={'type': 'histogram'})
        left_fig = px.box(
            data_frame=self.original_data,
            x=self.num_col,
            color=self.plotly_kwargs.get('color'),
            labels=self.plotly_kwargs.get('labels'),
            category_orders=self.plotly_kwargs.get('category_orders'),
        )
        # Create subplot structure
        fig = self._make_box_histogram_layout()
        # Add traces
        self._add_box_histogram_traces(fig, left_fig, right_fig)
        # Apply final styling
        self._style_box_histogram_view(fig)
        self.figure = fig
        if self.plotly_kwargs.get('color'):
            self.figure.update_layout(legend=fig_hist.layout.legend)

    def _make_box_histogram_layout(self) -> go.Figure:
        """Create layout for dual box-trim view."""
        if self.config.show_box:
            row_heights = self._calculate_row_heights()
            return make_subplots(
                rows=2, cols=2,
                row_heights=row_heights,
                vertical_spacing=0.05,
                shared_xaxes=True,
                specs=[
                    [{'rowspan': 2}, {'colspan': 1}],
                    [None, {'colspan': 1}]
                ]
            )
        return make_subplots(rows=1, cols=2, horizontal_spacing=0.1)

    def _add_box_histogram_traces(self, fig, left_fig, right_fig) -> None:
        """Add traces to dual box-trim figure."""
        # Add original boxplot (left top)
        for trace in left_fig.data:
            trace.x0 = None
            trace.y0 = None
            trace.showlegend = False if self.config.show_kde or self.config.show_hist or self.config.show_box else True
            fig.add_trace(trace, row=1, col=1)
        # Add trimmed histogram/boxplot (right)
        for trace in right_fig.data:
            if trace.type == 'box' and self.config.show_box:
                fig.add_trace(trace, row=1, col=2)
            else:
                fig.add_trace(trace, row=2 if self.config.show_box else 1,
                            col=2)

    def _style_box_histogram_view(self, fig) -> None:
        """Apply final styling to dual box-trim figure."""
        # Configure boxplots axes
        row_col = [(1, 1)]
        if self.config.show_box:
            row_col += [(1, 2)]
        for row, col in row_col:
            if row == 1 and col == 1:
                continue
            fig.update_xaxes(
                showticklabels=False,
                showline=False,
                showgrid=False,
                ticks='',
                row=row,
                col=col
            )
            fig.update_yaxes(visible=False, row=row, col=col)

        # Titles
        row = 2 if self.config.show_box else 1
        fig.update_xaxes(title=self.xaxis_title, row=row, col=2)
        fig.update_yaxes(title=self.yaxis_title, row=row, col=2)
        fig.update_xaxes(title=self.xaxis_title, row=1, col=1)


    def _create_qq_histogram_view(self) -> None:
        """Create view with histogram and QQ-plot."""
        fig_hist = px.histogram(**self.plotly_kwargs)
        self._set_axis_titles(fig_hist)
        rows = len(fig_hist._grid_ref) if hasattr(fig_hist, '_grid_ref') else 1
        cols = len(fig_hist._grid_ref[0]) if hasattr(fig_hist, '_grid_ref') and len(fig_hist._grid_ref) > 0 else 1
        # Create subplots figure
        left_fig = self._make_subplots_figure(fig_hist.data, rows, cols)
        data_for_qq = self.plotly_kwargs.get('data_frame')[self.num_col]
        right_fig = create_qqplot(x=data_for_qq)
        # Create subplot structure
        fig = self._make_qq_histogram_layout()
        # Add traces
        self._add_qq_histogram_traces(fig, left_fig, right_fig)
        self._style_qq_histogram_view(fig)
        self.figure = fig

    def _make_qq_histogram_layout(self) -> go.Figure:
        """Create layout for dual box-trim view."""
        if self.config.show_box:
            row_heights = self._calculate_row_heights()
            return make_subplots(
                rows=2, cols=2,
                row_heights=[0.07, 0.93],
                horizontal_spacing=0.1,
                vertical_spacing=0,
                specs=[
                    [{'colspan': 1}, {'rowspan': 2}],
                    [{'colspan': 1}, None]
                ]
            )
        return make_subplots(rows=1, cols=2, horizontal_spacing=0.1)

    def _add_qq_histogram_traces(self, fig, left_fig, right_fig) -> None:
        """Add traces to dual box-trim figure."""
        # Add histogram (left top)
        for trace in left_fig.data:
            if trace.type == 'box' and self.config.show_box:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace, row=2 if self.config.show_box else 1,
                            col=1)
        # Add qqplot
        for trace in right_fig.data:
            fig.add_trace(
                trace
                , row=1, col=2
            )
        for annotation in right_fig.layout.annotations:
            annotation.update(xref='paper', yref='paper')
            fig.add_annotation(annotation)

    def _style_qq_histogram_view(self, fig) -> None:
        """Apply final styling to dual box-trim figure."""
        # Configure axes
        if self.config.show_box:
            fig.update_xaxes(
                showticklabels=False,
                showline=False,
                ticks='',
                row=1,
                col=1
            )
            fig.update_yaxes(visible=False, row=1, col=1)

        # Titles
        row = 2 if self.config.show_box else 1
        fig.update_xaxes(title=self.xaxis_title, row=row, col=1)
        fig.update_yaxes(title=self.yaxis_title, row=row, col=1)
        fig.update_xaxes(title='Theoretical Quantiles', row=1, col=2)
        fig.update_yaxes(title='Sample Quantiles', row=1, col=2)
        fig.update_layout(showlegend=False)

    def _make_subplots_figure(self, traces: List[go.Histogram], rows:int, cols: int) -> go.Figure:
        """Create the complete subplots figure with all components."""
        show_box=self.config.show_box
        mode=self.config.mode
        # Determine subplot structure
        if show_box:
            row_heights = self._calculate_row_heights()
            new_rows = rows * 2
            row_heights = row_heights * rows
        else:
            new_rows = rows
            row_heights = None
        # Create base subplot structure
        fig = make_subplots(
            rows=new_rows,
            cols=cols,
            shared_xaxes=True,
            shared_yaxes=False if mode == 'dual_hist_trim' else True,
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            row_heights=row_heights,
            start_cell='top-left',
        )

        # Add all traces to the figure
        self._add_traces_to_subplots(fig, traces, rows, cols)
        return fig

    def _calculate_row_heights(self) -> Tuple[float, float]:
        """Calculate row heights based on number of categories."""
        if 'color' in self.plotly_kwargs:
            group_labels = self.plotly_kwargs['data_frame'][self.plotly_kwargs['color']].unique()
            categories_cnt = len(group_labels)

            # Adjust figure height based on number of categories
            height_increase = categories_cnt * 20  # Increase height by 20 pixels for each box plot
            base_height = 400
            total_height = base_height + height_increase
            self.plotly_kwargs.setdefault('height', total_height)

            # Set row height ratios based on number of categories
            if categories_cnt == 1:
                row_heights_box = 0.07
                row_heights_hist = 0.93
            elif categories_cnt <= 2:
                row_heights_box = 0.13
                row_heights_hist = 0.87
            elif categories_cnt <= 5:
                row_heights_box = 0.2
                row_heights_hist = 0.8
            elif categories_cnt <= 8:
                row_heights_box = 0.25
                row_heights_hist = 0.75
            else:
                row_heights_box = 0.3
                row_heights_hist = 0.7
        else:
            row_heights_box = 0.07
            row_heights_hist = 0.93

        return row_heights_box, row_heights_hist

    def _add_traces_to_subplots(self, fig: go.Figure, traces: List[go.Histogram], rows, cols):
        """Add traces to subplots with proper positioning."""
        num_col=self.num_col
        show_box=self.config.show_box
        show_kde=self.config.show_kde
        mode=self.config.mode
        histnorm = self.plotly_kwargs.get('histnorm')
        histnorm_label = self.HISTNORM_MAP[histnorm]
        for trace in traces:
            # Determine the subplot position for this trace
            row, col = self._get_row_col_from_axis(trace.xaxis, cols)
            # Calculate the actual row position in the subplot grid
            if show_box:
                last_row = rows * 2
                if rows ==  1:
                    hist_row = 2
                else:
                    hist_row = row * 2
                    # Reverse order for multi-row plots
                    hist_row = rows * 2 - hist_row + 2
            else:
                last_row = rows
                hist_row = rows - row + 1
            if histnorm is not None:
                trace.hovertemplate = trace.hovertemplate.replace(histnorm, histnorm_label)
            if num_col == self.plotly_kwargs.get('x') and histnorm_label != 'Count':
                trace.hovertemplate = trace.hovertemplate.replace('%{y}', '%{y:.3f}')
            if num_col == self.plotly_kwargs.get('y') and histnorm_label != 'Count':
                trace.hovertemplate = trace.hovertemplate.replace('%{x}', '%{x:.3f}')
            # Add the main trace (histogram)
            fig.add_trace(trace, row=hist_row, col=col)

            # Add KDE trace if enabled
            if show_kde:
                kde_trace = self._create_kde_trace(trace)
                if kde_trace:
                    fig.add_trace(kde_trace, row=hist_row, col=col)
            if col == 1:
                fig.update_yaxes(
                    title=self.yaxis_title
                    , row=hist_row, col=col
                )
            else:
                if not mode == 'dual_hist_trim':
                    fig.update_yaxes(showticklabels=False, row=hist_row, col=col)
            if hist_row == last_row:
                fig.update_xaxes(
                    title=self.xaxis_title
                    , row=hist_row, col=col
                )
            # Add boxplot if enabled
            if show_box:
                self._add_boxplot_trace(fig, trace, hist_row, col)

    def _add_boxplot_trace(self, fig: go.Figure, hist_trace: go.Histogram, hist_row: int, col: int):
        """Add a boxplot trace above the histogram."""
        box_row = hist_row - 1
        num_col=self.num_col
        show_box=self.config.show_box
        show_kde=self.config.show_kde
        show_hist=self.config.show_hist
        mode=self.config.mode
        # Create boxplot trace
        box = go.Box(
            x=hist_trace.x,
            showlegend=False if show_kde or show_hist else True,
            hovertemplate=self._create_hover_template(hist_trace, 'box'),
            marker_color=hist_trace.marker.color,
            legendgroup=hist_trace.name,
            name=hist_trace.name
        )

        # Add boxplot to figure
        fig.add_trace(box, row=box_row, col=col)
        # Configure axes for boxplot
        fig.update_xaxes(
            showticklabels=False,
            showline=False,
            ticks='',
            showgrid=False,
            row=box_row,
            col=col
        )
        fig.update_yaxes(
            visible=False,
            showline=False,
            showgrid=False,
            ticks='',
            matches=None,
            row=box_row,
            col=col
        )

    def _get_row_col_from_axis(self, axis_name, cols) -> Tuple[int, int]:
        """Determine row and column from axis name."""
        if not axis_name:
            return 1, 1
        # Axis name has format "xaxis", "xaxis2", "xaxis3", etc.
        if axis_name == 'x':
            axis_number = 1
        else:
            axis_number = int(axis_name.replace("x", "")) if axis_name.startswith("x") else 1
        row = (axis_number - 1) // cols + 1
        col = (axis_number - 1) % cols + 1
        return row, col

    def _add_kde_traces(self) -> None:
        """Add KDE traces to the histogram."""
        for trace in self.figure.data:
            if trace.type == 'histogram':
                kde_trace = self._create_kde_trace(trace)
                if kde_trace:
                    self.figure.add_trace(kde_trace)

    def _create_kde_trace(self, hist_trace: go.Histogram) -> Union[go.Scatter, None]:
        """Create KDE trace for a histogram trace."""
        from scipy.stats import gaussian_kde
        import numpy as np
        label_for_kde = hist_trace.name
        hist_data = hist_trace.x if hist_trace.x is not None else hist_trace.y
        if len(hist_data) < 2:
            warnings.warn(f'In group "{label_for_kde}" number of elements less then 2. KDE line cannot be constructed.')
            x_range = [0]
            y_values = [0]
        else:
            try:
                # Calculate KDE
                kde = gaussian_kde(hist_data)
                x_range = np.linspace(min(hist_data), max(hist_data), 500)
                y_values = kde(x_range)
            except Exception as e:
                warnings.warn(f"Failed to create KDE: {str(e)}")
                x_range = [0]
                y_values = [0]
        # Create hover template
        hovertemplate_kde = self._create_hover_template(hist_trace, 'kde')

        kde_trace = go.Scatter(
            x=x_range,
            y=y_values,
            mode='lines',
            name=label_for_kde,
            legendgroup=label_for_kde,
            line=dict(color=hist_trace.marker.color),
            hovertemplate=hovertemplate_kde,
            xaxis=hist_trace.xaxis,
            yaxis=hist_trace.yaxis,
        )
        if self.config.show_hist:
            kde_trace.showlegend = False
        elif self.config.mode == 'dual_hist_trim':
            kde_trace.showlegend = hist_trace.showlegend
        else:
            True
        return kde_trace


    def _create_hover_template(self, trace, trace_type) -> None:
        """The Universal Method for Creation Hover"""
        template_parts = []

        # General part for all types
        if 'color' in self.plotly_kwargs:
            color_col = self.plotly_kwargs['color']
            color_label = self.plotly_kwargs.get('labels', {}).get(color_col, color_col)
            template_parts.append(f"{color_label}: {trace.name}<br>")

        num_col = self.num_col
        col_label = self.plotly_kwargs.get('labels', {}).get(num_col, 'Value')
        histnorm_title = self.HISTNORM_MAP[self.plotly_kwargs.get('histnorm')]
        template_parts.append(f"{col_label}: %{{x:.2f}}<br>")
        # A specific part for each type
        if trace_type == ['histogram', 'kde']:
            if histnorm_title== 'Count':
                template_parts.append(histnorm_title)
            else:
                template_parts.append(f"{histnorm_title}: %{{y:.3f}}")

        return "".join(template_parts) + "<extra></extra>"

    def _apply_final_styling(self) -> None:
        """Apply final styling to the figure."""

        update_config = {
            'barmode': self.plotly_kwargs.get('barmode'),
            'title_text': self.plotly_kwargs.get('title')
            # 'boxmode': 'group'
        }
        if 'width' in self.plotly_kwargs:
            update_config['width'] = self.plotly_kwargs['width']
            self.figure_width = self.plotly_kwargs['width']
        elif self.config.mode != 'base':
            update_config['width'] = 900
            self.figure_width = 900
        else:
            update_config['width'] = 600
            self.figure_width = 600
        if 'height' in self.plotly_kwargs:
            update_config['height'] = self.plotly_kwargs['height']
            self.figure_height = self.plotly_kwargs['height']
        else:
            update_config['height'] = 400
            self.figure_height = 400
        if self.plotly_kwargs.get('color'):
            # So that the vertical order goes from top to bottom
            self.figure.data = self.figure.data[::-1]
            self.figure.update_layout(legend_traceorder='reversed')
            if self.config.legend_position == 'top':
                update_config.update({
                    'margin': dict(l=50, r=50, t=70, b=50)
                })
                # self.update_yaxes(domain = yaxis_domain if yaxis_domain else [0, 0.95])
                self.figure.update_layout(
                    legend = dict(
                        orientation="h"
                        , yanchor="top"
                        , y=1.09
                        , xanchor="center"
                        , x=0.5
                        , itemsizing="constant"
                    )
                )
            if not self.config.show_legend_title:
                update_config.update({
                    'legend_title_text': '' 
                })

        self.figure = CustomFigure(self.figure).update(**update_config)
        self.figure.update_xaxes(title_standoff=7)
        self.figure.update_yaxes(title_standoff=7)
        if self.config.xaxis_type == 'category':
            x_data = self.figure.data[0].x
            if x_data is not None:
                mask = np.isnan(x_data)
                if any(mask):
                    x_data = x_data[~mask]
                    self.figure.data[0].x = x_data
                self.figure.update_xaxes(
                    type='category',
                    tickvals=x_data,  
                    ticktext=[f"{float(val):.2f}" if val % 1 != 0 else int(val) for val in x_data], 
                )
        elif self.config.xaxis_type:
            self.figure.update_xaxes(
                type=self.config.xaxis_type
            )          
        if self.config.yaxis_type == 'category':
            y_data = self.figure.data[0].y
            if y_data is not None:
                mask = np.isnan(y_data)
                if any(mask):
                    y_data = y_data[~mask]
                    self.figure.data[0].y = y_data
                self.figure.update_yaxes(
                    type='category',
                    tickvals=y_data,  
                    ticktext=[f"{float(val):.2f}" if val % 1 != 0 else int(val) for val in y_data], 
                )
        elif self.config.yaxis_type:
            self.figure.update_yaxes(
                type=self.config.yaxis_type
            )                  
        # Hide elements based on config
        if not self.config.show_hist:
            self.figure.for_each_trace(
                lambda t: t.update(visible=False) if t.type == 'histogram' else None
            )
