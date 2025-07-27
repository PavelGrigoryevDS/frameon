from dataclasses import dataclass, field, fields
from typing import Optional, Union, List, Dict, Any, get_type_hints, Literal
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import display
from .custom_figure import CustomFigure

@dataclass
class CatCompareConfig:
    """Configuration for categorical comparison visualization."""
    trim_top_n_cat1: Optional[int] = None
    trim_top_n_cat2: Optional[int] = None
    barmode: str = 'group'
    text_auto: Union[bool, str] = False
    labels: Dict[str, str] = field(default_factory=dict)
    category_orders: Dict[str, List[str]] = field(default_factory=dict)
    hover_name: Optional[str] = None
    hover_data: Union[List, Dict] = field(default_factory=dict)
    return_figs: bool = False
    legend_position: str = 'top'
    heights: list = [400, 450, 450],
    width: int = 1000
    horizontal_spacing: Union[int, float] = 0.1
    visible_graphs: List[int] = field(default_factory=lambda: [1, 2, 3])
    fig_layouts: List[Dict[str, Any]] = field(default_factory=list)
    bargroupgap: float = None

class CatCompareBuilder:
    """Builds categorical comparison plots with enhanced configuration options."""
    def __init__(self, data_frame: pd.DataFrame, cat1: str, cat2: str):
        self.data_frame = data_frame
        self.plotly_kwargs = {}
        self.cat1 = cat1
        self.cat2 = cat2
        self.config = CatCompareConfig()
        self.top_n_cat1 = None
        self.top_n_cat2 = None
        self.figs = []
        
    def build(self, **kwargs) -> Union[List[CustomFigure], None]:
        """Main method to build and configure the categorical comparison plots."""
        self._separate_params(kwargs)
        self._validate_parameters()
        self._set_default_params_in_plotly_kwargs()
        self._process_category_orders()
        self.default_titles = self._generate_section_titles(self.cat1, self.cat2, self.config.labels)
        self._prepare_data()
        self._create_figures()
        
        if self.config.return_figs:
            return self.figs
        return None
    
    def _separate_params(self, kwargs: Dict[str, Any]) -> None:
        """
        Separate parameters between config and plotly_kwargs.
        """
        config_updates = {}
        plotly_updates = kwargs.pop('plotly_kwargs', {})
        config_fields = {f.name for f in fields(self.config)}
        config_types = get_type_hints(CatCompareConfig)

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

    def _generate_section_titles(self, cat1: str, cat2: str, labels: dict) -> list:
        """Generate automatic titles for each section using labels if available."""
        cat1_label = labels.get(cat1, cat1).title()
        cat2_label = labels.get(cat2, cat2).title()

        return [
            f"Distribution of {cat1_label} and {cat2_label}",
            f"{cat1_label} Distribution Across {cat2_label}",
            f"{cat2_label} Distribution Across {cat1_label}"
        ]

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if not isinstance(self.data_frame, pd.DataFrame):
            raise ValueError('data_frame must be pandas DataFrame')        
        if not self.cat1:
            raise ValueError("cat1 must be specified")
        if self.cat1 == self.cat2:
            raise ValueError('cat1 and cat2 cannot be the same')
        if not hasattr(self.data_frame, self.cat1):
            raise ValueError(f"Column {self.cat1} not found in DataFrame")

    def _set_default_params_in_plotly_kwargs(self) -> None:
        """Set default parameters for plotly kwargs"""
        # Set default label if not provided
        if 'data_for_px_bar' not in self.config.labels:
            self.config.labels['data_for_px_bar'] = '% of Total Count'
        if 'data_for_px_bar' not in self.config.hover_data and isinstance(self.config.hover_data, dict):    
            self.config.hover_data['data_for_px_bar'] = ':.1%'        

    def _prepare_data(self) -> None:
        """Prepare data for visualization."""
        self.df_cat1 = self.data_frame.groupby(self.data_frame[self.cat1], observed=True).size().to_frame('value_for_px_bar')
        self.df_cat1.index = self.df_cat1.index
        self.df_cat2 = self.data_frame.groupby(self.data_frame[self.cat2], observed=True).size().to_frame('value_for_px_bar')
        self.df_cat2.index = self.df_cat2.index
        self.df_cat1_by_cat2 = pd.crosstab(self.data_frame[self.cat1], self.data_frame[self.cat2])       
        self.df_cat2_by_cat1 = self.df_cat1_by_cat2.T    
        if self.config.trim_top_n_cat1:
            self.top_n_cat1 = self.df_cat1['value_for_px_bar'].nlargest(self.config.trim_top_n_cat1).index.tolist()
        if self.config.trim_top_n_cat2:
            self.top_n_cat2 = self.df_cat2['value_for_px_bar'].nlargest(self.config.trim_top_n_cat2).index.tolist()
            
    def _process_category_orders(self) -> None:
        """Process category orders based on counts and update plotly_kwargs.
        
        Supports string shortcuts for sorting:
        - 'category ascending' - sort categories alphabetically A-Z
        - 'category descending' - sort categories alphabetically Z-A
        - 'ascending' - sort by count (smallest to largest)
        - 'descending' - sort by count (largest to smallest)
        
        Examples:
            category_orders={'gender': 'descending'}  # Show genders by count (high to low)
            category_orders={'age_group': 'category ascending'}  # Alphabetical order
        """
        if not self.config.category_orders:
            return

        valid_methods = {
            'category ascending', 'category descending',
            'ascending', 'descending'
        }
        
        processed_orders = {}
        
        for col, order_spec in self.config.category_orders.items():
            if isinstance(order_spec, list):
                # Already processed order
                processed_orders[col] = list(map(str, order_spec))
                continue
                
            if isinstance(order_spec, str):
                if order_spec not in valid_methods:
                    raise ValueError(
                        f'Invalid sorting method "{order_spec}" for column "{col}". '
                        f'Valid methods: {sorted(valid_methods)}'
                    )
                
                # Get unique categories
                categories = self.data_frame[col].unique().astype(str)
                
                if 'category' in order_spec:
                    # Alphabetical sorting
                    ascending = 'ascending' in order_spec
                    processed_orders[col] = sorted(
                        categories, 
                        key=lambda x: str(x), 
                        reverse=not ascending
                    )
                else:
                    # Count-based sorting
                    ascending = 'ascending' in order_spec
                    counts = self.data_frame[col].value_counts()
                    processed_orders[col] = counts.sort_values(
                        ascending=ascending
                    ).index.astype(str).tolist()
        
        if processed_orders:
            self.config.category_orders = processed_orders

    def _create_figures(self) -> None:
        """Create all requested figures."""
        row_sets = [
            [(self.df_cat1, 'cat1', 'all'), (self.df_cat2, 'cat2', 'all')],  # Row 1
            [(self.df_cat1_by_cat2, 'cat1_by_cat2', 'index'), (self.df_cat1_by_cat2, 'cat1_by_cat2', 'columns')],  # Row 2
            [(self.df_cat2_by_cat1, 'cat2_by_cat1', 'index'), (self.df_cat2_by_cat1, 'cat2_by_cat1', 'columns')]   # Row 3
        ]
        
        for i, row in enumerate(row_sets):
            if (i + 1) in self.config.visible_graphs:
                fig = self._create_figure_for_row(row, i)
                self._apply_figure_layout(fig, i)
                self.figs.append(fig)
                fig.show()
    
    def _create_figure_for_row(self, row_sets: List, row_index: int) -> go.Figure:
        """Create a figure for a specific row of visualizations."""
        if row_index != 0:
            shared_yaxes = True
            horizontal_spacing=0.05
        else:
            shared_yaxes = False
            horizontal_spacing=self.config.horizontal_spacing
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=horizontal_spacing, shared_yaxes=shared_yaxes)
        
        for col, (df, graph_type, norm_mode) in enumerate(row_sets):
            df_prepared = self._prepare_df_for_visualization(df, graph_type, norm_mode)
            kwargs = self._get_graph_kwargs(graph_type)
            sub_fig = px.bar(df_prepared, **kwargs, **self.plotly_kwargs)
            for trace in sub_fig.data:
                trace.hovertemplate += '<br>Count = %{customdata}'
                fig.add_trace(trace, row=1, col=col + 1)
            self._update_subplot_axes(fig, sub_fig, col + 1, row_index)
        
        return fig
    
    def _prepare_df_for_visualization(self, df: pd.DataFrame, graph_type: str, norm_mode: str) -> pd.DataFrame:
        """Prepare dataframe for visualization with normalization."""
        # Normalize the crosstab
        normalized = self._normalize_data(df, norm_mode)
        # Apply trimming if needed
        if graph_type in ['cat1', 'cat2']:
            if self.top_n_cat1 and normalized.index.name == self.cat1:
                normalized = normalized.loc[self.top_n_cat1]
                df = df.loc[self.top_n_cat1]
            elif self.top_n_cat2 and normalized.index.name == self.cat2:
                normalized = normalized.loc[self.top_n_cat2]
                df = df.loc[self.top_n_cat2]
        else:
            if self.top_n_cat1:
                if normalized.index.name == self.cat1:
                    normalized = normalized.loc[self.top_n_cat1]
                    df = df.loc[self.top_n_cat1]
                else:
                    normalized = normalized[self.top_n_cat1]
                    df = df[self.top_n_cat1]
            if self.top_n_cat2:
                if normalized.index.name == self.cat2:
                    normalized = normalized.loc[self.top_n_cat2]
                    df = df.loc[self.top_n_cat2]
                else:
                    normalized = normalized[self.top_n_cat2]         
                    df = df[self.top_n_cat2]         
        # Combine original and normalized data
        result = pd.concat([normalized, df], axis=1, keys=['data_for_px_bar', 'customdata_for_px_bar'])
        # Transform to "long" format
        result = result.reset_index()
        cat1 = self.cat1
        cat2 = self.cat2
        if graph_type in ['cat1', 'cat2']:
            cat_col = cat1 if graph_type == 'cat1' else cat2
            result.columns = [cat_col, 'data_for_px_bar', 'customdata_for_px_bar']
            result_long = result.melt(id_vars=cat_col, var_name='type', value_name='value_for_px_bar')
            result_long[cat_col] = result_long[cat_col].astype(str)
        else:
            new_columns = result.columns.map(lambda x: f"{x[0]}__{x[1]}" if x[0] not in [cat1, cat2] else x[0]).tolist()
            result.columns = new_columns
            result_long = result.melt(
                id_vars=cat1 if graph_type == 'cat1_by_cat2' else cat2,
                var_name=f'type_{cat2 if graph_type == "cat1_by_cat2" else cat1}',
                value_name='value_for_px_bar'
            )
            result_long[['type', cat2 if graph_type == 'cat1_by_cat2' else cat1]] = (
                result_long[f'type_{cat2 if graph_type == "cat1_by_cat2" else cat1}'].str.split('__', n=1, expand=True)
            )
            result_long = result_long.drop(columns=f'type_{cat2 if graph_type == "cat1_by_cat2" else cat1}')
            result_long[cat1] = result_long[cat1].astype(str)
            result_long[cat2] = result_long[cat2].astype(str)

        # Transform to final format
        result_long = result_long.set_index(list(set(result_long.columns) - set(['value_for_px_bar'])))
        result_long = result_long.unstack(level='type').droplevel(0, axis=1).reset_index()

        return result_long
    
    def _normalize_data(self, data: pd.DataFrame, mode: str) -> pd.DataFrame:
        """Normalize data based on specified mode."""
        if mode == 'all':
            return data / data.sum().sum()
        elif mode == 'columns':
            return data / data.sum()
        elif mode == 'index':
            return data.div(data.sum(axis=1), axis=0)
        raise ValueError("Unsupported normalization mode.")
    
    def _get_graph_kwargs(self, graph_type: str) -> Dict[str, Any]:
        """Get keyword arguments for specific graph type."""
        base_kwargs = {
            'barmode': self.config.barmode,
            'text_auto': self.config.text_auto,
            'labels': self.config.labels,
            'category_orders': self.config.category_orders,
            'hover_name': self.config.hover_name,
            'hover_data': self.config.hover_data
        }
        
        if graph_type == 'cat1':
            return {**base_kwargs, 'y': self.cat1, 'x': 'data_for_px_bar', 'custom_data': 'customdata_for_px_bar'}
        elif graph_type == 'cat2':
            return {**base_kwargs, 'y': self.cat2, 'x': 'data_for_px_bar', 'custom_data': 'customdata_for_px_bar'}
        elif graph_type == 'cat1_by_cat2':
            return {**base_kwargs, 'y': self.cat1, 'x': 'data_for_px_bar', 'color': self.cat2, 'custom_data': 'customdata_for_px_bar'}
        elif graph_type == 'cat2_by_cat1':
            return {**base_kwargs, 'y': self.cat2, 'x': 'data_for_px_bar', 'color': self.cat1, 'custom_data': 'customdata_for_px_bar'}
        return base_kwargs
    
    def _update_subplot_axes(self, fig: go.Figure, sub_fig: go.Figure, col: int, row_index: int):
        """Update subplot axes configuration."""
        yaxis_props = sub_fig.layout.yaxis.to_plotly_json()
        yaxis_props.pop('anchor', None)
        yaxis_props.pop('domain', None)
        fig.update_yaxes(**yaxis_props, row=1, col=col)
        if row_index == 0:
            categoryarray=None
            if self.config.category_orders:
                if col == 1 and self.cat1 in self.config.category_orders:
                    categoryarray = self.config.category_orders[self.cat1]
                elif col == 2 and self.cat2 in self.config.category_orders:
                    categoryarray = self.config.category_orders[self.cat2]
            fig.update_yaxes(
                showgrid=False
                , title=None
                , categoryorder='array'
                , categoryarray=categoryarray[::-1] if categoryarray else None # For horisontal bar reversed order
                , row=1, col=col
            )
        else:
            fig.update_yaxes(
                showgrid=False
                , title_standoff=7
            )
            fig.update_yaxes(
                title=None
                , row=1, col=2
            )            
            fig.update_xaxes(
                showgrid=True
                , title_standoff=7
            )
            if col == 2:
                fig.update_traces(showlegend=False, row=1, col=2)
        fig.update_xaxes(title_text='% of Total Count')
        fig.update_xaxes(tickformat='.0%')
    
    def _apply_figure_layout(self, fig: go.Figure, row_index: int):
        """Apply final layout to figure."""
        fig.update_layout(
            height=self.config.heights[row_index],
            width=self.config.width
        )
        if row_index in [1, 2]:  
            legend_titles = [
                self.config.labels.get(self.cat2, self.cat2)
                , self.config.labels.get(self.cat1, self.cat1)
            ]
            fig.update_layout(
                yaxis_domain=[0, 0.9]
                , yaxis2_domain=[0, 0.9]
                , legend_title=legend_titles[row_index-1]
                , bargroupgap=self.config.bargroupgap
            )
            for x_pos, text in [(0.25, 'Normalized by axis'), (0.75, 'Normalized by legend')]:
                fig.add_annotation(
                    text=text, xref="paper", yref="paper",
                    xanchor="center", x=x_pos, y=0.95,
                    showarrow=False, font_size=12
                )
        if row_index == 0:  
            cat1_label = self.config.labels.get(self.cat1, self.cat1)
            cat2_label = self.config.labels.get(self.cat2, self.cat2)
            fig.update_layout(yaxis_domain=[0, 0.98], yaxis2_domain=[0, 0.98])
            for x_pos, text in [(0.25, cat1_label), (0.75, cat2_label)]:
                fig.add_annotation(
                    text=text, xref="paper", yref="paper",
                    xanchor="center", x=x_pos, y=1.04,
                    showarrow=False, font_size=12
                )        
        # Legend positioning
        legend_config = {
            'orientation': "h" if self.config.legend_position == 'top' else "v",
            'yanchor': "top",
            'y': 1.05 if self.config.legend_position == 'top' else 1,
            'xanchor': "center" if self.config.legend_position == 'top' else "left",
            'x': 0.5 if self.config.legend_position == 'top' else 1.1,
            'itemsizing': "constant"
        }
        fig.update_layout(legend=legend_config)
        fig.update_layout(
            title_text=self.default_titles[row_index]   
        )
        # Apply additional layout if provided
        if row_index < len(self.config.fig_layouts):
            fig.update_layout(**self.config.fig_layouts[row_index])
        fig = CustomFigure(fig)