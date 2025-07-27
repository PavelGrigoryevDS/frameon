import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, pearsonr, spearmanr, kendalltau, yeojohnson, boxcox
from typing import Union, Dict, List, Tuple, Generator, Optional, Literal, Any, get_type_hints
from dataclasses import dataclass, fields, field
from itertools import combinations
from IPython.display import display
from .custom_figure import CustomFigure

def shift_to_positive(x):
    min_val = x.min()
    if min_val <= 0:
        return x + (1 - min_val)
    return x

TRANSFORM_FUNCTIONS = {
    'log': lambda x: np.log(shift_to_positive(x)),
    'boxcox': lambda x: boxcox(shift_to_positive(x))[0], 
    'yeojohnson': lambda x: yeojohnson(x)[0],
    'sqrt': lambda x: np.sqrt(shift_to_positive(x, epsilon=0)),
    'reciprocal': lambda x: 1/(x + 1e-10) 
}

@dataclass
class PairplotConfig:
    """Configuration for pairplot visualization."""
    ranges: Optional[Dict[str, Tuple[float, float]]] = None
    color_mode: Optional[Literal['count', 'kde', 'category']] = None
    color_column: Optional[str] = None 
    display_mode: Literal['scatter', 'density_contour'] = 'scatter'
    correlation_method: Literal['pearson', 'spearman', 'kendall'] = 'pearson'
    transforms: Optional[Union[str, Dict[str, str]]] = None
    show_correlation: bool = True
    width: Optional[int] = None
    height: Optional[int] = None
    category_orders: Optional[Dict[str, List[str]]] = None
    labels: Optional[Dict[str, str]] = None
    bins: int = 20
    rows: Optional[int] = None
    cols: Optional[int] = None
    horizontal_spacing: float = 0.05
    vertical_spacing: float = 0.1
    generator_mode: bool = False
    plots_per_page: Optional[int] = None
    trendline: Optional[str] = None
    plot_bgcolor: Optional[str] = None
    title: Optional[str] = None
    show_legend_title: bool = False
    color_continuous_scale: Union[str, List[str]] = 'viridis'
    color_discrete_sequence: Optional[List[str]] = None
    renderer: str = None

class PairplotBuilder:
    """Builds pairplot visualizations with configurable options."""
    
    def __init__(self):
        self.config = PairplotConfig()
        self.figure = CustomFigure()
        self.max_density = 1
        self.df = pd.DataFrame()
        self.pairs = []
        self.coloraxis = None
    
    def build(self, **kwargs) -> Union[CustomFigure, Generator[CustomFigure, None, None]]:
        """Build and return the configured pairplot figure."""
        self._separate_params(kwargs)
        self._validate_params()
        self._prepare_data()
        
        if self.config.generator_mode:
            return self._generate_plots()
        else:
            if self.config.renderer is not None:
                self._create_figure()
                self.figure.show(
                    config=dict(dpi=200), 
                    renderer=self.config.renderer,
                    height=self.config.height,
                    width=self.config.width,
                )
            else:
                self._create_figure()
                return self.figure
    
    def _separate_params(self, kwargs: Dict[str, Any]) -> None:
        """Separate parameters between config and other attributes.
        
        Args:
            kwargs: Dictionary of parameters to separate
            
        Raises:
            TypeError: If parameter type doesn't match expected type
            ValueError: If parameter value is invalid for Literal type
        """
        config_updates = {}
        config_fields = {f.name for f in fields(self.config)}
        config_types = get_type_hints(PairplotConfig)
        
        # Extract required parameters
        self.df = kwargs.pop('data_frame')
        self.pairs = kwargs.pop('pairs', None)
        
        # If pairs is not declared, we use all numerical columns
        if self.pairs is None:
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            self.pairs = list(combinations(numeric_cols, 2))
            if not self.pairs:
                raise ValueError("No numeric columns found in dataframe for pairplot")
        for key, value in kwargs.items():
            if value is None:
                continue
                
            if key not in config_fields:
                continue
                
            expected_type = config_types[key]
            
            # Skip type checking for Any type
            if expected_type is Any:
                config_updates[key] = value
                continue
                
            # Handle Union types
            if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                if not any(self._is_instance(value, t) for t in expected_type.__args__):
                    raise TypeError(
                        f"Invalid type for '{key}'. Expected one of {expected_type.__args__}, "
                        f"got {type(value)}"
                    )
                    
            # Handle Literal types
            elif hasattr(expected_type, '__origin__') and expected_type.__origin__ is Literal:
                if value not in expected_type.__args__:
                    raise ValueError(
                        f"Invalid value '{value}' for {key}. "
                        f"Must be one of: {expected_type.__args__}"
                    )
                    
            # Handle ordinary types
            elif isinstance(expected_type, type) and not isinstance(value, expected_type):
                raise TypeError(
                    f"Invalid type for '{key}'. Expected {expected_type}, got {type(value)}"
                )
                
            config_updates[key] = value
            
        self.config.__dict__.update(config_updates)

    def _is_instance(self, value: Any, type_: Any) -> bool:
        """Helper method to safely check isinstance for complex types."""
        try:
            return isinstance(value, type_)
        except TypeError:
            # Handle cases where type checking isn't possible (e.g., subscripted generics)
            return True  # Assume valid if we can't check
    
    def _validate_params(self) -> None:
        """Validate pairplot parameters."""
        df = self.df
        if df.empty:
            raise ValueError("Input dataframe is empty")
            
        # Validate pairs
        # if not self.pairs:
        #     raise ValueError("pairs must contain at least one pair of column names")
            
        if self.config.rows is not None and self.config.cols is None:
            raise ValueError("When specifying rows, you must also specify cols")

        # Convert pairs to list of tuples if needed
        if all(isinstance(item, str) for item in self.pairs):
            self.pairs = list(combinations(self.pairs, 2))
        elif not all(isinstance(item, tuple) and len(item) == 2 for item in self.pairs):
            raise TypeError(
                "pairs must be either list of column names or list of (col1, col2) tuples"
            )

        # Validate ranges
        if self.config.ranges:
            for col, (min_val, max_val) in self.config.ranges.items():
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' in ranges not found in dataframe")
                if not isinstance(min_val, (float, int, type(None))) or not isinstance(max_val, (float, int, type(None))):
                    raise ValueError(f"Range for '{col}' must be tuple of (min, max) where each can be float or None")
        
        # Validate transforms
        if self.config.transforms:
            if isinstance(self.config.transforms, str):
                if self.config.transforms not in TRANSFORM_FUNCTIONS:
                    raise ValueError(
                        f"Invalid transform '{self.config.transforms}'. "
                        f"Must be one of: {list(TRANSFORM_FUNCTIONS.keys())} or a dict of {{column: transform}}"
                    )
                # We convert the line into a dictionary for all numerical columns
                numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
                self.config.transforms = {col: self.config.transforms for col in numeric_cols}
            elif isinstance(self.config.transforms, dict):
                for col, transform in self.config.transforms.items():
                    if col not in df.columns:
                        raise ValueError(f"Column '{col}' for transform not found")
                    if transform not in TRANSFORM_FUNCTIONS:
                        raise ValueError(
                            f"Invalid transform '{transform}'. Must be one of: {list(TRANSFORM_FUNCTIONS.keys())}"
                        )
            else:
                raise TypeError(
                    "transforms must be either a string (transform name) "
                    "or a dict of {column: transform}"
                )
        
        # Validate columns exist and are numeric
        for col1, col2 in self.pairs:
            if col1 not in df.columns or col2 not in df.columns:
                raise ValueError(f"Columns {col1} or {col2} not found in dataframe")
            if not (pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(self.df[col2])):
                raise ValueError(f"Columns {col1} and {col2} must be numeric")
            
        if self.config.display_mode not in ['scatter', 'density_contour']:
            raise ValueError("display_mode must be either 'scatter' or 'density_contour'")
            
        if self.config.correlation_method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError("correlation_method must be 'pearson', 'spearman', or 'kendall'")
            
        if self.config.display_mode == 'density_contour' and (self.config.color_mode or self.config.color_column):
            raise ValueError("Cannot use coloring with density_contour display mode")
        
        if self.config.color_mode == 'category' and not self.config.color_column:
            raise ValueError("color_column must be specified when color_mode='category'")

        if self.config.color_column and self.config.color_mode != 'category':
            raise ValueError("color_column can only be used with color_mode='category'")

        if self.config.color_mode not in [None, 'count', 'kde', 'category']:
            raise ValueError("color_mode must be one of: 'count', 'kde', 'category'")

        if self.config.display_mode == 'density_contour' and self.config.color_mode:
            raise ValueError("Cannot use color_mode with density_contour display mode")

        if self.config.color_mode == 'category' and self.config.color_column not in self.df.columns:
            raise ValueError(f"color_column '{self.config.color_column}' not found in dataframe")
    
    def _prepare_data(self) -> None:
        """Prepare data by applying transformations."""
        if self.config.transforms:
            self.df = self.df.copy()
            self.df = self.df.dropna(subset=list(self.config.transforms.keys()))
            # Initialize labels if not present
            if self.config.labels is None:
                self.config.labels = {}
            # Mapping of transform functions to their display names
            TRANSFORM_DISPLAY_NAMES = {
                'log': 'log',
                'boxcox': 'Box-Cox',
                'yeojohnson': 'Yeo-Johnson',
                'sqrt': '√',
                'reciprocal': '1/'
            }
            for col, transform in self.config.transforms.items():
                self.df[col] = TRANSFORM_FUNCTIONS[transform](self.df[col])
                # Only update label if not explicitly set by user
                display_name = TRANSFORM_DISPLAY_NAMES.get(transform, transform)
                if col in self.config.labels:
                    self.config.labels[col] = f"{display_name}({self.config.labels[col]})"
                else:
                    self.config.labels[col] = f"{display_name}({col})"
                
    def _create_figure(self) -> None:
        """Create the main figure."""
        # Determine grid size
        if self.config.cols is not None:
            if self.config.rows is None:
                self.config.rows = int(np.ceil(len(self.pairs) / self.config.cols))
        elif self.config.rows is None:
            if len(self.pairs) == 1:
                self.config.rows, self.config.cols = 1, 1
            elif len(self.pairs) == 2:
                self.config.rows, self.config.cols = 1, 2
            elif len(self.pairs) <= 4:
                self.config.rows, self.config.cols = 2, 2
            else:
                self.config.cols = min(3, int(np.ceil(np.sqrt(len(self.pairs)))))
                self.config.rows = int(np.ceil(len(self.pairs) / self.config.cols))
        
        # Check grid is sufficient
        if self.config.rows * self.config.cols < len(self.pairs):
            raise ValueError(
                f"Grid {self.config.rows}x{self.config.cols} is too small for {len(self.pairs)} plots. "
                f"Need at least {len(self.pairs)} subplots."
            )
        # Automatically adjust the indentation depending on the number of lines/columns
        if not self.config.vertical_spacing  and self.config.rows > 1:
            # For a large number of lines, we reduce vertical indentation
            self.config.vertical_spacing = min(0.1, 0.3 / self.config.rows)
        if not self.config.horizontal_spacing  and self.config.cols > 1:
            # For a large number of columns, we reduce horizontal indentation
            self.config.horizontal_spacing = min(0.05, 0.3 / self.config.cols)
        # Set figure dimensions
        BASE_SIZE = 300
        if self.config.width is None:
            self.config.width = self.config.cols * BASE_SIZE
            if self.config.color_mode:
                self.config.width += 100
        if self.config.height is None:
            self.config.height = self.config.rows * BASE_SIZE
            
        self.config.width = max(self.config.width, 300)
        self.config.height = max(self.config.height, 300)
        # Create figure
        self.figure = make_subplots(
            rows=self.config.rows,
            cols=self.config.cols,
            horizontal_spacing=self.config.horizontal_spacing,
            vertical_spacing=self.config.vertical_spacing
        )
        # Add traces for each pair
        for i, (col1, col2) in enumerate(self.pairs):
            row, col = divmod(i, self.config.cols)
            row += 1  # 1-based indexing
            col += 1
            # Get axis titles
            x_title = self.config.labels.get(col1, col1) if self.config.labels else col1
            y_title = self.config.labels.get(col2, col2) if self.config.labels else col2
            # Prepare data subset
            df_trim = self._prepare_data_subset(col1, col2)
            
            # Calculate correlation if needed
            corr_text = ""
            if self.config.show_correlation:
                corr_text = self._calculate_correlation(df_trim, col1, col2)
            # Add appropriate trace
            if self.config.display_mode == 'density_contour':
                self._add_density_contour_trace(df_trim, col1, col2, x_title, y_title, row, col)
            else:
                if self.config.color_mode:
                    self.max_density = self._add_colored_scatter_trace(
                        df_trim, col1, col2, x_title, y_title, row, col, i, corr_text
                    )
                else:
                    self._add_plain_scatter_trace(df_trim, col1, col2, x_title, y_title, row, col, corr_text)
            
            # Update axes
            self._update_axes(x_title, y_title, row, col)
        
        # Finalize layout
        self._finalize_figure_layout()
    
    def _generate_plots(self) -> Union[None, Generator[go.Figure, None, None]]:
        """Generator function that yields figures page by page."""
        combinations = self.pairs
        num_plots = len(combinations)
        
        if self.config.plots_per_page is None:
            self.config.plots_per_page = num_plots
            
        num_pages = (num_plots + self.config.plots_per_page - 1) // self.config.plots_per_page
        
        for page in range(num_pages):
            start_idx = page * self.config.plots_per_page
            end_idx = min((page + 1) * self.config.plots_per_page, num_plots)
            page_combinations = combinations[start_idx:end_idx]
            num_plots_page = len(page_combinations)

            # Calculate grid layout for this page
            if self.config.rows is not None and self.config.cols is None:
                raise ValueError("When specifying rows in generator mode, you must also specify cols")
        
            # Calculate grid layout for this page
            if self.config.cols is not None:
                page_cols = self.config.cols
                page_rows = int(np.ceil(num_plots_page / page_cols)) if self.config.rows is None else self.config.rows
            else:
                if num_plots_page == 1:
                    page_rows, page_cols = 1, 1
                elif num_plots_page == 2:
                    page_rows, page_cols = 1, 2
                elif num_plots_page <= 4:
                    page_rows, page_cols = 2, 2
                else:
                    page_cols = min(3, int(np.ceil(np.sqrt(num_plots_page))))
                    page_rows = int(np.ceil(num_plots_page / page_cols))
            # Set figure dimensions
            BASE_SIZE = 300
            if self.config.width is None:
                self.config.width = page_cols * BASE_SIZE
                if self.config.color_mode:
                    self.config.width += 100
            if self.config.height is None:
                self.config.height = page_rows * BASE_SIZE
                
            self.config.width = max(self.config.width, 300)
            self.config.height = max(self.config.height, 300)     

            self.figure = make_subplots(
                rows=page_rows,
                cols=page_cols,
                horizontal_spacing=self.config.horizontal_spacing,
                vertical_spacing=self.config.vertical_spacing
            )
            
            max_density = 1
            
            for i, (col1, col2) in enumerate(page_combinations):
                row, col = divmod(i, page_cols)
                row += 1
                col += 1
                
                # Get axis titles
                x_title = self.config.labels.get(col1, col1) if self.config.labels else col1
                y_title = self.config.labels.get(col2, col2) if self.config.labels else col2
                
                # Prepare data subset
                df_trim = self._prepare_data_subset(col1, col2)
                
                # Calculate correlation if needed
                corr_text = ""
                if self.config.show_correlation:
                    corr_text = self._calculate_correlation(df_trim, col1, col2)
                
                # Add appropriate trace
                if self.config.display_mode == 'density_contour':
                    self._add_density_contour_trace(df_trim, col1, col2, x_title, y_title, row, col)
                else:
                    if self.config.color_mode:
                        max_density = self._add_colored_scatter_trace(
                            df_trim, col1, col2, x_title, y_title, row, col, i, corr_text
                        )
                    else:
                        self._add_plain_scatter_trace(df_trim, col1, col2, x_title, y_title, row, col, corr_text)
                
                # Update axes
                self._update_axes(x_title, y_title, row, col)
            
            # Finalize figure layout
            self._finalize_figure_layout(max_density)
            
            if self.config.renderer is not None:
                self.figure.show(config=dict(dpi=200), renderer=self.config.renderer)
                yield
            else:
                yield self.figure 
    
    def _prepare_data_subset(self, col1: str, col2: str) -> pd.DataFrame:
        """Prepare a subset of data based on specified ranges."""
        df_trim = self.df.copy()
        # Apply range filtering if specified
        if self.config.ranges:
            for col in [col1, col2]:
                if col in self.config.ranges:
                    min_val, max_val = self.config.ranges[col]
                    if min_val is not None:
                        df_trim = df_trim[df_trim[col] >= min_val]
                    if max_val is not None:
                        df_trim = df_trim[df_trim[col] <= max_val]
        return df_trim.dropna(subset=[col1, col2])
    
    def _calculate_correlation(self, df: pd.DataFrame, col1: str, col2: str) -> str:
        """Calculate correlation and return formatted text."""
        if self.config.correlation_method == 'pearson':
            corr, pval = pearsonr(df[col1], df[col2])
        elif self.config.correlation_method == 'spearman':
            corr, pval = spearmanr(df[col1], df[col2])
        elif self.config.correlation_method == 'kendall':
            corr, pval = kendalltau(df[col1], df[col2])
        
        return f"r = {corr:.2f}<br> pval = ({pval:.1e})"
    
    def _add_density_contour_trace(self, df: pd.DataFrame, col1: str, col2: str, 
                                  x_title: str, y_title: str, row: int, col: int) -> None:
        """Add density contour trace to figure."""
        contour_fig = px.density_contour(
            df, 
            x=col1, 
            y=col2, 
            trendline=self.config.trendline,
            category_orders=self.config.category_orders,
            labels=self.config.labels
        )
        
        for trace in contour_fig.data:
            trace.update(
                hovertemplate=f"{x_title} = %{{x}}<br>{y_title} = %{{y}}<extra></extra>",
                showlegend=False
            )
            self.figure.add_trace(trace, row=row, col=col)
    
    def _add_colored_scatter_trace(self, df: pd.DataFrame, col1: str, col2: str,
                                 x_title: str, y_title: str, row: int, col: int,
                                 trace_idx: int, corr_text: Optional[str] = None) -> float:
        """Add colored scatter trace to figure and return max density."""
        # Prepare color data
        if self.config.color_mode == 'category':
            color_data = df[self.config.color_column]
            color_title = self.config.color_column
        elif self.config.color_mode in ['count', 'kde']:
            if self.config.color_mode == 'count':
                df['_x_sector_'] = pd.cut(df[col1], bins=self.config.bins)
                df['_y_sector_'] = pd.cut(df[col2], bins=self.config.bins)
                color_data = df.groupby(['_x_sector_', '_y_sector_'], observed=False)[col1].transform('count')
            else:  # 'kde'
                xy = np.vstack([df[col1], df[col2]])
                color_data = gaussian_kde(xy)(xy)
            color_title = 'Density'
            if not self.config.labels:
                self.config.labels = {}
            self.config.labels.update({'color': 'Density'})
        else:
            raise ValueError("Invalid color_mode")         
        max_density = color_data.max()
        
        # Create figure with px.scatter
        scatter_fig = px.scatter(
            df,
            x=col1,
            y=df[col2],
            color=color_data,
            color_continuous_scale=self.config.color_continuous_scale,
            color_discrete_sequence=self.config.color_discrete_sequence,
            render_mode='webgl',
            trendline=self.config.trendline,
            category_orders=self.config.category_orders,
            labels=self.config.labels
        )
        if not self.coloraxis and self.config.color_mode in ['count', 'kde']:
            self.coloraxis = scatter_fig.layout.coloraxis
        # Update marker style
        scatter_fig.update_traces(
            marker=dict(
                line=dict(width=0.5, color='white'),
                opacity=0.8
            ),
            selector=dict(mode='markers')
        )
        
        if corr_text:
            self.figure.add_annotation(
                xref='x domain',
                yref='y domain',
                x=0.98,
                y=0.02,
                xanchor='right',
                yanchor='bottom',
                text=corr_text,
                showarrow=False,
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
                borderpad=2,
                align="right",
                row=row,
                col=col
            )
        
        # Add to subplot
        for trace in scatter_fig.data:
            self.figure.add_trace(trace, row=row, col=col)
        
        # Hide colorbar for all but first plot
        if not (trace_idx == 0):
            self.figure.update_traces(showscale=False, selector=dict(row=row, col=col))
            self.figure.update_traces(showlegend=False)
        else:
            self.figure.update_traces(
                colorbar=dict(title=color_title),
                selector=dict(row=row, col=col)
            )
                
        return max_density
    
    def _add_plain_scatter_trace(self, df: pd.DataFrame, col1: str, col2: str,
                               x_title: str, y_title: str, row: int, col: int,
                               corr_text: Optional[str] = None) -> None:
        """Add plain scatter trace to figure using Plotly Express."""
        # Create figure with px.scatter
        scatter_fig = px.scatter(
            df,
            x=col1,
            y=col2,
            render_mode='webgl',
            trendline=self.config.trendline,
            category_orders=self.config.category_orders,
            labels=self.config.labels
        )
        
        # Update marker style
        scatter_fig.update_traces(
            marker=dict(
                line=dict(width=0.5, color='white')
            ),
            hovertemplate=f"{x_title} = %{{x}}<br>{y_title} = %{{y}}<extra></extra>",
            selector=dict(mode='markers')
        )
        
        # Add correlation annotation if provided
        if corr_text:
            self.figure.add_annotation(
                x=0.98,
                y=0.02,
                xref="paper",
                yref="paper",
                text=corr_text,
                showarrow=False,
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='rgba(0,0,0,0.2)',
                align="right",
                borderwidth=1,
                borderpad=2
            )
        # Add to subplot
        for trace in scatter_fig.data:
            self.figure.add_trace(trace, row=row, col=col)

    def _update_axes(self, x_title: str, y_title: str, row: int, col: int) -> None:
        """Update axes properties."""
        self.figure.update_xaxes(
            title_text=x_title,
            title_font=dict(size=12),
            row=row,
            col=col
        )
        self.figure.update_yaxes(
            title_text=y_title,
            title_font=dict(size=12),
            row=row,
            col=col
        )
    
    def _finalize_figure_layout(self, max_density: Optional[float] = None) -> None:
        """Finalize figure layout."""
        max_density = max_density or self.max_density
        if not self.config.title:
            title = "Variable Relationships"
            if self.config.display_mode == 'density_contour':
                title += " (Density Contours)"
            elif self.config.color_mode:
                if self.config.color_mode == 'category':
                    title += f" (Colored by {self.config.color_column.title()})"
                else:
                    title += f" (Colored by {self.config.color_mode.title()} Density)"
        else:
            title = self.config.title 
        self.figure.update_yaxes(
            title_standoff=5
        )
        self.figure.update_xaxes(
            title_standoff=5
        )
        layout_updates = dict(
            height=self.config.height,
            width=self.config.width,
            # margin=dict(l=50, r=50, t=80, b=50),
            title_text=title,
            title_y=0.97,
            plot_bgcolor=self.config.plot_bgcolor,
        )     
        if self.config.color_mode == 'category':
            if not self.config.show_legend_title:
                layout_updates.update({
                    'legend_title_text': '' ,
                    'margin': dict(l=50, r=50, t=80, b=50)
                })    
            self.figure.update_layout(
                legend=dict(
                    title=dict(text=self.config.color_column),
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5,
                    itemsizing="constant",
                    # bordercolor='rgba(0,0,0,0.1)',
                    # borderwidth=1,
                )        
            )    
            self.figure.update_traces(showlegend=True, row=1, col=1)                             
        if self.config.color_mode in ['count', 'kde']:
            self.figure.update_layout(coloraxis1=self.coloraxis)
            self.figure.update_layout(coloraxis1_colorbar_title_text = 'Density')
            # self.figure.update_layout(
            #     coloraxis1_colorbar_tickvals = [max_density * i / 10 for i in range(10)],
            #     coloraxis1_colorbar_ticktext=[f'{i * 10}%' for i in range(10)]
            # )
        if self.config.color_mode in ['kde']:
            for trace in self.figure.data:
                trace.hovertemplate = trace.hovertemplate.replace('marker.color', 'marker.color:.3f')
        self.figure.update_layout(legend_tracegroupgap=0)
        self.figure = CustomFigure(self.figure).update(**layout_updates)
    