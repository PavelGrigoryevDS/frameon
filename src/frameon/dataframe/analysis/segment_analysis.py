import pandas as pd
import numpy as np
import plotly.express as px
from typing import Union, Dict, Optional, Any, List, Literal, Tuple, Callable, TYPE_CHECKING
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import get_type_hints
import plotly.io as pio
from frameon.utils.plotting import CustomFigure
from IPython.display import display

if TYPE_CHECKING: # pragma: no cover
    from frameon.core.base import FrameOn

__all__ = ['SegmentAnalyzer']

class SegmentAnalyzer:
    def __init__(self, df: "FrameOn"):
        """Initialize with the main dataframe."""
        self._df = df
    
    def segment_polar(
        self,
        metrics: List[str],
        dimension: str,
        count_column: str,
        normalize_metric: bool = True,
        normalize_counts: bool = True, 
        text_auto: Union[bool, str] = True,  
        labels: Optional[dict] = None,  
        title: Optional[str] = None,
        agg_func: str = 'median',
        exclude_segments: Optional[List[str]] = None,
        include_segments: Optional[List[Any]] = None, 
        max_segments: Optional[int] = None, 
        width: int = 1000,
        height: int = 400,
        horizontal_spacing: Optional[float] = None,
    ) -> CustomFigure:
        """
        Creates a combined visualization showing segment distribution (left) and metric profiles (right).
        
        Parameters:
        -----------
        metrics : List[str]
            List of numeric columns to analyze
        dimension : str
            Column name used for segmentation
        count_column : str
            Column name to count unique values for segment sizes
        normalize_metric : bool, optional (default=True)
            Whether to normalize metrics to [0,1] range
        normalize_counts : bool, optional (default=True)
            Whether to normalize counts to percentages for bar chart
        text_auto : bool or str, optional
            Format string for bar chart labels (e.g. '.1%', '.3s', '.0f'). 
            Defaults to '.1%' when normalize_counts=True, '.3s' otherwise
        labels : Dict[str, str], optional
            Dictionary to rename metrics or dimension
            Format: {'original_name': 'display_name'}
        title : str, optional
            Main title for the entire visualization
        agg_func : str, optional (default='median')
            Aggregation function ('median', 'mean', 'sum', etc.)
        exclude_segments : List[str], optional
            List of segment values to exclude from analysis (e.g. ['Unknown', 'Other']). 
            These values will be removed from the specified dimension before processing.
            If None, all segments will be included. By default None
        include_segments : List[Any], optional
            Explicit list of segment values to include in analysis.
            If specified, only these segments will be analyzed.
            Mutually exclusive with exclude_segments. By default None
        max_segments : int, optional
            Maximum number of segments to display, ranked by segment size.
            If None, shows all segments. By default None            
        width : int, optional (default=1200)
            Figure width in pixels
        height : int, optional (default=600)
            Figure height in pixels
        horizontal_spacing: float
                Space between subplot columns in normalized plot coordinates. Must be a float between 0 and 1.

        Returns
        -------
        CustomFigure
            Interactive Plotly figure object          
        """
        df = self._df.copy()
        # Validate DataFrame is not empty
        if df.empty:
            raise ValueError(
                "Input DataFrame is empty."
            )           
        df[dimension] = df[dimension].astype(str).astype('category')
        if include_segments and exclude_segments:
            raise ValueError("Cannot use both include_segments and exclude_segments")
        # Validate input columns
        missing_columns = [
            col for col in [dimension, count_column] + metrics 
            if col not in self._df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Columns not found in DataFrame: {missing_columns}. "
                f"Available columns: {list(self._df.columns)}"
            )

        # Validate metrics list
        if not metrics:
            raise ValueError("Metrics list cannot be empty")

        # Validate dimension column has data
        if self._df[dimension].nunique() == 0:
            raise ValueError(f"Dimension column '{dimension}' contains no values")

        # Validate horizontal_spacing range
        if horizontal_spacing is not None and not 0 <= horizontal_spacing <= 1:
            raise ValueError("horizontal_spacing must be between 0 and 1")

        # Validate max_segments
        if max_segments is not None and max_segments <= 0:
            raise ValueError("max_segments must be positive integer")
  
        # 1. Prepare data
        columns_list = list(set([dimension, count_column] + metrics))
        plot_data = df[columns_list].copy()
        if include_segments:
            plot_data = plot_data[plot_data[dimension].isin(include_segments)]
        elif exclude_segments:
            plot_data = plot_data[~plot_data[dimension].isin(exclude_segments)]
        if plot_data.empty:
            raise ValueError("No data remaining after segment filtering")
        if max_segments is not None:
            top_segments = (
                plot_data.groupby(dimension, observed=False)[count_column]
                .count()
                .nlargest(max_segments)
                .index.tolist()
            )
            plot_data = plot_data[plot_data[dimension].isin(top_segments)]
        # 2. Calculate segment sizes
        segment_counts = plot_data.groupby(dimension, observed=True)[count_column].count().reset_index()
        segment_counts.columns = [dimension, 'Count']
        if normalize_counts:
            total = self._df[count_column].count()
            segment_counts['% of Total Count'] = segment_counts['Count'] / total
            display_col = '% of Total Count'
            display_col_format= '.1%'
        else:
            display_col = 'Count'
            display_col_format= '.3s'
        if text_auto and isinstance(text_auto, bool):
            text_auto = display_col_format
        elif text_auto and isinstance(text_auto, str):
            display_col_format = text_auto
        category_orders = {dimension: segment_counts.sort_values('Count', ascending=False)[dimension].tolist()}

        # 3. Aggregate metrics
        if agg_func in ['median', 'mean', 'sum', 'min', 'max']:
            segment_stats = plot_data.groupby(dimension, observed=True)[metrics].agg(agg_func)
        else:
            segment_stats = plot_data.groupby(dimension, observed=True)[metrics].apply(agg_func)
        
        # 4. Normalize if requested
        if normalize_metric:
            for metric in metrics:
                series = segment_stats[metric]
                
                if series.count() == 1:
                    segment_stats[metric] = 0.5
                    continue
                    
                if series.min() >= 0:
                    max_val = series.max()
                    segment_stats[metric] = 0.5 if max_val == 0 else series / max_val
                elif series.max() <= 0:
                    min_val = series.min()
                    segment_stats[metric] = 0.5 if min_val == 0 else 1 - (series / min_val)
                else:
                    pos_mask = series >= 0
                    neg_mask = ~pos_mask
                    result = series.copy()
                    
                    if pos_mask.any():
                        pos_vals = series[pos_mask]
                        result[pos_mask] = 0.5 + (pos_vals / (2 * pos_vals.max()))
                    
                    if neg_mask.any():
                        neg_vals = series[neg_mask]
                        result[neg_mask] = 0.5 * (1 - (neg_vals / neg_vals.min()))
                    
                    segment_stats[metric] = result
        
        # 5. Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.3, 0.7],
            specs=[[{"type": "bar"}, {"type": "polar"}]],
            horizontal_spacing=horizontal_spacing
        )

        # 6. Add count bar chart

        bar_fig = px.bar(
            segment_counts,
            x=display_col,
            y=dimension,
            color=dimension,
            orientation='h',
            category_orders=category_orders,
            labels=labels,
            hover_data={display_col: f':{display_col_format}'},
            text_auto=text_auto
        )
        
        for trace in bar_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # 7. Add polar chart
        polar_data = segment_stats.reset_index().melt(
            id_vars=[dimension],
            var_name='Metric',
            value_name='Value'
        )
        if labels:
            polar_data['Metric'] = polar_data['Metric'].replace(labels)
            
        polar_fig = px.line_polar(
            polar_data,
            r='Value',
            theta='Metric',
            color=dimension,
            line_close=True,
            category_orders=category_orders,
        )
        polar_fig.update_traces(showlegend=False)
        for trace in polar_fig.data:
            fig.add_trace(trace, row=1, col=2)
        fig.update_yaxes(
            autorange="reversed"
            , row=1, col=1
        )
        is_text_auto_false = text_auto == False
        fig.update_xaxes(
            title='% of Total Count' if normalize_counts else 'Count'
            , showticklabels=is_text_auto_false
            , showline=is_text_auto_false
            , ticks='' if not is_text_auto_false else None
            , showgrid=is_text_auto_false
            , tickformat='.0%' if normalize_counts else None
            , title_standoff = 5 if not is_text_auto_false else None
            , row=1, col=1
        )
        # 8. Update layout
        dimension_title = dimension if (not labels or dimension not in labels) else labels[dimension]
        fig.update_layout(
            title_text=title or f'Segment Analysis',
            title_y=0.96,
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                title=dimension_title,
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5
            ),
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1.1]),
                angularaxis=dict(direction='clockwise', rotation=90)
            ),
            bargap=0.3,
            margin=dict(l=50, r=50, t=90, b=50)
        )
        
        # 9. Update traces

        fig.update_traces(
            selector=dict(type='scatterpolar'),
            fill='toself',
            hovertemplate='<b>%{theta}</b>: %{r:.2f}<extra></extra>'
        )
        
        return CustomFigure(fig)  
    
    def segment_table(
        self,
        metrics: List[str],
        dimension: str,
        count_column: str,
        exclude_segments: Optional[List[str]] = None,
        include_segments: Optional[List[Any]] = None, 
        max_segments: Optional[int] = None, 
        agg_func: Union[str, Callable] = 'median',
        na_rep: str = 'NaN'
    ) -> None:
        """
        Create a gradient-styled table for segment analysis with client shares and aggregated metrics.
        
        The first row shows the percentage of clients in each segment, followed by aggregated metrics.
        
        Parameters
        ----------
        metrics : List[str]
            List of metric columns to aggregate and display
        dimension : str
            Column name used for segmentation (grouping)
        count_column : str
            Column with unique client identifiers for share calculation
        agg_func : Union[str, Callable], optional
            Aggregation function ('median', 'mean', etc.) or percentile in format 'pXX' (e.g., 'p25'), 
            by default 'median'
        na_rep : str, optional
            Representation for NaN values, by default 'NaN'
        exclude_segments : List[str], optional
            List of segment values to exclude from analysis (e.g. ['Unknown', 'Other']). 
            These values will be removed from the specified dimension before processing.
            If None, all segments will be included. By default None
        include_segments : List[Any], optional
            Explicit list of segment values to include in analysis.
            If specified, only these segments will be analyzed.
            Mutually exclusive with exclude_segments. By default None
        max_segments : int, optional
            Maximum number of segments to display, ranked by segment size.
            If None, shows all segments. By default None    
        Returns
        -------
            None                        
        """
        df = self._df.copy()
        
        # Apply segment filters
        if include_segments and exclude_segments:
            raise ValueError("Cannot use both include_segments and exclude_segments")
        
        if include_segments:
            df = df[df[dimension].isin(include_segments)]
        elif exclude_segments:
            df = df[~df[dimension].isin(exclude_segments)]        
        # Calculate client shares per segment
        segment_counts = df.groupby(dimension, observed=True)[count_column].count()

        if max_segments is not None:
            segment_counts = segment_counts.nlargest(max_segments)
            df = df[df[dimension].isin(segment_counts.index)]
        total_clients = self._df[count_column].count()
        segment_percentage = (segment_counts / total_clients).round(4)
        
        # Initialize result with client percentage
        result = pd.DataFrame([segment_percentage], index=['% of Total Count'])
        
        # Aggregate metrics based on specified function
        if isinstance(agg_func, str) and agg_func.startswith('p'):
            try:
                percentile = float(agg_func[1:]) / 100
                agg_df = df.groupby(dimension, observed=True)[metrics].quantile(percentile)
            except ValueError:
                raise ValueError("Percentile should be in format 'pXX' (e.g. 'p25')")
        else:
            agg_df = df.groupby(dimension, observed=True)[metrics].agg(agg_func)
        # Sorting segments
        sort_df = agg_df.copy()
        # Normalize metrics only to calculate the sorting
        for metric in metrics:
            series = sort_df[metric]
            if series.nunique() > 1:
                if series.min() >= 0:
                    max_val = series.max()
                    sort_df[metric] = series / max_val if max_val != 0 else 0.5
                else:
                    # For negative values
                    min_val, max_val = series.min(), series.max()
                    sort_df[metric] = (series - min_val) / (max_val - min_val)
        index_order = sort_df.sum(axis=1).sort_values(ascending=False).index
        agg_df = agg_df.loc[index_order].T
        result = result[index_order]
        
        result = pd.concat([result, agg_df])
        
        # Create formatted copy for display
        formatted_df = result.copy().T
        
        # Format share as percentage and metrics as floats
        formatted_df['% of Total Count'] = formatted_df['% of Total Count'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else na_rep)
        formatted_df[metrics] = formatted_df[metrics].map(lambda x: f"{x:.2f}" if pd.notna(x) else na_rep)
        formatted_df = formatted_df.T
               
        # Apply styling to the dataframe
        title = f"Segment Analysis for {dimension.replace('_', ' ').title()}" 
        styled_df = (
            formatted_df.style
            .set_caption(title) 
            .apply(self._gradient_row, axis=1)
            .set_properties(**{
                'border': '1px solid black',
                'text-align': 'center',
                'padding': '5px',
                'font-size': '12px'
            })
            .set_table_styles([
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", f"16px"),
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                        ("white-space", "nowrap"),  # Prevent caption from wrapping
                    ],
                },
                {
                    'selector': 'th',
                    'props': [('border', '1px solid black')]
                }])
        )
        
        display(styled_df)
      
    def _gradient_row(self, row):
        """Apply gradient styling to row based on numeric values"""
        # Gradient coloring function (works with original numeric values)
        def plotly_greens_gradient(val, min_val, max_val):
            """Convert value to color from green gradient"""
            if pd.isna(val):
                return "rgb(255,255,255)"
            
            try:
                normalized = (val - min_val) / (max_val - min_val) if max_val != min_val else 0
                r = int(247 - (247 - 0) * normalized)
                g = int(252 - (252 - 68) * normalized)
                b = int(245 - (245 - 27) * normalized)
                return f"rgb({r},{g},{b})"
            except (ValueError, TypeError):
                return "rgb(255,255,255)"
        numeric_values = pd.to_numeric(row.str.replace('%', ''), errors='coerce')
        valid_vals = numeric_values[~numeric_values.isna()]
        
        if len(valid_vals) == 0:
            return [''] * len(row)
            
        min_val, max_val = valid_vals.min(), valid_vals.max()
        
        styles = []
        for val in numeric_values:
            if pd.isna(val):
                styles.append('background-color: white; color: black')
            else:
                bg_color = plotly_greens_gradient(val, min_val, max_val)
                r, g, b = map(int, bg_color[4:-1].split(','))
                brightness = 0.299*r + 0.587*g + 0.114*b
                text_color = 'white' if brightness < 128 else 'black'
                styles.append(f'background-color: {bg_color}; color: {text_color}')
        return styles

    def metric_by_dimensions_table(
        self,
        metric: str,
        dimensions: List[str],
        agg_func: Union[str, Callable] = 'median',
        na_rep: str = 'NaN'
    ) -> None:
        """
        Create styled tables showing one metric across multiple dimensions.
        
        Parameters
        ----------
        metric : str
            Metric column to analyze
        dimensions : List[str]
            List of dimension columns for segmentation
        agg_func : Union[str, Callable], optional
            Aggregation function, by default 'median'
        na_rep : str, optional
            Representation for NaN values, by default 'NaN'
            
        Returns
        -------
            None
        """
        df = self._df
        styled_tables = []
        
        # Main title for all tables
        main_title = f"Analysis of '{metric}' across dimensions"
        print(main_title + "\n" + "="*len(main_title))
        
        for dimension in dimensions:
            # Aggregate data
            agg_df = df.groupby(dimension, observed=True)[metric].agg(agg_func)
            
            # Sort by metric value (descending)
            agg_df = agg_df.sort_values(ascending=False)
            
            # Convert to DataFrame for styling
            result = pd.DataFrame(agg_df)
            # Format values
            formatted_df = result.copy()
            formatted_df[metric] = formatted_df[metric].apply(lambda x: f"{x:.2f}" if pd.notna(x) else na_rep)
            formatted_df = formatted_df.T
            # Gradient coloring function
            def gradient_columns(col):
                numeric_values = pd.to_numeric(col.str.replace(na_rep, 'NaN'), errors='coerce')
                valid_vals = numeric_values[~numeric_values.isna()]
                
                if len(valid_vals) == 0:
                    return [''] * len(col)
                    
                min_val, max_val = valid_vals.min(), valid_vals.max()
                
                styles = []
                for val in numeric_values:
                    if pd.isna(val):
                        styles.append('background-color: white; color: black')
                    else:
                        # Greens gradient
                        normalized = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                        r = int(247 - (247 - 0) * normalized)
                        g = int(252 - (252 - 68) * normalized)
                        b = int(245 - (245 - 27) * normalized)
                        brightness = 0.299*r + 0.587*g + 0.114*b
                        text_color = 'white' if brightness < 128 else 'black'
                        styles.append(f'background-color: rgb({r},{g},{b}); color: {text_color}')
                return styles
            
            # Apply styling
            styled_table = (
                formatted_df.style
                .set_caption(f"By {dimension.replace('_', ' ').title()}")
                .apply(gradient_columns, axis=1)
                .hide(axis='index')
                .set_properties(**{
                    'border': '1px solid black',
                    'text-align': 'center',
                    'padding': '5px',
                    'font-size': '12px'
                })
                .set_table_styles([
                    {'selector': 'caption', 
                    'props': [
                        ("font-size", f"16px"),
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                        ("white-space", "nowrap"),  # Prevent caption from wrapping
                    ]},
                    {'selector': 'th, td',
                    'props': [('border', '1px solid black')]}
                ])
            )
            display(styled_table)
            
    def metric_by_dimensions_plot(
        self,
        metric: str,
        dimensions: List[str],
        color: Optional[str] = None,
        animation_frame: Optional[str] = None,
        facet_col_wrap: int = 3,
        facet_row_spacing: float = 0.1,
        facet_col_spacing: float = 0.05,
        labels: dict = None,
        category_orders: dict = None,
        agg_func: str = 'mean',
        sort_bars: bool = True,
        **plotly_kwargs
    ) -> CustomFigure:
        """
        Creates faceted horizontal bar charts for a metric across multiple dimensions.
 
        Features:
        
        - Each dimension in separate subplot
        - Bars sorted by metric value (descending)
        - Optional color and animation dimensions
        - Customizable facet grid layout

        Parameters
        ----------
        metric : str
            Numeric column to visualize
        dimensions : List[str]
            Columns for segmentation (each becomes a subplot)
        color : str, optional
            Column for color encoding
        animation_frame : str, optional
            Column for animation frames
        facet_col_wrap : int, optional (default=3)
            Max number of subplots per row
        facet_row_spacing : float, optional (default=0.1)
            Space between facet rows (0-1)
        facet_col_spacing : float, optional (default=0.05)
            Space between facet columns (0-1)
        labels : dict, optional
            Dictionary mapping column names to display names
        category_orders : dict, optional
            Specifies order of categories for dimensions
        agg_func : str, optional (default='mean')
            Aggregation function
        sort_bars : bool, optional (default=True)
            Whether to sort bars by value
        plotly_kwargs : dict, optional
            Additional arguments for Plotly Express

        Returns
        -------
        CustomFigure
            Interactive Plotly figure object

        Notes
        -----
        - If dimension has label in labels, category_orders must use the new label
        - plotly_kwargs are passed directly to px.bar()       
        """
        df = self._df
        if labels is not None:
            # Replace the names of Dimensions from the transmitted dictionary
            dimensions_for_rename = {dim: labels.get(dim, dim) for dim in dimensions}
        # Prepare data with proper sorting
        frames = []
        for dim in dimensions:
            group_cols = [dim, color, animation_frame] if color or animation_frame else [dim]
            group_cols = [col for col in group_cols if col is not None]
            
            dim_data = df.groupby(group_cols)[metric].agg(agg_func).reset_index()
            dim_data['Dimension'] = dimensions_for_rename.get(dim, dim) if labels else dim
            dim_data = dim_data.rename(columns={dim: 'Segment'})
            
            # # Pre-sort data if requested
            if sort_bars:
                if color:
                    # Sort by aggregate metric ignoring color groups
                    sort_order = dim_data.groupby('Segment')[metric].mean().sort_values(ascending=False).index
                    dim_data['Segment'] = pd.Categorical(dim_data['Segment'], categories=sort_order, ordered=True)
                    dim_data = dim_data.sort_values('Segment')
                else:
                    dim_data = dim_data.sort_values(metric, ascending=True)
            
            frames.append(dim_data)
        
        plot_data = pd.concat(frames)
        if 'title' not in plotly_kwargs:
            metric_label = labels.get(metric, metric) if labels else metric
            plotly_kwargs['title']=f"{metric_label} by Dimensions"
        # Create figure with all options
        fig = px.bar(
            plot_data,
            x=metric,
            y='Segment',
            color=color,
            animation_frame=animation_frame,
            facet_col='Dimension',
            facet_col_wrap=facet_col_wrap,
            facet_row_spacing=facet_row_spacing,
            facet_col_spacing=facet_col_spacing,
            labels=labels,
            barmode='group',
            category_orders=category_orders,
            **plotly_kwargs
        )
        
        fig.update_xaxes(showgrid=False) 
        # Axis and spacing controls
        fig.update_yaxes(matches=None, showticklabels=True, categoryorder='total ascending')
        fig.update_layout(
            title_y = 0.97 if color else None
            , margin=dict(t=80) if color else dict(t=70)
            , legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5
            ),
        )
        
        return CustomFigure(fig)