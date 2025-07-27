from typing import Union, Optional, Literal, Dict, Callable, List, Any, get_type_hints
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .custom_figure import CustomFigure

def parallel_categories(
    data_frame: pd.DataFrame,
    dimensions: list,
    color_mapping: dict = None,
    top_n_categories: dict = None,
    title: str = None,
    margin_l: int = 50,
    margin_r: int = 50,
    width: int = None,
    height: int = None,
    plotly_kwargs: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Create a parallel categories plot with customizable colors and category filtering.
    
    """
    if 'category_orders' in plotly_kwargs:
        plotly_kwargs.pop('category_orders')
    df = data_frame
    # Default color palette (Plotly's discrete colors)
    default_colors = [
        "#4C78A8", "#FF9E4A", "#57B16E", "#E25559", "#8B6BB7",
        "#A17C6B", "#E377C2", "#7F7F7F", "#B5BD4E", "#5BB0D9"
    ]
    
    # Filter data for top N categories if specified
    if top_n_categories:
        df_filtered = df.copy()
        for dim, n in top_n_categories.items():
            if dim in dimensions:
                # Get top n categories using pandas' value_counts (faster than Counter)
                top_cats = df[dim].value_counts().nlargest(n).index
                df_filtered = df_filtered[df_filtered[dim].isin(top_cats)]
    else:
        df_filtered = df.copy()
    
    # Prepare color mapping
    first_dim = dimensions[0]
    if color_mapping:
        # Create numerical codes for categories
        categories = df_filtered[first_dim].value_counts().index.tolist()
        color_scale = list(color_mapping.values())[:len(categories)]
        
        # Map categories to numerical codes (faster than using Counter)
        df_filtered['__color_code__'] = df_filtered[first_dim].map(
            {cat: idx for idx, cat in enumerate(categories)}
        )
    else:
        # Use default colors if no mapping provided
        categories = df_filtered[first_dim].value_counts().index.tolist()
        color_scale = default_colors[:len(categories)]
        df_filtered['__color_code__'] = df_filtered[first_dim].map(
            {cat: idx for idx, cat in enumerate(categories)}
        )
    
    # Create the plot
    fig = px.parallel_categories(
        df_filtered,
        dimensions=dimensions,
        color='__color_code__' if (color_mapping or len(categories) > 1) else None,
        color_continuous_scale=color_scale if (color_mapping or len(categories) > 1) else None,
        width=width if width else 1000,
        height=height if height else None,
        title=title if title else 'Parallel Categories',
        **plotly_kwargs
    )
    # Set order for dimensions by descending frequency of their categories
    for i, dim in enumerate(dimensions):
        # Get categories sorted by frequency (descending)
        sorted_categories = df_filtered[dim].value_counts().index.tolist()
        
        # Create new dimensions list with updated category order
        new_dimensions = []
        for j, d in enumerate(fig.data[0].dimensions):
            if j == i:  # Only modify the current dimension we're sorting
                new_dim = d
                new_dim['categoryorder'] = 'array'
                new_dim['categoryarray'] = sorted_categories
                new_dimensions.append(new_dim)
            else:
                new_dimensions.append(d)
        
        # Update the dimensions
        fig.update_traces(dimensions=new_dimensions)
        
    # Hide color scale for discrete categories
    if color_mapping or len(categories) > 1:
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(line_colorbar=None)
    fig.update_layout(margin=dict(l=margin_l, r=margin_r))
    
    return CustomFigure(fig)