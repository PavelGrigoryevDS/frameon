from typing import Union, Optional, Literal, Dict, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .custom_figure import CustomFigure
from frameon.utils.miscellaneous import style_dataframe, is_categorical_column
from scipy.stats import t
from IPython.display import display
import warnings

def create_plot_ci(
    data_frame: pd.DataFrame,
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
    plotly_kwargs: dict = None,
) -> CustomFigure:
    """
    Creates a plot with mean values and confidence intervals using t-statistics.
    """
    DEFAULT_WIDTH = 800
    CATEGORY_WIDTH = 150
    GROUP_WIDTH = 50
    df = data_frame
    # 1. Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")
    
    required_columns = [cat_col, num_col]
    if color:
        required_columns.append(color)
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    
    if orientation not in ['v', 'h']:
        raise ValueError("orientation must be either 'v' (vertical) or 'h' (horizontal)")
    
    if legend_position not in ['top', 'right', 'bottom', 'left']:
        raise ValueError("legend_position must be one of: 'top', 'right', 'bottom', 'left'")
    
    # 2. Data processing
    group_cols = [cat_col] + ([color] if color else [])
    
    if df[group_cols].isna().any().any():
        warnings.warn("Grouping columns contain NaN values which will be excluded", UserWarning)
        df = df.dropna(subset=group_cols)
    
    summary_df = df.groupby(group_cols, observed=False)[num_col].agg(["mean", "std", "count"]).reset_index()
    
    small_groups = summary_df[summary_df["count"] < min_group_size]
    if not small_groups.empty:
        warnings.warn(
            f"Some groups have fewer than {min_group_size} observations:\n"
            f"{small_groups[group_cols + ['count']].to_string(index=False)}\n"
            "Confidence intervals for these groups may be unreliable.",
            UserWarning
        )
    
    degrees_of_freedom = summary_df["count"] - 1
    t_score = t.ppf(1 - alpha / 2, degrees_of_freedom)
    
    summary_df["ci"] = t_score * summary_df["std"] / (summary_df["count"] ** 0.5)
    summary_df["mean"] = summary_df["mean"].round(2)
    summary_df["ci"] = summary_df["ci"].round(2)
    
    if show_summary:
        display(style_dataframe(
            summary_df,
            caption='Summary statistics',
            hide_columns=False,
            formatters={'mean': '{:.2f}', 'std': '{:.2f}', 'ci': '{:.2f}'}
        ))
    
    # 3. Position calculation for grouped points
    if color:
        # Get unique categories
        main_categories = summary_df[cat_col].unique()
        sub_categories = summary_df[color].unique()
        
        # Create mapping for main category positions
        main_positions = {cat: idx for idx, cat in enumerate(main_categories)}
        
        # Create offsets for subcategories within each main category
        sub_offsets = np.linspace(-group_spacing/2, group_spacing/2, len(sub_categories))
        sub_offset_map = {cat: offset for cat, offset in zip(sub_categories, sub_offsets)}
        
        # Calculate final positions
        summary_df['position'] = summary_df.apply(
            lambda row: main_positions[row[cat_col]] + sub_offset_map[row[color]],
            axis=1
        )
        
        # Set axis configuration
        if orientation == 'v':
            x_col = 'position'
            y_col = "mean"
            error_x = None
            error_y = "ci"
        else:
            x_col = "mean"
            y_col = 'position'
            error_x = "ci"
            error_y = None
    else:
        # No secondary category - simple positioning
        if orientation == 'v':
            x_col = cat_col
            y_col = "mean"
            error_x = None
            error_y = "ci"
        else:
            x_col = "mean"
            y_col = cat_col
            error_x = "ci"
            error_y = None
    
    # 4. Create figure with proper hover data
    hover_data = {
        cat_col: True,
        'mean': ':.2f',
        'ci': ':.2f'
    }
    if color:
        hover_data[color] = True
        hover_data['position'] = False
    
    if not labels:
        labels = {}
    labels.setdefault('mean', 'Mean')
    labels.setdefault('ci', f'{int((1-alpha)*100)}% CI')

    if color:
        n_main_cats = len(summary_df[cat_col].unique())
        n_sub_cats = len(summary_df[color].unique())
        
        # Width calculation: basic + 150 for each main category 
        # + 50 if there is a subcategoria
        width = max(DEFAULT_WIDTH, n_main_cats * CATEGORY_WIDTH)
        if n_sub_cats > 1:
            width = max(width, n_main_cats * n_sub_cats * GROUP_WIDTH)
    else:
        n_cats = len(summary_df[cat_col].unique())
        width = max(DEFAULT_WIDTH, n_cats * CATEGORY_WIDTH)
    if width > 1200:
        width = None
    plotly_kwargs.setdefault('width', width)

    fig = px.scatter(
        summary_df,
        x=x_col,
        y=y_col,
        color=color if color else None,
        error_y=error_y,
        error_x=error_x,
        title=title,
        labels=labels,
        category_orders=category_orders,
        hover_data=hover_data,
        **plotly_kwargs
    )
    
    # Update marker and error bar appearance
    fig.update_traces(
        marker=dict(size=marker_size),
        error_x=dict(width=error_bar_width),
        error_y=dict(width=error_bar_width)
    )
    
    # 5. Adjust axes for grouped points
    if color:
        tickvals = list(main_positions.values())
        ticktext = list(main_positions.keys())
        
        if orientation == 'v':
            fig.update_xaxes(
                tickvals=tickvals,
                ticktext=ticktext,
                title=labels.get(cat_col, cat_col))
        else:
            fig.update_yaxes(
                tickvals=tickvals,
                ticktext=ticktext,
                title=labels.get(cat_col, cat_col))
    
    # 6. Add annotations if needed
    if show_annotations:
        annotations = []
        for i, row in summary_df.iterrows():
            text = f"{row['mean']:{annotation_format}} ± {row['ci']:{annotation_format}}"
            if orientation == 'v':
                annotations.append(dict(
                    x=row[x_col],
                    y=row[y_col] + row['ci'],
                    text=text,
                    showarrow=False,
                    yshift=10,
                    xanchor='center'
                ))
            else:
                annotations.append(dict(
                    x=row[x_col] + row['ci'],
                    y=row[y_col],
                    text=text,
                    showarrow=False,
                    xshift=10,
                    yanchor='middle'
                ))
        fig.update_layout(annotations=annotations)

    fig_update_config = {}
    # Set legend properties
    if color:
        fig_update_config['legend_position'] = 'top'
        if show_legend_title == False:
            fig_update_config['legend_title_text'] = ''
    if orientation == 'v':
        fig_update_config['xaxis_showgrid'] = False
        fig.update_xaxes(zeroline=False)
    else:
        fig_update_config['yaxis_showgrid'] = False
        fig.update_yaxes(zeroline=False)
    # Apply updates
    fig = CustomFigure(fig)
    fig = fig.update(**fig_update_config)    
    return fig
