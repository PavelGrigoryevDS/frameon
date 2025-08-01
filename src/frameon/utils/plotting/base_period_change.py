import pandas as pd
import numpy as np
import plotly.express as px
from .custom_figure import CustomFigure
from typing import Union, List, Optional, Callable, Dict, Any

def period_change(
    df: pd.DataFrame,
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
    plotly_kwargs: Optional[Dict[str, Any]] = None
) -> CustomFigure:
    """
    Plot period-over-period changes for a given metric using pd.Grouper with enhanced customization.
    """
    labels = plotly_kwargs.pop('labels', {})
    hover_data = plotly_kwargs.pop('hover_data', {})
    # Validate parameters
    valid_periods = ['mom', 'wow', 'dod', 'yoy']
    if period not in valid_periods:
        raise ValueError(f"period must be one of {valid_periods}")
    
    # Set default colors if not provided
    if color_dict is None:
        color_dict = {
            'positive': '#57B16E',
            'negative': '#E25559'
        }
    
    # Convert to lists if single values passed
    group_cols = []
    for col in [facet_col, facet_row, animation_frame]:
        if col is not None and col not in group_cols:
            group_cols.append(col)
    
    # Validate all columns exist
    for col in [metric_col, date_col] + group_cols:
        if col not in df.columns:
            raise ValueError(f'Column "{col}" not found in dataframe')
    
    # Make sure date_col is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Determine frequency for pd.Grouper
    freq_map = {
        'mom': 'ME',
        'wow': 'W',
        'dod': 'D',
        'yoy': 'YE'
    }
    freq = freq_map[period]
    change_label_map = {
        'mom': 'Monthly Change',
        'wow': 'Weekly Change',
        'dod': 'Daily Change',  
        'yoy': 'Yearly Change'
    }
    change_col_label = change_label_map[period]
    # Group by period and other dimensions
    grouper = pd.Grouper(key=date_col, freq=freq)
    group_cols_with_grouper = [grouper] + group_cols
    
    # Aggregate data
    df_agg = df.groupby(group_cols_with_grouper, as_index=False)[metric_col].agg(agg_func)
    
    # Rename the grouped date column
    df_agg = df_agg.rename(columns={date_col: 'period'})
    
    # Create full index to ensure all periods are represented
    min_date = df_agg['period'].min()
    max_date = df_agg['period'].max()
    
    if period == 'wow':
        # For weekly grouping, we need to adjust to week start
        date_range = pd.date_range(min_date - pd.Timedelta(days=min_date.weekday()), 
                               max_date, 
                               freq=freq)
    else:
        date_range = pd.date_range(min_date, max_date, freq=freq)
    
    if group_cols:
        # Case with grouping columns
        full_index = pd.MultiIndex.from_product(
            [date_range] + [df_agg[col].unique() for col in group_cols],
            names=['period'] + group_cols
        )
    else:
        # Case with only date column
        full_index = date_range
        full_index.name = 'period'
    
    # Reindex to the full index
    df_agg = df_agg.set_index(['period'] + group_cols).reindex(full_index, fill_value=fill_value).reset_index()
    
    # Calculate change
    change_col = f'{metric_col}_change'
    if group_cols:
        df_agg[change_col] = df_agg.groupby(group_cols)[metric_col].pct_change()
    else:
        df_agg[change_col] = df_agg[metric_col].pct_change()
    
    # Add color column based on positive/negative values
    df_agg['change_direction'] = np.where(df_agg[change_col] >= 0, 'positive', 'negative')
    
    # Prepare default labels
    default_labels = {
        change_col: change_col_label,
        'change_direction': 'Change Direction'
    }
    if date_col not in plotly_kwargs:
        default_labels['period'] = 'Date'
    # Update default labels with user-provided labels
    if labels:
        default_labels.update(labels)
    if 'title' not in plotly_kwargs:
        plotly_kwargs['title'] = change_col_label
    if 'width' not in plotly_kwargs:
        plotly_kwargs['width'] = 800
    if 'height' not in plotly_kwargs:
        plotly_kwargs['height'] = 400
    # Convert datetime to string for better plotly static rendering
    df_agg['period'] = df_agg['period'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # Create plot with all parameters
    fig = px.bar(
        df_agg,
        x='period',
        y=change_col,
        color='change_direction',
        color_discrete_map=color_dict,
        facet_col=facet_col,
        facet_col_wrap=facet_col_wrap,
        facet_row=facet_row,
        hover_data={**hover_data, metric_col: False, 'change_direction': False, change_col: ':.1%'},
        animation_frame=animation_frame,
        labels=default_labels,
        **plotly_kwargs
    )
    
    # Add zero line and improve layout
    fig.update_layout(
        yaxis_zeroline=True,
        yaxis_tickformat='.0%',
        showlegend=False
    )
    
    return CustomFigure(fig)