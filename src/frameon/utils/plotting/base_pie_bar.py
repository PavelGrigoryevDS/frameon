from typing import Union, Optional, Literal,  List, Dict
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from frameon.utils.miscellaneous import is_categorical_column
from .custom_figure import CustomFigure
from .plot_utils import subplots
from frameon.utils.plotting.base_bar_line_area import BarLineAreaBuilder

def create_pie_bar(
    data_frame: pd.DataFrame,
    plotly_kwargs: dict,
    agg_func: Optional[Literal['mean', 'median', 'sum', 'count', 'nunique']] = 'count',
    agg_column: Optional[str] = None,
    category_orders: Optional[Dict[str, Union[str, List[str]]]] = None,
    trim_top_n_x: Optional[int] = None,
    trim_top_n_y: Optional[int] = None,
    trim_top_n_agg_func: Literal['mean', 'median', 'sum', 'count', 'nunique'] = 'count',   
    trim_top_n_direction: Literal['top', 'bottom'] = 'top', 
    norm_by: Optional[Union[str, Literal['all']]] = None,
    sort_by: Optional[str] = None,
    show_group_size: bool = True,
    min_group_size: Optional[int] = None,
    observed_for_groupby: bool = True,
    hole: float = None,
    label_for_others_in_pie: str = 'others',
    pie_textinfo: str = 'percent',
    agg_func_for_pie_others: str = 'sum',
    pull:  Optional[int] = None,
    horizontal_spacing: Optional[float] = None
) -> CustomFigure:
    """
    Creates a bar chart using the Plotly Express library.
    """
    # Validate required parameters
    if 'x' not in plotly_kwargs or 'y' not in plotly_kwargs:
        raise ValueError("Both x and y parameters must be specified")
    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError("data_frame must be a pandas DataFrame.")        
    if agg_func is None:
        raise ValueError("agg_func must be defined")     
    if agg_func_for_pie_others not in ['mean', 'median', 'sum', 'count', 'nunique', 'min', 'max']:
                raise ValueError(
                    f"Invalid agg_func '{agg_func_for_pie_others}'. Must be one of: "
                    "'mean', 'median', 'sum', 'count', 'nunique', 'min', 'max'"
                )      
    for col in ['color', 'animation_frame', 'facet_col', 'facet_row']:
        if col in plotly_kwargs:
            raise ValueError(f'{col} not supported')
    for col in ['x', 'y']:
        if plotly_kwargs.get(col) is not None:
            if not isinstance(plotly_kwargs[col], str):
                raise ValueError(f"Column '{col}' must be string type")     
            if plotly_kwargs[col] not in data_frame.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame. Available columns: {list(data_frame.columns)}")   
    if agg_column:
        if not isinstance(agg_column, str):
            raise ValueError("Column 'agg_column' must be string type")     
        if agg_column not in data_frame.columns:
            raise ValueError(f"Column 'agg_column' not found in the DataFrame. Available columns: {list(data_frame.columns)}")   
    config_for_bar_builder = {
        'trim_top_n_x': trim_top_n_x,
        'trim_top_n_y': trim_top_n_y,
        'trim_top_n_direction': trim_top_n_direction,
        'agg_column': agg_column,
        'agg_func': agg_func,
        'show_group_size': show_group_size,
        'min_group_size': min_group_size,
        'trim_top_n_agg_func': trim_top_n_agg_func,
        'observed_for_groupby': observed_for_groupby,
        'norm_by': norm_by,
        'sort_by': sort_by,
        'plotly_kwargs': plotly_kwargs
    }
    # Determine numeric and categorical columns
    x_col, y_col = plotly_kwargs['x'], plotly_kwargs['y']
    if agg_column:
        num_column = agg_column
        cat_column_axis = y_col if x_col == num_column else x_col
    else:
        if pd.api.types.is_numeric_dtype(data_frame[x_col]):
            num_column, cat_column_axis = x_col, y_col
        else:
            num_column, cat_column_axis = y_col, x_col
    # Adjust label for "others" if showing values
    if pie_textinfo == 'value' and label_for_others_in_pie == 'others':
        label_for_others_in_pie = 'sum others'

    # Create pie chart data
    # Group by categorical column and aggregate
    df_for_pie = (
        data_frame.groupby(cat_column_axis, observed=observed_for_groupby)
        .agg(
            num_column=(num_column, agg_func),
            count_for_show_group_size=(num_column, 'count')
        )
        .rename(columns={'num_column': num_column})
    )
    # Identify the category with maximum value
    max_cat_for_exclude = df_for_pie[num_column].idxmax()
    # Create bar chart figure for remaining categories
    df_for_bar = data_frame[data_frame[cat_column_axis] != max_cat_for_exclude]
    builder = BarLineAreaBuilder('bar')
    bar_fig = builder.build(data_frame=df_for_bar, **config_for_bar_builder)

    # Prepare pie chart data by combining all other categories into "others"
    df_for_pie = (
        df_for_pie.reset_index()
        .assign(new_cat_column=lambda df: df[cat_column_axis].astype(str).where(
            df[cat_column_axis] == max_cat_for_exclude, 
            label_for_others_in_pie
        ))
        .groupby('new_cat_column', observed=observed_for_groupby)
        .agg({
            num_column: agg_func_for_pie_others,
            'count_for_show_group_size': 'sum'
        })
        .reset_index()
        .sort_values(num_column, ascending=False)
    )
    
    # Format group size for display
    df_for_pie['count_for_show_group_size'] = df_for_pie['count_for_show_group_size'].apply(
        lambda x: f'{x}' if x <= 1000 else '>1000'
    )
    custom_data = df_for_pie['count_for_show_group_size']

    # if pie_textinfo == 'value':
    #     df_for_pie[num_column] = df_for_pie[num_column].round(decimal_places)
    # Create pie chart figure
    labels = plotly_kwargs.get('labels', {})
    labels['new_cat_column'] = labels.get(cat_column_axis, cat_column_axis)
    
    pie_fig = px.pie(df_for_pie
                     , values=num_column
                     , names='new_cat_column'
                     , labels=labels
                     , hover_data = {num_column: False} if norm_by == 'all' else {num_column: ':.2f'}
                     , hole=hole
    )
    pie_fig.update_traces(pull=pull) 
    # Update pie chart hover template if showing group size
    if show_group_size == True:
        for trace in pie_fig.data:
            trace.customdata = custom_data
            trace.hovertemplate = trace.hovertemplate.replace('customdata[0]', 'value')
            trace.hovertemplate += '<br>Group size = %{customdata[0]}'
    pie_fig.update_traces(textinfo=pie_textinfo)

    # pie_fig.update_traces(textinfo='value')
    # Combine pie and bar charts into subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type': 'pie'}, {'type': 'bar'}]],
        horizontal_spacing=horizontal_spacing,
    ).add_traces(
        pie_fig.data,
        rows=[1]*len(pie_fig.data),
        cols=[1]*len(pie_fig.data)
    ).add_traces(
        bar_fig.data,
        rows=[1]*len(bar_fig.data),
        cols=[2]*len(bar_fig.data)
    )
    xaxis_title_text= bar_fig.layout.xaxis.title.text 
    xaxis_showgrid= bar_fig.layout.xaxis.showgrid
    yaxis_title_text= bar_fig.layout.yaxis.title.text 
    yaxis_showgrid= bar_fig.layout.yaxis.showgrid
    xaxis_categoryarray = bar_fig.layout['xaxis']['categoryarray']
    xaxis_categoryorder = bar_fig.layout['xaxis']['categoryorder'] 
    yaxis_categoryarray = bar_fig.layout['yaxis']['categoryarray']
    yaxis_categoryorder = bar_fig.layout['yaxis']['categoryorder']     
    fig.update_xaxes(
        row=1,
        col=2,
        showgrid=xaxis_showgrid,
        title_text=xaxis_title_text,
        categoryarray=xaxis_categoryarray,
        categoryorder=xaxis_categoryorder
    )
    fig.update_yaxes(
        row=1,
        col=2,
        title_standoff=5,
        showgrid=yaxis_showgrid,
        title_text=yaxis_title_text,
        categoryarray=yaxis_categoryarray,
        categoryorder=yaxis_categoryorder   
    )    
    # Update layout with pie chart settings
    fig.update_layout(pie_fig.layout)

    # Set figure dimensions and layout
    width = plotly_kwargs.get('width', 900)
    height = plotly_kwargs.get('height', 400)
    fig.update_layout(legend_y=-0.1
                      , legend_x=0.11
                      , title_y=0.96
                      , margin = dict(l=50, r=50, b=50, t=70)
                      , legend_orientation='h'
                      , width=width
                      , height=height
                      , title_text = plotly_kwargs.get('title')
    )
    fig = CustomFigure(fig)
    fig = fig.update()
    return fig