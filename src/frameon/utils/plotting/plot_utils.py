import pandas as pd
import numpy as np
from typing import Union, Optional, Literal,  List, Dict
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def subplots(
    configs,
    title=None,
    width=None,
    height=None,
    rows=1,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=False,
    horizontal_spacing=None,
    specs=None,
    column_widths=None,
    row_heights=None,
    subplot_titles=None
):
    """
    Creates a figure with multiple subplots using Plotly.

    Parameters:
    ----------
    configs : list
        List of dictionaries containing configuration for each subplot.
        Each dictionary must contain the following keys:
        - fig : plotly.graph_objects.Figure
            The plotly figure object for the subplot.
        - layout : plotly.graph_objects.Layout
            The plotly layout object for the subplot.
        - row : int
            Row position of the subplot (1-indexed).
        - col : int
            Column position of the subplot (1-indexed).

        Optional keys:
        - is_margin : bool, default=False
            Boolean to indicate if the plot is a margin plot.
        - domain_x : list, default=None
            X-axis domain range as [min, max].
        - domain_y : list, default=None
            Y-axis domain range as [min, max].
        - showgrid_x : bool, default=True
            Show X-axis grid lines.
        - showgrid_y : bool, default=True
            Show Y-axis grid lines.
        - showticklabels_x : bool, default=True
            Show X-axis tick labels.
        - xaxis_visible : bool, default=True
            X-axis visibility.
        - yaxis_visible : bool, default=True
            Y-axis visibility.
        - show_yaxis_title : bool, default=True
            Whether to show the Y-axis title.

    title : str, default=''
        Main figure title.

    width : int, default=800
        Figure width in pixels.

    height : int, default=600
        Figure height in pixels.

    rows : int, default=1
        Number of rows in subplot grid.

    cols : int, default=1
        Number of columns in subplot grid.

    shared_xaxes : bool, default=False
        Share X axes between subplots.

    shared_yaxes : bool, default=False
        Share Y axes between subplots.

    horizontal_spacing : float, default=0.1
        Spacing between subplots horizontally (0 to 1).

    specs : list, default=None
        Subplot specifications, where each element defines the type of plot in that subplot.

    column_widths : list, default=None
        List of relative column widths.

    row_heights : list, default=None
        List of relative row heights.

    subplot_titles : list, default=None
        List of subplot titles.

    Returns:
    -------
    plotly.graph_objects.Figure
        The created figure with subplots.

    Example:
    -------
    configs = [
        dict(
            fig = fig_cnt.data[0]
            , layout = fig_cnt.layout
            , showgrid_y = False
            , row = 1
            , col = 1
        )
        , dict(
            fig = fig_sum.data[0]
            , layout = fig_sum.layout
            , showgrid_y = False
            , row = 1
            , col = 2
            , show_yaxis_title = False
        )
    ]
    """
    # Implementation of the function goes here

    # Create subplot layout
    # if subplot_titles is None:
    #     subplot_titles = []
    #     for fig in configs:
    #         subplot_titles.append(fig['layout'].title.text)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        column_widths=column_widths,
        row_heights=row_heights,
        specs=specs,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        horizontal_spacing=horizontal_spacing
    )

    # Process each subplot configuration
    if configs:
        for config in configs:
            if not config['fig']:
                continue

            # Set default values for configuration
            config['is_margin'] = config.get('is_margin', False)
            config['with_margin'] = config.get('with_margin', False)
            config['show_yaxis_title'] = config.get('show_yaxis_title', True)
            config['show_xaxis_title'] = config.get('show_xaxis_title', True)

            # Handle margin plot settings
            if config['is_margin']:
                config['domain_y'] = config.get('domain_y', [0.95, 1])
                config['showgrid_x'] = config.get('showgrid_x', False)
                config['showgrid_y'] = config.get('showgrid_y', False)
                config['showticklabels_x'] = config.get('showticklabels_x', False)
                config['xaxis_visible'] = config.get('xaxis_visible', False)
                config['yaxis_visible'] = config.get('yaxis_visible', False)
            else:
                # Handle regular plot settings
                if config['with_margin']:
                    config['domain_y'] = config.get('domain_y', [0, 0.9])
                config['showgrid_x'] = config.get('showgrid_x', True)
                config['showgrid_y'] = config.get('showgrid_y', True)
                config['showticklabels_x'] = config.get('showticklabels_x', True)
                config['xaxis_visible'] = config.get('xaxis_visible', True)
                config['yaxis_visible'] = config.get('yaxis_visible', True)

            # Set axis titles
            if config['show_xaxis_title']:
                config['xaxis_title_text'] = config['layout'].xaxis.title.text if 'layout' in config else None
            else:
                config['xaxis_title_text'] = None
            if config['show_yaxis_title']:
                config['yaxis_title_text'] = config['layout'].yaxis.title.text if 'layout' in config else None
            else:
                config['yaxis_title_text'] = None

            # Add trace and update axes
            fig.add_trace(config['fig'], row=config['row'], col=config['col'])

            # Update X axes
            fig.update_xaxes(
                row=config['row'],
                col=config['col'],
                showgrid=config['showgrid_x'],
                showticklabels=config['showticklabels_x'],
                # visible=config['xaxis_visible'],
                title_text=config['xaxis_title_text'],
                gridwidth=1,
                gridcolor="rgba(0, 0, 0, 0.1)"
            )

            # Update domains if specified
            if 'domain_x' in config:
                fig.update_xaxes(row=config['row'], col=config['col'], domain=config['domain_x'])
            if 'domain_y' in config:
                fig.update_yaxes(row=config['row'], col=config['col'], domain=config['domain_y'])
            # Update Y axes
            fig.update_yaxes(
                row=config['row'],
                col=config['col'],
                # showgrid=config['showgrid_y'],
                gridwidth=1,
                gridcolor="rgba(0, 0, 0, 0.1)",
                visible=config['yaxis_visible'],
                title_text=config['yaxis_title_text']
            )
            # fig.update_traces(selector=dict(type='pie'),
            #       marker=dict(colors=colorway_for_bar))
            fig_update(fig, xaxis_showgrid = config['showgrid_x'], yaxis_showgrid = config['showgrid_y'])

    # Adjust subplot titles position
    if subplot_titles:
        for i, _ in enumerate(subplot_titles):
            fig['layout']['annotations'][i-1]['y'] = 1.04


    # Update figure layout
    fig.update_layout(
        title_text=title,
        width=width,
        height=height,
        margin =  dict(l=50, r=50, b=50, t=50),
        # font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
        # title_font_size= 16,
        # xaxis_showticklabels=True,
        # xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        # yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        # xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        # yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        # xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        # yaxis_linecolor="rgba(0, 0, 0, 0.4)",
        # xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        # yaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        # legend_title_font_color='rgba(0, 0, 0, 0.7)',
        # legend_title_font_size=14,
        # legend_font_color='rgba(0, 0, 0, 0.7)',
        # hoverlabel=dict(bgcolor="white")
    )

    return fig