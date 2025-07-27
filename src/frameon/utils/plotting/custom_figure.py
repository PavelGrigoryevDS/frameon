from typing import Union, Optional, Literal,  List, Dict
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from pathlib import Path
import plotly.io as pio
import json

__all__ = []

# Load and register the custom template
_template_path = Path(__file__).parent / 'plotly_config.json'

with open(_template_path, 'r') as f:
    custom_template = json.load(f)

pio.templates['frameon_template'] = custom_template
pio.templates.default = 'frameon_template'

class CustomFigure(go.Figure):
    """Custom Figure class with extended update functionality."""
    def __init__(self, *args, **kwargs):
        """
        Initialize the CustomFigure.
        
        Parameters
        ----------
        args : tuple
            Positional arguments passed to go.Figure
        kwargs : dict
            Keyword arguments passed to go.Figure
        """
        super().__init__(*args, **kwargs)
            
    def update(
        self,
        title_text: Optional[str] = None,
        title_y:  Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        xaxis_title_text: Optional[str] = None,
        xaxis_showgrid: Optional[bool] = None,
        xaxis_tickformat: Optional[str] = None,
        xaxis_tickprefix: Optional[str] = None,
        xaxis_ticksuffix: Optional[str] = None,
        xaxis_dtick: Optional[float] = None,
        xaxis_ticktext: Optional[List[str]] = None,
        xaxis_tickvals: Optional[List[Union[int, float]]] = None,
        xaxis_range: Optional[List[Union[int, float]]] = None,
        xaxis_domain: Optional[List[float]] = None,
        yaxis_title_text: Optional[str] = None,
        yaxis_showgrid: Optional[bool] = None,
        yaxis_tickformat: Optional[str] = None,
        yaxis_tickprefix: Optional[str] = None,
        yaxis_ticksuffix: Optional[str] = None,
        yaxis_dtick: Optional[float] = None,
        yaxis_ticktext: Optional[List[str]] = None,
        yaxis_tickvals: Optional[List[Union[int, float]]] = None,
        yaxis_range: Optional[List[Union[int, float]]] = None,
        yaxis_domain: Optional[List[float]] = None,
        legend_title_text: Optional[str] = None,
        legend_position: Literal['top', 'right', 'bottom'] = None,
        showlegend: Optional[bool] = None,
        legend_x: Optional[float] = None,
        legend_y: Optional[float] = None,
        legend_xanchor: Literal['auto', 'left', 'center', 'right'] = None,
        legend_yanchor: Literal['auto', 'top', 'middle', 'bottom'] = None,
        legend_orientation: Literal['v', 'h'] = None,
        legend_itemsizing: Literal['trace', 'constant'] = None,
        legend_tracegroupgap: Optional[float] = None,
        textposition: Optional[str] = None,
        texttemplate: Optional[str] = None,
        textfont: Optional[Dict[str, Union[int, str]]] = None,
        hovertemplate: Optional[str] = None,
        hovermode: Literal['x', 'y', 'closest', 'x unified', False] = None,
        hoverlabel_align: Optional[str] = None,
        opacity: Optional[float] = None,
        bargap: Optional[float] = None,
        barmode: Optional[Literal['group', 'stack', 'overlay']] = None,
        boxmode: Optional[Literal['group', 'stack', 'overlay']] = None,
        bargroupgap: Optional[float] = None,
        margin: Optional[Dict[str, int]] = None,
        annotations: Optional[List[Dict]] = None,
        xgap: Optional[int] = None,
        ygap: Optional[int] = None,
        template: Optional[str] = None,
        hoverlabel: Optional[Dict[str, Union[int, str]]] = None,
        coloraxis: Optional[Dict] = None,
        plot_bgcolor: Optional[str] = None,
        **kwargs
    ) -> "CustomFigure":
        """
        Update the figure with consistent styling settings while maintaining all existing functionality.

        This method extends the standard Figure.update() method with additional convenient parameters
        for common styling tasks.

        Parameters
        ----------
        title_text : Optional[str]
            Main chart title displayed at the top of the figure.

            Example: title="Sales Performance 2023"

        title_y : Optional[float]
            Y postion title

            Example: title_y=0.97

        height : Optional[int]:
            Figure height in pixels.

            Example: height=600

        width : Optional[int]
            Figure width in pixels.

            Example: width=800

        xaxis_title_text : Optional[str]
            Label for the x-axis.

            Example: xaxis_title_text="Date"

        xaxis_showgrid : Optional[bool]
            Whether to show grid lines on the x-axis.

            Example: xaxis_showgrid=False to hide x-axis grid

        xaxis_tickformat : Optional[str]
            Format string for x-axis ticks.

            Examples:

            - '%Y-%m-%d' for dates
            - '.0f' for integers

        xaxis_tickprefix : Optional[str]
            Prefix for x-axis tick labels.

            Example: xaxis_tickprefix=' km'

        xaxis_ticksuffix : Optional[str]
            Suffix for x-axis tick labels.

            Example: xaxis_ticksuffix=' units'

        xaxis_dtick Optional[float]):
            Step size between x-axis ticks.

            Example: xaxis_dtick=1 for integer steps

        xaxis_ticktext : Optional[List[str]]
            Custom tick labels for the x-axis.

            Example: xaxis_ticktext=['Jan', 'Feb', 'Mar']

        xaxis_tickvals : Optional[List[Union[int, float]]]
            Values where custom tick labels should be placed on the x-axis.

            Example: xaxis_tickvals=[0, 1, 2]

        xaxis_range (Optional[List[Union[int, float]]]):
            Custom range for the x-axis.

            Example: xaxis_range=[0, 100]

        xaxis_domain (Optional[List[float]]):
            Domain range for x-axis (0 to 1).

            Example: xaxis_domain=[0.1, 0.9]

        yaxis_title_text (Optional[str]):
            Label for the y-axis.

            Example: yaxis_title_text="Revenue ($)"

        yaxis_showgrid (Optional[bool]):
            Whether to show grid lines on the y-axis.

            Example: yaxis_showgrid=True

        yaxis_tickformat (Optional[str]):
            Format string for y-axis ticks.

            Examples:

            - '.2f' for two decimals
            - '$,.0f' for currency without decimals

        yaxis_tickprefix (Optional[str]):
            Prefix for y-axis tick labels.

            Example: yaxis_tickprefix='$'

        yaxis_ticksuffix (Optional[str]):
            Suffix for y-axis tick labels.

            Example: yaxis_ticksuffix='%'

        yaxis_dtick (Optional[float]):
            Step size between y-axis ticks.

            Example: yaxis_dtick=10

        yaxis_ticktext (Optional[List[str]]):
            Custom tick labels for the y-axis.

            Example: yaxis_ticktext=['Low', 'Medium', 'High']

        yaxis_tickvals (Optional[List[Union[int, float]]]):
            Values where custom tick labels should be placed on the y-axis.

            Example: yaxis_tickvals=[0, 50, 100]

        yaxis_range (Optional[List[Union[int, float]]]):
            Custom range for the y-axis.

            Example: yaxis_range=[0, 100]

        yaxis_domain (Optional[List[float]]):
            Domain range for y-axis (0 to 1).

            Example: yaxis_domain=[0.1, 0.9]

        legend_position Literal[str]:
            The position of the legend ('top', 'right', 'bottom'). Default is 'top'.

            Example: legend_position='top'

        legend_title_text (Optional[str]):
            Title displayed above the legend.

            Example: legend_title_text="Product Categories"

        showlegend (Optional[bool]):
            Whether to display the legend.

            Example: showlegend=False to hide legend

        legend_x (Optional[float]):
            Legend x position (0 to 1).

            Example: legend_x=1.02

        legend_y (Optional[float]):
            Legend y position (0 to 1).

            Example: legend_y=1.0

        legend_xanchor (Optional[str]):
            Legend x anchor point.

            Options: 'auto', 'left', 'center', 'right'

        legend_yanchor (Optional[str]):
            Legend y anchor point.

            Options: 'auto', 'top', 'middle', 'bottom'

        legend_orientation (Optional[str]):
            Legend orientation.

            Options: 'v' (vertical), 'h' (horizontal)

        legend_itemsizing (Optional[str]):
            Legend item sizing mode.

            Options: 'trace', 'constant'

        legend_tracegroupgap (Optional([float])):
            Sets the amount of vertical space (in px) between legend groups. Number greater than or equal to 0

        textposition (Optional[str]):
            Position of text labels relative to data points.

            Options:

            - 'top'
            - 'bottom'
            - 'middle'
            - 'auto'
            - 'top center'
            - 'bottom center'

            Example: textposition='top center'

        texttemplate (Optional[str]):
            Template for text displayed on the plot.

            Example: texttemplate='%{y:.1f}%'

        textfont (Optional[Dict[str, Union[int, str]]]):
            Font settings for displayed text.

            Example: textfont=dict(size=12, color='red', family='Arial')

        hovertemplate (Optional[str]):
            Custom hover tooltip template.

            Example: hovertemplate='Date: %{x}<br>Value: %{y:.2f}$<extra></extra>'

        hovermode (Optional[str]):
            Hover interaction mode.

            Options:

            - 'x unified': single tooltip for all traces at x position
            - 'x': separate tooltip for each trace at x position
            - 'y': separate tooltip for each trace at y position
            - 'closest': tooltip for closest point
            - False: disable hover tooltips

        hoverlabel_align (Optional[str]):
            Hover label text alignment.

            Options: 'left', 'right', 'auto'

        opacity (Optional[float]):
            Opacity for figure elements (0 to 1)

        bargap (Optional[float]):
            Gap between bars in bar charts (0 to 1).

            Example: bargap=0.3 for 30% gap

        barmode, boxmode (Optional[str]):
            Determines the mode of the bars in bar charts.

            Options include:

            - 'group' (default): Bars are placed next to each other.
            - 'stack': Bars are stacked on top of each other.
            - 'overlay': Bars are drawn on top of each other.

            Example: barmode='stack' for stacked bars.

        bargroupgap (Optional[float]):
            Gap between bar groups (0 to 1).

            Example: bargroupgap=0.2 for 20% gap

        margin (Optional[Dict[str, int]]):
            Plot margins in pixels.

            Example: margin=dict(l=50, r=50, t=50, b=50)

        annotations (Optional[List[Dict]]):
            List of annotation dictionaries to add to the figure.

            Example: [dict(text='Note', x=0.5, y=0.5, showarrow=False)]

        xgap (Optional[int]):
            Horizontal gap for heatmap cells.

        ygap (Optional[int]):
            Vertical gap for heatmap cells.

        template (Optional[str]):
            Plotly template name for consistent styling.

            Options: "plotly", "plotly_white", "plotly_dark", "ggplot2",
            
                    "seaborn", "simple_white", "presentation",
                    
                    "xgridoff", "ygridoff", "non"

        hoverlabel (Optional[Dict[str, Union[int, str]]]):
            Hover label configuration.

            Example: hoverlabel=dict(bgcolor='white', bordercolor="black", font_size=12, align="left")

        coloraxis (Optional[Dict]):
            Settings for the color axis (coloraxis) passed to the update_layout function.

            Example of usage:

                coloraxis={
                    
                    'colorscale': 'Viridis',
                    'cmin': 0,
                    'cmax': 70,
                    'colorbar': {
                        
                        'title_text': 'Initial Colorbar',
                        'len': 0.7,
                        'thickness': 20,
                        'ticks': 'outside',
                        'outlinewidth': 0
                        
                    }
                    
                }
                
        plot_bgcolor : str, optional
            Sets the background color of the plotting area in-between x and y axes.
        kwargs:
            Additional arguments to pass to the standard Figure.update() method

        Returns
        -------
        CustomFigure
            Interactive Plotly figure object
        """
        # First apply our custom updates

        self._fig_update(
            title_text=title_text,
            title_y=title_y,
            xaxis_title_text=xaxis_title_text,
            yaxis_title_text=yaxis_title_text,
            legend_title_text=legend_title_text,
            height=height,
            width=width,
            showlegend=showlegend,
            xaxis_showgrid=xaxis_showgrid,
            yaxis_showgrid=yaxis_showgrid,
            xaxis_tickformat=xaxis_tickformat,
            yaxis_tickformat=yaxis_tickformat,
            xaxis_tickprefix=xaxis_tickprefix,
            xaxis_ticksuffix=xaxis_ticksuffix,
            yaxis_tickprefix=yaxis_tickprefix,
            yaxis_ticksuffix=yaxis_ticksuffix,
            hovertemplate=hovertemplate,
            xaxis_dtick=xaxis_dtick,
            yaxis_dtick=yaxis_dtick,
            xaxis_ticktext=xaxis_ticktext,
            xaxis_tickvals=xaxis_tickvals,
            yaxis_ticktext=yaxis_ticktext,
            yaxis_tickvals=yaxis_tickvals,
            texttemplate=texttemplate,
            textfont=textfont,
            opacity=opacity,
            textposition=textposition,
            template=template,
            hovermode=hovermode,
            bargap=bargap,
            barmode=barmode,
            boxmode=boxmode,
            bargroupgap=bargroupgap,
            legend_x=legend_x,
            legend_y=legend_y,
            legend_orientation=legend_orientation,
            xaxis_range=xaxis_range,
            yaxis_range=yaxis_range,
            margin=margin,
            xgap=xgap,
            ygap=ygap,
            yaxis_domain=yaxis_domain,
            xaxis_domain=xaxis_domain,
            legend_yanchor=legend_yanchor,
            legend_xanchor=legend_xanchor,
            legend_itemsizing=legend_itemsizing,
            annotations=annotations,
            hoverlabel=hoverlabel,
            coloraxis=coloraxis,
            legend_position=legend_position,
            plot_bgcolor=plot_bgcolor,
            legend_tracegroupgap=legend_tracegroupgap,
        )

        # Then call the original update method with any additional kwargs
        super().update(**kwargs)

        # Return the updated figure instance
        return self

    def _fig_update(
        self,
        title_text: Optional[str] = None,
        title_y: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        xaxis_title_text: Optional[str] = None,
        xaxis_showgrid: Optional[bool] = None,
        xaxis_tickformat: Optional[str] = None,
        xaxis_tickprefix: Optional[str] = None,
        xaxis_ticksuffix: Optional[str] = None,
        xaxis_dtick: Optional[float] = None,
        xaxis_ticktext: Optional[List[str]] = None,
        xaxis_tickvals: Optional[List[Union[int, float]]] = None,
        xaxis_range: Optional[List[Union[int, float]]] = None,
        xaxis_domain: Optional[List[float]] = None,
        yaxis_title_text: Optional[str] = None,
        yaxis_showgrid: Optional[bool] = None,
        yaxis_tickformat: Optional[str] = None,
        yaxis_tickprefix: Optional[str] = None,
        yaxis_ticksuffix: Optional[str] = None,
        yaxis_dtick: Optional[float] = None,
        yaxis_ticktext: Optional[List[str]] = None,
        yaxis_tickvals: Optional[List[Union[int, float]]] = None,
        yaxis_range: Optional[List[Union[int, float]]] = None,
        yaxis_domain: Optional[List[float]] = None,
        legend_position: Literal['top', 'right', 'bottom'] = None,
        legend_title_text: Optional[str] = None,
        showlegend: Optional[bool] = None,
        legend_x: Optional[float] = None,
        legend_y: Optional[float] = None,
        legend_xanchor: Literal['auto', 'left', 'center', 'right'] = None,
        legend_yanchor: Literal['auto', 'top', 'middle', 'bottom'] = None,
        legend_orientation: Literal['v', 'h'] = None,
        legend_itemsizing: Literal['trace', 'constant'] = None,
        legend_tracegroupgap: Optional[float] = None,
        textposition: Optional[str] = None,
        texttemplate: Optional[str] = None,
        textfont: Optional[Dict[str, Union[int, str]]] = None,
        hovertemplate: Optional[str] = None,
        hovermode: Literal['x', 'y', 'closest', 'x unified', False] = None,
        hoverlabel_align: Optional[str] = None,
        opacity: Optional[float] = None,
        bargap: Optional[float] = None,
        barmode: Optional[Literal['group', 'stack', 'overlay']] = None,
        boxmode: Optional[Literal['group', 'stack', 'overlay']] = None,
        bargroupgap: Optional[float] = None,
        margin: Optional[Dict[str, int]] = None,
        annotations: Optional[List[Dict]] = None,
        xgap: Optional[int] = None,
        ygap: Optional[int] = None,
        template: Optional[str] = None,
        hoverlabel: Optional[Dict[str, Union[int, str]]] = None,
        coloraxis: Optional[Dict] = None,
        plot_bgcolor: Optional[str] = None,
        ) -> None:
        """
        Internal method to handle the figure updates with our custom parameters.
        """
        # First, we set the position of the legend so that you can further use the parameters for the bastard of the legend
        if legend_position:
            self._set_legend_position(legend_position)
        # Layout updates
        layout_updates = {
            'title_text': title_text,
            'title_y': title_y,
            'width': width,
            'height': height,
            'legend_title_text': legend_title_text,
            'showlegend': showlegend,
            'template': template,
            'hovermode': hovermode,
            'bargap': bargap,
            'barmode': barmode,
            'boxmode': boxmode,
            'bargroupgap': bargroupgap,
            'margin': margin,
            'annotations': annotations,
            'coloraxis': coloraxis,
            'plot_bgcolor': plot_bgcolor,
        }

        # Update layout only if there are updates
        layout_updates = {k: v for k, v in layout_updates.items() if v is not None}
        if layout_updates:
            self.update_layout(**layout_updates)
        self.update_layout(barmode='group')
        # X-axis settings
        xaxis_updates = {
            'showgrid': xaxis_showgrid,
            'dtick': xaxis_dtick,
            'title_text': xaxis_title_text,
            'tickformat': xaxis_tickformat,
            'range': xaxis_range,
            'tickprefix': xaxis_tickprefix,
            'ticksuffix': xaxis_ticksuffix,
            'ticktext': xaxis_ticktext,
            'tickvals': xaxis_tickvals,
        }
        self.update_xaxes(**{k: v for k, v in xaxis_updates.items() if v is not None})

        # Y-axis settings
        yaxis_updates = {
            'showgrid': yaxis_showgrid,
            'dtick': yaxis_dtick,
            'title_text': yaxis_title_text,
            'tickformat': yaxis_tickformat,
            'range': yaxis_range,
            'tickprefix': yaxis_tickprefix,
            'ticksuffix': yaxis_ticksuffix,
            'ticktext': yaxis_ticktext,
            'tickvals': yaxis_tickvals,
        }
        self.update_yaxes(**{k: v for k, v in yaxis_updates.items() if v is not None})

        # Legend settings
        legend_updates = {
            'orientation': legend_orientation,
            'x': legend_x,
            'y': legend_y,
            'yanchor': legend_yanchor,
            'xanchor': legend_xanchor,
            'itemsizing': legend_itemsizing,
        }

        self.update_layout(legend={k: v for k, v in legend_updates.items() if v is not None})

        # Update traces if necessary
        trace_updates = {}
        if hovertemplate is not None:
            trace_updates['hovertemplate'] = hovertemplate
        if hoverlabel is not None:
            trace_updates['hoverlabel'] = hoverlabel
        if textposition is not None:
            trace_updates['textposition'] = textposition
        if texttemplate is not None:
            self.update_traces(texttemplate=texttemplate)
        if opacity is not None:
            trace_updates['opacity'] = opacity

        if trace_updates:
            self.update_traces(**trace_updates)

        if self.data and self.data[0].type == 'heatmap':
            self.update_traces(xgap=xgap, ygap=ygap)

        if textfont is not None:
            self.update_layout(font=textfont)

        if yaxis_domain is not None or xaxis_domain is not None:
            self.update_layout(
                yaxis=dict(domain=yaxis_domain) if yaxis_domain is not None else {},
                xaxis=dict(domain=xaxis_domain) if xaxis_domain is not None else {},
            )
        for trace in self.data:
            if trace.hovertemplate:
                trace.hovertemplate = re.sub(r'\s*=\s*', ' = ', trace.hovertemplate)

    def _set_legend_position(
        self,
        legend_position: Literal['top', 'right', 'bottom'] = 'top',
        yaxis_domain: Optional[List[float]] = None,
        orientation: Optional[Literal['h', 'v']] = None,
        yanchor: Optional[Literal['auto', 'top', 'middle', 'bottom']] = None,
        y: Optional[float] = None,
        xanchor: Optional[Literal['auto', 'left', 'center', 'right']] = None,
        x: Optional[float] = None,
        itemsizing: Optional[Literal['trace', 'constant']] = None
        ):
        """
        Sets the position of the legend in the plot.
        """
        if legend_position == 'top':
            self.update_yaxes(domain = yaxis_domain if yaxis_domain else [0, 0.95])
            self.update_layout(
                legend = dict(
                    orientation=orientation if orientation else "h"
                    , yanchor=yanchor if yanchor else "top"
                    , y=y if y else 1.06
                    , xanchor=xanchor if xanchor else "center"
                    , x=x if x else 0.5
                    , itemsizing=itemsizing if itemsizing else "constant"
                    # , bordercolor='rgba(0,0,0,0.1)'
                    # , borderwidth=1
                )
            )
        elif legend_position == 'right':
            self.update_layout(
                legend = dict(
                    xanchor=xanchor if xanchor else None
                    , yanchor=yanchor if yanchor else None
                    , orientation=orientation if orientation else "v"
                    , y=y if y else 1
                    , x=None
                    , itemsizing=itemsizing if itemsizing else "constant"
                )
            )
        elif legend_position == 'bottom':
            self.update_layout(
                legend = dict(
                    title_text='Score'
                    , orientation=orientation if orientation else "h"
                    , yanchor=yanchor if yanchor else "bottom"
                    , y=y if y else -0.15
                    , xanchor=xanchor if xanchor else "center"
                    , x=x if x else 0.5
                    , itemsizing=itemsizing if itemsizing else "constant"
                    # , bordercolor='rgba(0,0,0,0.1)'
                    # , borderwidth=1
                )
            )
        else:
            raise ValueError("legend_position must be 'top', 'right', or 'bottom'")
        return self
