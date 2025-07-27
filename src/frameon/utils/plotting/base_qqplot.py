import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.api import qqplot as sm_qqplot
from typing import Union, List, Dict, Optional
from frameon.utils.plotting import CustomFigure

def create_qqplot(
    x: Union[pd.Series, np.ndarray, List[float]],
    show_skew_curt: bool = True,
    point_color: str = 'rgba(40, 115, 168, 0.9)',
    line_color: str = 'rgba(226, 85, 89, 0.9)',
    point_size: int = 8,
    line_width: int = 2,
    title: str = 'Q-Q Plot',
    width: int = 400,
    height: int = 350,
    show_grid: bool = True,
    reference_line: str = '45',
    renderer: str = None
) -> Union[CustomFigure, None]:

    # Input validation
    if not isinstance(x, (pd.Series, np.ndarray, list)):
        raise TypeError("Input must be a pandas Series, numpy array, or list")

    if isinstance(x, list):
        x = np.array(x)

    if isinstance(x, (pd.Series, np.ndarray)):
        if len(x) < 2:
            raise ValueError("Input data must have at least 2 values")

    # Convert to pandas Series for consistency
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    if x.isna().any():
        x = x.dropna()

    # Calculate statistics
    skew = x.skew()
    kurt = x.kurtosis()

    # Create Q-Q plot using statsmodels
    fig = sm_qqplot(
        x,
        line=reference_line,
        fit=True
    )
    plt.close()  # Close the matplotlib plot

    # Extract data from the plot
    qqplot_data = fig.gca().lines

    # Get points and reference line data
    points_x = qqplot_data[0].get_xdata()  # Theoretical quantiles
    points_y = qqplot_data[0].get_ydata()  # Sample quantiles

    # Create Plotly figure
    fig = go.Figure()

    # Add data points
    fig.add_trace(go.Scatter(
        x=points_x,
        y=points_y,
        mode='markers',
        name='Data Points',
        marker=dict(
            color=point_color,
            size=point_size,
            opacity=0.8,
            line=dict(width=0, color='white')
        ),
        hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
    ))

    # Add reference line
    if reference_line is not None:
        line_x = qqplot_data[1].get_xdata()    # Reference line x
        line_y = qqplot_data[1].get_ydata()    # Reference line y
        fig.add_trace(go.Scattergl(
            x=line_x,
            y=line_y,
            mode='lines',
            name='Reference Line',
            line=dict(
                color=line_color,
                width=line_width,
                dash='solid'
            ),
            hoverinfo='none'
        ))

    # Calculate axis ranges with padding
    x_min, x_max = min(points_x), max(points_x)
    y_min, y_max = min(points_y), max(points_y)
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    # Add statistics annotation if requested
    if show_skew_curt:
        fig.add_annotation(
            x=0.98,
            y=0.02,
            xref='paper',
            yref='paper',
            text=f"Skew: {skew:.2f}<br>Kurt: {kurt:.2f}",
            showarrow=False,
            align='right',
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        )

    # Update figure layout
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
        width=width,
        height=height,
        xaxis=dict(showgrid=show_grid, zeroline=False),
        yaxis=dict(showgrid=show_grid, zeroline=False),
        margin=dict(l=50, r=20, b=50, t=50)
    )
    if reference_line != '45':
        fig.update_layout(
            xaxis_range=[x_min - x_padding, x_max + x_padding],
            yaxis_range=[y_min - y_padding, y_max + y_padding]
        )
    if renderer is not None:
        fig.show(config=dict(dpi=200), renderer=renderer, width=width, height=height)
    else:
        return CustomFigure(fig)