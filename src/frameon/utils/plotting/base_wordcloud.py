from typing import Optional, Union, List, Dict, Tuple, Callable, Any
import numpy as np
from wordcloud import WordCloud
import plotly.graph_objects as go
from matplotlib.colors import Colormap


def create_wordcloud_plotly(
    text: str,
    title: Optional[str] = None,
    max_words: int = 100,
    width: int = 800,
    height: int = 400,
    background_color: str = 'white',
    colormap: Union[str, Colormap] = 'viridis',
    margin: Optional[Union[int, Dict[str, int]]] = None,
    relative_scaling: float = 0.5,
    prefer_horizontal: float = 0.9,
    contour_width: int = 0,
    contour_color: str = 'black',
    random_state: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    stopwords: Optional[List[str]] = None,
    collocations: bool = True,
    normalize_plurals: bool = True,
    include_numbers: bool = False,
    min_word_length: int = 0,
    repeat: bool = False,
    scale: float = 1,
    min_font_size: int = 4,
    max_font_size: Optional[int] = None,
    font_path: Optional[str] = None,
    color_func: Optional[Callable] = None,
    regexp: Optional[str] = None,
    collocation_threshold: int = 30,
    return_fig: bool = False,
    scroll_zoom: bool = False,
    plotly_kwargs: Optional[Dict[str, Any]] = None
) -> Union[go.Figure, None]:
    """
    Generate an interactive word cloud visualization using Plotly.
    """
    # Process margin
    if margin is None:
        margin = dict(l=10, r=10, b=10, t=40 if title else 10)
    elif isinstance(margin, int):
        margin = dict(l=margin, r=margin, b=margin, t=margin + (40 if title else 0))
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=1000, # 2000
        height=500, # 1000
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        relative_scaling=relative_scaling,
        prefer_horizontal=prefer_horizontal,
        contour_width=contour_width,
        contour_color=contour_color,
        random_state=random_state,
        mask=mask,
        stopwords=stopwords,
        collocations=collocations,
        normalize_plurals=normalize_plurals,
        include_numbers=include_numbers,
        min_word_length=min_word_length,
        repeat=repeat,
        scale=scale,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        font_path=font_path,
        color_func=color_func,
        regexp=regexp,
        collocation_threshold=collocation_threshold
    ).generate(text)
    
    # Convert to numpy array
    wordcloud_array = np.asarray(wordcloud)
    
    # Create figure
    fig = go.Figure(
        go.Image(
            z=wordcloud_array,
            hoverinfo='skip',
            xaxis='x',
            yaxis='y'
        )
    )
    # Update layout
    fig.update_layout(
        title_text=title if title else '',
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        margin=margin,
        height=height,
        width=width,
        xaxis=dict(
            visible=False,
            # fixedrange=True,
            # range=[20, width - 20],
            scaleanchor='y'
        ),
        yaxis=dict(
            visible=False,
            # fixedrange=True,
            # range=[height - 20, 20],
            scaleanchor='x'
        )
    )
    # Apply any additional figure kwargs
    if plotly_kwargs:
        fig.update_layout(**plotly_kwargs)
    
    if not return_fig:
        fig.show(config={'scrollZoom': scroll_zoom})
    else:
        return fig