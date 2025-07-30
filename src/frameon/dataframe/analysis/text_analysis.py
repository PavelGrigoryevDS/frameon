import pandas as pd
import numpy as np
import plotly.express as px
from dataclasses import dataclass
from typing import Union, Dict, Optional, Tuple, Any, List, Literal, TYPE_CHECKING
from typing import get_type_hints
from plotly.subplots import make_subplots
from dataclasses import fields, field
from frameon.utils.plotting import CustomFigure
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download as nltk_download
from IPython.display import display
from wordcloud import WordCloud
import plotly.graph_objects as go
from frameon.utils.miscellaneous import style_dataframe, add_empty_columns_for_df, is_text_column
if TYPE_CHECKING: # pragma: no cover
    from frameon.core.base import FrameOn
    
__all__ = ['TextAnalyzer']

class TextAnalyzer:
    def __init__(self, df: "FrameOn"):
        """Initialize with the main dataframe."""
        self._df = df
       
    def sentiment(
        self,
        text_column: str,
        method: str = 'vader',
        clean_text: bool = True,
        show_stats: bool = False,
        show_distribution: bool = True,
        intensity_thresholds: Tuple[float, float] = (0.05, 0.05),
        height: int = None, 
        width: int = None,        
    ) -> Union[CustomFigure, None]:
        """
        Perform comprehensive sentiment analysis on text data with multiple visualization options.
        
        Parameters:
        -----------
        text_column : str
            Name of the column containing the text data.
        method : str, optional (default='vader')
            Sentiment analysis method to use ('textblob' or 'vader').
        clean_text : bool, optional (default=True)
            Whether to clean the text before analysis (removes punctuation and converts to lowercase).
        show_stats : bool, optional (default=True)
            Whether to display sentiment statistics table.
        show_distribution : bool, optional (default=True)
            Whether to display the sentiment distribution plot.
        intensity_thresholds : Tuple[float, float], optional (default=(0.05, 0.05))
            Thresholds for classifying sentiment intensity (neutral_low, neutral_high).
        height, width : int, optional
            Height and width for plot 
                        
        Returns:
        --------
            Union[CustomFigure, None]
        """
        if text_column not in self._df:
            raise ValueError(f'Column "{text_column}" does not exist in the DataFrame')
        df = self._df[[text_column]].dropna()
        if df.empty:
            raise ValueError('After dropping missing values, the text column is empty')
        if not is_text_column(df[text_column]):
            raise ValueError(f'text_column must contain text')
        # Initialize sentiment analyzer
        if method == 'vader':
            nltk_download('vader_lexicon', quiet=True)
            sia = SentimentIntensityAnalyzer()
        
        def analyze_sentiment(text: str) -> float:
            """Analyze text sentiment and return polarity score."""
                
            if method == 'textblob':
                blob = TextBlob(text)
                return round(blob.sentiment.polarity, 2)
            elif method == 'vader':
                return round(sia.polarity_scores(text)['compound'], 2)
            else:
                raise ValueError(f"Unsupported method: {method}")
        
        def categorize_sentiment(score: float) -> str:
            """Categorize sentiment with intensity levels."""
            neutral_low, neutral_high = intensity_thresholds
            if score > neutral_high:
                return "Positive" if score > 0.5 else "Slightly Positive"
            elif score < -neutral_low:
                return "Negative" if score < -0.5 else "Slightly Negative"
            else:
                return "Neutral"

        if clean_text:
            df[text_column] = df[text_column].str.replace(r'[^\w\s]', '', regex=True).str.lower()    
        # Apply sentiment analysis
        df['sentiment'] = df[text_column].apply(analyze_sentiment)
        df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)

        # Count the occurrences of each sentiment
        sentiment_counts = df['sentiment_category'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        sentiment_counts['Percentage'] = (sentiment_counts['Count'] / sentiment_counts['Count'].sum() * 100).round(1)
        
        if show_stats:
            # Create formatted version for display
            sentiment_counts_display = sentiment_counts.copy()
            sentiment_counts_display['Percentage'] = sentiment_counts_display['Percentage'].astype(str) + '%'
            sentiment_counts_display['Count (Percentage)'] = (sentiment_counts_display['Count'].astype(str) + 
                                                            ' (' + sentiment_counts_display['Percentage'] + ')')
            sentiment_counts_display = sentiment_counts_display[['Sentiment', 'Count (Percentage)']]
            sentiment_counts_display.columns = ['col1', 'col2']
            
            avg_sentiment = round(df['sentiment'].mean(), 2)
            sentiment_75_pct = round(df['sentiment'].quantile(0.75), 2)
            median_sentiment = round(df['sentiment'].median(), 2)
            sentiment_25_pct = round(df['sentiment'].quantile(0.25), 2)
            sentiment_min = round(df['sentiment'].min(), 2)
            sentiment_max = round(df['sentiment'].max(), 2)

            # Create result DataFrame
            statistics = pd.DataFrame(
                {
                    "Mean": [avg_sentiment],
                    "Max": [sentiment_max],
                    "75%": [sentiment_75_pct],
                    "Median": [median_sentiment],
                    "25%": [sentiment_25_pct],
                    "Min": [sentiment_min],
                }
            )
            statistics = statistics.T.astype(str).reset_index()
            statistics.columns = ['col1', 'col2']
            res_df = pd.concat([statistics, sentiment_counts_display], axis=1).T.reset_index(drop=True).T
            res_df = res_df.fillna('')
            add_empty_columns_for_df(res_df, [2])
            display(style_dataframe(
                res_df,
                caption='Sentiment Analysis Statistics',
            ))
        
        if show_distribution:
            # Create subplots with 1 row and 2 columns
            fig = make_subplots(rows=2, cols=2, 
                            #    column_widths=[0.5, 0.5],
                            horizontal_spacing=0.15,
                            specs=[
                                [{'colspan': 1}, {'rowspan': 2}],
                                [{'colspan': 1}, None]
                            ],                              
                            subplot_titles=("Sentiment Distribution", "Sentiment Proportion"))
            
            # Left plot - Histogram
            fig.add_trace(
                px.histogram(
                    x=df['sentiment'],
                    histnorm='probability',
                    nbins=50,
                ).update_traces(hovertemplate="Score: %{x}<br>Probability: %{y}<extra></extra>").data[0],                
                row=2, col=1
            )

            # Right plot - Horizontal bar chart with percentages
            sentiment_counts = sentiment_counts.sort_values('Count')
            fig.add_trace(
                px.bar(
                    sentiment_counts,
                    x='Percentage',
                    y='Sentiment',
                    text=sentiment_counts['Percentage'].astype(str) + '%',
                    hover_data={
                        'Percentage': ':.1f',    # Format percentage with one decimal place in hover
                        'Count': True,           # Show raw count as-is
                        'Sentiment': False       # Hide sentiment from hover because it is shown as y-axis label
                    },
                    custom_data='Count',
                ).data[0],  
                row=1, col=2
            )       
            # Left plot - Box
            fig.add_trace(
                px.box(
                    x=df['sentiment'],
                ).update_traces(hovertemplate="Score: %{x}<extra></extra>").data[0],
                row=1, col=1
            )        
            
            # Update layout
            fig.update_layout(
                title_text="Sentiment Analysis",
                showlegend=False,
                height=height if height else 400,
                width=width if width else 900,
                margin=dict(l=50, r=50, b=50, t=80),
                plot_bgcolor='white',
            )
            
            # Update x-axis for histogram
            fig.update_xaxes(
                title_text="Sentiment Score",
                row=2, col=1
            )
            
            # Update y-axis for histogram
            fig.update_yaxes(
                title_text="Probability",
                row=2, col=1,
                domain=[0, 0.92]
            )

            # Update x-axis for box
            fig.update_xaxes(
                visible=False,
                row=1, col=1
            )
            
            # Update y-axis for box
            fig.update_yaxes(
                visible=False,
                row=1, col=1,
                domain=[0.93, 1]
            )

            # Update x-axis for bar chart (remove axis and title)
            fig.update_xaxes(
                visible=False,
                showgrid=False,
                showticklabels=False,
                title_text="",
                row=1, col=2,
            )
            
            # Update y-axis for bar chart
            fig.update_yaxes(
                title_text="",
                row=1, col=2
            )
            # print(fig)
            return CustomFigure(fig)
        
    def word_frequency(
        self,
        text_column: str,
        n: int = 10,
        show: Literal["top", "bottom", "both"] = "top",
        title: Optional[str] = None,
        horizontal_spacing: Optional[float] = None,
        text_auto=False,
        height: int = None, 
        width: int = None, 
        **wordcloud_kwargs: Any
    ) -> CustomFigure:
        """
        Generate bar plots showing the most/least frequent words in a text column.
        
        Parameters:
        -----------
        text_column : str
            Name of the column containing text data to analyze.
        n : int, optional (default=10)
            Number of top/bottom words to display. When show="both", this will show
            n words for both top and bottom (total 2n words).
        show : Literal["top", "bottom", "both"], optional (default="top")
            Which words to display:
            
            - "top": Only show top n most frequent words
            - "bottom": Only show bottom n least frequent words
            - "both": Show both top and bottom n words in separate subplots
            
        title : Optional[str], optional (default=None)
            Custom title for the plot. If None, a default title will be generated.
        horizontal_spacing : float
            For show = 'both'. Space between subplot columns in normalized plot coordinates. Must be a float between 0 and 1.
            
        text_auto : bool
            text_auto parameter pass to plotly
        height, width : int, optional
            Height and width for plot 
        wordcloud_kwargs : Any
            Additional keyword arguments to pass to WordCloud.process_text().
            
            Common options include:
            
            - stopwords: Set of stopwords to exclude
            - max_words: Maximum number of words to consider
            - collocations: Whether to include bigrams
            
        Returns:
        --------
            A Plotly figure object containing the requested bar plots.          
        """
        valid_show_parameter = ["top", "bottom", "both"]
        # Process text and get word frequencies
        wordcloud = WordCloud(**wordcloud_kwargs)
        word_freq = wordcloud.process_text(self._df[text_column].str.cat(sep=' '))

        sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare data based on show parameter
        if show == "both":
            # Get top and bottom n words
            top_words = sorted_word_freq[:n]
            top_words.reverse()
            bottom_words = sorted_word_freq[-n:] if len(sorted_word_freq) > n else sorted_word_freq
            # Create DataFrames
            top_df = pd.DataFrame(top_words, columns=['word', 'freq'])
            bottom_df = pd.DataFrame(bottom_words, columns=['word', 'freq'])
            
            # Create subplots
            fig = make_subplots(rows=1, cols=2, horizontal_spacing=horizontal_spacing, subplot_titles=(
                f"Top {n} Most Frequent Words", 
                f"Bottom {n} Least Frequent Words"
            ))
            
            # Add top words bar chart
            fig.add_trace(
                px.bar(
                    x=top_df['freq'],
                    y=top_df['word'],
                    text_auto=text_auto
                ).data[0],
                row=1, col=1
            )
            
            # Add bottom words bar chart
            fig.add_trace(
                px.bar(
                    x=bottom_df['freq'],
                    y=bottom_df['word'],
                    text_auto=text_auto
                ).data[0],
                row=1, col=2
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>Frequency: %{x}<extra></extra>")
            # Update layout for dual plot
            fig.update_layout(
                title_text=title or f"Word Frequency Analysis (Top & Bottom {n})",
                showlegend=False,
                height= height if height else max(400, n * 20),  # Dynamic height based on n
                width=width if width else 900,
                margin=dict(l=50, r=50, b=50, t=80),
            )
            fig.update_xaxes(title_text="Frequency")
            
        else:
            # Get either top or bottom words
            if show == "top":
                words = sorted_word_freq[:n]
                words.reverse()
                default_title = f"Top {n} Most Frequent Words"
            elif show == 'bottom':
                words = sorted_word_freq[-n:] if len(sorted_word_freq) > n else sorted_word_freq
                default_title = f"Bottom {n} Least Frequent Words"
            else:
                raise ValueError(f'Invalid show parameter must be one of {valid_show_parameter}')
            df = pd.DataFrame(words, columns=['word', 'freq'])
            
            # Create single plot
            fig = go.Figure()
            fig.add_trace(
                px.bar(
                    x=df['freq'],
                    y=df['word'],
                    text_auto=text_auto,
                    
                ).update_traces(hovertemplate="<b>%{y}</b><br>Frequency: %{x}<extra></extra>").data[0]
            )
            
            # Update layout for single plot
            fig.update_layout(
                title_text=title or default_title,
                showlegend=False,
                height= height if height else max(400, n * 20),  # Dynamic height based on n
                width=width if width else 500,
                margin=dict(l=50, r=50, b=50, t=50),
                xaxis_title_text="Frequency",
                # yaxis_title_text="Word"
            )
        
        return CustomFigure(fig)        