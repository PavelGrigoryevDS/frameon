from enum import Enum, auto
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pandas.io.formats.style import Styler
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels import robust
from wordcloud import WordCloud
import matplotlib as plt
from frameon.utils.plotting import CustomFigure
from frameon.utils.miscellaneous import (
    add_empty_columns_for_df, format_count_with_percentage,
    format_number, get_column_type, is_categorical_column,
    is_datetime_column, is_float_column, is_int_column,
    is_text_column, style_dataframe, validate_is_DataFrame
)
from frameon.utils.plotting.base_histogram import HistogramBuilder
from frameon.utils.plotting.base_wordcloud import create_wordcloud_plotly

__all__ = ['SeriesOnInfo']

class SeriesOnInfo:
    """
    Class containing methods for Series information analysis.
    """
    def __init__(self, series: pd.Series):
        self._series = series
        self.visualization_config = {
            'sizes': {'width': 800, 'height': 500},
            'font': {'family': "Arial", 'size': 12},
            'template': 'plotly_white'
        }        
        
    def info(
        self, 
        plot=True, 
        column_type: Optional[Literal['numeric', 'categorical', 'text', 'datetime']] = None,
        hist_mode: Literal['base', 'dual_hist_trim', 'dual_hist_qq'] = 'base',
        lower_quantile: Optional[Union[float, int]] = 0,
        upper_quantile: Optional[Union[float, int]] = 1,
        top_n: int = 10,
        max_words: int = 100,
        height: int = None, 
        width: int = None,
        labels: dict = None,
        title: str = None,
        show_text: bool = True,
        xaxis_type: str = None,
        yaxis_type: str = None,      
        renderer: str = None,
        **plotly_kwargs      
        ) -> Union[None, go.Figure]:
        """
        Generate combined report with summary and histogram
        
        Parameters:
        -----------
        plot : bool, optional
            Whether to return plot
        column_type : str, optional
            Type for column, if None type detect automaticaly. Can be one of 'numeric', 'categorical', 'text', 'datetime'.
        hist_mode : str, optional
            Specifies the type of histogram to be displayed (only for numeric columns). Available options are:
            
            - 'base': Display simple histogram
            - 'dual_hist_trim': Displays a dual plot with the original histogram on the left
                and a trimmed histogram (based on quantiles) on the right.
            - 'dual_hist_qq': Displays a dual plot with the original histogram on the left
                and a QQ-plot on the right.
                            
            Default is 'dual_hist_trim'.
            
        lower_quantile : float, optional
            The lower quantile for data filtering (default is 0).
        upper_quantile : float, optional
            The upper quantile for data filtering (default is 1).
        title : str
            Title for plot
        labels : dict, optional
            A dictionary mapping column names to their display names.
        top_n: int
            Number of top categories for bar chart (default: 10)
        height : int
            Height for plot
        width : int
            Width for plot
        show_text : bool
            Whether to show text in bar chart
        xaxis_type : str, optional
            Type of the X-axis for histogram ['-', 'linear', 'log', 'date', 'category', 'multicategory']
        yaxis_type : str, optional
            Type of the Y-axis for histogram ['-', 'linear', 'log', 'date', 'category', 'multicategory']   
        max_words : int
            Maximum words for word cloud (default: 100)
        renderer : str, optional
            The Plotly renderer to use for displaying the figure. Common options include:
            
            - 'browser': Opens plot in default web browser
            - 'notebook': Renders plot in Jupyter notebook output
            - 'png': Static PNG image
            - 'svg': Static SVG image
            - 'json': Returns figure as JSON string
            
            If None, uses Plotly's default renderer.
            See Plotly documentation for full list of available renderers.            
        plotly_kwargs
            Additional keyword arguments to pass to the Plotly Express histogram. Default is None
            
        Returns:
        --------
            None                  
        """
        if self._series.empty:
            raise ValueError(
                "Series is empty."
            )   
        if column_type and column_type not in ['numeric', 'categorical', 'text', 'datetime']:
            raise ValueError("column_type must be on of 'numeric', 'categorical', 'text', 'datetime'")
        is_numeric = pd.api.types.is_numeric_dtype(self._series)

        if column_type == 'numeric' and not is_numeric:
            try:
                self._series = self._series.astype(float)
            except Exception as e:
                raise ValueError(f'Cannot convert column to numeric type: {str(e)}')
        if column_type not in ['datetime', 'text', 'categorical'] and (is_numeric or column_type == 'numeric'):
            summary = self._generate_full_numeric_summary()
            display(summary)
            if plot:
                if hist_mode not in ['base', 'dual_hist_trim', 'dual_hist_qq']:
                    raise ValueError("hist_mode must be on of 'base', dual_hist_trim' or 'dual_hist_qq'")
                return self._generate_histogram(hist_mode, lower_quantile, upper_quantile, title, height, width, labels, xaxis_type, yaxis_type, plotly_kwargs)

        if column_type != 'datetime' and (column_type in ['text', 'categorical'] or pd.api.types.is_string_dtype(self._series.dropna())):
            if pd.api.types.is_datetime64_any_dtype(self._series):
                raise ValueError('Column cannot be datetime when processing as text/categorical')
            if not pd.api.types.is_string_dtype(self._series):
                try:
                    self._series = self._series.astype('string')
                except Exception as e:
                    raise ValueError(f'Cannot convert column to text type: {str(e)}')
            is_categegory = column_type != 'text' and (column_type == 'categorical' or is_categorical_column(self._series))
            summary = self._generate_summary_for_categorical(is_categegory)
            display(summary)
            if plot:
                fig =  self._generate_combined_charts(is_categegory, top_n, max_words, title, height, width, show_text, labels)
                if renderer is not None:
                    return fig.show(config=dict(dpi=200), renderer=renderer, height=fig.layout.height, width=fig.layout.width)
                else:
                    return fig
        is_datetime = pd.api.types.is_datetime64_any_dtype(self._series)
        if column_type == 'datetime' and not is_datetime:
            raise ValueError('Column must be datetime when column_type="datetime"')
        if is_datetime or column_type == 'datetime':
            summary = self._generate_full_datetime_summary()
            display(summary)
    
    # ========
    # Datetime
    # ========
    def _generate_basic_stats_datetime(self) -> pd.DataFrame:
        """
        Returns basic datetime statistics with exactly specified fields.
        """
        valid_dates = self._series.dropna()
        timedeltas = valid_dates.sort_values().diff().dropna() if len(valid_dates) > 1 else pd.Series(dtype='timedelta64[ns]')
        ram = round(self._series.memory_usage(deep=True) / 1048576)
        if ram == 0:
            ram = "<1 Mb"
        stats = {
            "First date": valid_dates.min().date() if not valid_dates.empty else "N/A",
            "Last date": valid_dates.max().date() if not valid_dates.empty else "N/A",
            "Avg Days Frequency": str(round(timedeltas.mean().total_seconds() / (24 * 3600), 2)) if not timedeltas.empty else "N/A",
            "Min Days Interval": timedeltas.min().days if not timedeltas.empty else "N/A",
            "Max Days Interval": timedeltas.max().days if not timedeltas.empty else "N/A",
            "Memory Usage": ram
        }
        return pd.DataFrame(stats.items(), columns=["Metric", "Value"])

    def _generate_data_quality_stats_datetime(self) -> pd.DataFrame:
        """
        Returns data quality stats for datetime column.
        """
        valid_dates = self._series.dropna()
        value_counts = self._series.value_counts()
        len_column = len(self._series)
        values = self._series.count()
        formatted_values = format_count_with_percentage(values, len_column)
        zeros = (valid_dates == 0).sum()
        formatted_zeros = format_count_with_percentage(zeros, len_column)  if zeros > 0 else "---"
        missings = self._series.isna().sum()
        formatted_missings = format_count_with_percentage(missings, len_column)  if missings > 0 else "---"
        distinct = valid_dates.nunique()
        formatted_distinct = format_count_with_percentage(distinct, len_column)  if distinct > 0 else "---"
        duplicates = self._series.duplicated().sum()
        formatted_duplicates = format_count_with_percentage(duplicates, len_column)  if duplicates > 0 else "---"
        dup_values = self._series.value_counts()[self._series.value_counts() > 1].count()
        formatted_dup_values = format_count_with_percentage(dup_values, len_column)  if dup_values > 0 else "---"
        stats = {
            "Values": formatted_values,
            "Zeros": formatted_zeros,
            "Missings": formatted_missings,
            "Distinct": formatted_distinct,
            "Duplicates": formatted_duplicates,
            "Dup. Values": formatted_dup_values
        }
        return pd.DataFrame(stats.items(), columns=["Metric", "Value"])

    def _generate_temporal_stats_datetime(self) -> pd.DataFrame:
        """
        Returns temporal patterns stats for datetime column.
        """
        valid_dates = self._series.dropna()
        stats = {}
        len_column = len(self._series)
        if not valid_dates.empty:
            date_range = pd.date_range(valid_dates.min(), valid_dates.max())
            
            valid_days = valid_dates.dt.normalize().unique()
            date_range_days = date_range.normalize().unique()
            
            all_days = len(date_range_days)
            missing_days = len(set(date_range_days) - set(valid_days))
            
            all_years = date_range.year.nunique()
            missing_years = len(set(date_range.year) - set(valid_dates.dt.year))
            
            all_weeks = date_range.to_period("W").nunique()
            missing_weeks = len(set(zip(date_range.year, date_range.isocalendar().week)) - 
                                set(zip(valid_dates.dt.year, valid_dates.dt.isocalendar().week)))
            
            all_months = date_range.to_period("M").nunique()
            missing_months = len(set(zip(date_range.year, date_range.month)) - 
                                    set(zip(valid_dates.dt.year, valid_dates.dt.month)))

            stats = {
                "Missing Years": format_count_with_percentage(missing_years, all_years) if missing_years > 0 else "---",
                "Missing Months": format_count_with_percentage(missing_months, all_months) if missing_months > 0 else "---",
                "Missing Weeks": format_count_with_percentage(missing_weeks, all_weeks) if missing_weeks > 0 else "---",
                "Missing Days": format_count_with_percentage(missing_days, all_days) if missing_days > 0 else "---",
                "Weekend Percentage": f"{valid_dates.dt.weekday.isin([5,6]).mean():.1%}",
                "Most Common Weekday": valid_dates.dt.day_name().mode()[0] if not valid_dates.empty else "N/A"
            }
        
        return pd.DataFrame(stats.items(), columns=["Metric", "Value"])
    
    def _generate_full_datetime_summary(self) -> pd.DataFrame:
        """
        Combines all datetime statistics into a single DataFrame.
        """
        column_name = self._series.name
        basic_stats = self._generate_basic_stats_datetime()
        data_quality_stats = self._generate_data_quality_stats_datetime()
        temporal_stats = self._generate_temporal_stats_datetime()
        # Concatenate all DataFrames along the columns (axis=1)
        full_summary = pd.concat([basic_stats, data_quality_stats, temporal_stats], axis=1)

        full_summary.columns = pd.MultiIndex.from_tuples(
            [('Summary', 'Metric'), ('Summary', 'Value'),
            ('Data Quality Stats', 'Metric'), ('Data Quality Stats', 'Value'),
            ('Temporal Stats', 'Metric'), ('Temporal Stats', 'Value')]

        )
        full_summary = add_empty_columns_for_df(full_summary, [2, 4])
        caption = f'Summary Statistics for "{column_name}" (Type: Datetime)'
        styled_summary = style_dataframe(full_summary, level=1, caption=caption, header_alignment='center')
        return styled_summary    

    # ========
    # Numeric
    # ========

    def _generate_summary_for_numeric(self) -> pd.DataFrame:
        """
        Generates basic summary statistics for numeric columns in HTML-friendly format.
        
        Note: Results are cached based on column name
        """
        len_column = len(self._series)
        # Calculate basic metrics
        values = self._series.count()
        values = format_count_with_percentage(values, len_column)
        
        # Missing values
        missing = self._series.isna().sum()
        missing = format_count_with_percentage(missing, len_column) if missing else '---'
        
        # Distinct values
        distinct = self._series.nunique()
        distinct = format_count_with_percentage(distinct, len_column)

        # Non-Duplicate 
        unique_once = self._series.value_counts()
        unique_once = (unique_once == 1).sum()
        unique_once = format_count_with_percentage(unique_once, len_column)

        # Duplicates
        duplicates = self._series.duplicated().sum()
        duplicates = format_count_with_percentage(duplicates, len_column) if duplicates > 0 else "---"

        # Count of values with duplicates
        values_with_duplicates = self._series.value_counts()[self._series.value_counts() > 1].count()        
        values_with_duplicates = format_count_with_percentage(values_with_duplicates, len_column) if duplicates != "---" else "---"

        # Zeros and negatives
        zeros = (self._series == 0).sum()
        zeros = format_count_with_percentage(zeros, len_column) if zeros > 0 else "---"
        
        negative = (self._series < 0).sum()
        negative = format_count_with_percentage(negative, len_column) if negative > 0 else "---"

        # Infinite
        infinite = np.isinf(self._series).sum()
        infinite = format_count_with_percentage(infinite, len_column) if infinite > 0 else "---"

        # Memory usage
        ram = round(self._series.memory_usage(deep=True) / 1048576)
        if ram == 0:
            ram = "<1 Mb"
        
        df_res = pd.DataFrame({
            "Total": [values],
            "Missing": [missing],
            "Distinct": [distinct],
            "Non-Duplicate": [unique_once],
            "Duplicates": [duplicates],
            "Dup. Values": [values_with_duplicates],
            "Zeros": [zeros],
            "Negative": [negative],
            "Memory Usage": [ram]
        }).T.reset_index()
        df_res.columns = ['Metric', 'Value']
        return df_res
    
    def _generate_percentiles_for_numeric(self) -> pd.DataFrame:
        """
        Generates percentile statistics for numeric columns.
        """
        percentiles = {
            "Max": self._series.max(),
            "99%": self._series.quantile(0.99),
            "95%": self._series.quantile(0.95),
            "75%": self._series.quantile(0.75),
            "50%": self._series.median(),
            "25%": self._series.quantile(0.25),
            "5%": self._series.quantile(0.05),
            "1%": self._series.quantile(0.01),
            "Min": self._series.min()
        }
        
        formatted = {k: format_number(v) for k, v in percentiles.items()}
        df_res = pd.DataFrame(formatted, index=[0]).T.reset_index()
        df_res.columns = ['Metric', 'Value']
        return df_res
    
    def _generate_stats_for_numeric(self) -> pd.DataFrame:
        """
        Generates statistical measures for numeric columns including trimmed mean.
        
        Args:
            column: Numeric pandas Series
            
        Returns:
            DataFrame with statistics in [Metric, Value] format
        """
     
        clean_col = self._series.dropna()
        
        stats_dict = {
            "Mean": clean_col.mean(),
            "Trimmed Mean (10%)": stats.trim_mean(clean_col, proportiontocut=0.1),
            "Mode": clean_col.mode()[0] if len(clean_col.mode()) == 1 else "Multiple",
            "Range": clean_col.max() - clean_col.min(),
            "IQR": clean_col.quantile(0.75) - clean_col.quantile(0.25),
            "Std": clean_col.std(),
            "MAD": robust.mad(clean_col),
            "Kurt": clean_col.kurtosis(),
            "Skew": clean_col.skew()
        }
        
        formatted = {
            k: format_number(v) if isinstance(v, (int, float)) else v 
            for k, v in stats_dict.items()
        }
        
        result = pd.DataFrame(formatted.items(), columns=['Metric', 'Value'])
        return result

    def _generate_value_counts_for_numeric(self, top_n: int = 9) -> pd.DataFrame:
        """
        Generates value counts for numeric columns with binning.
        """
        # Handle empty/missing data
        clean_col = self._series.dropna()
        len_column = len(self._series)
        if len(clean_col) == 0:
            return pd.DataFrame({"Message": ["No numeric data available"]})
        # Get top distinct values
        top_values = clean_col.value_counts().head(top_n)
        # Create a DataFrame for results
        df_res = pd.DataFrame({
            'Value': top_values.index,
            'Count': top_values.values,
        })
        df_res['Value'] = df_res['Value'].apply(lambda x: f'{x:.2f}' if x % 1 != 0 else f'{x:.0f}')
        # Format the Percent column to show "<1%" for values less than 1
        df_res['Count'] = df_res['Count'].apply(lambda x: format_count_with_percentage(x, len_column))
        return df_res.reset_index(drop=True)
    
    def _combine_numeric_stats(self, *stats_dfs: pd.DataFrame) -> pd.DataFrame:
        """
        Combines multiple statistics DataFrames into a single styled summary.
        
        Args:
            *stats_dfs: Variable number of DataFrames to combine
            
        Returns:
            Styled DataFrame with multi-level columns
        """
        # Concatenate all DataFrames along the columns (axis=1)
        combined_df = pd.concat(stats_dfs, axis=1)
        
        # Create multi-level column names
        column_tuples = []
        for i, df in enumerate(stats_dfs):
            section_name = [
                'Summary', 'Percentiles', 
                'Detailed Stats', 'Value Counts'
            ][i]
            for col in df.columns:
                column_tuples.append((section_name, col))
                
        combined_df.columns = pd.MultiIndex.from_tuples(column_tuples)
        return combined_df

    def _generate_full_numeric_summary(self) -> pd.DataFrame:
        """
        Generates complete styled summary for numeric columns.
        
        Args:
            column: Numeric pandas Series to analyze
            
        Returns:
            Styled DataFrame with all statistics
        """
        column_name = self._series.name
        column_type = get_column_type(self._series)
        
        # Generate all component statistics
        stats = [
            self._generate_summary_for_numeric(),
            self._generate_percentiles_for_numeric(),
            self._generate_stats_for_numeric(),
            self._generate_value_counts_for_numeric()
        ]
        
        # Combine and style
        full_summary = self._combine_numeric_stats(*stats)
        full_summary = add_empty_columns_for_df(full_summary, [2, 4, 6])
        caption = f'Summary Statistics for "{column_name}" (Type: {column_type})'
        return style_dataframe(
            full_summary, 
            level=1, 
            caption=caption, 
            header_alignment='center'
        )

    def _generate_histogram(
        self, 
        hist_mode: Literal['dual_hist_trim', 'dual_hist_qq'] = 'dual_hist_trim',
        lower_quantile: Optional[Union[float, int]] = None,
        upper_quantile: Optional[Union[float, int]] = None,
        title: str = None,
        height: str = None, 
        width: str = None,
        labels: dict = None,
        xaxis_type: str = None,
        yaxis_type: str = None,    
        plotly_kwargs: dict = None,       
        ) -> go.Figure:
        """
        Generates an histogram with box plot for numeric data.
        """
        builder = HistogramBuilder()
        annotations = []
        if title:
            if (lower_quantile or upper_quantile) and hist_mode == 'dual_hist_trim':
                quantile_for_title = ' (Right: '
                if lower_quantile:
                    quantile_for_title += f"from {lower_quantile} "
                if upper_quantile:
                    quantile_for_title += f"to {upper_quantile} "
                quantile_for_title += 'Quantile)'
                title += quantile_for_title
        else:
            if hist_mode == 'dual_hist_trim':
                title = f'Distribution of {self._series.name}'
                annotations=[
                    dict(
                        x=0.25,   
                        y=1.09,  
                        xref='paper',
                        yref='paper',
                        text="Original",
                        showarrow=False,
                        xanchor="center",
                        # font=dict(size=14)
                    ),
                    dict(
                        x=0.75,   
                        y=1.09,  
                        xref='paper',
                        yref='paper',
                        text=f"Trimmed from {lower_quantile:.2f} to {upper_quantile:.2f} Quantiles",
                        showarrow=False,
                        xanchor="center",
                        # font=dict(size=14)
                    )
                ]
            elif hist_mode == 'base':
                title = f'Distribution of {self._series.name}'
            else:
                title = f'Distribution and Q-Q Plot of {self._series.name}'
        if hist_mode in ['dual_hist_trim', 'dual_hist_qq']:
            height = 350 if not height else height
            width = 800 if not width else width
        else:
            height = 350 if not height else height
            width = 450 if not width else width
        if not labels:
            labels = dict()
        labels.setdefault(self._series.name, 'Value')
        params = dict(
            x=self._series,
            mode=hist_mode,
            title=title,
            labels=labels,
            xaxis_type=xaxis_type,
            yaxis_type=yaxis_type,
            renderer='jpg' if hist_mode == 'dual_hist_qq' else None,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            height = height,
            width = width,
            plotly_kwargs = plotly_kwargs,
            # width=config['sizes']['width'],
            # height=config['sizes']['height'],        
        )
        fig = builder.build(**params)
        if annotations and hist_mode == 'dual_hist_trim':
            fig.update_layout(annotations=annotations)
        return fig

    # ========
    # Categorical
    # ========
    def _generate_summary_for_categorical(self, is_categegory: bool) -> pd.DataFrame:
        """
        Generates comprehensive summary for categorical/text columns
        Returns DataFrame with metrics in consistent format
        """
        column_name = self._series.name
        # Basic counts
        total = len(self._series)
        non_null = self._series.count()
        missing = self._series.isna().sum()
        empty = (self._series.str.strip() == "").sum()
        
        # Distinct values analysis
        distinct = self._series.nunique()
        unique_once = self._series.value_counts().eq(1).sum()
        most_common = self._series.mode()
        most_common_count = (self._series == most_common[0]).sum() if len(most_common) > 0 else 0
        
        # Duplicates analysis
        exact_duplicates = self._series.duplicated().sum()
        
        # Fuzzy duplicates (case and whitespace insensitive)
        fuzzy_duplicates = (
            self._series.str.lower()
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
            .duplicated()
            .sum()
        )

        # Count of values with duplicates
        values_with_duplicates = self._series.value_counts()[self._series.value_counts() > 1].count()
        
        # Text length analysis 
        lengths = self._series.str.len()
        avg_len = lengths.mean()
        median_len = lengths.median()
        min_len = lengths.min()
        max_len = lengths.max()
        word_counts = self._series.str.split().str.len()
        avg_words = word_counts.mean()
        digit_count = self._series.str.count(r'\d') 
        avg_digit_ratio = (digit_count / lengths.replace(0, np.nan)).mean()
        most_common_length = lengths.mode()

        if not most_common_length.empty:
            most_common_length_value = most_common_length[0]
            most_common_length_count = lengths.value_counts().iloc[0]
            most_common_length_str = f"{most_common_length_value} ({most_common_length_count / len(lengths):.1%})"
        else:
            most_common_length_str = "N/A"

        ram = round(self._series.memory_usage(deep=True) / 1048576)
        if ram == 0:
            ram = "<1 Mb"
        # Prepare metrics
        quality_metrics = {
            "Total Values": format_count_with_percentage(total, total),
            "Missing Values": format_count_with_percentage(missing, total) if missing else '---',
            "Empty Strings": format_count_with_percentage(empty, total) if empty else '---',
            "Distinct Values": format_count_with_percentage(distinct, total) if distinct else '---',
            "Non-Duplicates": format_count_with_percentage(unique_once, total) if unique_once else '---',
            "Exact Duplicates": format_count_with_percentage(exact_duplicates, total) if exact_duplicates else '---',
            "Fuzzy Duplicates": format_count_with_percentage(fuzzy_duplicates, total) if fuzzy_duplicates else '---',
            "Values with Duplicates": format_count_with_percentage(values_with_duplicates, total) if values_with_duplicates else '---',
            "Memory Usage": ram,
        }
        text_metrics = {
            "Avg Word Count": f"{avg_words:.1f}" if not pd.isna(avg_words) else "N/A",
            "Max Length (chars)": f"{max_len:.1f}" if not pd.isna(max_len) else "N/A",
            "Avg Length (chars)": f"{avg_len:.1f}" if not pd.isna(avg_len) else "N/A",
            "Median Length (chars)": f"{median_len:.1f}" if not pd.isna(median_len) else "N/A",
            "Min Length (chars)": f"{min_len:.1f}" if not pd.isna(min_len) else "N/A",
            "Most Common Length": most_common_length_str,
            "Avg Digit Ratio": f'{avg_digit_ratio:.2f}'
        }
        
        quality_metrics_df = pd.DataFrame(quality_metrics.items(), columns=["Metric", "Value"])
        text_metrics_df = pd.DataFrame(text_metrics.items(), columns=["Metric", "Value"])
        value_counts_df = self._generate_value_counts_for_categorical()
        value_counts_df.columns = ["Metric", "Value"]
        # Concatenate all DataFrames along the columns (axis=1)
        if is_categegory:
            full_summary = pd.concat([quality_metrics_df, text_metrics_df, value_counts_df], axis=1)
            full_summary.columns = pd.MultiIndex.from_tuples(
                [('Summary', 'Metric'), ('Summary', 'Value'),
                ('Text Metrics', 'Metric'), ('Text Metrics', 'Value'),
                ('Value Counts', 'Metric'), ('Value Counts', 'Value')]
            )
            full_summary = add_empty_columns_for_df(full_summary, [2, 4])
        else:
            full_summary = pd.concat([quality_metrics_df, text_metrics_df], axis=1)
            full_summary.columns = pd.MultiIndex.from_tuples(
                [('Summary', 'Metric'), ('Summary', 'Value'),
                ('Text Metrics', 'Metric'), ('Text Metrics', 'Value')]
            )          
            full_summary = add_empty_columns_for_df(full_summary, [2])
        column_type = get_column_type(self._series)
        caption = f'Summary Statistics for "{column_name}" (Type: {column_type})'
        styled_summary = style_dataframe(full_summary, level=1, caption=caption, header_alignment='center')
        return styled_summary
        
    def _generate_value_counts_for_categorical(self, top_n: int = 9) -> pd.DataFrame:
        """
        Analyzes categorical value distribution and returns formatted counts.
        
        Args:
            column: Categorical pandas Series to analyze
            top_n: Number of top values to show (default: 10)
            
        Returns:
            DataFrame with columns: ['Value', 'Count', 'Percent']
            where Count is formatted as "count (percentage%)"
        """
        # Calculate frequencies
        total_count = len(self._series)
        value_counts = self._series.value_counts()
        
        # Handle empty data
        if total_count == 0:
            return pd.DataFrame({"Message": ["No data available"]})
        
        # Prepare top values
        results = []
        for val, count in value_counts.head(top_n).items():
            results.append({
                'Value': str(val)[:50] + '...' if len(str(val)) > 50 else str(val),
                'Count': count,
            })
        
        # Create DataFrame
        df_result = pd.DataFrame(results)
        # Format counts with percentages
        df_result['Count'] = df_result.apply(
            lambda row: format_count_with_percentage(row['Count'], total_count),
            axis=1
        )
        
        # # Add missing values row if present
        # if self._series.isna().sum() > 0:
        #     missing_row = {
        #         'Value': 'Missing',
        #         'Count': self._series.isna().sum(),
        #         'Percent': (self._series.isna().sum() / total_count) * 100
        #     }
        #     df_result = pd.concat([
        #         df_result,
        #         pd.DataFrame([missing_row])
        #     ], ignore_index=True)
        
        # # Add empty strings row if present (for string columns)
        # if pd.api.types.is_string_dtype(self._series) and (self._series == "").sum() > 0:
        #     empty_count = (self._series == "").sum()
        #     empty_row = {
        #         'Value': 'Empty Strings',
        #         'Count': empty_count,
        #         'Percent': (empty_count / total_count) * 100
        #     }
        #     df_result = pd.concat([
        #         df_result,
        #         pd.DataFrame([empty_row])
        #     ], ignore_index=True)
        
        return df_result[['Value', 'Count']]  # Return same columns as numeric version
    
    def _generate_bar_chart(
        self, top_n: int = 10, show_text: bool = True, labels: dict = None
    ) -> go.Figure:
        """
        Generates an interactive horizontal bar chart for categorical data.

        Args:
            column: Categorical pandas Series to visualize
            top_n: Number of top categories to display (default: 10)

        Returns:
            Plotly Figure object with these features:
            
            - Horizontal bars sorted by frequency
            - Value annotations
            - Adaptive text wrapping
            - Smart truncation for long labels
            - Responsive design settings

        Visual Features:
        
        - Top N categories by count
        - Percentage and absolute value labels
        - Color gradient by frequency
        - Dynamic label sizing
        - Mobile-optimized layout
        """
        # Prepare data - count values and calculate percentages
        value_counts = self._series.value_counts().nlargest(top_n)
        percentages = (value_counts / len(self._series)) * 100
        df_plot = pd.DataFrame(
            {
                "Category": value_counts.index,
                "Count": value_counts.values,
                "Percentage": percentages.round(1),
            }
        )

        # Truncate long category names
        max_label_length = 30
        df_plot["DisplayName"] = df_plot["Category"].apply(
            lambda x: (
                (x[:max_label_length] + "...")
                if len(x) > max_label_length
                else x
            )
        )

        # Create figure
        fig = px.bar(
            df_plot,
            x="Count",
            y="DisplayName",
            orientation="h",
            text="Percentage" if show_text else None,
            custom_data='Percentage',
            height=400,
            width=700,
            labels=labels,
        )

        # Style configuration
        fig.update_traces(
            texttemplate="%{text}%" if show_text else None,
            textfont_size=12,
            marker_line_width=0.5,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "Count: %{x:,}<br>"
                + "Percentage: %{customdata}%"
                + "<extra></extra>"
            ),
        )

        # Layout configuration
        fig.update_layout(
            title=f"Top {top_n} Categories in {self._series.name}",
            # title_x=0.5,
            # title_font_size=16,
            # margin=dict(
            #     l=120, r=40, t=80, b=40
            # ),  # Extra left margin for labels
            xaxis_title="Count",
            yaxis_title=None,
            yaxis={"categoryorder": "total ascending"},
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis={"gridcolor": "#f0f0f0", "tickformat": ","},
            uniformtext_minsize=8,
            uniformtext_mode="hide",
        )

        return CustomFigure(fig)

    def _generate_wordcloud(self, 
                       max_words: int = 100,
                       background_color: str = 'white',
                       colormap='viridis',
                       width: int = 800,
                       height: int = 400,
                       collocations: bool = True) -> plt.figure:
        """
        Generates a word cloud visualization for categorical/text data
        
        Args:
            column: Categorical or text column to analyze
            max_words: Maximum number of words to display (default: 100)
            background_color: Background color (default: 'white')
            width: Image width in pixels (default: 800)
            height: Image height in pixels (default: 400)
            stopwords: Set of words to exclude (default: None)
            collocations: Whether to include bigrams (default: True)
            
        Returns:
            Matplotlib Figure object with the word cloud visualization
        """
        # Prepare text data
        text = ' '.join(self._series.dropna().astype(str))
        # Create the word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            colormap=colormap,
            collocations=collocations,
        ).generate(text)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        fig.tight_layout()
        return fig
    
    def _generate_combined_charts(self, is_categegory: bool, top_n: int = 10, max_words: int = 100, title=None,
                                  height: int = None, width: int = None, show_text: bool = True, labels: dict = None) -> go.Figure:
        """
        Generates combined visualization with bar chart and properly scaled word cloud.
        Args:
            column: Categorical/text pandas Series to visualize
            top_n: Number of top categories for bar chart (default: 10)
            max_words: Maximum words for word cloud (default: 100)
            height, width : int
                Height and width for plot
        Returns:
            Plotly Figure with both visualizations (word cloud maintains aspect ratio)
        """
        if is_categegory:
            fig = self._generate_bar_chart(top_n, show_text, labels)
            fig.update_yaxes(categoryorder="total ascending")
            default_title = f'Value counts of {self._series.name}'
            height = height if height else 350
            width = width if width else 450
            fig.update_layout(
                title_text=default_title if not title else title,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=10, r=10, b=10, t=50),
                height=height,
                width=width,
            )
        else:
            title = title if title else f'WordCloud of {self._series.name}'
            text = ' '.join(self._series.dropna().astype(str))
            height = height if height else 400
            width = width if width else 700
            fig = create_wordcloud_plotly(text, max_words=max_words, width=width, height=height, return_fig=True)
        return CustomFigure(fig)
    
    def show_quantiles(self, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]) -> None:
        """
        Calculates specified quantile statistics for numeric columns.
        
        Parameters:
        -----------
        quantiles (list): A list of quantiles to calculate. Default is [0.05, 0.25, 0.5, 0.75, 0.95].
        
        Returns:
        --------
        pd.DataFrame: A DataFrame containing the quantile statistics.
        """
        quantiles = sorted(quantiles)  
        quantile_values = {f"{q}": self._series.quantile(q) for q in quantiles}
              
        formatted = {k: format_number(v) for k, v in quantile_values.items()}
        df_res = pd.DataFrame(formatted, index=[0]).T.reset_index()
        df_res.columns = ['Quantile', 'Value']
        caption = f'Quantiles in {self._series.name}'
        styled_df_res = style_dataframe(df_res, caption=caption, hide_columns=False)
        display(styled_df_res)    