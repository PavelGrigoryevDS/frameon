from typing import Optional, Union, List, Dict, Tuple, Any
from enum import Enum, auto
import io
import base64
from pandas.io.formats.style import Styler
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display
import numpy as np
from plotly.subplots import make_subplots
from frameon.utils.miscellaneous import style_dataframe, add_empty_columns_for_df, format_number, format_count_with_percentage, validate_is_DataFrame, is_categorical_column, is_text_column, is_int_column, is_datetime_column, is_float_column, get_column_type

__all__ = ['FrameOnInfo']

class FrameOnInfo:
    """
    Class containing methods for DataFrame information analysis.
    """
    def __init__(self, df: pd.DataFrame):
        self._df = df 

    def info(self) -> Styler:
        """
        Generates a styled overview table with key statistics about the DataFrame.

        Returns:
        --------
            A pandas Styler object with formatted overview table ready for HTML display          
        """
        # Calculate basic statistics
        total_rows = self._df.shape[0]
        total_cols = self._df.shape[1]
        ram = round(self._df.__sizeof__() / 1_048_576)
        if ram == 0:
            ram = "<1 Mb"
        # Calculate missing values
        total_cells = total_rows * total_cols
        missing_cells = self._df.isna().sum().sum()
        missing_cells = format_count_with_percentage(missing_cells, total_cells)  if missing_cells else '---'
        col_types = {
            'Text': sum(is_text_column(self._df[col]) for col in self._df.columns),
            'Categorical': sum(is_categorical_column(self._df[col]) for col in self._df.columns),
            'Int': sum(is_int_column(self._df[col]) for col in self._df.columns),
            'Float': sum(is_float_column(self._df[col]) for col in self._df.columns),
            'Datetime': sum(is_datetime_column(self._df[col]) for col in self._df.columns)
        }        
        # Calculate duplicate statistics
        exact_dups = self._calculate_duplicates_in_df(exact=True)
        fuzzy_dups = self._calculate_duplicates_in_df(exact=False)

        # Create summary table
        summary_data = {
            "Rows": format_number(total_rows),
            "Features": total_cols,
            "Missing cells": missing_cells,
            "Exact Duplicates": exact_dups,  
            "Fuzzy Duplicates": fuzzy_dups, 
            "Memory Usage (Mb)": ram,
        }

        # Convert to DataFrame and style
        summary_df = pd.DataFrame.from_dict(
            summary_data, orient="index", columns=["Value"]
        ).reset_index(names='Metric')
        # Convert to DataFrame and style
        col_types_df = pd.DataFrame.from_dict(
            col_types, 
            orient="index", 
            columns=["Value"]
        ).reset_index(names='Metric')
        col_types_df['Value'] = col_types_df['Value'].astype(str)
        # Concatenate all DataFrames along the columns (axis=1)
        full_summary = pd.concat([summary_df, col_types_df], axis=1)
        full_summary.columns = pd.MultiIndex.from_tuples(
            [('Summary', 'Metric'), ('Summary', 'Value'),
            ('Column Types', 'Metric'), ('Column Types', 'Value')]

        )
        full_summary = add_empty_columns_for_df(full_summary, [2])
        caption='Dataframe Overview'
        styled_summary = style_dataframe(full_summary, level=1, caption=caption, header_alignment='center')
        return styled_summary  

    def _calculate_duplicates_in_df(self, exact: bool = True) -> str:
        """
        Calculate duplicate rows statistics with configurable matching strictness.

        Args:
            exact: If True, finds exact duplicates. If False, finds fuzzy matches
                  (ignoring case and whitespace differences for string columns)

        Returns:
            Formatted string with count and percentage of duplicates.
            Returns "---" if no duplicates found.
        """
        try:
            if exact:
                # Count exact duplicates
                dup_count = self._df.duplicated().sum()
            else:
                # Check if there are any string columns
                if not any(self._df.dtypes == "object"):
                    return "---"  # No string columns to check for fuzzy duplicates

                # Count fuzzy duplicates (normalized strings)
                dup_count = (
                    self._df.apply(
                        lambda col: (
                            col.str.lower()
                            .str.strip()
                            .str.replace(r"\s+", " ", regex=True)
                            if col.dtype == "object"
                            else col
                        )
                    )
                    .duplicated(keep=False)
                    .sum()
                )

            if dup_count == 0:
                return "---"

            # Calculate percentage
            total_rows = len(self._df)
            percentage = (dup_count / total_rows) * 100

            # Format percentage based on magnitude
            if 0 < percentage < 1:
                percentage_str = "<1"
            elif 99 < percentage < 100:
                percentage_str = f"{percentage:.1f}".replace("100.0", "99.9")
            else:
                percentage_str = f"{round(percentage)}"

            # Format final output
            formatted_count = format_number(dup_count)
            return f"{formatted_count} ({percentage_str}%)"

        except Exception as e:
            # Graceful fallback for any calculation errors
            # print(f"Duplicate calculation error: {str(e)}")
            return "---"