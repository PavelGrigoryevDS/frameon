from typing import Any, Union
import pandas as pd
import numpy as np
from IPython.display import display
from typing import Union, List, Dict, Tuple, Any, Optional, Literal
from pandas.io.formats.style import Styler

__all__ = []

def validate_is_DataFrame(df: pd.DataFrame):
    """Validate input DataFrame"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be pandas DataFrame")
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

def is_categorical_column(column: pd.Series) -> bool:
    """
    Determines if a column should be treated as categorical based on its properties.
    
    Args:
        col: Name of the column to check
        
    Returns:
        bool: True if column is considered categorical, False otherwise
        
    Logic:
        - Column is categorical if:
        1. Ratio of unique values to total values < 50% (low cardinality), OR
        2. Column has pandas Categorical dtype
    """
    
    return (not pd.api.types.is_datetime64_any_dtype(column) and 
            not pd.api.types.is_numeric_dtype(column) and 
            (column.nunique() / len(column) < 0.2) or 
            isinstance(column.dtype, pd.CategoricalDtype))

def is_text_column(column: pd.Series) -> bool:
    """
    Determines if a column contains text data (non-categorical strings).
    
    Args:
        col: Name of the column to check
        
    Returns:
        bool: True if column contains text data, False otherwise
        
    Logic:
        - Column is text if:
        1. Has dtype 'object', AND
        2. Does not meet categorical criteria
    """
    
    return (pd.api.types.is_string_dtype(column.dropna()) and 
            not is_categorical_column(column))

def is_int_column(column: pd.Series) -> bool:
    """
    Determines if a column contains integer values (including nullable integers).
    
    Args:
        col: Name of the column to check
        
    Returns:
        bool: True if column contains only integers, False otherwise
        
    Logic:
        - Column is integer if:
        1. Is numeric dtype, AND
        2. All non-null values are whole numbers
    """
    if not pd.api.types.is_numeric_dtype(column):
        return False
        
    col_clean = column.dropna()
    if col_clean.empty:
        return False
        
    # Handle nullable integer types (Int8, Int16, etc.)
    if pd.api.types.is_integer_dtype(column):
        return True
        
    # Check if all values are whole numbers
    return (col_clean % 1 == 0).all()

def is_datetime_column(column: pd.Series) -> bool:
    """
    Determines if a column contains datetime values.
    
    Args:
        col: Name of the column to check
        
    Returns:
        bool: True if column contains datetime values, False otherwise
        
    Note:
        - Recognizes both datetime64[ns] and timezone-aware datetimes
    """
        
    return pd.api.types.is_datetime64_any_dtype(column)

def is_float_column(column: pd.Series) -> bool:
    """
    Determines if a column contains floating-point values (excluding integers).
    
    Args:
        col: Name of the column to check
        
    Returns:
        bool: True if column contains floating-point numbers, False otherwise
        
    Logic:
        - Column is float if:
        1. Is numeric dtype, AND
        2. Contains decimal numbers (not all whole numbers), AND
        3. Is not a datetime column
    """
    
    # First check if it's numeric and not datetime
    if not pd.api.types.is_numeric_dtype(column) or is_datetime_column(column):
        return False
        
    # Exclude integer columns
    if is_int_column(column):
        return False
        
    # Handle empty columns
    col_clean = column.dropna()
    if col_clean.empty:
        return False
        
    # Check if contains decimal numbers
    return not (col_clean % 1 == 0).all()    

def get_column_type(column: pd.Series) -> str:
    """
    Determines the data type of a specified column in the DataFrame.
    Parameters:
    - col (str): The name of the column for which to determine the data type.
    Returns:
    - str: A string representing the type of the column. Possible values are:
        - "Categorical" for categorical columns
        - "Text" for string (text) columns
        - "Integer" for integer columns
        - "Float" for float (decimal) columns
        - "Datetime" for datetime columns
        - "Unknown" if the column type is not recognized.
    """
    if is_categorical_column(column):
        return "Categorical"
    elif is_text_column(column):
        return "Text"
    elif is_int_column(column):
        return "Integer"
    elif is_float_column(column):
        return "Float"
    elif is_datetime_column(column):
        return "Datetime"
    else:
        return "Unknown"    
    
def format_number(num: Union[int, float]) -> str:
        """
        Format numbers with appropriate scaling and decimal precision for display.

        Args:
            num: Number to format (int or float)

        Returns:
            Formatted string with:
            - Thousands separators for large numbers
            - Appropriate unit suffixes (k, m, b)
            - Smart decimal precision

        Examples:
            >>> _format_number(1234.5678)
            '1,234.57'

            >>> _format_number(1_500_000)
            '1.50m'

            >>> _format_number(3_000_000_000)
            '3.00b'
        """
        # Handle None/NaN cases
        if pd.isna(num):
            return "NA"

        # Handle zero case
        if num == 0:
            return "0"

        abs_num = abs(num)

        # Determine appropriate scaling
        if abs_num < 1_000:
            # Small numbers - format with 2 decimals if not whole
            if isinstance(num, int) or num.is_integer():
                return f"{int(num):,}"
            return f"{num:,.2f}"

        elif abs_num < 1_000_000:
            scaled = num / 1_000
            return f"{scaled:,.2f}k"

        elif abs_num < 1_000_000_000:
            scaled = num / 1_000_000
            return f"{scaled:,.2f}m"

        elif abs_num < 1_000_000_000_000:
            scaled = num / 1_000_000_000
            return f"{scaled:,.2f}b"

        else:
            scaled = num / 1_000_000_000_000
            return f"{scaled:,.2f}t"

def format_count_with_percentage(
    count: int, total: int, precision: int = 0
) -> str:
    """
    Formats a count with its percentage of total in a human-readable way.

    Args:
        count: The subset count to format
        total: The total count for percentage calculation
        precision: Number of decimal places for percentages (default: 1)

    Returns:
        Formatted string in "count (percentage%)" format.
        Special cases:
        - Returns "0" if total is 0
        - Returns "count (100%)" if count equals total
        - Handles edge percentages (<1%, >99%)

    Examples:
        >>> _format_count_with_percentage(5, 100)
        '5 (5.0%)'

        >>> _format_count_with_percentage(999, 1000)
        '999 (99.9%)'

        >>> _format_count_with_percentage(1, 1000)
        '1 (<1%)'
    """
    # Handle zero total case
    if total == 0:
        return "0"

    # Handle 100% case
    if count == total:
        return f"{format_number(count)} (100%)"

    percentage = (count / total) * 100

    # Format percentage based on magnitude
    if percentage < 1:
        percentage_str = f"<{10**-precision}"  # <1% or <0.1% etc
    elif percentage > 99 and percentage < 100:
        percentage_str = f"{100 - 10**-precision:.{precision}f}"  # 99.9%
    else:
        percentage_str = f"{percentage:.{precision}f}"

    # Remove trailing .0 for whole numbers when precision=1
    if precision == 1 and percentage_str.endswith(".0"):
        percentage_str = percentage_str[:-2]

    return f"{format_number(count)} ({percentage_str}%)"    

def add_empty_columns_for_df(df: pd.DataFrame, positions: list) -> pd.DataFrame:
    """
    Adds empty columns to the DataFrame at specified positions.
    Args:
        df (pd.DataFrame): The input DataFrame to which empty columns will be added.
        positions (list): A list of indices after which to insert empty columns. Must verify 0 <= loc <= len(columns).
    Returns:
        pd.DataFrame: The modified DataFrame with empty columns added.
    Raises:
        Exception: If there is an error inserting the empty columns at the specified positions.
    """
    # Sort positions in descending order to avoid index shifting issues
    for i, pos in enumerate(positions):
        try:
            # Insert empty column with 10 spaces, each column must be unique
            df.insert(pos + i, ' ' * (i + 1), " " * 10)  
        except Exception as e:
            print(f"Error inserting at position {pos}: {e}")
    df = df.fillna('')  # Fill NaN values with empty strings
    return df

def style_dataframe(df: pd.DataFrame, hide_index: bool = True, 
                    hide_columns: bool = True, level: int = 0, formatters: dict = None,
                    caption: str = None, header_alignment='left', caption_font_size=16) -> Styler:
    """
    Styles the given DataFrame with options to hide index and columns.
    Args:
        df (pd.DataFrame): The input DataFrame to be styled.
        hide_index (bool, optional): Whether to hide the index. Defaults to True.
        hide_columns (bool, optional): Whether to hide the specified level of columns. Defaults to True.
        level (int, optional): The level of the columns to hide if hide_columns is True. Defaults to 0.
        caption (str, optional): The caption for the styled DataFrame. Defaults to "Datetime Summary Statistics".
    Returns:
        pd.io.formats.style.Styler: The styled DataFrame.
    """
    if caption:
        styled_df = df.style.set_caption(caption)
    else:
        styled_df = df.style
    styled_df = (styled_df.set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [
                                ("font-size", f"{caption_font_size}px"),
                                ("text-align", "left"),
                                ("font-weight", "bold"),
                                ("white-space", "nowrap"),  # Prevent caption from wrapping
                            ],
                        },
                        {
                            "selector": "th",
                            "props": [("text-align", header_alignment)]  # Align header text to the left
                        }
                    ]
                )
                .set_properties(**{"text-align": "left"}))
    if formatters:
        styled_df = styled_df.format(formatters)    
    if hide_index:
        styled_df = styled_df.hide(axis="index")
    if hide_columns:
        styled_df = styled_df.hide(axis="columns", level=level)
    return styled_df        

def analyze_anomalies_all_categories(
    df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    pct_diff_threshold: float = 0,
    include_columns: Optional[Union[str, List[str]]] = None,
    exclude_columns: Optional[Union[str, List[str]]] = None,    
) -> Union[None, pd.DataFrame]:
    """
    Enhanced analysis of anomalies across all categorical columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original DataFrame (must contain value column and all categories)
    anomaly_df : pd.DataFrame
        DataFrame with anomalies
    pct_diff_threshold : float
        Minimum % difference to include in results (from -100 to 100)
    include_columns : str or List[str], optional
        Specific categorical columns to include (None for all)
    exclude_columns : str or List[str], optional
        Categorical columns to exclude from analysis

    Returns:
    --------
    dict
        {
            'long_format': DataFrame with columns [Column, Category, % Diff],
            'wide_format': Pivoted DataFrame with categories as rows
        }
    """
    # Identify categorical columns (exclude 'value' column)
    # categorical_cols = [col for col in df.columns 
    #                    if col != 'value' and is_categorical_column(df[col])]
    categorical_cols = [col for col in df.columns 
                       if is_categorical_column(df[col])]
    if isinstance(include_columns, str):
        include_columns = [include_columns]
    if isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]
    # Apply column filters
    if include_columns is not None:
        categorical_cols = [col for col in categorical_cols if col in include_columns]
    if exclude_columns is not None:
        categorical_cols = [col for col in categorical_cols if col not in exclude_columns]    
    
    if not categorical_cols:
        return None
    all_results = []
    for col in categorical_cols:
        try:
            # Calculate base statistics
            total_count = len(df)
            anomaly_count = len(anomaly_df)
            
            # Group data for percentage calculations
            anomaly_stats = (
                anomaly_df.groupby(col, observed=False, dropna=False)
                .size()
                .reset_index(name='anomaly_count')
            )
            category_stats = (
                df.groupby(col, observed=False, dropna=False)
                .size()
                .reset_index(name='category_count')
            )
            # Merge and calculate metrics
            result = pd.merge(
                category_stats,
                anomaly_stats,
                on=col,
                how='left'
            )
            result[['anomaly_count', 'category_count']] = result[['anomaly_count', 'category_count']].fillna(0)
            # Filter out categories with zero anomalies
            result = result[result['anomaly_count'] > 0]
            
            # Calculate all metrics
            result['anomaly_pct'] = result['anomaly_count'] / anomaly_count * 100
            result['category_pct'] = result['category_count'] / total_count * 100
            result['anomaly_rate'] = result['anomaly_count'] / result['category_count'] * 100
            result['pct_diff'] = result['anomaly_pct'] - result['category_pct']
            
            # Apply thresholds and limits
            result = result[result['pct_diff'] >= pct_diff_threshold]
            if not result.empty:
                # Format counts as strings
                result['category_count'] = result['category_count'].astype(int).astype(str)
                result['anomaly_count'] = result['anomaly_count'].astype(int).astype(str)
                
                # Format percentages with 2 decimal places
                result['category_pct'] = result['category_pct'].round(1).astype(str) + '%'
                result['anomaly_pct'] = result['anomaly_pct'].round(1).astype(str) + '%'
                result['pct_diff'] = result['pct_diff']
                result['anomaly_rate'] = result['anomaly_rate'].round(1).astype(str) + '%'
                
                # Create display version
                display_df = result[[
                    col,
                    'category_count',
                    'anomaly_count',
                    'category_pct',
                    'anomaly_pct',
                    'pct_diff',
                    'anomaly_rate'
                ]].copy()
                
                display_df.columns = [
                    'Category',
                    'Total',
                    'Anomaly',
                    'Total %',
                    'Anomaly %',
                    '% Diff',
                    'Anomaly Rate'
                ]
                
                # Add column name as additional information
                display_df['Column'] = col
                display_df = display_df[['Column', 'Category', 'Total', 'Anomaly', 'Anomaly Rate', 
                                       'Total %', 'Anomaly %', '% Diff']]
                
                all_results.append(display_df)
                
        except Exception as e:
            print(f"Error processing column {col}: {str(e)}")
            continue

    if not all_results:
        return pd.DataFrame(columns=['Column', 'Category', '% Diff'])
    
    # Combine results
    result = pd.concat(all_results, ignore_index=True)

    return result.sort_values('% Diff', ascending=False)

def restore_full_index(
    df: pd.DataFrame,
    date_col: str,
    group_cols: list[str],
    freq: str = 'ME',
    fill_value: Union[str, int, float] = 0
) -> pd.DataFrame:
    """
    Restores a full index for a DataFrame by filling in missing dates and categories.
    This function takes a DataFrame, a date column, and a list of grouping columns.
    It creates a full MultiIndex by generating all possible combinations of dates
    (within the range of the date column) and unique values of the grouping columns.
    Missing values are filled with the specified fill_value.
    
    Parameters:
    ----------
    df (pd.DataFrame): The input DataFrame containing the data.
    date_col (str): The name of the column in `df` that contains the dates.
    group_cols (list of str): A list of column names in `df` that are used for grouping.
    freq (str, optional): The frequency for the date range. Default is 'ME' (month end).
    fill_value (str, int, float, optional): The value to fill missing entries with. Default is 0.
    
    Returns:
        pd.DataFrame: A DataFrame with a full index, where missing dates and categories are filled in.
    """
    # Check if date_col exists in the DataFrame
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in the DataFrame. Available columns: {list(df.columns)}")
    
    # Check if all group_cols exist in the DataFrame
    for col in group_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame. Available columns: {list(df.columns)}")

    # Generate the full date range based on the minimum and maximum dates in the DataFrame
    date_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq=freq)

    # Create a MultiIndex from the Cartesian product of the date range and unique values of the grouping columns
    full_index = pd.MultiIndex.from_product(
        [date_range] + [df[col].unique() for col in group_cols],
        names=[date_col] + group_cols
    )

    # Set the index of the DataFrame to the date and grouping columns, then reindex to the full index
    df = df.set_index([date_col] + group_cols).reindex(full_index, fill_value=fill_value).reset_index()

    return df


def convert_to_dataframe(
    data: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, dict, list]] = None,
    x: Optional[Union[str, pd.Series, np.ndarray, list]] = None,
    y: Optional[Union[str, pd.Series, np.ndarray, list]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convert various input formats to a DataFrame suitable for plotting.
    Handles cases similar to plotly express histogram.
    
    Parameters
    ----------
    data : DataFrame, Series, array-like, or dict, optional
        Input data structure
    x : str, Series, array-like, or list, optional
        Column name or data for x-axis
    y : str, Series, array-like, or list, optional
        Column name or data for y-axis
    **kwargs : dict
        Additional arguments
        
    Returns
    -------
    pd.DataFrame
        DataFrame suitable for plotting
    """
    
    # Check if data is already a DataFrame
    if data is not None:
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, pd.Series):
            if data.name is None:
                data = data.to_frame(name='value')
            else:
                data = data.to_frame()
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            data.columns = range(data.shape[1])
        elif isinstance(data, list):
            if all(isinstance(row, list) for row in data):
                data = pd.DataFrame(data)
                data.columns = range(data.shape[1])
            else:
                data = pd.DataFrame({'value': data})
        elif isinstance(data, dict):
            data = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported type for 'data'. Must be one of DataFrame, Series, array-like, or dict.")
    
    # Check if x is provided and is a valid type
    if x is not None:
        if isinstance(x, pd.Series):
            col_name = x.name if x.name is not None else 'x'
            data = x.to_frame(name=col_name)
            kwargs['x'] = col_name
        elif isinstance(x, np.ndarray):
            data = pd.DataFrame({'x': x})
            kwargs['x'] = 'x'
        elif isinstance(x, list):
            data = pd.DataFrame({'x': x})
            kwargs['x'] = 'x'
        elif isinstance(x, str):
            if data is not None and x in data.columns:
                kwargs['x'] = x
            else:
                raise ValueError("Unsupported type for 'x'. Must be a str (column name in data), Series, or array-like.")
        else:
            raise ValueError("Unsupported type for 'x'. Must be a str (column name in data), Series, or array-like.")
    
    # Check if y is provided and is a valid type
    if y is not None:
        if isinstance(y, pd.Series):
            col_name = y.name if y.name is not None else 'y'
            if data is not None:
                data[col_name] = y
            else:
                data = y.to_frame(name=col_name)
            kwargs['y'] = col_name
        elif isinstance(y, np.ndarray):
            if data is not None:
                data['y'] = y
            else:
                data = pd.DataFrame({'y': y})
            kwargs['y'] = 'y'
        elif isinstance(y, list):
            if data is not None:
                data['y'] = y
            else:
                data = pd.DataFrame({'y': y})
            kwargs['y'] = 'y'
        elif isinstance(y, str):
            if data is not None and y in data.columns:
                kwargs['y'] = y
            else:
                raise ValueError("Unsupported type for 'y'. Must be a str (column name in data), Series, or array-like.")
        else:
            raise ValueError("Unsupported type for 'y'. Must be a str (column name in data), Series, or array-like.")
    
    # If no valid data or x/y provided, raise an error
    if data is None:
        raise ValueError("Either 'data' or 'x/y' data must be provided. Supported types for 'data' are DataFrame, Series, array-like, or dict.")
    
    return data
