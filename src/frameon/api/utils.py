import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Tuple, Literal, Optional
from textwrap import dedent
from pprint import pformat
import warnings
from IPython.display import display
from frameon.utils.miscellaneous import style_dataframe
__all__ = [
    'analyze_join_keys',
    'find_inconsistent_mappings',
    'haversine_vectorized',
]

def analyze_join_keys(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: Union[str, List[str]] = None,
    left_on: Union[str, List[str]] = None,
    right_on: Union[str, List[str]] = None,
    short_result: bool = True,
    only_coverage: bool = False,
    how: Literal['all', 'inner', 'left', 'right', 'outer'] = 'all'
) -> None:
    """
    Analyzes key relationship and coverage between two DataFrames.
    Prints a formatted summary of the results and returns detailed metrics.

    Parameters:
    -----------
    left_df : pd.DataFrame
        Left DataFrame for analysis
    right_df : pd.DataFrame
        Right DataFrame for analysis
    on : str or list, optional
        Column name(s) present in both DataFrames (like pd.merge)
    left_on : str or list, optional
        Column name(s) in left DataFrame (like pd.merge)
    right_on : str or list, optional
        Column name(s) in right DataFrame (like pd.merge)
    short_result : bool, optional
        Whether to show short result
    only_coverage: bool, default False
        Whether to show only coverage between tables
    how : str
        Join type to show in short result. Can be one of 'all', 'inner', 'left', 'right', 'outer'.
        
    Returns:
    --------
    None
    """
    # Input validation
    if left_df.empty or right_df.empty:
        raise ValueError("Input DataFrames cannot be empty")
    if short_result and how not in ['all', 'inner', 'left', 'right', 'outer']:
        raise ValueError("Invalid how. Can be one of ['all', 'inner', 'left', 'right', 'outer']")
    # Resolve keys (like pd.merge)
    keys = _resolve_keys(on, left_on, right_on)
    left_key, right_key = keys['left'], keys['right']

    # Validate keys exist in DataFrames
    for key in left_key:
        if key not in left_df.columns:
            raise ValueError(f"Left key column '{key}' not found in left DataFrame")
    for key in right_key:
        if key not in right_df.columns:
            raise ValueError(f"Right key column '{key}' not found in right DataFrame")

    # Prepare key data - handle composite keys by creating temporary columns
    left_key_str = left_df[left_key].astype(str).agg('|'.join, axis=1) if len(left_key) > 1 else left_df[left_key[0]]
    right_key_str = right_df[right_key].astype(str).agg('|'.join, axis=1) if len(right_key) > 1 else right_df[right_key[0]]

    # Calculate coverage metrics
    left_unique = set(left_key_str.dropna().unique())
    right_unique = set(right_key_str.dropna().unique())
    common_keys = left_unique & right_unique
    if only_coverage:
        print(f'Left only keys: {len(left_unique - right_unique)}')
        print(f'Right only keys: {len(right_unique - left_unique)}')
        return
    
    # Calculate key statistics
    left_counts = left_key_str.value_counts(dropna=False)
    right_counts = right_key_str.value_counts(dropna=False)
    left_total_duplicates = left_df[left_key].duplicated(keep=False).sum()
    right_total_duplicates = right_df[right_key].duplicated(keep=False).sum()

    # Determine relationship type
    max_left = left_counts.max()
    max_right = right_counts.max()

    relationship_map = {
        (1, 1): 'one-to-one (1:1)',
        (1, float('inf')): 'one-to-many (1:N)',
        (float('inf'), 1): 'many-to-one (N:1)',
        (float('inf'), float('inf')): 'many-to-many (N:M)'
    }

    # Calculate missing values
    left_missing = left_key_str.isna().sum()
    right_missing = right_key_str.isna().sum()
    inner_size, left_size, right_size, outer_size = _calculate_join_sizes(left_df, right_df, left_key, right_key).values()
    # Prepare results
    result = {
        'keys': {
            'left': left_key,
            'right': right_key
        },
        'relationship': {
            'type': relationship_map[
                (1 if max_left == 1 else float('inf'),
                 1 if max_right == 1 else float('inf'))
            ],
            'left_duplicates': (left_counts > 1).mean(),
            'right_duplicates': (right_counts > 1).mean(),
            'left_duplicate_count': (left_counts > 1).sum(),
            'right_duplicate_count': (right_counts > 1).sum(),
            'left_total_duplicates': left_total_duplicates,
            'right_total_duplicates': right_total_duplicates,
        },
        'join_sizes': {
            'left_rows': len(left_df),
            'right_rows': len(right_df),
            'inner': inner_size, 
            'left': left_size, 
            'right': right_size, 
            'outer': outer_size, 
        },
        'coverage': {
            'left_in_right': len(common_keys) / len(left_unique) if len(left_unique) > 0 else 0,
            'right_in_left': len(common_keys) / len(right_unique) if len(right_unique) > 0 else 0,
            'left_only': len(left_unique - right_unique),
            'right_only': len(right_unique - left_unique),
            'common': len(common_keys),
            'left_missing': left_missing,
            'right_missing': right_missing,
            'left_missing_pct': left_missing / len(left_df) if len(left_df) > 0 else 0,
            'right_missing_pct': right_missing / len(right_df) if len(right_df) > 0 else 0,
        },
        'counts': {
            'left': len(left_unique),
            'right': len(right_unique),
        },
        'samples': {
            'left_only': list(left_unique - right_unique)[:5],
            'right_only': list(right_unique - left_unique)[:5]
        }
    }
    _print_analysis_report(result, short_result=short_result, how=how)

def _calculate_join_sizes(left_df: pd.DataFrame, right_df: pd.DataFrame, left_key: str, right_key: str) -> dict:
    """
    Accurately calculates the sizes of joins with consideration of duplicate keys.
    Optimized version for large DataFrames.
    """
    left_keys = left_df[left_key].dropna()
    right_keys = right_df[right_key].dropna()

    left_counts = left_keys.value_counts(dropna=True)
    right_counts = right_keys.value_counts(dropna=True)

    common_keys = left_counts.index.intersection(right_counts.index)

    inner_size = (left_counts[common_keys] * right_counts[common_keys]).sum()
    left_only_size = left_counts.sum() - left_counts[common_keys].sum()
    right_only_size = right_counts.sum() - right_counts[common_keys].sum()

    return {
        'inner': int(inner_size),
        'left': int(inner_size + left_only_size),
        'right': int(inner_size + right_only_size),
        'outer': int(inner_size + left_only_size + right_only_size),
    }

def _print_analysis_report(results: Dict, short_result: bool, how: Literal['all', 'inner', 'left', 'right', 'outer'] = 'all') -> None:
    """Formats the analysis results into a comprehensive report."""
    keys = results['keys']
    rel = results['relationship']
    cov = results['coverage']
    cnt = results['counts']
    smp = results['samples']
    
    def format_percent(value):
        if value == 1.0:
            return "100%"
        elif value == 0.0:
            return "0%"

        rounded = round(value * 100, 3)
        
        if rounded >= 99.95:
            return "99.9%"
        elif rounded <= 0.05:
            return "0.1%"
        else:
            formatted = f"{value:.1%}"
            return formatted.replace(".0%", "%") if ".0%" in formatted else formatted

    report = f"""
    {' Join Analysis Report ':=^80}

    Relationship:
    - Type: {rel['type']}
    - Left total duplicates (keep=False): {rel['left_total_duplicates']:,} ({format_percent(rel['left_total_duplicates']/results['join_sizes']['left_rows'])} of rows)
    - Right total duplicates (keep=False): {rel['right_total_duplicates']:,} ({format_percent(rel['right_total_duplicates']/results['join_sizes']['right_rows'])} of rows)

    Coverage:
    - Left-only keys: {cov['left_only']:,}
    - Right-only keys: {cov['right_only']:,}

    Counts:
    - Left keys: {cnt['left']:,} unique
    - Right keys: {cnt['right']:,} unique
    - Common keys: {cov['common']:,}

    Missing Values:
    - Left missing: {cov['left_missing']:,} ({format_percent(cov['left_missing_pct'])})
    - Right missing: {cov['right_missing']:,} ({format_percent(cov['right_missing_pct'])})

    Join Result Sizes:
    - Left table size: {results['join_sizes']['left_rows']:,} rows
    - Right table size: {results['join_sizes']['right_rows']:,} rows
    - Inner join size: {results['join_sizes']['inner']:,} rows
    - Left join size: {results['join_sizes']['left']:,} rows
    - Right join size: {results['join_sizes']['right']:,} rows
    - Outer join size: {results['join_sizes']['outer']:,} rows
    """
    if short_result:
        output = []
        output.append(f"Type: {rel['type']}\n")
        output.append(f"Left-only keys: {cov['left_only']:,}\n")
        output.append(f"Right-only keys: {cov['right_only']:,}\n")
        output.append(f"Left table size: {results['join_sizes']['left_rows']:,} rows\n")
        output.append(f"Right table size: {results['join_sizes']['right_rows']:,} rows\n")
        report = pd.DataFrame({
            'Type': f"{rel['type'][-4:-1]}",
            'Left-only keys': f"{cov['left_only']:,}",
            'Right-only keys': f"{cov['right_only']:,}",
            'Left size': f"{results['join_sizes']['left_rows']:,}",
            'Right size': f"{results['join_sizes']['right_rows']:,}",
        }, index=[0])
        if how == 'inner':
            report['Inner join size'] = f"{results['join_sizes']['inner']:,}"
        elif how == 'left':
            report['Left join size'] = f"{results['join_sizes']['left']:,}"
        elif how == 'right':
            report['Right join size'] = f"{results['join_sizes']['right']:,}"
        elif how == 'outer':
            report['Outer join size'] = f"{results['join_sizes']['outer']:,}"
        else:
            report['Inner join size'] = f"{results['join_sizes']['inner']:,}"
            report['Left join size'] = f"{results['join_sizes']['left']:,}"
            report['Right join size'] = f"{results['join_sizes']['right']:,}"
            report['Outer join size'] = f"{results['join_sizes']['outer']:,}"
        display(style_dataframe(
            report
            , hide_columns=False
            , caption='Join info'
        ))
    else:
            print(dedent(report))

def _resolve_keys(
    on: Union[str, List[str]] = None,
    left_on: Union[str, List[str]] = None,
    right_on: Union[str, List[str]] = None
) -> Dict[str, List[str]]:
    """
    Resolves and validates key columns for analysis (consistent with pd.merge).

    Parameters:
    -----------
    on : str or list, optional
        Column name(s) present in both DataFrames
    left_on : str or list, optional
        Column name(s) in left DataFrame
    right_on : str or list, optional
        Column name(s) in right DataFrame

    Returns:
    --------
    dict
        Dictionary with resolved keys:
        - 'left': list of left key columns
        - 'right': list of right key columns

    Raises:
    -------
    ValueError
        If key configuration is invalid
    """
    if on is not None:
        if isinstance(on, str):
            on = [on]
        return {'left': on, 'right': on}

    if left_on is not None and right_on is not None:
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        if len(left_on) != len(right_on):
            raise ValueError(
                f"Mismatched key counts: left has {len(left_on)} keys, "
                f"right has {len(right_on)} keys"
            )
        return {'left': left_on, 'right': right_on}

    raise ValueError(
        "Must specify either 'on' (for same-named keys) "
        "or both 'left_on' and 'right_on' (for differently named keys)"
    )

def find_inconsistent_mappings(
    df: pd.DataFrame,
    key_column: str,
    value_column: str,
    return_summary: bool = False,
    verbose: bool = True,
    sample_size: int = 3
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Identifies rows where keys map to inconsistent value.
    Useful for detecting data integrity issues and many-to-many relationships.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to analyze
    key_column : str
        Column containing keys to check for consistency
    value_column : str
        Column containing values that should be consistently mapped
    return_summary : bool, optional
        Whether to return summary statistics or inconsistent rows
    verbose : bool, optional
        Whether to print a formatted summary of the results
    sample_size : int, optional
        Number of sample inconsistencies to display when verbose=True

    Returns:
    --------
    pd.DataFrame or dict
        If return_summary=False:
            DataFrame containing all rows with inconsistent mappings
        Else:
            Dictionary with summary statistics of inconsistencies
    """
    # Calculate value counts per key
    value_counts = (
        df.groupby(key_column, observed=True)[value_column]
        .agg(['nunique', 'count', lambda x: list(x.unique())])
        .rename(columns={
            'nunique': 'unique_values',
            'count': 'total_rows',
            '<lambda>': 'sample_values'
        })
    )

    # Find inconsistent keys
    inconsistent_keys = value_counts[value_counts['unique_values'] > 1]

    if return_summary:
        result = {
            'inconsistent_keys_count': len(inconsistent_keys),
            'affected_rows': inconsistent_keys['total_rows'].sum(),
            'inconsistency_rate': len(inconsistent_keys) / len(value_counts) if len(value_counts) > 0 else 0,
            'value_distribution': inconsistent_keys['unique_values'].value_counts().to_dict(),
            'most_inconsistent_keys': inconsistent_keys.nlargest(sample_size, 'unique_values').to_dict('index')
        }

        if verbose:
            print(_format_inconsistency_summary(result, key_column, value_column))

        return result

    # Return all rows with inconsistent mappings
    result = df[df[key_column].isin(inconsistent_keys.index)]
    if result.empty:
        warnings.warn(f"No inconsistent mappings found between {key_column} and {value_column}")
    elif verbose:
        sample = result.groupby(key_column).head(1).head(sample_size)
        print(f"\nSample of inconsistent mappings (showing first {sample_size} keys):")
        print(sample.to_string(index=False))

    return result

def _format_inconsistency_summary(results: Dict, key_col: str, value_col: str) -> str:
    """Formats inconsistency analysis results into a human-readable report."""
    report = f"""
    {' Inconsistent Mappings Analysis ':=^80}

    Columns analyzed:
    - Key: {key_col}
    - Value: {value_col}

    Summary:
    
    - Inconsistent keys: {results['inconsistent_keys_count']:,}
    - Affected rows: {results['affected_rows']:,}
    - Inconsistency rate: {results['inconsistency_rate']:.1%}

    Value Distribution:
    {pformat(results['value_distribution'], indent=4)}

    Most Inconsistent Keys (sample):
    {pformat(results['most_inconsistent_keys'], indent=4)}

    {'='*80}
    """
    return dedent(report)

def haversine_vectorized(
    lat1: Union[float, np.ndarray, pd.Series],
    lon1: Union[float, np.ndarray, pd.Series],
    lat2: Union[float, np.ndarray, pd.Series],
    lon2: Union[float, np.ndarray, pd.Series],
    unit: str = 'km'
) -> Union[float, np.ndarray]:
    """
    Calculate great-circle distances between geographic points using vectorized operations.
    Supports multiple distance units and efficient batch processing.

    Parameters:
    -----------
    lat1 : float, array-like
        Latitude(s) of first point(s) in degrees
    lon1 : float, array-like
        Longitude(s) of first point(s) in degrees
    lat2 : float, array-like
        Latitude(s) of second point(s) in degrees
    lon2 : float, array-like
        Longitude(s) of second point(s) in degrees
    unit : str, optional
        Unit for returned distances: 'km' (kilometers), 'm' (meters), or 'mi' (miles)

    Returns:
    --------
    float or ndarray
        Distance(s) between points in requested units
    """
    # Validate units
    valid_units = {'km', 'm', 'mi'}
    if unit not in valid_units:
        raise ValueError(f"Invalid unit. Must be one of {valid_units}")
    # Validate coordinates
    lat1 = np.array(lat1)
    lon1 = np.array(lon1)
    lat2 = np.array(lat2)
    lon2 = np.array(lon2)
    if not ((-90 <= lat1[~np.isnan(lat1)]) & (lat1[~np.isnan(lat1)] <= 90)).all():
        raise ValueError("Latitude(s) of first point(s) must be between -90 and 90 degrees")
    if not ((-180 <= lon1[~np.isnan(lon1)]) & (lon1[~np.isnan(lon1)] <= 180)).all():
        raise ValueError("Longitude(s) of first point(s) must be between -180 and 180 degrees")
    if not ((-90 <= lat2[~np.isnan(lat2)]) & (lat2[~np.isnan(lat2)] <= 90)).all():
        raise ValueError("Latitude(s) of second point(s) must be between -90 and 90 degrees")
    if not ((-180 <= lon2[~np.isnan(lon2)]) & (lon2[~np.isnan(lon2)] <= 180)).all():
        raise ValueError("Longitude(s) of second point(s) must be between -180 and 180 degrees")
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)

    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in km
    r = 6371.0

    # Convert to requested units
    if unit == 'm':
        return c * r * 1000
    elif unit == 'mi':
        return c * r * 0.621371
    return c * r
