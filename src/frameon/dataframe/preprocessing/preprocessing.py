import warnings
from enum import auto
from typing import (TYPE_CHECKING, Callable, Dict, List, Literal, Optional,
                    Sequence, Any, Tuple, Union)
from sklearn.experimental import enable_iterative_imputer
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from scipy.stats import t
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
from statsmodels.robust import mad
from datetime import timedelta
from pandas.tseries.offsets import DateOffset

from frameon.utils.miscellaneous import (analyze_anomalies_all_categories,
                                              get_column_type,
                                              is_categorical_column,
                                              is_datetime_column, is_float_column,
                                              is_int_column, is_text_column,
                                              style_dataframe)
from frameon.utils.plotting import plot_utils

if TYPE_CHECKING: # pragma: no cover
    from frameon.core.base import FrameOn, SeriesOn


__all__ = ['FrameOnPreproc']

class FrameOnPreproc:
    """
    Class containing methods for Dataframe preprocessing.
    """

    def __init__(self, df: "FrameOn"):
        self._df = df
 
    def impute_missing(
        self,
        target_cols: Union[str, Sequence[str]],
        auxiliary_cols: Union[Literal['all'], str, Sequence[str]],
        method: Literal['simple', 'knn', 'iterative'] = 'simple',
        strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = 'median',
        n_neighbors: int = 5,
        sample_size: Optional[int] = None,
        random_state: int = 42,
        standardize: bool = False,
        imputer_params: Optional[Dict[str, Any]] = None,
        inplace: bool = False
    ) -> Union[None, pd.DataFrame]:
        """
        Perform missing value imputation on specified numerical columns.
        
        Parameters:
        -----------
        target_cols : str or list
            Numerical columns to impute (must contain missing values)
            
        auxiliary_cols : str or list, default='all'
            Columns to use as features for imputation. Can include:
            
            - Numerical columns (used directly)
            - Categorical columns (one-hot encoded)
            - Datetime columns (feature engineered)
            
            Does not include text columns.
            
        method : {'simple', 'knn', 'iterative'}, default='simple'
            Imputation strategy:
            
            - simple: Fast univariate imputation
            - knn: Nearest neighbors-based imputation
            - iterative: Multivariate imputation using chained equations
            
        strategy : str, default='median'
            Strategy for SimpleImputer: ['mean', 'median', 'most_frequent', 'constant']
            
        n_neighbors : int, default=5
            Number of neighbors for KNNImputer
            
        sample_size : int, optional
            Subsample size for large datasets optimization
            
        random_state : int, default=42
            Random seed for reproducibility
            
        standardize : bool, default=False
            Whether to standardize features before imputation.
            Recommended for knn and iterative methods.
            
        imputer_params : dict, optional
            Additional parameters for IterativeImputer:
            
            - estimator: sklearn estimator (default=BayesianRidge())
            - max_iter: int (default=10)
            - tol: float (default=1e-3)
            
        inplace : bool, default=False
            Whether to modify the original DataFrame
            
        Returns:
        --------
        pd.DataFrame or None
            DataFrame with imputed values or None if inplace=True
        """
        # Validate not empty DataFrame
        if self._df.empty:
            raise ValueError(
                "DataFrame is empty."
            )   
        # Validate method
        valid_methods = {'simple', 'knn', 'iterative'}
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
        
        # Validate sample size
        if sample_size and sample_size > len(self._df):
            sample_size = len(self._df)
            
        # Valid strategies
        valid_strategies = {'mean', 'median', 'most_frequent', 'constant'}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of: {valid_strategies}")
        # Validate and prepare input parameters
        target_cols = self._validate_target_cols(target_cols)
        auxiliary_cols = self._prepare_auxiliary_cols(auxiliary_cols, target_cols)
        
        # Check at least one non-NA value exists in each target column
        for col in target_cols:
            if self._df[col].notna().sum() == 0:
                raise ValueError(f"Target column '{col}' contains only NA values - nothing to impute from")
            
        # Prepare data matrix for imputation
        full_data, target_idx = self._prepare_data(
            target_cols,
            auxiliary_cols,
            method,
            standardize
        )

        # Prepare TRAINING data (with sampling if specified)
        if sample_size and len(self._df) > sample_size:
            train_data = resample(
                full_data, 
                n_samples=sample_size, 
                random_state=random_state
            )
        else:
            train_data = full_data.copy()   

        # Create and configure imputer
        imputer = self._create_imputer(
            method,
            strategy,
            n_neighbors,
            random_state,
            imputer_params or {}
        )
        
        # Perform imputation
        imputer.fit(train_data)
        imputed_values = imputer.transform(full_data)
        
        # Merge results back into original DataFrame
        result_df = self._merge_results(
            imputed_values[:, target_idx],
            target_cols,
            full_data.index
        )
        return self._return_result(result_df, inplace)

    def restore_full_index(
        self,
        date_cols: Union[str, List[str]],
        group_cols: Union[str, List[str]],
        freq: Optional[Union[str, timedelta, DateOffset, Dict[str, Union[str, timedelta, DateOffset]]]] = None,
        fill_value: Optional[Union[str, int, float]] = None,
        inplace: bool = False
    ) -> Union[None, pd.DataFrame]:
        """
        Restores a full index for a DataFrame by filling in missing dates and categories.
        This function takes a DataFrame, a date column, and a list of grouping columns.
        It creates a full MultiIndex by generating all possible combinations of dates
        (within the range of the date column) and unique values of the grouping columns.
        Missing values are filled with the specified fill_value.
        
        Parameters:
        -----------
        date_col : str
            The name of the column in `df` that contains the dates.
        group_cols : list of str
            A list of column names in `df` that are used for grouping.
        freq : str, optional
            The frequency for the date range. Default is 'ME' (month end).
        fill_value : str, int, float, optional
            The value to fill missing entries with. Default is 0.
        inplace : bool, default=False
            Whether to modify the original DataFrame
            
        Returns:
        --------
        pd.DataFrame or None
            DataFrame with imputed values or None if inplace=True
        """
        df = self._df
        # Convert to lists if single values passed
        date_cols = [date_cols] if isinstance(date_cols, str) else date_cols
        group_cols = [group_cols] if isinstance(group_cols, str) else group_cols

        # Check all columns exist in DataFrame
        missing_cols = [col for col in date_cols + group_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")        
        # Convert to lists if single values passed
        date_cols = [date_cols] if isinstance(date_cols, str) else date_cols
        group_cols = [group_cols] if isinstance(group_cols, str) else group_cols
        index_cols = date_cols + group_cols
        duplicates = df[index_cols].duplicated(keep=False)
        
        if duplicates.any():
            dup_data = df[duplicates].sort_values(index_cols)
            
            error_msg = (
                f"Found {duplicates.sum()} duplicate rows in index columns:\n"
                f"Index columns: {index_cols}\n\n"
                "Solutions:\n"
                "1. Use groupby().agg() to aggregate duplicates\n"
                "2. Add unique identifier column\n"
                "3. Use drop_duplicates() if appropriate"
            )
            raise ValueError(error_msg)

        # Validate all columns exist
        for col in date_cols + group_cols:
            if col not in df.columns:
                raise ValueError(f'Date column "{col}" not found for time-based operations')

        # Generate date ranges for each date column
        date_ranges = []
        for col in date_cols:
            date_ranges.append(pd.date_range(df[col].min(), df[col].max(), freq=freq))

        # Create full index
        if group_cols:
            # Case with grouping columns
            full_index = pd.MultiIndex.from_product(
                date_ranges + [df[col].unique() for col in group_cols],
                names=date_cols + group_cols
            )
        else:
            # Case with only date columns
            if len(date_ranges) == 1:
                full_index = date_ranges[0]
            else:
                full_index = pd.MultiIndex.from_product(date_ranges, names=date_cols)
            if len(date_cols) == 1:
                full_index.name = date_cols[0]
        # Reindex to the full index
        result_df = df.set_index(date_cols + group_cols).reindex(full_index, fill_value=fill_value).reset_index()
        return self._return_result(result_df, inplace)

    def _validate_target_cols(self, target_cols: Union[str, Sequence[str]]) -> List[str]:
        """Validate and normalize target columns."""
        # Convert to list if single column
        if isinstance(target_cols, str):
            target_cols = [target_cols]
            
        # Check all target columns exist and are numerical
        numeric_cols = self._df.select_dtypes(include=np.number).columns
        missing = [col for col in target_cols if col not in self._df.columns]
        if missing:
            raise ValueError(f"Target columns not found in DataFrame: {missing}")
            
        non_numeric = [col for col in target_cols if col not in numeric_cols]
        if non_numeric:
            raise ValueError(f"Target columns must be numerical: {non_numeric}")
            
        # Check for missing values
        if self._df[target_cols].notna().all().all():
            warnings.warn("No missing values found in target columns!")
            
        return target_cols

    def _prepare_auxiliary_cols(
        self,
        auxiliary_cols: Union[str, Sequence[str]],
        target_cols: Sequence[str]
    ) -> List[str]:
        """Prepare and validate auxiliary columns."""
        # Handle 'all' case
        if auxiliary_cols == 'all':
            auxiliary_cols = [col for col in self._df.columns 
                             if col not in target_cols and not is_text_column(self._df[col])]
        
        # Convert to list if single column
        elif isinstance(auxiliary_cols, str):
            auxiliary_cols = [auxiliary_cols]
            
        # Remove any target columns that were accidentally included
        auxiliary_cols = [col for col in auxiliary_cols if col not in target_cols]
        
        # Validate columns exist
        missing = [col for col in auxiliary_cols if col not in self._df.columns]
        if missing:
            raise ValueError(f"Auxiliary columns not found in DataFrame: {missing}")
            
        return auxiliary_cols

    def _prepare_data(
        self,
        target_cols: Sequence[str],
        auxiliary_cols: Sequence[str],
        method: str,
        standardize: bool,
    ) -> Tuple[pd.DataFrame, Union[slice, List[int]]]:
        """
        Prepare data matrix for imputation.
        
        Returns:
        --------
        tuple: (processed_data, target_indices)
        """
        # Separate numerical and categorical auxiliary columns
        num_cols = [col for col in auxiliary_cols 
                   if pd.api.types.is_numeric_dtype(self._df[col])]
        cat_cols = [col for col in auxiliary_cols 
                   if is_categorical_column(self._df[col])]
        
        # For simple method, only use target columns
        if method == 'simple':
            processed = self._df[target_cols].copy()
            target_idx = slice(None)  # All columns are targets
        else:
            # Process numerical features
            num_data = self._df[num_cols].values if num_cols else np.empty((len(self._df), 0))
            
            # Process categorical features with one-hot encoding
            if cat_cols:
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                cat_data = encoder.fit_transform(self._df[cat_cols])
            else:
                cat_data = np.empty((len(self._df), 0))
                
            # Combine all features
            processed = np.hstack([
                self._df[target_cols].values,  # Target columns first
                num_data,
                cat_data
            ])

            # Target columns are always first in the array
            target_idx = slice(0, len(target_cols))
            
            # Standardize if requested
            if standardize:
                processed = StandardScaler().fit_transform(processed)
        
        # Convert to DataFrame for sampling
        processed = pd.DataFrame(processed, index=self._df.index)
            
        return processed, target_idx

    def _create_imputer(
        self,
        method: str,
        strategy: str,
        n_neighbors: int,
        random_state: int,
        imputer_params: Dict[str, Any]
    ):
        """Create and configure the appropriate imputer."""
        if method == 'simple':
            return SimpleImputer(strategy=strategy, **imputer_params)
            
        elif method == 'knn':
            return KNNImputer(n_neighbors=n_neighbors)
            
        elif method == 'iterative':
            estimator = imputer_params.get('estimator', BayesianRidge())
            return IterativeImputer(
                estimator=estimator,
                max_iter=imputer_params.get('max_iter', 10),
                random_state=random_state,
                **{k:v for k,v in imputer_params.items() 
                  if k not in ['estimator', 'max_iter']}
            )
            
        raise ValueError(f"Invalid method: {method}. Choose from ['simple', 'knn', 'iterative']")

    def _merge_results(
        self,
        imputed_values: np.ndarray,
        target_cols: Sequence[str],
        index: pd.Index
    ) -> pd.DataFrame:
        """
        Efficiently merge imputed values back into original DataFrame.
        
        Only updates missing values in target columns while preserving:
        - Original data types
        - Non-target columns
        - Index and column order
        """
        # Create copy of original data (shallow copy for efficiency)
        result_df = self._df.copy()
        
        # Convert imputed values to DataFrame for alignment
        imputed_df = pd.DataFrame(imputed_values, columns=target_cols, index=index)
        
        # Vectorized update of only missing values
        for col in target_cols:
            mask = self._df[col].isna()
            result_df.loc[mask, col] = imputed_df.loc[mask, col]
            
        return result_df

    def _return_result(
        self,
        result_df: pd.DataFrame,
        inplace: bool
    ) -> Union[None, pd.DataFrame]:
        """Handle in-place modification and return results."""
        if inplace:
            self._df = result_df
            return None
        return result_df
    
    def find_optimal_k_for_knn_imputer(
        self, 
        target_cols: Union[str, List[str]],
        auxiliary_cols: Union[str, List[str]] = 'all',        
        max_k: int = 15, 
        n_jobs: int = -1,
        metric: str = 'nan_euclidean'
    ) -> Dict[str, Union[int, float]]: 
        """
        Find optimal number of neighbors for KNNImputer using advanced metrics.
        
        Parameters:
        -----------
        target_cols : str or list
            Numerical columns to impute (must contain missing values)
            
        auxiliary_cols : str or list, default='all'
            Columns to use as features for imputation. Can include:
            
            - Numerical columns (used directly)
            - Categorical columns (one-hot encoded)
            - Datetime columns (feature engineered)
            
            Does not include text columns.

        max_k : int, default=15
            Maximum number of neighbors to test
            
        n_jobs : int, default=-1
            Number of parallel jobs to run
            
        metric : str, default='nan_euclidean'
            Distance metric to use ('nan_euclidean', 'nan_manhattan')
            
        Returns:
        --------
        dict
            Contains:
            
            - 'optimal_k': Best K value
            - 'best_score': Corresponding score
            - 'all_scores': Scores for all tested K values
            - 'elbow_point': Alternative K suggestion from elbow method
        """
        # Validate and prepare input parameters
        target_cols = self._validate_target_cols(target_cols)
        auxiliary_cols = self._prepare_auxiliary_cols(auxiliary_cols, target_cols)
        # Prepare standardized data
        processed, _ = self._prepare_data(
            target_cols=target_cols,
            auxiliary_cols=auxiliary_cols,
            method='knn',
            standardize=True,
        )
        
        # Initialize results storage
        results = {
            'scores': [],
            'optimal_k': 2,
            'best_score': -1,
            'elbow_point': None
        }
        
        # Parallel K evaluation
        with Parallel(n_jobs=n_jobs) as parallel:
            scores = parallel(
                delayed(self._evaluate_k)(k, processed, metric)
                for k in range(2, max_k + 1)
            )
            results['scores'] = scores
        
        # Find best K by silhouette
        results['best_score'] = max(results['scores'])
        results['optimal_k'] = results['scores'].index(results['best_score']) + 2
        
        # Calculate elbow point
        results['elbow_point'] = self._find_elbow_point(results['scores'])
        
        return {
            'optimal_k': results['optimal_k'],
            'best_score': results['best_score'],
            'all_scores': dict(zip(range(2, max_k+1), results['scores'])),
            'elbow_point': results['elbow_point']
        }

    def _evaluate_k(self, k: int, data: Union[pd.DataFrame, np.ndarray], metric: str) -> Union[int, float]:
        """Evaluate single K value with multiple metrics"""
        try:
            # Impute with current K
            imputer = KNNImputer(n_neighbors=k, metric=metric)
            imputed = imputer.fit_transform(data)
            
            # Calculate multiple quality metrics
            sil_score = silhouette_score(imputed, metric=metric)
            variance = np.nanvar(imputed, axis=0).mean()
            
            # Combined score (adjust weights as needed)
            return 0.7 * sil_score + 0.3 * variance
        except:
            return -1

    def _find_elbow_point(self, scores: List[float]) -> int:
        """
        Find elbow point in the scores curve using the maximum distance method.
        
        Parameters:
        -----------
        scores : List[float]
            List of evaluation scores for different K values
            
        Returns:
        --------
        int
            The K value at which the elbow occurs (1-based index)
        """
        if any(np.isnan(s) for s in scores):
            warnings.warn("NaN values found in scores, using default K=2")
            return 2
        if not scores or len(scores) < 2:
            return 2  # Default minimal K value
        
        try:
            # Convert to numpy arrays
            x = np.arange(len(scores))
            y = np.array(scores)
            
            # Get coordinates of the line from first to last point
            first_point = np.array([x[0], y[0]])
            last_point = np.array([x[-1], y[-1]])
            line_vec = last_point - first_point
            
            # Normalize the line vector
            line_vec_norm = line_vec / np.linalg.norm(line_vec)
            
            # Vector from first point to each point on the curve
            vec_from_first = np.column_stack((x - first_point[0], y - first_point[1]))
            
            # Scalar projection of each point onto the line
            scalar_prod = np.sum(vec_from_first * line_vec_norm, axis=1)
            
            # Vector from each point to its projection on the line
            vec_to_line = vec_from_first - np.outer(scalar_prod, line_vec_norm)
            
            # Calculate distances and find maximum
            distances = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
            elbow_index = np.argmax(distances)
            
            # Return K value (add 2 because K starts at 2)
            return elbow_index + 2
            
        except Exception as e:
            warnings.warn(f"Elbow point detection failed: {str(e)}. Using default K=2")
            return 2