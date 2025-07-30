from typing import Optional, Union, List, Literal, TYPE_CHECKING, Dict
import pandas as pd
import numpy as np
from frameon.utils.miscellaneous import style_dataframe, analyze_anomalies_all_categories
from IPython.display import display, HTML
import itertools
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from frameon.utils.plotting import BarLineAreaBuilder, CustomFigure
from scipy.stats import pearsonr
if TYPE_CHECKING: # pragma: no cover
    from frameon.core.base import FrameOn

__all__ = ['FrameOnAnomaly']

class FrameOnAnomaly:
    """
    Class containing unified methods for DataFrame anomaly detection and analysis.
    """
    def __init__(self, df: "FrameOn"):
        self._df = df
    
    # ====================== CORE METHODS ======================
    
    def anomalies_report(
        self,
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'] = 'missing',
        exact: bool = True,
        normalize_text: bool = True,
        method: Literal['iqr', 'zscore', 'quantile'] = 'quantile',
        threshold: float = 0.05,
        show_combinations: bool = True,
        show_sample: bool = True,
        show_by_categories: bool = True,
        show_correlation_matrix: bool = True,
        sample_size: int = 5,
        pct_diff_threshold: float = 5,
        include_columns: Optional[Union[str, List[str]]] = None,
        exclude_columns: Optional[Union[str, List[str]]] = None,    
        height: int = None, 
        width: int = None,        
    ) -> Union[None, go.Figure]:
        """
        Generates a comprehensive report for rows containing specified anomalies across ALL columns.
        
        The method identifies ENTIRE ROWS that contain at least one anomaly of the specified type,
        then provides detailed analysis including:
        
        - Statistics by column
        - Combination patterns
        - Category distributions
        - Sample anomalous rows
        - Correlation analysis between anomalies

        Key Features:
        
        - Works at ROW LEVEL - flags rows with ANY occurrence of the specified anomaly type
        
        Parameters:
        -----------
        anomaly_type : str
            Type of anomaly: 'missing', 'duplicate', 'outlier', 'zero', 'negative'
        exact : bool
            For duplicates: exact matching (True) or fuzzy matching (False)
        normalize_text : bool
            For duplicates: normalize text for fuzzy matching
        method : str
            For outliers: detection method ('iqr', 'zscore', 'quantile')
        threshold : float
            For outliers: detection threshold. Interpretation depends on method:
            
            - 'iqr': multiplier for IQR (typical values 1.5-3.0)
            - 'zscore': cutoff value in standard deviations (typical values 2.0-3.0)
            - 'quantile': probability threshold (0 < threshold < 1, typical values 0.05-0.01)

        show_combinations : bool
            Whether to show combinations analysis
        show_sample : bool
            Whether to show sample from DataFrame with anomalies        
        show_by_categories : bool, optional
            Whether to show count of anomalies by categories
        show_correlation_matrix : bool
            Whether to show correlation matrix 
        sample_size : int
            Number of sample rows to display
        pct_diff_threshold : float, default 1.0
            Only for show_by_categories. Minimum % difference to include in results (from -100 to 100)
        include_columns : str or List[str], optional
            Only for show_by_categories. Specific categorical columns to include (None for all)
        exclude_columns : str or List[str], optional
            Only for show_by_categories. Categorical columns to exclude from analysis    
        height, width : int
            Height and width for correlation matrix plot     
             
        Returns:
        --------
        Union[None, go.Figure]    
        """
        # Validate DataFrame is not empty
        if self._df.empty:
            raise ValueError(
                "DataFrame is empty."
            )   
        # Validate threshold based on method
        if anomaly_type == 'outlier':
            if method == 'quantile' and not (0 < threshold < 1):
                raise ValueError("For 'quantile' method, threshold must be between 0 and 1")
            elif method in ('iqr', 'zscore') and threshold <= 0:
                raise ValueError(f"For '{method}' method, threshold must be positive")
        # Get anomaly data
        if anomaly_type == 'missing':
            anomaly_mask = self._df.isna().any(axis=1)
        elif anomaly_type == 'duplicate':
            anomaly_mask = self._get_duplicate_mask(exact, normalize_text)
        elif anomaly_type == 'outlier':
            anomaly_mask = self._get_anomaly_mask(None, 'outlier', method, threshold)
        elif anomaly_type == 'zero':
            anomaly_mask = self._get_anomaly_mask(None, 'zero')
        elif anomaly_type == 'negative':
            anomaly_mask = self._get_anomaly_mask(None, 'negative')
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

        anomaly_df = self._df[anomaly_mask]
        
        if anomaly_df.empty:
            print(f"No {anomaly_type} found in the DataFrame")
            return
        
        # 1. Basic stats (only for duplicates)
        if anomaly_type == 'duplicate':
            display(style_dataframe(
                self._duplicates_stats(exact, normalize_text),
                caption=f"{anomaly_type.capitalize()} Statistics",
                hide_columns=False,
                hide_index=False
            ))
        
        # 2. By-column analysis (for all anomaly types)
        by_col_df = self._get_result_for_detect_anomalies(
            anomaly_type=anomaly_type,
            method=method,
            threshold=threshold,
        )
        if by_col_df is not None:
            display(style_dataframe(
                by_col_df,
                caption=f"{anomaly_type.capitalize()}s by Column",
                hide_columns=False,
                hide_index=False,
                formatters={'Count': '{:.0f}', 'Percent': '{:.1%}'}
            ))
        
        # 3. Combinations analysis (optional)
        if show_combinations:
            self.anomalies_combinations(
                n=2,
                anomaly_type=anomaly_type,
                method=method,
                threshold=threshold
            )
        
        # 4. Category breakdown (optional)
        if show_by_categories:
            results = analyze_anomalies_all_categories(
                df=self._df,
                anomaly_df=anomaly_df,
                pct_diff_threshold=pct_diff_threshold,
                include_columns=include_columns,
                exclude_columns=exclude_columns,
            )
            if results is not None:
                caption = f"{anomaly_type.capitalize()}s distribution across categories"
                if anomaly_type == 'outlier':
                    caption += f" (method: {method}, threshold: {threshold})"
                
                display(style_dataframe(
                    results,
                    caption=caption,
                    hide_columns=False,
                    formatters={'% Diff': '{:.1f}%'}
                ))
        # 5. Sample rows with original formatting
        if show_sample:
            if not sample_size:
                anomaly_df_len = anomaly_df.shape[0]
                sample_n = 5 if anomaly_df_len >= 5 else anomaly_df_len
            else: 
                sample_n = sample_size
            sample_n = min(sample_n, len(anomaly_df))
            sample_df = anomaly_df.sample(sample_n)
            for col in sample_df.columns:
                if pd.api.types.is_numeric_dtype(sample_df[col]):
                    sample_df[col] = sample_df[col].astype(str)
            
            display(style_dataframe(
                sample_df,
                caption=f"Sample {anomaly_type.capitalize()}s Rows",
                hide_columns=False,
                hide_index=False,
            ))
        if show_correlation_matrix:
            return self.anomalies_corr_matrix(anomaly_type, height=height, width=width)

    def anomalies_corr_matrix(
        self,
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'],
        text_size: int = 14,
        cell_border_color: str = 'lightgray', 
        column_labels: Optional[Dict[str, str]] = None, 
        height: int = None, 
        width: int = None,
    ) -> go.Figure:
        """Advanced correlation matrix showing only columns with anomalies
        
        Parameters:
        -----------
        anomaly_type : str
            Type of anomalies to analyze ('missing' or other types)
        text_size : int, optional
            Font size for matrix text (default: 14)
        cell_border_color : str, optional
            Color for cell borders (default: 'lightgray')
        column_labels : dict, optional
            Dictionary mapping column names to display labels
        height, width : int
            Height and width for plot
            
        Returns:
        --------
        go.Figure
        
        """
        # 1. Prepare anomaly data
        if anomaly_type == 'missing':
            corr_data = self._df.isna().astype(float)
        else:
            corr_data = pd.DataFrame()
            for col in self._df.columns:
                if pd.api.types.is_numeric_dtype(self._df[col]):
                    mask = self._get_anomaly_mask([col], anomaly_type, 'iqr', 1.5)
                    if mask.any():  # Only include columns with anomalies
                        corr_data[col] = mask.astype(float)
        
        # Filter only columns with anomalies
        corr_data = corr_data.loc[:, corr_data.any()]
        
        if corr_data.empty or len(corr_data.columns) < 2:
            print("Not enough columns with anomalies for correlation analysis")
            return None

        # 2. Calculate matrices
        cols = corr_data.columns
        corr_matrix = corr_data.corr().round(2)
        p_values = pd.DataFrame(
            np.ones((len(cols), len(cols))),
            columns=cols,
            index=cols
        )
        
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i < j:
                    _, p_val = pearsonr(corr_data[col1], corr_data[col2])
                    p_values.loc[col1, col2] = p_val
                    p_values.loc[col2, col1] = p_val
        
        # 3. Create combined matrix for visualization
        visual_matrix = pd.DataFrame(
            index=cols,
            columns=cols,
            dtype=object
        )
        
        # Apply custom column labels if provided
        display_cols = cols if column_labels is None else [column_labels.get(col, col) for col in cols]
        
        # Fill diagonal with column names
        np.fill_diagonal(visual_matrix.values, '')
        
        # 4. Create color matrix
        color_matrix = pd.DataFrame(
            np.zeros((len(cols), len(cols))),
            columns=cols,
            index=cols
        )
        # Fill upper triangle with correlation coefficients
        for i in range(1, len(cols)):
            for j in range(i): 
                visual_matrix.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}" 
                # Fill upper triangle with correlation values
                color_matrix.iloc[i, j] = corr_matrix.iloc[i, j]  
                
        # Fill lower triangle with p-values
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                p_val = p_values.iloc[i, j]
                visual_matrix.iloc[i, j] = f"{p_val:.3f}" 
                # Fill lower triangle with special value for significant p-values
                if p_val < 0.05:
                    color_matrix.iloc[i, j] = 0.1  # Special value for significant p-values
                else:
                    color_matrix.iloc[i, j] = np.nan
        
        # Set diagonal to NaN (no color)
        np.fill_diagonal(color_matrix.values, np.nan)
        
        # 5. Create visualization
        fig = go.Figure()
        
        # Main heatmap for correlations
        heatmap = go.Heatmap(
            z=color_matrix.values,
            x=cols,
            y=cols,
            zmin=-1,
            zmax=1,
            colorscale='RdBu_r',
            showscale=True,
            colorbar=dict(
                title='Correlation',
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=['-1 (Neg)', '-0.5', '0', '0.5', '1 (Pos)']
            ),
            text=visual_matrix.values,
            texttemplate="%{text}",
            textfont=dict(size=text_size),
            hoverinfo="text",
            hovertext=[
                [f"<b>{display_cols[i]}</b> vs <b>{display_cols[j]}</b><br>"
                f"Correlation: {corr_matrix.iloc[i,j]:.2f}<br>"
                f"p-value: {p_values.iloc[i,j]:.4f}"
                for j in range(len(cols))] 
                for i in range(len(cols))],
            xgap=1,  # Add gaps between cells
            ygap=1,  # Add gaps between cells
        )
        
        fig.add_trace(heatmap)
        upper_triangle_mask = np.triu(np.ones_like(color_matrix, dtype=bool), k=1)
        # Add separate trace for significant p-values (0.1 values)
        if (color_matrix == 0.1).any().any():
            fig.add_trace(go.Heatmap(
                z=np.where(
                    (color_matrix == 0.1) & upper_triangle_mask, # Only the lower triangle
                    0.1,  # Meaning for backlighting
                    np.nan  # All other cells
                ),
                x=cols,
                y=cols,
                zmin=0,
                zmax=1,
                colorscale=[[0, 'rgba(0,128,0,0.3)'], [1, 'rgba(0,128,0,0.3)']],
                showscale=False,
                hoverinfo='skip',
                xgap=1,
                ygap=1,
            ))
        
        # Add grid lines for all cells (including empty ones)
        fig.add_shape(
            type="rect",
            x0=-0.5, y0=-0.5, x1=len(cols)-0.5, y1=len(cols)-0.5,
            line=dict(color=cell_border_color, width=1),
            fillcolor="rgba(0,0,0,0)",
            layer="below"
        )
        
        for i in range(len(cols)):
            for j in range(len(cols)):
                fig.add_shape(
                    type="rect",
                    x0=i-0.5, y0=j-0.5, x1=i+0.5, y1=j+0.5,
                    line=dict(color=cell_border_color, width=1),
                    fillcolor="rgba(0,0,0,0)",
                    layer="below"
                )
        n_cols = len(cols)
        min_height, max_height = 400, 1200
        min_width, max_width = 500, 1200
        # Smooth increase (logarithmic/quadratic)
        if not height:
            height = min(
                max(
                    int(400 + (n_cols - 2) ** 1.1 * 30),  # Formula: basic size + (number of columns)^1.1 *coefficient
                    min_height
                ),
                max_height
            )

        # Smooth increase in width (linear with saturation)
        if not width:
            width = min(
                max(
                    int(500 + (n_cols - 2) * 50),  # For each column +50px
                    min_width
                ),
                max_width
            )
        # Update layout
        fig.update_layout(
            title = f"{anomaly_type.capitalize()} Anomaly Correlation Matrix<br>" \
                "<sup>Lower triangle: coefficients | Upper: p-values (green if p < 0.05)</sup>",
            width=width,
            height=height,
            xaxis=dict(
                showgrid=False,
                linecolor='black',  # Add border at the top and bottom
                linewidth=1,
                mirror=True,  # This mirrors the axis line to the top
                showline=True,
                # ticks='',  # Remove ticks
                # tickvals=[]  # Remove tick labels
            ),
            yaxis=dict(
                autorange="reversed",
                showgrid=False,
                linecolor='black',  # Add border at the left and right
                linewidth=1,
                mirror=True,  # This mirrors the axis line to the right
                showline=True,
                # ticks='',  # Remove ticks
                # tickvals=[]  # Remove tick labels
            ),
            font=dict(size=text_size),
            plot_bgcolor='white',
            coloraxis={'colorscale': 'RdBu_r', 'cmin': -1, 'cmax': 1}
        )
        
        return CustomFigure(fig)

    def detect_anomalies(
        self,
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'] = 'missing',
        columns: Optional[Union[str, List[str]]] = None,
        method: Literal['iqr', 'zscore', 'quantile'] = 'quantile',
        threshold: float = 0.05,
        display_count: bool = True, 
        return_mode: Literal[False, 'combined', 'by_column'] = False
    ) -> Union[None, pd.DataFrame]:
        """
        Unified method to detect different types of anomalies by column.
        
        Parameters:
        -----------
        anomaly_type : str, default 'missing'
            Type of anomaly to detect
        columns : Optional[Union[str, List[str]]], default None
            Columns to detect (all if None)
        method : str, default 'iqr'
            For outliers: detection method
        threshold : float
            For outliers: detection threshold. Interpretation depends on method:
            
            - 'iqr': multiplier for IQR (typical values 1.5-3.0)
            - 'zscore': cutoff value in standard deviations (typical values 2.0-3.0)
            - 'quantile': probability threshold (0 < threshold < 1, typical values 0.05-0.01)
            
        display_count : bool, default True
            Whether to display count of anomalies
            
        return_mode : bool or str, default False
            Return mode specification:
            
            - False: don't return results (just display)
            - 'combined': return single DataFrame with all anomalies
            - 'by_column': return dict of {column: anomaly_df}
            
        Returns:
        --------
        Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
        
            Depending on return_mode:
            
            - False: None
            - 'combined': DataFrame with all anomalies
            - 'by_column': dict of {column: anomaly_df}
        """
        results = self._get_result_for_detect_anomalies(
            anomaly_type=anomaly_type,
            columns=columns,
            method=method,
            threshold=threshold,
        )
        if results is None:
            return
        if display_count:
            formatted = results.copy()
            formatted['Count'] = formatted['Count'].astype(int).astype(str)
            formatted['Percent'] = formatted['Percent'].apply(lambda x: f"{x:.2%}")
            
            caption = f"{anomaly_type.capitalize()}s by Column"
            if anomaly_type == 'outlier':
                caption += f" (method: {method}, threshold: {threshold})"
            display(style_dataframe(
                formatted,
                caption=caption,
                hide_index=False,
                hide_columns=False
            ))
            
        if return_mode:
            if isinstance(columns, str):
                columns = [columns]
            if columns is None:
                columns = self._df.columns
            
            if return_mode == 'by_column':
                # Return dictionary of {column: anomaly_df}
                anomalies_dict = {}
                for col in columns:
                    mask = self._get_anomaly_mask([col], anomaly_type, method, threshold)
                    if mask.any():
                        anomalies_dict[col] = self._df[mask]
                
                if not anomalies_dict:
                    print(f"No {anomaly_type} found in specified columns")
                    return None
                return anomalies_dict
            elif return_mode == 'combined':
                # Return combined DataFrame
                mask = self._get_anomaly_mask(columns, anomaly_type, method, threshold)
                if not mask.any():
                    print(f"No {anomaly_type} found in specified columns")
                    return None
                return self._df[mask]
            else:
                raise ValueError('Invalid return_mode. Can be one of false, by_column, combined')
    
    def anomalies_combinations(
        self,
        n: int = 2,
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'] = 'missing',
        method: Literal['iqr', 'zscore', 'quantile'] = 'quantile',
        threshold: float = 0.05
    ) -> None:
        """
        Unified method to analyze co-occurring anomalies in column combinations.
        
        Parameters:
        -----------
        n : int, default 2
            Number of columns in combinations
        anomaly_type : str, default 'missing'
            Type of anomaly to analyze
        method : str, default 'quantile'
            For outliers: detection method
        threshold : float, default 0.05
            For outliers: detection threshold
            
        Returns:
        --------
        None
        """
        if n < 2:
            raise ValueError("Combination size must be at least 2")
            
        if anomaly_type in ['zero', 'negative', 'outlier']:
            columns = self._df.select_dtypes(include=np.number).columns.tolist()
        else:
            columns = self._df.columns.tolist()
            
        if len(columns) == 1:
            return
        if n > len(columns):
            raise ValueError(f"Combination size ({n}) cannot exceed number of columns ({len(columns)})")
        
        # Get columns with at least some anomalies
        if anomaly_type == 'missing':
            columns = [col for col in columns if self._df[col].isna().any()]
        elif anomaly_type == 'duplicate':
            columns = [col for col in columns if self._df[col].duplicated().any()]
        elif anomaly_type == 'outlier':
            columns = [col for col in columns if self._get_anomaly_mask([col], 'outlier', method, threshold).any()]
        elif anomaly_type == 'zero':
            columns = [col for col in columns if (self._df[col] == 0).any()]
        elif anomaly_type == 'negative':
            columns = [col for col in columns if (self._df[col] < 0).any()]
        
        if len(columns) < 2:
            print(f"Not enough columns with {anomaly_type} to analyze combinations")
            return
        
        # Special case for pairwise combinations
        if n == 2:
            combos = itertools.combinations(columns, 2)
            result_df = pd.DataFrame([], index=columns, columns=columns)
            
            for col1, col2 in combos:
                mask1 = self._get_anomaly_mask([col1], anomaly_type, method, threshold)
                mask2 = self._get_anomaly_mask([col2], anomaly_type, method, threshold)
                co_anomalies = (mask1 & mask2).sum()
                col1_anomalies = mask1.sum()
                col2_anomalies = mask2.sum()
                if co_anomalies == 0:
                    result_df.loc[col2, col1] = f'0'
                else:
                    col2_percent = co_anomalies / col2_anomalies
                    formated_value = f'< {(col2_percent):.1%}' if col2_percent >= 0.01 else '< under 1%'
                    col1_percent = co_anomalies / col1_anomalies
                    formated_value += f' / ^ {(col1_percent):.1%}' if col1_percent >= 0.01 else ' / ^ under 1%'
                    result_df.loc[col2, col1] = formated_value
                
            
            caption = f"Co-occurring {anomaly_type.capitalize()}s (Pairwise)"
            if anomaly_type == 'outlier':
                caption += f" (method: {method}, threshold: {threshold})"
            
            display(style_dataframe(
                result_df.fillna(""),
                caption=caption,
                hide_index=False,
                hide_columns=False
            ))
            return
            
        # Handle combinations for n >= 3
        column_combinations = itertools.combinations(columns, n)
        results = []
        
        for cols in column_combinations:
            mask = pd.Series(True, index=self._df.index)
            for col in cols:
                mask &= self._get_anomaly_mask([col], anomaly_type, method, threshold)
            co_anomalies = mask.sum()
            
            if co_anomalies > 0:
                results.append({
                    'columns': " | ".join(cols),
                    'count': co_anomalies
                })
        
        if not results:
            print(f"No co-occurring {anomaly_type} found in any {n}-column combinations")
            return
            
        results_df = pd.DataFrame(results).sort_values('count', ascending=False)
        
        caption = f"Co-occurring {anomaly_type.capitalize()} ({n}-way combinations)"
        if anomaly_type == 'outlier':
            caption += f" (method: {method}, threshold: {threshold})"
        
        display(style_dataframe(
            results_df,
            caption=caption,
            hide_columns=False,
            formatters={'count': "{:,}".format},
        ))
    
    def anomalies_by_categories(
        self,
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'] = 'missing',
        custom_mask: Optional[Union[pd.Series, np.ndarray, list]] = None,
        exact: bool = True,
        normalize_text: bool = True,
        method: Literal['iqr', 'zscore', 'quantile'] = 'quantile',
        threshold: float = 0.05,
        pct_diff_threshold: float = 0,
        include_columns: Optional[Union[str, List[str]]] = None,
        exclude_columns: Optional[Union[str, List[str]]] = None,    
    ) -> None:
        """
        Unified method to analyze rows containing specified anomalies across ALL columns by categories.

        Key Features:
        
        - Works at ROW LEVEL - flags rows with ANY occurrence of the specified anomaly type  
        
        Parameters:
        -----------
        anomaly_type : str, default 'missing'
            Type of anomaly to analyze
        custom_mask : Union[pd.Series, np.ndarray, list], optional
            Boolean custom mask to detect anomalies.
            If provided, overrides `anomaly_type`.
        exact : bool
            For duplicates: exact matching (True) or fuzzy matching (False)
        normalize_text : bool
            For duplicates: normalize text for fuzzy matching
        method : str, default 'quantile'
            For outliers: detection method
        threshold : float, default 0.05
            For outliers: detection threshold
        pct_diff_threshold : float, default 1.0
            Minimum % difference to include in results (from -100 to 100)
        include_columns : str or List[str], optional
            Specific categorical columns to include (None for all)
        exclude_columns : str or List[str], optional
            Categorical columns to exclude from analysis   
            
        Returns:
        --------
        None            
        """
        if self._df.empty:
            raise ValueError(
                "DataFrame is empty."
            )   
        # Validate threshold based on method
        if anomaly_type == 'outlier':
            if method == 'quantile' and not (0 < threshold < 1):
                raise ValueError("For 'quantile' method, threshold must be between 0 and 1")
            elif method in ('iqr', 'zscore') and threshold <= 0:
                raise ValueError(f"For '{method}' method, threshold must be positive")
        if custom_mask is not None:
            anomaly_mask = pd.Series(custom_mask, index=self._df.index, dtype=bool)
            anomaly_type = "custom"
        else:
            # Get anomaly data
            if anomaly_type == 'missing':
                anomaly_mask = self._df.isna().any(axis=1)
            elif anomaly_type == 'duplicate':
                anomaly_mask = self._get_duplicate_mask(exact, normalize_text)
            elif anomaly_type == 'outlier':
                anomaly_mask = self._get_anomaly_mask(None, 'outlier', method, threshold)
            elif anomaly_type == 'zero':
                anomaly_mask = self._get_anomaly_mask(None, 'zero')
            elif anomaly_type == 'negative':
                anomaly_mask = self._get_anomaly_mask(None, 'negative')
            else:
                raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        anomaly_df = self._df[anomaly_mask]
        
        if anomaly_df.empty:
            print(f"No {anomaly_type} found in the DataFrame")
            return
        
        results = analyze_anomalies_all_categories(
            df=self._df,
            anomaly_df=anomaly_df,
            pct_diff_threshold=pct_diff_threshold,
            include_columns=include_columns,
            exclude_columns=exclude_columns,
        )
        caption = f"{anomaly_type.capitalize()}s distribution across categories"
        if anomaly_type == 'outlier':
            caption += f" (method: {method}, threshold: {threshold})"
        
        display(style_dataframe(
            results,
            caption=caption,
            hide_columns=False,
            formatters={'% Diff': '{:.1f}%'}
        ))    
        
    def detect_simultaneous_anomalies(
        self,
        columns: Union[str, List[str]],
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'] = 'missing',
        method: Literal['iqr', 'zscore', 'quantile'] = 'quantile',
        threshold: float = 0.05,
        return_results: bool = False,
        display_report: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Detect and analyze rows with simultaneous anomalies of specified type in multiple columns.
        
        Parameters:
        -----------
        columns : Union[str, List[str]]
            Column or list of columns to check for simultaneous anomalies
        anomaly_type : str, default 'missing'
            Type of anomaly to detect: 'missing', 'duplicate', 'outlier', 'zero', 'negative'
        method : str, default 'quantile'
            For outliers: detection method ('iqr', 'zscore', 'quantile')
        threshold : float, default 0.05
            For outliers: detection threshold
        return_results : bool, default False
            Whether to return the DataFrame with anomalous rows
        display_report : bool, default True
            Whether to display a detailed report about the anomalies
            
        Returns:
        --------
        Optional[pd.DataFrame]
            If return_results is True, returns DataFrame with rows containing simultaneous anomalies.
            Otherwise returns None.
        """
        if isinstance(columns, str):
            columns = [columns]
        
        if len(columns) < 2:
            raise ValueError("At least two columns must be specified for simultaneous anomaly detection")
        
        # Get masks for each column
        masks = []
        for col in columns:
            masks.append(self._get_anomaly_mask([col], anomaly_type, method, threshold))
        
        # Combine masks with AND logic (simultaneous anomalies)
        combined_mask = pd.Series(True, index=self._df.index)
        for mask in masks:
            combined_mask &= mask
        
        anomaly_df = self._df[combined_mask]
        total_rows = self._df.shape[0]
        anomaly_count = anomaly_df.shape[0]
        
        if display_report:
            if anomaly_count == 0:
                print(f"No simultaneous {anomaly_type} anomalies found in all specified columns: {columns}")
                return None
            
            # Calculate individual column stats
            col_stats = []
            for col, mask in zip(columns, masks):
                col_count = mask.sum()
                col_stats.append({
                    'Column': col,
                    'Individual Anomalies': col_count,
                    '% of Total': f"{col_count/total_rows:.2%}",
                    '% in Simultaneous': f"{anomaly_count/col_count:.2%}" if col_count > 0 else "N/A"
                })
            
            stats_df = pd.DataFrame(col_stats)
            
            # Display report
            display(style_dataframe(
                stats_df,
                caption=f"Simultaneous {anomaly_type.capitalize()} Anomalies Analysis",
                hide_index=True,
                hide_columns=False,
            ))
            
            if return_results:
                return anomaly_df
        else:
            if return_results:
                return anomaly_df if not anomaly_df.empty else None

    def anomalies_over_time(
        self,
        time_column: str,
        freq: str = 'D',
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'] = 'missing',
        custom_mask: Optional[Union[pd.Series, np.ndarray, list]] = None,
        method: Literal['iqr', 'zscore', 'quantile'] = 'quantile',
        threshold: float = 0.05,
        title: Optional[str] = None,
        exact: bool = True,
        normalize_text: bool = True,
    ) -> Union[None, go.Figure]:
        """
        Plot anomalies across ALL columns over time using resampling.
        
        Key Features:
        
        - Works at ROW LEVEL - flags rows with ANY occurrence of the specified anomaly type  
        
        Parameters:
        -----------
        time_column : str
            Name of the datetime column in the parent DataFrame
        freq : str, default 'D'
            Resampling frequency (e.g., 'D', 'W', 'M', 'Y')
        anomaly_type : str, default 'missing'
            Type of anomaly to count ('missing', 'duplicate', 'outlier', 'zero', 'negative')
        custom_mask : Union[pd.Series, np.ndarray, list], optional
            Boolean custom mask to detect anomalies.
            If provided, overrides `anomaly_type`.
        method : str, default 'quantile'
            For outliers: detection method
        threshold : float
            For outliers: detection threshold. Interpretation depends on method:
            
            - 'iqr': multiplier for IQR (typical values 1.5-3.0)
            - 'zscore': cutoff value in standard deviations (typical values 2.0-3.0)
            - 'quantile': probability threshold (0 < threshold < 1, typical values 0.05-0.01)
            
        exact : bool
            For duplicates: exact matching (True) or fuzzy matching (False)
        normalize_text : bool
            For duplicates: normalize text for fuzzy matching
        title : Optional[str]
            Custom plot title
            
        Returns:
        --------
            Union[None, go.Figure]            
        """
        # Get parent DataFrame
        if anomaly_type == 'outlier':
            if method == 'quantile' and not (0 < threshold < 1):
                raise ValueError("For 'quantile' method, threshold must be between 0 and 1")
            elif method in ('iqr', 'zscore') and threshold <= 0:
                raise ValueError(f"For '{method}' method, threshold must be positive")
        if time_column not in self._df.columns:
            raise ValueError(f"Time column '{time_column}' not found in DataFrame")
            
        if not pd.api.types.is_datetime64_any_dtype(self._df[time_column]):
            raise ValueError(f"Column '{time_column}' must be datetime type")
        
        # Get anomaly mask
        if custom_mask is not None:
            anomaly_mask = pd.Series(custom_mask, index=self._df.index, dtype=bool)
            anomaly_type = "custom"
        else:
            # Get anomaly data
            if anomaly_type == 'missing':
                anomaly_mask = self._df.isna().any(axis=1)
            elif anomaly_type == 'duplicate':
                anomaly_mask = self._get_duplicate_mask(exact, normalize_text)
            elif anomaly_type == 'outlier':
                anomaly_mask = self._get_anomaly_mask(None, 'outlier', method, threshold)
            elif anomaly_type == 'zero':
                anomaly_mask = self._get_anomaly_mask(None, 'zero')
            elif anomaly_type == 'negative':
                anomaly_mask = self._get_anomaly_mask(None, 'negative')
            else:
                raise ValueError(f"Unknown anomaly type: {anomaly_type}")
            
        if not anomaly_mask.any():
            print(f"No {anomaly_type} found in specified columns")
            return None
        # Create temporary DataFrame with time and anomalies
        temp_df = pd.DataFrame({
            'date': self._df[time_column],
            'anomaly': anomaly_mask.astype(int),
        })

        labels = pd.Series(dict(
            date = 'Date'
            , anomaly = f'{anomaly_type.capitalize()} Count'
        ))
        builder = BarLineAreaBuilder('line')
        if anomaly_type == 'custom':
            default_title = f"Custom Anomalies Over Time ({freq})"
        else:
            default_title = f"{anomaly_type.capitalize()} Anomalies Over Time ({freq})"
        if anomaly_type == 'outlier':
            default_title += f" (method: {method}, threshold: {threshold})"
        params = dict(
            data_frame = temp_df
            , x=labels.index[0]
            , y=labels.index[1]
            , agg_func='sum'
            , freq=freq
            , labels=labels
            , title=title or default_title
        )
        return builder.build(**params)
    
    # ====================== HELPER METHODS ======================
    
    def _get_anomaly_mask(
        self,
        columns: Optional[List[str]],
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'],
        method: Literal['iqr', 'zscore', 'quantile'] = 'quantile',
        threshold: float = 0.05
    ) -> pd.Series:
        """Helper method to get boolean mask for specified anomalies"""
        if isinstance(columns, str):
            columns = [columns]
        if columns is None:
            columns = self._df.columns
        if anomaly_type == 'missing':
            return self._df[columns].isna().any(axis=1)
        elif anomaly_type == 'duplicate':
            return self._df.duplicated(subset=columns, keep=False)
        elif anomaly_type == 'outlier':
            mask = pd.Series(False, index=self._df.index)
            for col in columns:
                col_data = self._df[col]
                if not pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
                    continue
                if method == 'iqr':
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    mask |= (col_data < lower) | (col_data > upper)
                elif method == 'zscore':
                    zscore = np.abs((col_data - col_data.mean()) / col_data.std())
                    mask |= zscore > threshold
                elif method == 'quantile':
                    lower = col_data.quantile(threshold)
                    upper = col_data.quantile(1 - threshold)
                    mask |= (col_data < lower) | (col_data > upper)
            return mask
        elif anomaly_type == 'zero':
            return (self._df[columns] == 0).any(axis=1)
        elif anomaly_type == 'negative':
            num_columns = self._df[columns].select_dtypes(include=np.number).columns.tolist()
            return (self._df[num_columns] < 0).any(axis=1)
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
    
    def _get_result_for_detect_anomalies(
        self,
        anomaly_type: Literal['missing', 'duplicate', 'outlier', 'zero', 'negative'] = 'missing',
        columns: Optional[Union[str, List[str]]] = None,
        method: Literal['iqr', 'zscore', 'quantile'] = 'quantile',
        threshold: float = 0.05,
    ) -> Optional[pd.DataFrame]:
        """
        Method for get result to detect different types of anomalies by column.
        """
        if columns is None:
            columns = self._df.columns
        elif isinstance(columns, str):
            columns = [columns]
        if anomaly_type in ['zero', 'negative', 'outlier']:
            columns = [col for col in columns if pd.api.types.is_numeric_dtype(self._df[col])]
            if not columns:
                print(f"No numeric columns to analyze {anomaly_type}")
                return None
        
        # Get counts
        if anomaly_type == 'missing':
            counts = self._df[columns].isna().sum()
        elif anomaly_type == 'duplicate':
            counts = pd.Series({col: self._df[col].duplicated().sum() for col in columns})
        elif anomaly_type == 'outlier':
            counts = pd.Series(index=columns, dtype=int)
            for col in columns:
                mask = self._get_anomaly_mask([col], 'outlier', method, threshold)
                counts[col] = mask.sum()
        elif anomaly_type == 'zero':
            counts = (self._df[columns] == 0).sum()
        elif anomaly_type == 'negative':
            num_columns = self._df[columns].select_dtypes(include=np.number).columns.tolist()
            counts = (self._df[num_columns] < 0).sum()
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        counts = counts[counts > 0]
        if counts.empty:
            print(f"No {anomaly_type} found in specified columns")
            return None
        
        # Prepare results
        results = pd.DataFrame({
            'Count': counts,
            'Percent': counts / len(self._df)
        })
        return results
    
    def _get_duplicate_mask(self, exact: bool, normalize_text: bool) -> pd.Series:
        """Generate duplicate mask based on parameters"""
        if exact:
            return self._df.duplicated(keep='first') 
        
        # Fuzzy duplicate detection
        df_to_check = self._df
        if normalize_text:
            df_to_check = self._df.apply(
                lambda col: (
                    col.str.lower()
                    .str.strip()
                    .str.replace(r'\s+', ' ', regex=True)
                    if pd.api.types.is_string_dtype(col)
                    else col
                )
            )
        return df_to_check.duplicated(keep='first')

    def _duplicates_stats(self, exact: bool, normalize_text: bool) -> pd.DataFrame:
        """
        Return duplicate statistics DataFrame.
        """
        data = []
        keep_options = ['first', False]
        exact_options = [True, False] if exact else [False]
        
        for keep in keep_options:
            row = {'keep': keep}
            for exact in exact_options:
                count = self._get_duplicate_mask_with_keep(keep, exact, normalize_text).sum()
                col_name = 'exact' if exact else 'fuzzy'
                row[col_name] = count
            data.append(row)
        
        return pd.DataFrame(data)
        
    def _get_duplicate_mask_with_keep(
        self,
        keep: Literal['first', 'last', False],
        exact: bool,
        normalize_text: bool
    ) -> pd.Series:
        if exact:
            return self._df.duplicated(keep=keep)
        
        # Fuzzy duplicate detection
        df_to_check = self._df
        if normalize_text:
            df_to_check = self._df.apply(
                lambda col: (
                    col.str.lower()
                    .str.strip()
                    .str.replace(r'\s+', ' ', regex=True)
                    if pd.api.types.is_string_dtype(col)
                    else col
                )
            )
        return df_to_check.duplicated(keep=keep)

