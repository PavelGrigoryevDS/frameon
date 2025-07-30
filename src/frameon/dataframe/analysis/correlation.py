import pandas as pd
import numpy as np
import plotly.express as px
from dataclasses import dataclass
from typing import Union, Dict, Optional, Tuple, Any, List, Literal, TYPE_CHECKING
from scipy.stats import pearsonr, spearmanr, kendalltau
from plotly.subplots import make_subplots
from dataclasses import fields, field
import plotly.graph_objects as go
from IPython.display import display
from frameon.utils.plotting import CustomFigure
import warnings

if TYPE_CHECKING: # pragma: no cover
    from frameon.core.base import FrameOn

__all__ = ['CorrelationAnalyzer']

class CorrelationAnalyzer:
    def __init__(self, df: "FrameOn"):
        """Initialize with the main dataframe."""
        self._df = df
    
    def corr_matrix(
        self,
        method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
        text_size: int = 12,
        cell_border_color: str = 'lightgray', 
        column_labels: Optional[Dict[str, str]] = None, 
        height: int = None, 
        width: int = None,
        title: str = "Correlation Matrix",
        labels: dict = None,
        significance_level: float = 0.05,
        decimal_places_coef: int = 2,
        decimal_places_pval: int = 2
    ) -> CustomFigure:
        """Builds an advanced correlation matrix for numeric columns in the dataframe.
        
        Parameters:
        -----------
        method : str, optional
            Correlation method ('pearson', 'spearman', or 'kendall') (default: 'pearson')
        text_size : int, optional
            Font size for matrix text (default: 14)
        cell_border_color : str, optional
            Color for cell borders (default: 'lightgray')
        column_labels : dict, optional
            Dictionary mapping column names to display labels
        height, width : int, optional
            Height and width for plot (auto-calculated if None)
        title : str, optional
            Title for the plot (default: "Correlation Matrix")
        labels : dict, optional
            Labels for rename columns.
        significance_level : float, optional
            Threshold for statistical significance (default: 0.05)
            
        Returns
        -------
        CustomFigure
            Interactive Plotly figure object                      
        """
        # Select only numeric columns
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError("Not enough numeric columns for correlation analysis (need at least 2)")
        
        corr_data = self._df[numeric_cols]
        if labels:
            corr_data = corr_data.rename(columns=labels)
        # 2. Calculate matrices
        cols = corr_data.columns
        corr_matrix = corr_data.corr(method=method).round(2)

        p_values = pd.DataFrame(
            np.ones((len(cols), len(cols))),
            columns=cols,
            index=cols
        )
        
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i < j:
                    mask = corr_data[[col1, col2]].notna().all(axis=1)
                    x = corr_data.loc[mask, col1]
                    y = corr_data.loc[mask, col2]
                    if len(x) < 2: 
                        warnings.warn(
                            f"Cannot compute correlation between '{col1}' and '{col2}'. "
                            f"Only {len(x)} valid pairs available. Returning NaN.",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        p_val = np.nan
                    else:
                        _, p_val = CorrelationAnalyzer._calculate_correlation(method, x, y)
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
                visual_matrix.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.{decimal_places_coef}f}" 
                # Fill upper triangle with correlation values
                color_matrix.iloc[i, j] = corr_matrix.iloc[i, j]  

        # Fill lower triangle with p-values
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                p_val = p_values.iloc[i, j]
                visual_matrix.iloc[i, j] = f"{p_val:.{decimal_places_pval}f}" 
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
                ticktext=['-1 (Neg)', '-0.5', '0', '0.5', '1 (Pos)'],
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
            x0=-0.5, y0=-0.5, 
            x1=len(cols)-0.5, y1=len(cols)-0.5,
            line=dict(color=cell_border_color, width=1),
            fillcolor="rgba(0,0,0,0)",
            layer="below"
        )

        for i in range(len(cols)+1):
            fig.add_shape(
                type="line",
                x0=-0.5, y0=i-0.5,
                x1=len(cols)-0.5, y1=i-0.5,
                line=dict(color=cell_border_color, width=1),
                layer="below"
            )

        for j in range(len(cols)+1):
            fig.add_shape(
                type="line",
                x0=j-0.5, y0=-0.5,
                x1=j-0.5, y1=len(cols)-0.5,
                line=dict(color=cell_border_color, width=1),
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
            title = f"{title} • {method.title()}<br>" \
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
            # font=dict(size=text_size),
            plot_bgcolor='white',
            coloraxis={'colorscale': 'RdBu_r', 'cmin': -1, 'cmax': 1}
        )

        return CustomFigure(fig)
    
    @staticmethod
    def _calculate_correlation(method, x, y):
        """Helper function to compute correlation based on selected method"""
        if len(x) < 2:
            return np.nan, np.nan
            
        if method == 'pearson':
            return pearsonr(x, y)
        elif method == 'spearman':
            return spearmanr(x, y)
        elif method == 'kendall':
            return kendalltau(x, y)
        else:
            raise ValueError(f"Unknown correlation method: {method}")