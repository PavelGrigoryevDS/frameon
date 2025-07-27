import pandas as pd
import numpy as np
import plotly.express as px
from dataclasses import dataclass
from typing import Union, Dict, Optional, Any, List, Literal, Tuple
from plotly.subplots import make_subplots
from dataclasses import fields, field
import plotly.graph_objects as go
from typing import get_type_hints
import plotly.io as pio

@dataclass
class RFMConfig:
    user_id_col: str
    date_col: str
    revenue_col: str
    order_id_col: Optional[str] = None
    now_date: Optional[pd.Timestamp] = None
    score_bins: Literal[3, 5] = 3
    height: Optional[int] = None
    width: Optional[int] = None
    show_colorbar: bool = True
    color_continuous_scale: str = 'Greens'
    text_auto: Union[bool, str] = True
    lower_quantile: Union[int, float] = 0
    upper_quantile: Union[int, float] = 1
    distr_plot_type: Literal['box', 'violin'] = 'box'
    return_rfm: Optional[bool] = False
    pair_for_slice: Optional[str] = 'fm'
    plots: List[str] = field(
        default_factory=lambda: [
            'hist',
            'heat',
            'heat_pairs',
            'heat_sliced',
            'seg_bar',
            'seg_tree',
            'scatter_sliced',
            'distr_grid',
            'bar_sliced'            
        ]
    )

class RFMAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """Initialize with the main dataframe."""
        self.df = df
        self.config_rfm = RFMConfig(user_id_col='', date_col='', revenue_col='')
        self.df_rfm = None

    def rfm(
        self,
        user_id_col: str,
        date_col: str,
        revenue_col: str,
        order_id_col: Optional[str] = None,
        now_date: Optional[pd.Timestamp] = None,
        score_bins: Literal[3, 5] = 3,
        show_colorbar: bool = True,
        plots: Optional[List[str]] = None,
        color_continuous_scale: str = 'Greens',
        text_auto: Union[bool, str] = '.3s',
        lower_quantile: Union[int, float] = 0,
        upper_quantile: Union[int, float] = 1,
        distr_plot_type: Literal['box', 'violin'] = 'box',
        pair_for_slice: Optional[str] = 'fm',
        return_rfm: Optional[bool] = False
        ) -> Dict[str, go.Figure]:
        """
        Perform comprehensive RFM (Recency, Frequency, Monetary) analysis with visualization.

        The analysis consists of:
        
        1. Calculating RFM scores for each customer
        2. Segmenting customers based on RFM scores
        3. Generating interactive visualizations of the results

        Parameters
        ----------
        user_id_col : str
            Column name containing unique customer identifiers
        date_col : str
            Column name with transaction dates (will be converted to datetime)
        revenue_col : str
            Column name with transaction amounts/values
        order_id_col : str, optional
            Column name with order identifiers (if not provided, will use date_col for frequency calculation)
        now_date : pd.Timestamp, optional
            Reference date for recency calculation (defaults to max date in date_col)
        score_bins : {3, 5}, default=3
            Number of scoring bins (3 or 5) for RFM segmentation:
            
            - 3 bins: Low/Medium/High scores
            - 5 bins: More granular scoring (1-5)
            
        plots: List of visualizations to generate. Available options:
        
            - None: all visualizations
            - 'hist': Distribution histograms with boxplots
            - 'heat': Recency vs Frequency heatmap
            - 'heat_pairs': Pairwise score combinations
            - 'heat_sliced': Heatmaps sliced by third score
            - 'scatter_sliced': Scatter plots sliced by score
            - 'distr_grid': 3x2 grid of distributions
            - 'seg_bar': Customer segments bar chart
            - 'seg_tree': Customer segments treemap

            Default: all visualizations
        
        return_rfm : bool, optional
            If True, will return DataFrame with rfm result (key in result: 'df_rfm'):
            Default: False
        
        show_colorbar : bool, default=True
            Whether to show color scale bars in heatmaps
        color_continuous_scale : str, default='Greens'
            Plotly color scale name for heatmap visualization
        text_auto : bool or str, default=True
            Controls display of values in heatmap cells (True/False or formatting string)
        lower_quantile : float, default=0
            Lower quantile for trimming extreme values (0-1)
        upper_quantile : float, default=1
            Upper quantile for trimming extreme values (0-1)
        distr_plot_type: Literal['box', 'violin'] = 'violin'
            Type of plot for distr_grid
        pair_for_slice : str, optional
            Which RFM pair to plot in heat_sliced and bar_sliced visualizations.
            Possible values: 'rf', 'fm', 'rm' (for Recency-Frequency, Frequency-Monetary,
            Recency-Monetary pairs respectively).
            Only affects 'heat_sliced' and 'bar_sliced' plots.
            
        Returns
        -------
        Dict[str, go.Figure]
            Dictionary of DataFrame and Plotly Figure objects containing these keys:
            
            - 'df_rfm': DataFrame with rfm metrics, scores and segments
            - 'hist': Distribution histograms with boxplots
            - 'heat': Recency vs Frequency heatmap with Monetary values
            - 'heat_pairs': Three heatmaps showing counts between RF, FM and RM scores
            - 'heat_sliced': Heatmaps showing pairwise relationships sliced by third score
            - 'bar_sliced': Bars showing pairwise relationships sliced by third score
            - 'scatter_sliced': Scatter plots showing raw values relationships sliced by score
            - 'distr_grid': 3x2 grid showing raw distributions split by score groups
            - 'seg_bar': Customer segments distribution (bar chart)
            - 'seg_tree': Customer segments visualization (treemap)         
        """

        # Separate parameters into config
        self._rfm_separate_params(locals())

        # Validate parameters
        self._validate_rfm_parameters()
        
        # Set segment names
        self._init_segment_names()
        
        # Prepare RFM data
        self._prepare_rfm_data()

        # Calculate RFM scores
        self._calculate_rfm_scores()

        # Assign segments
        self._assign_segments()

        # Create visualizations
        result = self._create_visualizations()
        if self.config_rfm.return_rfm:
            result['df_rfm'] = self.df_rfm
        return result

    def _rfm_separate_params(self, kwargs: Dict[str, Any]) -> None:
        """Separate and validate parameters for config."""
        config_updates = {}
        config_fields = {f.name for f in fields(self.config_rfm)}
        config_types = get_type_hints(RFMConfig)
        for key, value in kwargs.items():
            if value is None:
                continue
            if not isinstance(key, str):
                raise TypeError(f"Parameter name must be string, got {type(key)}")
            if key in config_fields:
                expected_type = config_types[key]

                # Handle Literal types
                if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Literal:
                    if value not in expected_type.__args__:
                        raise ValueError(
                            f"Invalid value '{value}' for {key}. Must be one of: {expected_type.__args__}"
                        )

                # Handle regular types
                elif isinstance(expected_type, type) and not isinstance(value, expected_type):
                    raise TypeError(
                        f"Invalid type for '{key}'. Expected {expected_type}, got {type(value)}"
                    )

                config_updates[key] = value

        self.config_rfm.__dict__.update(config_updates)

    def _validate_rfm_parameters(self) -> None:
        """Validate all RFM parameters including DataFrame columns.
        
        Raises:
            ValueError: If required columns are missing or invalid values provided
            TypeError: If parameter types are incorrect
        """
        config = self.config_rfm
        df_columns = set(self.df.columns)
        
        # Validate DataFrame is not empty
        if self.df.empty:
            raise ValueError(
                "Input DataFrame is empty. RFM analysis requires non-empty data."
            )        
            
        # Validate required columns exist in DataFrame
        required_columns = {
            'user_id_col': config.user_id_col,
            'order_id_col': config.order_id_col,
            'date_col': config.date_col,
            'revenue_col': config.revenue_col
        }
        
        missing_columns = [
            col_name for param_name, col_name in required_columns.items() 
            if col_name and col_name not in df_columns
        ]
        
        if missing_columns:
            available_cols = '\n- '.join(sorted(df_columns))
            raise ValueError(
                f"Missing required columns in DataFrame:"
                f"- {', '.join(missing_columns)}"
            )

        # Validate date column type
        if not pd.api.types.is_datetime64_any_dtype(self.df[config.date_col]):
            raise TypeError(
                f"Date column '{config.date_col}' must be datetime type. "
                f"Found {self.df[config.date_col].dtype}. "
                "Use pd.to_datetime() to convert."
            )

        # Validate score_bins
        if config.score_bins not in {3, 5}:
            raise ValueError(f"score_bins must be 3 or 5, got {config.score_bins}")

        # Validate pair_for_slice
        if config.pair_for_slice not in {None, 'rf', 'fm', 'rm'}:
            raise ValueError(
                f"pair_for_slice must be None or one of ['rf','fm','rm'], "
                f"got '{config.pair_for_slice}'"
            )

        # 5. Validate quantile ranges
        if not 0 <= config.lower_quantile < config.upper_quantile <= 1:
            raise ValueError(
                f"Invalid quantile range: lower={config.lower_quantile}, "
                f"upper={config.upper_quantile}. Must satisfy 0 <= lower < upper <= 1"
            )
            
        if not self.config_rfm.user_id_col or not self.config_rfm.date_col:
            raise ValueError("user_id_col and date_col are required parameters")
        
    def _prepare_rfm_data(self) -> None:
        """Prepare the base RFM dataframe."""
        config = self.config_rfm
        df = self.df
        
        # Validate date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[config.date_col]):
            raise TypeError(
                f"Date column '{config.date_col}' must be pandas datetime type. "
                f"Found {df[config.date_col].dtype}. "
                "Please convert using pd.to_datetime() first."
            )
        # Set reference date if not provided
        if config.now_date is None:
            config.now_date = df[config.date_col].max()

        # Group by customer
        self.df_rfm = (
            df.groupby(config.user_id_col)
            .agg(
                recency=(config.date_col, lambda x: (config.now_date - x.max()).days),
                frequency=(config.order_id_col if config.order_id_col else config.date_col, 'nunique'),
                monetary=(config.revenue_col, 'sum')
            )
            .reset_index()
        )

    def _calculate_rfm_scores(self) -> None:
        """Calculate R, F, M scores."""
        config = self.config_rfm
        df_rfm = self.df_rfm
        # Handle case with single user
        if len(df_rfm) == 1:
            df_rfm['recency_score'] = config.score_bins
            df_rfm['frequency_score'] = config.score_bins
            df_rfm['monetary_score'] = config.score_bins
            df_rfm['rfm_score'] = f"{config.score_bins}{config.score_bins}{config.score_bins}"
            return        
        recency_labels = list(range(config.score_bins, 0, -1))
        frequency_labels = list(range(1, config.score_bins + 1))
        monetary_labels = list(range(1, config.score_bins + 1))
        # Calculate scores with qcut
        df_rfm['recency_score'] = pd.qcut(
            df_rfm['recency'].rank(method='first'),
            q=config.score_bins,
            labels=recency_labels
        )

        df_rfm['frequency_score'] = pd.qcut(
            df_rfm['frequency'].rank(method='first'),
            q=config.score_bins,
            labels=frequency_labels
        )

        df_rfm['monetary_score'] = pd.qcut(
            df_rfm['monetary'].rank(method='first'),
            q=config.score_bins,
            labels=monetary_labels
        )
        # Create RFM segment string
        df_rfm['rfm_score'] = (
            df_rfm['recency_score'].astype(str) +
            df_rfm['frequency_score'].astype(str) +
            df_rfm['monetary_score'].astype(str)
        )

    def _assign_segments(self) -> None:
        """Assign segment names based on RFM scores."""
        df_rfm = self.df_rfm

        # Map segments using predefined names
        df_rfm['rfm_segment'] = df_rfm['rfm_score'].replace(self.segment_names, regex=True) 
        # Handle unknown segments
        unknown_segments = df_rfm[df_rfm['rfm_segment'].isna()]['rfm_score'].unique()
        if len(unknown_segments) > 0:
            print(f"Warning: {len(unknown_segments)} unknown segments detected")
            # Fill any missing segments with 'Unknown'
            df_rfm['rfm_segment'] = df_rfm['rfm_segment'].fillna('Unknown')

    def _init_segment_names(self):
        # Choose the desired dictionary
        if self.config_rfm.score_bins == 3:
            self.segment_names = {
                r'333': 'Champions',
                r'332|323': 'Loyal',
                r'313': 'Potential Loyalists',
                r'233': 'Recent Customers',
                r'133|132|311': 'At Risk',
                r'321|312|232': 'Need Attention',
                r'331|322|223|222': 'Promising',
                r'231|221|213|212|123': 'About to Sleep',
                r'211|131|122|121|113|112': 'Hibernating',
                r'111': 'Lost'
            }
        elif self.config_rfm.score_bins == 5:
            self.segment_names = {
                r'555|554|544|545|454|455|445': 'Champions',
                r'155|154|144|214|215|115|114|113': 'Cannot Lose Them',
                r'543|444|435|355|354|345|344|335': 'Loyal',
                r'553|551|552|541|542|533|532|531|452|451|442|441|431|453|433|432|423|353|352|351|342|341|333|323': 'Potential Loyalists',
                r'512|511|422|421|412|411|311': 'Recent Customers',
                r'255|254|245|244|253|252|243|242|235|234|225|224|153|152|145|143|142|135|134|133|125|124': 'At Risk',
                r'535|534|443|434|343|334|325|324': 'Need Attention',
                r'525|524|523|522|521|515|514|513|425|424|413|414|415|315|314|313': 'Promising',
                r'331|321|312|221|213|231|241|251': 'About To Sleep',
                r'332|322|233|232|223|222|132|123|122|212|211': 'Hibernating', 
                r'111|112|121|131|141|151': 'Lost',
            }
        else:
            raise ValueError(f'Unknown score_bins: {self.config_rfm.score_bins}')

    def _create_visualizations(self) -> go.Figure:
        """Create all RFM visualizations."""
        config = self.config_rfm
        df_rfm = self.df_rfm
        # Create visualizations
        available_viz = {
            'hist': self._make_histograms,
            'heat': self._make_heatmap,
            'heat_pairs': self._make_pairwise_heatmaps,
            'heat_sliced': self._make_pairwise_heatmaps_by_score,
            'bar_sliced': self._make_pairwise_bar_by_score,
            'seg_bar': self._make_segment_bar,
            'seg_tree': self._make_segment_treemap,
            'scatter_sliced': self._make_pairwise_scatters_by_score,
            'distr_grid': self._make_raw_distribution_grid
        }
        invalid_viz = set(config.plots) - set(available_viz.keys())
        if invalid_viz:
            raise ValueError(f"Invalid visualization types: {invalid_viz}. "
                        f"Available options: {list(available_viz.keys())}")
            
        return {name: fn() for name, fn in available_viz.items() 
            if name in config.plots}

    def _make_histograms(self) -> go.Figure:
        """Add R, F, M histograms to figure with optional quantile trimming."""
        df_rfm = self.df_rfm

        # Check if need to cut the data
        trim_data = (hasattr(self.config_rfm, 'lower_quantile') and
                    hasattr(self.config_rfm, 'upper_quantile') and
                    (self.config_rfm.lower_quantile > 0 or
                    self.config_rfm.upper_quantile < 1))

        if trim_data:
            # Create a trimmed version of the data
            df_trimmed = df_rfm.copy()
            for col in ['recency', 'frequency', 'monetary']:
                lower = df_rfm[col].quantile(self.config_rfm.lower_quantile)
                upper = df_rfm[col].quantile(self.config_rfm.upper_quantile)
                df_trimmed[col] = df_rfm[col].clip(lower, upper)

            # Create subtitles with information about circumcision
            trim_info = f"<br><sup>Top row: Original distributions | Bottom row: Trimmed: "
            if self.config_rfm.lower_quantile > 0:
                trim_info += f"from {self.config_rfm.lower_quantile} "
            if self.config_rfm.upper_quantile <1:
                trim_info += f"to {self.config_rfm.upper_quantile} "
            trim_info += "quantile</sup>"
        else:
            trim_info = ""


        if trim_data:
            fig = make_subplots(
                rows=4, cols=3,
                horizontal_spacing=0.07,
                vertical_spacing=0.01,
            )
            # Add both data sets (original and cut)
            for i, col in enumerate(['recency', 'frequency', 'monetary'], start=1):
                # Original data
                fig.add_trace(
                    go.Histogram(
                        x=df_rfm[col],
                        name='Full data',
                        histnorm='probability',
                        marker_color='#4C78A8',
                        showlegend=False,
                        hovertemplate=f'{col.title()}: %{{x}}<br>Probability: %{{y:.2f}}<extra></extra>'
                    ),
                    row=2, col=i
                )

                # Cut data
                fig.add_trace(
                    go.Histogram(
                        x=df_trimmed[col],
                        name='Trimmed data',
                        histnorm='probability',
                        marker_color='#4C78A8',
                        showlegend=False,
                        hovertemplate=f'{col.title()}: %{{x}}<br>Probability: %{{y:.2f}}<extra></extra>'
                    ),
                    row=4, col=i
                )

                # box for both sets
                for r, data in [(1, df_rfm), (3, df_trimmed)]:
                    fig.add_trace(
                        go.Box(
                            x=data[col],
                            name='Boxplot',
                            marker_color='#4C78A8',
                            showlegend=False,
                            hovertemplate=f'{col.title()}: %{{x}}<extra></extra>'
                        ),
                        row=r, col=i
                    )

                    # Setting axes for box
                    fig.update_xaxes(
                        showticklabels=False,
                        showline=False,
                        ticks='',
                        showgrid=False,
                        row=r, col=i
                    )
                    fig.update_yaxes(
                        visible=False,
                        showline=False,
                        showgrid=False,
                        ticks='',
                        row=r, col=i
                    )

            # Setting up areas of graphs
            for i in range(1, 4):
                fig.update_yaxes(domain=[0.95, 1], row=1, col=i)
                fig.update_yaxes(domain=[0.55, 0.93], row=2, col=i)
                fig.update_yaxes(domain=[0.4, 0.45], row=3, col=i)
                fig.update_yaxes(domain=[0, 0.38], row=4, col=i)

            # Signatures of the axes
            fig.update_xaxes(title_text="Recency", row=4, col=1)
            fig.update_xaxes(title_text="Frequency", row=4, col=2)
            fig.update_xaxes(title_text="Monetary", row=4, col=3)

            fig.update_yaxes(title_text="Probability", row=2, col=1)
            fig.update_yaxes(title_text="Probability", row=4, col=1)
            fig.update_layout(
                height=600
                , width=900
                , title_text=f"RFM Score Distributions{trim_info}",
            )
        else:
            fig = make_subplots(
                rows=2, cols=3,
                horizontal_spacing=0.07,
                vertical_spacing=0.01,
            )
            # Add both both data sets (original and cut)
            for i, col in enumerate(['recency', 'frequency', 'monetary'], start=1):
                # Original data
                fig.add_trace(
                    go.Histogram(
                        x=df_rfm[col],
                        name='Full data',
                        histnorm='probability',
                        marker_color='#4C78A8',
                        showlegend=False,
                        hovertemplate=f'{col.title()}: %{{x}}<br>Probability: %{{y:.2f}}<extra></extra>'
                    ),
                    row=2, col=i
                )
                fig.add_trace(
                    go.Box(
                        x=df_rfm[col],
                        name='Boxplot',
                        marker_color='#4C78A8',
                        showlegend=False,
                        hovertemplate=f'{col.title()}: ' + '%{x}<extra></extra>'
                    ),
                    row=1, col=i
                )

                # Setting axes for box
                fig.update_xaxes(
                    showticklabels=False,
                    showline=False,
                    ticks='',
                    showgrid=False,
                    row=1, col=i
                )
                fig.update_yaxes(
                    visible=False,
                    showline=False,
                    showgrid=False,
                    ticks='',
                    row=1, col=i
                )

            # Setting up areas of graphs
            for i in range(1, 4):
                fig.update_yaxes(domain=[0.93, 1], row=1, col=i)
                fig.update_yaxes(domain=[0, 0.9], row=2, col=i)

            # Signatures of the axes
            fig.update_xaxes(title_text="Recency", row=2, col=1)
            fig.update_xaxes(title_text="Frequency", row=2, col=2)
            fig.update_xaxes(title_text="Monetary", row=2, col=3)

            fig.update_yaxes(title_text="Probability", row=2, col=1)
            fig.update_layout(
                height=350
                , width=900
                , title_text=f"RFM Score Distributions",
            )
        return fig

    def _make_heatmap(self) -> go.Figure:
        """Add RFM heatmap to figure."""
        config = self.config_rfm
        df_rfm = self.df_rfm

        # Create pivot table
        heatmap_data = pd.pivot_table(
            df_rfm,
            index='recency_score',
            columns='frequency_score',
            values='monetary',
            aggfunc='median',
            observed=False
        ).fillna(0)
        # Reverse for descending from top
        heatmap_data = heatmap_data.iloc[::-1]
        # Create heatmap
        heatmap = px.imshow(
            heatmap_data,
            text_auto=config.text_auto,
            color_continuous_scale=config.color_continuous_scale,
            labels=dict(
                x="Frequency Score",
                y="Recency Score",
                color="Monetary"
            ),
            aspect='auto'
        )
        heatmap.update_layout(
            title='RFM Heatmap'
            , xaxis_type='category'
            , yaxis_type='category'
            , height=400
            , width=500
        )
        
        return heatmap

    def _make_pairwise_heatmaps(self) -> go.Figure:
        """Create pairwise heatmaps for RF, FM and RM combinations."""
        config = self.config_rfm
        df_rfm = self.df_rfm
        
        fig = make_subplots(
            rows=1, 
            cols=3,
            subplot_titles=('Recency vs Frequency', 'Frequency vs Monetary', 'Recency vs Monetary'),
            horizontal_spacing=0.07
        )
        
        # Pairs for heatmap
        pairs = [
            ('recency_score', 'frequency_score'),
            ('frequency_score', 'monetary_score'), 
            ('recency_score', 'monetary_score')
        ]
        
        for i, (x, y) in enumerate(pairs, 1):
            # Create a summary table with the number of customers
            heatmap_data = df_rfm.groupby([x, y], observed=False).size().unstack(fill_value=0)
            
            # For recent_score make the reverse order
            if x == 'recency_score':
                heatmap_data = heatmap_data.iloc[::-1]
                
            # Add heatmap
            heatmap = px.imshow(
                heatmap_data,
                text_auto=config.text_auto,
                color_continuous_scale=config.color_continuous_scale,
                labels=dict(x=y.replace('_score', '').title(),
                        y=x.replace('_score', '').title(),
                        color="Count"),
                aspect='auto'
            )
            
            fig.add_trace(heatmap.data[0], row=1, col=i)
            fig.update_xaxes(title_text=y.replace('_score', '').title(), type='category', title_standoff=5, row=1, col=i)
            fig.update_yaxes(title_text=x.replace('_score', '').title(), type='category', title_standoff=5, row=1, col=i)
        
        fig.update_annotations(font_size=14)
        
        fig.update_layout(
            coloraxis=dict(colorscale=config.color_continuous_scale, colorbar_title_text='Count'),
            height=330,
            width=900,
            title_text='Pairwise RFM Score Combinations'
        )
        
        return fig

    def _make_pairwise_heatmaps_by_score(self) -> go.Figure:
        """Create pairwise heatmaps for FM combinations grouped by recency score.
        
        Returns:
            go.Figure: Plotly figure with grid of heatmaps (rows are recency scores, 
                    columns are FM pairs)
        """
        config = self.config_rfm
        df_rfm = self.df_rfm
        
        # Get min and max counts across all heatmaps for consistent coloring
        min_count = float('inf')
        max_count = -float('inf')
        # Determine which pair to plot based on config
        if config.pair_for_slice == 'rf':
            x, y, slice_var = 'recency_score', 'frequency_score', 'monetary_score'
        elif config.pair_for_slice == 'fm':
            x, y, slice_var = 'frequency_score', 'monetary_score', 'recency_score'
        elif config.pair_for_slice == 'rm':
            x, y, slice_var = 'recency_score', 'monetary_score', 'frequency_score'
        else:  # default
            x, y, slice_var = 'frequency_score', 'monetary_score', 'recency_score'        
        # First pass to determine color scale range
        for score_val in range(1, config.score_bins + 1):
            filtered_df = df_rfm[df_rfm[slice_var] == score_val]
            heatmap_data = filtered_df.groupby([y, x], observed=False).size().unstack(fill_value=0)
            current_max = heatmap_data.max().max()
            current_min = heatmap_data.min().min()
            if current_max > max_count:
                max_count = current_max
            if current_min < min_count:
                min_count = current_min

        # Create subplot grid
        name_map = {
            'recency_score': 'R',
            'frequency_score': 'F', 
            'monetary_score': 'M'
        }
        name_x = name_map[x]
        name_y = name_map[y]
        name_slice_var = name_map[slice_var]
        # Calculate subplot grid dimensions
        n_rows = (config.score_bins + 2) // 3  # This gives 1 row for 3 bins, 2 rows for 5 bins
        n_cols = min(3, config.score_bins)
        
        # Generation of subplot_titles
        subplot_titles = []
        for score_val in range(1, config.score_bins + 1):
            subplot_titles.append(f"{name_x} vs {name_y} ({name_slice_var}={score_val})")
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.06,
            vertical_spacing=0.06,
            shared_yaxes=False,
            shared_xaxes=False
        )
        
        # Second pass to create actual heatmaps
        for score_val in range(1, config.score_bins + 1):
            filtered_df = df_rfm[df_rfm[slice_var] == score_val]
            
            heatmap_data = filtered_df.groupby([y, x], observed=False).size().unstack(fill_value=0)
            heatmap_data = heatmap_data[sorted(heatmap_data.columns)]
            # Calculate row and column position
            row_idx = (score_val - 1) // n_cols + 1
            col_idx = (score_val - 1) % n_cols + 1
            
            # Create imshow and get its trace
            heatmap = px.imshow(
                heatmap_data,
                text_auto=config.text_auto,
                color_continuous_scale=config.color_continuous_scale,
                labels=dict(color="Count"),
                zmin=min_count,
                zmax=max_count,
                aspect='auto'
            )
            
            # Add the trace from imshow
            fig.add_trace(
                heatmap.data[0], 
                row=row_idx, 
                col=col_idx
            )
            
            # Add hover
            fig.update_traces(
                hovertemplate=(
                    f"{name_y}: %{{y}}<br>"
                    f"{name_x}: %{{x}}<br>"
                    f"{name_slice_var}: {score_val}<br>"
                    "Count: %{z}<extra></extra>"
                ),
                selector=dict(type='heatmap'),  
                row=row_idx,
                col=col_idx
            )
            
            # Show x-axis labels on bottom row only
            show_xlabels = (row_idx == n_rows) or (n_rows == 2 and col_idx == 3)
            fig.update_xaxes(
                visible=True if show_xlabels else False,
                title_text=name_x if show_xlabels else None,
                row=row_idx,
                col=col_idx,
                type='category',
                title_standoff=5,
                showticklabels=True if show_xlabels else False,
            )
            
            # Show y-axis labels on first column only
            show_ylabels = (col_idx == 1)
            fig.update_yaxes(
                title_text=name_y if show_ylabels else None,
                row=row_idx,
                col=col_idx,
                type='category',
                title_standoff=5,
                showticklabels=show_ylabels,
            )
        
        # Add colorbar
        fig.update_layout(
            coloraxis=dict(
                colorscale=config.color_continuous_scale,
                cmin=min_count,
                cmax=max_count,
                colorbar=dict(
                    title='Count',
                    len=0.8 if n_rows == 2 else None,
                    y=0.5,
                    yanchor='middle'
                )
            ),
            height=500 if n_rows == 2 else 300,
            width=800,
            title='Customer Distribution by RFM Scores',
            margin=dict(t=70)
        )
        fig.update_annotations(font_size=14)
        return fig

    def _make_pairwise_bar_by_score(self) -> go.Figure:
        """Create pairwise barcharts for RFM combinations using bar plots with faceting.
        
        Returns:
            go.Figure: Plotly figure with faceted bar charts showing distribution
        """
        config = self.config_rfm
        df_rfm = self.df_rfm
        
        # Determine which pair to plot based on config
        if config.pair_for_slice == 'rf':
            x, color, facet = 'frequency_score', 'recency_score', 'monetary_score'
            x_title, color_title, facet_title = 'F', 'R', 'M'
        elif config.pair_for_slice == 'fm':
            x, color, facet = 'frequency_score', 'monetary_score', 'recency_score'
            x_title, color_title, facet_title = 'F', 'M', 'R'
        elif config.pair_for_slice == 'rm':
            x, color, facet = 'recency_score', 'monetary_score', 'frequency_score'
            x_title, color_title, facet_title = 'R', 'M', 'F'
        else:  # default
            x, color, facet = 'recency_score', 'frequency_score', 'monetary_score'
            x_title, color_title, facet_title = 'R', 'F', 'M'        
        # Prepare data - group by all three scores and count occurrences
        grouped_df = (
            df_rfm.groupby(
                ['recency_score', 'frequency_score', 'monetary_score'],
                observed=False
            )
            .size()
            .reset_index(name='count')
        )
        
        # Create the bar plot with faceting
        fig = px.bar(
            grouped_df,
            x=x,
            y='count',
            color=color,
            facet_col=facet,
            facet_col_wrap=3,
            category_orders={
                'recency_score': list(range(1, config.score_bins + 1)),
                'frequency_score': list(range(1, config.score_bins + 1)),
                'monetary_score': list(range(1, config.score_bins + 1))
            },
            labels={
                'recency_score': 'Recency Score',
                'frequency_score': 'Frequency Score',
                'monetary_score': 'Monetary Score',
                'count': 'Count'
            },
            barmode='group',
            title='Customer Distribution by RFM Scores',
            facet_col_spacing=0.05
        )
        
        # Update layout for better readability
        fig.update_layout(
            height=350 * ((config.score_bins + 2) // 3),  # Adjust height based on number of rows
            width=1100,
            margin=dict(t=100),
            showlegend=True,
            title_y = 0.97,
        )
        
        # Update facet/subplot titles
        # fig.for_each_annotation(lambda a: a.update(text=f"{facet_title}={a.text.split('=')[-1]}"))
        # Legend positioning
        legend_config = {
            'orientation': "h",
            'yanchor': "top",
            'y': 1.09,
            'xanchor': "center",
            'x': 0.5,
            'itemsizing': "constant"
        }
        fig.update_layout(legend=legend_config)        
        # Update axes
        fig.update_xaxes(type='category')
        fig.update_xaxes(title_text=x_title, row=1, col=3)
        
        return fig

    def _make_segment_bar(self) -> go.Figure:
        """Add segment bar chart to figure."""
        df_rfm = self.df_rfm

        # Count customers per segment
        segment_stats = (
            df_rfm['rfm_segment']
            .value_counts()
            .reset_index()
            .assign(percentage=lambda x: x['count'] / x['count'].sum() * 100)
            .sort_values('count', ascending=True)
        )
        # Create bar chart
        bar = px.bar(
            segment_stats,
            x='count',
            y='rfm_segment',
            orientation='h',
            text=segment_stats['percentage'].apply(lambda x: f'{x:.1f}%'),
            custom_data=['count', 'percentage']
        )
        bar.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Count: %{customdata[0]:,}<br>"
                "Percentage: %{customdata[1]:.1f}%<br>"
                "<extra></extra>"
            ),
        )
        # Update axes
        bar.update_xaxes(visible=False,)
        bar.update_yaxes(title_text="Segment")
        bar.update_layout(
            title = 'Customer Segments Distribution'
            , height=400
            , width=600
        )
        return bar

    def _make_segment_treemap(self) -> go.Figure:
        """Add segment treemap to figure."""
        df_rfm = self.df_rfm

        # Count customers per segment
        segment_stats = (
            df_rfm['rfm_segment']
            .value_counts()
            .reset_index()
            .assign(percentage=lambda x: x['count'] / x['count'].sum())
        )
        # Create color mapping
        segment_colors = {
            'Champions': '#5A7EBA',  
            'Loyal': '#6BA368',      
            'Potential Loyalists': '#D4A95D',  
            'Recent Customers': '#E49E84',     
            'Promising': '#A38671',  
            'At Risk': '#E68C56',    
            'Need Attention': '#8A6EAD',  
            'Cannot Lose Them': '#B05F6D',  
            'About to Sleep': '#6C9D9A',  
            'Hibernating': '#9E9E9E',  
            'Lost': '#C38D9E',        
        }
        # Create treemap
        treemap = px.treemap(
            segment_stats,
            path=[px.Constant('All'), 'rfm_segment'],
            values='count',
            color='rfm_segment',
            color_discrete_map=segment_colors,
            custom_data=['percentage'],
            hover_name='rfm_segment'
        )
        treemap.update_traces(
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Customers: %{value:,}<br>"
                "Percentage: %{customdata[0]:.1%}<br>"
                "<extra></extra>"
            ),
            marker=dict(cornerradius=5),
            textinfo="label+value",
            texttemplate="<b>%{label}</b><br>%{value:,}<br>%{customdata[0]:.0%}",
            textfont=dict(size=14)
        )
        # Update color for 'All'
        treemap_colors = list(treemap.data[0].marker.colors)
        treemap_colors[0] = '#E6E6FA'
        treemap.data[0].marker.colors = treemap_colors
        # Custom hover for 'All' to remove percentage
        treemap.data[0].hovertemplate = [
            "<b>All Customers</b><br>Total: %{value:,}<br><extra></extra>"
            if label == 'All' else t
            for label, t in zip(treemap.data[0].labels, treemap.data[0].hovertemplate)
        ]
        treemap.update_layout(
            title='Customer Segments Distribution',
            margin=dict(t=50, l=0, r=0, b=0),
            title_x=0.01,
            title_y=0.97,
            width=900
        )
        return treemap
    
    def _make_pairwise_scatters_by_score(self) -> go.Figure:
        """Create pairwise scatter plots for RF, FM and RM combinations grouped by third variable score.
        
        Returns:
            go.Figure: Plotly figure with grid of scatter plots (rows are third variable scores, 
                    columns are RFM pairs)
        """
        config = self.config_rfm
        df_rfm = self.df_rfm
        
        # Estimation for names
        name_map = {
            'recency_score': 'R',
            'frequency_score': 'F', 
            'monetary_score': 'M',
            'recency': 'R',
            'frequency': 'F',
            'monetary': 'M'
        }

        # Generation of headlines
        subplot_titles = []
        for score_val in range(config.score_bins, 0, -1):
            for x, y in [('recency', 'frequency'),
                        ('frequency', 'monetary'),
                        ('recency', 'monetary')]:
                # We determine the third variable
                third_var = next(v for v in ['recency', 'frequency', 'monetary'] 
                            if v not in (x, y))
                pair_name = f"{name_map[x]} vs {name_map[y]}"
                subplot_titles.append(f"{pair_name} ({name_map[third_var]}={score_val})")
                
        # Create a chart of graphs
        fig = make_subplots(
            rows=config.score_bins, 
            cols=3,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.07,
            vertical_spacing=0.04,
            shared_xaxes=True            
        )

        # Fill out Scatter plots
        for score_var in ['recency_score', 'frequency_score', 'monetary_score']:
            for score_val in range(1, config.score_bins + 1):
                filtered_df = df_rfm[df_rfm[score_var] == score_val]
                
                col_idx = 0
                for x, y in [('recency', 'frequency'), 
                            ('frequency', 'monetary'),
                            ('recency', 'monetary')]:
                    col_idx += 1
                    
                    # We miss if there is a coincidence with the third variable
                    third_var = next(v for v in ['recency', 'frequency', 'monetary'] 
                                if v not in (x, y))
                    if score_var.startswith(third_var):
                        continue
                    
                    # Determine row (reverse order)
                    row_idx = config.score_bins - score_val + 1
                    
                    # Create Scatter Plot
                    scatter = px.scatter(
                        filtered_df,
                        x=x,
                        y=y,
                        opacity=0.6,
                        hover_data={
                            x: ':.1f',
                            y: ':.1f',
                            'recency': ':.1f',
                            'frequency': ':.0f',
                            'monetary': ':.1f'
                        }
                    )
                    
                    # Add the trace
                    fig.add_trace(
                        scatter.data[0],
                        row=row_idx,
                        col=col_idx
                    )
                    
                    # Set Hover
                    fig.update_traces(
                        hovertemplate=(
                            f"<b>{name_map[x]}</b>: %{{x:.1f}}<br>"
                            f"<b>{name_map[y]}</b>: %{{y:.1f}}<br>"
                            f"<b>{name_map[third_var]}</b>: {score_val}<br>"
                            "<extra></extra>"
                        ),
                        selector=dict(type='scatter'),
                        row=row_idx,
                        col=col_idx
                    )
                    
                    # Only show x-axis labels on bottom row
                    show_xlabels = (row_idx == config.score_bins)
                    fig.update_xaxes(
                        title_text=name_map[x] if show_xlabels else None,
                        row=row_idx,
                        col=col_idx,
                        title_standoff=5,
                    )
                    
                    # Only show y-axis labels on first column
                    fig.update_yaxes(
                        title_text=name_map[y],
                        row=row_idx,
                        col=col_idx,
                        title_standoff=5,
                    )

        # General settings
        fig.update_layout(
            height=250 * config.score_bins,
            width=1000,
            title_text='RFM Raw Values Distribution by Score Groups',
            margin=dict(t=70),
            showlegend=False
        )
        
        # Setting headlines
        fig.update_annotations(font_size=14)
        
        return fig    
    
    def _make_raw_distribution_grid(self) -> go.Figure:
        """
        Create 3x2 grid showing raw distributions split by score groups.
        """
        config = self.config_rfm
        df_rfm = self.df_rfm
        plot_type = config.distr_plot_type
        default_colors = pio.templates[pio.templates.default].layout.colorway
        score_colors = default_colors[:config.score_bins]
        # Abbreviations for names
        name_map = {
            'recency': 'Recency',
            'frequency': 'Frequency',
            'monetary': 'Monetary',
            'recency_score': 'R Score',
            'frequency_score': 'F Score',
            'monetary_score': 'M Score'
        }
        
        # Create a 3x2 grid with shared y-axis
        fig = make_subplots(
            rows=3, 
            cols=2,
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
            shared_yaxes=True  # A total axis Y for all figs
        )
        
        # Main row cycle (RAW variables)
        for row, raw_var in enumerate(['recency', 'frequency', 'monetary'], 1):
            
            # Determine two score variables (not current RAW)
            score_vars = [v for v in ['recency_score', 'frequency_score', 'monetary_score'] 
                        if not v.startswith(raw_var)]
            
            # First column - split by first score variable
            for score_val in range(1, config.score_bins+1):
                filtered_df = df_rfm[df_rfm[score_vars[0]] == score_val]
                
                if plot_type == 'box':
                    trace = go.Box(
                        y=filtered_df[raw_var],
                        name=f"{score_val}",
                        legendgroup=f"Score {score_val}",
                        showlegend=False,
                        marker_color=score_colors[score_val-1],
                        width=0.5
                    )
                else:  # violin
                    trace = go.Violin(
                        y=filtered_df[raw_var],
                        name=f"{score_val}",
                        legendgroup=f"Score {score_val}",
                        showlegend=False,
                        marker_color=score_colors[score_val-1],
                        width=0.7,
                        box=dict(
                            visible=True,
                            width=0.3,  
                        ),
                    )
                
                fig.add_trace(trace, row=row, col=1)
            
            # Second column - split by second score variable
            for score_val in range(1, config.score_bins+1):
                filtered_df = df_rfm[df_rfm[score_vars[1]] == score_val]
                
                if plot_type == 'box':
                    trace = go.Box(
                        y=filtered_df[raw_var],
                        name=f"{score_val}",
                        legendgroup=f"Score {score_val}",
                        showlegend=False,
                        marker_color=score_colors[score_val-1],
                        width=0.5
                    )
                else:  # violin
                    trace = go.Violin(
                        y=filtered_df[raw_var],
                        name=f"{score_val}",
                        legendgroup=f"Score {score_val}",
                        showlegend=False,
                        marker_color=score_colors[score_val-1],
                        width=0.7,
                        box=dict(
                            visible=True,
                            width=0.3,  
                        ),
                    )
                
                fig.add_trace(trace, row=row, col=2)
            
            # Y-axis labels
            fig.update_yaxes(
                title_text=name_map[raw_var],
                title_standoff=5,
                row=row, col=1
            )
            
            # X-axis labels
            fig.update_xaxes(
                title_text=name_map[score_vars[0]],
                title_standoff=5,
                row=row, col=1
            )
            fig.update_xaxes(
                title_text=name_map[score_vars[1]],
                title_standoff=5,
                row=row, col=2
            )
        for score_val in range(1, config.score_bins+1):
            fig.data[score_val-1].showlegend = True
        # General settings
        if plot_type == 'box':
            fig.update_layout(boxmode='group')
        else:
            fig.update_layout(violinmode='group')
        fig.update_layout(
            title_y=0.98,
            height=900,  # Increased height for 3 lines
            width=800,   # Reduced width for 2 columns
            margin=dict(t=80),
            title_text="Raw Value Distributions by Score Groups",
            legend = dict(
                title='Score Group'
                , orientation='h'
                , yanchor="bottom"
                , y=1.01
                , xanchor= "center"
                , x=0.5
                , itemsizing="constant"
            )  
        )
        
        return fig