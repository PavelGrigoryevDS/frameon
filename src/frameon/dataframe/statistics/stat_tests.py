from dataclasses import dataclass, field, fields
import contextlib
import textwrap
import warnings
from typing import (Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar,
                    Union, get_type_hints)

import joblib
import numpy as np
import pandas as pd
import patsy
import re
from IPython.display import display
import plotly.graph_objects as go
from joblib import Parallel, cpu_count, delayed
import seaborn as sns
from matplotlib import pyplot as plt
import pingouin as pg
from plotly.subplots import make_subplots
from rich.console import Console
from rich.markdown import Markdown
from scikit_posthocs import posthoc_dunn
from scipy import stats
from scipy.optimize import root_scalar
from scipy.stats import bootstrap as scipy_bootstrap, norm
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                            RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression,
                                     mutual_info_classif, mutual_info_regression)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (accuracy_score, log_loss, mean_squared_error,
                           roc_auc_score, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from tqdm import tqdm
from frameon.utils.plotting import CustomFigure
from frameon.utils.miscellaneous import style_dataframe


rich_console = Console(highlight=False)
rprint = rich_console.print
@dataclass
class FeatureImportanceResult:
    """Class for storing feature importance analysis results"""
    importance_df: pd.DataFrame
    model: Optional[object] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
@dataclass
class TestConfig:
    alpha: float = 0.05
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
    equal_var: bool = False
    reference_group: Optional[str] = None
    h0: Optional[str] = None
    h1: Optional[str] = None

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

@dataclass
class BootstrapResult:
    """Container for bootstrap results with accessible attributes."""
    table: pd.DataFrame
    bootstrap_distribution: np.ndarray
    plot: go.Figure

class StatisticalTests:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize StatisticalTests with a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for statistical tests. Column validation will be performed
            when specific tests are run.
        """
        self._validate_dataframe(df)
        self.df = df
        self.dv = None
        self.between = None
        self.config = TestConfig()

    def _validate_dataframe(self, df) -> None:
        """Validate DataFrame."""
        if df is None:
            raise ValueError("DataFrame cannot be None")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if df.empty:
            raise ValueError("DataFrame cannot be empty")

    def _validate_alternative(self, alternative: str) -> str:
        """Validate and standardize alternative hypothesis parameter."""
        valid_alternative = ["two-sided", "2s", 'greater', "g", 'less', "l"]
        if alternative not in valid_alternative:
            raise ValueError(
                f"alternative must be one of {valid_alternative}, but got {alternative}")

        alternative_map = {
            "two-sided": "two-sided",
            "2s": "two-sided",
            "greater": "greater",
            "g": "greater",
            "less": "less",
            "l": "less"
        }
        return alternative_map[alternative]

    def _validate_config(self, updates: Dict[str, Any]) -> None:
        """Validate and update configuration parameters."""
        config_updates = {}
        config_fields = {f.name for f in fields(self.config)}
        config_types = get_type_hints(TestConfig)

        for key, value in updates.items():
            if value is None:
                continue

            if not isinstance(key, str):
                raise TypeError(f"Parameter name must be string, got {type(key)}")

            if key not in config_fields:
                raise ValueError(f"'{key}' is not a valid configuration parameter")

            expected_type = config_types[key]
            self._validate_type(key, value, expected_type)
            config_updates[key] = value

        self.config.__dict__.update(config_updates)

    def _validate_type(self, key: str, value: Any, expected_type: Any) -> None:
        """Validate a single value against its expected type."""
        # Handle Optional types
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            expected_types = [t for t in expected_type.__args__ if t is not type(None)]
            if not any(isinstance(value, t) for t in expected_types):
                raise TypeError(
                    f"Invalid type for '{key}'. Expected one of {expected_types}, got {type(value)}"
                )
            return

        # Handle Literal types
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Literal:
            if value not in expected_type.__args__:
                raise ValueError(
                    f"Invalid value '{value}' for {key}. Must be one of: {expected_type.__args__}"
                )
            return

        # Handle regular types
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Invalid type for '{key}'. Expected {expected_type}, got {type(value)}"
            )

    def normality(
        self,
        dv: str,
        between: str,
        alpha: float = 0.05,
        method: str = 'jarque_bera',
        show_qqplot: bool = False,
        width: int = 9,
        height: int = 6
    ) -> None:
        """
        Perform normality tests for each group and display QQ-plots.

        Parameters:
        -----------
        dv : str
            Name of the dependent variable column.
        between : str
            Name of the grouping variable column.
        alpha : float, optional
            Significance level (default: 0.05).
        method : str, optional
            Normality test method ('shapiro', 'normaltest', 'jarque_bera').
        show_qqplot : bool, optional
            Whether to show QQ-plots for each group (default: True).
        width : int, optional
            Width of the figure in inches (default: 12).
        height : int, optional
            Height of the figure in inches (default: 8).
            
        Returns:
        --------
            None            
        """
        # Update and validate config
        kwargs = dict(alpha=alpha)
        self._validate_config(kwargs)
        # Validate data
        self.dv = dv
        self.between = between
        self._validate_dv_and_between(test_name=None)

        # Check for missing values
        df, na_warnings = self._check_and_handle_na(dv, between)
        text_output = ""
        if na_warnings:
            text_output += "\n".join(na_warnings) + '\n'

        # Prepare report header
        groups = df[between].unique()
        text = f"""
        [bold]Normality Test Report[/bold]\n
        - Dependent Variable: {dv}
        - Grouping Variable: {between}
        - Alpha Level: {alpha}
        - Test Method: {method}\n"""
        text_output += textwrap.dedent(text)

        # Add hypothesis formulation
        h0 = "H0: The data follows a normal distribution"
        h1 = "H1: The data does not follow a normal distribution"

        text = f"""
        [bold]🔹 Hypothesis Formulation[/bold]\n
        - {h0}
        - {h1}"""
        text_output += textwrap.dedent(text)
        rprint(text_output)
        # Perform normality tests for each group
        norm_results = []
        for group in groups:
            group_data = df[df[between] == group][dv]

            # Run the specified normality test
            if method == 'shapiro':
                test_result = pg.normality(group_data, method='shapiro')
            elif method == 'normaltest':
                test_result = pg.normality(group_data, method='normaltest')
            elif method == 'jarque_bera':
                test_result = pg.normality(group_data, method='jarque_bera')
            else:
                raise ValueError(f"Unknown normality test method: {method}")

            test_result['group'] = str(group)
            norm_results.append(test_result)

        # Combine results and display
        all_results = pd.concat(norm_results)
        all_results['Reject H0'] = all_results['pval'] < alpha
        display(all_results.set_index('group').round(2))
        # Check results and print conclusion
        non_normal_groups = all_results[all_results['pval'] < alpha]['group'].tolist()
        if non_normal_groups:
            rprint(f"[bright_red]Warning: Non-normal distributions detected in groups: {', '.join(non_normal_groups)}[/bright_red]")
        else:
            rprint("[green]All groups appear normally distributed[/green]")

        # Create QQ-plots for each group if requested
        if show_qqplot and len(groups) > 0:
            n_groups = len(groups)
            n_cols = 3
            n_rows = (n_groups + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(width, height),
                squeeze=False
            )

            point_kwargs = {
                "color": "#2873a8",
            }

            for i, (group, ax) in enumerate(zip(groups, axes.flatten())):
                group_data = df[df[between] == group][dv]

                # Calculate statistics
                skew_val = group_data.skew()
                kurt_val = group_data.kurtosis()

                pg.qqplot(group_data, confidence=False, ax=ax, **point_kwargs)


                r_squared = None
                for text in ax.texts:
                    text_content = text.get_text()
                    if '$R^2=' in text_content:
                        match = re.search(r'\$R\^2=([0-9.]+)\$', text_content)
                        if match:
                            r_squared = float(match.group(1))
                            text.remove()
                            break

                stats_text = f"Skew: {skew_val:.2f}\nKurt: {kurt_val:.2f}\nR2: {r_squared:.3f}"
                ax.text(
                    0.98,
                    0.02,
                    stats_text,
                    transform=ax.transAxes,
                    ha='right',
                    va='bottom',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2')
                )
                ax.lines[1].set_color('#e25559')
                ax.set_title(f"Group: {group}")
                ax.grid(True, alpha=0.3, linestyle='--')
                if i % n_cols != 0:
                    ax.set_ylabel('')
                if i < (n_rows - 1) * n_cols:
                    ax.set_xlabel('')

            for i in range(len(groups), n_rows * n_cols):
                fig.delaxes(axes.flatten()[i])

            plt.tight_layout()
            plt.show()

    def levene(
        self,
        dv: str,
        between: str,
        alpha: float = 0.05,
        return_results: bool = False
    ) -> Optional[Tuple[float, float, bool]]:
        """
        Perform Levene's test for equality of variances.

        Parameters:
        -----------
        dv : str
            Name of the dependent variable column.
        between : str
            Name of the column containing group labels.
        alpha : float, optional
            Significance level (default: 0.05).
        return_results : bool, optional
            If True, returns test results (default: False).
            
        Returns:
        --------
            Optional[Tuple[float, float, bool]]                  
        """
        # Update and validate config
        kwargs = dict(
            alpha=alpha
        )
        self.dv = dv
        self.between = between
        self._validate_config(kwargs)
        self._validate_dv_and_between(test_name='levene')
        df, na_warnings = self._check_and_handle_na(dv, between)
        groups = df[between].unique()
        group1, group2 = groups[0], groups[1]
        # Check missings
        text_output = ""
        if na_warnings:
            text_output += "\n".join(na_warnings) +'\n'
        text = f"""
        [bold]Levene's Test Report[/bold]\n
        - Dependent Variable: {dv}
        - Groups: {group1} vs {group2}
        - Alpha Level: {alpha}"""
        text_output += textwrap.dedent(text)
        rprint(text_output)
        result = pg.homoscedasticity(
            data=df,
            dv=dv,
            group=between,
            method='levene',
            alpha=self.config.alpha
        )

        W, p_value = result.loc['levene', ['W', 'pval']]
        equal_var = p_value >= self.config.alpha

        if return_results:
            return W, p_value, equal_var

        display(result)

        if p_value < self.config.alpha:
            p_value_str = f"[bright_red]{p_value:.4f}[/bright_red]"
        else:
            p_value_str = f"[green]{p_value:.4f}[/green]"
        conclusion = (
            f"At alpha = {self.config.alpha}, we {'[bright_red]reject[/bright_red]' if p_value < self.config.alpha else '[green]fail to reject[/green]'} "
            f"the null hypothesis that the variances are equal (p={p_value_str})."
        )
        rprint(f"[bold]Conclusion:[/bold] {conclusion}", highlight=False)

    def _validate_dv_and_between(self, test_name: str) -> None:
        """Validate parameters for statistical tests.

        Handles different test requirements:

        - ttest/mannwhitney: exactly 2 groups
        - anova/kruskal: 2+ groups (per factor)
        - factorial designs: multiple factors

        Parameters:
        -----------
        test_name : str
            Type of test being performed ('ttest', 'mannwhitney', 'anova', 'kruskal')
        """
        # Basic validation
        dv = self.dv
        df = self.df
        between = self.between
        if dv is None or between is None:
            raise ValueError("Both dv and between must be specified")

        if dv not in df.columns:
            raise ValueError(f"Dependent variable column '{dv}' not found in DataFrame")

        if not pd.api.types.is_numeric_dtype(df[dv]):
            raise ValueError(f"Dependent variable '{dv}' must be numeric")

        # Handle single factor vs multiple factors
        between_factors = [between] if isinstance(between, str) else between

        for factor in between_factors:
            if factor not in df.columns:
                raise ValueError(f"Between factor column '{factor}' not found in DataFrame")

            groups = df[factor].unique()
            n_groups = len(groups)

            # Test-specific validations
            if test_name in ['ttest', 'mannwhitney']:
                if n_groups != 2:
                    raise ValueError(
                        f"For {test_name}, between factor must have exactly 2 groups. "
                        f"Found {n_groups} in '{factor}': {groups}"
                    )

                if self.config.reference_group:
                    if self.config.reference_group not in groups:
                        raise ValueError(
                            f"reference_group '{self.config.reference_group}' must be one of {groups}"
                        )
                    if self.config.alternative != 'two-sided' and self.config.reference_group != groups[0]:
                        groups = groups[::-1]
                elif self.config.alternative != 'two-sided':
                    raise ValueError("For one-tailed tests, reference_group must be specified")

            elif test_name in ['anova', 'kruskal']:
                if n_groups < 2:
                    raise ValueError(
                        f"For {test_name}, between factor must have at least 2 groups. "
                        f"Found {n_groups} in '{factor}': {groups}"
                    )

                if n_groups < 3 and test_name == 'anova' and len(between_factors) == 1:
                    rprint(
                        f"[yellow]Warning: ANOVA with only 2 groups in '{factor}' is equivalent to t-test[/yellow]"
                    )

        # # Additional checks for factorial designs
        # if test_name in ['anova', 'kruskal'] and len(between_factors) > 1:
        #     if df[between_factors].duplicated().any():
        #         raise ValueError(
        #             "Duplicate rows detected when combining factors. "
        #             "Ensure factorial design is balanced or use repeated measures."
        #         )

    def _check_and_handle_na(self, dv: str, between: Union[str, List[str]]) -> Tuple:
        """
        Check for and handle missing values in the dataset.

        This method:

        1. Identifies NA values in both the dependent and grouping variables
        2. Reports which groups have missing data (for each factor in multi-factor designs)
        3. Drops rows with missing values
        4. Returns warnings about missing data

        Parameters:
        -----------
        dv : str
            Name of the dependent variable column
        between : str or list of str
            Name of the grouping variable column(s)

        Returns:
        --------
        list
            List of warning messages about missing data (empty if no NAs found)
        """
        df = self.df
        na_info = []

        # Convert between to list if single factor
        between_factors = [between] if isinstance(between, str) else between
        total_rows = len(df)

        # Check for missing values in each grouping variable
        if between_factors:
            for factor in between_factors:
                na_count = df[factor].isna().sum()
                if na_count > 0:
                    na_percentage = (na_count / total_rows) * 100
                    na_info.append(
                        f"⚠️ Found {na_count} ({na_percentage:.2f}%) "
                        f"missing values in grouping variable '{factor}'"
                    )

        # Check for missing values in DV
        na_dv = df[dv].isna().sum()
        if na_dv > 0:
            na_percentage = (na_dv / total_rows) * 100
            na_info.append(
                f"⚠️ Found {na_dv} ({na_percentage:.2f}%) "
                f"missing values in dependent variable '{dv}'"
            )
        if between_factors:
            # For each factor, report DV missingness by group if the factor has < 20 groups
            for factor in between_factors:
                if df[factor].nunique() <= 20:  # Only show if reasonable number of groups
                    na_by_group = (df.groupby(factor, observed=True)[dv]
                                .apply(lambda x: x.isna().sum())
                                .reset_index(name='na_count'))

                    for _, row in na_by_group[na_by_group['na_count'] > 0].iterrows():
                        group = row[factor]
                        group_size = len(df[df[factor] == group])
                        na_pct = (row['na_count'] / group_size) * 100
                        na_info.append(
                            f"⚠️ Group '{group}' in factor '{factor}' has "
                            f"{row['na_count']} ({na_pct:.2f}%) missing values in '{dv}'"
                        )

        # Remove rows with any missing values in the specified columns
        initial_rows = len(df)
        cols_to_check = [dv]
        if between_factors:
            cols_to_check += between_factors
        df = df.dropna(subset=cols_to_check)
        removed_rows = initial_rows - len(df)

        # Add removal info if any rows were dropped
        if removed_rows > 0:
            removal_pct = (removed_rows / initial_rows) * 100
            na_info.append(
                f"Removed {removed_rows} rows ({removal_pct:.2f}% of data) "
                f"with missing values in: {', '.join(cols_to_check)}"
            )

        return df, na_info

    def ttest(
        self,
        dv: str,
        between: str,
        alpha: float = 0.05,
        alternative: str = 'two-sided',
        reference_group: Optional[str] = None,
        correction: Optional[Union[str, bool]] = None,
        h0: Optional[str] = None,
        h1: Optional[str] = None
    ) -> None:
        """
        Perform comprehensive independent samples t-test analysis with step-by-step reporting.

        Parameters:
        -----------
        dv : str
            Name of the dependent variable column.
        between : str
            Name of the column containing group labels.
        alpha : float, optional
            Significance level (default: 0.05).
        alternative : str, optional
            Alternative hypothesis ('two-sided', '2s', 'greater', 'g', 'less', 'l').
        reference_group : Optional[str], optional
            Label name in `between` column to define reference group for the alternative hypothesis.
        correction : str or bool
            Correction for pingouin ttest. If not set, then it is determined automatically
        h0 : Optional[str], optional
            Custom null hypothesis text.
        h1 : Optional[str], optional
            Custom alternative hypothesis text.
            
        Returns:
        --------
            None                  
        """
        # Standardize alternative
        alternative = self._validate_alternative(alternative)
        # Update and validate config
        self._validate_config({
            'alpha': alpha,
            'alternative': alternative,
            'reference_group': reference_group,
            'h0': h0,
            'h1': h1
        })

        # Validate data and get samples
        self.dv = dv
        self.between = between
        self._validate_dv_and_between(test_name='ttest')
        # Check missings
        df, na_warnings = self._check_and_handle_na(dv, between)
        groups = df[between].unique()
        sample1 = df[df[between] == groups[0]][dv]
        sample2 = df[df[between] == groups[1]][dv]
        group1, group2 = groups[0], groups[1]

        text_output = ""
        if na_warnings:
            text_output += "\n".join(na_warnings) +'\n'

        text = f"""
        [bold]T-Test Report[/bold]\n
        - Dependent Variable: {dv}
        - Between Factor: {between} ({group1} vs {group2})
        - Alpha Level: {alpha}
        - Alternative: {alternative}\n"""
        text_output += textwrap.dedent(text)

        # Hypotheses formulation
        h0 = self.config.h0 or f"H0: The mean of {group1} equals the mean of {group2}"
        h1 = self.config.h1 or {
            'two-sided': f"H1: The mean of {group1} differs from the mean of {group2}",
            'greater': f"H1: The mean of {reference_group or group1} is greater than {group2 if reference_group else group2}",
            'less': f"H1: The mean of {reference_group or group1} is less than {group2 if reference_group else group2}"
        }[alternative]
        text = f"""
        [bold]🔹 Hypothesis Formulation[/bold]\n
        - {h0}
        - {h1}\n\n"""
        text_output += textwrap.dedent(text)

        # Descriptive statistics
        desc_stats = (
            df.groupby(between, observed=True)[dv]
            .agg(['count', 'mean', 'std', 'min', 'max'])
            .rename_axis('Group')
            .reset_index()
        )
        desc_stats[['mean', 'std', 'min', 'max']] = desc_stats[['mean', 'std', 'min', 'max']].round(2).astype(str)
        group_counts = df[between].value_counts()
        if any(desc_stats['count'] < 30):
            text = (
                "[bold]🔹 Descriptive Statistics[/bold]\n\n"
                "⚠️ Warning: Small sample size (<30) detected in one or more groups"
            )
            text_output += text
        else:
            text_output += "[bold]🔹 Descriptive Statistics[/bold]"
        rprint(text_output)
        text_output = ''
        display(desc_stats.style.hide(axis="index"))

        # Homogeneity of variance
        text = "[bold]🔹 Homogeneity of Variance (Levene's Test)[/bold]"
        rprint(text, highlight=False)
        levene_result = pg.homoscedasticity(
            data=df,
            dv=dv,
            group=between,
            method='levene',
            alpha=alpha
        )
        display(levene_result.round(3).astype(str))

        W, p_value_levene = levene_result.loc['levene', ['W', 'pval']]
        equal_var = p_value_levene >= alpha
        if correction is None:
            if equal_var:
                text = (
                    f"- Using Student's t-test - "
                    f"equal variances assumed (pval = {p_value_levene:.3f} ≥ {alpha})\n\n"
                )
            else:
                text = (
                    f"- Using Welch's t-test - "
                    f"unequal variances detected (pval = {p_value_levene:.3f} < {alpha})\n\n"
                )
        else:
            if correction == False:
                text = (
                    f"- Using Student's t-test - \n\n"
                )
            elif correction == True:
                text = (
                    f"- Using Welch's t-test - \n\n"
                )  
            elif correction == 'auto':
                text = ''
        text_output += text
        # Perform t-test
        text_output += "[bold]🔹 T-Test Results[/bold]"
        rprint(text_output)
        text_output = ''
        ttest_result = pg.ttest(
            x=sample1,
            y=sample2,
            alternative=alternative,
            correction=not equal_var if correction is None else correction,
            confidence=1 - alpha
        )
        display(ttest_result.round(3).astype(str))

        # Effect size
        cohen_d = ttest_result['cohen-d'].values[0]
        effect_size = "small" if abs(cohen_d) < 0.5 else "medium" if abs(cohen_d) < 0.8 else "large"
        text_output += f"\n- [bold]Effect Size:[/bold] Cohen's d = {cohen_d:.2f} ({effect_size} effect)\n"

        p_value = ttest_result['p-val'].values[0]
        if p_value < self.config.alpha:
            p_value_str = f"[bright_red]{p_value:.4f}[/bright_red]"
        else:
            p_value_str = f"[green]{p_value:.4f}[/green]"
        conclusion = (
            f"At alpha = {self.config.alpha}, we {'[bright_red]reject[/bright_red]' if p_value < self.config.alpha else '[green]fail to reject[/green]'} "
            f"the null hypothesis (p-val = {p_value_str})."
        )
        rprint(f"[bold]🔹 Conclusion:[/bold] {conclusion}", highlight=False)

    def mwu(
        self,
        dv: str,
        between: str,
        alpha: float = 0.05,
        alternative: str = 'two-sided',
        reference_group: Optional[str] = None,
        h0: Optional[str] = None,
        h1: Optional[str] = None
    ) -> None:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test) with comprehensive reporting.

        Parameters:
        -----------
        dv : str
            Name of the dependent variable column.
        between : str
            Name of the column containing group labels.
        alpha : float, optional
            Significance level (default: 0.05).
        alternative : str, optional
            Alternative hypothesis ('two-sided', '2s', 'greater', 'g', 'less', 'l').
        reference_group : Optional[str], optional
            Label name in `between` column to define reference group for the alternative hypothesis..
        h0 : Optional[str], optional
            Custom null hypothesis text.
        h1 : Optional[str], optional
            Custom alternative hypothesis text.
            
        Returns:
        --------
            None                  
        """
        # Standardize alternative
        alternative = self._validate_alternative(alternative)
        # Update and validate config
        self._validate_config({
            'alpha': alpha,
            'alternative': alternative,
            'reference_group': reference_group,
            'h0': h0,
            'h1': h1
        })

        # Validate data and get samples
        self.dv = dv
        self.between = between
        self._validate_dv_and_between(test_name='mannwhitney')
        # Check missings
        df, na_warnings = self._check_and_handle_na(dv, between)
        groups = df[between].unique()
        sample1 = df[df[between] == groups[0]][dv]
        sample2 = df[df[between] == groups[1]][dv]
        group1, group2 = groups[0], groups[1]

        text_output = ""
        if na_warnings:
            text_output += "\n".join(na_warnings) +'\n'

        # Prepare report
        text = f"""
        [bold]Mann-Whitney U Test Report[/bold]\n
        - Dependent Variable: {dv}
        - Between Factor: {between} ({group1} vs {group2})
        - Alpha Level: {alpha}
        - Alternative: {alternative}\n"""
        text = textwrap.dedent(text)
        text_output += text

        # Hypotheses formulation
        h0 = self.config.h0 or f"H0: The distribution of {group1} equals the distribution of {group2}"
        h1 = self.config.h1 or {
            'two-sided': f"H1: The distribution of {group1} differs from the distribution of {group2}",
            'greater': f"H1: The distribution of {reference_group or group1} is stochastically greater than {group2 if reference_group else group2}",
            'less': f"H1: The distribution of {reference_group or group1} is stochastically less than {group2 if reference_group else group2}"
        }[alternative]

        text = f"""
        [bold]🔹 Hypothesis Formulation[/bold]\n
        - {h0}
        - {h1}\n\n"""
        text = textwrap.dedent(text)
        text_output += text

        # Descriptive statistics
        desc_stats = (
            df.groupby(between, observed=True)[dv]
            .agg(['count', 'median', 'mean', 'std', 'min', 'max'])
            .rename_axis('Group')
            .reset_index()
        )
        desc_stats[['median', 'mean', 'std', 'min', 'max']] = desc_stats[['median', 'mean', 'std', 'min', 'max']].round(2).astype(str)
        if any(desc_stats['count'] < 30):
            text = (
                "[bold]🔹 Descriptive Statistics[/bold]\n\n"
                "⚠️ Warning: Small sample size (<30) detected in one or more groups"
            )
            text_output += text
        else:
            text_output += "[bold]🔹 Descriptive Statistics[/bold]"
        rprint(text_output)
        text_output = ''
        display(desc_stats.style.hide(axis="index"))

        # Perform Mann-Whitney test
        rprint("[bold]🔹 Mann-Whitney U Test Results[/bold]")

        result = pg.mwu(
            x=sample1,
            y=sample2,
            alternative=alternative,
        )
        display(result.round(3).astype(str))

        # Effect sizes interpretation
        rbc = result['RBC'].values[0]
        cles = result['CLES'].values[0]

        rbc_effect = "small" if abs(rbc) < 0.3 else "medium" if abs(rbc) < 0.5 else "large"
        cles_interpretation = "small" if abs(cles - 0.5) < 0.1 else "medium" if abs(cles - 0.5) < 0.2 else "large"

        text = f"""
        [bold]Effect Sizes:[/bold]
        - Rank-Biserial Correlation (RBC) = {rbc:.2f} ({rbc_effect} effect)
        - Common Language Effect Size (CLES) = {cles:.2f} ({cles_interpretation} effect)
        """
        text_output += textwrap.dedent(text)

        # Conclusion
        p_value = result['p-val'].values[0]
        if p_value < self.config.alpha:
            p_value_str = f"[bright_red]{p_value:.4f}[/bright_red]"
        else:
            p_value_str = f"[green]{p_value:.4f}[/green]"

        conclusion = (
            f"At alpha = {self.config.alpha}, we {'[bright_red]reject[/bright_red]' if p_value < self.config.alpha else '[green]fail to reject[/green]'} "
            f"the null hypothesis (p-val = {p_value_str})."
        )
        text_output += f"\n[bold]🔹 Conclusion:[/bold] {conclusion}"
        rprint(text_output)

    def anova(
        self,
        dv: str,
        between: Union[str, List[str]],
        alpha: float = 0.05,
        effsize: str = 'np2',
        h0: Optional[str] = None,
        h1: Optional[str] = None
    ) -> None:
        """
        Perform ANOVA analysis with automatic variance checking and appropriate post-hoc tests.

        Handles both one-way and factorial designs:

        - Checks homogeneity of variance (Levene's test)
        - Uses Welch's ANOVA when variances differ
        - Calculates achieved power
        - Performs appropriate post-hoc tests (Tukey/Games-Howell)

        Parameters:
        -----------
        dv : str
            Name of dependent variable column
        between : str or list of str
            Name of between-factor column(s)
        alpha : float, optional
            Significance level (default 0.05)
        effsize : str, optional
            Effect size type ('np2' for partial eta-squared or 'n2' for eta-squared)
        h0 : str, optional
            Custom null hypothesis text
        h1 : str, optional
            Custom alternative hypothesis text
            
        Returns:
        --------
            None                  
        """
        # Validate data and handle missing values
        self.dv = dv
        self.between = between
        self._validate_dv_and_between(test_name='anova')
        if effsize not in ['np2', 'n2']:
            raise ValueError("effsize must be either 'np2' or 'n2'")
        df, na_warnings = self._check_and_handle_na(dv, between)
        text_output = ""
        if na_warnings:
            text_output += "\n".join(na_warnings) +'\n'

        # Prepare between factors (convert string to list if single factor)
        between_factors = [between] if isinstance(between, str) else between
        is_oneway = len(between_factors) == 1

        # Prepare report header
        design_type = "One-way" if is_oneway else f"{len(between_factors)}-way Factorial"
        text = f"""
        [bold]{design_type} ANOVA Report[/bold]\n
        - Dependent Variable: {dv}
        - Between Factors: {', '.join(between_factors)}
        - Alpha Level: {alpha}\n"""
        text_output += textwrap.dedent(text)
        # text = textwrap.dedent(text)
        # rprint(text)

        # Hypothesis formulation
        if is_oneway:
            factor = between[0] if isinstance(between, list) else between
            groups = df[factor].unique()
            groups_str = ", ".join(groups[:3]) + (f", ... ({len(groups)} total)" if len(groups) > 3 else "")

            h0 = h0 or f"H0: All group means are equal ({groups_str})"
            h1 = h1 or f"H1: At least one group mean differs among {groups_str}"
        else:
            # For multi ANOVA
            factors_str = ", ".join(between)
            h0 = h0 or f"H0: No main effects or interactions for factors: {factors_str}"
            h1 = h1 or f"H1: Significant main effects or interactions for factors: {factors_str}"
        text = f"""
        [bold]🔹 Hypothesis Formulation[/bold]\n
        - {h0}
        - {h1}\n
        """
        text_output += textwrap.dedent(text)
        # text = textwrap.dedent(text)
        # rprint(text)
        # Descriptive statistics
        if is_oneway:
            text_output += "[bold]🔹 Descriptive Statistics[/bold]"
            # For one-way ANOVA or single factor
            single_factor = between[0] if isinstance(between, list) else between
            desc_stats = (
                df.groupby(single_factor, observed=True)[dv]
                .agg(['count', 'mean', 'std', 'min', 'max', 'median'])
                .rename_axis('Group')
                .reset_index()
            )
            desc_stats[['mean', 'std', 'min', 'max', 'median']] = desc_stats[['mean', 'std', 'min', 'max', 'median']].round(2)

            # Add warning for small groups
            if any(desc_stats['count'] < 30):
                small_groups = desc_stats[desc_stats['count'] < 30]['Group'].tolist()
                text_output += f"\n\n⚠️ [yellow]Warning: Small sample size (<30) in groups: {', '.join(map(str, small_groups))}[/yellow]"
            rprint(text_output)
            text_output = ''
            display(desc_stats.round(3).astype(str).style.hide(axis="index"))

        # Check homogeneity of variance
        text_output += "[bold]🔹 Homogeneity of Variance Check[/bold]"
        if is_oneway:
            levene_result = pg.homoscedasticity(
                data=df,
                dv=dv,
                group=between_factors[0],
                method='levene',
                alpha=alpha
            )
            rprint(text_output)
            text_output = ''
            display(levene_result.round(3).astype(str))
            levene_pval = levene_result['pval'].iloc[0]
            equal_var = levene_pval >= alpha
            if equal_var:
                text_output += (
                    f"- Using standard ANOVA - "
                    f"Levene's test showed equal variances (pval = {levene_pval:.3f} ≥ {alpha})\n\n"
                )
            else:
                text_output += (
                    f"- Using Welch's ANOVA - "
                    f"Levene's test showed unequal variances (pval = {levene_pval:.3f} < {alpha})\n\n"
                )
        else:
            equal_var = True
            text_output += "\n\n- In a multifactorial ANOVA, homoscedasticity (equal variances) is assumed without explicit testing.\n\n"
        # Perform appropriate ANOVA
        text_output += f"[bold]🔹 {'Welch' if not equal_var else ''} ANOVA Results[/bold]"
        if equal_var:
            aov = pg.anova(
                data=df,
                dv=dv,
                between=between_factors,
                # detailed=True,
                effsize=effsize
            )
        else:
            if not is_oneway:
                raise ValueError("Welch's ANOVA only available for one-way design. Consider data transformation.")
            aov = pg.welch_anova(
                data=df,
                dv=dv,
                between=between_factors[0]
            )
        rprint(text_output)
        text_output = ''
        aov_styled = aov.round(3).astype(str).style.hide(axis='index')
        display(aov_styled)

        # Calculate achieved power
        if is_oneway and equal_var:  # Power calculation for one-way normal ANOVA
            text_output += "[bold]🔹 Power Analysis[/bold]\n\n"
            k = df[between_factors[0]].nunique() # Number of groups
            n = df.shape[0] / k # Number of observations per group
            eta_sq = aov.loc[0, 'np2'] if 'np2' in aov.columns else aov.loc[0, 'n2']
            achieved_power = pg.power_anova(
                eta_squared=eta_sq,
                k=k,
                n=n,
                alpha=alpha
            )
            text_output += f"- Achieved power: [bold]{achieved_power:.4f}[/bold]\n\n"
        # else:
        #     rprint("⚠️ Power calculation not available for this design\n")

        # Post-hoc tests if significant
        sig_effects = aov[aov['p-unc'] < alpha]
        if not sig_effects.empty:
            text_output += f"[bold]🔹 Post-hoc Analysis[/bold]"
            rprint(text_output)
            text_output = ''
            if is_oneway:
                # One-way post-hoc
                if equal_var:
                    ph = pg.pairwise_tukey(
                        data=df,
                        dv=dv,
                        between=between_factors[0]
                    )
                    caption ="Tukey HSD for equal variances"
                else:
                    ph = pg.pairwise_gameshowell(
                        data=df,
                        dv=dv,
                        between=between_factors[0]
                    )
                    caption= "Games-Howell for unequal variances"
                ph = ph.round(3).astype(str)
                display(style_dataframe(ph, hide_columns=False, caption=caption, caption_font_size=14))

                text_output = f"[bold]🔹 Conclusion:[/bold] [bright_red]Significant effects found (p-unc = {aov['p-unc'].iloc[0]:.3f})[/bright_red]\n"
            else:
                # Factorial design post-hoc
                for factor in between_factors:
                    if factor in sig_effects['Source'].tolist():
                        ph = pg.pairwise_tukey(
                            data=df,
                            dv=dv,
                            between=factor
                        )
                        caption = f"Tukey HSD for {factor}"
                        ph = ph.round(3).astype(str)
                        display(style_dataframe(ph, hide_columns=False, caption=caption, caption_font_size=14))
                text_output = '[bold]🔹 Conclusion:[/bold] [bright_red]Significant effects found[/bright_red]\n\n'
                for _, row in sig_effects.iterrows():
                    effect = row['Source']
                    p = row['p-unc']
                    text_output += f"- {effect}: p-unc = {p:.3f}\n"
            if equal_var:
                p_name = 'p-tukey'
                sig_comparisons = ph[ph['p-tukey'].astype(float) < alpha]
            else:
                p_name = 'pval'
                sig_comparisons = ph[ph['pval'].astype(float)  < alpha]
            if not sig_comparisons.empty:
                text_output += "\n[bold]Significant pairwise comparisons:[/bold]\n"
                for _, row in sig_comparisons.iterrows():
                    text_output += f"- {row['A']} vs {row['B']}: {p_name} = {float(row[p_name]):.3f}\n"
            rprint(text_output.strip())
        else:
            if is_oneway:
                rprint(f"[bold]🔹 Conclusion:[/bold] [green]No significant effects found[/green]\n")
            else:
                rprint("[bold]🔹 Conclusion:[/bold] [green]No significant main effects or interactions found[/green]")

    def kruskal(
        self,
        dv: str,
        between: str,
        alpha: float = 0.05,
        h0: Optional[str] = None,
        h1: Optional[str] = None
    ) -> None:
        """
        Perform Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA).

        Parameters:
        -----------
        dv : str
            Name of dependent variable column
        between : str
            Name of grouping variable column
        alpha : float, optional
            Significance level (default 0.05)
        h0 : str, optional
            Custom null hypothesis text
        h1 : str, optional
            Custom alternative hypothesis text
            
        Returns:
        --------
            None                  
        """

        # Validate data and handle missing values
        self.dv = dv
        self.between = between
        self._validate_dv_and_between(test_name='kruskal')
        df, na_warnings = self._check_and_handle_na(dv, between)
        text_output = ""
        if na_warnings:
            text_output += "\n".join(na_warnings) +'\n'

        # Get group names for reporting
        groups = df[between].unique()
        groups_str = ", ".join(groups[:3]) + (f", ... ({len(groups)} total)" if len(groups) > 3 else "")

        # Prepare report header
        text = f"""
        [bold]Kruskal-Wallis Test Report[/bold]\n
        - Dependent Variable: {dv}
        - Grouping Variable: {between}
        - Groups: {groups_str}
        - Alpha Level: {alpha}
        """
        text_output += textwrap.dedent(text)

        # Hypothesis formulation
        h0 = h0 or f"H0: The distributions of all groups are equal ({groups_str})"
        h1 = h1 or f"H1: At least one group distribution differs among {groups_str}"

        text = f"""
        [bold]🔹 Hypothesis Formulation[/bold]\n
        - {h0}
        - {h1}\n
        """
        text_output += textwrap.dedent(text)
        # Descriptive statistics
        text_output += "[bold]🔹 Descriptive Statistics[/bold]"
        desc_stats = (
            df.groupby(between, observed=True)[dv]
            .agg(['count', 'median', 'mean', 'std', 'min', 'max'])
            .rename_axis('Group')
            .reset_index()
        )
        desc_stats[['median', 'mean', 'std', 'min', 'max']] = desc_stats[['median', 'mean', 'std', 'min', 'max']].round(2)

        if any(desc_stats['count'] < 30):
            small_groups = desc_stats[desc_stats['count'] < 30]['Group'].tolist()
            text_output += f"\n\n⚠️ [yellow]Warning: Small sample size (<30) in groups: {', '.join(map(str, small_groups))}[/yellow]"
        rprint(text_output)
        text_output = ''
        display(desc_stats.round(3).astype(str).style.hide(axis="index"))

        # Perform Kruskal-Wallis test
        rprint("\n[bold]🔹 Kruskal-Wallis Test Results[/bold]")
        kw_result = pg.kruskal(
            data=df,
            dv=dv,
            between=between
        )
        display(kw_result.round(3).astype(str).style.hide(axis="index"))
        kw_H = kw_result['H'].iloc[0]
        kw_punc = kw_result['p-unc'].iloc[0]
        # Post-hoc tests if significant
        if kw_punc < alpha:
            rprint("\n[bold]🔹 Post-hoc Analysis (Dunn's Test)[/bold]")
            dunn_results = posthoc_dunn(
                df,
                val_col=dv,
                group_col=between,
                p_adjust='holm'
            )
            # ph = dunn_results.stack().reset_index()
            upper_triangle = np.triu(np.ones(dunn_results.shape), k=1).astype(bool)
            ph = dunn_results.where(upper_triangle).stack().reset_index()
            ph.columns = ['A', 'B', 'p-corr']
            # ph = ph.drop_duplicates(subset=['A', 'B'])
            display(ph.round(3).astype(str).style.hide(axis="index"))

            text = (f"[bold]🔹 Conclusion:[/bold] [bright_red]Significant difference found (p-unc = {kw_punc:.3f})[/bright_red]")
            # Print significant comparisons
            sig_comparisons = ph[ph['p-corr'] < alpha]
            if not sig_comparisons.empty:
                text += "\n\n[bold]Significant pairwise comparisons:[/bold]\n"
                for _, row in sig_comparisons.iterrows():
                    text +=f"- {row['A']} vs {row['B']}: pval = {row['p-corr']:.3f}\n"
            rprint(text.strip())
        else:
            rprint(f"\n[bold]🔹 Conclusion:[/bold] [green]No significant difference found[/green]")

    def chi2_independence(self, x, y, test='pearson', alpha=0.05) -> None:
        """
        Categorical variables association report using pingouin.chi2_independence

        Parameters
        ----------
        x, y : str
            Names of categorical variables in DataFrame
        test : str
            Type of test to perform. Options: 'pearson', 'cressie-read', 'log-likelihood',
            'freeman-tukey', 'mod-log-likelihood', 'neyman', 'all'
        alpha : float
            Significance level

        Returns
        -------
            None
        """
        df = self.df
        # 1. Input validation
        if alpha < 0 or alpha > 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha:.3f}")

        for col in [x, y]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
        if test not in ['pearson', 'cressie-read', 'log-likelihood', 'freeman-tukey', 'mod-log-likelihood', 'neyman', 'all']:
            raise ValueError(f"Invalid test type: {test}. Choose from 'pearson', 'cressie-read', 'log-likelihood', 'freeman-tukey', 'mod-log-likelihood', 'neyman', or 'all'.")
        # 2. Missing data check
        na_info = []
        for col in [x, y]:
            na_count = df[col].isna().sum()
            if na_count > 0:
                na_pct = na_count / len(df) * 100
                na_info.append(f"⚠️ Found {na_count} ({na_pct:.1f}%) missing values in '{col}'")
        output_text = ''
        if na_info:
            output_text += "\n".join(na_info)
            df = df.dropna(subset=[x, y])
            output_text += f"\nRemoved {na_count} rows with missing values\n\n"
        # 1. Contingency table
        cont_table = pd.crosstab(df[x], df[y])
        n = cont_table.sum().sum()

        output_text += f"[bold]Categorical Association Report[/bold]\n\n"
        output_text += f"- {x} vs {y}\n"
        output_text += f"- Total observations: {n}\n\n"

        groups_x = df[x].unique()
        groups_y = df[y].unique()
        groups_str_x = ", ".join(map(str, groups_x[:3])) + (f", ... ({len(groups_x)} total)" if len(groups_x) > 3 else "")
        groups_str_y = ", ".join(map(str, groups_y[:3])) + (f", ... ({len(groups_y)} total)" if len(groups_y) > 3 else "")

        h0 = f"H0: {x} and {y} are independent ({groups_str_x} vs {groups_str_y})"
        h1 = f"H1: {x} and {y} are associated ({groups_str_x} vs {groups_str_y})"
        text = f"""
        [bold]🔹 Hypothesis Formulation[/bold]\n
        - {h0}
        - {h1}\n\n"""
        output_text += textwrap.dedent(text)

        output_text += "[bold]🔹 Contingency Table[/bold]"
        rprint(output_text)
        output_text = ''
        display(cont_table)

        # 2. Statistical tests
        rprint("[bold]🔹 Statistical Tests[/bold]")
        _, _, chi2 = pg.chi2_independence(df, x, y)
        if test != 'all':
            chi2 = chi2[chi2['test'] == test]
        if chi2.empty:
            raise ValueError('Result is empty')
        # Add interpretations
        def interpret_cramer(v):
            if pd.isna(v): return ""
            if v < 0.1: return "Very weak"
            elif v < 0.3: return "Weak"
            elif v < 0.5: return "Moderate"
            else: return "Strong"


        chi2['cramer strength'] = chi2['cramer'].apply(interpret_cramer)
        display(chi2.round(3).astype(str).style.hide(axis='index'))

        # rprint(f"\n{conclusion}")
        best_test = chi2.loc[chi2['pval'].idxmin()]
        p_value = best_test['pval']
        p_value_str = f"[bright_red]{p_value:.3f}[/bright_red]" if p_value < alpha else f"[green]{p_value:.3f}[/green]"
        # Build conclusion based on test type
        if test != 'all':
            conclusion = (
                f"At alpha = {alpha}, we {'[bright_red]reject[/bright_red]' if p_value < alpha else '[green]fail to reject[/green]'} "
                f"the null hypothesis of independence ({test} test, p = {p_value_str})."
            )
        else:
            conclusion = (
                f"At alpha = {alpha}, based on the most significant test ({best_test['test']}), we "
                f"{'[bright_red]reject[/bright_red]' if p_value < alpha else '[green]fail to reject[/green]'} "
                f"the null hypothesis of independence (p = {p_value_str})."
            )

        # Add effect size interpretation if significant
        if p_value > alpha:
            strength = interpret_cramer(best_test.get('cramer', float('nan')))
            if not pd.isna(best_test.get('cramer')):
                conclusion += (
                    f"\nFound significant association between {x} and {y} "
                    f"(Cramer's V = {best_test['cramer']:.2f}, {strength} effect)."
                )
            else:
                conclusion += f"\nFound significant association between {x} and {y}."
        else:
            conclusion += f"\nNo significant association found between {x} and {y}."

        rprint(f"[bold]🔹 Conclusion:[/bold] {conclusion}")

    def bootstrap(
        self,
        dv: str,
        between: Optional[str] = None,
        reference_group: Optional[str] = None,
        statistic: Union[str, Callable] = "mean_diff",
        n_resamples: int = 9999,
        confidence_level: float = 0.95,
        method: str = 'BCa',
        n_jobs: int = -1,
        batch: Optional[int] = None,
        rng: Optional[Union[int, np.random.Generator]] = None,
        parallel: bool = False,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
        plot: bool = False,
        return_results: bool = False,
        **kwargs
    ) -> BootstrapResult:
        """
        Perform bootstrap analysis on DataFrame columns with parallel and non-parallel modes.

        Parameters
        ----------
        dv : str
            Name of the column containing the dependent variable (metric).

        between : str, optional
            Name of the column containing the grouping variable. If None, performs one-sample bootstrap.

        reference_group : str, optional
            When between is specified, defines which group should be considered as the first group 
            in the comparison (e.g., 'control' group). The statistic will be calculated as:
            statistic(first_group) - statistic(second_group).
            Required when between is specified.

        statistic : str or callable
            Statistic to compute. Can be:

            - String name of built-in statistic:

                - For one sample: 'mean', 'median'
                - For two samples: 'mean_diff', 'median_diff'

            - Callable function that computes the statistic. The function must:

                - Accept axis parameter for vectorized operations
                - Return scalar value when axis=None
                - Handle numpy array inputs

        n_resamples : int, optional
            Number of bootstrap resamples to perform (default: 9999).

        confidence_level : float, optional
            Confidence level for intervals (0 < level < 1).

        method : {'percentile', 'basic', 'BCa'}, optional
            Method for computing confidence interval:

            - 'percentile': Simple percentile method
            - 'basic': Basic (reverse percentile) method
            - 'BCa': Bias-corrected and accelerated (recommended)

        n_jobs : int, optional
            Number of parallel jobs to run:

            - -1: All available cores
            - >0: Specified number of jobs

        batch : int, optional
            The number of resamples to process in each vectorized call to statistic.

        rng : int or numpy.random.Generator, optional
            Random number generator or seed:

            - None: New random state
            - int: Seed for new random generator
            - Generator: Use existing generator

        parallel : bool, optional
            Whether to use parallel processing. If True:

            - Uses joblib for parallel computation
            - Ignores method parameter and calculates intervals manually
            - Generally faster for large n_resamples

        alternative : {'two-sided', 'less', 'greater'}, optional
            Type of confidence interval:

            - 'two-sided': Two-sided interval (default)
            - 'less': One-sided interval (upper bound only)
            - 'greater': One-sided interval (lower bound only)

        plot : bool, optional
            Whether to show the plot of the bootstrap distribution (default: False)

        return_results : bool, optional
            Whether to return result.

        kwargs
            Additional arguments passed to the statistic function.

        Returns
        -------
        BootstrapResult
            Object with attributes:

            - table: Pandas DataFrame with results (method, statistic, n_resamples, standard_error, ci_low, ci_high)
            - bootstrap_distribution: Array of bootstrap resample statistics

        Notes
        -----
        Examples of statistic functions:
            1) Lambda function for one-sample mean:

                lambda x, axis=-1: np.mean(x, axis=axis)

            2) Defined function for two-sample ratio:

                def ratio(x, y, axis=-1):
                    return np.mean(x, axis=axis)/np.mean(y, axis=axis)

            3) Function with additional parameters:

                def trimmed_mean(x, axis=-1, trim=0.1):
                    return scipy.stats.trim_mean(x, trim, axis=axis)

            4) Vectorized two-sample statistic:

                def mean_diff(x, y, axis=-1):
                    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

        Important Notes:
            - For BCa method, at least 1000 resamples are recommended for stable results.
            - Parallel processing is generally faster but uses more memory.
            - When parallel=True, confidence intervals are always calculated manually.
            - If random_state is provided, it takes precedence for backward compatibility.
        """
        df = self.df
        # Validate inputs
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        if parallel and (not isinstance(n_jobs, int) or n_jobs is None or (n_jobs < 0 and n_jobs != -1)):
            raise ValueError('n_jobs must be -1 or positive integer')

        if n_resamples < 1:
            raise ValueError("n_resamples must be positive")

        if alternative not in ['two-sided', 'less', 'greater']:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

        if method == 'BCa' and n_resamples < 1000 and not parallel:
            warnings.warn("BCa method recommended to use at least 1000 resamples for stable results")
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if not isinstance(dv, str) or dv not in df.columns:
            raise ValueError(f"dv must be a single column name present in DataFrame")
        
        if between is not None:
            if not isinstance(between, str) or between not in df.columns:
                raise ValueError(f"between must be a column name present in DataFrame")
            
            groups = self.df[between].unique()
            if len(groups) != 2:
                raise ValueError(f"between must have exactly 2 unique groups, found {len(groups)}")
            
            if reference_group is None:
                raise ValueError("reference_group parameter required when between is specified - "
                            "define which group should be first in comparison (e.g., reference_group='control')")
            elif reference_group not in groups:
                raise ValueError(f"reference_group group '{reference_group}' not found in {between} column")
            
        if not isinstance(statistic, (str, Callable)):
            raise TypeError("statistic must be either string or callable. "
                        "If callable, it should accept axis parameter for vectorized operations. "
                        "Example: def my_stat(x, axis=-1): return np.mean(x, axis=axis)")
        df, na_warnings = self._check_and_handle_na(dv, between)
        if na_warnings:
            text_output = "\n".join(na_warnings)
            rprint(text_output)
        # Prepare data
        if between is None:
            # One-sample 
            data = (df[dv].values,)
            default_stat = 'mean'
        else:
            # Two-sample 
            group1 = df[df[between] == reference_group][dv].values
            group2 = df[df[between] != reference_group][dv].values
            data = (group1, group2)
            default_stat = 'mean_diff'
        # Define statistic function
        if isinstance(statistic, str):
            if between is None:
                if statistic == 'mean':
                    def statistic_func(x, axis=-1): return np.mean(x, axis=axis)
                elif statistic == 'median':
                    def statistic_func(x, axis=-1): return np.median(x, axis=axis)
                else:
                    raise ValueError(f"Unknown single-sample statistic: {statistic}")
            else:
                if statistic == 'mean_diff':
                    def statistic_func(x, y, axis=-1): return np.mean(x, axis=axis) - np.mean(y, axis=axis)
                elif statistic == 'median_diff':
                    def statistic_func(x, y, axis=-1): return np.median(x, axis=axis) - np.median(y, axis=axis)
                else:
                    raise ValueError(f"Unknown two-sample statistic: {statistic}")
        else:
            statistic_func = statistic

        # Calculate original statistic
        if between is None:
            original_stat = statistic_func(data[0], **kwargs)
        else:
            original_stat = statistic_func(data[0], data[1], **kwargs)

        # Parallel bootstrap implementation
        if parallel:
            if n_jobs == -1:
                n_jobs = cpu_count()

            n_resamples_per_job = n_resamples // n_jobs
            remainder = n_resamples % n_jobs
            actual_resamples = n_resamples_per_job * n_jobs + remainder

            if method == 'BCa' and actual_resamples < 1000:
                warnings.warn(f"BCa method using {actual_resamples} resamples (recommended >= 1000)")

            # Handle both new (rng) and legacy (random_state) parameters
            if isinstance(rng, np.random.Generator):
                seeds = rng.integers(0, 2**32-1, size=n_jobs)
            else:
                rng = np.random.default_rng(rng)
                seeds = rng.integers(0, 2**32-1, size=n_jobs)

            def parallel_bootstrap_task(i):
                # Adjust n_resamples for the last job to include the remainder
                if i == n_jobs - 1:
                    n_resamples_for_task = n_resamples_per_job + remainder
                else:
                    n_resamples_for_task = n_resamples_per_job
                res = scipy_bootstrap(
                    data,
                    statistic_func,
                    vectorized=True,
                    n_resamples=n_resamples_for_task,
                    batch=batch,
                    method='percentile',  # We'll compute BCa/basic ourselves
                    random_state=seeds[i]
                )
                return res.bootstrap_distribution
            with tqdm_joblib(tqdm(desc="Bootstrapping", total=n_jobs)) as progress_bar:
                bootstrap_dists = Parallel(n_jobs=n_jobs)(delayed(parallel_bootstrap_task)(i) for i in range(n_jobs))
            bootstrap_dist = np.concatenate(bootstrap_dists)
            # Adjust alpha based on alternative hypothesis
            if alternative == 'two-sided':
                alpha = (1 - confidence_level) / 2
                ci_low, ci_high = None, None  # Initialize both
            elif alternative == 'less':
                alpha = 1 - confidence_level
                ci_low, ci_high = -np.inf, None  # Only upper bound
            elif alternative == 'greater':
                alpha = 1 - confidence_level
                ci_low, ci_high = None, np.inf  # Only lower bound

            # Compute confidence interval based on method (if not parallel) or manually (if parallel)
            if method == 'percentile':
                # Always calculate manually in parallel mode
                if ci_low is None:
                    ci_low = np.percentile(bootstrap_dist, 100 * alpha)
                if ci_high is None:
                    ci_high = np.percentile(bootstrap_dist, 100 * (1 - alpha))
            elif method == 'basic':
                # Basic (reverse percentile) method
                stat = original_stat if between is None else statistic_func(*data)
                if ci_low is None:
                    ci_low = 2 * stat - np.percentile(bootstrap_dist, 100 * (1 - alpha))
                if ci_high is None:
                    ci_high = 2 * stat - np.percentile(bootstrap_dist, 100 * alpha)
            elif method == 'BCa':
                # Bias-corrected and accelerated method
                if len(data[0]) < 2:
                    raise ValueError("Need at least 2 observations for BCa")

                stat = original_stat if between is None else statistic_func(*data)

                # Bias correction (z0)
                prop_less = np.mean(bootstrap_dist < stat)
                z0 = norm.ppf(np.clip(prop_less, 1e-12, 1-1e-12))  # Clip to avoid infinities

                # Acceleration factor (a) using jackknife
                if between is None:
                    # Single sample case
                    n = len(data[0])
                    jack_stats = np.array([
                        statistic_func(np.delete(data[0], i)) for i in range(n)
                    ])
                else:
                    # Two sample case - more accurate approach matching scipy
                    n1 = len(data[0])
                    n2 = len(data[1])
                    jack_stats = []

                    # Leave-one-out for first sample
                    for i in range(n1):
                        jack_stats.append(statistic_func(np.delete(data[0], i), data[1]))

                    # Leave-one-out for second sample
                    for i in range(n2):
                        jack_stats.append(statistic_func(data[0], np.delete(data[1], i)))

                    jack_stats = np.array(jack_stats)

                # Calculate acceleration factor
                jack_mean = np.mean(jack_stats)
                u = jack_mean - jack_stats  # Leave-one-out deviations
                a = np.sum(u**3) / (6 * np.sum(u**2)**1.5)  # Skewness measure

                # Adjusted percentiles function with safeguards
                def get_bca_quantile(alpha):
                    z_alpha = norm.ppf(np.clip(alpha, 1e-12, 1-1e-12))
                    numerator = z0 + z_alpha
                    denominator = 1 - a * numerator

                    # Fallback to percentile if denominator issues arise
                    if np.abs(denominator) < 1e-8 or not np.isfinite(denominator):
                        warnings.warn("BCa adjustment unstable, falling back to percentile")
                        return alpha

                    adjusted_alpha = norm.cdf(z0 + numerator / denominator)
                    return np.clip(adjusted_alpha, 0, 1)  # Ensure valid probability

                # Calculate adjusted percentiles based on alternative
                if ci_low is None:
                    adjusted_low = get_bca_quantile(alpha)
                    ci_low = np.percentile(bootstrap_dist, 100 * adjusted_low)
                if ci_high is None:
                    adjusted_high = get_bca_quantile(1 - alpha)
                    ci_high = np.percentile(bootstrap_dist, 100 * adjusted_high)
            else:
                raise ValueError(f"Unknown method: {method}")
            # Standard error (std of bootstrap distribution)
            std_error = f'{np.std(bootstrap_dist, ddof=1):.2f}'
        else:
            if method == 'BCa' and n_resamples < 1000:
                warnings.warn(f"BCa method using {n_resamples} resamples (recommended >= 1000)")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = scipy_bootstrap(
                    data,
                    statistic_func,
                    vectorized=True,
                    n_resamples=n_resamples,
                    method=method,
                    random_state=rng,
                    batch=batch
                )
                bootstrap_dist = res.bootstrap_distribution
            actual_resamples = len(bootstrap_dist)

            ci_low = res.confidence_interval.low
            ci_high = res.confidence_interval.high
            std_error = f'{res.standard_error:.2f}'
        # Format CI based on alternative
        if alternative == 'two-sided':
            ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]"
        elif alternative == 'less':
            ci_str = f"[-inf, {ci_high:.2f}]"
        else:  # greater
            ci_str = f"[{ci_low:.2f}, inf]"

        # Create result table
        result_table = pd.DataFrame({
            'method': [method],
            'statistic': [f'{original_stat:.2f}'],
            'n_resamples': [actual_resamples],
            'standard_error': [std_error],
            'ci': [ci_str],
            'alternative': [alternative]
        })
        if plot:
            fig = self._plot_bootstrap_distribution(
                bootstrap_dist,
                ci_low if ci_low is not None else -np.inf,
                ci_high if ci_high is not None else np.inf,
            )
        else:
            fig = None
        if not return_results:
            display(result_table.style.hide(axis='index'))
            if plot:
                CustomFigure(fig).show()
        else:
            return BootstrapResult(table=result_table, bootstrap_distribution=bootstrap_dist, plot=fig)

    def _plot_bootstrap_distribution(self, bootstrap_dist, ci_low, ci_high, title="Bootstrap Distribution") -> go.Figure:
        """Plot interactive histogram of bootstrap distribution with confidence intervals."""
        # Human-readable number formatting
        def human_readable(x):
            for unit in ['', 'k', 'M', 'B']:
                if abs(x) < 1000:
                    return f"{x:.2f}{unit}"
                x /= 1000
            return f"{x:.2f}T"
        # Create histogram data
        try:
            hist, bin_edges = np.histogram(bootstrap_dist, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        except Exception as e:
            print(f"An error in creating histogram: {e}")
            return go.Figure()
        # Create figure
        fig = go.Figure()

        # Add histogram bars with conditional coloring
        inside_ci = (bin_centers >= ci_low) & (bin_centers <= ci_high)
        colors = np.where(inside_ci, '#4C78A8', '#FF9E4A')

        fig.add_trace(go.Bar(
            x=bin_centers,
            y=hist,
            marker_color=colors,
            hovertemplate='Density = %{y:.2f}<br>Statistic Value = %{x:.2f}',
            showlegend=False,
        ))
        # Add confidence interval lines
        if ci_low != np.inf:
            fig.add_vline(
                x=ci_low,
                line_width=2,
                line_dash="dash",
                line_color="#4C78A8",
                annotation_text=f"CI Lower: {human_readable(ci_low)}",
                annotation_font_color = "#4C78A8",
                annotation_position="top",
            )
        if ci_high != np.inf:
            fig.add_vline(
                x=ci_high,
                line_width=2,
                line_dash="dash",
                line_color="#4C78A8",
                annotation_text=f"CI Upper: {human_readable(ci_high)}",
                annotation_font_color = "#4C78A8",
                annotation_position="top",
            )

        # Update layout
        fig.update_layout(
            title_text=title,
            xaxis_title="Statistic Value",
            yaxis_title="Density",
            width=600,
            height=400,
            bargap=0,
            margin=dict(l=50, r=50, b=50, t=50),
        )

        return fig

    def ols(
        self,
        formula: str,
        cov_type: Literal['nonrobust', 'HC0', 'HC1', 'HC2', 'HC3'] = "HC3",
        show_summary: bool = True,
        show_plots: bool = False,
        show_vif: bool = True,
        missing: str = 'drop',
        p_adjust: Optional[str] = None,
        return_results: bool = False,
        **kwargs: Any
    ):
        """
        Fit Ordinary Least Squares (OLS) regression.

        Best for:

        - Continuous normally distributed outcomes
        - Homoscedastic data (constant variance)
        - Linear relationships
        - When you need simple interpretable coefficients

        Parameters:
        -----------
        formula : str
            Patsy formula (e.g., 'y ~ x1 + x2 + C(category)')
        cov_type : str, optional
            Covariance matrix type: 'nonrobust', 'HC0'-'HC3' (default: 'HC3')
        show_summary : bool
            Whether to print summary (default: True)
        show_diagnostic_plots : bool
            Whether to show diagnostic plots (default: False)
        show_vif : bool
            Whether to show VIF for multicollinearity (default: True)
        show_plots : bool
            Whether to show diagnostic plots (default: False)
        missing : str
            How to handle missing values: 'drop', 'none', 'raise' (default: 'drop')
        p_adjust : str
            Method used for testing and adjustment of p-values. Can be either the full name or initial letters. Available methods are:

                - 'bonf': one-step Bonferroni correction
                - 'sidak': one-step Sidak correction
                - 'holm': step-down method using Bonferroni adjustments
                - 'fdr_bh': Benjamini/Hochberg FDR correction
                - 'fdr_by': Benjamini/Yekutieli FDR correction
                - None: pass-through option (no correction applied)

        return_results: bool
            Whether to return results.
        kwargs : Any
            Additional arguments for sm.OLS

        Returns:
        --------
        RegressionResultsWrapper
            OLS regression results
        """
        if not isinstance(formula, str):
            raise ValueError("formula must be a string")
        y, X = patsy.dmatrices(formula, self.df, return_type='dataframe')

        if show_vif:
            self._check_multicollinearity(X)

        model = sm.OLS(y, X, missing=missing, **kwargs)

        fit_kwargs = {'cov_type': cov_type, 'use_t': True}

        results = model.fit(**fit_kwargs)

        if show_summary:
            display(results.summary())

        if show_plots:
            self._show_minimal_diagnostic_plots(results)

        if return_results:
            return results

    def rlm(
        self,
        formula: str,
        show_summary: bool = True,
        show_plots: bool = False,
        show_vif: bool = True,
        missing: str = 'drop',
        p_adjust: Optional[str] = None,
        return_results: bool = False,
        M = None,
        **kwargs: Any
    ):
        """
        Fit Robust Linear Model (RLM) using M-estimators.

        Best for:

        - Data with outliers
        - When OLS assumptions violated
        - Heavy-tailed error distributions

        Parameters:
        -----------
        formula : str
            Patsy formula (e.g., 'y ~ x1 + x2')
        show_summary : bool
            Whether to print summary (default: True)
        show_plots : bool
            Whether to show diagnostic plots (default: False)
        show_vif : bool
            Whether to show VIF (default: True)
        missing : str
            How to handle missing values (default: 'drop')
        M : RobustNorm, optional
            Robust norm function from statsmodels.robust.norms

            Default is HuberT(). Alternatives:

            - HuberT(t=1.345)
            - RamsayE(a=0.3)
            - AndrewWave(a=1.339)
            - TrimmedMean(c=0.1)
            - Hampel(a=1.7,b=3.4,c=8.5)

        p_adjust : str
            Method used for testing and adjustment of p-values. Can be either the full name or initial letters. Available methods are:

                - 'bonf': one-step Bonferroni correction
                - 'sidak': one-step Sidak correction
                - 'holm': step-down method using Bonferroni adjustments
                - 'fdr_bh': Benjamini/Hochberg FDR correction
                - 'fdr_by': Benjamini/Yekutieli FDR correction
                - None: pass-through option (no correction applied)

        return_results: bool
            Whether to return results.
        kwargs : Any
            Additional arguments for sm.RLM

        Returns:
        --------
        RLMResults
            Robust regression results
        """
        if not isinstance(formula, str):
            raise ValueError("formula must be a string")
        y, X = patsy.dmatrices(formula, self.df, return_type='dataframe')

        if show_vif:
            self._check_multicollinearity(X)

        model = sm.RLM(y, X, missing=missing, M=M, **kwargs)

        results = model.fit()

        if show_summary:
            display(results.summary())

        if show_plots:
            self._show_minimal_diagnostic_plots(results)

        if return_results:
            return results

    def glm(
        self,
        formula: str,
        family: sm.families.Family = sm.families.Gaussian(),
        cov_type: Literal['nonrobust', 'HC0', 'HC1', 'HC2', 'HC3'] = "HC3",
        show_summary: bool = True,
        show_plots: bool = False,
        show_vif: bool = True,
        missing: str = 'drop',
        p_adjust: Optional[str] = None,
        return_results: bool = False,
        **kwargs: Any
    ):
        """
        Fit Generalized Linear Model (GLM).

        Best for:

        - Non-normal distributions (binary, count, etc.)
        - When you need link functions
        - Extending linear models to exponential family

        Parameters:
        -----------
        formula : str
            Patsy formula (e.g., 'y ~ x1 + x2')
        family : Family
            Exponential family distribution (default: Gaussian)

            Common options:

            - sm.families.Gaussian() - Normal distribution
            - sm.families.Binomial() - Logistic regression
            - sm.families.Poisson() - Poisson regression
            - sm.families.Gamma() - Gamma regression
            - sm.families.NegativeBinomial() - Negative binomial

        cov_type : str
            Covariance matrix type (default: 'HC3')
        show_summary : bool
            Whether to print summary (default: True)
        show_plots : bool
            Whether to show diagnostic plots (default: False)
        show_vif : bool
            Whether to show VIF (default: True)
        missing : str
            How to handle missing values (default: 'drop')
        p_adjust : str
            Method used for testing and adjustment of p-values. Can be either the full name or initial letters. Available methods are:

                - 'bonf': one-step Bonferroni correction
                - 'sidak': one-step Sidak correction
                - 'holm': step-down method using Bonferroni adjustments
                - 'fdr_bh': Benjamini/Hochberg FDR correction
                - 'fdr_by': Benjamini/Yekutieli FDR correction
                - None: pass-through option (no correction applied)

        return_results: bool
            Whether to return results.
        kwargs : Any
            Additional arguments for sm.GLM

        Returns:
        --------
        GLMResults
            GLM regression results
        """
        if not isinstance(formula, str):
            raise ValueError("formula must be a string")
        y, X = patsy.dmatrices(formula, self.df, return_type='dataframe')

        if show_vif:
            self._check_multicollinearity(X)

        model = sm.GLM(y, X, family=family, missing=missing, **kwargs)

        fit_kwargs = {'cov_type': cov_type, 'use_t': True}

        results = model.fit(**fit_kwargs)
        if show_summary:
            display(results.summary())

        # Apply p-value adjustment if requested
        if p_adjust:
            adj_df = self._adjust_regression_pvalues(results, p_adjust)
            caption = f"Adjusted p-values (method: {p_adjust}):"

            display(style_dataframe(
                adj_df[['coef', 'p-unc', 'p-corr']]
                , hide_index=False
                , hide_columns=False
                , caption=caption
                , caption_font_size=16
                , formatters={
                    'coef': '{:.2f}'
                    , 'p-unc': '{:.3f}'
                    , 'p-corr': '{:.3f}'
                }
            ))
            if show_plots:
                self._show_minimal_diagnostic_plots(results)

            if return_results:
                return results

    def quantreg(
        self,
        formula: str,
        q: float = 0.5,
        show_summary: bool = True,
        show_plots: bool = False,
        show_vif: bool = True,
        missing: str = 'drop',
        return_results: bool = False,
        **kwargs: Any
    ):
        """
        Fit Quantile Regression (median by default).

        Best for:

        - Non-normal error distributions
        - Heteroscedastic data
        - Analyzing conditional quantiles
        - Extreme value analysis

        Parameters:
        -----------
        formula : str
            Patsy formula (e.g., 'y ~ x1 + x2')
        q : float
            Quantile to estimate (0.5 for median)
        show_summary : bool
            Whether to print summary (default: True)
        show_plots : bool
            Whether to show diagnostic plots (default: False)
        show_vif : bool
            Whether to show VIF (default: True)
        missing : str
            How to handle missing values (default: 'drop')
        return_results: bool
            Whether to return results.
        kwargs : Any
            Additional arguments for sm.QuantReg

        Returns:
        --------
        QuantRegResults
            Quantile regression results
        """
        if not isinstance(formula, str):
            raise ValueError("formula must be a string")
        if not 0 < q < 1:
            raise ValueError("Quantile q must be between 0 and 1")

        y, X = patsy.dmatrices(formula, self.df, return_type='dataframe')

        if show_vif:
            self._check_multicollinearity(X)

        model = sm.QuantReg(y, X, missing=missing, **kwargs)
        results = model.fit(q=q)

        if show_summary:
            display(results.summary())

        if show_plots:
            self._show_minimal_diagnostic_plots(results)

        if return_results:
            return results

    def ordered_model(
        self,
        formula: str,
        distr: str = 'logit',
        method: str = 'bfgs',
        p_adjust: Optional[str] = None,
        show_summary: bool = True,
        return_results: bool = False,
        **kwargs
    ):
        """
        Fit an ordinal regression model (Proportional Odds Model).

        Parameters:
        -----------
        formula : str
            Patsy formula specifying the model (e.g., 'y ~ x1 + C(x2)')
        distr : str, optional
            Distribution for the model: 'logit' (default), 'probit', or 'loglog'
        method : str, optional
            Optimization method: 'bfgs' (default), 'newton', 'lbfgs', etc.
        p_adjust : str
            Method used for testing and adjustment of p-values. Can be either the full name or initial letters. Available methods are:

                - 'bonf': one-step Bonferroni correction
                - 'sidak': one-step Sidak correction
                - 'holm': step-down method using Bonferroni adjustments
                - 'fdr_bh': Benjamini/Hochberg FDR correction
                - 'fdr_by': Benjamini/Yekutieli FDR correction
                - None: pass-through option (no correction applied)

        show_summary : bool, optional
            Whether to print model summary (default: True)
        return_results : bool, optional
            Whether to return fitted model results (default: False)
        kwargs
            Additional arguments passed to OrderedModel.fit()

        Returns:
        --------
            OrderedModelResults (if return_results=True)
        """
        if not pd.api.types.is_categorical_dtype(self.df[formula.split('~')[0].strip()]):
            raise TypeError("Dependent variable must be ordered categorical. Use pd.Categorical")
        # Prepare data
        X = patsy.dmatrix(formula.split('~')[1], self.df, return_type='dataframe')
        X = X.drop('Intercept', axis=1)  # Remove intercept for OrderedModel
        y = self.df[formula.split('~')[0].strip()]

        # Fit model
        model = OrderedModel(y, X, distr=distr)
        result = model.fit(method=method, **kwargs)

        # Apply p-value adjustment if requested
        if p_adjust and hasattr(result, 'pvalues'):
            pvals = result.pvalues[1:]  # Exclude threshold parameters
            _, pvals_corrected = pg.multicomp(pvals, method=p_adjust)

            # Create summary DataFrame with adjusted p-values
            summary_df = pd.DataFrame({
                'coef': result.params,
                'std_err': result.bse,
                'p_value': result.pvalues,
                'p_adj': np.concatenate([
                    [np.nan] * (len(result.params) - len(pvals_corrected)),  # For thresholds
                    pvals_corrected
                ])
            })

            if show_summary:
                display(result.summary())
                print("\n🔹 Adjusted p-values (method: {}):".format(p_adjust))
                display(summary_df[['coef', 'p_value', 'p_adj']].round(4))
        elif show_summary:
            display(result.summary())

        if return_results:
            return results

    def mixedlm(
        self,
        formula: str,
        groups: str,
        show_summary: bool = True,
        show_plots: bool = False,
        show_vif: bool = True,
        missing: str = 'drop',
        return_results: bool = False,
        **kwargs: Any
    ):
        """
        Fit Linear Mixed Effects Model.

        Best for:

        - Clustered/hierarchical data
        - Repeated measures
        - Longitudinal studies
        - Random effects modeling

        Parameters:
        -----------
        formula : str
            Patsy formula for fixed effects
        groups : str
            Column name for group membership
        show_summary : bool
            Whether to print summary (default: True)
        show_plots : bool
            Whether to show diagnostic plots (default: False)
        show_vif : bool
            Whether to show VIF (default: True)
        missing : str
            How to handle missing values (default: 'drop')
        return_results: bool
            Whether to return results.
        kwargs : Any
            Additional arguments for sm.MixedLM

        Returns:
        --------
        MixedLMResults
            Mixed effects model results
        """
        if not isinstance(formula, str):
            raise ValueError("formula must be a string")
        if groups not in self.df.columns:
            raise ValueError(f"Group variable '{groups}' not found in data")
        y, X = patsy.dmatrices(formula, self.df, return_type='dataframe')
        if show_vif:
            self._check_multicollinearity(X)
        model = sm.MixedLM(y, X, missing=missing, groups=groups, **kwargs)
        results = model.fit()

        if show_summary:
            display(results.summary())

        if show_plots:
            self._show_minimal_diagnostic_plots(results)
        if return_results:
            return results

    def _check_multicollinearity(self, X: pd.DataFrame) -> None:
        """
        Calculate and display Variance Inflation Factors (VIF).

        Parameters:
        -----------
        X : pd.DataFrame
            Design matrix from patsy.dmatrices

        Notes:
        ------
        - VIF > 10 indicates severe multicollinearity
        - VIF > 5 suggests moderate multicollinearity
        """
        if 'Intercept' in X.columns:
            X = X.drop('Intercept', axis=1)
        if X.shape[1] == 1:
            print("VIF requires at least two predictors. Skipping check.")
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        vif_df = pd.DataFrame({
            'Variable': X.columns,
            'VIF': vif
        }).sort_values('VIF', ascending=False)

        display(style_dataframe(vif_df.round(3).astype(str), hide_columns=False, caption='Variance Inflation Factors'))
        if (vif_df['VIF'] > 10).any():
            rprint("[red]Warning: VIF > 10 indicates severe multicollinearity[/red]")
        elif (vif_df['VIF'] > 5).any():
            rprint("[yellow]Warning: VIF > 5 suggests moderate multicollinearity[/yellow]")

    def _show_minimal_diagnostic_plots(
        self,
        results,
    ) -> None:
        """
        Show minimalist diagnostic plots with enhanced styling options.

        Parameters:
        -----------
        results : Regression results object
            Results from any statsmodels regression
        """
        # Default plot style
        default_style = {
            'figsize': (8, 3.3),
            'color': '#4C78A8',
            'grid_style': {
                'alpha': 0.3,
                'linestyle': '--',
                'linewidth': 0.5,
                'color': 'gray'
            },
            'transparent': True,
            'dpi': 100
        }
        if hasattr(results, 'resid'):
            residuals = results.resid
        elif hasattr(results, 'resid_response'):  # For GLM
            residuals = results.resid_response
        else:
            raise AttributeError("Cannot find residuals in results object")
        # Merge default style with user-provided style
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=default_style['figsize'])

            # Residuals vs Fitted
            sns.scatterplot(
                x=results.fittedvalues,
                y=residuals,
                color=default_style['color'],
                edgecolor='none',
                ax=ax1
            )
            ax1.axhline(0, color='r', linestyle='--', lw=1)
            ax1.set_title('Residuals vs Fitted', pad=10)
            ax1.set_xlabel('Fitted values')
            ax1.collections[0].set_edgecolor('none')
            ax1.set_ylabel('Residuals')

            # Remove spines and ticks
            # for spine in ax1.spines.values():
            #     spine.set_visible(False)
            ax1.tick_params(axis='both', which='both', length=0)

            # Add grid
            ax1.grid(True, **default_style['grid_style'])

            # Q-Q plot
            pg.qqplot(residuals, ax=ax2)
            ax2.set_title('Q-Q Plot: Residual Normality Check', pad=10)
            ax2.collections[0].set_color(default_style['color'])
            ax2.collections[0].set_edgecolor('none')
            ax2.set_xlabel('Theoretical Quantiles')
            ax2.set_ylabel('Sample Quantiles')

            # Remove spines and ticks for Q-Q plot
            # for spine in ax2.spines.values():
            #     spine.set_visible(False)
            ax2.tick_params(axis='both', which='both', length=0)

            # Add grid for Q-Q plot
            ax2.grid(True, **default_style['grid_style'])

            plt.tight_layout()
            plt.show()

        except ImportError:
            rprint("[yellow]⚠️ matplotlib/seaborn not available - plots skipped[/yellow]")

    def feature_importance_analysis(
        self,
        target_column: str,
        feature_columns: List[str],
        problem_type: str = 'auto',
        test_size: float = 0.2,
        n_estimators: int = 100,
        top_n_features: int = 20,
        show_plots: bool = True,
        use_statistical: bool = True,
        use_l1: bool = True,
        use_mutual_info: bool = True,
        return_results: bool = False,
        horizontal_spacing: float = 0.15,
        vertical_spacing: float = 0.1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        show_values: bool = True,
    ) -> Union[None, Dict[str, FeatureImportanceResult]]:
        """
        Advanced feature importance analysis using multiple methods and models.

        Parameters:
        -----------
        target_column : str
            Name of the target variable column
        feature_columns : list
            List of feature column names
        problem_type : str, optional (default='auto')
            Problem type: 'classification', 'regression' or 'auto' (auto-detection)
        test_size : float, optional (default=0.2)
            Size of the test set for evaluating importance on holdout data
        n_estimators : int, optional (default=100)
            Number of trees in ensemble models
        top_n_features : int, optional (default=20)
            Number of top features to display
        show_plots : bool, optional (default=True)
            Whether to show plots
        use_statistical : bool, optional (default=True)
            Whether to use statistical methods (F-test).
            Statistical tests measure the strength of the relationship between
            each feature and the target. Good for linear relationships.
        use_l1 : bool, optional (default=True)
            Whether to use L1 regularization (Lasso/Logistic Regression).
            L1 regularization performs feature selection by shrinking some
            coefficients to exactly zero. Useful for identifying a small subset
            of important features.
        use_mutual_info : bool, optional (default=True)
            Whether to use mutual information method.
            Mutual information measures the dependency between variables, capturing
            both linear and non-linear relationships. Good for detecting complex
            patterns that other methods might miss.
        return_results : bool, optional (default=True)
            Whether to return results.
        horizontal_spacing, vertical_spacing : float
            Spacing for plotly subplots.
        height, width : int, optional
            height, width for plotly figure
        show_values ; bool
            Whether to show text values on bars

        Returns:
        --------
        dict
            Dictionary with results (FeatureImportanceResult for each method)
        """
        df = self.df
        # Input data validation
        if not all(col in df.columns for col in [target_column] + feature_columns):
            raise ValueError("Some specified columns are missing from the dataframe")

        if len(feature_columns) < 2:
            raise ValueError("At least 2 features must be specified")
        # Prepare data
        df = df[[target_column] + feature_columns]
        initial_rows = len(df)
        df = df.dropna(subset=[target_column])
        removed_rows = initial_rows - len(df)

        if removed_rows > 0:
            removal_pct = (removed_rows / initial_rows) * 100
            rprint(f"[yellow]⚠️ Removed {removed_rows} rows ({removal_pct:.2f}%) with missing values[/yellow]")

        # Handle categorical variables
        categorical_cols = [
            var for var in feature_columns
            if not pd.api.types.is_numeric_dtype(df[var]) or df[var].nunique() < 10
        ]

        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols)
            feature_columns = [col for col in df.columns if col != target_column]

        X = df[feature_columns]
        y = df[target_column]

        if (X == 0).mean().mean() > 0.8:
            warnings.warn("High sparsity in data after one-hot encoding")

        # Determine problem type
        if problem_type == 'auto':
            if y.dtype == 'object' or y.nunique() < 10:
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        scaler = StandardScaler()
        numeric_cols = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols:
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size)

        results = {}

        # Metrics for model evaluation
        if problem_type == 'classification':
            n_classes = y.nunique()
            if n_classes == 1:
                raise ValueError("Only one class present in target variable")
            if n_classes == 2:  # Binary classification
                metric_func = roc_auc_score
                metric_name = 'ROC AUC'
                eval_metric = 'roc_auc'
                predict_method = 'predict_proba'
            else:  # Multiclass classification
                metric_func = lambda y_true, y_pred: log_loss(y_true, y_pred)
                metric_name = 'Log Loss'
                eval_metric = 'neg_log_loss'
                predict_method = 'predict_proba'
        else:
            metric_func = r2_score
            metric_name = 'R2'
            eval_metric = 'neg_mean_squared_error'
            predict_method = 'predict'

        if problem_type == 'classification' and n_classes > 2:
            # Test check on the main models
            test_models = [
                RandomForestClassifier(n_estimators=1),
                GradientBoostingClassifier(n_estimators=1),
                LogisticRegression()
            ]
            if not all(hasattr(m, 'predict_proba') for m in test_models):
                warnings.warn("Some models don't support predict_proba, falling back to accuracy")
                metric_func = accuracy_score
                metric_name = 'accuracy'
                predict_method = 'predict'
        # 1. Random Forest Feature Importance
        if problem_type == 'classification':
            rf_model = RandomForestClassifier(n_estimators=n_estimators)
        else:
            rf_model = RandomForestRegressor(n_estimators=n_estimators)

        rf_model.fit(X_train, y_train)
        importances = rf_model.feature_importances_
        rf_importance = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
        rf_importance = rf_importance.sort_values('Importance', ascending=False)

        # Evaluate model performance
        if predict_method == 'predict_proba':
            y_pred = getattr(rf_model, predict_method)(X_test)
            if n_classes == 2:
                y_pred = y_pred[:, 1]
        else:
            y_pred = rf_model.predict(X_test)

        try:
            metric_value = metric_func(y_test, y_pred)
        except Exception as e:
            warnings.warn(f"Failed to calculate {metric_name}: {str(e)}")
            metric_value = None

        results['random_forest'] = FeatureImportanceResult(
            importance_df=rf_importance,
            model=rf_model,
            metric_name=metric_name,
            metric_value=metric_value
        )
        # 2. Gradient Boosting Feature Importance
        if problem_type == 'classification':
            gb_model = GradientBoostingClassifier(n_estimators=n_estimators)
        else:
            gb_model = GradientBoostingRegressor(n_estimators=n_estimators)

        gb_model.fit(X_train, y_train)
        importances = gb_model.feature_importances_
        gb_importance = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
        gb_importance = gb_importance.sort_values('Importance', ascending=False)

        # Evaluate model performance
        if predict_method == 'predict_proba':
            y_pred = getattr(gb_model, predict_method)(X_test)
            if n_classes == 2:
                y_pred = y_pred[:, 1]
        else:
            y_pred = gb_model.predict(X_test)

        try:
            metric_value = metric_func(y_test, y_pred)
        except Exception as e:
            warnings.warn(f"Failed to calculate {metric_name}: {str(e)}")
            metric_value = None

        results['gradient_boosting'] = FeatureImportanceResult(
            importance_df=gb_importance,
            model=gb_model,
            metric_name=metric_name,
            metric_value=metric_value
        )
        # display(rf_importance)
        # 3. Statistical Feature Selection
        if use_statistical:
            try:
                if problem_type == 'classification':
                    selector = SelectKBest(score_func=f_classif, k='all')
                else:
                    selector = SelectKBest(score_func=f_regression, k='all')

                selector.fit(X, y)
                stats_importance = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': selector.scores_,
                    'P-value': selector.pvalues_
                })
                stats_importance = stats_importance.sort_values('Importance', ascending=False)

                results['statistical_f_test'] = FeatureImportanceResult(
                    importance_df=stats_importance,
                    metric_name='F-statistic'
                )
            except Exception as e:
                warnings.warn(f"Statistical feature selection (F-test) failed: {str(e)}")
        # display(stats_importance)
        # 4. Mutual Information
        if use_mutual_info:
            try:
                if problem_type == 'classification':
                    mi_scores = mutual_info_classif(X, y)
                else:
                    mi_scores = mutual_info_regression(X, y)

                mi_importance = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': mi_scores
                })
                mi_importance = mi_importance.sort_values('Importance', ascending=False)

                results['mutual_information'] = FeatureImportanceResult(
                    importance_df=mi_importance,
                    metric_name='Mutual Information'
                )
            except Exception as e:
                warnings.warn(f"Mutual information calculation failed: {str(e)}")
                use_mutual_info = False
        # display(mi_importance)
        # 5. L1-based Feature Importance
        if use_l1:
            try:
                if problem_type == 'classification':
                    l1_model = LogisticRegression(
                        penalty='l1',
                        solver='liblinear',
                        max_iter=1000
                    )
                else:
                    l1_model = Lasso(alpha=0.01)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    l1_model.fit(X_train, y_train)

                if problem_type == 'classification':
                    coef = l1_model.coef_[0] if len(l1_model.coef_.shape) > 1 else l1_model.coef_
                else:
                    coef = l1_model.coef_

                l1_importance = pd.DataFrame({'Feature': feature_columns, 'Importance': np.abs(coef)})
                l1_importance = l1_importance.sort_values('Importance', ascending=False)
                # Evaluate model performance
                if predict_method == 'predict_proba':
                    y_pred = getattr(l1_model, predict_method)(X_test)
                    if n_classes == 2:
                        y_pred = y_pred[:, 1]
                else:
                    y_pred = l1_model.predict(X_test)

                try:
                    metric_value = metric_func(y_test, y_pred)
                except Exception as e:
                    warnings.warn(f"Failed to calculate {metric_name}: {str(e)}")
                    metric_value = None


                results['l1_regularization'] = FeatureImportanceResult(
                    importance_df=l1_importance,
                    model=l1_model,
                    metric_name='L1 Coefficients',
                    metric_value=metric_value
                )
            except Exception as e:
                warnings.warn(f"L1 regularization failed: {str(e)}")
                use_l1 = False
        # display(l1_importance)
        # Visualization
        if show_plots:
            self._plot_feature_importance(
                results,
                problem_type=problem_type,
                top_n_features=top_n_features,
                use_statistical=use_statistical,
                use_l1=use_l1,
                use_mutual_info=use_mutual_info,
                horizontal_spacing=horizontal_spacing,
                vertical_spacing=vertical_spacing,
                height=height,
                width=width,
                show_values=show_values
            )
        if return_results:
            return results

    def _plot_feature_importance(
        self,
        results: Dict[str, FeatureImportanceResult],
        problem_type: str,
        top_n_features: int = 20,
        use_statistical: bool = True,
        use_l1: bool = True,
        use_mutual_info: bool = True,
        horizontal_spacing: float = 0.15,
        vertical_spacing: float = 0.1,
        height: int = None,
        width: int = None,
        show_values: bool = True
    ) -> None:
        """Visualize feature importance analysis results"""

        # List to store traces
        traces = []
        subplot_titles = []

        # Add Random Forest and Gradient Boosting Importance
        rf_importance_df = results['random_forest'].importance_df
        gb_importance_df = results['gradient_boosting'].importance_df

        rf_name = 'Random Forest'
        gb_name = 'Gradient Boosting'

        rf_name += f' ({results["random_forest"].metric_name}'
        gb_name += f' ({results["gradient_boosting"].metric_name}'

        # Combine RF and GB into one DataFrame
        combined_importance = pd.merge(
            rf_importance_df,
            gb_importance_df,
            on='Feature',
            suffixes=('_rf', '_gb')
        )

        # Sort by combined importance (sum of RF and GB)
        combined_importance['Combined Importance'] = combined_importance['Importance_rf'] + combined_importance['Importance_gb']
        combined_importance = combined_importance.sort_values(
            by='Combined Importance',
            ascending=False
        ).head(top_n_features)

        combined_importance_df = pd.melt(
            combined_importance,
            id_vars='Feature',
            var_name='Model',
            value_name='Importance'
        )

        # Add RF and GB traces to the same subplot
        traces.append(go.Bar(
            x=combined_importance_df[combined_importance_df['Model'] == 'Importance_rf']['Importance'],
            y=combined_importance_df[combined_importance_df['Model'] == 'Importance_rf']['Feature'],
            name=rf_name,
            orientation='h',
            marker_color='#4C78A8',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>',
            text=[f'{x:.2f}' for x in combined_importance_df[combined_importance_df['Model'] == 'Importance_rf']['Importance']] if show_values else None,
        ))
        traces.append(go.Bar(
            x=combined_importance_df[combined_importance_df['Model'] == 'Importance_gb']['Importance'],
            y=combined_importance_df[combined_importance_df['Model'] == 'Importance_gb']['Feature'],
            name=gb_name,
            orientation='h',
            marker_color='#FF9E4A',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>',
            text=[f'{x:.2f}' for x in combined_importance_df[combined_importance_df['Model'] == 'Importance_gb']['Importance']] if show_values else None,
        ))
        subplot_titles.append('Random Forest/Gradient Boosting Importance')

        # 3. Statistical Feature Selection
        if use_statistical and 'statistical_f_test' in results:
            stats_top = results['statistical_f_test'].importance_df.head(top_n_features)
            traces.append(go.Bar(
                x=stats_top['Importance'],
                y=stats_top['Feature'],
                name='Statistical',
                orientation='h',
                marker_color='#57B16E',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>',
                text=[f'{x:.2f}' for x in stats_top['Importance']] if show_values else None,
            ))
            subplot_titles.append('Statistical Feature Scores')

        # 4. Mutual Information
        if use_mutual_info and 'mutual_information' in results:
            mutual_info_top = results['mutual_information'].importance_df.head(top_n_features)
            traces.append(go.Bar(
                x=mutual_info_top['Importance'],
                y=mutual_info_top['Feature'],
                name='Mutual',
                orientation='h',
                marker_color='#E25559',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>',
                text=[f'{x:.2f}' for x in mutual_info_top['Importance']] if show_values else None
            ))
            subplot_titles.append('Mutual Information')

        # 5. L1-based Feature Importance
        if use_l1 and 'l1_regularization' in results:
            l1_top = results['l1_regularization'].importance_df.head(top_n_features)
            traces.append(go.Bar(
                x=l1_top['Importance'],
                y=l1_top['Feature'],
                name='L1 Regularization',
                orientation='h',
                marker_color='#8B6BB7',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>',
                text=[f'{x:.2f}' for x in l1_top['Importance']] if show_values else None
            ))
            subplot_titles.append('L1 Regularization Coefficients')

        n_plots = len(traces)

        # Determine subplot layout
        if n_plots == 1:
            rows, cols = 1, 1
        elif n_plots == 2:
            rows, cols = 1, 2
        else:
            rows, cols = 2, 2


        # Create subplot figure
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing
        )

        # Add traces to the figure
        for i, trace in enumerate(traces, -1):
            row = (i // cols) + 1
            col = (i % cols) + 1
            if i in [-1, 0]:
                row, col = 1, 1
            fig.add_trace(trace, row=row, col=col)
        legend_config = {
            'orientation': "h",
            'yanchor': "top",
            'y': 1.1 ,
            'xanchor': "center",
            'x': 0.5,
            'itemsizing': "constant"
        }
        # Update layout
        if not height:
            height=700 if n_plots > 1 else 400
        if not width:
            width=900 if n_plots > 1 else 600
        fig.update_layout(
            height=height,
            width=width,
            title_text=f"Feature Importance Analysis - {problem_type.capitalize()} Problem",
            title_y = 0.97,
            showlegend=True,
            barmode='group',
            margin= dict(l=50, r=50, b=50, t=100),
            legend=legend_config
        )
        for col in [1, 2]:
            fig.update_xaxes(title_text="Importance Score", row=2, col=col)
        fig.update_yaxes(autorange="reversed")
        CustomFigure(fig).show()

    def _adjust_regression_pvalues(self, results, p_adjust: str):
        """
        Adjust p-values for multiple comparisons in regression results.

        Parameters:
        -----------
        results : statsmodels.RegressionResults
            Fitted regression model results object
        p_adjust : str, optional
            Method for p-value adjustment. Options:

            - 'holm': Holm-Bonferroni (default)
            - 'bonferroni': Bonferroni correction
            - 'fdr_bh': Benjamini-Hochberg FDR
            - Other methods supported by pingouin.multicomp

        Returns:
        --------
        pd.DataFrame
            DataFrame with coefficients and adjusted p-values

        Notes:
        ------
        - Intercept is excluded from adjustment
        - Uses pingouin.multicomp for adjustment
        - Returns a copy of results.summary_df with added 'p_adj' column
        """
        if not hasattr(results, 'pvalues'):
            raise ValueError("Results object must have pvalues attribute")

        # Get p-values excluding intercept
        pvals = results.pvalues[1:]

        # Apply p-value adjustment
        _, pvals_corrected = pg.multicomp(pvals, method=p_adjust)

        # Create results DataFrame with adjusted p-values
        if hasattr(results, 'summary_df'):
            results_df = results.summary_df.copy()
        else:
            results_df = pd.DataFrame({
                'coef': results.params,
                'std_err': results.bse,
                'p-unc': results.pvalues
            })

        # Add adjusted p-values (leave intercept as NaN)
        results_df['p-corr'] = np.concatenate([
            [np.nan],  # Intercept
            pvals_corrected
        ])

        return results_df