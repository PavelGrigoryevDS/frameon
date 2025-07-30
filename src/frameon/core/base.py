import pandas as pd
from typing import Optional, Any, Union, overload, TYPE_CHECKING
from functools import cached_property
import copy

if TYPE_CHECKING: # pragma: no cover
    from frameon.series.explore import SeriesOnExplore
    from frameon.series.preprocessing import SeriesOnPreproc
    from frameon.dataframe.explore import FrameOnExplore
    from frameon.dataframe.preprocessing import FrameOnPreproc
    from frameon.dataframe.analysis import FrameOnAnalysis
    from frameon.dataframe.visualization import FrameOnViz
    from frameon.dataframe.statistics import FrameOnStats


class SeriesOn(pd.Series):
    """
    Enhanced pandas Series with additional functionality while maintaining performance.
    """
    _parent_df = None  # Default value to avoid attribute errors

    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False, _parent_df=None):
        """
        Initialize SeriesOn while properly handling parent DataFrame reference.
        
        Uses the same signature as pd.Series for maximum compatibility.
        """
        super().__init__(data, index, dtype, name, copy, fastpath)
        self._parent_df = _parent_df

    @property
    def parent_df(self) -> Optional["FrameOn"]:
        return self._parent_df

    @property
    def _constructor(self):
        """Return the constructor that should be used for all new instances."""
        return SeriesOn

    @property
    def _constructor_expanddim(self):
        """Return the constructor for DataFrame results (from 1D to 2D)."""
        return FrameOn

    # Method namespaces
    @cached_property
    def explore(self) -> "SeriesOnExplore":
        """Access exploratory data analysis methods."""
        from frameon.series.explore import SeriesOnExplore
        return SeriesOnExplore(self)

    @cached_property
    def preproc(self) -> "SeriesOnPreproc":
        """Access data preprocessing methods."""
        from frameon.series.preprocessing import SeriesOnPreproc
        return SeriesOnPreproc(self)

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other SeriesOn objects."""
        self = super().__finalize__(other, method, **kwargs)
        if isinstance(other, SeriesOn):
            self._parent_df = other._parent_df
        return self
    
    def __dir__(self):
        return sorted(set(super().__dir__() + [
            'explore',
            'preproc',
            'parent_df'
        ]))

class FrameOn(pd.DataFrame):
    """
    Enhanced pandas DataFrame that properly handles construction and maintains performance.
    """
    _metadata = ['_name_df', '_plotly_settings']
    _name_df = None  # Default value to avoid attribute errors

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, name_df=None):
        """
        Initialize FrameOn while maintaining pandas compatibility.
        """
        super().__init__(data, index, columns, dtype, copy)
        self._name_df = name_df

    @property
    def _constructor(self):
        """Return the constructor that should be used for all new instances."""
        def _smart_constructor(data=None, index=None, columns=None, dtype=None, copy=False):
            return FrameOn(
                data=data,
                index=index,
                columns=columns,
                dtype=dtype,
                copy=copy,
                name_df=self._name_df
            )
        return _smart_constructor

    @property
    def _constructor_sliced(self):
        """Return the constructor for Series results (from 2D to 1D)."""
        def _smart_constructor_sliced(data=None, index=None, name=None, dtype=None, copy=False):
            return SeriesOn(
                data=data,
                index=index,
                name=name,
                dtype=dtype,
                copy=copy,
                _parent_df=self
            )
        return _smart_constructor_sliced

    # Method namespaces
    @cached_property
    def explore(self) -> "FrameOnExplore":
        """Access exploratory data analysis methods."""
        from frameon.dataframe.explore import FrameOnExplore
        return FrameOnExplore(self)

    @cached_property
    def preproc(self) -> "FrameOnPreproc":
        """Access data preprocessing methods."""
        from frameon.dataframe.preprocessing import FrameOnPreproc
        return FrameOnPreproc(self)

    @cached_property
    def analysis(self) -> "FrameOnAnalysis":
        """Access advanced analysis methods."""
        from frameon.dataframe.analysis import FrameOnAnalysis
        return FrameOnAnalysis(self)

    @cached_property
    def viz(self) -> "FrameOnViz":
        """Access visualization methods."""
        from frameon.dataframe.visualization import FrameOnViz
        return FrameOnViz(self)

    @cached_property
    def stats(self) -> "FrameOnStats":
        """Access statistical methods."""
        from frameon.dataframe.statistics import FrameOnStats
        return FrameOnStats(self)

    @overload
    def __getitem__(self, key: str) -> SeriesOn: ...
    
    @overload
    def __getitem__(self, key: list[str]) -> "FrameOn": ...

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, pd.Series):
            series = SeriesOn(result, _parent_df=self)
            return series.__finalize__(self)
        elif isinstance(result, pd.DataFrame):
            df = FrameOn(result)
            return df.__finalize__(self)
        return result

    def __getattr__(self, name: str) -> "SeriesOn":
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if name in self.columns:
                return self[name]
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other FrameOn objects."""
        self = super().__finalize__(other, method, **kwargs)
        if isinstance(other, FrameOn):
            self._name_df = other._name_df
            if hasattr(other, 'viz') and hasattr(other.viz, '_plotly_settings'):
                if not hasattr(self, 'viz'):
                    from frameon.dataframe.visualization import FrameOnViz
                    self.viz = FrameOnViz(self)
                self.viz._plotly_settings = copy.deepcopy(other.viz._plotly_settings)
        return self

    def __dir__(self):
        """Add custom methods to autocomplete."""
        return sorted(set(
            super().__dir__() + 
            [
                'explore', 
                'preproc', 
                'analysis', 
                'viz', 
                'stats', 
            ]
        ))
