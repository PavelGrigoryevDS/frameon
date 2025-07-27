.. _api.series:

Series-level
============

.. currentmodule:: frameon.series.explore

Exploration Methods
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/series/explore
   :template: autosummary/method_series_explore.rst

   SeriesOnExplore.info
   SeriesOnExplore.detect_anomalies
   SeriesOnExplore.detect_outliers
   SeriesOnExplore.anomalies_by_categories
   SeriesOnExplore.anomalies_over_time
   SeriesOnExplore.detect_window_outliers
   SeriesOnExplore.plot_rolling_anomaly_rate

.. currentmodule:: frameon.series.preprocessing

Preprocessing Methods
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/series/preprocessing
   :template: autosummary/method_series_preproc.rst

   SeriesOnPreproc.to_categorical
   SeriesOnPreproc.normalize_string_series
   SeriesOnPreproc.transform_numeric
   SeriesOnPreproc.fill_missing_by_category
   SeriesOnPreproc.impute_missing
   SeriesOnPreproc.calc_target_category_share
   SeriesOnPreproc.check_group_counts
