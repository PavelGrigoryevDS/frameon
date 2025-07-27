.. _api.dataframe:

DataFrame-level
===============
.. currentmodule:: frameon.dataframe.explore

Exploration Methods
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/dataframe/explore
   :template: autosummary/method_dataframe_explore.rst

   FrameOnExplore.info
   FrameOnExplore.anomalies_report
   FrameOnExplore.detect_anomalies
   FrameOnExplore.anomalies_corr_matrix
   FrameOnExplore.anomalies_combinations
   FrameOnExplore.anomalies_by_categories
   FrameOnExplore.detect_simultaneous_anomalies
   FrameOnExplore.anomalies_over_time

.. currentmodule:: frameon.dataframe.preprocessing

Preprocessing Methods
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/dataframe/preprocessing
   :template: autosummary/method_dataframe_preproc.rst

   FrameOnPreproc.impute_missing
   FrameOnPreproc.restore_full_index

.. currentmodule:: frameon.dataframe.analysis

Analysis Methods
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/dataframe/analysis
   :template: autosummary/method_dataframe_analysis.rst

   FrameOnAnalysis.cohort
   FrameOnAnalysis.corr_matrix
   FrameOnAnalysis.rfm
   FrameOnAnalysis.segment_polar
   FrameOnAnalysis.metric_by_dimensions_plot
   FrameOnAnalysis.sentiment
   FrameOnAnalysis.word_frequency

.. currentmodule:: frameon.dataframe.visualization

Visualization Methods
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/dataframe/visualization
   :template: autosummary/method_dataframe_viz.rst

   FrameOnViz.bar
   FrameOnViz.line
   FrameOnViz.area
   FrameOnViz.box
   FrameOnViz.heatmap
   FrameOnViz.pairplot
   FrameOnViz.pie_bar
   FrameOnViz.histogram
   FrameOnViz.cat_compare
   FrameOnViz.wordcloud
   FrameOnViz.parallel_categories
   FrameOnViz.period_change

.. currentmodule:: frameon.dataframe.statistics

Statistics Methods
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/dataframe/statistics
   :template: autosummary/method_dataframe_stats.rst

   FrameOnStats.normality
   FrameOnStats.levene
   FrameOnStats.ttest
   FrameOnStats.mwu
   FrameOnStats.anova
   FrameOnStats.kruskal
   FrameOnStats.chi2_independence
   FrameOnStats.bootstrap
   FrameOnStats.ols
   FrameOnStats.rlm
   FrameOnStats.glm
   FrameOnStats.quantreg
   FrameOnStats.ordered_model
   FrameOnStats.mixedlm
   FrameOnStats.feature_importance_analysis