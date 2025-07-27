API Reference
=============

.. currentmodule:: frameon

Frameon extends pandas DataFrames and Series with additional analytical capabilities while preserving all native functionality.

Namespaces
----------

.. list-table::
   :widths: 30 70
   :header-rows: 1
   
   * - Namespace
     - Description
   * - ``.explore``
     - Data exploration
   * - ``.preproc``
     - Data cleaning and transformation
   * - ``.analyze``
     - Analytical methods (only for DataFrame-level)
   * - ``.viz``
     - Data visualization (only for DataFrame-level)
   * - ``.stats``
     - Statistical analysis (only for DataFrame-level)

Core Components
---------------

.. list-table::
   :widths: 30 70
   :header-rows: 1
   
   * - Component
     - Description
   * - :doc:`DataFrame-level <dataframe/index>`
     - Extended methods for full datasets
   * - :doc:`Series-level <series/index>`
     - Enhanced column operations
   * - :doc:`Utilities <utils>`
     - Helper functions
   * - :doc:`CustomFigure <customfigure>`
     - CustomFigure class for customize Plotly figure


.. toctree::
   :hidden:
   :caption: API Reference
   
   dataframe/index
   series/index
   utils
   customfigure