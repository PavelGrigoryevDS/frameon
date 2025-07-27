.. role:: badge

frameon
=======

.. grid:: 3
    :gutter: 3
    :class-container: card-grid
    
    .. grid-item-card::
        :class-card: doc-card
        :class-header: doc-card-header
        :text-align: center
        :link: getting_started.html
        
        🚀 **Quick Start**
        
        Installation and basic usage
        
        +++
        Get started →

    .. grid-item-card::
        :class-card: doc-card
        :class-header: doc-card-header
        :text-align: center
        :link: api/index.html
        
        📚 **API Reference**
        
        Detailed method documentation
        
        +++
        Explore API →

    .. grid-item-card::
        :class-card: doc-card
        :class-header: doc-card-header
        :text-align: center
        :link: examples_gallery/index.html
        
        🧪 **Examples**
        
        Practical usage scenarios
        
        +++
        View examples →

About Frameon
-------------

Frameon extends pandas DataFrames and Series with analysis methods while keeping all original functionality intact.

Key principles:

- **Seamless integration**: Works with existing pandas DataFrames and Series
- **Non-intrusive**: All pandas methods remain unchanged and fully available
- **Modular access**: Additional functionality organized in clear namespaces
- **Dual-level access**: Methods available for both entire DataFrames and individual columns

Method Levels
-------------

Frameon provides methods at two levels:

1. **DataFrame-level** - operate on the entire dataframe:

   .. code-block:: python

      df.explore.info()    # Summary for all columns
      df.viz.bar()         # Visualization using multiple columns

2. **Series-level** - work with individual columns:

   .. code-block:: python

      df['age'].explore.info()                # Summary for single column
      df['price'].preproc.to_categorical()    # Convert specific column to categorical data

Key points:

- Same namespaces (like ``.explore``) provide different methods for DataFrames and Series
- DataFrame methods focus on relationships *between* columns
- Series methods focus on operations *within* a single column

Built Upon
----------

Frameon utilizes the following open-source libraries as foundational components:

.. rst-class:: powered-by

- `pandas <https://pandas.pydata.org/>`_ - Core data structures
- `numpy <https://numpy.org/>`_ - Numerical computing
- `plotly <https://plotly.com/python/>`_ - Interactive visualization
- `scipy <https://www.scipy.org/>`_ - Scientific computing
- `pingouin <https://pingouin-stats.org/>`_ - Statistics
- `scikit-learn <https://scikit-learn.org/>`_ - Machine learning
- `statsmodels <https://www.statsmodels.org/>`_ - Statistical modeling
- `nltk <https://www.nltk.org/>`_ - Text processing


Core Features
-------------

- **Data exploration**: Quick insights and summaries
- **Preprocessing**: Common data cleaning operations
- **Advanced analysis**: Statistical tests and cohort analysis
- **Visualization**: Extended plotting capabilities

Getting Started
---------------

1. Install the package:
   
   .. code-block:: bash

      pip install frameon

2. Wrap your DataFrame:

   .. code-block:: python

      from frameon import FrameOn as fo
      df = fo(your_dataframe)

3. Start exploring:

   .. code-block:: python

      df.explore.info()           # For entire DataFrame
      df['col'].explore.info()    # For individual column

`Full guide → <getting_started.html>`_

.. toctree::
   :hidden:

   getting_started
   examples_gallery/index
   api/index
   changelog