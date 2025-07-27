Getting Started
===============

Installation
------------

.. admonition:: Recommended
   :class: important

   Use a virtual environment to prevent conflicts with existing package versions. 
   Frameon requires specific versions of dependencies that may affect other packages.

Basic Installation
~~~~~~~~~~~~~~~~~~

Using pip:

.. code-block:: bash

   pip install frameon

Using poetry:

.. code-block:: bash

   poetry add frameon

Installation with Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python's built-in venv:

.. code-block:: bash

   # Create and activate environment
   python -m venv frameon_env
   source frameon_env/bin/activate 
   
   pip install frameon

Poetry (manages virtual env automatically):

.. code-block:: bash

   # In your project directory
   poetry init  # For new projects
   poetry add frameon

Access Patterns
---------------

**DataFrame Operations**

.. code-block:: python

   df.explore.method()    # Dataset exploration
   df.preproc.method()    # Table transformations
   df.analyze.method()    # Analytical methods
   df.viz.method()        # Visualizations
   df.stats.method()      # Statistical analysis

**Series Operations**

.. code-block:: python

   df['col'].explore.method()    # Column analysis
   df['col'].preproc.method()    # Value transformations

Basic Usage
-----------

1. **Import and wrap your DataFrame**:

   .. code-block:: python

      import pandas as pd
      from frameon import FrameOn as fo
      
      # Create or load your DataFrame
      df = pd.read_csv('your_data.csv')
      
      # Add Frameon functionality
      df = fo(df)

2. **Explore your data**:

   - For **DataFrame-level** operations:

     .. code-block:: python

        df.explore.info()
        df.viz.bar()

   - For **column-specific** operations:

     .. code-block:: python

        df['price'].explore.info()
        df['date'].preproc.to_categorical()

Pandas Compatibility
--------------------

All standard operations remain available:

.. code-block:: python

   df.groupby('category').mean()
   df.query('price > 100')

Next Steps
----------

- Explore :doc:`examples <examples_gallery/index>`
- Check :doc:`complete API reference <api/index>`
