# Configuration file for the Sphinx documentation builder.
import os
import sys
from datetime import datetime
import frameon

# Add src directory to path for autodoc
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
project = 'frameon'
copyright = f'{datetime.now().year}, Pavel Grigoryev'
author = 'Pavel Grigoryev'
release = version = str(frameon.__version__)
language = "en"
# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_nb',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_plotly_directive',
    'sphinx.ext.intersphinx',
]

# Plotly configuration
plotly_include_source = True
plotly_html_show_formats = False
plotly_html_show_sourcelink = False
plotly_preview = True

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_attributes_as_param_list = False

# MyST-NB configuration for Jupyter notebooks
nb_execution_mode = 'auto'  # Execute notebooks
nb_execution_timeout = 300  # Increased timeout
myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'deflist',
    'html_admonition',
    'html_image',
    'colon_fence',
    'linkify',
    'replacements',
    'smartquotes',
    'substitution',
]

# Sphinx Gallery configuration
# sphinx_gallery_conf = {
#     'examples_dirs': ['../../examples/dataframe', '../../examples/series'],
#     'gallery_dirs': ['examples_gallery/dataframe', 'examples_gallery/series'],
#     'filename_pattern': r'.*\.ipynb',
#     'ignore_pattern': r'__init__\.py',
#     'run_stale_examples': True,
#     'min_reported_time': 5,
#     'image_scrapers': ('matplotlib',),
#     'capture_repr': ('_repr_html_', '__repr__'),
#     'plot_gallery': True,
#     'download_all_examples': True,
#     'within_subsection_order': 'FileNameSortKey',
#     'backreferences_dir': None,
#     'doc_module': ('frameon'),
#     'thumbnail_size': (400, 280),
#     'default_thumb_file': '_static/no_image.png',
#     'abort_on_example_error': False,
#     'notebook_images': True,
#     'junit': os.path.join('..', 'test-results', 'sphinx-gallery', 'junit.xml'),
#     'log_level': {'backreference_missing': 'warning'},
# }

templates_path = ['_templates']
exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store',
    '**.ipynb_checkpoints'
]

html_sidebars = {
    "getting_started": [],  # No sidebar for getting_started
    "changelog": [],  # No sidebar for changelog
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'

# Configure autosummary
autosummary_generate = True
autosummary_imported_members = True
autoclass_content = 'both'
add_module_names = False
html_show_sourcelink = False
autosummary_context = {
    'custom_template': True
}

# Ignore autosummary warnings
# suppress_warnings = [
#     'toc.not_included',
#     'autosummary'
# ]
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'github_url': '',
    'twitter_url': '',
    'use_edit_page_button': False,
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'icon_links': [
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/frameon/',
            'icon': 'fa-brands fa-python',
        },
        {
            "name": "GitHub",
            "url": "https://github.com/PavelGrigoryevDS/frameon",
            "icon": "fa-brands fa-github",
        },     
    ],
    'show_nav_level': 3,
    'navigation_depth': 4,
    'show_toc_level': 2,
    'navbar_start': ['navbar-logo'],  
    'logo': {
        'text': project, 
    },
}
      
def setup(app):
    app.add_css_file("custom.css")
    app.add_js_file("custom.js")
 
        
# Autodoc options
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'inherited-members': True, 
    'show-inheritance': True,    
    'special-members': '__init__',
    'exclude-members': '__weakref__, __.*, _.*', 
    'undoc-members': False,
    'private-members': False,
}
