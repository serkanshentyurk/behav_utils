"""Sphinx configuration for the behav_utils documentation site.

Build from the repository root, after ``pip install -e ".[docs]"``::

    sphinx-build -W --keep-going -b html docs docs/_build/html

Every notebook in ``docs/tutorials`` is executed on each build; a failing cell, or any warning, fails
the build. CI runs the same command.
"""

import behav_utils

project = 'behav_utils'
author = 'Serkan Shentyurk'
copyright = '2026, Serkan Shentyurk'
release = behav_utils.__version__
version = '.'.join(release.split('.')[:2])

extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Headings down to level 3 get GitHub-style anchors, so links such as config_guide.md#go-no-go resolve.
myst_heading_anchors = 3

# Notebooks run on every build, and an error in any cell fails it.
nb_execution_mode = 'force'
nb_execution_raise_on_error = True
nb_execution_timeout = 600
nb_merge_streams = True

# API pages: each sub-package page documents the names in its __all__; docstrings are Google style, and
# types come from the annotations.
autodoc_default_options = {'members': True, 'member-order': 'bysource'}
autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True
napoleon_use_rtype = False

html_theme = 'pydata_sphinx_theme'
html_title = f'behav_utils {release}'
html_show_sourcelink = False
html_theme_options = {
    'icon_links': [
        {'name': 'GitHub', 'url': 'https://github.com/serkanshentyurk/behav_utils', 'icon': 'fa-brands fa-github'},
    ],
    'header_links_before_dropdown': 6,
    'navigation_with_keys': False,
}
