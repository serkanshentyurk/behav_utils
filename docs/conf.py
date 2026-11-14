"""Sphinx configuration for the behav_utils documentation site.

Build from the repository root, after ``pip install -e ".[docs]"``::

    sphinx-build -W --keep-going -b html docs docs/_build/html

Every notebook in ``docs/tutorials`` is executed on each build; a failing cell, or any warning, fails
the build. The docs workflow runs the same command and sets ``DOCS_VERSION`` to the folder it publishes into
(``dev``, or a release's major.minor); the site's address is ``Documentation`` in pyproject.toml.
"""

import os
import re
import tomllib
from pathlib import Path

import behav_utils

ROOT = Path(__file__).resolve().parent.parent
SITE = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['urls']['Documentation'].rstrip('/')
DOCS_VERSION = os.environ.get('DOCS_VERSION', 'dev')

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
    'header_links_before_dropdown': 7,
    'navigation_with_keys': False,
    'navbar_end': ['version-switcher', 'theme-switcher', 'navbar-icon-links'],
    'switcher': {'json_url': f'{SITE}/versions.json', 'version_match': DOCS_VERSION},
    'check_switcher': False,
    'show_version_warning_banner': True,
}


def _hide_private_parameters(app, what, name, obj, options, signature, return_annotation):
    """Leave parameters whose names start with ``_`` out of signatures: private, as everywhere in the
    library (``AnimalData`` and ``SessionData`` carry two such fields)."""
    if not signature or '_' not in signature:
        return None
    shown = re.sub(r',\s*\*{0,2}_\w+(?:=[^,()\[\]{}]*)?', '', signature)
    shown = re.sub(r'\(\s*\*{0,2}_\w+(?:=[^,()\[\]{}]*)?(?:,\s*|(?=\)))', '(', shown)
    return (shown, return_annotation) if shown != signature else None


def setup(app):
    app.connect('autodoc-process-signature', _hide_private_parameters)
