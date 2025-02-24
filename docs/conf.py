"""
Configuration file for the Sphinx documentation builder.

This file is configured for the quantFin project. It sets up the Sphinx
extensions required to automatically generate documentation from docstrings
(using autodoc, napoleon, and autosummary) and adds the project source directory
to sys.path so that the instruments package can be documented.
"""

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'quantFin'
copyright = '2025, Diljit Singh'
author = 'Diljit Singh'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]
autosummary_generate = True

suppress_warnings = ['autodoc.duplicate_object']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']


# -- Event handler -----------------------------------------------------------
def skip_duplicate_members(app, what, name, obj, skip, options):
    # Skip duplicate documentation for dataclass fields
    if name in {"strike", "maturity", "is_call"}:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_duplicate_members)
