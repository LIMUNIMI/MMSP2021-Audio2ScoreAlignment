# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# changing extensions of pyx files and removing cython specific syntax
for root, dirs, files in os.walk('../'):
    for file in files:
        # change 'pyx' to 'py' if needed
        old_fname = os.path.join(root, file)
        if file.endswith('.pyx'):
            new_fname = old_fname[:-4] + '.py'
        elif file.endswith('.py'):
            new_fname = old_fname
        else:
            continue

        # change all 'cimport' to 'import'
        with open(old_fname, 'rt') as old_file:
            text = old_file.read().replace('cimport', 'import')
        with open(new_fname, 'wt') as new_file:
            new_file.write(text)

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'ASMD'
copyright = '2020, Federico Simonetta'
author = 'Federico Simonetta'

# The full version, including alpha/beta/rc tags
release = '0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode'
]
# autodoc_docstring_signature = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['*.so', '*.pyx']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'yummy-sphinx-theme'
# html_theme_path = ["_themes", ]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# some issue with master doc
master_doc = 'index'
