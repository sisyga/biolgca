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
# import sys
# sys.path.insert(0, os.path.abspath('.'))

#import pathlib
import sys
sys.path.insert(0, os.path.abspath('../..'))
#sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())


# -- Project information -----------------------------------------------------

project = 'BIO-LGCA'
copyright = '2024, TUD Dresden University of Technology'
author = 'Simon Syga, Bianca Güttner'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'sphinx.ext.duration',  # measure duration of sphinx processing
	'sphinx.ext.doctest',  # include test snippets
	'sphinx.ext.autodoc',
	'sphinx.ext.autosummary',
	'numpydoc',
	'sphinx_autodoc_typehints'
	#'sphinx.ext.napoleon'  # enable numpydoc-style docstring parsing
	#'myst_parser'
	#'m2r2'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
# standard: 'alabaster'
# matplotlib: 'sphinxdoc'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static/']
#html_css_files = ['css/custom.css']
html_style = 'css/methods.css'

#def setup(app):
#    app.add_css_file("css/custom.css")

# generate autosummary even if no references
# configured like https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion/
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_imported_members = False  # Do not include imported members
autodoc_mock_imports = ["matplotlib", "sympy", "mpl_toolkits"]

#autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
add_module_names = False # Remove namespaces from class/method signatures

# change root document
root_doc = 'index'

# formatting
add_function_parentheses = True  # default True
add_module_names = True  # default True
# To Do: html_theme_options = ...  see https://www.sphinx-doc.org/en/master/usage/theming.html#builtin-themes
# To Do: other HTML config https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# alabaster customization
# https://alabaster.readthedocs.io/en/latest/customization.html#theme-options
# to do: github button and read the rest
html_theme_options = {
    'fixed_sidebar' : True,
    'logo': '../../../images/biolgca_v2.png',
}
#html_logo = '../images/biolgca_v2.png'
