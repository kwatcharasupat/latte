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

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))


# -- Project information -----------------------------------------------------

project = "Latte"
copyright = "2021, Karn N. Watcharasupat, Junyoung Lee, and Alexander Lerch"
author = "Karn N. Watcharasupat, Junyoung Lee, and Alexander Lerch"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    # "sphinx.ext.intersphinx",
    "autoapi.extension",
    "numpydoc",
    "m2r2",
]

autodoc_typehints = "description"
add_module_names = False

source_suffix = [".rst", ".md"]

autoapi_type = "python"
autoapi_dirs = ["../src/latte/"]
autoapi_options = [
    "members",
    # "undoc-members",
    # "inherited-members",
    "show-inheritance",
    "show-module-summary",
    # "imported-members",
]
autoapi_add_toctree_entry = False
autoclass_content = "class"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchmetrics": ("https://torchmetrics.readthedocs.io/en/stable/", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/stable", None),
    # "tensorflow": ("https://www.tensorflow.org/api_docs/python/tf", None)
}

templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
