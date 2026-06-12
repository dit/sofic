import os

import pensive

on_rtd = os.environ.get("READTHEDOCS", None) == "True"

needs_sphinx = "4.0"

primary_domain = "py"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

project = "pensive"
copyright = "2026, pensive contributors"  # noqa: A001
version = pensive.__version__
release = pensive.__version__

exclude_patterns = ["_build"]
add_module_names = False
pygments_style = "sphinx"
modindex_common_prefix = ["pensive."]
todo_include_todos = not on_rtd

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

autodoc_member_order = "bysource"
