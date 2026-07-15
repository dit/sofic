import os

import sofic

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
    "sphinxcontrib.bibtex",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]

bibtex_bibfiles = ["references.bib"]

ipython_mplbackend = "agg"
ipython_execlines = [
    "import numpy as np",
    "np.set_printoptions(legacy='1.25')",
    "np.random.seed(0)",
    "import sofic",
]
ipython_savefig_dir = "images/"
ipython_warning_is_error = False

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

project = "sofic"
copyright = "2026, sofic contributors"  # noqa: A001
version = sofic.__version__
release = sofic.__version__

exclude_patterns = ["_build"]
add_module_names = False
pygments_style = "sphinx"
modindex_common_prefix = ["sofic."]
todo_include_todos = not on_rtd

# -- Math macros (single source of truth for HTML and PDF) ---------------------

_MACROS = {
    "op": [r"\operatorname{#1}\left[#2\right]", 2],
    "H": [r"\op{H}{#1}", 1],
    "I": [r"\op{I}{#1}", 1],
    "Cmu": [r"C_\mu", 0],
    "Emu": [r"E", 0],
    "chimu": [r"\chi", 0],
    "Cpm": [r"C_\pm", 0],
    "rhomu": [r"\rho_\mu", 0],
    "bmu": [r"b_\mu", 0],
    "rmu": [r"r_\mu", 0],
    "hmu": [r"h_\mu", 0],
    "ind": r"\mathrel{\large\text{$\perp\mkern-10mu\perp$}}",
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {"tex": {"macros": _MACROS}}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
htmlhelp_basename = "soficdoc"

_LATEX_RENEW = {"H"}


def _macros_to_latex(macros, renew):
    lines = []
    for name, defn in macros.items():
        cmd = "renewcommand" if name in renew else "newcommand"
        if isinstance(defn, list):
            tex_def, nargs = defn
            lines.append(rf"\{cmd}{{\{name}}}[{nargs}]{{{tex_def}}}")
        else:
            lines.append(rf"\{cmd}{{\{name}}}{{{defn}}}")
    return "\n".join(lines)


latex_elements = {
    "preamble": "\n".join(
        [
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",
            "",
            _macros_to_latex(_MACROS, _LATEX_RENEW),
        ]
    ),
}

latex_documents = [
    ("index", "sofic.tex", "sofic Documentation", "sofic Contributors", "manual"),
]

man_pages = [("index", "sofic", "sofic Documentation", ["sofic Contributors"], 1)]

texinfo_documents = [
    (
        "index",
        "sofic",
        "sofic Documentation",
        "sofic Contributors",
        "sofic",
        "Stochastic symbol generators in Python.",
        "Science",
    ),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "dit": ("https://dit.readthedocs.io/en/latest/", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

autodoc_member_order = "bysource"
