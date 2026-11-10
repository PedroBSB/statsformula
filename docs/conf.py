"""Sphinx configuration for the statsformula documentation."""

import ast
import contextlib
import io
import os
import runpy
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import statsformula  # noqa: E402

project = "statsformula"
author = "Pedro Albuquerque"
copyright = "2026, Pedro Albuquerque"
release = statsformula.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]
templates_path = []

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

html_theme = "furo"
html_title = "statsformula %s" % release
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/PedroBSB/statsformula/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [],
}

# ---------------------------------------------------------------------------
# examples gallery: run every script in examples/ and write one page per script
# ---------------------------------------------------------------------------

EXAMPLES = os.path.join(ROOT, "examples")
GALLERY = os.path.join(HERE, "examples")


def _build_gallery(app):
    os.makedirs(GALLERY, exist_ok=True)
    scripts = sorted(f for f in os.listdir(EXAMPLES) if f.endswith(".py"))
    toc = []
    for name in scripts:
        path = os.path.join(EXAMPLES, name)
        source = open(path, encoding="utf-8").read()
        doc = ast.get_docstring(ast.parse(source)) or name
        first, _, body = doc.partition("\n")
        title, sep, subtitle = first.partition(": ")
        if sep and len(title) > 8:
            body = "*" + subtitle.strip().rstrip(".") + ".*\n\n" + body.strip()
        else:
            title = first
        out = io.StringIO()
        cwd = os.getcwd()
        try:
            os.chdir(EXAMPLES)
            with warnings.catch_warnings(), contextlib.redirect_stdout(out):
                warnings.simplefilter("ignore")
                runpy.run_path(path, run_name="__main__")
        except Exception as e:  # keep building; the page shows the error
            out.write("\n[example failed: %r]\n" % (e,))
        finally:
            os.chdir(cwd)
        stem = name[:-3]
        with open(os.path.join(GALLERY, stem + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(out.getvalue())
        page = ["# " + title.strip().rstrip("."), "", body.strip(), "",
                "**Script:** [`examples/%s`](https://github.com/PedroBSB/statsformula/blob/main/examples/%s)"
                % (name, name), "", "```{literalinclude} ../../examples/%s" % name,
                ":language: python", ":lines: %d-" % (_after_docstring(source) + 1), "```", "",
                "## Output", "", "```{literalinclude} %s.txt" % stem, ":language: text", "```", ""]
        with open(os.path.join(GALLERY, stem + ".md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(page))
        toc.append(stem)
    index = ["# Examples", "",
             "Worked examples on real data. Every dataset ships with scikit-learn, statsmodels, "
             "semopy or statsformula itself (`sf.load_data()`), so each script runs as is. The "
             "pages are generated when the documentation is built, and the output shown is the "
             "scripts' actual output. The test suite runs the same scripts.", "",
             "```{toctree}", ":maxdepth: 1", ""] + toc + ["```", ""]
    with open(os.path.join(GALLERY, "index.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(index))


def _after_docstring(source):
    mod = ast.parse(source)
    if mod.body and isinstance(mod.body[0], ast.Expr) and isinstance(getattr(mod.body[0], "value", None), ast.Constant):
        return mod.body[0].end_lineno + 1
    return 0


def setup(app):
    app.connect("builder-inited", _build_gallery)

# case-insensitive file systems: formula/Formula, terms/Terms, as_formula/as_Formula
autosummary_filename_map = {
    "statsformula.formula": "statsformula.formula-function",
    "statsformula.as_formula": "statsformula.as_formula-function",
    "statsformula.terms": "statsformula.terms-function",
}
# scikit-learn's inherited docstrings reference names that are ambiguous here
suppress_warnings = ["ref.python"]
