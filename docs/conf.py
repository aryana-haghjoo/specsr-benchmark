"""Sphinx configuration for the specsrbench documentation."""

from __future__ import annotations

import importlib.metadata

project = "specsrbench"
author = "Aryana Haghjoo"
copyright = "2026, Aryana Haghjoo"

try:
    release = importlib.metadata.version("specsrbench")
except importlib.metadata.PackageNotFoundError:  # docs built from a bare checkout
    release = "0.0.0.dev0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# Generate stub pages for the API reference automatically.
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Heavy or optional imports must not block a docs build. `specsr` and torch are
# needed only by the ML arm, and pywt/skimage only by two of the deconvolvers;
# none of them is needed to read a docstring.
autodoc_mock_imports = [
    "torch",
    "specsr",
    "huggingface_hub",
    "pywt",
    "skimage",
    "astropy",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Render "Attributes" as :ivar: fields. Without this, napoleon emits
# .. attribute:: directives that collide with the annotations autodoc already
# documents on a dataclass, producing duplicate-description warnings.
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

templates_path = ["_templates"]
# GUARDS.md is pulled into guides/guards.md with `{include}`, and the README
# links to it on GitHub. Excluding it from the source list keeps it doing both
# jobs without also becoming a second, orphan copy of the same page.
#
# The `*_STATE.md` pattern covers development notes that live in docs/ in a
# working checkout but are not part of the site. Matching a pattern rather than
# a filename means a local build produces the same site as a released one, and
# a new note does not silently become a page.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store",
                    "*_STATE.md", "GUARDS.md"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist"]

html_theme = "furo"
html_title = f"specsrbench {version}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/aryana-haghjoo/specsr-benchmark/",
    "source_branch": "main",
    "source_directory": "docs/",
}
