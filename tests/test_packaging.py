"""The version is written down twice, so something has to hold them together.

``pyproject.toml`` and ``src/specsrbench/__init__.py`` each carry the version,
and the release procedure bumps both by hand.  Nothing else compares them:
``publish.yml`` checks the *tag* against the version ``python -m build``
produced, which comes from ``pyproject.toml`` alone.  So a bump that misses
``__init__.py`` sails through -- PyPI serves 0.1.2 while every installed copy
reports ``__version__ == "0.1.1"``, permanently, since a released version
cannot be replaced.

That is the quiet kind of wrong this repository keeps re-learning: a number
duplicated by hand, with no test recomputing it.
"""
from __future__ import annotations

import re

import pytest

from conftest import REPO


def _pyproject_version() -> str:
    # tomllib is 3.11+, and the package supports 3.10.  Parsed properly where
    # the parser exists; the regex below is the same question asked of the
    # same line, not a second source of truth.
    tomllib = pytest.importorskip("tomllib")
    return tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]


def _dunder_version() -> str:
    body = (REPO / "src" / "specsrbench" / "__init__.py").read_text()
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', body, re.M)
    assert m, "src/specsrbench/__init__.py declares no __version__"
    return m.group(1)


def test_the_two_recorded_versions_agree():
    assert _pyproject_version() == _dunder_version(), (
        "pyproject.toml and src/specsrbench/__init__.py disagree about the "
        "version; publish.yml only checks the tag against pyproject.toml, so "
        "this would ship a package that misreports its own version")


def test_the_version_is_a_release_number():
    """A tag is ``v<version>``, so the version has to look like one."""
    v = _dunder_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.-]?(?:a|b|rc|dev)\d+)?", v), \
        f"{v!r} is not a version publish.yml can match against a v-tag"


def test_the_installed_package_reports_the_same_version():
    """Importing it must agree with reading it.

    The two functions above read files.  This one asks the package, which is
    what a user gets from ``specsrbench.__version__``.
    """
    import specsrbench

    assert specsrbench.__version__ == _dunder_version()
