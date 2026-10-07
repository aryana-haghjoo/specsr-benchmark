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


# ── the release statement (paper, Data Availability) ──────────────────────────
# The paper tells a reader where to get the package.  Those identifiers are
# hand-copied into the manuscript, which is the same shape of mistake as the
# duplicated version above: nothing recomputes them, and a repository rename or
# a new archive would leave the paper pointing at a dead link forever, since a
# published paper cannot be corrected as easily as a released package.  These
# tests assert the statement against what the package itself declares.
#
# Until 0fcc577 this was a full appendix (``sec:software``) that also pinned a
# version and stated the Python floor, the dependencies, the extras and the
# tutorial sample, each with a test here.  The revision cut it down to the
# Data Availability paragraph, and the tests for claims the paper no longer
# makes went with it; both are in git history if the appendix comes back.

RELEASE_SECTION = r"\section*{Data Availability}"


def _paper_tex() -> str:
    paper = REPO / "paper" / "paper.tex"
    if not paper.exists():
        pytest.skip("paper.tex not present")
    return paper.read_text(encoding="utf-8")


def _release_statement() -> str:
    """The Data Availability section alone, line breaks removed.

    Scoped to the section so that a URL quoted elsewhere in the paper cannot
    stand in for the one the reader is actually told to use.
    """
    tex = _paper_tex()
    start = tex.find(RELEASE_SECTION)
    assert start >= 0, (
        "paper.tex no longer carries the Data Availability statement; the "
        "tests below would then silently assert nothing")
    end = tex.find(r"\bibliography", start)
    return tex[start:end if end >= 0 else None].replace("\n", "")


def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib")
    return tomllib.loads((REPO / "pyproject.toml").read_text())["project"]


def test_the_paper_has_a_release_statement():
    _release_statement()


def test_the_release_statement_states_the_declared_repository():
    """The link in the statement must be the one the package declares."""
    repo = _pyproject()["urls"]["Repository"]
    assert repo.rstrip("/") in _release_statement(), (
        f"pyproject declares Repository = {repo}, which the Data Availability "
        "statement does not state; the paper is pointing somewhere else")


def test_the_release_statement_names_the_pypi_project():
    """The install instruction has to name the project that is published."""
    name = _pyproject()["name"]
    assert f"pip install {name}" in _release_statement(), \
        f"the Data Availability statement does not tell the reader to install {name!r}"


def test_the_release_statement_cites_the_concept_doi_that_the_readme_records():
    """The archive DOI is written down in README.md and again in the paper.

    Cite the *concept* DOI, not a version DOI: it resolves to the latest
    release, so it stays correct after the next one, which a version DOI
    printed in a paper does not.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*\[(10\.5281/zenodo\.\d+)\]", readme)
    assert m, "README.md no longer marks which Zenodo DOI is the concept DOI"
    concept = m.group(1)
    assert concept in _release_statement(), (
        f"README records {concept} as the concept DOI; the Data Availability "
        "statement states a different one, or none")


def test_the_release_statement_states_the_license_the_package_ships():
    assert "MIT license" in _release_statement(), \
        "the Data Availability statement does not state the license"
    assert (REPO / "LICENSE").read_text(encoding="utf-8").lstrip().startswith(
        "MIT License"), "LICENSE is no longer MIT, but the paper says it is"
