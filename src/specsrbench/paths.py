"""Where the inputs and outputs live, and how they are found.

Three layers, in the order they are consulted:

1. **Environment variables.**  ``SPECSRBENCH_CACHE``, ``SPECSRBENCH_SETS`` and
   ``SPECSRBENCH_FIGURES`` override everything.  An installed wheel has no repo
   around it, so this is the only mechanism a ``pip install`` user has.
2. **A repo root walked up from the caller.**  A directory is the root if it
   holds ``cache_logR_tuned/``.  Note what is *not* in that test: anything to do
   with the manuscript.  The figures used to require the paper source to sit
   beside the cache, which made every one of them unbuildable anywhere except
   inside one particular checkout.
3. **The current directory.**  So that ``specsrbench figures all`` in a fresh
   clone does the obvious thing.

Directories are *not* created on import.  A module that only reads should fail
with "no such directory" rather than silently make an empty one and then report
a missing file inside it -- the failure mode that sends people looking for a
corrupted cache when the real problem is that they are in the wrong directory.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["repo_root", "cache_dir", "sets_dir", "figures_dir", "describe"]

#: Marks a directory as the project root.  The tuned cache is what every figure
#: reads, so its presence is the property that actually matters.
_ROOT_MARKER = "cache_logR_tuned"


def _walk_up(start: Path) -> Path | None:
    for p in [start, *start.parents]:
        if (p / _ROOT_MARKER).is_dir():
            return p
    return None


def repo_root(start: Path | str | None = None) -> Path:
    """Directory holding ``cache_logR_tuned/``, or the current directory."""
    if (env := os.environ.get("SPECSRBENCH_ROOT")):
        return Path(env).expanduser().resolve()
    here = Path(start or Path.cwd()).resolve()
    found = _walk_up(here)
    if found is not None:
        return found
    # Also try the installed location: a git checkout used in place, where the
    # caller's cwd is somewhere else entirely.
    found = _walk_up(Path(__file__).resolve())
    return found if found is not None else here


def cache_dir() -> Path:
    """The tuned cache every figure reads."""
    if (env := os.environ.get("SPECSRBENCH_CACHE")):
        return Path(env).expanduser().resolve()
    return repo_root() / "cache_logR_tuned"


def sets_dir() -> Path:
    """The paired evaluation/calibration/tuning sets the cache is built from.

    Separate from :func:`cache_dir` because the two have different lifetimes:
    the sets come out of ``specsr`` and the JADES tree and change only when the
    models or the split do, while the cache is rebuilt every time a classical
    parameter moves.
    """
    if (env := os.environ.get("SPECSRBENCH_SETS")):
        return Path(env).expanduser().resolve()
    return repo_root() / "cache_logR"


def figures_dir() -> Path:
    """Where figure PDFs are written."""
    if (env := os.environ.get("SPECSRBENCH_FIGURES")):
        return Path(env).expanduser().resolve()
    return repo_root() / "figures"


def describe() -> str:
    """One line per resolved path, for printing at the top of a run."""
    rows = [("root", repo_root()), ("cache", cache_dir()),
            ("sets", sets_dir()), ("figures", figures_dir())]
    return "\n".join(
        f"  {name:8s} {path}{'' if path.exists() else '   (absent)'}"
        for name, path in rows)
