"""Rebuilding the cache the figures read, from JADES DR4 and the Hub.

Six stages, each consuming what the one before it wrote::

    predictions   paired dataset + Hub checkpoints -> ML predictions
    sets          predictions -> eval / calib / tune sets, galaxy-disjoint
    lsf           raw JADES x1d + line fits       -> the measured LSF kernel
    tune          tune set + kernel               -> classical_params.json
    classical     eval set + kernel + parameters  -> the classical caches
    lines         every reconstruction            -> Gaussian fits, S/N, summary

Only ``lsf`` needs the raw JADES tree, and only ``predictions`` needs ``torch``
and network access to the Hub; the rest run on what the earlier stages wrote.

The order matters in a way that is easy to get wrong.  ``lsf`` reads the line
fits that ``lines`` writes, so the very first build of a fresh tree runs
``lines`` once against the shipped kernel, then ``lsf``, then ``tune`` ->
``classical`` -> ``lines`` again on the measured one.  ``classical`` refuses to
write a cache built with the shipped kernel, which is what stops that first
pass being mistaken for a finished one.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np

__all__ = ["STAGES", "run_stage", "require_npz"]


def require_npz(path: Path, produced_by: str):
    """Load an input, or say which stage produces it.

    A build stage that cannot find its input is the normal state of a fresh
    clone, not a corruption.  Saying so, and naming the command that fixes it,
    is the difference between a one-line fix and an afternoon spent looking for
    a cache that was never there.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"missing {path}\n"
            f"  this is written by:  {produced_by}\n"
            f"  see `specsrbench paths` for where inputs are being looked for")
    return np.load(path, allow_pickle=True)

#: stage name -> module implementing it.
STAGES: dict[str, str] = {
    "predictions": "predictions",
    "sets": "sets",
    "lsf": "lsf",
    "tune": "tune",
    "classical": "classical_cache",
    "lines": "lines",
}


def run_stage(stage: str, args=None) -> int:
    """Run one build stage.  ``args`` is the parsed CLI namespace, if any."""
    mod = import_module(f".{STAGES[stage]}", __package__)
    argv: list[str] = []
    if args is not None:
        if stage == "lsf" and getattr(args, "jades_root", None):
            argv += ["--jades-root", str(args.jades_root)]
        if stage == "predictions" and getattr(args, "dataset", None):
            argv += ["--dataset", str(args.dataset)]
        if getattr(args, "nproc", None):
            argv += ["--nproc", str(args.nproc)]
        if getattr(args, "dry_run", False):
            argv += ["--dry-run"]
    print(f"\n{'=' * 78}\nbuild stage: {stage}\n{'=' * 78}")
    return mod.main(argv)
