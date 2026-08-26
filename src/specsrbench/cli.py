"""``specsrbench`` -- one entry point for every product in this repository.

    specsrbench figures all              # the six paper figures, from the cache
    specsrbench figures 4                # just one
    specsrbench build all                # the cache, from JADES DR4 + the Hub
    specsrbench paths                    # where everything is being looked for

The figure commands need only the committed cache; the build commands need the
raw JADES tree and, for the ML arm, the checkpoints on the Hugging Face Hub.
``specsrbench build`` prints what it is missing rather than guessing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, paths


def _add_figures(sub):
    p = sub.add_parser("figures", help="Build paper figures from the cache.")
    p.add_argument("which", nargs="+",
                   help="'all', a figure number (1-6), or a name "
                        "(toy, qualitative, residuals, mae, per-line-snr, redshift)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="where to write the PDFs (default: figures/)")
    p.add_argument("--cache", type=Path, default=None,
                   help="cache directory to read (default: cache_logR_tuned/)")


def _add_build(sub):
    p = sub.add_parser("build", help="Rebuild the cache the figures read.")
    p.add_argument("stage", nargs="+", help="'all' or one of: "
                   + ", ".join(BUILD_STAGES))
    p.add_argument("--jades-root", type=Path, default=None,
                   help="raw JADES DR4 tree, needed by the 'lsf' stage")
    p.add_argument("--dataset", type=Path, default=None,
                   help="paired dataset built by `specsr build` (predictions stage)")
    p.add_argument("--nproc", type=int, default=None,
                   help="worker processes; this box is shared, so leave headroom")
    p.add_argument("--dry-run", action="store_true",
                   help="print the stages and their inputs, run nothing")


#: Ordered.  Each stage consumes what the one before it wrote.
BUILD_STAGES = ("predictions", "sets", "lsf", "tune", "classical", "lines")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="specsrbench",
        description="Benchmarking deep learning against classical deconvolution "
                    "for galaxy spectral super-resolution.")
    ap.add_argument("--version", action="version", version=f"specsrbench {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="<command>")
    _add_figures(sub)
    _add_build(sub)
    sub.add_parser("paths", help="Print the directories that will be used.")

    args = ap.parse_args(argv)
    if args.command is None:
        ap.print_help()
        return 2

    if args.command == "paths":
        print(paths.describe())
        return 0

    if args.command == "figures":
        from . import figures
        names = list(figures.REGISTRY) if "all" in args.which else args.which
        cache = None
        if args.cache is not None:
            from .data import load_cache
            cache = load_cache(args.cache)
        written = []
        for name in names:
            written.append(figures.build(name, cache=cache, outdir=args.outdir))
        print(f"\n{len(written)} figure(s) written:")
        for w in written:
            print(f"  {w}")
        return 0

    if args.command == "build":
        from .build import run_stage
        stages = list(BUILD_STAGES) if "all" in args.stage else args.stage
        unknown = [s for s in stages if s not in BUILD_STAGES]
        if unknown:
            print(f"unknown stage(s): {', '.join(unknown)}\n"
                  f"choose from: {', '.join(BUILD_STAGES)}", file=sys.stderr)
            return 2
        for stage in stages:
            run_stage(stage, args)
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
