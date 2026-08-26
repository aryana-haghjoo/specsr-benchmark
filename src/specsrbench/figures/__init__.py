"""One module per paper figure, and the registry the CLI dispatches through.

Each module exposes ``build(cache=None, outdir=None) -> Path``: it draws its
figure, prints the numbers the paper quotes from it, and returns the path it
wrote.  Nothing else in the package imports one figure from another, so a
figure can be rebuilt on its own without dragging the rest of the suite in.

These were six Jupyter notebooks until 2026-08-25.  Notebooks were a poor fit
for a product that has to rebuild identically: their outputs are committed
alongside their code, so a stale number can sit in a diff looking like data;
they cannot be imported, so five of the six carried a copied ``load()`` helper
and colour table; and "run all the figures" meant driving ``nbconvert`` and
parsing stdout to find out whether it had worked.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

__all__ = ["REGISTRY", "build", "build_all", "FIGURE_FILES"]

#: figure name -> (module, output filename).  The name is what the CLI takes.
REGISTRY: dict[str, tuple[str, str]] = {
    "toy":          ("fig1_toy_methods",   "fig_toy_1d.pdf"),
    "qualitative":  ("fig2_qualitative",   "fig_jades_qualitative.pdf"),
    "residuals":    ("fig3_residual_maps", "fig_residual_maps.pdf"),
    "mae":          ("fig4_mae_summary",   "fig_mae_summary.pdf"),
    "per-line-snr": ("fig5_per_line_snr",  "fig_jades_per_line_snr.pdf"),
    "redshift":     ("fig6_redshift_mae",  "fig_redshift_mae.pdf"),
}

#: Paper figure number -> registry name, for ``specsrbench figures 4``.
BY_NUMBER: dict[str, str] = {
    "1": "toy", "2": "qualitative", "3": "residuals",
    "4": "mae", "5": "per-line-snr", "6": "redshift",
}

FIGURE_FILES: tuple[str, ...] = tuple(f for _, f in REGISTRY.values())


def resolve(name: str) -> str:
    """Accept a registry name, a paper figure number, or ``fig4``."""
    n = name.strip().lower().removeprefix("figure").removeprefix("fig").strip("_- ")
    if n in REGISTRY:
        return n
    if name.strip().lower() in REGISTRY:
        return name.strip().lower()
    if n in BY_NUMBER:
        return BY_NUMBER[n]
    raise KeyError(
        f"unknown figure {name!r}; choose from "
        f"{', '.join(REGISTRY)} or 1-{len(BY_NUMBER)}")


def build(name: str, cache=None, outdir: Path | None = None) -> Path:
    """Build one figure by name or number."""
    key = resolve(name)
    module, _ = REGISTRY[key]
    mod = import_module(f".{module}", __package__)
    return mod.build(cache=cache, outdir=outdir)


def build_all(cache=None, outdir: Path | None = None,
              skip: tuple[str, ...] = ()) -> dict[str, Path]:
    """Build every figure, in paper order, returning what was written."""
    out: dict[str, Path] = {}
    for number, key in BY_NUMBER.items():
        if key in skip:
            continue
        print(f"\n{'=' * 78}\nFigure {number}: {key}\n{'=' * 78}")
        out[key] = build(key, cache=cache, outdir=outdir)
    return out
