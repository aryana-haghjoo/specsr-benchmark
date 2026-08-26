"""The one matplotlib style every figure in the paper uses.

Each notebook opened with the same 24-line ``rcParams`` block, copied by hand.
They had not drifted, but nothing was stopping them: a figure whose tick
direction quietly differs from its neighbours' is not something a test catches
and not something a reader can unsee.
"""
from __future__ import annotations

import matplotlib

__all__ = ["PAPER_RC", "use_paper_style", "use_agg"]

PAPER_RC: dict[str, object] = {
    "figure.dpi":       110,
    "savefig.dpi":      300,
    "font.size":        10,
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "axes.edgecolor":   "black",
    "axes.linewidth":   0.8,
    "axes.labelcolor":  "black",
    "axes.titlesize":   10,
    "axes.labelsize":   10,
    "xtick.color":      "black",
    "ytick.color":      "black",
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":        True,
    "ytick.right":      True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
    "legend.frameon":   False,
}


def use_agg() -> None:
    """Select the non-interactive backend, before ``pyplot`` is imported.

    Figure building is a batch job; on a headless machine an interactive
    backend fails at import, which is a confusing way to be told there is no
    display.
    """
    matplotlib.use("Agg")


def use_paper_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_RC)
