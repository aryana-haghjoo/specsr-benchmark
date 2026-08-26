"""Figure 6 -- reconstruction error against redshift.

MAE in six equal-count redshift bins, with the population per bin beneath.
Equal-count rather than equal-width: the sample runs from z = 0.31 to 13.86 and
is very far from uniform, so equal-width bins put almost everything in the
first two and leave the rest reporting the error of a handful of galaxies.

What the panel is for is the degeneracy between redshift and difficulty.  Lines
move across the detector with redshift, and the four diagnostic lines leave the
grid entirely at the top of the range, so a method that looks better at high z
may only be being scored on continuum.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths, style
from ..data import load_cache
from ..methods import NON_HR, registry

N_ZBINS = 6


def compute(cache, n_bins: int = N_ZBINS):
    """``(bin labels, counts, {method: MAE per bin}, bin midpoints, edges)``."""
    z = np.asarray(cache.z)
    edges = np.percentile(z, np.linspace(0, 100, n_bins + 1))
    mids = 0.5 * (edges[:-1] + edges[1:])
    labels = [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(edges[:-1], edges[1:])]

    x_high = cache.x_high
    mae_by_z = {k: [] for k in NON_HR}
    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (z >= lo) & (z < hi)
        counts.append(int(mask.sum()))
        for k in NON_HR:
            diff = (cache.arrays[k][mask] - x_high[mask]).ravel()
            diff = diff[np.isfinite(diff)]
            mae_by_z[k].append(float(np.mean(np.abs(diff))) if diff.size else np.nan)
    return labels, counts, mae_by_z, mids, edges


def build(cache=None, outdir: Path | None = None) -> Path:
    style.use_agg()
    import matplotlib.pyplot as plt
    import pandas as pd

    style.use_paper_style()
    cache = cache or load_cache()
    outdir = Path(outdir) if outdir else paths.figures_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    reg = registry(include_hr=False)

    print(cache.summary())
    labels, counts, mae_by_z, mids, edges = compute(cache)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                   gridspec_kw={"height_ratios": [4, 1]})
    for k in NON_HR:
        ax1.plot(mids, mae_by_z[k], marker="o", label=reg[k].label,
                 color=reg[k].color, lw=1.5, ms=5)
    ax1.set_ylabel("MAE (normalised flux)")
    ax1.legend(fontsize=8)
    ax1.set_xticks(mids)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    ax2.bar(mids, counts, width=np.diff(edges) * 0.85, color="steelblue", alpha=0.7)
    ax2.set_ylabel("N spectra")
    ax2.set_xlabel("Redshift range")
    ax2.set_xticks(mids)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    out = outdir / "fig_redshift_mae.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame({
        "z range": labels, "N": counts,
        **{reg[k].label: [round(v, 5) for v in vals] for k, vals in mae_by_z.items()},
    })
    print(df.to_string(index=False))
    print(f"Saved → {out}")
    return out
