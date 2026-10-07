"""Figure 6 -- reconstruction error against redshift, and why it has that shape.

Six equal-count redshift bins (~95 galaxies each), plotted at each bin's median
redshift with bootstrap error bars.  Equal-count rather than equal-width: the
sample runs from z = 0.31 to 13.86 and is very far from uniform, so equal-width
bins put almost everything in the first two and leave the rest reporting the
error of a handful of galaxies.  Median rather than midpoint, because the top
bin spans z = 5.0-13.9 while three quarters of its galaxies lie below z = 7.1:
drawn at its midpoint (z = 9.4) it stretched the curves flat.

Three panels, because raw MAE by itself answers the wrong question:

(a) MAE, with the *reference noise floor* -- the MAE a perfect reconstruction
    would score against the noisy grating spectrum, ``sqrt(2/pi) * sigma``
    averaged over valid pixels.  The fluxes are z-scored per spectrum, so as
    galaxies get fainter at high z the reference's own noise takes a growing
    share of its unit variance, and every method's error rises with it.
(b) Scale-free MAE, the Table 1 metric, which shrinkage cannot improve.
(c) The amplitude ratio, sigma_method / sigma_HR over the bin's valid pixels.

Together they show that the gap between SR2 and the classical cluster in (a) is
the amplitude gap in (c), not a reconstruction advantage: in (b) it closes.

Every quantity uses the ``valid`` mask and the per-spectrum averaging of
Table 1.  An earlier version pooled all pixels including the 3% where the
reference is padding, and so disagreed with the numbers the text quotes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths, style
from ..data import load_cache
from ..methods import LINES, NON_HR, registry
from ..metrics import bootstrap_std, mae_by_spectrum, mae_scalefree_by_spectrum, std_ratio

N_ZBINS = 6
#: Lines closer than this to either end of the grid are not counted as inside
#: it -- the same margin ``Cache.line_positions`` uses.
EDGE_MARGIN_UM = 0.05


def bin_index(z: np.ndarray, n_bins: int = N_ZBINS) -> tuple[np.ndarray, np.ndarray]:
    """``(edges, index)`` for equal-count bins; the top edge is inclusive."""
    z = np.asarray(z, dtype=np.float64)
    edges = np.percentile(z, np.linspace(0, 100, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, n_bins - 1)
    return edges, idx


def line_count_changes(wl_lo: float, wl_hi: float,
                       merge_dz: float = 0.15) -> list[tuple[float, str]]:
    """Redshifts at which one of the four diagnostic lines enters or leaves.

    Changes closer than ``merge_dz`` share a marker ([O III] and H-beta enter
    within 0.06 of each other).
    """
    events = []
    for _key, label, rest in LINES:
        name = label.split(r" $\lambda")[0]
        events.append(((wl_lo + EDGE_MARGIN_UM) / rest - 1.0, name, "in"))
        events.append(((wl_hi - EDGE_MARGIN_UM) / rest - 1.0, name, "out"))
    events.sort()
    out: list[tuple[float, str]] = []
    group = [events[0]]
    for ev in events[1:] + [None]:
        if ev is not None and ev[0] - group[0][0] < merge_dz and ev[2] == group[0][2]:
            group.append(ev)
            continue
        out.append((group[0][0], ", ".join(g[1] for g in group) + f" {group[0][2]}"))
        group = [ev] if ev is not None else []
    return out


def compute(cache, n_bins: int = N_ZBINS) -> dict:
    """Per-bin statistics for every method, and the reference noise floor."""
    z = np.asarray(cache.z, dtype=np.float64)
    x, valid = cache.x_high, cache.valid
    edges, idx = bin_index(z, n_bins)

    per_mae = {k: mae_by_spectrum(cache.arrays[k], x, valid) for k in NON_HR}
    per_sf = {k: mae_scalefree_by_spectrum(cache.arrays[k], x, valid) for k in NON_HR}
    err = np.where(valid, cache.x_high_err, np.nan)
    per_floor = np.sqrt(2.0 / np.pi) * np.nanmean(err, axis=1)
    per_zero = np.nanmean(np.abs(np.where(valid, x, np.nan)), axis=1)

    res = {"edges": edges, "counts": [], "z_median": [],
           "labels": [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(edges[:-1], edges[1:])],
           "mae": {k: [] for k in NON_HR}, "mae_err": {k: [] for k in NON_HR},
           "sf": {k: [] for k in NON_HR}, "sf_err": {k: [] for k in NON_HR},
           "amp": {k: [] for k in NON_HR}, "floor": [], "zero": []}
    for b in range(n_bins):
        s = idx == b
        res["counts"].append(int(s.sum()))
        res["z_median"].append(float(np.median(z[s])))
        res["floor"].append(float(np.mean(per_floor[s])))
        res["zero"].append(float(np.mean(per_zero[s])))
        for k in NON_HR:
            res["mae"][k].append(float(np.mean(per_mae[k][s])))
            res["mae_err"][k].append(bootstrap_std(per_mae[k][s]))
            res["sf"][k].append(float(np.mean(per_sf[k][s])))
            res["sf_err"][k].append(bootstrap_std(per_sf[k][s]))
            res["amp"][k].append(std_ratio(cache.arrays[k][s], x[s], valid[s]))
    res["line_changes"] = line_count_changes(float(cache.wl_high.min()),
                                             float(cache.wl_high.max()))
    return res


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
    r = compute(cache)
    zm, edges = np.asarray(r["z_median"]), r["edges"]
    x_max = 8.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2.2, 1.4]})
    for ax in axes:
        for b in range(len(zm)):                       # the bins themselves
            if b % 2 == 0:
                ax.axvspan(edges[b], min(edges[b + 1], x_max), color="0.93", lw=0, zorder=0)
        for zc, _text in r["line_changes"]:            # where a diagnostic line enters/leaves
            if 0 < zc < x_max:
                ax.axvline(zc, color="0.55", ls=":", lw=0.9, zorder=1)

    panels = [(axes[0], "mae", "mae_err", "MAE (normalized flux)"),
              (axes[1], "sf", "sf_err", "Scale-free MAE")]
    for ax, key, ekey, ylabel in panels:
        for k in NON_HR:
            ax.errorbar(zm, r[key][k], yerr=r[ekey][k], marker="o", ms=4, lw=1.4,
                        capsize=2, color=reg[k].color, label=reg[k].label, zorder=3)
        ax.set_ylabel(ylabel)
    axes[0].plot(zm, r["floor"], color="k", ls="--", lw=1.4, zorder=2,
                 label="Reference noise floor")
    axes[0].plot(zm, r["zero"], color="0.45", ls=":", lw=1.6, zorder=2,
                 label="All-zero spectrum")
    axes[0].legend(fontsize=8, ncol=3, loc="lower right")

    for k in NON_HR:
        axes[2].plot(zm, r["amp"][k], marker="o", ms=4, lw=1.4, color=reg[k].color)
    axes[2].axhline(1.0, color="k", lw=0.8, ls="-", zorder=1)
    axes[2].set_ylabel(r"$\sigma_\mathrm{method}/\sigma_\mathrm{HR}$")
    axes[2].set_ylim(0, 1.3)
    axes[2].set_xlabel("Redshift (median of each bin)")
    axes[2].set_xlim(0, x_max)

    for ax, letter in zip(axes, "abc"):
        ax.text(0.008, 0.97, f"({letter})", transform=ax.transAxes, va="top",
                ha="left", fontsize=10, fontweight="bold")
    top = axes[0].get_ylim()[1]
    for zc, text in r["line_changes"]:
        if 0 < zc < x_max:
            axes[0].text(zc + 0.04, top, text, rotation=90, va="top", ha="left",
                         fontsize=7, color="0.35")

    plt.tight_layout()
    out = outdir / "fig_redshift_mae.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame({
        "z range": r["labels"], "N": r["counts"], "z med": np.round(zm, 2),
        "floor": np.round(r["floor"], 3),
        **{f"{reg[k].label}": np.round(r["mae"][k], 3) for k in NON_HR},
    })
    print(df.to_string(index=False))
    sf = pd.DataFrame({"z range": r["labels"],
                       **{reg[k].label: np.round(r["sf"][k], 4) for k in NON_HR}})
    print("scale-free MAE\n" + sf.to_string(index=False))
    amp = pd.DataFrame({"z range": r["labels"],
                        **{reg[k].label: np.round(r["amp"][k], 2) for k in NON_HR}})
    print("amplitude ratio\n" + amp.to_string(index=False))
    print(f"Saved → {out}")
    return out
