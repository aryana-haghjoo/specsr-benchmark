"""Figure 4 -- global reconstruction fidelity, five ways.

Five bar panels over the eight methods.  The top row is the paper's central
argument read left to right: raw MAE, where SR2 leads by 30%; the amplitude
ratio, where it sits at 0.54 of the reference's scale; and the scale-free MAE,
where every method is level.  Absolute error against a noisy reference falls
when you shrink toward zero, whatever the reconstruction quality, and the third
panel is the same comparison with that route closed.  The bottom row is MAE
normalised by the reference's own flux uncertainty, and per-spectrum RMSE.

These panels replaced the manuscript's global-fidelity table on 2026-10-02, so
the numbers a reader used to look up there are printed beside the bars of the
amplitude and scale-free panels.  ``tests/test_paper_consistency.py`` checks
:func:`compute` against ``summary_final.csv`` row by row.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths, style
from ..data import load_cache
from ..methods import ORDER, registry
from ..metrics import mae_scalefree_by_spectrum, std_ratio

# Figure 4 spells out that the TV method is Wiener-prefiltered; the other
# panels have less room and write "TV".
LABELS = {"TV": "Wiener + TV"}

#: ``E[|N(0, sigma)| / sigma]`` -- where the uncertainty-normalised panel would
#: sit for a reconstruction that is perfect up to the reference's own noise.
GAUSSIAN_ABS_FLOOR = float(np.sqrt(2.0 / np.pi))

N_BOOT = 1000
SEED = 42


def compute(cache):
    """One row per method, plus the two reference lines the panels draw."""
    reg = registry(label_overrides=LABELS)
    x_high = cache.x_high
    err_safe, floor, mean_unc = cache.x_high_err_floored()
    rng = np.random.default_rng(SEED)

    rows, per_sample_rmse = [], {}
    for key in ORDER:
        arr = cache.arrays[key]
        valid = (np.isfinite(arr) & np.isfinite(x_high)
                 & np.isfinite(err_safe) & (err_safe > 0))
        diff = np.where(valid, arr - x_high, np.nan)
        abs_diff = np.abs(diff)
        unc_abs = abs_diff / err_safe

        mae_by_spec = np.nanmean(abs_diff, axis=1)
        unc_by_spec = np.nanmean(unc_abs, axis=1)
        rmse_by_spec = np.sqrt(np.nanmean(diff ** 2, axis=1))
        per_sample_rmse[key] = rmse_by_spec
        finite = diff[np.isfinite(diff)]

        # Resample spectra, not pixels: pixels within a spectrum are correlated
        # and resampling them reports an error bar several times too small.
        boot = rng.choice(np.arange(len(mae_by_spec)),
                          size=(N_BOOT, len(mae_by_spec)), replace=True)
        # The two diagnostics that tell shrinkage from reconstruction, over the
        # same ``valid`` mask as every other global number in the paper.
        sf_by_spec = mae_scalefree_by_spectrum(arr, x_high, cache.valid)
        rows.append({
            "Method": reg[key].label,
            "MAE": float(np.nanmean(mae_by_spec)),
            "MAE_std": float(np.nanstd(np.nanmean(mae_by_spec[boot], axis=1))),
            "RMSE": float(np.sqrt(np.nanmean(finite ** 2))),
            "RMSE_spec_std": float(np.nanstd(rmse_by_spec)),
            "RMSE_spec_median": float(np.nanmedian(rmse_by_spec)),
            "Bias": float(np.nanmean(finite)),
            "UncNorm_MAE": float(np.nanmean(unc_by_spec)),
            "UncNorm_MAE_std": float(np.nanstd(np.nanmean(unc_by_spec[boot], axis=1))),
            "UncNorm_MAE_med": float(np.nanmedian(np.nanmedian(unc_abs, axis=1))),
            "AmpRatio": std_ratio(arr, x_high, cache.valid),
            "MAE_scalefree": float(np.nanmean(sf_by_spec)),
            "MAE_scalefree_std": float(np.nanstd(np.nanmean(sf_by_spec[boot], axis=1))),
        })
    return rows, per_sample_rmse, floor, mean_unc


def build(cache=None, outdir: Path | None = None) -> Path:
    style.use_agg()
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import pandas as pd

    style.use_paper_style()
    cache = cache or load_cache()
    outdir = Path(outdir) if outdir else paths.figures_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    reg = registry(label_overrides=LABELS)

    print(cache.summary())
    rows, _rmse, floor, mean_unc = compute(cache)

    df = pd.DataFrame(rows)
    print("=== Global reconstruction statistics ===")
    print(df.round(4).to_string(index=False))
    print(f"\nMean normalized flux uncertainty: {mean_unc:.4f}")
    print(f"Gaussian noise-only E[|N(0, sigma)| / sigma]: {GAUSSIAN_ABS_FLOOR:.4f}")
    print(f"1st percentile uncertainty floor: {floor:.4f} normalized flux")

    keys = [k for k in ORDER if k != "HR"]
    by_label = {r["Method"]: r for r in rows}
    take = lambda field: [by_label[reg[k].label][field] for k in keys]  # noqa: E731
    names = [reg[k].label for k in keys]
    colours = [reg[k].color for k in keys]
    y = np.arange(len(keys))

    # Three panels over two, with the lower pair centred under the upper row.
    # Two grids rather than one: in a single grid the lower row's tick labels
    # force a gap between the upper panels that they do not need.  Every panel
    # is the same width, so the lower grid is two panels and one gap wide.
    fig = plt.figure(figsize=(14.8, 8.8))
    left, right, wspace = 0.08, 0.99, 0.08
    width = (right - left) / (3 + 2 * wspace)
    half = width * (2 + wspace) / 2
    mid = (left + right) / 2
    top = fig.add_gridspec(1, 3, left=left, right=right, top=0.92, bottom=0.57,
                           wspace=wspace)
    low = fig.add_gridspec(1, 2, left=mid - half, right=mid + half, top=0.44,
                           bottom=0.09, wspace=wspace)
    ax_mae = fig.add_subplot(top[0])
    ax_amp = fig.add_subplot(top[1], sharey=ax_mae)
    ax_sf = fig.add_subplot(top[2], sharey=ax_mae)
    ax_unc = fig.add_subplot(low[0], sharey=ax_mae)
    ax_rmse = fig.add_subplot(low[1], sharey=ax_mae)
    for unlabelled in (ax_amp, ax_sf, ax_rmse):
        unlabelled.tick_params(labelleft=False)
    bar_kw = dict(color=colours, alpha=0.85, error_kw=dict(elinewidth=0.8, capsize=3))

    def annotate(ax, values, fmt):
        """Print each bar's value at its end: these panels stand in for a table."""
        # A column at the right edge, clear of the bars and their error bars.
        for yi, v in zip(y, values):
            ax.text(0.98, yi, f"{v:{fmt}}", va="center", ha="right", fontsize=8,
                    transform=ax.get_yaxis_transform())

    ax = ax_mae
    ax.barh(y, take("MAE"), xerr=take("MAE_std"), **bar_kw)
    ax.axvline(mean_unc, color="black", lw=0.8, ls="--", alpha=0.75,
               label="Mean flux uncertainty")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("MAE (normalized flux)")
    ax.set_title("(a) Mean Absolute Error")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.invert_yaxis()

    ax = ax_amp
    amps = take("AmpRatio")
    ax.barh(y, amps, **bar_kw)
    ax.axvline(1.0, color="black", lw=0.8, ls="--", alpha=0.75)
    annotate(ax, amps, ".2f")
    ax.set_xlim(0.0, max(amps) * 1.3)
    ax.set_xlabel(r"$\sigma_{\rm method} / \sigma_{\rm HR}$  (dashed: reference)")
    ax.set_title("(b) Amplitude ratio")

    ax = ax_sf
    sfs = take("MAE_scalefree")
    ax.barh(y, sfs, xerr=take("MAE_scalefree_std"), **bar_kw)
    annotate(ax, sfs, ".4f")
    ax.set_xlim(0.0, max(sfs) * 1.3)
    ax.set_xlabel("MAE after optimal rescaling")
    ax.set_title("(c) Scale-free MAE")

    ax = ax_unc
    ax.barh(y, take("UncNorm_MAE"), xerr=take("UncNorm_MAE_std"), **bar_kw)
    ax.axvline(GAUSSIAN_ABS_FLOOR, color="black", lw=0.8, ls="--", alpha=0.75,
               label="Gaussian noise floor")
    ax.set_xlabel(r"Mean $|residual| / \sigma_{flux}$")
    ax.set_title("(d) Uncertainty-normalized MAE")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = ax_rmse
    rmses, rmse_stds = take("RMSE"), take("RMSE_spec_std")
    ax.barh(y, rmses, xerr=rmse_stds, **bar_kw)
    ax.axvline(1.0, color="black", lw=0.8, ls=":", alpha=0.7,
               label="Predict-zero baseline")
    ax.set_xlim(max(0.0, min(rmses) - max(rmse_stds) * 0.3),
                max(rmses) + max(rmse_stds) * 1.05)
    ax.set_xlabel("Mean per-spectrum RMSE")
    ax.set_title("(e) Root Mean Square Error")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.legend(handles=[mpatches.Patch(color=reg[k].color, label=reg[k].label)
                        for k in keys],
               loc="upper center", ncol=len(keys), bbox_to_anchor=(mid, 1.0),
               fontsize=9, frameon=False, handlelength=1.2, handleheight=0.9)

    out = outdir / "fig_mae_summary.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")
    return out
