"""Figure 4 -- global reconstruction fidelity, three ways.

Three bar panels over the nine methods: raw MAE, MAE normalised by the
reference's own flux uncertainty, and per-spectrum RMSE.  The table printed
alongside is Table A2 of the paper, and ``tests/test_paper_consistency.py``
checks the manuscript against it row by row.

Read the MAE panel with the amplitude column beside it.  SR2 leads it by 30%,
and does so by producing a spectrum at 0.54 of the reference's scale: absolute
error against a noisy reference falls when you shrink toward zero, whatever the
reconstruction quality.  ``MAE_scalefree`` in ``summary_final.csv`` is the same
comparison with that route closed, and there SR2 places eighth of nine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths, style
from ..data import load_cache
from ..methods import ORDER, registry

# Figure 4 spells out that the TV method is Wiener-prefiltered; the other
# panels have less room and write "TV".
LABELS = {"TV": "Wiener + TV"}

#: E[|N(0, sigma)| / sigma] -- where the uncertainty-normalised panel would sit
#: for a reconstruction that is perfect up to the reference's own noise.
GAUSSIAN_ABS_FLOOR = float(np.sqrt(2.0 / np.pi))

N_BOOT = 1000
SEED = 42


def compute(cache):
    """The rows of Table A2, plus the two reference lines the panels draw."""
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
    print("=== Table A2: Global Reconstruction Statistics ===")
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

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), sharey=True)

    ax = axes[0]
    ax.barh(y, take("MAE"), xerr=take("MAE_std"), color=colours, alpha=0.85,
            error_kw=dict(elinewidth=0.8, capsize=3))
    ax.axvline(mean_unc, color="black", lw=0.8, ls="--", alpha=0.75,
               label="Mean flux uncertainty")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("MAE (normalized flux)")
    ax.set_title("Mean Absolute Error")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.invert_yaxis()

    ax = axes[1]
    ax.barh(y, take("UncNorm_MAE"), xerr=take("UncNorm_MAE_std"), color=colours,
            alpha=0.85, error_kw=dict(elinewidth=0.8, capsize=3))
    ax.axvline(GAUSSIAN_ABS_FLOOR, color="black", lw=0.8, ls="--", alpha=0.75,
               label="Gaussian noise floor")
    ax.set_xlabel(r"Mean $|residual| / \sigma_{flux}$")
    ax.set_title("Uncertainty-normalized MAE")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[2]
    rmses, rmse_stds = take("RMSE"), take("RMSE_spec_std")
    ax.barh(y, rmses, xerr=rmse_stds, color=colours, alpha=0.85,
            error_kw=dict(elinewidth=0.8, capsize=3))
    ax.axvline(1.0, color="black", lw=0.8, ls=":", alpha=0.7,
               label="Predict-zero baseline")
    ax.set_xlim(max(0.0, min(rmses) - max(rmse_stds) * 0.3),
                max(rmses) + max(rmse_stds) * 1.05)
    ax.set_xlabel("Mean per-spectrum RMSE")
    ax.set_title("Root Mean Square Error")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.legend(handles=[mpatches.Patch(color=reg[k].color, label=reg[k].label)
                        for k in keys],
               loc="upper center", ncol=len(keys), bbox_to_anchor=(0.5, 1.04),
               fontsize=9, frameon=False, handlelength=1.2, handleheight=0.9)

    plt.tight_layout(w_pad=1.4)
    out = outdir / "fig_mae_summary.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")
    return out
