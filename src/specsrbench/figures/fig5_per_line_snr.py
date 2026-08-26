"""Figure 5 -- per-line behaviour: S/N, detection, false detection, width bias.

Four rows over the four diagnostic lines.  The first two say how much signal a
method reports and how often it reports any; the last two are what stop those
being read as quality.

* **Median S/N**, over the subset where the reference itself detects the line.
* **Detection fraction** at S/N > 5, with the reference's own rate marked.
* **False detection rate** -- how often a method reports S/N > 5 where the
  reference says the line is absent (S/N < 3).  This is the row where SR2
  separates from every classical method: 0.30 on Hbeta and 0.44 on [O II],
  against <= 0.09 for anything classical.
* **FWHM bias** in nanometres against the reference's fitted width.

A method can lead the first row by inventing lines, and only the third row
shows it.  They are drawn together for that reason.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths, style
from ..data import load_cache
from ..methods import LINES, NON_HR, ORDER, registry

#: The reference detects a line above this; below TRUE_ABSENT it says there is
#: none, and anything a method "finds" there is false.
HR_DETECT_THRESH = 5.0
TRUE_ABSENT_THRESH = 3.0
#: Widths are only meaningful where the reference line is solidly detected.
FWHM_MIN_SN = 5.0


def compute(cache) -> dict[str, dict[str, list[float]]]:
    """The four rows, each as ``{method key: [value per line]}``."""
    reg = registry()
    abs_snr, det_frac, fdr, fwhm_bias = {}, {}, {}, {}
    for key in ORDER:
        abs_snr[key], det_frac[key], fdr[key], fwhm_bias[key] = [], [], [], []
        label = reg[key].label
        for lname, _disp, _rest in LINES:
            hr_sn = cache.snr[f"HR_{lname}"]
            sn_m = cache.snr[f"{key}_{lname}"]
            real = np.isfinite(hr_sn) & (hr_sn > HR_DETECT_THRESH)
            sub = sn_m[real & np.isfinite(sn_m)]
            abs_snr[key].append(float(np.median(sub)) if sub.size else np.nan)
            det_frac[key].append(float(np.nanmean(sn_m > HR_DETECT_THRESH)))

            if key == "HR":
                fdr[key].append(np.nan)
                fwhm_bias[key].append(0.0)   # zero by definition
                continue

            fit_sn_hr = cache.fits[f"HR target_{lname}_sn"]
            fit_sn_m = cache.fits[f"{label}_{lname}_sn"]
            absent = np.isfinite(fit_sn_hr) & (fit_sn_hr < TRUE_ABSENT_THRESH)
            valid = absent & np.isfinite(fit_sn_m)
            fdr[key].append(float(np.nanmean(fit_sn_m[valid] > HR_DETECT_THRESH))
                            if valid.sum() else np.nan)

            hr_fwhm = 2.355 * cache.fits[f"HR target_{lname}_sigma"] * 1e3
            m_fwhm = 2.355 * cache.fits[f"{label}_{lname}_sigma"] * 1e3
            hr_sn2 = cache.fits[f"HR target_{lname}_sn"]
            v = np.isfinite(hr_fwhm) & np.isfinite(m_fwhm) & (hr_sn2 > FWHM_MIN_SN)
            fwhm_bias[key].append(float(np.nanmedian(m_fwhm[v] - hr_fwhm[v])))
    return {"abs_snr": abs_snr, "det_frac": det_frac,
            "fdr": fdr, "fwhm_bias": fwhm_bias}


def build(cache=None, outdir: Path | None = None) -> Path:
    style.use_agg()
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    style.use_paper_style()
    cache = cache or load_cache()
    outdir = Path(outdir) if outdir else paths.figures_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    reg = registry()

    print(cache.summary())
    print(f"Loaded fit_data: {len(cache.fits)} arrays")
    d = compute(cache)

    methods_show = list(NON_HR) + ["HR"]
    rows = [d["abs_snr"], d["det_frac"], d["fdr"], d["fwhm_bias"]]
    row_labels = [
        f"Median S/N\n(HR S/N > {HR_DETECT_THRESH} subset)",
        f"Detection fraction\n(S/N > {HR_DETECT_THRESH})",
        f"False detection rate\n(HR S/N < {TRUE_ABSENT_THRESH})",
        "FWHM bias (nm)",
    ]
    x_limits = [None, (0, 1.05), (0, 0.75), None]

    for li, (_lname, disp, _r) in enumerate(LINES):
        print(f"  {disp:24s} median S/N  SR2 {d['abs_snr']['SR2'][li]:8.1f}"
              f"   HR {d['abs_snr']['HR'][li]:8.1f}"
              f"   FDR SR2 {d['fdr']['SR2'][li]:.3f}")

    fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharey="row")
    for row_i, (data, ylabel, xlim) in enumerate(zip(rows, row_labels, x_limits)):
        for li, (_lname, ldisplay, _rest) in enumerate(LINES):
            ax = axes[row_i, li]
            vals = [data[m][li] for m in methods_show]
            colours = [reg[m].color for m in methods_show]
            # The reference has no meaningful false-detection rate or width
            # bias against itself; leave those bars off rather than at zero.
            if row_i in (2, 3) and not np.isnan(vals[-1]):
                vals[-1] = np.nan
            ax.barh(np.arange(len(methods_show)), vals, color=colours, alpha=0.85)
            ax.set_yticks(np.arange(len(methods_show)))
            ax.set_yticklabels([reg[m].label for m in methods_show] if li == 0
                               else [""] * len(methods_show))
            if row_i == 0:
                ax.set_title(ldisplay)
            if row_i == 1:
                ax.axvline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
            if xlim:
                ax.set_xlim(*xlim)
            ax.set_xlabel(ylabel.replace("\n", " "))
            ax.invert_yaxis()
        axes[row_i, 0].set_ylabel(ylabel)

    fig.legend(handles=[mpatches.Patch(color=reg[m].color, label=reg[m].label)
                        for m in methods_show],
               loc="upper center", ncol=len(methods_show),
               bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False,
               handlelength=1.2, handleheight=0.9)

    plt.tight_layout()
    out = outdir / "fig_jades_per_line_snr.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
    return out
