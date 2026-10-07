"""Figure 5 -- per-line behaviour: S/N, line shape and position, detection.

Six rows over the four diagnostic lines, ordered so that what an observer
measures from a line comes first:

* **Median S/N**, over the subset where the reference itself detects the line.
* **FWHM_pred - FWHM_obs**, in nm, against the width fitted to the grating
  reference ("obs").
* **(FWHM_pred - FWHM_obs) / FWHM_obs** -- the same bias as a fraction of the
  reference width, which is what makes the scale plain: the grating lines are
  only ~2-3 nm wide, so the classical reconstructions overestimate widths by
  roughly 4-19 times the true width, SR2 by about a third of it.
* **Line-center offset**, the median ``|v_pred - v_obs|`` in km/s between the
  centre fitted to a reconstruction and the centre fitted to the reference.
  The fit bounds each centre to +-10 nm of its catalog position; a fit at the
  bound found no line there, and where many are (classical H-beta, up to 54%)
  the median offset is a lower limit.  The fraction at the bound is printed.
* **Detection fraction** at S/N > 5, with the reference's own rate marked.
* **False detection rate** -- how often a method reports S/N > 5 where the
  reference says the line is absent (S/N < 3).  This is the row where SR2
  separates from every classical method: 0.30 on Hbeta and 0.44 on [O II],
  against <= 0.09 for anything classical.

A method can lead the first row by inventing lines, and only the last row
shows it.  They are drawn together for that reason.

Widths and centres use spectra where the reference line is detected at
S/N > 5, so that "obs" is a measurement and not a fit to noise.
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
#: Widths and centres are only meaningful where the reference line is solidly
#: detected.
FWHM_MIN_SN = 5.0
#: ``build.lines.fit_gauss`` bounds each centre to +- this of the catalog one.
MU_BOUND_UM = 0.01
C_KMS = 299792.458


def compute(cache) -> dict[str, dict[str, list[float]]]:
    """Every row, each as ``{method key: [value per line]}``.

    Besides the six plotted rows, ``centre_bias`` (the signed median offset,
    km/s) and ``centre_at_bound`` (fraction of centres at the fit bound) are
    returned for the text and the tests.
    """
    reg = registry()
    keys = ("abs_snr", "det_frac", "fdr", "fwhm_bias", "fwhm_rel", "fwhm_ratio",
            "centre_offset", "centre_bias", "centre_at_bound")
    out = {k: {m: [] for m in ORDER} for k in keys}
    z = np.asarray(cache.z, dtype=np.float64)
    for key in ORDER:
        label = reg[key].label
        for lname, _disp, rest in LINES:
            hr_sn = cache.snr[f"HR_{lname}"]
            sn_m = cache.snr[f"{key}_{lname}"]
            real = np.isfinite(hr_sn) & (hr_sn > HR_DETECT_THRESH)
            sub = sn_m[real & np.isfinite(sn_m)]
            out["abs_snr"][key].append(float(np.median(sub)) if sub.size else np.nan)
            out["det_frac"][key].append(float(np.nanmean(sn_m > HR_DETECT_THRESH)))

            if key == "HR":        # nothing to compare the reference with
                for k in keys[2:]:
                    out[k][key].append(np.nan)
                continue

            fit_sn_hr = cache.fits[f"HR target_{lname}_sn"]
            fit_sn_m = cache.fits[f"{label}_{lname}_sn"]
            absent = np.isfinite(fit_sn_hr) & (fit_sn_hr < TRUE_ABSENT_THRESH)
            valid = absent & np.isfinite(fit_sn_m)
            out["fdr"][key].append(float(np.nanmean(fit_sn_m[valid] > HR_DETECT_THRESH))
                                   if valid.sum() else np.nan)

            hr_fwhm = 2.355 * cache.fits[f"HR target_{lname}_sigma"] * 1e3
            m_fwhm = 2.355 * cache.fits[f"{label}_{lname}_sigma"] * 1e3
            hr_mu = cache.fits[f"HR target_{lname}_mu"]
            m_mu = cache.fits[f"{label}_{lname}_mu"]
            v = (np.isfinite(hr_fwhm) & np.isfinite(m_fwhm) & (fit_sn_hr > FWHM_MIN_SN)
                 & np.isfinite(hr_mu) & np.isfinite(m_mu))
            out["fwhm_bias"][key].append(float(np.median(m_fwhm[v] - hr_fwhm[v])))
            out["fwhm_rel"][key].append(
                float(np.median((m_fwhm[v] - hr_fwhm[v]) / hr_fwhm[v])))
            out["fwhm_ratio"][key].append(float(np.median(m_fwhm[v] / hr_fwhm[v])))
            dv = C_KMS * (m_mu[v] - hr_mu[v]) / hr_mu[v]
            out["centre_offset"][key].append(float(np.median(np.abs(dv))))
            out["centre_bias"][key].append(float(np.median(dv)))
            mu0 = rest * (1.0 + z[v])
            out["centre_at_bound"][key].append(
                float(np.mean(np.abs(m_mu[v] - mu0) > 0.99 * MU_BOUND_UM)))
    return out


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
    rows = [
        ("abs_snr", f"Median S/N\n(HR S/N > {HR_DETECT_THRESH:g})", None, None),
        ("fwhm_bias", r"$\mathrm{FWHM_{pred}} - \mathrm{FWHM_{obs}}$ (nm)", None, None),
        ("fwhm_rel", r"$(\mathrm{FWHM_{pred}} - \mathrm{FWHM_{obs}})\,/\,\mathrm{FWHM_{obs}}$",
         None, None),
        ("centre_offset", r"Line-center offset" "\n" r"$|v_\mathrm{pred} - v_\mathrm{obs}|$ (km/s)",
         None, None),
        ("det_frac", f"Detection fraction\n(S/N > {HR_DETECT_THRESH:g})", (0, 1.05), 1.0),
        ("fdr", f"False detection rate\n(HR S/N < {TRUE_ABSENT_THRESH:g})", (0, 0.75), None),
    ]

    for li, (_lname, disp, _r) in enumerate(LINES):
        print(f"  {disp:24s} median S/N  SR2 {d['abs_snr']['SR2'][li]:8.1f}"
              f"   HR {d['abs_snr']['HR'][li]:8.1f}"
              f"   FDR SR2 {d['fdr']['SR2'][li]:.3f}")
    print("  FWHM bias / FWHM_obs, centre offset (km/s), at bound, per line:")
    for m in NON_HR:
        cells = "  ".join(f"{d['fwhm_rel'][m][i]:5.2f} {d['centre_offset'][m][i]:5.0f}"
                          f" {100 * d['centre_at_bound'][m][i]:3.0f}%"
                          for i in range(len(LINES)))
        print(f"    {reg[m].label:12s} {cells}")

    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 15.5), sharey="row")
    ypos = np.arange(len(methods_show))
    for row_i, (key, label, xlim, ref) in enumerate(rows):
        for li, (_lname, ldisplay, _rest) in enumerate(LINES):
            ax = axes[row_i, li]
            vals = [d[key][m][li] for m in methods_show]
            ax.barh(ypos, vals, color=[reg[m].color for m in methods_show], alpha=0.85)
            if ref is not None:
                ax.axvline(ref, color="black", lw=0.8, ls="--", alpha=0.6)
            if xlim:
                ax.set_xlim(*xlim)
            if row_i == 0:
                ax.set_title(ldisplay)
            ax.set_xlabel(label.replace("\n", " "), fontsize=9)
            # Shared y: label the first column only.  Setting blank labels on
            # the others used to blank the shared formatter, and with it every
            # method name in the figure.
            ax.set_yticks(ypos)
            ax.tick_params(axis="y", labelleft=(li == 0))
        axes[row_i, 0].set_yticklabels([reg[m].label for m in methods_show], fontsize=9)
        axes[row_i, 0].invert_yaxis()
        axes[row_i, 0].set_ylabel(label, fontsize=10)

    fig.legend(handles=[mpatches.Patch(color=reg[m].color, label=reg[m].label)
                        for m in methods_show],
               loc="upper center", ncol=len(methods_show),
               bbox_to_anchor=(0.5, 1.015), fontsize=9, frameon=False,
               handlelength=1.2, handleheight=0.9)

    plt.tight_layout()
    out = outdir / "fig_jades_per_line_snr.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
    return out
