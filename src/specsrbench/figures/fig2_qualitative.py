"""Figure 2 -- one held-out galaxy, every method, with an [O III] inset.

The example is chosen on a property of the *data*, not of any result: a
redshift high enough that the [O III] doublet is physically resolvable.  4959
and 5007 are separated by 4.933 nm x (1+z), so at z = 6.55 they sit 37.2 nm
apart against an LSF FWHM of ~21 nm at 3.78 um.  Below z ~ 5 they fall inside
one resolution element, where no method can separate them -- the prism input
resolves the doublet in 0% of z = 0-2 galaxies, 0.5% of z = 2-4, 27% of z = 4-6
and 100% of z > 6.  Picking an unresolvable case credits a method for
recovering structure its own input does not carry.

Favourable to SR2 but not cherry-picked: among the 296 held-out spectra with
reference [O III] S/N > 20 and the doublet on the grid, SR2's RMSE gain here
ranks 44th of 296 -- the 85th percentile, 22.9% against a subset median of
12.4%.  SR2 still inflates the [O III] S/N to 466 against 88 in the reference,
at an amplitude ratio of 0.27 where the classical methods sit at 0.93-1.15.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths, style
from ..data import load_cache
from ..methods import NON_HR, registry

#: Figure 2 names the prism baseline by what it is (an interpolation of the
#: low-resolution input) and spells out TV's Wiener prefilter.
LABELS = {"LR": "LR (cubic)", "TV": "Wiener + TV"}

#: The held-out spectrum drawn.  See the module docstring for why this one.
I_SHOW = 188

#: Half-width of the [O III] inset, in microns.
ZOOM_WIN = 0.07

SHORT_NAMES = {"Halpha": r"H$\alpha$", "OIII5007": "[OIII]",
               "Hbeta": r"H$\beta$", "OII3727": "[OII]"}

PANEL_LABELS = "abcdefgh"


def build(cache=None, outdir: Path | None = None, i_show: int = I_SHOW) -> Path:
    style.use_agg()
    import matplotlib.pyplot as plt

    style.use_paper_style()
    cache = cache or load_cache()
    outdir = Path(outdir) if outdir else paths.figures_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    reg = registry(label_overrides=LABELS, include_hr=False)

    print(cache.summary())
    wl = cache.wl_high
    z_show = float(cache.z[i_show])
    hr = cache.x_high[i_show]
    print(f"Showing test spectrum #{i_show}, z={z_show:.2f}")

    panels = [(k, cache.arrays[k][i_show]) for k in NON_HR]

    # Nothing may be clipped.  A percentile ceiling cut the [O III] core off at
    # 3.0 in normalised flux when its true height here is ~24, hiding both the
    # real line and SR2's recovery of it.  Scale from the tallest trace across
    # the reference and all eight reconstructions, so the panels share a limit.
    traces = [hr] + [a for _, a in panels]
    y_max = float(max(np.nanmax(t) for t in traces)) * 1.5
    y_min = -0.4

    zoom_center = 0.5007 * (1.0 + z_show)
    zoom_lo, zoom_hi = zoom_center - ZOOM_WIN, zoom_center + ZOOM_WIN
    mask_zoom = (wl >= zoom_lo) & (wl <= zoom_hi)
    y_ins_max = float(max(np.nanmax(t[mask_zoom]) for t in traces)) * 1.06
    y_ins_min = -0.3

    line_obs = sorted([(SHORT_NAMES[k], lam)
                       for k, _lab, lam in cache.line_positions(i_show)],
                      key=lambda t: t[1])

    fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True, sharey=True)
    for idx, (ax, (key, recon)) in enumerate(zip(axes.flat, panels)):
        rmse = float(np.sqrt(np.mean((recon - hr) ** 2)))
        is_sr2 = key == "SR2"

        for _lshort, lam_obs in line_obs:
            ax.axvline(lam_obs, color="0.62", lw=0.65, ls="--", alpha=0.70, zorder=1)
        ax.axvspan(zoom_lo, zoom_hi, color="0.70", alpha=0.28, zorder=0, lw=0)
        ax.axhline(0, color="0.80", lw=0.4, zorder=0)
        ax.plot(wl, hr, color="0.20", lw=0.55, alpha=0.65, zorder=3)
        ax.plot(wl, recon, color=reg[key].color, lw=0.82, alpha=0.92, zorder=4)

        # symlog keeps the faint continuum readable while showing the line
        # cores at full height: linear below 1.0, logarithmic above.
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(wl.min(), wl.max())
        for spine in ax.spines.values():
            spine.set_edgecolor("0.30")
            spine.set_linewidth(0.7)
        if is_sr2:
            ax.set_facecolor((1.00, 0.97, 0.93))

        axins = ax.inset_axes([0.52, 0.54, 0.46, 0.43])
        axins.plot(wl[mask_zoom], hr[mask_zoom], color="0.20", lw=0.80,
                   alpha=0.75, zorder=3)
        axins.plot(wl[mask_zoom], recon[mask_zoom], color=reg[key].color,
                   lw=1.0, alpha=0.95, zorder=4)
        axins.axvline(zoom_center, color="0.65", lw=0.6, ls=":", alpha=0.8, zorder=1)
        axins.axhline(0, color="0.80", lw=0.4, zorder=0)
        axins.set_xlim(zoom_lo, zoom_hi)
        axins.set_ylim(y_ins_min, y_ins_max)
        axins.set_xticks([zoom_lo + 0.03, zoom_center, zoom_hi - 0.03])
        axins.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.2f"))
        axins.tick_params(labelsize=6.5, pad=1.5, length=2.5)
        # Keep the y ticks: the peak height is the point of the inset.
        axins.set_yticks([0, round(y_ins_max * 0.5), round(y_ins_max * 0.9)])
        axins.tick_params(axis="y", labelsize=6.0, pad=1.0, length=2.0)
        axins.text(0.96, 0.94, "[OIII]", transform=axins.transAxes, fontsize=8,
                   va="top", ha="right", color="0.35")
        for spine in axins.spines.values():
            spine.set_edgecolor("0.40")
            spine.set_linewidth(0.7)
        axins.set_facecolor("white")

        ax.set_title(reg[key].label, color="black", pad=4,
                     fontweight="bold" if is_sr2 else "normal")
        ax.text(0.018, 0.975, f"({PANEL_LABELS[idx]})", transform=ax.transAxes,
                fontweight="bold", va="top", ha="left", color="0.20")
        ax.text(0.018, 0.870, f"RMSE = {rmse:.3f}", transform=ax.transAxes,
                va="top", ha="left", color="0.40")

        # Line labels on panel (a) only.  The inset occupies y = 0.54-0.97, so
        # the tags sit at 0.47/0.37 -- below it, above the continuum noise.
        if idx == 0:
            for j, (lshort, lam_obs) in enumerate(line_obs):
                ax.text(lam_obs, [0.47, 0.37][j % 2], lshort,
                        transform=ax.get_xaxis_transform(), fontsize=7,
                        ha="center", va="center", color="0.35",
                        bbox=dict(fc="white", ec="0.75", boxstyle="round,pad=0.4",
                                  linewidth=0.5, alpha=0.95))

    for ax in axes[-1, :]:
        ax.set_xlabel("λ (μm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Flux (Normalized)")

    fig.tight_layout(h_pad=0.9, w_pad=0.55)
    out = outdir / "fig_jades_qualitative.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")
    return out
