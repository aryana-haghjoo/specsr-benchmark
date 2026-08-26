"""Figure 3 -- residual maps: every held-out spectrum, sorted by redshift.

Each panel is ``method - reference`` for all 572 spectra, stacked with redshift
running up the vertical axis, with the scatter of the residual plotted beneath
against the reference's own noise floor.

The dashed tracks are the loci ``lambda_obs = lambda_rest (1 + z)`` for eight
strong lines, following Figure 5 of Paper 1.  They are what makes the map
readable: a diagonal ridge that follows a track is a real line the method is
getting wrong, and one that does not is instrumental.  Without them every ridge
is unlabelled and could be either.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths, style
from ..data import load_cache
from ..methods import NON_HR, registry

#: Figure 3 draws on a diverging colour map, where deeppink and teal read as
#: data rather than as legend entries.
COLORS = {"LR": "gray", "TV": "mediumseagreen"}

#: Rest wavelengths (um) of the lines traced over each map.
EM_LINE_TRACKS: dict[str, float] = {
    r"Ly$\alpha$": 0.1216, r"[OII]": 0.3727, r"H$\beta$": 0.4861,
    r"[OIII]": 0.5007, r"H$\alpha$": 0.6563, r"HeI": 1.0830,
    r"Pa$\beta$": 1.2818, r"Pa$\alpha$": 1.8751,
}

#: Where along each track its name tag goes, as a fraction of the visible span.
#: Hand-tuned for this panel geometry: Hbeta and [OIII] run ~10 rows apart
#: everywhere, so they can only be separated in wavelength, and [OII] crowds
#: them near the top.
_TRACK_LABEL_FRAC = {
    r"Ly$\alpha$": 0.30, r"[OII]": 0.33, r"H$\beta$": 0.23, r"[OIII]": 0.44,
    r"H$\alpha$": 0.44, r"HeI": 0.50, r"Pa$\beta$": 0.75, r"Pa$\alpha$": 0.65,
}

VMAX = 1.5
N_COLS = 4


def build(cache=None, outdir: Path | None = None) -> Path:
    style.use_agg()
    import matplotlib.gridspec as gridspec
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    style.use_paper_style()
    cache = cache or load_cache()
    outdir = Path(outdir) if outdir else paths.figures_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    reg = registry(color_overrides=COLORS)

    print(cache.summary())
    wl, z = cache.wl_high, np.asarray(cache.z)
    n = cache.n_spectra
    sort_idx = np.argsort(z)
    z_sorted = z[sort_idx]
    hr_s = cache.x_high[sort_idx]

    # The reference's own high-frequency scatter, as the floor each method's
    # residual is judged against.
    hr_hp = cache.x_high - gaussian_filter1d(cache.x_high, sigma=10, axis=1)
    hr_noise_floor = 1.4826 * np.nanmedian(
        np.abs(hr_hp - np.nanmedian(hr_hp, axis=0)), axis=0)

    methods_list = [(k, cache.arrays[k][sort_idx]) for k in NON_HR]
    n_mrows = len(methods_list) // N_COLS

    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(n_mrows * 2, N_COLS, height_ratios=[3, 1] * n_mrows,
                           hspace=0.35, wspace=0.06, left=0.06, right=0.92,
                           top=0.97, bottom=0.06)

    # Label rows by redshift rather than row number, as in Paper 1: the row
    # index is an artefact of the sort, the redshift is the physical axis.
    # With origin='lower' the lowest-z spectrum sits at the bottom and the line
    # tracks run lower-left to upper-right.  (matplotlib's default origin puts
    # row 0 at the top, which flipped every track vertically.)
    ztick_pos, ztick_lab = [], []
    for zval in (0, 1, 2, 3, 4, 5, 10):
        if z_sorted[0] <= zval <= z_sorted[-1]:
            ztick_pos.append(min(int(np.searchsorted(z_sorted, zval)), n - 1))
            ztick_lab.append(str(zval))

    wl_track = np.linspace(float(wl[0]), float(wl[-1]), 300)

    for i, (key, arr) in enumerate(methods_list):
        group, col = i // N_COLS, i % N_COLS
        ax_map = fig.add_subplot(gs[group * 2, col])
        ax_sct = fig.add_subplot(gs[group * 2 + 1, col])

        resid = arr - hr_s
        rsc = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=0)), axis=0)

        ax_map.imshow(resid, aspect="auto", origin="lower", vmin=-VMAX, vmax=VMAX,
                      cmap="RdBu_r", extent=[wl.min(), wl.max(), 0, n])
        ax_map.set_title(reg[key].label, pad=3)
        ax_map.set_xticks([])
        ax_map.set_yticks(ztick_pos)
        if col == 0:
            ax_map.set_yticklabels(ztick_lab)
            ax_map.set_ylabel(r"Redshift $z$", fontsize=9)
        else:
            ax_map.set_yticklabels([])

        for lname, lam_rest in EM_LINE_TRACKS.items():
            z_track = wl_track / lam_rest - 1.0
            idx_track = np.interp(z_track, z_sorted, np.arange(n),
                                  left=-1, right=n + 1)
            vis = (z_track >= z_sorted[0]) & (z_track <= z_sorted[-1])
            if vis.sum() < 2:
                continue
            ax_map.plot(wl_track[vis], idx_track[vis], color="k", lw=0.6,
                        ls="--", alpha=0.5, zorder=5)
            j = int(_TRACK_LABEL_FRAC.get(lname, 0.5) * (vis.sum() - 1))
            # Bold on an opaque white halo: RdBu_r is saturated enough that a
            # plain black tag disappears on the ridge it annotates.
            ax_map.text(wl_track[vis][j], idx_track[vis][j], lname, fontsize=7.5,
                        ha="center", va="bottom", color="black", zorder=6,
                        fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="w")])
        ax_map.set_xlim(wl.min(), wl.max())
        ax_map.set_ylim(0, n)

        ax_sct.plot(wl, rsc, lw=1.0, label=r"$\sigma_\mathrm{resid}$")
        ax_sct.plot(wl, hr_noise_floor, lw=0.9, ls="--", color="gray",
                    label="HR noise floor")
        ax_sct.set_ylim(0, 1.8)
        if group == n_mrows - 1:
            ax_sct.set_xlabel(r"$\lambda$ ($\mu$m)", fontsize=9)
        if col == 0:
            ax_sct.set_ylabel("Scatter", fontsize=8)
            ax_sct.legend(fontsize=7, loc="upper right")
        else:
            ax_sct.set_yticks([])

    cax = fig.add_axes([0.93, 0.10, 0.015, 0.85])
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("RdBu_r"),
                               norm=plt.Normalize(vmin=-VMAX, vmax=VMAX))
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label("Residual (method $-$ HR)", fontsize=9)

    out = outdir / "fig_residual_maps.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
    return out
