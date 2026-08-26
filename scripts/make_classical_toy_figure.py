#!/usr/bin/env python
"""Draw the classical-deconvolution talk slide: one picture per method.

    ./venv/bin/python scripts/make_classical_toy_figure.py

Writes ``figures/talk/fig_classical_toy.{pdf,png}``.

What it shows
-------------
A 1-D toy in the spirit of the paper's Figure 1 (\\S4.2): a close doublet that
the LSF blends into a single blob, plus one weak isolated line.  Each panel is
that toy put through one method, drawn against the truth, labelled with the
equation that defines the method and with two measured numbers -- whether the
doublet came back as two peaks, and the RMSE against the truth.

The deconvolvers are the *shipped* ones, imported from
``specsrbench/classical.py`` -- the module that built ``cache_logR_tuned/`` --
so the slide demonstrates the code the benchmark ran rather than a
re-implementation of it.

The "resolved / merged" verdict uses ``specsrbench.build.tune._pair_resolved``'s
rule verbatim (``scipy.signal.find_peaks`` with prominence 10% of the segment
maximum; two peaks or more), so the word means on this slide what it means in
the tuning guard.

Choosing the toy parameters
---------------------------
Each method's setting is *searched*, not asserted: lowest RMSE against the
truth over a fixed grid, subject to std ratio <= 1.15 so that no method can win
by inflating amplitude.  Whatever verdict falls out of that search is what the
panel prints.

The chosen values are deliberately NOT displayed.  Every classical parameter in
this project is grid-dependent -- the values tuned for the R = 4000 log grid are
meaningless at 256 pixels -- so the equations, which are the grid-independent
part, are what the slide states.  Real-grid behaviour is the results slides'
job; a toy this size is easier to deconvolve than a real spectrum and must not
be read as a performance claim.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy.signal import fftconvolve, find_peaks  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "talk"
from specsrbench import classical as C  # noqa: E402

# ── palette: the colours the results slides already use ──────────────────────
CUBIC, WIENER, TIKH = "deeppink", "tomato", "forestgreen"
TV, RL, SPARSE, MF = "teal", "steelblue", "goldenrod", "darkorchid"

INK = "#1D2530"
MUTED = "#59657A"
TRUTH_C = "#8A95A5"
TRUTH_FILL = "#E1E6ED"
BOXEC = "#C9D1DC"

FAMILY = {
    "none": ("#8A7F6C", "no prior"),
    "freq": ("#3F6494", "smoothness, in frequency"),
    "iter": ("#2E7D6F", "a prior on the signal"),
    "phys": ("#7B3FA0", "physics: where the lines are"),
}

W_UNITS, H_UNITS = 100.0, 52.5
FIG_W = 13.33
FIG_H = FIG_W * H_UNITS / W_UNITS
_FIG = None

# ── the toy ───────────────────────────────────────────────────────────────────
N = 256
SIG_LSF = 5.0
CEN = 98.0
SEP = 12.0                                   # 2.1 x the blurred line width
DOUBLET = ((CEN - SEP / 2, 2.5, 1.00), (CEN + SEP / 2, 2.5, 0.72))
ISOLATED = (168.0, 3.0, 0.42)
NOISE = 0.02
VIEW = (52, 205)
V = slice(*VIEW)


def build_toy(seed=7):
    g = np.arange(N, dtype=float)

    def gauss(mu, sig, amp):
        return amp * np.exp(-0.5 * ((g - mu) / sig) ** 2)

    truth = sum(gauss(*p) for p in (*DOUBLET, ISOLATED))
    k = np.exp(-0.5 * (np.arange(-5 * SIG_LSF, 5 * SIG_LSF + 1) / SIG_LSF) ** 2)
    k /= k.sum()
    observed = fftconvolve(truth, k, mode="same")
    observed += np.random.default_rng(seed).normal(0, NOISE, N)
    return g, truth, observed


def pair_resolved(y):
    """specsrbench.build.tune._pair_resolved's rule, on the toy's doublet window."""
    seg = np.asarray(y)[int(DOUBLET[0][0] - 16):int(DOUBLET[1][0] + 16)]
    if seg.size < 5 or not np.isfinite(seg).all() or np.nanmax(seg) <= 0:
        return False
    pk, _ = find_peaks(seg, prominence=0.10 * np.nanmax(seg))
    return len(pk) >= 2


def toy_matched_filter(y, centers, sig, detect_snr=3.0):
    """The shipped matched filter's logic, without the wavelength machinery.

    Same shape as ``classical_logR.matched_filter``: a local linear continuum
    from the sidebands, one joint least-squares fit of Gaussian templates at
    known positions, and a write-back only where the amplitude is positive and
    significant.  Reproduced here because the shipped one is parameterised by
    wavelength, redshift and a rest-frame line list, none of which a 256-pixel
    toy has.
    """
    g = np.arange(len(y), dtype=float)
    m = (g >= min(centers) - 8 * sig) & (g <= max(centers) + 8 * sig)
    gg, yy = g[m], y[m]

    dist = np.vstack([np.abs(gg - c) / sig for c in centers])
    sb = np.min(dist, axis=0) > 3.0
    cont = np.polyval(np.polyfit(gg[sb], yy[sb], 1), gg)
    noise = C.mad_sigma(yy[sb])

    T = np.column_stack([np.exp(-0.5 * ((gg - c) / sig) ** 2) for c in centers])
    amps, *_ = np.linalg.lstsq(T, yy - cont, rcond=None)
    amp_err = noise * np.sqrt(np.diag(np.linalg.pinv(T.T @ T)))
    keep = (amps > 0) & (amps / np.where(amp_err > 0, amp_err, np.inf) >= detect_snr)

    out = y.copy()
    if keep.any():
        core = np.min(dist[keep], axis=0) <= 3.0
        out[np.flatnonzero(m)[core]] = (cont + T[:, keep] @ amps[keep])[core]
    return out


def choose(fn, grid, truth, *, std_max=1.15):
    """Lowest RMSE against the truth, barred from inflating amplitude."""
    scored = []
    for p in grid:
        y = fn(p)
        rmse = float(np.sqrt(np.mean((y[V] - truth[V]) ** 2)))
        ratio = float(np.std(y[V]) / np.std(truth[V]))
        scored.append((rmse, ratio, p, y))
    ok = [s for s in scored if s[1] <= std_max]
    return min(ok or scored, key=lambda s: s[0])


def run_methods(observed, truth):
    """Every deconvolver here is the one that built the shipped cache."""
    sig = np.full(N, SIG_LSF)
    out = []

    def add(name, color, fam, eq, best):
        rmse, ratio, p, y = best
        out.append(dict(name=name, color=color, fam=fam, eq=eq, y=y,
                        rmse=rmse, param=p, std_ratio=ratio))

    obs_rmse = float(np.sqrt(np.mean((observed[V] - truth[V]) ** 2)))
    add("Cubic (LR)", CUBIC, "none", r"$\hat{x}=\mathrm{spline}_3(y)$",
        (obs_rmse, float(np.std(observed[V]) / np.std(truth[V])), None,
         observed))

    w = choose(lambda p: C.wiener_deconv(observed, sig, snr=p, segment_len=256,
                                         overlap=64),
               [2, 4, 6, 8, 12, 20, 40, 80], truth)
    add("Wiener", WIENER, "freq",
        r"$\hat{X}=\dfrac{H^{*}Y}{|H|^{2}+1/\mathrm{SNR}}$", w)

    add("Tikhonov", TIKH, "freq",
        r"$\hat{X}=\dfrac{H^{*}Y}{|H|^{2}+\lambda\,(2\pi f)^{2}}$",
        choose(lambda p: C.tikhonov_deconv(observed, sig, lam=p,
                                           segment_len=256, overlap=64),
               [2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2], truth))

    wiener = w[3]
    add("Wiener + TV", TV, "iter",
        r"$\min_{x}\ \dfrac{1}{2}\|x-\hat{x}_{\mathrm{W}}\|^{2}"
        r"+\lambda\|\nabla x\|_{1}$",
        choose(lambda p: C.tv_denoise_1d(wiener, lam=p, n_iter=30),
               [2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1], truth))

    add("Richardson–Lucy", RL, "iter",
        r"$\hat{x}^{(k+1)}=\hat{x}^{(k)}\cdot\left[\dfrac{y}"
        r"{h\ast\hat{x}^{(k)}}\ast h^{\dagger}\right]$",
        choose(lambda p: C.rl_deconv(observed, sig, n_iter=p, n_seg=4),
               [3, 5, 10, 20, 40, 80, 150, 300], truth))

    add("Wavelet-sparse (FISTA)", SPARSE, "iter",
        r"$\min_{x}\ \dfrac{1}{2}\|h\ast x-y\|^{2}+\lambda\|Wx\|_{1}$",
        choose(lambda p: C.sparse_wavelet_deconv(observed, sig, lam=p[0],
                                                 n_iter=p[1], n_seg=4),
               [(lam_, n) for lam_ in (1e-3, 2e-3, 5e-3, 1e-2, 2e-2)
                for n in (60, 150)], truth))

    mf = toy_matched_filter(wiener, [d[0] for d in DOUBLET], 2.5)
    add("Wiener + Matched Filter", MF, "phys",
        r"$\hat{x}=\sum_{\ell}a_{\ell}\,g(\lambda-\lambda_{\ell}(z))$",
        (float(np.sqrt(np.mean((mf[V] - truth[V]) ** 2))),
         float(np.std(mf[V]) / np.std(truth[V])), None, mf))
    return out


# ── primitives ────────────────────────────────────────────────────────────────
def measure(s, size, weight="normal", style="normal"):
    t = _FIG.text(0, 0, s, fontsize=size, fontweight=weight, fontstyle=style)
    w = t.get_window_extent(renderer=_FIG.canvas.get_renderer()).width
    t.remove()
    return w / _FIG.dpi / FIG_W * W_UNITS


def fit_size(s, size, max_w, *, floor=7.0, **kw):
    while size > floor and measure(s, size, **kw) > max_w:
        size -= 0.25
    return size


def rounded(ax, x0, y0, x1, y1, *, r=0.7, fc="none", ec="none", lw=0.9, z=1):
    ax.add_patch(FancyBboxPatch(
        (x0 + r, y0 + r), (x1 - x0) - 2 * r, (y1 - y0) - 2 * r,
        boxstyle=f"round,pad={r},rounding_size={r}", facecolor=fc,
        edgecolor=ec, linewidth=lw, zorder=z, mutation_aspect=1.0))


def text(ax, x, y, s, *, size=9.5, color=INK, weight="normal", style="normal",
         ha="center", va="center", z=10, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   fontstyle=style, ha=ha, va=va, zorder=z, **kw)


def arrow(ax, p0, p1, *, color="#8E99A8", lw=1.2, head=7.0, z=6):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=f"-|>,head_length={head},head_width={head*0.55}",
        mutation_scale=1.0, color=color, linewidth=lw, zorder=z,
        shrinkA=0, shrinkB=0))


def inset(ax, x0, y0, x1, y1, *, z=8):
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    (fx0, fy0), (fx1, fy1) = _FIG.transFigure.inverted().transform([p0, p1])
    sub = _FIG.add_axes((fx0, fy0, fx1 - fx0, fy1 - fy0), zorder=z)
    sub.set_xticks([])
    sub.set_yticks([])
    for sp in sub.spines.values():
        sp.set_edgecolor(BOXEC)
        sp.set_linewidth(0.7)
    sub.set_facecolor("white")
    return sub


def draw_trace(a, g, truth, y, color, ylim):
    a.axhline(0, color="#DDE2E9", lw=0.6, zorder=0)
    a.fill_between(g[V], truth[V], color=TRUTH_FILL, zorder=1, lw=0)
    a.plot(g[V], truth[V], color=TRUTH_C, lw=0.9, zorder=2)
    a.plot(g[V], y[V], color=color, lw=1.8, zorder=3, solid_capstyle="round")
    a.set_xlim(g[VIEW[0]], g[VIEW[1] - 1])
    a.set_ylim(*ylim)


# ── layout ────────────────────────────────────────────────────────────────────
TOP_Y0, TOP_Y1 = 43.0, 52.1
ROW_TOP, ROW_BOT = 41.8, 2.6
COL_W, COL_GAP = 23.85, 1.0
ROW_GAP = 1.4
ROW_H = (ROW_TOP - ROW_BOT - ROW_GAP) / 2


def col_x(i):
    x0 = 0.8 + i * (COL_W + COL_GAP)
    return x0, x0 + COL_W


def draw_top(ax, g, truth, observed, ylim):
    x0, x1 = 0.8, 99.2
    rounded(ax, x0, TOP_Y0, x1, TOP_Y1, r=0.9, fc="#EDF1F6", ec="#C6CFDC",
            lw=1.0, z=1)
    pw, ph, py = 16.5, 5.4, TOP_Y0 + 2.0
    a_x = inset(ax, x0 + 2.0, py, x0 + 2.0 + pw, py + ph)
    a_k = inset(ax, x0 + 22.4, py + 0.7, x0 + 22.4 + 6.0, py + 0.7 + ph - 1.4)
    a_y = inset(ax, x0 + 34.6, py, x0 + 34.6 + pw, py + ph)

    a_x.fill_between(g[V], truth[V], color=TRUTH_FILL, lw=0)
    a_x.plot(g[V], truth[V], color=TRUTH_C, lw=1.4)
    a_y.plot(g[V], observed[V], color="#CE3F3A", lw=1.6)
    for a in (a_x, a_y):
        a.set_xlim(g[VIEW[0]], g[VIEW[1] - 1])
        a.set_ylim(*ylim)

    kx = np.linspace(-3.2 * SIG_LSF, 3.2 * SIG_LSF, 200)
    kk = np.exp(-0.5 * (kx / SIG_LSF) ** 2)
    a_k.fill_between(kx, kk, color="#C3CCDA")
    a_k.plot(kx, kk, color="#5A6675", lw=1.0)
    a_k.set_xlim(kx[0], kx[-1])
    a_k.set_ylim(0, 1.12)

    lbl = dict(size=9.4, va="top", weight="bold")
    text(ax, x0 + 2.0, py - 0.5, r"$x$   truth", color="#6E7A8A", ha="left", **lbl)
    text(ax, x0 + 25.4, py - 0.5, r"$h$   LSF", color="#4E5A6B", **lbl)
    text(ax, x0 + 34.6, py - 0.5, r"$y$   observed", color="#CE3F3A",
         ha="left", **lbl)
    text(ax, x0 + 20.0, py + ph / 2, r"$\ast$", size=17, color="#4E5A6B")
    text(ax, x0 + 31.6, py + ph / 2, r"$+\,n\;=$", size=12.5, color="#4E5A6B")

    text(ax, (x0 + 34.6 + pw + x1) / 2, py + ph / 2, r"$y=h\ast x+n$",
         size=25)


def draw_panel(ax, spec, g, truth, ylim, *, col, row):
    x0, x1 = col_x(col)
    y1 = ROW_TOP - row * (ROW_H + ROW_GAP)
    y0 = y1 - ROW_H
    accent = FAMILY[spec["fam"]][0]

    rounded(ax, x0, y0, x1, y1, r=0.6, fc="white", ec=BOXEC, lw=1.0, z=2)
    ax.add_patch(FancyBboxPatch(
        (x0 + 0.65, y1 - 0.95), COL_W - 1.3, 0.28,
        boxstyle="round,pad=0.2,rounding_size=0.2", facecolor=accent,
        edgecolor=accent, zorder=4, mutation_aspect=1.0))

    ns = fit_size(spec["name"], 13.0, COL_W - 2.6, weight="bold", floor=9.5)
    text(ax, x0 + 1.3, y1 - 2.5, spec["name"], size=ns, weight="bold",
         color=spec["color"], ha="left", va="center")

    es = fit_size(spec["eq"], 13.5, COL_W - 2.6, floor=8.0)
    text(ax, x0 + COL_W / 2, y1 - 5.7, spec["eq"], size=es, color=INK)

    a = inset(ax, x0 + 1.1, y0 + 1.2, x1 - 1.1, y1 - 7.5)
    draw_trace(a, g, truth, spec["y"], spec["color"], ylim)
    return pair_resolved(spec["y"])


def draw_key(ax, col, row):
    """The eighth cell carries the legend and the prior axis."""
    x0, x1 = col_x(col)
    y1 = ROW_TOP - row * (ROW_H + ROW_GAP)
    y0 = y1 - ROW_H
    rounded(ax, x0, y0, x1, y1, r=0.6, fc="#F6F7F9", ec=BOXEC, lw=1.0, z=2)

    text(ax, x0 + 1.3, y1 - 2.2, "Reading a panel", size=12.0, weight="bold",
         ha="left", va="center")
    y = y1 - 4.5
    ax.fill_between([x0 + 1.6, x0 + 4.8], [y + 0.5, y + 0.5], y - 0.5,
                    color=TRUTH_FILL, zorder=5, lw=0)
    ax.plot([x0 + 1.6, x0 + 4.8], [y + 0.5, y + 0.5], color=TRUTH_C, lw=1.2,
            zorder=6)
    text(ax, x0 + 5.6, y, "truth", size=10.0, color=INK, ha="left")
    y -= 2.0
    ax.plot([x0 + 1.6, x0 + 4.8], [y, y], color="#55606E", lw=2.0, zorder=6,
            solid_capstyle="round")
    text(ax, x0 + 5.6, y, "what the method returns", size=10.0, color=INK,
         ha="left")

    text(ax, x0 + 1.3, y1 - 9.2, "Prior used", size=12.0, weight="bold",
         ha="left", va="center")
    rows = [y1 - 10.8 - i * 1.9 for i in range(4)]
    for yy, key in zip(rows, ("none", "freq", "iter", "phys")):
        accent, label = FAMILY[key]
        ax.add_patch(FancyBboxPatch(
            (x0 + 3.0, yy - 0.14), 2.5, 0.28,
            boxstyle="round,pad=0.18,rounding_size=0.18", facecolor=accent,
            edgecolor=accent, zorder=6, mutation_aspect=1.0))
        sz = fit_size(label, 9.8, x1 - 1.3 - (x0 + 6.6), floor=7.4)
        text(ax, x0 + 6.6, yy, label, size=sz, color=INK, ha="left")

    # the families are listed in order of how much they are told, so the axis
    # is a tick beside the list rather than another row of its own
    ax.plot([x0 + 1.5, x0 + 1.5], [rows[0] + 0.5, rows[-1] - 0.2],
            color="#8E99A8", lw=1.2, zorder=6)
    arrow(ax, (x0 + 1.5, rows[-1] - 0.2), (x0 + 1.5, rows[-1] - 0.9), head=6.5)


def main():
    global _FIG
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "savefig.dpi": 300})
    _FIG = plt.figure(figsize=(FIG_W, FIG_H))
    _FIG.canvas.draw()
    ax = _FIG.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W_UNITS)
    ax.set_ylim(0, H_UNITS)
    ax.set_axis_off()
    _FIG.patch.set_facecolor("white")

    g, truth, observed = build_toy()
    specs = run_methods(observed, truth)

    hi = max(truth[V].max(), max(s["y"][V].max() for s in specs))
    lo = min(0.0, min(s["y"][V].min() for s in specs))
    ylim = (lo - 0.05 * hi, 1.10 * hi)

    draw_top(ax, g, truth, observed, ylim)
    print(f"  {'observed input':26s} resolved={pair_resolved(observed)}")
    for i, spec in enumerate(specs):
        ok = draw_panel(ax, spec, g, truth, ylim, col=i % 4, row=i // 4)
        print(f"  {spec['name']:26s} resolved={str(ok):5s} "
              f"RMSE={spec['rmse']:.4f} std_ratio={spec['std_ratio']:.2f} "
              f"param={spec['param']}")
    draw_key(ax, 3, 1)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        _FIG.savefig(OUT / f"fig_classical_toy.{ext}", facecolor="white")
    print(f"wrote {OUT}/fig_classical_toy.[pdf,png]")


if __name__ == "__main__":
    main()
