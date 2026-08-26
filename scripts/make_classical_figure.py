#!/usr/bin/env python
"""Draw the classical-deconvolution schematic used in the conference talk.

    ./venv/bin/python scripts/make_classical_figure.py

Writes ``figures/talk/fig_classical_methods.{pdf,png}``.

Why a script and not a drawing tool
-----------------------------------
The top band is real data: the high-resolution grating reference, the modelled
prism LSF at that wavelength, and the prism spectrum the methods actually see
are read from ``cache_logR_tuned/`` -- the same cache the benchmark figures are
built from.  A schematic whose curves come from the cache cannot drift away
from the results it introduces.

Every parameter printed in a method box is read out of
``classical_params.json`` at draw time, so the slide quotes the settings the
shipped cache was actually built with.  This is not paranoia: a written-down
parameter and the one a cache was built with have drifted apart in this project
before, in both directions, and a slide is the least likely place anyone would
look for the discrepancy.

Text is wrapped by measuring it with the renderer rather than by guessing a
character count, and every box is sized from its own wrapped content, so a
longer sentence grows its box instead of silently overflowing it.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "cache_logR_tuned"
OUT = REPO / "figures" / "talk"


def shipped_params():
    """The parameters the shipped cache was actually built with.

    Read from ``classical_params.json`` -- the one home for them -- rather than
    from the build module, which would load the whole evaluation cache and pull
    in specsr just to draw a slide.  This used to parse a literal out of the
    builder's source; when the parameters moved into the JSON file that parse
    silently found nothing, which is the failure mode the single-home rule
    exists to prevent.
    """
    record = json.loads((CACHE / "classical_params.json").read_text())
    return {k: dict(v) for k, v in record["tuned"].items()}


TUNED = shipped_params()
N_MF_LINES = 67          # emission templates in specsr's LINE_LIST_REST_AA
                         # after classical_cache.is_emission_template

# ── palette ───────────────────────────────────────────────────────────────────
# The method colours are the ones the results slides already use (COLORS in
# specsrbench.methods), so a colour means the same method
# across the whole talk.
CUBIC, WIENER, TIKH = "deeppink", "tomato", "forestgreen"
TV, RL, SPARSE = "teal", "steelblue", "goldenrod"
MF, ML = "darkorchid", "darkorange"

INK = "#1D2530"
MUTED = "#59657A"
FRAME = "#9AA4B2"
ARROWC = "#46566B"
PANEL_TOP = "#EDF1F6"
BOXFC = "#FFFFFF"
BOXEC = "#C9D1DC"
HR_C, LR_C = "#6E56A8", "#CE3F3A"

W_UNITS, H_UNITS = 100.0, 46.5
FIG_W = 13.33
FIG_H = FIG_W * H_UNITS / W_UNITS

_FIG = None      # set in main(); needed for renderer-based text measurement


# ── text metrics ──────────────────────────────────────────────────────────────
def measure(s, size, weight="normal", style="normal"):
    """Width of ``s`` in canvas units, measured with the real renderer."""
    t = _FIG.text(0, 0, s, fontsize=size, fontweight=weight, fontstyle=style)
    w = t.get_window_extent(renderer=_FIG.canvas.get_renderer()).width
    t.remove()
    return w / _FIG.dpi / FIG_W * W_UNITS


def line_step(size, lh=1.30):
    """Baseline-to-baseline distance in canvas units."""
    return size * lh / 72.0 * (H_UNITS / FIG_H)


def wrap_fit(s, size, max_w, **kw):
    """Greedy word wrap to a width in canvas units."""
    lines, cur = [], ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if cur and measure(trial, size, **kw) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def fit_size(s, size, max_w, *, floor=8.0, **kw):
    """Largest font size <= ``size`` at which ``s`` fits in ``max_w``."""
    while size > floor and measure(s, size, **kw) > max_w:
        size -= 0.25
    return size


# ── primitives ────────────────────────────────────────────────────────────────
def rounded(ax, x0, y0, x1, y1, *, r=0.7, fc="none", ec="none", lw=0.9,
            ls="solid", z=1):
    ax.add_patch(FancyBboxPatch(
        (x0 + r, y0 + r), (x1 - x0) - 2 * r, (y1 - y0) - 2 * r,
        boxstyle=f"round,pad={r},rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        zorder=z, mutation_aspect=1.0))


def arrow(ax, p0, p1, *, color=ARROWC, lw=1.3, z=6, rad=0.0, head=7.0,
          ls="solid"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=f"-|>,head_length={head},head_width={head*0.55}",
        connectionstyle=f"arc3,rad={rad}", mutation_scale=1.0, color=color,
        linewidth=lw, linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, *, size=9.5, color=INK, weight="normal", style="normal",
         ha="center", va="center", z=10, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   fontstyle=style, ha=ha, va=va, zorder=z, **kw)


def para(ax, x, y, s, *, max_w, size=9.0, color=MUTED, lh=1.30, **kw):
    """Left-aligned wrapped paragraph anchored at its top; returns bottom y."""
    lines = wrap_fit(s, size, max_w)
    step = line_step(size, lh)
    for i, ln in enumerate(lines):
        text(ax, x, y - i * step, ln, size=size, color=color, ha="left",
             va="top", **kw)
    return y - len(lines) * step


def para_height(s, size, max_w, lh=1.30):
    return len(wrap_fit(s, size, max_w)) * line_step(size, lh)


def inset(ax, x0, y0, x1, y1, *, z=8):
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    (fx0, fy0), (fx1, fy1) = _FIG.transFigure.inverted().transform([p0, p1])
    sub = _FIG.add_axes((fx0, fy0, fx1 - fx0, fy1 - fy0), zorder=z)
    sub.set_xticks([])
    sub.set_yticks([])
    for sp in sub.spines.values():
        sp.set_edgecolor(FRAME)
        sp.set_linewidth(0.7)
    sub.set_facecolor("white")
    return sub


# ── geometry ──────────────────────────────────────────────────────────────────
# Only the outer edges and the top band are fixed.  The header strips, the
# footer and therefore the space left for the method boxes are all measured
# from their own wrapped text in resolve_geometry(), so editing a sentence
# moves the layout instead of overflowing a box that was sized by hand.
TOP_Y0, TOP_Y1 = 36.3, 46.3
BUS_Y = 35.5
HDR_TOP = 34.7
HDR_BOT = BOX_TOP = BOX_BOT = FOOT_Y1 = 0.0     # set by resolve_geometry()
FOOT_Y0 = 0.3

COLUMNS = [
    dict(x0=0.8, x1=15.8, accent="#8A7F6C", head="#F4F1EA",
         title="No deconvolution",
         prior="No prior at all. Adds no information."),
    dict(x0=17.1, x1=39.6, accent="#3F6494", head="#EBF0F7",
         title="Linear filters",
         prior="A smoothness prior, applied in frequency: one\u2009multiply per FFT segment."),
    dict(x0=40.9, x1=68.5, accent="#3F6494", head="#EBF0F7",
         title="Iterative methods",
         prior="A prior on the signal itself, imposed by stepping the forward model."),
    dict(x0=69.8, x1=84.5, accent="#7B3FA0", head="#F4ECF8",
         title="Physics prior",
         prior="Told where the lines are: redshift + line list."),
    dict(x0=85.8, x1=99.2, accent="#C06A18", head="#FCF0E2",
         title="Learned prior",
         prior="Told nothing. Infers it from the data."),
]

NAME_SIZE, EQ_SIZE, BODY_SIZE, TAG_SIZE = 11.0, 10.0, 8.5, 8.2
PAD_X, PAD_TOP, PAD_BOT = 1.6, 1.05, 0.55
HDR_SIZE, FOOT_SIZE = 8.6, 8.5

FOOT_LEAD = ("Every classical method has to be handed the LSF; the network is "
             "not.  Wiener + MF is the fair comparator — it is given the same "
             "prior the network has to learn.")
FOOT_NOTE = ("All parameters retuned on a galaxy-disjoint 40-spectrum set.  "
             "MAE alone is not a safe objective — it rewards smoothing, "
             "shrinkage, blurring and merging — so the search is guarded on "
             "line S/N, amplitude, FWHM and [O III] doublet survival.")


def resolve_geometry():
    """Size the header strips and the footer to their text, then hand what is
    left to the method boxes."""
    global HDR_BOT, BOX_TOP, BOX_BOT, FOOT_Y1
    hdr_h = 1.4 + 0.3 + 0.6 + max(
        para_height(c["prior"], HDR_SIZE, c["x1"] - c["x0"] - 2.2)
        for c in COLUMNS)
    HDR_BOT = HDR_TOP - hdr_h
    BOX_TOP = HDR_BOT - 1.0

    foot_h = (1.25 + para_height(FOOT_LEAD, 9.2, 95.0)
              + para_height(FOOT_NOTE, FOOT_SIZE, 95.0) + 0.55)
    FOOT_Y1 = FOOT_Y0 + foot_h
    BOX_BOT = FOOT_Y1 + 0.7


# ── data for the top band ─────────────────────────────────────────────────────
def forward_problem_data(i=188, lam_lo=3.715, lam_hi=3.845):
    """The pair the methods are handed, plus the LSF modelled at that lambda.

    Spectrum 188 is the figure-2 example: z = 6.55, where [O III] 4959/5007
    sit 37 nm apart, so the blur the top band illustrates is blur the data can
    actually show being undone.
    """
    wl = np.load(CACHE / "wl_high.npy")
    xh = np.load(CACHE / "x_high.npy")[i]
    xl = np.load(CACHE / "x_low.npy")[i]
    sig = np.load(CACHE / "sigma_pix.npy")
    z = float(np.load(CACHE / "z_test.npy")[i])

    m = (wl >= lam_lo) & (wl <= lam_hi)
    pix = int(np.argmin(np.abs(wl - 0.5 * (lam_lo + lam_hi))))
    sig_nm = float(sig[pix]) * float(np.gradient(wl)[pix]) * 1e3
    lines_nm = [0.4959 * (1 + z) * 1e3, 0.5007 * (1 + z) * 1e3]
    return dict(wl=wl[m] * 1e3, hr=xh[m], lr=xl[m], z=z, lines_nm=lines_nm,
                sig_nm=sig_nm, fwhm_nm=2.3548 * sig_nm)


def draw_top_band(ax, d):
    x0, x1 = 0.8, 99.2
    rounded(ax, x0, TOP_Y0, x1, TOP_Y1, r=0.9, fc=PANEL_TOP, ec="#C6CFDC",
            lw=1.0, z=1)
    text(ax, x0 + 2.0, TOP_Y1 - 1.5, "The problem every method below is solving",
         size=13.0, weight="bold", ha="left")
    text(ax, x1 - 2.0, TOP_Y1 - 1.5,
         rf"JADES galaxy at $z={d['z']:.2f}$,  [O III] $\lambda\lambda$4959,5007",
         size=9.2, color=MUTED, ha="right", style="italic")

    pw, ph, py = 15.5, 4.7, TOP_Y0 + 2.3
    a_hr = inset(ax, x0 + 2.0, py, x0 + 2.0 + pw, py + ph)
    a_k = inset(ax, x0 + 21.4, py + 0.7, x0 + 21.4 + 6.4, py + 0.7 + ph - 1.4)
    a_lr = inset(ax, x0 + 33.6, py, x0 + 33.6 + pw, py + ph)

    for a, y, c in ((a_hr, d["hr"], HR_C), (a_lr, d["lr"], LR_C)):
        for lam in d["lines_nm"]:
            a.axvline(lam, color="#9AA4B2", lw=0.7, ls=(0, (2, 2)), zorder=1)
        a.plot(d["wl"], y, color=c, lw=1.2, zorder=2)
        a.set_xlim(d["wl"][0], d["wl"][-1])
        a.margins(y=0.16)
    lo = min(a_hr.get_ylim()[0], a_lr.get_ylim()[0])
    hi = max(a_hr.get_ylim()[1], a_lr.get_ylim()[1])
    a_hr.set_ylim(lo, hi)
    a_lr.set_ylim(lo, hi)

    g = np.linspace(-3.2 * d["sig_nm"], 3.2 * d["sig_nm"], 200)
    k = np.exp(-0.5 * (g / d["sig_nm"]) ** 2)
    a_k.fill_between(g, k, color="#C3CCDA")
    a_k.plot(g, k, color="#5A6675", lw=1.0)
    a_k.set_xlim(g[0], g[-1])
    a_k.set_ylim(0, 1.12)

    lbl = dict(size=8.8, va="top", weight="bold")
    text(ax, x0 + 2.0, py - 0.5, r"$x$   reference, $R\approx1000$",
         color=HR_C, ha="left", **lbl)
    text(ax, x0 + 24.6, py - 0.5,
         rf"$h$   LSF, {d['fwhm_nm']:.0f} nm FWHM", color="#4E5A6B",
         ha="center", **lbl)
    text(ax, x0 + 33.6, py - 0.5, r"$y$   what we observe, $R\approx100$",
         color=LR_C, ha="left", **lbl)
    text(ax, x0 + 18.7, py + ph / 2, r"$\circledast$", size=15, color="#4E5A6B")
    text(ax, x0 + 30.6, py + ph / 2, r"$+\,n\;\;=$", size=12.5, color="#4E5A6B")

    tx = x0 + 52.0
    maxw = (x1 - 2.0) - tx
    text(ax, tx, TOP_Y1 - 3.5, r"$y \;=\; h(\lambda) \circledast x \;+\; n$",
         size=15.5, ha="left", va="center")
    text(ax, tx + 22.0, TOP_Y1 - 3.5,
         r"$H(f)=e^{-2\pi^{2}f^{2}\sigma^{2}}$", size=12.0, color=MUTED,
         ha="left", va="center")
    y = para(ax, tx, TOP_Y1 - 5.0,
             "Inverting this is ill-posed. The LSF is a low-pass: above "
             "f ≈ 1/(2πσ) the signal is gone and only noise is left to "
             "amplify. Nothing recovers it from the data alone.",
             max_w=maxw, size=9.4, color=INK)
    text(ax, tx, y - 0.7,
         "Every method below is one choice of prior for what to put back.",
         size=11.0, weight="bold", ha="left", va="top")


# ── method boxes ──────────────────────────────────────────────────────────────
def method_height(*, eq, body, tag, max_w):
    h = PAD_TOP + line_step(NAME_SIZE)
    if eq:
        h += line_step(EQ_SIZE) + 0.15
    elif tag:
        h += line_step(TAG_SIZE) + 0.15
    h += para_height(body, BODY_SIZE, max_w) + PAD_BOT
    return h


def method(ax, col, y_top, *, color, name, body, eq=None, tag=None,
           eq_size=EQ_SIZE):
    c = COLUMNS[col]
    x0, x1 = c["x0"], c["x1"]
    tx = x0 + PAD_X
    max_w = x1 - PAD_X - tx
    h = method_height(eq=eq, body=body, tag=tag, max_w=max_w)

    rounded(ax, x0, y_top - h, x1, y_top, r=0.6, fc=BOXFC, ec=BOXEC, lw=1.0, z=3)
    ax.add_patch(FancyBboxPatch(
        (x0 + 0.45, y_top - h + 0.7), 0.28, h - 1.4,
        boxstyle="round,pad=0.22,rounding_size=0.22", facecolor=color,
        edgecolor=color, zorder=4, mutation_aspect=1.0))

    y = y_top - PAD_TOP
    # The two-stage methods must not read as independent filters. Where there
    # is an equation the tag shares the name row; where there is not, it takes
    # the equation's row, which keeps long names from having to shrink.
    name_w = max_w
    if tag and eq:
        text(ax, x1 - PAD_X, y, tag, size=TAG_SIZE, color=WIENER,
             style="italic", ha="right", va="top")
        name_w = max_w - measure(tag, TAG_SIZE, style="italic") - 1.0
    ns = fit_size(name, NAME_SIZE, name_w, weight="bold", floor=8.6)
    text(ax, tx, y, name, size=ns, weight="bold", color=color, ha="left",
         va="top")
    y -= line_step(NAME_SIZE)
    if eq:
        es = fit_size(eq, eq_size, max_w, floor=8.0)
        text(ax, tx, y, eq, size=es, color=INK, ha="left", va="top")
        y -= line_step(eq_size) + 0.15
    elif tag:
        text(ax, tx, y, tag, size=TAG_SIZE, color=WIENER, style="italic",
             ha="left", va="top")
        y -= line_step(TAG_SIZE) + 0.15
    para(ax, tx, y, body, max_w=max_w, size=BODY_SIZE)
    return h


def draw_column_headers(ax):
    for c in COLUMNS:
        rounded(ax, c["x0"], HDR_BOT, c["x1"], HDR_TOP, r=0.7, fc=c["head"],
                ec=c["accent"], lw=1.1, z=2)
        ts = fit_size(c["title"], 12.0, c["x1"] - c["x0"] - 2.2, weight="bold",
                      floor=9.5)
        text(ax, c["x0"] + 1.1, HDR_TOP - 1.4, c["title"], size=ts,
             weight="bold", color=c["accent"], ha="left", va="center")
        para(ax, c["x0"] + 1.1, HDR_TOP - 2.4, c["prior"],
             max_w=c["x1"] - c["x0"] - 2.2, size=HDR_SIZE, color=INK)


def draw_methods(ax):
    gap = 0.7
    used = {}

    def stack(col, specs):
        y = BOX_TOP
        for kw in specs:
            y -= method(ax, col, y, **kw) + gap
        used[col] = BOX_TOP - (y + gap)

    stack(0, [dict(
        color=CUBIC, name="Cubic (LR)", eq=r"$\hat{x}=\mathrm{spline}_3(y)$",
        body="The prism spectrum resampled onto the R = 4000 grid. Nothing is "
             "deconvolved. This is the floor: a method that does not beat it "
             "has bought sharpness with no information.")])

    stack(1, [
        dict(color=WIENER, name="Wiener",
             eq=r"$W=H^{*}/(|H|^{2}+1/\mathrm{SNR})$,    "
                rf"$\mathrm{{SNR}}={TUNED['wiener']['snr']:.0f}$",
             body="Minimum-mean-square-error filter under a flat noise prior. "
                  "Normalised to unit gain at f = 0 so it cannot lower the "
                  "error just by shrinking the spectrum toward zero."),
        dict(color=TIKH, name="Tikhonov",
             eq=r"$W=H^{*}/(|H|^{2}+\lambda(2\pi f)^{2})$,    "
                rf"$\lambda={TUNED['tikhonov']['lam']:.0f}$",
             body="The penalty grows with frequency instead of being constant: "
                  "high-frequency noise is suppressed harder than by Wiener, "
                  "while the smooth continuum passes through untouched."),
    ])

    stack(2, [
        dict(color=RL, name="Richardson–Lucy",
             eq=r"$\hat{x}^{(k+1)}=\hat{x}^{(k)}\cdot\left[(y/h\circledast"
                r"\hat{x}^{(k)})\circledast h^{\dagger}\right]$",
             body="Non-negative, Poisson likelihood. Sharpens peaks and "
                  "amplifies noise together "
                  f"({TUNED['rl']['n_iter']} iteration)."),
        dict(color=SPARSE, name="Wavelet-sparse (FISTA)",
             eq=r"$\min\;\frac{1}{2}|h\circledast x-y|^{2}+\lambda|Wx|_{1}$,"
                rf"    $\lambda={TUNED['sparse']['lam']}$",
             body="Daubechies-4, 4 levels: lines are sparse at fine scales, "
                  f"continuum at coarse ({TUNED['sparse']['n_iter']} iteration)."),
        dict(color=TV, name="Wiener + Total Variation", tag="after Wiener",
             eq=r"$\min\;\frac{1}{2}|x-\hat{x}_{\mathrm{W}}|^{2}"
                r"+\lambda|\nabla x|_{1}$,    "
                rf"$\lambda={TUNED['tv']['lam']}$",
             body="Edge-preserving denoiser. Keeps sharp jumps, but can leave "
                  f"piecewise-flat staircases ({TUNED['tv']['n_iter']} iterations)."),
    ])

    stack(3, [dict(
        color=MF, name="Wiener + Matched Filter", tag="after Wiener",
        body=f"Gaussian templates are fitted to {N_MF_LINES} rest-frame emission "
             "features placed at the catalogue redshift. Overlapping windows are "
             "fitted jointly, and a line is written back only where its amplitude "
             "is positive and its local significance reaches "
             f"{TUNED['mf']['detect_snr']:.0f}σ.")])

    stack(4, [dict(
        color=ML, name="ML (SR2)",
        body="The three-stage network of Paper 1: a conservative CNN, a redshift "
             "head, then an attention refiner over emission-line tokens. It is "
             "given neither the LSF nor the redshift — both are learned from "
             "~25,000 JWST prism/grating pairs.")])

    for col, h in sorted(used.items()):
        room = BOX_TOP - BOX_BOT
        flag = "  <-- OVERFLOW" if h > room else ""
        print(f"  column {col}: {h:5.1f} of {room:.1f} units{flag}")


def draw_flow(ax):
    x_lr = 0.8 + 33.6 + 15.5 / 2
    arrow(ax, (x_lr, TOP_Y0 - 0.1), (x_lr, BUS_Y + 0.3), lw=1.5)
    xs = [c["x0"] + min(7.0, (c["x1"] - c["x0"]) / 2) for c in COLUMNS]
    ax.plot([min(xs), max(xs)], [BUS_Y, BUS_Y], color=ARROWC, lw=1.4, zorder=5,
            solid_capstyle="round")
    for x in xs:
        arrow(ax, (x, BUS_Y), (x, HDR_TOP + 0.25), lw=1.4)


def draw_footer(ax):
    rounded(ax, 0.8, FOOT_Y0, 99.2, FOOT_Y1, r=0.6, fc="#F5F7F9", ec="#D3DAE3",
            lw=0.9, z=1)
    ay = FOOT_Y1 - 0.8
    mask = dict(facecolor="#F5F7F9", edgecolor="none", pad=2.0)
    ax.plot([3.2, 82.0], [ay, ay], color="#9AA5B4", lw=1.1, zorder=5)
    arrow(ax, (82.0, ay), (96.8, ay), color="#9AA5B4", lw=1.1, head=6.5)
    text(ax, 3.4, ay, "none", size=8.4, color=MUTED, ha="left", bbox=mask)
    text(ax, 96.6, ay, "learned", size=8.4, color=MUTED, ha="right", bbox=mask)
    text(ax, 49.9, ay, "increasing prior information", size=9.6, color=INK,
         weight="bold", ha="center", va="center", bbox=mask)

    y = para(ax, 3.2, FOOT_Y1 - 1.30, FOOT_LEAD, max_w=95.0, size=9.2, color=INK)
    para(ax, 3.2, y - 0.15, FOOT_NOTE, max_w=95.0, size=FOOT_SIZE, color=MUTED)


def main():
    global _FIG
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "savefig.dpi": 300})
    _FIG = plt.figure(figsize=(FIG_W, FIG_H))
    _FIG.canvas.draw()                     # renderer needed by measure()
    ax = _FIG.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W_UNITS)
    ax.set_ylim(0, H_UNITS)
    ax.set_axis_off()
    _FIG.patch.set_facecolor("white")

    resolve_geometry()
    draw_column_headers(ax)
    draw_methods(ax)
    draw_footer(ax)
    draw_top_band(ax, forward_problem_data())
    draw_flow(ax)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        _FIG.savefig(OUT / f"fig_classical_methods.{ext}", facecolor="white")
    print(f"wrote {OUT}/fig_classical_methods.[pdf,png]")


if __name__ == "__main__":
    main()
