"""The results prose must match the cache, not just the summary table.

The global-fidelity table has been guarded since the first rebuild and stayed
correct.  The paragraphs around it were not guarded, and after the classical
methods were retuned every classical number in Sections 4.3-4.7 was left
quoting the pre-retune cache -- line S/N ratios, detection fractions,
false-detection rates, FWHM biases, flux-ratio rankings and the redshift bins.
Two of those stale numbers had also inverted a qualitative claim.  These tests
recompute each quoted quantity and assert the manuscript states it.

On 2026-10-01 the prose was cut back to one or two rounded numbers per claim,
the rest being left to the figures and tables.  Where a number left the text
its string guard went with it, but the *claim* the sentence still makes is
recomputed here -- "every classical method falls below the reference" is as
easy to leave stale as "$0.22$ to $0.71$" was.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from conftest import REPO, SRC

PAPER = REPO / "paper" / "paper.tex"
LINES = ["Halpha", "Hbeta", "OII3727", "OIII5007"]
# snr.npz key prefix -> fit_params_cache.npz method name
SNR_KEY = {"LR": "Cubic (LR)", "Wiener": "Wiener", "Tikhonov": "Tikhonov",
           "TV": "TV", "RL": "R-L", "Sparse": "Sparse", "MF": "Wiener + MF",
           "SR2": "ML (SR2)"}
CLASSICAL = [k for k in SNR_KEY if k != "SR2"]


@pytest.fixture(scope="module")
def tex():
    if not PAPER.exists():
        pytest.skip("paper.tex not present")
    return PAPER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def snr(cache):
    return np.load(cache / "snr.npz", allow_pickle=True)


@pytest.fixture(scope="module")
def fits(cache):
    return np.load(cache / "fit_params_cache.npz", allow_pickle=True)


def _states(tex, value, fmt=".2f"):
    return f"${value:{fmt}}$" in tex


def _said(tex, text):
    """``text`` appears verbatim, except that any space may be a line break."""
    return re.search(re.escape(text).replace(r"\ ", r"\s+"), tex) is not None


# ── per-line S/N (Section 4.3) ────────────────────────────────────────────────
@pytest.fixture(scope="module")
def snr_ratios(snr):
    """Median per-spectrum S/N ratio vs the HR reference, over the HR>5 subset."""
    out = {}
    for key in SNR_KEY:
        out[key] = []
        for line in LINES:
            hr, me = snr[f"HR_{line}"], snr[f"{key}_{line}"]
            sel = np.isfinite(hr) & (hr > 5) & np.isfinite(me)
            out[key].append(float(np.median(me[sel] / hr[sel])))
    return out


def test_sr2_line_snr_ratios(tex, snr_ratios):
    sr2 = dict(zip(LINES, snr_ratios["SR2"]))
    assert min(sr2.values()) > 1, "SR2 no longer exceeds the reference for every line"
    ha = sr2.pop("Halpha")
    assert _said(tex, f"by a factor of ${ha:.2f}$ for \\ha"), \
        f"paper does not state SR2's Halpha S/N ratio {ha:.2f}"
    assert round(ha) == 3, f"abstract says three times the reference's; it is {ha:.2f}"
    lo, hi = min(sr2.values()), max(sr2.values())
    assert _said(tex, f"${lo:.1f}$--${hi:.1f}$ for the others"), \
        f"paper does not state the other lines' S/N ratios {lo:.1f}-{hi:.1f}"


def test_classical_line_snr_range(tex, snr_ratios):
    arr = np.array([snr_ratios[k] for k in CLASSICAL])
    assert arr.max() < 1, f"a classical method exceeds the reference ({arr.max():.2f})"
    assert _said(tex, "Every classical method falls below the grating reference "
                      "for every line")


# ── amplitude recovery (Sections 4.3 and 5) ───────────────────────────────────
@pytest.fixture(scope="module")
def amp_recovery(fits):
    out = {}
    for line in LINES:
        h = fits[f"HR target_{line}_amp"]
        s = fits[f"ML (SR2)_{line}_amp"]
        hs = fits[f"HR target_{line}_sn"]
        v = np.isfinite(h) & np.isfinite(s) & (hs > 5) & (h != 0)
        out[line] = 100.0 * float(np.median(s[v] / h[v]))
    return out


def test_amplitude_recovery_range(tex, amp_recovery):
    lo, hi = min(amp_recovery.values()), max(amp_recovery.values())
    assert f"{lo:.0f}--{hi:.0f}\\%" in tex, \
        f"paper does not state the SR2 amplitude range {lo:.0f}-{hi:.0f}%"
    ha = amp_recovery["Halpha"]
    assert 45 <= ha <= 55, f"paper says about half for Halpha; it is {ha:.1f}%"
    assert ha == hi, "paper says the other three lines are lower still than Halpha"
    assert _said(tex, "about half the reference value for \\ha\\ and lower still")


# ── detection fractions (Section 4.3) ─────────────────────────────────────────
def test_detection_fractions(tex, snr):
    def frac(key, line):
        return 100.0 * float(np.nanmean(snr[f"{key}_{line}"] > 5))

    for line in ("Hbeta", "OII3727"):
        sr2, hr = frac("SR2", line), frac("HR", line)
        assert sr2 > hr, f"SR2 no longer out-detects the reference for {line}"
        assert _said(tex, f"${sr2:.0f}\\%$ against ${hr:.0f}\\%$"), \
            f"paper does not state the {line} detection fractions {sr2:.0f}% / {hr:.0f}%"
    hi = max(frac(k, line) for k in CLASSICAL for line in ("Hbeta", "OII3727"))
    assert _said(tex, f"the classical methods reach at most ${hi:.0f}\\%$"), \
        f"paper does not state the classical weak-line detection ceiling {hi:.0f}%"


# ── false-detection rates (Section 4.4) ───────────────────────────────────────
@pytest.fixture(scope="module")
def fdr(fits):
    out = {}
    for key, name in SNR_KEY.items():
        out[key] = []
        for line in LINES:
            h = fits[f"HR target_{line}_sn"]
            s = fits[f"{name}_{line}_sn"]
            v = np.isfinite(h) & (h < 3) & np.isfinite(s)
            out[key].append(float(np.mean(s[v] > 5)))
    return out


def test_sr2_weak_line_fdr(tex, fdr):
    hb, oii = (fdr["SR2"][LINES.index(k)] for k in ("Hbeta", "OII3727"))
    assert len(re.findall(rf"reaches \${hb:.2f}\$ and \${oii:.2f}\$", tex)) == 1, \
        f"results do not state the SR2 weak-line FDRs {hb:.2f} / {oii:.2f}"
    assert round(oii, 1) == 0.4, "paper says roughly two in five for [O II]"


def test_strong_line_fdr_band(tex, fdr):
    """For Halpha and [O III] every method, SR2 included, sits in one band."""
    arr = np.array([[fdr[k][LINES.index(line)] for line in ("Halpha", "OIII5007")]
                    for k in SNR_KEY])
    band = f"${arr.min():.2f}$--${arr.max():.2f}$"
    assert tex.count(band) == 1, \
        f"results do not state the strong-line FDR band {band}"


def test_no_classical_method_exceeds_the_stated_weak_line_fdr(tex, fdr):
    arr = np.array([fdr[k] for k in CLASSICAL])
    worst = max(arr[:, LINES.index("Hbeta")].max(), arr[:, LINES.index("OII3727")].max())
    stated = [float(v) for v in
              re.findall(r"No classical method exceeds \$([0-9.]+)\$", tex)]
    assert len(stated) == 1, "paper does not bound the classical weak-line FDR"
    # The discussion no longer repeats the numbers; it says "several times".
    assert _said(tex, "several times that of any classical method")
    for line in ("Hbeta", "OII3727"):
        i = LINES.index(line)
        assert fdr["SR2"][i] > 3 * arr[:, i].max(), \
            f"SR2's {line} FDR is not several times the worst classical method's"
    for v in stated:
        assert worst <= v < worst + 0.01, \
            f"paper bounds the classical weak-line FDR at {v}, cache says {worst:.3f}"


# ── FWHM bias (Section 4.6) ───────────────────────────────────────────────────
@pytest.fixture(scope="module")
def fwhm_bias(fits):
    out = {}
    for key, name in SNR_KEY.items():
        out[key] = []
        for line in LINES:
            hf = 2.355 * fits[f"HR target_{line}_sigma"] * 1e3
            mf = 2.355 * fits[f"{name}_{line}_sigma"] * 1e3
            hs = fits[f"HR target_{line}_sn"]
            v = np.isfinite(hf) & np.isfinite(mf) & (hs > 5)
            out[key].append(float(np.nanmedian(mf[v] - hf[v])))
    return out


def test_sr2_fwhm_bias_is_subnanometre(tex, fwhm_bias):
    worst = max(abs(v) for v in fwhm_bias["SR2"])
    assert worst < 1.0, f"SR2 FWHM bias reaches {worst:.2f} nm; paper claims sub-nanometre"


def test_classical_fwhm_bias_range(tex, fwhm_bias):
    arr = np.array([fwhm_bias[k] for k in CLASSICAL])
    lo, hi = arr.min(), arr.max()
    assert f"${lo:.0f}$--${hi:.0f}$\\,nm" in tex, \
        f"paper does not state the classical FWHM bias range {lo:.0f}-{hi:.0f} nm"
    assert f"${lo:.0f}$\\,nm" in tex, \
        f"paper does not state the best classical FWHM bias of {lo:.0f} nm"
    assert "$16$\\,nm" not in tex, "the pre-retune 16 nm best-classical FWHM survives"


# ── flux ratios (Section 4.5) ─────────────────────────────────────────────────
# Ratios of *integrated* fluxes, sqrt(2 pi) A sigma, with Hbeta and [O III]
# from the joint fit (specsrbench.build.lines).  They were ratios of fitted
# amplitudes until 2026-10-07, which is not what a flux ratio is: SR2 looked
# biased (its lines are narrower than the classical ones by different factors
# at Halpha and Hbeta) when in flux it is unbiased and noisy.
def _flux_ratio_stats(fits, num, den, strict=None):
    def col(name, line, q):
        return fits[f"{name}_{line}_{q}"]
    out = {}
    for name in SNR_KEY.values():
        v = np.ones(len(col(name, num, "flux")), dtype=bool)
        for who in ("HR target", name):
            for line in (num, den):
                v &= np.isfinite(col(who, line, "flux")) & (col(who, line, "flux_amp") > 0.01)
        if strict is not None:
            v &= strict
        rm = np.log10(col(name, num, "flux")[v] / col(name, den, "flux")[v])
        rh = np.log10(col("HR target", num, "flux")[v] / col("HR target", den, "flux")[v])
        out[name] = {"mae": float(np.mean(np.abs(rm - rh))),
                     "bias": float(np.median(rm - rh)),
                     "std": float(np.std(rm)), "std_hr": float(np.std(rh)),
                     "n": int(v.sum())}
    return out


@pytest.fixture(scope="module")
def flux_ratio_stats(fits):
    return {"balmer": _flux_ratio_stats(fits, "Halpha", "Hbeta"),
            "o3hb": _flux_ratio_stats(fits, "OIII5007", "Hbeta")}


@pytest.fixture(scope="module")
def flux_ratio_logmae(flux_ratio_stats):
    return {k: {m: s["mae"] for m, s in v.items()} for k, v in flux_ratio_stats.items()}


def _two_worst(d):
    return set(sorted(d, key=d.get)[-2:])


def test_flux_ratio_ranking(tex, flux_ratio_logmae):
    for key, d in flux_ratio_logmae.items():
        assert min(d, key=d.get) == "Cubic (LR)", \
            f"interpolation no longer has the smallest {key} error"
        assert _two_worst(d) == {"ML (SR2)", "Wiener + MF"}, \
            f"SR2 and the matched filter are no longer the two worst on {key}"
    assert _said(tex, "No method improves on the input")
    assert _said(tex, "are the two worst on both")
    assert not re.search(r"second of (the )?eight", tex), \
        "paper still ranks SR2 second on the Balmer decrement (the amplitude-ratio result)"
    assert "integrated line fluxes" in " ".join(tex.split())


def test_flux_ratio_bias_and_scatter(tex, flux_ratio_stats):
    b, o = flux_ratio_stats["balmer"], flux_ratio_stats["o3hb"]
    for d in (b, o):
        assert abs(d["ML (SR2)"]["bias"]) < 0.01, "SR2's median ratio is no longer unbiased"
    for k in CLASSICAL:
        bias = b[SNR_KEY[k]]["bias"]
        assert 0.05 < bias < 0.25, f"{k}'s Balmer offset is {bias:+.2f}, paper says ~0.1 high"
    assert np.median([b[SNR_KEY[k]]["bias"] for k in CLASSICAL]) == pytest.approx(0.1, abs=0.03)
    sr2 = b["ML (SR2)"]
    assert 1.8 < sr2["std"] / sr2["std_hr"] < 2.6, "SR2's Balmer scatter is not about twice"
    assert f"${sr2['std']:.2f}$ vs.\\ ${sr2['std_hr']:.2f}$" in tex, \
        f"paper does not state the Balmer widths {sr2['std']:.2f} vs {sr2['std_hr']:.2f}"


def test_sr2_integrated_flux_range(tex, fits):
    vals = []
    for line in LINES:
        h, s = fits[f"HR target_{line}_flux"], fits[f"ML (SR2)_{line}_flux"]
        v = np.isfinite(h * s) & (fits[f"HR target_{line}_sn"] > 5) & (h > 0)
        vals.append(100.0 * float(np.median(s[v] / h[v])))
    assert f"integrated flux {min(vals):.0f}--{max(vals):.0f}\\%" in " ".join(tex.split()), \
        f"paper does not state the SR2 flux range {min(vals):.0f}-{max(vals):.0f}%"


def test_flux_ratio_spans(tex, flux_ratio_logmae):
    allv = (list(flux_ratio_logmae["balmer"].values())
            + list(flux_ratio_logmae["o3hb"].values()))
    m = re.search(r"\{\\sim\}([0-9.]+)\$--\$([0-9.]+)\\,\\mathrm\{dex\}", tex)
    assert m, "paper does not state the overall log-ratio error range"
    assert float(m.group(1)) == pytest.approx(min(allv), abs=0.02)
    assert float(m.group(2)) == pytest.approx(max(allv), abs=0.02)


# ── line width as a ratio, and line centres (Section 4.6) ─────────────────────
FIG5_LINES = ["Halpha", "OIII5007", "Hbeta", "OII3727"]      # figure 5's column order
FIG5_CLASSICAL = ["LR", "Wiener", "Tikhonov", "TV", "RL", "Sparse", "MF"]


@pytest.fixture(scope="module")
def fig5(cache):
    from specsrbench.data import load_cache
    from specsrbench.figures.fig5_per_line_snr import compute
    return compute(load_cache(cache))


def _span(fig5, key, methods, line=None, scale=1.0):
    cols = [FIG5_LINES.index(line)] if line else range(4)
    vals = [scale * fig5[key][m][j] for m in methods for j in cols]
    return min(vals), max(vals)


def _kms(v):
    """How the paper typesets a velocity: thousands separated by {,}."""
    return f"{v:,.0f}".replace(",", "{,}")


def test_reference_line_widths(tex, fits):
    med = [float(np.median(2.355e3 * fits[f"HR target_{ln}_sigma"][
        np.isfinite(fits[f"HR target_{ln}_sigma"]) & (fits[f"HR target_{ln}_sn"] > 5)]))
        for ln in FIG5_LINES]
    lo, hi = np.floor(min(med)), np.ceil(max(med))
    assert _said(tex, f"only ${lo:.0f}$--${hi:.0f}$\\,nm wide"), \
        f"paper does not bracket the reference line widths {min(med):.1f}-{max(med):.1f} nm"


def test_fwhm_relative_bias(tex, fig5):
    """Figure 5 row 3: (FWHM_pred - FWHM_obs) / FWHM_obs, and the ratio it implies."""
    lo, hi = _span(fig5, "fwhm_ratio", FIG5_CLASSICAL)
    assert _said(tex, f"make them ${lo:.0f}$--${hi:.0f}$ times as broad"), \
        f"paper does not state the classical width ratio {lo:.1f}-{hi:.1f}"
    lo, hi = _span(fig5, "fwhm_rel", ["SR2"])
    lo_pct, hi_pct = 100 * round(lo, 1), 100 * round(hi, 1)
    assert _said(tex, f"broadens them by ${lo_pct:.0f}$--${hi_pct:.0f}\\%$"), \
        f"paper does not state the SR2 relative width bias {lo:.2f}-{hi:.2f}"


def test_line_centre_offsets(tex, fig5):
    def stated(pattern):
        m = re.search(pattern, tex)
        assert m, f"paper does not state a centre offset matching {pattern}"
        return float(m.group(1).replace("{,}", ""))

    hi = _span(fig5, "centre_offset", ["SR2"])[1]
    bound = stated(r"line\s+within \$\{\\sim\}([0-9{},]+)\$\\,km")
    assert 0.9 * bound <= hi <= bound, f"paper bounds SR2's centres at {bound}, worst is {hi:.0f}"
    assert hi / 299792.458 <= 0.001, "SR2's worst centre offset exceeds dz/(1+z) = 0.001"

    ha = _span(fig5, "centre_offset", FIG5_CLASSICAL, "Halpha")
    assert ha[0] <= _span(fig5, "centre_offset", ["SR2"], "Halpha")[0], \
        "paper says the classical methods match SR2 on Halpha centres; they do not"
    for line in ("OIII5007", "Hbeta", "OII3727"):
        lo = _span(fig5, "centre_offset", FIG5_CLASSICAL, line)[0]
        assert lo > _span(fig5, "centre_offset", ["SR2"], line)[1], \
            f"a classical {line} centre is as good as SR2's, contrary to the text"

    oiii = _span(fig5, "centre_offset", FIG5_CLASSICAL, "OIII5007")[1]
    weak = max(_span(fig5, "centre_offset", FIG5_CLASSICAL, ln)[1]
               for ln in ("Hbeta", "OII3727"))
    assert oiii == pytest.approx(
        stated(r"up to \$\{\\sim\}([0-9{},]+)\$\\,km\\,s\$\^\{-1\}\$ for \\oiii"), rel=0.05)
    assert weak == pytest.approx(
        stated(r"up to\s+\$\{\\sim\}([0-9{},]+)\$\\,km\\,s\$\^\{-1\}\$ for \\hb"), rel=0.05)


def test_line_centre_shifts_are_systematic(tex, fig5):
    hi = _span(fig5, "centre_bias", FIG5_CLASSICAL, "OIII5007")[1]
    assert hi < 0, "not every classical [O III] centre is shifted blueward"
    assert _said(tex, "which is drawn blueward toward $\\lambda$4959 and \\hb")


def test_line_centre_bound_fractions(tex, fig5):
    """"Many fits reach the bound", so the classical offsets are lower limits."""
    for line in ("Hbeta", "OII3727"):
        frac = _span(fig5, "centre_at_bound", FIG5_CLASSICAL, line, scale=100)[1]
        assert frac > 25, f"only {frac:.0f}% of classical {line} fits reach the bound"
    assert _said(tex, "many fits reach the fitting bound and the offsets are lower limits")


# ── redshift dependence (Section 4.7) ─────────────────────────────────────────
@pytest.fixture(scope="module")
def z_bins(cache, x_high, reconstructions):
    z = np.load(cache / "z_test.npy")
    valid = np.asarray(np.load(SRC / "eval_set.npz", allow_pickle=True)["valid_high"],
                       dtype=bool)
    edges = np.quantile(z, np.linspace(0, 1, 7))
    rows = []
    for i in range(6):
        lo, hi = edges[i], edges[i + 1]
        sel = (z >= lo) & (z <= hi) if i == 5 else (z >= lo) & (z < hi)
        row = {}
        for k, a in reconstructions.items():
            d = np.where(valid[sel], a[sel] - x_high[sel], np.nan)
            row[k] = float(np.nanmean(np.nanmean(np.abs(d), axis=1)))
        rows.append(row)
    return rows


def test_sr2_is_best_in_every_redshift_bin(z_bins):
    for i, row in enumerate(z_bins):
        best = min(row, key=row.get)
        assert best == "ML (SR2)", f"bin {i} is led by {best}, not SR2"


@pytest.fixture(scope="module")
def fig6(cache):
    from specsrbench.data import load_cache
    from specsrbench.figures.fig6_redshift_mae import compute
    return compute(load_cache(cache))


def test_figure6_plots_the_numbers_the_text_quotes(fig6, z_bins):
    """The figure once pooled every pixel, padding included; the text did not."""
    for key, name in [("SR2", "ML (SR2)"), ("Wiener", "Wiener"), ("MF", "Wiener + MF")]:
        got = np.round(fig6["mae"][key], 3)
        want = np.round([row[name] for row in z_bins], 3)
        assert np.array_equal(got, want), f"{key}: figure {got} vs text {want}"


def test_redshift_noise_floor(tex, fig6, cache):
    assert fig6["floor"][0] < fig6["floor"][-1], "the noise floor does not rise with redshift"
    from specsrbench.data import load_cache
    c = load_cache(cache)
    err = np.where(c.valid, c.x_high_err, np.nan)
    floor = np.sqrt(2 / np.pi) * np.nanmean(err, axis=1)
    w = np.nanmean(np.abs(np.where(c.valid, c.arrays["Wiener"] - c.x_high, np.nan)), axis=1)
    r2 = np.corrcoef(floor, w)[0, 1] ** 2
    assert r2 > 0.5, f"the noise floor explains only {100 * r2:.0f}% of Wiener's MAE"
    assert _said(tex, "it accounts for most of the trend")


def test_redshift_knee_is_where_wiener_error_stops_growing(tex, fig6, cache):
    """Noise-subtracted MSE: an unbiased estimate of error against the truth."""
    from specsrbench.data import load_cache
    from specsrbench.figures.fig6_redshift_mae import bin_index
    c = load_cache(cache)
    err2 = np.where(c.valid, c.x_high_err, np.nan) ** 2
    d2 = np.where(c.valid, c.arrays["Wiener"] - c.x_high, np.nan) ** 2
    _edges, idx = bin_index(c.z)
    own = [np.nanmean(d2[idx == b]) - np.nanmean(err2[idx == b]) for b in range(6)]
    peak = int(np.argmax(own))
    assert all(np.diff(own[:peak + 1]) > 0), f"Wiener's own error is not rising to its peak: {own}"
    assert _said(tex, f"grows only up to $z \\approx {fig6['z_median'][peak]:.1f}$"), (
        f"paper does not place the knee at the bin where it peaks "
        f"(z = {fig6['z_median'][peak]:.1f})")


def test_redshift_amplitude_trend(tex, fig6):
    sr2 = fig6["amp"]["SR2"]
    assert sr2[0] > sr2[-1], "SR2's amplitude does not fall with redshift"
    assert _said(tex, f"falls from ${sr2[0]:.2f}$ in the lowest bin "
                      f"to ${sr2[-1]:.2f}$ in the highest"), \
        f"paper does not state SR2's amplitude trend {sr2[0]:.2f}-{sr2[-1]:.2f}"
    cls = np.array([fig6["amp"][k] for k in fig6["amp"] if k != "SR2"])
    assert 0.9 < cls.min() and cls.max() < 1.25, \
        f"classical amplitudes span {cls.min():.2f}-{cls.max():.2f}; paper says near unity"
    assert _said(tex, "every classical method stays near unity")


def test_a_spectrum_of_zeros_beats_the_classical_methods(tex, fig6):
    """Above the stated redshift, and only there, zeros out-score every classical method."""
    cls = [k for k in fig6["mae"] if k != "SR2"]
    beats = [fig6["zero"][b] < min(fig6["mae"][k][b] for k in cls) for b in range(6)]
    first = beats.index(True)
    assert all(beats[first:]) and not any(beats[:first]), f"zeros win in bins {beats}"
    assert _said(tex, f"above $z \\approx {fig6['edges'][first]:.1f}$ it scores "
                      "lower than every classical"), \
        f"paper does not state the redshift above which zeros win (z = {fig6['edges'][first]:.2f})"


def test_redshift_scalefree_cluster(tex, fig6):
    sf = fig6["sf"]
    spread = max(100 * (max(sf[k][b] for k in sf) - min(sf[k][b] for k in sf))
                 / min(sf[k][b] for k in sf) for b in range(6))
    assert spread < 5, f"the scale-free metric separates the methods by {spread:.1f}% in a bin"
    extremes = [sf["SR2"][b] in (min(sf[k][b] for k in sf), max(sf[k][b] for k in sf))
                for b in range(6)]
    assert not all(extremes), "SR2 is the extreme method in every bin"
    assert _said(tex, "no method separates from the others at any redshift")


def test_mf_versus_wiener_direction(tex, z_bins):
    """The paper once claimed the MF degrades faster than Wiener; it does not."""
    worse = [i for i, r in enumerate(z_bins) if r["Wiener + MF"] > r["Wiener"]]
    assert not worse, f"MF is worse than Wiener in bins {worse}; prose says otherwise"
    assert not re.search(r"\\gls\{mf\} degrades faster than the filter", tex), \
        "paper still claims the MF degrades faster than Wiener"


# ── fit-bound artefacts (Section 4.6) ─────────────────────────────────────────
# The sub-nanometre FWHM result only means anything if the widths are measured
# rather than pinned at the fitting bound, so the paper states how often they
# are -- which makes it a number that has to be recomputed like any other.
def test_width_bound_fractions(tex, fits, cache, wave):
    z = np.load(cache / "z_test.npy")
    rest = {"Halpha": 0.6563, "OIII5007": 0.5007,
            "Hbeta": 0.4861, "OII3727": 0.3727}
    dpix = np.gradient(wave)

    def pinned(sig, line):
        mu = rest[line] * (1.0 + z)
        dx = np.interp(mu, wave, dpix)
        return (sig <= 0.5 * dx * 1.01) | (sig >= 0.12 * 0.99)

    worst, both = {}, 0
    for name, key in (("ML (SR2)", "sr2"), ("HR target", "hr")):
        worst[key] = max(
            100.0 * float(np.mean(pinned(fits[f"{name}_{line_key}_sigma"], line_key)
                                  [np.isfinite(fits[f"{name}_{line_key}_sigma"])]))
            for line_key in rest)
    for line_key in rest:
        a, b = fits[f"ML (SR2)_{line_key}_sigma"], fits[f"HR target_{line_key}_sigma"]
        ok = np.isfinite(a) & np.isfinite(b)
        both += int((pinned(a, line_key) & pinned(b, line_key) & ok).sum())

    m = re.search(r"fewer than \$([0-9]+)\\%\$ of the \\gls\{sr2\} and reference fits", tex)
    assert m, "paper does not bound the fraction of fits at the width bound"
    for key in ("sr2", "hr"):
        assert worst[key] < float(m.group(1)), \
            f"{worst[key]:.1f}% of {key} fits sit at the width bound; paper says < {m.group(1)}%"
    assert both < 0.01 * len(z) * len(rest), f"both fits are pinned together in {both} cases"


# ── residual-map streaks (Section 4.1) ────────────────────────────────────────
# The prose called the red diagonals "systematic underestimation of the line
# flux".  The maps plot method - reference on RdBu_r, where red is the method
# reading *high*, so the claim inverted the figure it described.  What the
# streaks show is line flux spread into broad wings, with a thin depleted core
# -- and SR2's core is the most depleted of any method, which is the amplitude
# deficit seen pixel by pixel.  These numbers are what the text now quotes.
RESID_BANDS = {"core": (0.0, 3.0), "wing": (3.0, 10.0)}


@pytest.fixture(scope="module")
def line_residuals(reconstructions, x_high, wave, cache):
    z = np.load(cache / "z_test.npy")
    out = {}
    for line, lam_rest in (("Halpha", 0.6563), ("OIII5007", 0.5007)):
        centers = lam_rest * (1.0 + z)
        d_nm = (wave[None, :] - centers[:, None]) * 1000.0
        on_grid = (centers > wave[0] + 0.05) & (centers < wave[-1] - 0.05)
        per_method = {}
        for name, arr in reconstructions.items():
            resid = np.asarray(arr, float) - x_high
            bands = {}
            for band, (lo, hi) in RESID_BANDS.items():
                m = (np.abs(d_nm) >= lo) & (np.abs(d_nm) < hi) & on_grid[:, None]
                bands[band] = float(np.nanmean(resid[m]))
            per_method[name] = bands
        out[line] = per_method
    return out


def test_residual_streaks_are_wings_not_missing_flux(line_residuals):
    """The direction of the claim, independent of the numbers quoted."""
    for line, per_method in line_residuals.items():
        for name, bands in per_method.items():
            if name == "ML (SR2)":
                continue
            assert bands["wing"] > 0, (
                f"{name} does not read high in the {line} wings; the paper's "
                "description of the red streaks no longer holds")
            assert bands["core"] < 0, f"{name} does not read low in the {line} core"


def test_sr2_core_deficit_is_the_largest_and_wings_the_smallest(tex, line_residuals):
    for line, label in (("Halpha", "Halpha"), ("OIII5007", "[OIII]")):
        per_method = line_residuals[line]
        sr2 = per_method["ML (SR2)"]["core"]
        worst_classical = min(v["core"] for k, v in per_method.items()
                              if k != "ML (SR2)")
        assert sr2 < worst_classical, (
            f"SR2's {label} core deficit ({sr2:.2f}) is no longer the largest "
            f"(classical worst {worst_classical:.2f}); the paper says it is")
    ha = line_residuals["Halpha"]
    wings = [v["wing"] for k, v in ha.items() if k != "ML (SR2)"]
    assert ha["ML (SR2)"]["wing"] < 0.5 * min(wings), \
        "SR2's Halpha wings are no longer almost absent beside the classical ones"
    assert _said(tex, "Its wings are almost absent, but its core deficit is the "
                      "largest of any method")


# ── flux-ratio robustness check (Section 4.5) ────────────────────────────────
@pytest.fixture(scope="module")
def flux_ratio_strict(fits, snr):
    """Log-ratio MAE with both lines detected at S/N > 5 in the reference.

    The amplitude threshold used in the main comparison is weak, and a ratio is
    most sensitive to its faintest line.  This is the check that the section's
    conclusions do not depend on that threshold.
    """
    def one(num, den):
        strict = (snr[f"HR_{num}"] > 5) & (snr[f"HR_{den}"] > 5)
        return {m: s["mae"] for m, s in _flux_ratio_stats(fits, num, den, strict).items()}
    return {"balmer": one("Halpha", "Hbeta"), "o3hb": one("OIII5007", "Hbeta")}


def test_strict_flux_ratio_conclusions_are_unchanged(tex, flux_ratio_strict, flux_ratio_logmae):
    b, o = flux_ratio_strict["balmer"], flux_ratio_strict["o3hb"]
    for key, d in flux_ratio_strict.items():
        assert min(d, key=d.get) == "Cubic (LR)", \
            f"interpolation is no longer best on {key} under the strict cut"
        assert _two_worst(d) == {"ML (SR2)", "Wiener + MF"}, \
            f"SR2 and the matched filter are no longer the two worst on {key} under the strict cut"
    assert _said(tex, "leaves these conclusions unchanged")
    for name in b:
        assert b[name] < flux_ratio_logmae["balmer"][name] + 0.01 and \
            o[name] < flux_ratio_logmae["o3hb"][name] + 0.01, \
            f"{name}'s error does not fall under the strict cut"
    allv = list(b.values()) + list(o.values())
    assert f"${min(allv):.2f}$--${max(allv):.2f}\\,$dex" in tex, \
        f"paper does not state the strict-cut span {min(allv):.2f}-{max(allv):.2f}"


# ── 1D toy demonstration (Section 3.4) ────────────────────────────────────────
# The toy RMSEs are quoted in the prose and were unguarded: the classical
# panels are deterministic (only the CNN panel is not), so there is no reason
# for the paper to state a number the generator does not reproduce.
def test_toy_claims_hold(tex):
    pytest.importorskip("skimage")
    pytest.importorskip("pywt")
    from specsrbench.figures import fig1_toy_methods as toy

    _x, true, lsf, _blurred, obs = toy.make_toy()
    centers = [p[0] for p in toy.TRUE_PARAMS]
    sigma = float(np.mean([p[1] for p in toy.TRUE_PARAMS]))

    def rmse(a):
        return float(np.sqrt(np.mean((a - true) ** 2)))

    def peak(a):
        return float(a[150:210].max() / true[150:210].max())

    got = {
        "matched filter": toy.m_matched_filter(obs, lsf, centers, sigma),
        "Tikhonov": toy.m_tikhonov(obs, lsf, lam=0.01),
    }
    for name, arr in got.items():
        assert _states(tex, rmse(arr), ".3f"), \
            f"paper does not state the toy {name} RMSE of {rmse(arr):.3f}"

    # "Recovers most of the peak amplitude but amplifies the noise."
    rl = toy.m_richardson_lucy(obs, lsf, n_iter=30)
    assert peak(rl) > 0.75, f"Richardson-Lucy recovers only {peak(rl):.0%} of the peak"
    assert rl.min() <= obs.min() * 0.9, \
        "Richardson-Lucy no longer leaves excursions as deep as the noisy input"

    # The toy parameters are conventional, not optimized: stronger Tikhonov
    # regularization buys less ringing and a lower RMSE for a lower peak.
    assert "tuned separately for the toy" not in tex, \
        "the toy parameters are fixed conventional values, not tuned"
    loose, tight = got["Tikhonov"], toy.m_tikhonov(obs, lsf, lam=0.3)
    assert rmse(tight) < rmse(loose) and tight.min() > loose.min(), \
        "stronger Tikhonov regularization no longer reduces the toy ringing"
    assert peak(tight) < peak(loose), \
        "stronger Tikhonov regularization no longer costs peak amplitude"
