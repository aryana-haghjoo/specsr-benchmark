"""The results prose must match the cache, not just the summary table.

The global-fidelity table has been guarded since the first rebuild and stayed
correct.  The paragraphs around it were not guarded, and after the classical
methods were retuned every classical number in Sections 4.3-4.7 was left
quoting the pre-retune cache -- line S/N ratios, detection fractions,
false-detection rates, FWHM biases, flux-ratio rankings and the redshift bins.
Two of those stale numbers had also inverted a qualitative claim.  These tests
recompute each quoted quantity and assert the manuscript states it.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from conftest import REPO, SRC

PAPER = REPO / "paper.tex"
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
    for line, val in zip(LINES, snr_ratios["SR2"]):
        assert _states(tex, val), f"paper does not state SR2 {line} S/N ratio {val:.2f}"


def test_classical_line_snr_range(tex, snr_ratios):
    arr = np.array([snr_ratios[k] for k in CLASSICAL])
    lo, hi = arr.min(), arr.max()
    assert f"${lo:.2f}$ to\n${hi:.2f}$" in tex or f"${lo:.2f}$ to ${hi:.2f}$" in tex, \
        f"paper does not state the classical S/N ratio range {lo:.2f}-{hi:.2f}"


def test_classical_line_snr_leaders(tex, snr_ratios):
    """Which classical method leads each line changed when they were retuned."""
    arr = np.array([snr_ratios[k] for k in CLASSICAL])
    for j, line in enumerate(LINES):
        best = CLASSICAL[int(np.argmax(arr[:, j]))]
        val = arr[:, j].max()
        if line in ("Halpha", "OIII5007", "OII3727"):
            assert _states(tex, val), \
                f"paper does not state the best classical {line} ratio {val:.2f} ({best})"


def test_median_halpha_snr(tex, snr):
    hr, me = snr["HR_Halpha"], snr["SR2_Halpha"]
    real = np.isfinite(hr) & (hr > 5)
    a = float(np.median(me[real & np.isfinite(me)]))
    b = float(np.median(hr[real]))
    assert _states(tex, a, ".1f"), f"paper does not state SR2 median Halpha S/N {a:.1f}"
    assert _states(tex, b, ".1f"), f"paper does not state HR median Halpha S/N {b:.1f}"


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
    assert _states(tex, ha, ".1f") or f"{ha:.1f}\\%" in tex, \
        f"paper does not state the Halpha amplitude recovery {ha:.1f}%"


# ── detection fractions (Section 4.3) ─────────────────────────────────────────
def test_detection_fractions(tex, snr):
    for line in ("Hbeta", "OII3727"):
        for key in ("SR2", "HR"):
            v = 100.0 * float(np.nanmean(snr[f"{key}_{line}"] > 5))
            assert f"{v:.1f}\\%" in tex, \
                f"paper does not state the {key} {line} detection fraction {v:.1f}%"
    arr = np.array([[100.0 * float(np.nanmean(snr[f"{k}_{line_key}"] > 5))
                     for line_key in ("Hbeta", "OII3727")] for k in CLASSICAL])
    lo, hi = arr.min(), arr.max()
    assert f"{lo:.1f}\\%" in tex and f"{hi:.1f}\\%" in tex, \
        f"paper does not state the classical weak-line detection span {lo:.1f}-{hi:.1f}%"


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
    for line, val in zip(LINES, fdr["SR2"]):
        if line in ("Hbeta", "OII3727"):
            assert _states(tex, val, ".3f"), \
                f"paper does not state the SR2 {line} FDR {val:.3f}"


def test_classical_weak_line_fdr_bounds(tex, fdr):
    arr = np.array([fdr[k] for k in CLASSICAL])
    for j, line in enumerate(LINES):
        if line not in ("Hbeta", "OII3727"):
            continue
        lo, hi = arr[:, j].min(), arr[:, j].max()
        assert _states(tex, lo, ".3f"), \
            f"paper does not state the classical {line} FDR floor {lo:.3f}"
        assert _states(tex, hi, ".3f"), \
            f"paper does not state the classical {line} FDR ceiling {hi:.3f}"


def test_no_classical_method_exceeds_the_stated_weak_line_fdr(tex, fdr):
    arr = np.array([fdr[k] for k in CLASSICAL])
    worst = 100.0 * max(arr[:, LINES.index("Hbeta")].max(),
                        arr[:, LINES.index("OII3727")].max())
    m = re.search(r"manufactures a weak line at more than an? \$?([0-9]+)\\%\$? rate", tex)
    assert m, "paper does not bound the classical weak-line false-detection rate"
    assert worst <= float(m.group(1)), \
        f"paper claims no classical method exceeds {m.group(1)}%, cache says {worst:.1f}%"


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
@pytest.fixture(scope="module")
def flux_ratio_logmae(fits):
    def one(num, den):
        hn, hd = fits[f"HR target_{num}_amp"], fits[f"HR target_{den}_amp"]
        out = {}
        for name in SNR_KEY.values():
            mn, md = fits[f"{name}_{num}_amp"], fits[f"{name}_{den}_amp"]
            v = (np.isfinite(hn) & np.isfinite(hd) & np.isfinite(mn) & np.isfinite(md)
                 & (hn > 0.01) & (hd > 0.01) & (mn > 0.01) & (md > 0.01))
            out[name] = float(np.mean(np.abs(np.log10(mn[v] / md[v])
                                             - np.log10(hn[v] / hd[v]))))
        return out
    return {"balmer": one("Halpha", "Hbeta"), "o3hb": one("OIII5007", "Hbeta")}


def test_balmer_ranking_is_stated_honestly(tex, flux_ratio_logmae):
    b = flux_ratio_logmae["balmer"]
    order = sorted(b, key=b.get)
    rank = order.index("ML (SR2)") + 1
    assert rank > 1, "SR2 now leads the Balmer decrement; the prose says it does not"
    assert not re.search(r"nominally the best of any method", tex), \
        "paper still calls the ML Balmer log-MAE the best of any method"
    for name in order[:rank]:
        assert _states(tex, b[name], ".3f"), \
            f"paper does not state the Balmer log-MAE {b[name]:.3f} for {name}"


def test_flux_ratio_spans(tex, flux_ratio_logmae):
    b = flux_ratio_logmae["balmer"]
    lo, hi = min(b.values()), max(b.values())
    assert f"${lo:.3f}$--${hi:.3f}$" in tex, \
        f"paper does not state the Balmer field span {lo:.3f}-{hi:.3f}"
    allv = list(b.values()) + list(flux_ratio_logmae["o3hb"].values())
    m = re.search(r"\{\\sim\}([0-9.]+)\$--\$([0-9.]+)\\,\\mathrm\{dex\}", tex)
    assert m, "paper does not state the overall log-ratio error range"
    assert float(m.group(1)) == pytest.approx(min(allv), abs=0.02)
    assert float(m.group(2)) == pytest.approx(max(allv), abs=0.02)


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
            if k == "ML (SR1)":
                continue
            d = np.where(valid[sel], a[sel] - x_high[sel], np.nan)
            row[k] = float(np.nanmean(np.nanmean(np.abs(d), axis=1)))
        rows.append(row)
    return rows


def test_sr2_is_best_in_every_redshift_bin(tex, z_bins):
    for i, row in enumerate(z_bins):
        best = min(row, key=row.get)
        assert best == "ML (SR2)", f"bin {i} is led by {best}, not SR2"
    assert _states(tex, z_bins[0]["ML (SR2)"], ".3f"), \
        f"paper does not state the lowest-bin SR2 MAE {z_bins[0]['ML (SR2)']:.3f}"
    assert _states(tex, z_bins[-1]["ML (SR2)"], ".3f"), \
        f"paper does not state the highest-bin SR2 MAE {z_bins[-1]['ML (SR2)']:.3f}"


def test_wiener_band_across_redshift(tex, z_bins):
    lo = z_bins[0]["Wiener"]
    hi = z_bins[-1]["Wiener"]
    assert _states(tex, lo, ".3f") and _states(tex, hi, ".3f"), \
        f"paper does not state the Wiener redshift band {lo:.3f}-{hi:.3f}"


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

    for key in ("sr2", "hr"):
        assert f"${worst[key]:.1f}\\%$" in tex, \
            f"paper does not state the {key} width-bound fraction {worst[key]:.1f}%"
    n_fits = f"{len(z) * len(rest):,}".replace(",", "{,}")
    assert f"${both}$ of the ${n_fits}$" in tex, \
        (f"paper does not state that both fits are pinned together in {both} of "
         f"{len(z) * len(rest)} line fits")
