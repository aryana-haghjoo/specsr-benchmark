"""The scientific guards, checked against the shipped cache.

Each of these encodes a failure that actually reached a committed analysis.
They are the reason the tuning is constrained rather than a free minimisation
of MAE.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import (CLASSICAL, LINES, STD_RATIO_HI, STD_RATIO_LO,
                      mae, mae_scalefree, std_ratio)

BASE = "Cubic (LR)"          # the no-deconvolution baseline


# ── guard 1: amplitude ────────────────────────────────────────────────────────
@pytest.mark.parametrize("method", CLASSICAL)
def test_classical_methods_preserve_amplitude(method, reconstructions, x_high, valid):
    """No published classical method may rescale the spectrum.

    Regression: Wiener tuned to snr=0.8 had DC gain 0.444 and scored best on
    MAE while sitting at 42% of the reference's amplitude.
    """
    r = std_ratio(reconstructions[method], x_high, valid)
    assert STD_RATIO_LO <= r <= STD_RATIO_HI, \
        f"{method} amplitude ratio {r:.3f} outside [{STD_RATIO_LO}, {STD_RATIO_HI}]"


def test_ml_amplitude_deficit_is_still_present(reconstructions, x_high, valid):
    """The ML model does NOT satisfy the amplitude guard, and the paper says so.

    This is a characterisation test: if SR2 is ever retrained and this passes,
    the paper's central claim about shrinkage needs revisiting rather than the
    test being deleted.
    """
    r = std_ratio(reconstructions["ML (SR2)"], x_high, valid)
    assert r < STD_RATIO_LO, \
        f"SR2 amplitude ratio is now {r:.3f}; the shrinkage argument must be rechecked"


# ── guard 2: line preservation ────────────────────────────────────────────────
@pytest.mark.parametrize("method", CLASSICAL)
def test_classical_methods_do_not_destroy_lines(method, fits):
    """A filter that erases the lines scores well on MAE. Regression: Tikhonov
    at lam=1e6 reached the best MAE with median Halpha S/N of 0.75."""
    key = {"Wiener": "Wiener", "Tikhonov": "Tikhonov", "TV": "TV",
           "R-L": "R-L", "Sparse": "Sparse", "Wiener + MF": "Wiener + MF"}[method]
    for l in LINES:
        got = np.nanmedian(fits[f"{key}_{l}_sn"])
        base = np.nanmedian(fits[f"{BASE}_{l}_sn"])
        assert got >= 0.5 * base, \
            f"{method}/{l}: median S/N {got:.2f} vs baseline {base:.2f}"


# ── guard 3: must not blur ────────────────────────────────────────────────────
def _fwhm_bias(fits, method):
    bias = []
    for l in LINES:
        hr_sn = fits[f"HR target_{l}_sn"]
        sm, sh = fits[f"{method}_{l}_sigma"], fits[f"HR target_{l}_sigma"]
        m = np.isfinite(hr_sn) & (hr_sn > 5) & np.isfinite(sm) & np.isfinite(sh)
        if m.sum():
            bias.append(np.median(2.355 * (sm[m] - sh[m])) * 1000)
    return float(np.mean(bias))


@pytest.mark.parametrize("method", CLASSICAL)
def test_deconvolution_does_not_broaden_lines(method, fits):
    """A deconvolution must not leave lines wider than doing nothing.

    Regression: the unit-gain Wiener filter has max(W)/W(0) == 1 for every
    snr <= 1, making it a pure low-pass.  Tuned on MAE alone it landed at
    snr=0.3 and broadened lines to 55 nm against a 30 nm baseline.
    """
    got, base = _fwhm_bias(fits, method), _fwhm_bias(fits, BASE)
    assert got <= base * 1.10, \
        f"{method} FWHM bias {got:.1f} nm exceeds the no-deconvolution baseline {base:.1f} nm"


def test_wiener_filter_can_amplify(cache):
    """The shipped Wiener setting must actually deconvolve, not just smooth."""
    import json
    import specsrbench.classical  # noqa: F401  (the package must be importable)
    params = json.loads((cache / "classical_params.json").read_text())
    snr = params["tuned"]["wiener"]["snr"]
    f = np.fft.rfftfreq(params["tuned"]["wiener"]["segment_len"])
    H = np.exp(-2.0 * (np.pi * f * 25.0) ** 2)
    W = H / (H ** 2 + 1.0 / snr)
    assert W.max() / W[0] > 1.0, \
        f"Wiener at snr={snr} is a pure low-pass (max gain {W.max() / W[0]:.3f}); it cannot sharpen"


# ── the headline result ───────────────────────────────────────────────────────
def test_scalefree_spread_is_small(reconstructions, x_high, valid):
    """The paper's claim: on a scale-free metric no method stands out."""
    vals = {k: mae_scalefree(v, x_high, valid) for k, v in reconstructions.items()}
    spread = (max(vals.values()) - min(vals.values())) / min(vals.values())
    assert spread < 0.05, f"scale-free spread is {spread:.1%}, paper claims ~1%"


def test_ml_does_not_lead_on_the_scalefree_metric(reconstructions, x_high, valid):
    """The paper states SR2 ranks near the bottom once shrinkage is removed."""
    vals = {k: mae_scalefree(v, x_high, valid) for k, v in reconstructions.items()}
    order = sorted(vals, key=vals.get)
    assert order.index("ML (SR2)") >= len(order) - 3, \
        f"SR2 now ranks {order.index('ML (SR2)') + 1} of {len(order)} on scale-free MAE"
    assert vals["ML (SR2)"] > vals[BASE], \
        "SR2 now beats cubic interpolation scale-free; the paper's claim must be updated"


def test_ml_still_leads_on_raw_mae(reconstructions, x_high, valid):
    """The other half of the paper's point: raw MAE ranks it first."""
    vals = {k: mae(v, x_high, valid) for k, v in reconstructions.items()}
    assert min(vals, key=vals.get).startswith("ML"), \
        "raw MAE no longer favours the ML model; the contrast in the paper is stale"


def test_summary_csv_matches_recomputation(cache, reconstructions, x_high, valid):
    """The committed summary must agree with a fresh computation."""
    import csv
    path = cache / "summary_final.csv"
    if not path.exists():
        pytest.skip("summary_final.csv not present")
    rows = {r["Method"]: r for r in csv.DictReader(path.open())}
    for name, arr in reconstructions.items():
        if name not in rows:
            continue
        assert float(rows[name]["MAE"]) == pytest.approx(mae(arr, x_high, valid), abs=1e-3)
        assert float(rows[name]["std_ratio"]) == pytest.approx(
            std_ratio(arr, x_high, valid), abs=1e-3)


# ── guard 4: must not merge a resolvable line pair ────────────────────────────
# Guards 1-3 are all statistics of a Gaussian fitted to a single line, and a
# Gaussian fitted to a *blended* doublet has much the same amplitude, S/N and
# width as one fitted to a separated pair.  All three passed while Wiener,
# Tikhonov and Wiener+TV resolved the [OIII] doublet in 0% of the spectra where
# their own input resolves it in 43%.  Peak structure is the only thing that
# sees this, so it is measured from peaks and not from a fit.
OIII_5007 = 0.5007
PAIR_MIN_UM = 3.0


def _pair_resolved(y, wave, z_i):
    from scipy.signal import find_peaks

    c = OIII_5007 * (1.0 + z_i)
    m = (wave >= c - 0.045) & (wave <= c + 0.045)
    seg = np.asarray(y)[m]
    if seg.size < 5 or not np.isfinite(seg).all() or np.nanmax(seg) <= 0:
        return False
    pk, _ = find_peaks(seg, prominence=0.10 * np.nanmax(seg))
    return len(pk) >= 2


@pytest.fixture(scope="session")
def pair_survival(cache, wave, x_high, reconstructions):
    """Fraction of resolvable [OIII] pairs each method keeps resolved.

    Selection is on the grating truth alone -- the pair must actually be there
    to be destroyed -- over the part of the band where it is wider than the LSF.
    """
    z = np.load(cache / "z_test.npy")
    idx = [i for i in np.where(OIII_5007 * (1.0 + z) > PAIR_MIN_UM)[0]
           if _pair_resolved(x_high[i], wave, z[i])]
    assert len(idx) > 30, f"only {len(idx)} resolvable pairs; the guard is toothless"
    return {k: float(np.mean([_pair_resolved(v[i], wave, z[i]) for i in idx]))
            for k, v in reconstructions.items()}


@pytest.mark.parametrize("method", CLASSICAL)
def test_deconvolution_does_not_merge_close_line_pairs(method, pair_survival, cache):
    """A filter must not destroy a separation carried by its own input.

    A method is allowed to fail this only if the tuner recorded that no setting
    passes it -- an undocumented failure means the caches drifted from the
    parameters, which is exactly how the merging methods reached the paper.
    """
    import json

    got, base = pair_survival[method], pair_survival[BASE]
    if got >= base:
        return
    key = {"Wiener": "wiener", "Tikhonov": "tikhonov", "TV": "tv",
           "R-L": "rl", "Sparse": "sparse", "Wiener + MF": "mf"}[method]
    failed = json.loads((cache / "classical_params.json").read_text()).get(
        "failed_guards", {})
    assert key in failed, (
        f"{method} keeps only {got:.0%} of resolvable pairs against a "
        f"{base:.0%} no-deconvolution baseline, and no failure is recorded "
        f"for it in classical_params.json")


def test_the_merging_guard_has_teeth(pair_survival):
    """At least one method must beat the baseline, or the measure is degenerate."""
    base = pair_survival["Cubic (LR)"]
    better = [k for k, v in pair_survival.items() if v > base]
    assert better, "no method exceeds the no-deconvolution pair baseline"
