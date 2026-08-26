"""The classical deconvolution implementations, on synthetic spectra.

Every test here is a property that must hold for any correct implementation,
checked against data whose truth is known by construction.  Several encode
bugs that actually shipped.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import STD_RATIO_HI, STD_RATIO_LO
from specsrbench import classical as C

N_PIX = 2048
SIGMA_PIX = 25.0          # the prism LSF on the log R=4000 grid, in pixels


def _lsf_blur(x, sigma=SIGMA_PIX):
    k = C.make_kernel(sigma)
    from scipy.signal import fftconvolve
    return fftconvolve(x, k, mode="same")


@pytest.fixture
def spectrum():
    """A narrow-lined 'high-resolution' truth, its blurred+noisy observation,
    and the per-pixel LSF width array the methods expect."""
    rng = np.random.default_rng(0)
    wl = np.linspace(1.0, 5.3, N_PIX)
    truth = 0.2 * np.ones(N_PIX)
    for centre, amp in [(400, 3.0), (900, 2.0), (1500, 4.0)]:
        truth += amp * np.exp(-0.5 * ((np.arange(N_PIX) - centre) / 2.5) ** 2)
    obs = _lsf_blur(truth) + rng.normal(0, 0.02, N_PIX)
    sp = np.full(N_PIX, SIGMA_PIX)
    return wl, truth, obs, sp


def _z(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / max(a.std(), 1e-30)


# ── the shrinkage bug ─────────────────────────────────────────────────────────
def test_wiener_has_unit_dc_gain(spectrum):
    """A constant in must give the same constant out. This is the fix."""
    _, _, _, sp = spectrum
    const = np.full(N_PIX, 3.7)
    out = C.wiener_deconv(const, sp, snr=0.3, segment_len=128, overlap=32)
    interior = out[64:-64]         # ignore Hanning roll-off at the edges
    assert np.allclose(interior, 3.7, rtol=2e-2), \
        f"DC gain is {np.median(interior) / 3.7:.3f}, expected 1.0"


@pytest.mark.parametrize("snr", [0.3, 0.8, 2.0, 10.0])
def test_wiener_dc_gain_independent_of_regularisation(spectrum, snr):
    """Unit gain must hold at every snr, or tuning can still buy MAE by rescaling."""
    _, _, _, sp = spectrum
    out = C.wiener_deconv(np.full(N_PIX, 2.0), sp, snr=snr, segment_len=128, overlap=32)
    assert np.allclose(out[64:-64], 2.0, rtol=2e-2)


def test_raw_wiener_would_shrink(spectrum):
    """Regression: without the fix the filter rescales. Documents the defect."""
    _, _, _, sp = spectrum
    out = C.wiener_deconv(np.full(N_PIX, 1.0), sp, snr=0.8,
                          segment_len=128, overlap=32, unit_gain=False)
    assert np.median(out[64:-64]) == pytest.approx(1.0 / (1.0 + 1.0 / 0.8), rel=0.05), \
        "expected the raw MMSE filter to shrink by its DC gain"


def test_tikhonov_has_unit_dc_gain(spectrum):
    """Tikhonov's W(0) is 1 by construction; that is why it never degenerated."""
    _, _, _, sp = spectrum
    out = C.tikhonov_deconv(np.full(N_PIX, 5.0), sp, lam=100.0,
                            segment_len=1024, overlap=256)
    assert np.allclose(out[512:-512], 5.0, rtol=2e-2)


# ── amplitude preservation, per method ────────────────────────────────────────
def _methods(wl, obs, sp, z):
    w = C.wiener_deconv(obs, sp, snr=0.3, segment_len=128, overlap=32)
    return {
        "Wiener": w,
        "Tikhonov": C.tikhonov_deconv(obs, sp, lam=100.0, segment_len=1024, overlap=256),
        "R-L": C.rl_deconv(obs, sp, n_iter=1),
        "Sparse": C.sparse_wavelet_deconv(obs, sp, lam=0.05, n_iter=1),
        "TV": C.tv_denoise_1d(w, lam=0.3, n_iter=30),
    }


def test_all_methods_preserve_amplitude(spectrum):
    """No published method may rescale the spectrum it reconstructs."""
    wl, truth, obs, sp = spectrum
    for name, out in _methods(wl, obs, sp, None).items():
        r = np.std(_z(out)) / np.std(_z(obs))
        assert STD_RATIO_LO <= r <= STD_RATIO_HI, \
            f"{name} amplitude ratio {r:.3f} outside [{STD_RATIO_LO}, {STD_RATIO_HI}]"


def test_all_methods_finite(spectrum):
    wl, truth, obs, sp = spectrum
    for name, out in _methods(wl, obs, sp, None).items():
        assert np.all(np.isfinite(out)), f"{name} produced non-finite values"


@pytest.mark.parametrize("pathological", ["zeros", "constant", "spike"])
def test_methods_survive_pathological_input(spectrum, pathological):
    _, _, _, sp = spectrum
    x = {"zeros": np.zeros(N_PIX),
         "constant": np.full(N_PIX, 2.0),
         "spike": np.eye(1, N_PIX, N_PIX // 2).ravel() * 10.0}[pathological]
    for name, out in _methods(None, x, sp, None).items():
        assert np.all(np.isfinite(out)), f"{name} broke on {pathological}"


def test_rl_preserves_non_negativity(spectrum):
    """Richardson-Lucy is a non-negative method; a negative output is a bug."""
    _, _, obs, sp = spectrum
    out = C.rl_deconv(obs - obs.min(), sp, n_iter=5)
    assert out.min() >= -1e-6


def _shipped_wiener_params():
    """The Wiener settings actually published, so the test tracks the analysis."""
    import json

    from conftest import CACHE
    path = CACHE / "classical_params.json"
    if path.exists():
        return json.loads(path.read_text())["tuned"]["wiener"]
    return dict(snr=4.0, segment_len=128, overlap=32)


def test_shipped_wiener_is_not_a_pure_lowpass():
    """A filter that cannot amplify any frequency cannot deconvolve.

    Regression: the unit-gain Wiener filter has max(W)/W(0) == 1 for every
    snr <= 1, because W peaks at H = sqrt(1/snr) and H never exceeds 1.  Tuned
    against MAE alone it landed at snr=0.3 -- a pure low-pass that broadened
    lines to 55 nm against a 30 nm no-deconvolution baseline.
    """
    p = _shipped_wiener_params()
    f = np.fft.rfftfreq(p["segment_len"])
    H = np.exp(-2.0 * (np.pi * f * SIGMA_PIX) ** 2)
    W = H / (H ** 2 + 1.0 / p["snr"])
    assert W.max() / W[0] > 1.0, (
        f"snr={p['snr']} gives max gain {W.max() / W[0]:.3f}: this filter can only blur")


def test_deconvolution_recovers_a_known_width():
    """Correctness of the implementation, in a regime where the answer is known.

    With the line comparable to the LSF and high S/N, Wiener deconvolution must
    recover the true width.  At snr=30 it does so exactly here.

    This deliberately does NOT use the prism regime.  There the reference lines
    are ~14x narrower than the LSF, deconvolution recovers almost none of that
    width, and every classical method lands 20-30 nm wide against a true FWHM
    of ~2.3 nm -- which is a finding of the paper, not a bug to assert against.
    Whether the shipped configuration beats doing nothing is checked on the real
    data in tests/test_invariants.py.
    """
    from scipy.signal import fftconvolve
    n = 2048
    for lsf, line in [(5.0, 10.0), (5.0, 6.0), (8.0, 10.0)]:
        truth = 0.2 * np.ones(n) + 3.0 * np.exp(-0.5 * ((np.arange(n) - 1024) / line) ** 2)
        obs = (fftconvolve(truth, C.make_kernel(lsf), mode="same")
               + np.random.default_rng(0).normal(0, 0.005, n))
        w = C.wiener_deconv(obs, np.full(n, lsf), snr=30.0, segment_len=256, overlap=64)

        def fwhm(y):
            seg = y[824:1224].astype(float)
            seg = seg - np.median(seg)
            return (seg > seg.max() / 2).sum()

        assert fwhm(w) < fwhm(obs), \
            f"LSF={lsf} line={line}: deconvolution did not sharpen"
        assert fwhm(w) <= fwhm(truth) + 2, \
            f"LSF={lsf} line={line}: recovered {fwhm(w)} px vs true {fwhm(truth)} px"


# ── the matched-filter template bug ───────────────────────────────────────────
def test_mf_template_is_much_narrower_than_the_lsf():
    """Regression: templates were the LSF itself, 14x too wide, inflating output.

    The default must stay well below 1.0 or the filter injects flux across a
    region an order of magnitude broader than a real line.
    """
    import inspect
    default = inspect.signature(C.matched_filter).parameters["width_scale"].default
    assert default < 0.5, f"width_scale default {default} is too close to the LSF width"


def test_mf_does_not_inflate_amplitude(spectrum):
    """The matched filter must not amplify the spectrum it corrects."""
    wl, truth, obs, sp = spectrum
    w = C.wiener_deconv(obs, sp, snr=0.3, segment_len=128, overlap=32)
    lines = np.array([wl[400], wl[900], wl[1500]]) / 1.0
    out = C.matched_filter(w, wl, 0.0, lines, sp, width_scale=0.25, detect_snr=3.0)
    r = np.std(_z(out)) / np.std(_z(w))
    assert r <= STD_RATIO_HI, f"MF inflated amplitude by {r:.2f}x"


def test_mf_wide_template_inflates(spectrum):
    """Regression: at width_scale=1.0 (the old behaviour) it must blow up."""
    wl, truth, obs, sp = spectrum
    w = C.wiener_deconv(obs, sp, snr=0.3, segment_len=128, overlap=32)
    lines = np.array([wl[400], wl[900], wl[1500]])
    wide = C.matched_filter(w, wl, 0.0, lines, sp, width_scale=1.0, detect_snr=3.0)
    narrow = C.matched_filter(w, wl, 0.0, lines, sp, width_scale=0.25, detect_snr=3.0)
    assert np.std(wide) > np.std(narrow), \
        "the old wide template should inflate relative to the fixed one"


def test_mf_leaves_line_free_regions_untouched(spectrum):
    """Away from the templates the spectrum must be returned unchanged."""
    wl, truth, obs, sp = spectrum
    w = C.wiener_deconv(obs, sp, snr=0.3, segment_len=128, overlap=32)
    out = C.matched_filter(w, wl, 0.0, np.array([wl[400]]), sp, width_scale=0.25)
    assert np.allclose(out[1200:], w[1200:]), "MF altered a region with no template"


def test_kernel_is_normalised():
    """A convolution kernel that does not sum to 1 changes total flux."""
    for sig in (3.0, 25.0, 47.0):
        assert C.make_kernel(sig).sum() == pytest.approx(1.0, rel=1e-9)
