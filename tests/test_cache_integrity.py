"""Structural checks on cache_logR_tuned/, the cache every figure reads.

These catch the class of bug where a file is present and loadable but wrong:
sentinel values mistaken for data, a grid that is not what it claims, arrays
that disagree on how many spectra there are.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import CACHE, LINES, METHODS, SRC

N_EXPECTED = 572
PIX_EXPECTED = 6671


def test_all_arrays_agree_on_shape(reconstructions, x_high):
    assert x_high.shape == (N_EXPECTED, PIX_EXPECTED)
    for name, arr in reconstructions.items():
        assert arr.shape == x_high.shape, f"{name} has shape {arr.shape}"


def test_no_non_finite_values(reconstructions):
    for name, arr in reconstructions.items():
        bad = (~np.isfinite(arr)).sum()
        assert bad == 0, f"{name} has {bad} non-finite values"


def test_grid_is_log_at_constant_R(wave):
    """The grid must be the specsr DEFAULT_GRID, not a linear one."""
    assert wave.shape == (PIX_EXPECTED,)
    R = wave[:-1] / np.diff(wave)
    assert R.min() > 3900 and R.max() < 4100, \
        f"resolving power spans {R.min():.0f}-{R.max():.0f}, expected ~4000"
    assert wave[0] == pytest.approx(1.0, abs=1e-6)
    assert wave[-1] == pytest.approx(5.3, abs=1e-3)
    # a linear grid would have constant spacing; this must not
    d = np.diff(wave)
    assert d.max() / d.min() > 4, "grid spacing is nearly constant -- is this linear?"


def test_normalisation_is_self_consistent(eval_set, x_high):
    """x_high must equal (flux_high - hi_mean)/hi_std, the scale everything uses."""
    fh = np.asarray(eval_set["flux_high"], float)
    m = np.asarray(eval_set["hi_mean"], float)
    s = np.asarray(eval_set["hi_std"], float)
    assert np.abs(x_high - (fh - m) / s).max() < 1e-5


def test_flux_errors_have_no_sentinels_inside_the_mask(cache, valid):
    """Regression: flux_high_err marks invalid pixels with 1.0 against fluxes of
    ~1e-21.  Left unmasked it drove the mean normalised uncertainty to 3e18."""
    d = np.load(cache / "flux_high_err.npz")
    fe = d["flux_high_err"]
    inside = fe[valid]
    inside = inside[np.isfinite(inside)]
    assert inside.max() < 1e-15, \
        f"sentinel-scale error ({inside.max():.2e}) inside valid_high"
    std = np.nanstd(d["flux_high"], axis=1)
    ratio = np.nanmean(fe / np.where(std < 1e-25, 1e-25, std)[:, None])
    assert 0.1 < ratio < 5.0, f"mean normalised flux uncertainty is {ratio:.3g}"


def test_split_sets_are_disjoint():
    """The tuning set must share no galaxy with the evaluation set, or the
    classical methods are tuned on their own test data."""
    if not (SRC / "tune_set.npz").exists():
        pytest.skip("cache_logR/ not present")
    ev = np.load(SRC / "eval_set.npz", allow_pickle=True)
    tu = np.load(SRC / "tune_set.npz", allow_pickle=True)
    for key in ("parent_id", "row_index"):
        a, b = set(np.asarray(ev[key]).tolist()), set(np.asarray(tu[key]).tolist())
        assert not (a & b), f"tune and eval sets share {len(a & b)} {key} values"
    if (SRC / "calib_set.npz").exists():
        ca = np.load(SRC / "calib_set.npz", allow_pickle=True)
        shared = set(np.asarray(ev["row_index"]).tolist()) & set(np.asarray(ca["row_index"]).tolist())
        assert not shared, f"calib and eval share {len(shared)} rows"


def test_eval_galaxies_are_unique(eval_set):
    """One spectrum per galaxy: augmented siblings must not be in the eval set."""
    pid = np.asarray(eval_set["parent_id"])
    assert len(np.unique(pid)) == len(pid), "duplicate parent_id in the eval set"


def test_fit_params_covers_every_method_and_line(fits):
    for m in METHODS:
        for l in LINES:
            for k in ("amp", "sigma", "sn"):
                key = f"{m}_{l}_{k}"
                assert key in fits.files, f"missing {key}"
                assert fits[key].shape == (N_EXPECTED,)


def test_line_fits_mostly_succeed(fits):
    """A method whose fits collapse is broken even if the array is present."""
    for m in METHODS:
        for l in LINES:
            frac = np.isfinite(fits[f"{m}_{l}_sn"]).mean()
            assert frac > 0.85, f"{m}/{l}: only {frac:.0%} of fits succeeded"


def test_fitted_widths_are_not_pinned_to_the_bound(fits, wave, cache):
    """A width fitted at its bound is not a measurement.

    The floor tracks the grid (half a pixel).  It used to be a fixed 0.001 um,
    chosen when a pixel was 0.0016 um; on this grid a pixel is 0.00025 um at
    1 um and a real R=2700 line has sigma ~0.00016 um, so the old floor pinned
    the reference's own widths at the boundary and reported them too wide.
    """
    z = np.load(cache / "z_test.npy")
    rest = dict(zip(LINES, (0.6563, 0.5007, 0.4861, 0.3727)))
    for l in LINES:
        sig = fits[f"HR target_{l}_sigma"]
        ok = np.isfinite(sig)
        mu = rest[l] * (1.0 + z[ok])
        dx = np.interp(mu, wave, np.gradient(wave))
        at_bound = (sig[ok] <= 0.5 * dx * 1.01).mean()
        assert at_bound < 0.10, \
            f"HR {l}: {at_bound:.0%} of fitted widths sit at the fit floor"
        # and the old fixed floor must genuinely have been too coarse here
        assert np.median(sig[ok]) < 0.0016, \
            f"HR {l} median width {np.median(sig[ok]):.5f} um -- grid may not be the log one"


# ── the kernel the caches were deconvolved with ───────────────────────────────
# Added 2026-08-25.  eval_set.npz ships a sigma_pix that nothing generates and
# that does not describe the data: it is roughly constant in nanometres, where a
# spectrograph's LSF is fixed in detector pixels.  For a day the tuner used the
# measured kernel while build_classical_logR.py went on building the caches with
# the shipped one, and nothing in this suite could tell -- every guard was
# evaluated against whichever array happened to be in the cache.

def test_cache_kernel_is_the_measured_one(cache, sigma_pix):
    """sigma_pix.npy must be the derived LSF, not the shipped array."""
    measured = cache / "sigma_pix_measured.npy"
    if not measured.exists():
        pytest.skip("sigma_pix_measured.npy not present "
                    "(run `specsrbench build lsf`)")
    np.testing.assert_allclose(
        sigma_pix, np.load(measured), rtol=1e-9,
        err_msg="the cache's kernel is not the one derive_lsf.py measured")


def test_kernel_is_constant_in_detector_pixels(sigma_pix, wave, eval_set):
    """The measured LSF is flat in prism pixels; the shipped one is flat in nm.

    This is the property that distinguishes them, so assert it directly rather
    than comparing to a stored copy of the right answer.
    """
    sigma_nm = sigma_pix * np.gradient(wave) * 1e3
    shipped_nm = np.asarray(eval_set["sigma_pix"], float) * np.gradient(wave) * 1e3
    band = (wave > 1.5) & (wave < 5.0)
    spread = sigma_nm[band].max() / sigma_nm[band].min()
    shipped_spread = shipped_nm[band].max() / shipped_nm[band].min()
    assert spread > 2.0, (
        f"the cache's LSF varies by only {spread:.2f}x in nm across 1.5-5 um; "
        "a kernel that is flat in nanometres is the shipped one, not the data's")
    assert shipped_spread < 1.5, "the shipped array is no longer flat in nm; recheck"


def test_caches_were_built_with_the_parameters_they_record(cache, reconstructions):
    """Rebuild a slice of each cache from classical_params.json and compare.

    A number is only safe if a test recomputes it; the same is true of a cache.
    This is the check that was missing when the recorded parameters and the
    shipped caches disagreed -- the tuner had retuned every method against the
    derived kernel and the caches were still the pre-retune build.
    """
    import json
    from specsrbench import classical as C

    params = json.loads((cache / "classical_params.json").read_text())
    assert params["kernel"].startswith("derived"), \
        f"caches record kernel {params['kernel']!r}; run `specsrbench build lsf`"
    tuned = params["tuned"]
    sig, _ = C.load_sigma_pix(cache, np.load(SRC / "eval_set.npz", allow_pickle=True))
    x_low = np.load(cache / "x_low.npy")[:4]

    for name, fn, key in [("Wiener", C.wiener_deconv, "wiener"),
                          ("Tikhonov", C.tikhonov_deconv, "tikhonov"),
                          ("R-L", C.rl_deconv, "rl")]:
        got = np.stack([fn(s, sig, **tuned[key]) for s in x_low])
        np.testing.assert_allclose(
            got, reconstructions[name][:4], rtol=0, atol=1e-4,
            err_msg=(f"{name}_cache.npy does not reproduce from the recorded "
                     f"parameters {tuned[key]} and the recorded kernel -- the "
                     "cache and classical_params.json are out of step"))
