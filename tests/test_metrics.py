"""The metrics themselves, on synthetic data.

These exist because the analysis was once wrong for a reason no data-level
check could catch: MAE against a noisy reference is partly minimised by
shrinking the estimate toward zero.  The tests below pin down which metric has
that weakness and which does not, so the guard can never be silently dropped.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import mae, mae_scalefree, std_ratio


@pytest.fixture
def synthetic():
    """A noisy 'reconstruction' of a noisy reference, on 200 x 500 pixels."""
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 1, (200, 500))
    pred = truth + rng.normal(0, 0.8, (200, 500))
    mask = np.ones_like(truth, dtype=bool)
    return pred, truth, mask


def test_mae_is_lowered_by_shrinking(synthetic):
    """MAE rewards shrinkage. This is the defect the analysis walked into.

    If this test ever fails, the premise behind the amplitude guard has changed
    and the guard should be re-derived rather than removed.
    """
    pred, truth, mask = synthetic
    full = mae(pred, truth, mask)
    best_k, best = 1.0, full
    for k in np.linspace(0.1, 1.0, 19):
        m = mae(pred * k, truth, mask)
        if m < best:
            best_k, best = k, m
    assert best < full, "expected some k<1 to beat k=1 on MAE"
    assert best_k < 0.95, f"optimal shrink factor {best_k:.2f} unexpectedly close to 1"


def test_scalefree_mae_is_invariant_under_rescaling(synthetic):
    """The scale-free metric cannot be moved by a global rescale, at all."""
    pred, truth, mask = synthetic
    base = mae_scalefree(pred, truth, mask)
    for k in (0.1, 0.44, 0.5, 2.0, 10.0):
        assert mae_scalefree(pred * k, truth, mask) == pytest.approx(base, rel=1e-9), \
            f"scale-free MAE moved under rescale by {k}"


def test_scalefree_mae_still_penalises_real_error(synthetic):
    """Invariance to scale must not mean insensitivity to everything."""
    pred, truth, mask = synthetic
    rng = np.random.default_rng(1)
    worse = pred + rng.normal(0, 1.5, pred.shape)
    assert mae_scalefree(worse, truth, mask) > mae_scalefree(pred, truth, mask)


def test_std_ratio_detects_shrinkage(synthetic):
    """The amplitude diagnostic must catch what MAE misses."""
    pred, truth, mask = synthetic
    assert std_ratio(pred * 0.44, truth, mask) == pytest.approx(
        0.44 * std_ratio(pred, truth, mask), rel=1e-9)
    assert std_ratio(pred * 0.44, truth, mask) < 0.9, \
        "a 0.44 rescale must fall outside the amplitude guard"


def test_guard_would_have_rejected_the_shipped_bug(synthetic):
    """Regression: the exact failure that shipped.

    A Wiener filter at snr=0.8 has DC gain 1/(1+1/0.8) = 0.444.  MAE preferred
    it; the amplitude guard must reject it.
    """
    from conftest import STD_RATIO_LO, STD_RATIO_HI
    pred, truth, mask = synthetic
    dc_gain = 1.0 / (1.0 + 1.0 / 0.8)
    assert dc_gain == pytest.approx(0.4444, abs=1e-3)
    shrunk = pred * dc_gain
    assert mae(shrunk, truth, mask) < mae(pred, truth, mask), \
        "the bug's premise: shrinking improved MAE"
    assert not (STD_RATIO_LO <= std_ratio(shrunk, truth, mask) <= STD_RATIO_HI), \
        "amplitude guard failed to reject the shipped bug"


def test_nan_masking_is_respected():
    """Masked pixels must not contribute. The flux_high_err sentinel bug."""
    rng = np.random.default_rng(2)
    truth = rng.normal(0, 1, (10, 100))
    pred = truth.copy()
    mask = np.ones_like(truth, dtype=bool)
    mask[:, :20] = False
    pred[:, :20] = 1e18          # sentinel garbage outside the mask
    assert mae(pred, truth, mask) == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(std_ratio(pred, truth, mask))
