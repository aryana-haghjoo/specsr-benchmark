"""Reconstruction metrics, and the four guards that keep tuning honest.

MAE alone is not a safe objective, in four separate ways, and no one guard
catches the others:

1. *Smoothing* -- erase every line and no line-shaped residual is incurred.
2. *Shrinkage* -- scale toward zero and absolute error against a noisy
   reference falls, whatever the reconstruction quality.
3. *Blurring* -- a unit-gain Wiener filter with ``snr <= 1`` peaks at zero
   frequency, so it can only broaden lines, and the amplitude guard is
   width-invariant.
4. *Merging* -- all three of the above can pass while two close lines are
   smeared into one peak, because a Gaussian fitted to a blended doublet has
   much the same amplitude, S/N and width as one fitted to a separated pair.

:func:`mae_scalefree` and :func:`std_ratio` are what make (2) visible, and they
live here rather than in a build script because the tuner, the cache build, the
figures and the tests all have to agree on them.  They did not, once: the tuner
retuned every method against a corrected kernel while the build went on using
the old parameters, and every classical number in the paper was wrong for a day
while the whole test suite passed.
"""
from __future__ import annotations

import numpy as np

__all__ = ["zscore", "mae", "mae_scalefree", "std_ratio", "rmse", "bias",
           "bootstrap_std", "fwhm_from_sigma", "global_stats"]

#: Amplitude guard band.  Outside it a reconstruction is shrinking toward zero
#: or inflating, and MAE cannot tell you which.
STD_RATIO_LO, STD_RATIO_HI = 0.90, 1.15


def zscore(arr: np.ndarray) -> np.ndarray:
    """Per-spectrum z-score, the units every cached reconstruction is in."""
    a = np.asarray(arr, dtype=np.float64)
    mu = np.nanmean(a, axis=1, keepdims=True)
    sd = np.nanstd(a, axis=1, keepdims=True)
    return (a - mu) / np.where(sd < 1e-25, 1e-25, sd)


def _masked_diff(pred, truth, mask) -> np.ndarray:
    return np.where(mask, np.asarray(pred, dtype=np.float64) - truth, np.nan)


def mae(pred, truth, mask) -> float:
    """Mean over spectra of the per-spectrum mean absolute error."""
    return float(np.nanmean(np.nanmean(np.abs(_masked_diff(pred, truth, mask)), axis=1)))


def std_ratio(pred, truth, mask) -> float:
    """Output scale over target scale.  1.0 means amplitude was preserved."""
    return (float(np.nanstd(np.where(mask, pred, np.nan)))
            / float(np.nanstd(np.where(mask, truth, np.nan))))


def mae_scalefree(pred, truth, mask) -> float:
    """MAE after rescaling each spectrum by its own least-squares optimal gain.

    Invariant under any global rescale of ``pred`` by construction, which is
    the entire point: shrinkage cannot improve it.  This is the metric on which
    SR2 ranks eighth of nine while leading the raw-MAE table by 30%.
    """
    p = np.asarray(pred, dtype=np.float64)
    num = np.nansum(np.where(mask, p * truth, np.nan), axis=1)
    den = np.nansum(np.where(mask, p * p, np.nan), axis=1)
    k = np.where(den > 0, num / den, 1.0)[:, None]
    return float(np.nanmean(np.nanmean(
        np.abs(np.where(mask, p * k - truth, np.nan)), axis=1)))


def rmse(pred, truth, mask) -> float:
    d = _masked_diff(pred, truth, mask)
    return float(np.sqrt(np.nanmean(d[np.isfinite(d)] ** 2)))


def bias(pred, truth, mask) -> float:
    d = _masked_diff(pred, truth, mask)
    return float(np.nanmean(d[np.isfinite(d)]))


def bootstrap_std(per_spectrum, n_boot: int = 1000, seed: int = 42) -> float:
    """Standard deviation of the mean under resampling of spectra.

    Spectra, not pixels: pixels within one spectrum are correlated, and
    resampling them would report an error bar several times too small.
    """
    v = np.asarray(per_spectrum, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(v), size=(n_boot, len(v)), replace=True)
    return float(np.nanstd(np.nanmean(v[idx], axis=1)))


def fwhm_from_sigma(sigma_um) -> np.ndarray:
    """Gaussian FWHM in nanometres from a fitted sigma in microns."""
    return 2.355 * np.asarray(sigma_um, dtype=np.float64) * 1e3


def global_stats(name: str, pred, truth, mask, *, n_boot: int = 1000,
                 seed: int = 42) -> dict:
    """One row of the global-fidelity table, guards included.

    ``MAE`` and ``MAE_scalefree`` are reported together, always.  Quoting the
    first without the second is how this paper twice came to state a conclusion
    that the scale-free number reverses.
    """
    a = np.asarray(pred, dtype=np.float64)
    v = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(truth)
    d = np.where(v, a - truth, np.nan)
    mae_by_spec = np.nanmean(np.abs(d), axis=1)
    return {
        "Method": name,
        "MAE": float(np.nanmean(mae_by_spec)),
        "MAE_err": bootstrap_std(mae_by_spec, n_boot=n_boot, seed=seed),
        "MAE_scalefree": mae_scalefree(a, truth, v),
        "std_ratio": std_ratio(a, truth, v),
        "RMSE": rmse(a, truth, v),
        "Bias": bias(a, truth, v),
    }
