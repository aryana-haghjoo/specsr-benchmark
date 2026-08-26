"""Shared fixtures.

Tests are split by what they need:

* ``test_metrics`` and ``test_algorithms`` are pure -- synthetic arrays only,
  no data files, no specsr, no torch.  They run anywhere in seconds.
* ``test_cache_integrity``, ``test_invariants`` and ``test_paper_consistency``
  need ``cache_logR_tuned/`` and skip cleanly without it.
* ``test_figures`` is the end-to-end pass -- it builds all six figures and
  checks the numbers they print -- and is marked ``slow``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "cache_logR_tuned"
SRC = REPO / "cache_logR"

# Import the package from the source tree when it is not installed, so the
# suite runs on a fresh clone without a build step.
sys.path.insert(0, str(REPO / "src"))

METHODS = ["Cubic (LR)", "Wiener", "Tikhonov", "TV", "R-L", "Sparse",
           "Wiener + MF", "ML (SR2)", "HR target"]
CLASSICAL = ["Wiener", "Tikhonov", "TV", "R-L", "Sparse", "Wiener + MF"]
LINES = ["Halpha", "OIII5007", "Hbeta", "OII3727"]

# Amplitude guard.  A reconstruction outside this band is either shrinking the
# spectrum toward zero or inflating it, and MAE cannot tell you which.
STD_RATIO_LO, STD_RATIO_HI = 0.90, 1.15


def _need(path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO)} not present")


@pytest.fixture(scope="session")
def cache():
    _need(CACHE / "x_high.npy")
    return CACHE


@pytest.fixture(scope="session")
def eval_set():
    _need(SRC / "eval_set.npz")
    return np.load(SRC / "eval_set.npz", allow_pickle=True)


@pytest.fixture(scope="session")
def valid(eval_set):
    return np.asarray(eval_set["valid_high"], dtype=bool)


@pytest.fixture(scope="session")
def x_high(cache):
    return np.load(cache / "x_high.npy")


@pytest.fixture(scope="session")
def fits(cache):
    _need(cache / "fit_params_cache.npz")
    return np.load(cache / "fit_params_cache.npz")


@pytest.fixture(scope="session")
def reconstructions(cache):
    """Every method's reconstruction, in the target's normalised units."""
    ml = np.load(cache / "ml_inference_cache.npz")
    out = {
        "Cubic (LR)": np.load(cache / "x_low.npy"),
        "Wiener": np.load(cache / "wiener_cache.npy"),
        "Tikhonov": np.load(cache / "tikhonov_cache.npy"),
        "TV": np.load(cache / "tv_cache.npy"),
        "R-L": np.load(cache / "rl_cache.npy"),
        "Sparse": np.load(cache / "sparse_cache.npy"),
        "Wiener + MF": np.load(cache / "mf_cache.npy"),
        "ML (SR1)": ml["sr1_mean"],
        "ML (SR2)": ml["sr2_mean"],
    }
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


@pytest.fixture(scope="session")
def sigma_pix(cache):
    return np.load(cache / "sigma_pix.npy")


@pytest.fixture(scope="session")
def wave(cache):
    return np.load(cache / "wl_high.npy")


# ── metrics under test ────────────────────────────────────────────────────────
def mae(pred, truth, mask):
    d = np.where(mask, np.asarray(pred, float) - truth, np.nan)
    return float(np.nanmean(np.nanmean(np.abs(d), axis=1)))


def std_ratio(pred, truth, mask):
    return (float(np.nanstd(np.where(mask, pred, np.nan)))
            / float(np.nanstd(np.where(mask, truth, np.nan))))


def mae_scalefree(pred, truth, mask):
    """MAE after rescaling each spectrum by its own least-squares optimal gain.

    Invariant under any global rescale of ``pred``, by construction, which is
    the whole point: shrinkage cannot improve it.
    """
    p = np.asarray(pred, float)
    num = np.nansum(np.where(mask, p * truth, np.nan), axis=1)
    den = np.nansum(np.where(mask, p * p, np.nan), axis=1)
    k = np.where(den > 0, num / den, 1.0)[:, None]
    return float(np.nanmean(np.nanmean(np.abs(np.where(mask, p * k - truth, np.nan)), axis=1)))
