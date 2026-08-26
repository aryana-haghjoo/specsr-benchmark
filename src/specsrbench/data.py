"""Loading the tuned cache: one object, loaded once, shared by every figure.

Each of the six notebooks opened the cache itself, with its own ``load()``
helper and its own list of ``.npy`` names.  The lists had to be kept in step by
hand, and the error message when one drifted was ``FileNotFoundError`` naming a
file that had simply been renamed.

Arrays load lazily and are held after first access, so building all six figures
in one process reads the 267 MB cache once rather than six times.
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

from . import paths
from .methods import LINES, METHODS

__all__ = ["Cache", "load_cache"]


class MissingCache(FileNotFoundError):
    """Raised with the command that regenerates what is missing."""


class Cache:
    """The arrays every figure reads, from one directory."""

    def __init__(self, directory: Path | str | None = None):
        self.dir = Path(directory) if directory is not None else paths.cache_dir()

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _load(self, name: str, *, rebuild: str = "specsrbench build all"):
        p = self.dir / name
        if not p.exists():
            raise MissingCache(
                f"missing {p}\n"
                f"  the cache directory is {self.dir}\n"
                f"  set SPECSRBENCH_CACHE to point elsewhere, or rebuild with:\n"
                f"      {rebuild}")
        return np.load(p, allow_pickle=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Cache {self.dir} n={self.n_spectra} pix={self.n_pixels}>"

    # ── grids and labels ──────────────────────────────────────────────────────
    @cached_property
    def wl_high(self) -> np.ndarray:
        """The log constant-R grid, in microns (6,671 points, R=4000)."""
        return self._load("wl_high.npy")

    @cached_property
    def wl_low(self) -> np.ndarray:
        return self._load("wl_low.npy")

    @cached_property
    def z(self) -> np.ndarray:
        """Spectroscopic redshift of each held-out galaxy."""
        return self._load("z_test.npy")

    @cached_property
    def sigma_pix(self) -> np.ndarray:
        """The LSF the caches were actually deconvolved with, in pixels.

        A copy of the derived kernel, written by the build so that nothing
        downstream has to infer which kernel produced the arrays beside it.
        ``eval_set.npz`` ships a different ``sigma_pix`` that does not describe
        the data; never read that one.
        """
        return self._load("sigma_pix.npy")

    @property
    def n_spectra(self) -> int:
        return int(len(self.z))

    @property
    def n_pixels(self) -> int:
        return int(len(self.wl_high))

    # ── reconstructions ───────────────────────────────────────────────────────
    @cached_property
    def _ml(self):
        return self._load("ml_inference_cache.npz")

    @cached_property
    def arrays(self) -> dict[str, np.ndarray]:
        """Every method's reconstruction, keyed by canonical name.

        In the target's normalised (per-spectrum z-scored) units, which is what
        makes the MAE columns comparable across spectra of wildly different
        brightness.
        """
        out: dict[str, np.ndarray] = {}
        for m in METHODS:
            if m.array is not None:
                out[m.key] = np.asarray(self._load(m.array), dtype=np.float64)
        out["SR2"] = np.asarray(self._ml["sr2_mean"], dtype=np.float64)
        return out

    @cached_property
    def sr1(self) -> np.ndarray:
        return np.asarray(self._ml["sr1_mean"], dtype=np.float64)

    @cached_property
    def zhat(self) -> np.ndarray:
        """The redshift head's estimate, as SR2 conditioned on it."""
        return np.asarray(self._ml["zhat"], dtype=np.float64)

    @property
    def x_high(self) -> np.ndarray:
        """The grating reference every method is scored against."""
        return self.arrays["HR"]

    @property
    def x_low(self) -> np.ndarray:
        """The prism input, cubic-interpolated onto the high-resolution grid."""
        return self.arrays["LR"]

    # ── line measurements ─────────────────────────────────────────────────────
    @cached_property
    def snr(self) -> dict[str, np.ndarray]:
        """Per-line S/N, keyed ``{method_key}_{line_key}`` (9 x 4 = 36)."""
        f = self._load("snr.npz", rebuild="specsrbench build lines")
        return {k: f[k] for k in f.files}

    @cached_property
    def fits(self) -> dict[str, np.ndarray]:
        """Gaussian fit parameters, keyed ``{label}_{line}_{amp,sigma,sn}``."""
        f = self._load("fit_params_cache.npz", rebuild="specsrbench build lines")
        return {k: f[k] for k in f.files}

    # ── flux uncertainties ────────────────────────────────────────────────────
    @cached_property
    def _err(self):
        return self._load("flux_high_err.npz")

    @cached_property
    def valid(self) -> np.ndarray:
        """Pixels where the grating reference is real, not padding."""
        return np.asarray(self._err["valid_high"], dtype=bool)

    @cached_property
    def x_high_err(self) -> np.ndarray:
        """Reference flux uncertainty, in the same normalised units as ``x_high``.

        Scaled by each spectrum's own standard deviation, matching the z-score
        applied to the fluxes; the mean is *not* subtracted, because an
        uncertainty is a width and has no offset.

        Invalid pixels arrive as NaN.  They are marked in the raw product by a
        sentinel of 1.0 against fluxes of order 1e-21; left unmasked, that one
        value drives the mean normalised uncertainty to 3e18 instead of ~0.5.
        The masking happens at build time, and
        ``tests/test_cache_integrity.py`` asserts it here.
        """
        err = np.asarray(self._err["flux_high_err"], dtype=np.float64)
        std = np.asarray(self._err["hi_std"], dtype=np.float64).reshape(-1, 1)
        return err / np.where(std < 1e-25, 1e-25, std)

    def x_high_err_floored(self) -> tuple[np.ndarray, float, float]:
        """``(errors, floor, mean)`` with a 1st-percentile floor applied.

        A handful of pixels carry an uncertainty of essentially zero, and
        dividing a residual by one of those produces an ``inf`` that swallows
        the mean.  The floor is the 1st percentile of the finite positive
        values, computed on this cache rather than hard-coded.
        """
        e = self.x_high_err
        finite = e[np.isfinite(e) & (e > 0)]
        floor = float(np.nanpercentile(finite, 1))
        safe = np.maximum(e, floor)
        return safe, floor, float(np.nanmean(safe))

    # ── convenience ───────────────────────────────────────────────────────────
    def line_positions(self, i: int) -> list[tuple[str, str, float]]:
        """``(key, label, observed wavelength)`` for lines inside the grid."""
        z = float(self.z[i])
        lo, hi = float(self.wl_high.min()), float(self.wl_high.max())
        return [(k, lab, rest * (1 + z)) for k, lab, rest in LINES
                if lo + 0.05 < rest * (1 + z) < hi - 0.05]

    def summary(self) -> str:
        return (f"Loaded {self.n_spectra} held-out spectra from {self.dir.name}\n"
                f"  grid  {self.wl_high.min():.2f}-{self.wl_high.max():.2f} um "
                f"({self.n_pixels} pix, log R=4000)\n"
                f"  z     [{self.z.min():.3f}, {self.z.max():.3f}]")


_CACHED: dict[Path, Cache] = {}


def load_cache(directory: Path | str | None = None) -> Cache:
    """The cache at ``directory``, reusing an already-loaded one if possible."""
    d = Path(directory) if directory is not None else paths.cache_dir()
    if d not in _CACHED:
        _CACHED[d] = Cache(d)
    return _CACHED[d]
