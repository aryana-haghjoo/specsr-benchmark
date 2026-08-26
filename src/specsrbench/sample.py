"""The small tutorial dataset: 24 held-out spectra, fetched from the Hub.

The benchmark proper reads a 267 MB cache that this package does not ship and
most readers will never rebuild -- it needs the raw JADES DR4 tree, the Hub
weights and a few CPU-hours.  That is a poor first experience for someone who
has just run ``pip install specsrbench`` and wants to see what the thing does.

This module is the other end of that scale.  One 1.9 MB archive, downloaded on
first use and cached by ``huggingface_hub`` thereafter, carrying everything the
benchmark needs on a small subset: the prism input, the grating reference, the
measured line-spread function, the SR2 prediction, and the tuned classical
parameters.  Every classical baseline runs on it in seconds with no torch, no
survey data and no configuration.

What the numbers off it mean
----------------------------
24 galaxies, not 572.  The *ordering* of the methods reproduces and so does the
lesson -- SR2 leads raw MAE by ~30% at 0.58 of the reference amplitude, and
ranks last once that is corrected for -- but the individual figures carry the
error bar of a 24-spectrum sample and are not the paper's.  Quote the paper for
the paper's numbers.

The galaxies are the evaluation set sorted by redshift and sampled at evenly
spaced ranks, so they are held out by construction and span z = 0.31 to 13.86
rather than being chosen for how good they look.  ``tests/test_tutorial_sample.py``
checks that against the split.

.. warning::

   ``sigma_pix`` here is the *derived* kernel, the one
   :func:`specsrbench.classical.load_sigma_pix` returns.  The evaluation set
   ships a different array under that name which does not describe the data --
   roughly constant in nanometres where a real spectrograph is fixed in
   detector pixels, and up to 2.3x too broad at 5 um.  Deconvolving with it
   merges line pairs the input still resolves.  Nothing in this archive carries
   that array.
"""
from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path

import numpy as np

from .methods import LINES

__all__ = ["Sample", "load_sample", "sample_path", "DEFAULT_REPO", "FILENAME"]

#: Hub dataset repo holding the archive.  Override with ``SPECSRBENCH_SAMPLE_REPO``.
DEFAULT_REPO = "aryana-haghjoo/specsr-benchmark"
#: Branch, tag or commit.  Override with ``SPECSRBENCH_SAMPLE_REVISION``.
DEFAULT_REVISION = "main"
#: The one file in it.
FILENAME = "specsrbench_sample.npz"

#: Which deconvolver runs each classical method, and which parameter block in
#: the archive configures it.  The parameters are *not* restated here: they are
#: read from the archive, which copied them from the tuner's own output file.
#: Two hand-maintained copies of these six numbers is exactly what made every
#: classical number in the paper wrong for a day on 2026-08-24.
_CLASSICAL = ("Wiener", "Tikhonov", "TV", "R-L", "Sparse", "Wiener + MF")


def sample_path(repo_id: str | None = None, revision: str | None = None) -> Path:
    """Local path to the archive, downloading it if necessary.

    ``SPECSRBENCH_SAMPLE`` points at a local ``.npz`` and wins outright: no
    network, no Hub account, and the way to work against a rebuilt sample
    before it is published.
    """
    if (local := os.environ.get("SPECSRBENCH_SAMPLE")):
        p = Path(local).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                f"SPECSRBENCH_SAMPLE is set to {local!r}, which does not exist")
        return p

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "huggingface_hub is missing. It is a *base* dependency of "
            "specsrbench, so this usually means a partial install:\n"
            "    pip install --upgrade --force-reinstall specsrbench\n"
            "Or work offline -- download\n"
            f"    https://huggingface.co/datasets/{DEFAULT_REPO}/blob/main/{FILENAME}\n"
            "by hand and point SPECSRBENCH_SAMPLE at it."
        ) from exc

    return Path(hf_hub_download(
        repo_id=repo_id or os.environ.get("SPECSRBENCH_SAMPLE_REPO", DEFAULT_REPO),
        filename=FILENAME,
        repo_type="dataset",
        revision=(revision or os.environ.get("SPECSRBENCH_SAMPLE_REVISION",
                                             DEFAULT_REVISION)),
    ))


class Sample:
    """24 held-out spectra and everything needed to benchmark against them.

    Attributes mirror :class:`specsrbench.data.Cache` where they mean the same
    thing, so code written against one mostly reads on the other.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._d = np.load(self.path, allow_pickle=True)

    # ── grids and labels ──────────────────────────────────────────────────────
    @cached_property
    def wave(self) -> np.ndarray:
        """The log constant-R grid, in microns (6,671 points, R = 4000)."""
        return np.asarray(self._d["wave"], dtype=np.float64)

    @cached_property
    def sigma_pix(self) -> np.ndarray:
        """The measured LSF width, in detector pixels, per grid point."""
        return np.asarray(self._d["sigma_pix"], dtype=np.float64)

    @cached_property
    def z(self) -> np.ndarray:
        """Spectroscopic redshift, ascending."""
        return np.asarray(self._d["z"], dtype=np.float64)

    @cached_property
    def z_pred(self) -> np.ndarray:
        """The redshift head's estimate, which SR2 was conditioned on."""
        return np.asarray(self._d["z_pred"], dtype=np.float64)

    @cached_property
    def mf_lines(self) -> np.ndarray:
        """Rest wavelengths the matched filter places templates at, in microns.

        Carried here so the matched filter runs without ``specsr``, which owns
        the line list and pulls in torch.
        """
        return np.asarray(self._d["mf_lines"], dtype=np.float32)

    # ── spectra ───────────────────────────────────────────────────────────────
    @cached_property
    def x_low(self) -> np.ndarray:
        """The prism input, cubic-interpolated onto the high-resolution grid.

        This is both the input every deconvolver takes *and* the no-deconvolution
        baseline every one of them has to beat.
        """
        return np.asarray(self._d["x_low"], dtype=np.float64)

    @cached_property
    def x_high(self) -> np.ndarray:
        """The grating reference every method is scored against."""
        return np.asarray(self._d["x_high"], dtype=np.float64)

    @cached_property
    def x_high_err(self) -> np.ndarray:
        """Reference flux uncertainty, normalised like ``x_high``; NaN where invalid."""
        return np.asarray(self._d["x_high_err"], dtype=np.float64)

    @cached_property
    def sr2(self) -> np.ndarray:
        """The SR2 deep-learning prediction, precomputed (no torch needed)."""
        return np.asarray(self._d["sr2"], dtype=np.float64)

    @cached_property
    def valid(self) -> np.ndarray:
        """Pixels where the grating reference is real, not padding."""
        return np.asarray(self._d["valid"], dtype=bool)

    # ── metadata ──────────────────────────────────────────────────────────────
    @cached_property
    def params(self) -> dict:
        """The tuned classical parameters, as ``classical_params.json`` records them."""
        return json.loads(str(self._d["params"]))

    @cached_property
    def tuned(self) -> dict:
        """Just the per-method parameter blocks, keyed as :func:`reconstruct` expects."""
        return self.params["tuned"]

    @cached_property
    def provenance(self) -> dict:
        """Where these spectra came from, and what produced the arrays beside them."""
        return json.loads(str(self._d["provenance"]))

    @property
    def n_spectra(self) -> int:
        return int(len(self.z))

    @property
    def n_pixels(self) -> int:
        return int(len(self.wave))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Sample n={self.n_spectra} pix={self.n_pixels} from {self.path.name}>"

    def summary(self) -> str:
        return (f"{self.n_spectra} held-out spectra from {self.path.name}\n"
                f"  grid  {self.wave.min():.2f}-{self.wave.max():.2f} um "
                f"({self.n_pixels} pix, log R=4000)\n"
                f"  z     [{self.z.min():.3f}, {self.z.max():.3f}]\n"
                f"  LSF   {self.sigma_pix.min():.2f}-{self.sigma_pix.max():.2f} pix "
                f"({self.provenance['kernel']})")

    # ── running the benchmark ─────────────────────────────────────────────────
    def line_positions(self, i: int) -> list[tuple[str, str, float]]:
        """``(key, label, observed wavelength)`` for lines inside the grid."""
        z = float(self.z[i])
        lo, hi = float(self.wave.min()), float(self.wave.max())
        return [(k, lab, rest * (1 + z)) for k, lab, rest in LINES
                if lo + 0.05 < rest * (1 + z) < hi - 0.05]

    def reconstruct(self, methods=None) -> dict[str, np.ndarray]:
        """Run the classical baselines at the tuned parameters.

        Returns ``{display name: array}`` including the two reconstructions that
        need no work -- ``"Cubic (LR)"``, the prism input, and ``"ML (SR2)"``,
        the precomputed prediction -- so the result is the full comparison table
        rather than only the part that had to be computed.

        Takes a few seconds for all six on 24 spectra.
        """
        from . import classical as C

        want = list(methods) if methods is not None else list(_CLASSICAL)
        unknown = [m for m in want if m not in _CLASSICAL]
        if unknown:
            raise KeyError(f"unknown method(s) {unknown}; known: {list(_CLASSICAL)}")

        p, sig, low = self.tuned, self.sigma_pix, self.x_low
        runners = {
            "Wiener":      lambda: [C.wiener_deconv(s, sig, **p["wiener"]) for s in low],
            "Tikhonov":    lambda: [C.tikhonov_deconv(s, sig, **p["tikhonov"]) for s in low],
            "TV":          lambda: [C.tv_denoise_1d(s, **p["tv"]) for s in low],
            "R-L":         lambda: [C.rl_deconv(s, sig, **p["rl"]) for s in low],
            "Sparse":      lambda: [C.sparse_wavelet_deconv(s, sig, **p["sparse"])
                                    for s in low],
            "Wiener + MF": lambda: [C.matched_filter(s, self.wave, zz, self.mf_lines,
                                                     sig, **p["mf"])
                                    for s, zz in zip(low, self.z)],
        }
        out = {"Cubic (LR)": self.x_low}
        for name in want:
            out[name] = np.asarray(runners[name](), dtype=np.float64)
        out["ML (SR2)"] = self.sr2
        return out


def load_sample(path: Path | str | None = None, *, repo_id: str | None = None,
                revision: str | None = None) -> Sample:
    """The tutorial sample, downloading it from the Hub on first use.

    Parameters
    ----------
    path
        A local ``.npz`` to read instead of fetching anything.
    repo_id, revision
        Override the Hub dataset repo and the revision taken from it.
    """
    return Sample(path if path is not None else sample_path(repo_id, revision))
