"""Derive the prism->grating line-spread function from the instrument itself.

``cache_logR/eval_set.npz`` ships a ``sigma_pix`` array that every classical
deconvolution uses as its kernel.  Nothing in either repo generates it, and it
does not describe the data: it is roughly constant in *nanometres* across the
band, whereas a spectrograph has a fixed LSF in *detector pixels*.  Measured
against the paired spectra it is up to 2.3x too broad at 5 um and ~25% too
narrow at 1.5 um, which is enough to stop Wiener, Tikhonov and Wiener+TV
resolving a line pair their own input still resolves.

``x_low`` is real JWST NIRSpec PRISM data and ``x_high`` is real grating data
(specsr resamples both onto the log grid and convolves nothing), so the kernel
relating them is the genuine prism-vs-grating resolution difference.  This
script derives it in three steps:

1. Measure the prism dispersion d(lambda)/dpixel from the raw JADES x1d
   products -- a property of the disperser, not of any target.
2. Measure the effective kernel width from the paired data itself, as
   sigma_eff^2 = sigma_LR^2 - sigma_HR^2 on Gaussian fits to the four
   diagnostic lines.
3. Express (2) in units of (1).  It comes out constant to within a few per
   cent across a factor of three in wavelength, which is the check that this is
   an instrumental LSF and not a curve fitted to noise.  The fitted constant
   times the dispersion is the kernel.

Writes ``sigma_pix_measured.npy`` into the cache directory.  Requires the raw
JADES DR4 tree, which is not in this repository and is not redistributable
here: set ``SPECSR_JADES_ROOT`` or pass ``--jades-root``.

    specsrbench build lsf --jades-root /path/to/JADES/DR4
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

from .. import paths
from . import require_npz

#: Only a default, and only for the machine this was developed on.  Everyone
#: else passes --jades-root or sets SPECSR_JADES_ROOT.
DEFAULT_JADES = Path.home() / "Documents/GitHub/JADES_data/DR4"

LINES = {"Halpha": 0.6563, "OIII5007": 0.5007, "Hbeta": 0.4861, "OII3727": 0.3727}
# Wavelength bins for the kernel measurement.  Wide enough that each holds
# enough well-detected lines to take a stable median.
BINS = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.5)]


def prism_dispersion(jades_root: Path, n_files: int = 40):
    """Median native prism wavelength solution, and d(lambda)/dpixel."""
    from astropy.io import fits

    pat = str(jades_root / "*/spectra/clear-prism/*/*_x1d.fits")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"no prism x1d products under {pat}")
    waves = []
    for f in files[:n_files]:
        try:
            with fits.open(f) as hdul:
                w = np.sort(np.asarray(hdul["EXTRACT3PIX1D"].data["WAVELENGTH"], float))
        except Exception:
            continue
        w = w[np.isfinite(w)]
        if w.size > 600:
            waves.append(w)
    if not waves:
        raise RuntimeError("no readable prism spectra")
    n = min(len(w) for w in waves)
    W = np.median(np.stack([w[:n] for w in waves]), axis=0)
    return W, np.gradient(W), len(waves)


def measured_kernel(fits_cache, z):
    """sigma_eff(lambda) in microns, from Gaussian fits to the paired lines."""
    rows = []
    for line, rest in LINES.items():
        s_hr = fits_cache[f"HR target_{line}_sigma"]
        s_lr = fits_cache[f"Cubic (LR)_{line}_sigma"]
        sn = fits_cache[f"HR target_{line}_sn"]
        lam = rest * (1.0 + z)
        ok = np.isfinite(s_hr) & np.isfinite(s_lr) & (sn > 10) & (s_lr > s_hr)
        for i in np.where(ok)[0]:
            rows.append((lam[i], float(np.sqrt(s_lr[i] ** 2 - s_hr[i] ** 2))))
    return np.asarray(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="specsrbench build lsf")
    ap.add_argument("--jades-root", "--jades", dest="jades", type=Path,
                    default=Path(os.environ.get("SPECSR_JADES_ROOT", DEFAULT_JADES)))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    SRC, OUT = paths.sets_dir(), paths.cache_dir()
    if args.out is None:
        args.out = OUT / "sigma_pix_measured.npy"

    E = require_npz(SRC / "eval_set.npz", "specsrbench build sets")
    wave = np.asarray(E["wave"], float)
    # Absent from any set built by `specsrbench build sets` -- see its
    # docstring.  Still present in the historical one, where the comparison
    # is worth printing because it is the whole reason this stage exists.
    stored = (np.asarray(E["sigma_pix"], float)
              if "sigma_pix" in E.files else None)
    fits_cache = np.load(OUT / "fit_params_cache.npz", allow_pickle=True)
    z = np.load(OUT / "z_test.npy")

    print(f"prism dispersion from {args.jades}")
    Wp, dlp, nfiles = prism_dispersion(args.jades)
    print(f"  averaged {nfiles} x1d products, {Wp.min():.2f}-{Wp.max():.2f} um\n")

    rows = measured_kernel(fits_cache, z)
    print(f"effective kernel from {len(rows)} well-detected lines\n")
    print(f"{'window (um)':>13s}{'n':>5s}{'sigma_eff':>12s}{'prism dlam':>12s}{'prism px':>10s}")
    lam_c, k_px = [], []
    for lo, hi in BINS:
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if m.sum() < 20:
            continue
        lam = float(np.median(rows[m, 0]))
        sig = float(np.median(rows[m, 1]))
        d = float(np.interp(lam, Wp, dlp))
        lam_c.append(lam)
        k_px.append(sig / d)
        print(f"{lo:6.1f}-{hi:5.1f}{int(m.sum()):5d}{sig*1e3:10.2f} nm"
              f"{d*1e3:10.2f} nm{sig/d:10.2f}")

    k = float(np.median(k_px))
    spread = float(np.max(k_px) - np.min(k_px))
    print(f"\n  LSF = {k:.2f} prism pixels (sigma) = {2.355*k:.2f} px FWHM")
    print(f"  spread across the band: {spread:.2f} px "
          f"({100*spread/k:.0f}%) -- a real LSF is constant in detector pixels")
    if spread / k > 0.25:
        print("  WARNING: not constant enough to be an instrumental LSF; "
              "do not trust this kernel", file=sys.stderr)

    sigma_um = k * np.interp(wave, Wp, dlp)
    sigma_pix = sigma_um / np.gradient(wave)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, sigma_pix)

    if stored is None:
        print(f"\n{'lam':>6s}{'derived sigma_pix':>19s}")
        for lam0 in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
            j = int(np.argmin(np.abs(wave - lam0)))
            print(f"{lam0:6.1f}{sigma_pix[j]:19.1f}")
    else:
        print(f"\n{'lam':>6s}{'derived sigma_pix':>19s}{'shipped':>10s}"
              f"{'shipped/derived':>17s}")
        for lam0 in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
            j = int(np.argmin(np.abs(wave - lam0)))
            print(f"{lam0:6.1f}{sigma_pix[j]:19.1f}{stored[j]:10.1f}"
                  f"{stored[j] / sigma_pix[j]:17.2f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
