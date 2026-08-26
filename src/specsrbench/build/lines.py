"""Assemble a drop-in cache on the log constant-R grid: arrays, line fits, S/N.

Produces the cache directory :mod:`specsrbench.data` loads: the reconstructions
as plain arrays, a Gaussian fit of every diagnostic line in every method, the
S/N those imply, and the summary table the paper and the tests are checked
against.

Two departures from the original linear-grid build, both forced by the finer
grid:

* ``sigma_bounds`` had a fixed 0.001 um floor, set when one pixel was 0.001601
  um everywhere.  On this grid a pixel is 0.00025 um at 1 um, and a real R=2700
  grating line has sigma ~0.00016 um -- below the old floor, so the HR
  reference's own lines would fit at the boundary and come out systematically
  too wide.  The floor is now half the local pixel.
* ``flux_high`` / ``flux_high_err`` come from ``eval_set.npz`` rather than the
  135 MB raw dataset, which removed figure 4's only raw-data dependency.

The method label ``R-L (30 it)`` becomes ``R-L``: 30 iterations is not what the
retuned pipeline runs, and a label that states a wrong iteration count is worse
than one that states none.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import curve_fit

from .. import classical as C
from .. import paths
from . import require_npz

SRC = paths.sets_dir()
OUT = paths.cache_dir()
NPROC = 24        # this box is shared; leave headroom

E = require_npz(SRC / "eval_set.npz", "specsrbench build sets")
WAVE = np.asarray(E["wave"], dtype=np.float64)
DPIX = np.gradient(WAVE)
X_LOW = np.asarray(E["x_low"], dtype=np.float64)
X_HIGH = np.asarray(E["x_high"], dtype=np.float64)
VALID = np.asarray(E["valid_high"], dtype=bool)
Z = np.asarray(E["z_true"], dtype=np.float64)
HI_M = np.asarray(E["hi_mean"], dtype=np.float64)
HI_S = np.asarray(E["hi_std"], dtype=np.float64)
N = X_LOW.shape[0]

LINES = [("Halpha", 0.6563), ("OIII5007", 0.5007),
         ("Hbeta", 0.4861), ("OII3727", 0.3727)]

# fit_params_cache label -> snr.npz label
LABELS = [
    ("Cubic (LR)", "LR"),
    ("Wiener", "Wiener"),
    ("Tikhonov", "Tikhonov"),
    ("TV", "TV"),
    ("R-L", "RL"),
    ("Sparse", "Sparse"),
    ("Wiener + MF", "MF"),
    ("ML (SR2)", "SR2"),
    ("HR target", "HR"),
]


def _g(x, amp, mu, sig, c0, c1):
    return c0 + c1 * (x - mu) + amp * np.exp(
        -0.5 * ((x - mu) / np.clip(sig, 1e-12, None)) ** 2)


def fit_gauss(x, y, mu0, fit_halfwin=0.25, sb_gap=0.03, sb_width=0.12,
              core_halfwin=0.05, sigma_hi=0.12, mu_bounds_half=0.01):
    """(amp, sigma, S/N) or (nan, nan, nan).  Lower sigma bound tracks the grid."""
    fit_m = (x >= mu0 - fit_halfwin) & (x <= mu0 + fit_halfwin)
    core_m = np.abs(x - mu0) <= core_halfwin
    sb_m = ((np.abs(x - mu0) >= sb_gap) & (np.abs(x - mu0) <= sb_gap + sb_width)
            & (~core_m) & fit_m)
    xx, yy = x[fit_m], y[fit_m]
    y_sb = y[sb_m & np.isfinite(y)]
    if xx.size < 15 or y_sb.size < 30:
        return np.nan, np.nan, np.nan
    sc = max(1.4826 * np.median(np.abs(y_sb - np.median(y_sb))), 1e-3)
    c0e = float(np.median(y_sb))
    dx = float(np.median(np.diff(xx))) if xx.size > 1 else 0.002
    res = yy - c0e
    hi_r, lo_r = float(np.nanmax(res)), float(np.nanmin(res))
    amp0 = hi_r if abs(hi_r) >= abs(lo_r) else lo_r
    if abs(amp0) < 1e-6:
        amp0 = 1e-6
    sigma_lo = max(0.5 * dx, 1e-6)          # half a pixel, not a fixed 0.001 um
    sig0 = float(np.clip(2 * dx, sigma_lo, sigma_hi))
    lo = [-np.inf, mu0 - mu_bounds_half, sigma_lo, -np.inf, -np.inf]
    hi = [np.inf, mu0 + mu_bounds_half, sigma_hi, np.inf, np.inf]
    try:
        popt, _ = curve_fit(_g, xx, yy, p0=[amp0, mu0, sig0, c0e, 0.0],
                            bounds=(lo, hi), sigma=np.full_like(xx, sc),
                            absolute_sigma=True, maxfev=5000)
        return float(popt[0]), float(popt[2]), abs(float(popt[0])) / sc
    except Exception:
        return np.nan, np.nan, np.nan


def _task(job):
    label, lname, lam0, arr = job
    amps = np.full(N, np.nan)
    sigs = np.full(N, np.nan)
    sns = np.full(N, np.nan)
    clipped = 0
    for i in range(N):
        mu0 = lam0 * (1.0 + Z[i])
        amps[i], sigs[i], sns[i] = fit_gauss(WAVE, arr[i], mu0)
        if np.isfinite(sigs[i]):
            j = int(np.argmin(np.abs(WAVE - mu0)))
            fit_m = (WAVE >= mu0 - 0.25) & (WAVE <= mu0 + 0.25)
            dx = float(np.median(np.diff(WAVE[fit_m]))) if fit_m.sum() > 1 else DPIX[j]
            if sigs[i] <= 0.5 * dx * 1.01 or sigs[i] >= 0.12 * 0.99:
                clipped += 1
    return label, lname, amps, sigs, sns, clipped


def main(argv=None) -> int:
    """Fit every line in every reconstruction and write the summary."""
    global NPROC
    ap = argparse.ArgumentParser(prog="specsrbench build")
    ap.add_argument("--nproc", type=int, default=NPROC,
                    help="worker processes (this box is shared)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be read and written, run nothing")
    args = ap.parse_args([] if argv is None else argv)
    NPROC = args.nproc
    if args.dry_run:
        print(f"  reads  {SRC}\n  writes {OUT}\n  nproc  {NPROC}")
        return 0

    OUT.mkdir(exist_ok=True)
    print(f"Assembling drop-in cache in {OUT}  ({N} spectra x {len(WAVE)} px)\n")

    # ── 1. plain arrays the figures load by name ──────────────────────────────
    np.save(OUT / "wl_high.npy", WAVE)
    np.save(OUT / "x_high.npy", X_HIGH)
    np.save(OUT / "x_low.npy", X_LOW)
    np.save(OUT / "z_test.npy", Z)
    # The kernel the caches were actually deconvolved with, not the one
    # eval_set.npz ships -- writing the shipped array here is what left the
    # guards and the line fits measuring against a different kernel than
    # the caches were built with.
    sigma_pix, kernel_src = C.load_sigma_pix(OUT, E)
    np.save(OUT / "sigma_pix.npy", sigma_pix)
    print(f"  kernel: {kernel_src}")
    np.save(OUT / "wl_low.npy", WAVE)

    np.savez(OUT / "ml_inference_cache.npz",
             sr1_mean=(np.asarray(E["sr1"], dtype=np.float64) - HI_M) / HI_S,
             sr2_mean=(np.asarray(E["sr2"], dtype=np.float64) - HI_M) / HI_S,
             zhat=np.asarray(E["z_pred"], dtype=np.float64))

    # figure 4 normalises residuals by the flux errors; supply them from the
    # eval set so it no longer needs the 135 MB raw dataset.
    #
    # flux_high_err marks invalid pixels with a sentinel of 1.0, which against
    # fluxes of ~1e-21 is not an error bar but a flag.  All 115,940 such pixels
    # lie outside valid_high.  Left in, they drive the mean normalised flux
    # uncertainty to 3.2e18 instead of ~0.6.  Blank them to NaN so the
    # figure's nan-aware statistics skip them, which also restricts its MAE
    # to valid pixels -- the same masking the summary tables use.
    # flux_high stays unmasked: the figure uses it only to derive the
    # per-spectrum scale, and hi_std -- the scale x_high itself was built with
    # -- is exactly np.std over all pixels.  Masking it would put the residuals
    # and their error bars on scales differing by up to 10%.
    valid_h = np.asarray(E["valid_high"], dtype=bool)
    fe = np.where(valid_h, np.asarray(E["flux_high_err"], dtype=np.float64), np.nan)
    np.savez(OUT / "flux_high_err.npz",
             flux_high=np.asarray(E["flux_high"], dtype=np.float64),
             flux_high_err=fe, hi_std=HI_S, valid_high=valid_h)
    print("  wrote arrays, ml_inference_cache.npz, flux_high_err.npz")

    # ── 2. line fits for every method x line ─────────────────────────────────
    arrays = {
        "Cubic (LR)": X_LOW,
        "Wiener": np.load(OUT / "wiener_cache.npy").astype(np.float64),
        "Tikhonov": np.load(OUT / "tikhonov_cache.npy").astype(np.float64),
        "TV": np.load(OUT / "tv_cache.npy").astype(np.float64),
        "R-L": np.load(OUT / "rl_cache.npy").astype(np.float64),
        "Sparse": np.load(OUT / "sparse_cache.npy").astype(np.float64),
        "Wiener + MF": np.load(OUT / "mf_cache.npy").astype(np.float64),
        "ML (SR2)": (np.asarray(E["sr2"], dtype=np.float64) - HI_M) / HI_S,
        "HR target": X_HIGH,
    }
    jobs = [(lab, ln, l0, arrays[lab]) for lab, _ in LABELS for ln, l0 in LINES]
    print(f"\n  fitting {len(jobs)} method x line combinations "
          f"({len(jobs) * N:,} Gaussian fits)...")
    t0 = time.perf_counter()
    with Pool(NPROC) as p:
        results = p.map(_task, jobs)
    print(f"  done in {time.perf_counter() - t0:.0f}s\n")

    snr_label = dict(LABELS)
    fit_data, snr_data = {}, {}
    print(f"  {'method':13s} {'line':10s}  valid   median S/N   sigma at bound")
    for label, lname, amps, sigs, sns, clipped in results:
        fit_data[f"{label}_{lname}_amp"] = amps
        fit_data[f"{label}_{lname}_sigma"] = sigs
        fit_data[f"{label}_{lname}_sn"] = sns
        snr_data[f"{snr_label[label]}_{lname}"] = sns
        nv = int(np.isfinite(sns).sum())
        print(f"  {label:13s} {lname:10s} {nv:4d}/{N}   {np.nanmedian(sns):8.2f}   "
              f"{clipped:4d} ({100.0 * clipped / max(nv, 1):.1f}%)")

    np.savez(OUT / "fit_params_cache.npz", **fit_data)
    np.savez(OUT / "snr.npz", **snr_data)
    for m in ("Tikhonov", "TV", "Sparse"):
        np.savez(OUT / f"{m.lower()}_snr.npz",
                 **{k: v for k, v in snr_data.items() if k.startswith(m + "_")})
    print(f"\n  wrote fit_params_cache.npz ({len(fit_data)} arrays), "
          f"snr.npz ({len(snr_data)} arrays)")

    # ── 3. the summary the paper and tests are checked against ───────────────
    # Written here, at the end of the chain, so it cannot fall out of step with
    # the caches: a stale summary is exactly the drift tests/test_invariants.py
    # exists to catch.
    import csv
    sdT = float(np.nanstd(np.where(VALID, X_HIGH, np.nan)))
    with (OUT / "summary_final.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Method", "MAE", "MAE_scalefree", "std_ratio"]
                   + [f"SN_{ln}" for ln, _ in LINES])
        rows = []
        for label, snl in LABELS:
            a = np.asarray(arrays[label], dtype=np.float64)
            d = np.where(VALID, a - X_HIGH, np.nan)
            num = np.nansum(np.where(VALID, a * X_HIGH, np.nan), axis=1)
            den = np.nansum(np.where(VALID, a * a, np.nan), axis=1)
            k = np.where(den > 0, num / den, 1.0)[:, None]
            rows.append([
                label,
                float(np.nanmean(np.nanmean(np.abs(d), axis=1))),
                float(np.nanmean(np.nanmean(
                    np.abs(np.where(VALID, a * k - X_HIGH, np.nan)), axis=1))),
                float(np.nanstd(np.where(VALID, a, np.nan))) / sdT,
                *[float(np.nanmedian(snr_data[f"{snl}_{ln}"])) for ln, _ in LINES],
            ])
        for r in sorted(rows, key=lambda r: r[2]):
            w.writerow([r[0]] + [f"{v:.4f}" for v in r[1:]])
    print("  wrote summary_final.csv")
    print(f"\nDone. {OUT}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
