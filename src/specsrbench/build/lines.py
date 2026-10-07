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
from scipy.optimize import curve_fit, least_squares

from .. import classical as C
from .. import paths
from . import require_npz

SRC = paths.sets_dir()
OUT = paths.cache_dir()
NPROC = 24        # this box is shared; leave headroom



# ── inputs ────────────────────────────────────────────────────────────────────
# Loaded by `_load()` at the top of `main()`, not at import.  Importing a module
# must not require a cache to exist: it made three modules un-importable on any
# machine without the data, which broke the API documentation build and would
# have broken `from specsrbench.build import tune` for everyone else.
#
# The names stay module-level globals rather than becoming a context object
# because the worker functions below close over them and are dispatched through
# `multiprocessing.Pool`.  On fork, a worker inherits whatever the parent had
# set by the time the pool was created, which is after `_load()` has run.
E = None
WAVE = None
DPIX = None
X_LOW = None
X_HIGH = None
VALID = None
Z = None
HI_M = None
HI_S = None
N = None


def _load():
    """Read this stage's inputs into the module globals."""
    global E, WAVE, DPIX, X_LOW, X_HIGH, VALID, Z, HI_M, HI_S, N
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
              core_halfwin=0.05, sigma_hi=0.12, mu_bounds_half=0.01,
              return_mu=False):
    """(amp, sigma, S/N) or (nan, nan, nan).  Lower sigma bound tracks the grid.

    With ``return_mu`` the fitted centre is appended.  It is bounded to
    ``mu0 +- mu_bounds_half`` (10 nm), so a centre offset between two fits can
    never exceed 20 nm, and one at the bound means the fit found no line there.
    """
    nan = (np.nan,) * (4 if return_mu else 3)
    fit_m = (x >= mu0 - fit_halfwin) & (x <= mu0 + fit_halfwin)
    core_m = np.abs(x - mu0) <= core_halfwin
    sb_m = ((np.abs(x - mu0) >= sb_gap) & (np.abs(x - mu0) <= sb_gap + sb_width)
            & (~core_m) & fit_m)
    xx, yy = x[fit_m], y[fit_m]
    y_sb = y[sb_m & np.isfinite(y)]
    if xx.size < 15 or y_sb.size < 30:
        return nan
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
        out = (float(popt[0]), float(popt[2]), abs(float(popt[0])) / sc)
        return out + (float(popt[1]),) if return_mu else out
    except Exception:
        return nan


C_KMS = 299792.458
# H-beta, [O III] 4959 and [O III] 5007: three lines within 9000 km/s that the
# prism merges into one feature.
HBETA_OIII_REST_UM = (0.486133, 0.495891, 0.500684)
#: The lines whose flux comes from the joint fit, and their column in it.
JOINT_LINES = {"Hbeta": 0, "OIII5007": 2}


def fit_hbeta_oiii(wavelength, flux, z, *, window_kms: float = 20000.0,
                   sigma_lo_kms: float = 40.0, sigma_hi_kms: float = 3000.0):
    """Joint fit of H-beta and the [O III] doublet: three Gaussians, one width.

    The model ``specsr.linefit.fit_hbeta_oiii`` uses for paper 1's line-flux
    figure, ported so the flux of a line means the same thing in both papers.
    A single Gaussian cannot measure the *flux* of these lines at prism
    resolution: centred on H-beta it widens until it holds the [O III] lines as
    well, and centred on 5007 it absorbs 4959.  Fitting the three together at
    fixed centres with a shared velocity width assigns the blended flux to the
    right line.  Amplitudes are bounded non-negative.

    Returns ``(amps, sigmas_um)``, each of length three in the order H-beta,
    4959, 5007, or ``None`` when the lines fall off the grid, the window holds
    a non-finite pixel, or the fit fails.
    """
    centres = np.asarray(HBETA_OIII_REST_UM) * (1.0 + float(z))
    lam0 = 0.5 * (centres[1] + centres[2])
    v_all = (wavelength - lam0) / lam0 * C_KMS
    m = np.abs(v_all) <= window_kms
    if centres[0] < wavelength[0] or centres[2] > wavelength[-1] or m.sum() < 30:
        return None
    v, y = v_all[m], flux[m]
    if not np.isfinite(y).all():
        return None
    vc = (centres - lam0) / lam0 * C_KMS

    def model(p):
        g = p[0] + p[1] * v
        for amp, v0 in zip(p[3:], vc, strict=True):
            g = g + amp * np.exp(-0.5 * ((v - v0) / p[2]) ** 2)
        return g

    med = float(np.median(y))
    amp0 = max(float(np.max(y) - med), 1e-6)
    p0 = [med, 0.0, 300.0, amp0 * 0.3, amp0 / 3.33, amp0]
    lo = [-np.inf, -np.inf, sigma_lo_kms, 0.0, 0.0, 0.0]
    hi = [np.inf, np.inf, sigma_hi_kms, np.inf, np.inf, np.inf]
    r = least_squares(lambda p: model(p) - y, p0, bounds=(lo, hi),
                      max_nfev=4000, method="trf")
    if not r.success:
        return None
    # An amplitude pinned at its lower bound comes back as a denormal-sized
    # positive number, not as zero; it is a non-detection and is returned as one.
    amps = np.array([a if a > 1e-4 * amp0 else 0.0 for a in r.x[3:]])
    return amps, float(r.x[2]) / C_KMS * centres


def joint_fits(wave, arr, z):
    """``(amps, sigmas_um)``, each ``(n, 3)``, from :func:`fit_hbeta_oiii`."""
    amps = np.full((len(z), 3), np.nan)
    sigs = np.full((len(z), 3), np.nan)
    for i in range(len(z)):
        r = fit_hbeta_oiii(wave, arr[i], z[i])
        if r is not None:
            amps[i], sigs[i] = r
    return amps, sigs


def line_fluxes(label, fit_data, joint):
    """Integrated flux of each diagnostic line, ``sqrt(2 pi) * amp * sigma``.

    H-alpha and [O II] take the single-Gaussian fit already in ``fit_data``;
    H-beta and [O III] 5007 take the joint fit.  ``_flux_amp`` is the amplitude
    of whichever fit supplied the flux, which is what a detection cut on the
    flux measurement has to be applied to.
    """
    amps, sigs = joint
    out = {}
    for lname, _ in LINES:
        if lname in JOINT_LINES:
            a, sg = amps[:, JOINT_LINES[lname]], sigs[:, JOINT_LINES[lname]]
        else:
            a, sg = fit_data[f"{label}_{lname}_amp"], fit_data[f"{label}_{lname}_sigma"]
        out[f"{label}_{lname}_flux"] = np.sqrt(2.0 * np.pi) * a * sg
        out[f"{label}_{lname}_flux_amp"] = np.asarray(a, dtype=np.float64).copy()
    return out


def _joint_task(job):
    label, arr = job
    return label, joint_fits(WAVE, arr, Z)


def method_arrays():
    """Every reconstruction, and the reference, keyed by its fit-cache label."""
    return {
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


def add_line_fluxes(fit_data, arrays):
    """Add the ``_flux`` / ``_flux_amp`` arrays for every method to ``fit_data``."""
    with Pool(NPROC) as p:
        joint = dict(p.map(_joint_task, [(lab, arrays[lab]) for lab, _ in LABELS]))
    for lab, _ in LABELS:
        fit_data.update(line_fluxes(lab, fit_data, joint[lab]))
    return fit_data


def _task(job):
    label, lname, lam0, arr = job
    amps = np.full(N, np.nan)
    sigs = np.full(N, np.nan)
    sns = np.full(N, np.nan)
    mus = np.full(N, np.nan)
    clipped = 0
    for i in range(N):
        mu0 = lam0 * (1.0 + Z[i])
        amps[i], sigs[i], sns[i], mus[i] = fit_gauss(WAVE, arr[i], mu0, return_mu=True)
        if np.isfinite(sigs[i]):
            j = int(np.argmin(np.abs(WAVE - mu0)))
            fit_m = (WAVE >= mu0 - 0.25) & (WAVE <= mu0 + 0.25)
            dx = float(np.median(np.diff(WAVE[fit_m]))) if fit_m.sum() > 1 else DPIX[j]
            if sigs[i] <= 0.5 * dx * 1.01 or sigs[i] >= 0.12 * 0.99:
                clipped += 1
    return label, lname, amps, sigs, sns, mus, clipped


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
    global SRC, OUT
    SRC, OUT = paths.sets_dir(), paths.cache_dir()
    if args.dry_run:
        print(f"  reads  {SRC}\n  writes {OUT}\n  nproc  {NPROC}")
        return 0

    _load()

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
    arrays = method_arrays()
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
    for label, lname, amps, sigs, sns, mus, clipped in results:
        fit_data[f"{label}_{lname}_amp"] = amps
        fit_data[f"{label}_{lname}_sigma"] = sigs
        fit_data[f"{label}_{lname}_mu"] = mus
        fit_data[f"{label}_{lname}_sn"] = sns
        snr_data[f"{snr_label[label]}_{lname}"] = sns
        nv = int(np.isfinite(sns).sum())
        print(f"  {label:13s} {lname:10s} {nv:4d}/{N}   {np.nanmedian(sns):8.2f}   "
              f"{clipped:4d} ({100.0 * clipped / max(nv, 1):.1f}%)")

    # Integrated fluxes, with H-beta and [O III] from a joint fit.  The flux
    # ratios of section 4.5 are built from these, not from the amplitudes.
    add_line_fluxes(fit_data, arrays)
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
