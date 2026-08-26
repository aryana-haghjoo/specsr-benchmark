"""Rebuild the classical caches on the log constant-R grid with retuned parameters.

Parameters come from :mod:`specsrbench.build.tune`, chosen on the 40-spectrum
``tune_set`` that shares no galaxy with the 572-spectrum evaluation set.
Writes to ``cache_logR_tuned/``; never touches ``cache/`` or ``cache_logR/``.

Three methods have their optimum at a degenerate boundary and are recorded both
ways, at the tuned setting and at the setting paper 2 currently publishes, so
the difference is visible rather than buried.
"""
from __future__ import annotations

import argparse
import json
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .. import classical as C
from .. import paths
from . import require_npz

SRC = paths.sets_dir()
OUT = paths.cache_dir()
NPROC = 24        # this box is shared; leave headroom

# ── parameters ───────────────────────────────────────────────
#
# Read from the file `specsrbench build tune` writes, never restated here.
# They used to be a literal dict in this file, kept in step with the tuner by
# hand, and on 2026-08-24 they fell out of step: the tuner retuned every method
# against the derived kernel and this file went on building the caches with the
# old values and the shipped one.  Every classical number in the paper was
# wrong for a day because two files disagreed about six numbers.
#
# The tuner selects each setting on the 40-spectrum tune_set (no galaxy shared
# with the evaluation set) by minimising MAE subject to four guards, each
# catching something none of the others can:
#
#   (1) smoothing  -- median line S/N >= 0.9x the no-deconvolution baseline.
#       A filter that erases every line incurs no line-shaped residual, so MAE
#       rewards it.
#   (2) shrinkage  -- output std within [0.90, 1.15] of the target's.  MAE also
#       rewards scaling toward zero, and guard (1) is blind to it: line S/N is
#       amplitude over sideband noise, so a global rescale leaves it unchanged.
#   (3) blurring   -- median line FWHM bias <= the baseline's.  A unit-gain
#       Wiener filter with snr <= 1 peaks at zero frequency, so it cannot
#       amplify anything and can only broaden, while scoring well on MAE.
#       Guard (2) is width-invariant and cannot see it.
#   (4) merging    -- the fraction of resolvable [OIII] pairs kept resolved must
#       be at least the no-deconvolution baseline's.  Guards (1)-(3) are all
#       single-line Gaussian-fit statistics, and a Gaussian fitted to a blended
#       doublet has much the same amplitude, S/N and width as one fitted to a
#       separated pair, so all three pass a filter that merges the doublet.
#
# Guard (2) is why the Wiener filter is normalised to unit DC gain and why the
# matched filter's template width is a fraction of the LSF rather than 1.0: at
# width_scale=1 it injects flux over a region ~14x broader than a real line,
# reaching std=2.16.  Guard (4) is why Tikhonov is retained rather than tuned --
# no setting passes it and the amplitude guard together, which the tuner records
# in ``failed_guards``.
FALLBACK_TUNED = {
    "wiener":   dict(snr=8.0, segment_len=128, overlap=32),
    "tikhonov": dict(lam=10.0, segment_len=512, overlap=128),
    "rl":       dict(n_iter=1),
    "sparse":   dict(lam=0.05, n_iter=1),
    "tv":       dict(lam=0.1, n_iter=30),
    "mf":       dict(window_nsigma=4.0, detect_snr=5.0, core_nsigma=3.0,
                     sideband_nsigma=2.0, width_scale=0.2),
}

TUNER_RECORD: dict = {}
TUNED: dict = {}
PARAMS_SRC = ""


def _load_params():
    """The parameters the caches are built with, and where they came from.

    Resolved when the stage runs rather than when the module is imported, so
    that pointing SPECSRBENCH_CACHE somewhere else actually changes which
    parameters are read instead of silently reusing whichever directory
    happened to be current at import.
    """
    global TUNER_RECORD, TUNED, PARAMS_SRC
    params_path = OUT / "classical_params.json"
    if params_path.exists():
        TUNER_RECORD = json.loads(params_path.read_text())
        TUNED = {k: dict(v) for k, v in TUNER_RECORD["tuned"].items()}
        PARAMS_SRC = str(params_path)
    else:                                                # pragma: no cover
        TUNER_RECORD = {}
        TUNED = {k: dict(v) for k, v in FALLBACK_TUNED.items()}
        PARAMS_SRC = "FALLBACK -- run `specsrbench build tune` first"

# what paper 2 currently publishes, carried over from the linear grid
AS_PUBLISHED = {
    "wiener":   dict(snr=10.0, segment_len=128, overlap=32),
    "tikhonov": dict(lam=0.1, segment_len=128, overlap=32),
    "rl":       dict(n_iter=30),
    "sparse":   dict(lam=0.05, n_iter=150),
    "tv":       dict(lam=0.02, n_iter=30),
}

# ── inputs ────────────────────────────────────────────────────────────────────
# Loaded by `_load()` at the top of `main()`, not at import.  Importing a module
# must not require a cache to exist: it made three modules un-importable on any
# machine without the data, which broke the API documentation build and would
# have broken `from specsrbench.build import ...` for everyone else.
#
# The names stay module-level globals rather than becoming a context object
# because the worker functions below close over them and are dispatched through
# `multiprocessing.Pool`.  On fork, a worker inherits whatever the parent had
# set by the time the pool was created, which is after `_load()` has run.
E = WAVE = SIGMA_PIX = KERNEL_SRC = None
X_LOW = X_HIGH = VALID = Z = HI_M = HI_S = MF_LINES = None


def is_emission_template(name):
    """Line-list entries the matched filter writes back: emission only.

    Breaks and absorption features are excluded -- a matched filter that places
    a positive template at a Balmer break is fitting a step with a Gaussian.
    """
    excluded = ("Lyman_limit", "Balmer_break", "D4000_break", "CaK", "CaH",
                "Gband", "Mg_b", "NaD", "DIB", "TiO", "FeII_UV",
                "FeII_opt_blend", "MnII", "MgI", "CaII_triplet")
    return not name.startswith(excluded)


def _load():
    """Read this stage's inputs into the module globals."""
    global E, WAVE, SIGMA_PIX, KERNEL_SRC, X_LOW, X_HIGH, VALID, Z
    global HI_M, HI_S, MF_LINES
    # Imported here rather than at module scope: specsr is an optional extra,
    # and only this stage needs its line list.
    from specsr.models.lines import LINE_LIST_REST_AA

    E = require_npz(SRC / "eval_set.npz", "specsrbench build sets")
    WAVE = np.asarray(E["wave"], dtype=np.float64)
    SIGMA_PIX, KERNEL_SRC = C.load_sigma_pix(OUT, E)
    X_LOW = np.asarray(E["x_low"], dtype=np.float64)
    X_HIGH = np.asarray(E["x_high"], dtype=np.float64)
    VALID = np.asarray(E["valid_high"], dtype=bool)
    Z = np.asarray(E["z_true"], dtype=np.float64)
    HI_M = np.asarray(E["hi_mean"], dtype=np.float64)
    HI_S = np.asarray(E["hi_std"], dtype=np.float64)
    MF_LINES = np.asarray(
        [w for n, w in LINE_LIST_REST_AA if is_emission_template(n)],
        dtype=np.float32) * 1e-4


def _w(s, **k):
    return C.wiener_deconv(s, SIGMA_PIX, **k)


def _t(s, **k):
    return C.tikhonov_deconv(s, SIGMA_PIX, **k)


def _r(s, **k):
    return C.rl_deconv(s, SIGMA_PIX, **k)


def _sp(s, **k):
    return C.sparse_wavelet_deconv(s, SIGMA_PIX, **k)


def _tv(s, **k):
    return C.tv_denoise_1d(s, **k)


def _mf(a, **k):
    s, z = a
    return C.matched_filter(s, WAVE, z, MF_LINES, SIGMA_PIX, **k)


def run(label, fn, arrs, **kw):
    t0 = time.perf_counter()
    with Pool(NPROC) as p:
        out = np.array(p.map(partial(fn, **kw), list(arrs)))
    print(f"  {label:26s} {time.perf_counter() - t0:6.1f}s", flush=True)
    return out


def stats(name, arr):
    a = np.asarray(arr, dtype=np.float64)
    v = VALID & np.isfinite(a) & np.isfinite(X_HIGH)
    d = np.where(v, a - X_HIGH, np.nan)
    mae_by = np.nanmean(np.abs(d), axis=1)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(mae_by), size=(1000, len(mae_by)), replace=True)

    # Scale-free diagnostics.  Raw MAE can be lowered by shrinking the estimate
    # toward zero, so report alongside it (a) the amplitude actually retained
    # and (b) MAE after each spectrum is rescaled by its own least-squares
    # optimal gain, which no amount of shrinkage can improve.
    std_ratio = (float(np.nanstd(np.where(v, a, np.nan)))
                 / float(np.nanstd(np.where(v, X_HIGH, np.nan))))
    num = np.nansum(np.where(v, a * X_HIGH, np.nan), axis=1)
    den = np.nansum(np.where(v, a * a, np.nan), axis=1)
    k = np.where(den > 0, num / den, 1.0)[:, None]
    mae_sf = float(np.nanmean(np.nanmean(np.abs(np.where(v, a * k - X_HIGH, np.nan)), axis=1)))
    return {
        "Method": name,
        "MAE": float(np.nanmean(mae_by)),
        "MAE_err": float(np.nanstd(np.nanmean(mae_by[idx], axis=1))),
        "MAE_scalefree": mae_sf,
        "std_ratio": std_ratio,
        "RMSE": float(np.sqrt(np.nanmean(d[np.isfinite(d)] ** 2))),
        "Bias": float(np.nanmean(d[np.isfinite(d)])),
    }


def main(argv=None) -> int:
    """Rebuild the six classical caches at the tuned parameters."""
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

    _load_params()
    _load()

    OUT.mkdir(exist_ok=True)
    print(f"Rebuilding classical caches on {X_LOW.shape[0]} held-out spectra "
          f"x {X_LOW.shape[1]} pixels (log R=4000)")
    print(f"  kernel:     {KERNEL_SRC}")
    print(f"  parameters: {PARAMS_SRC}")
    if KERNEL_SRC.startswith("SHIPPED"):
        raise SystemExit(
            "refusing to build with the shipped kernel: it does not describe "
            "the data (see specsrbench.build.lsf).  Run\n"
            "    specsrbench build lsf --jades-root <JADES DR4 tree>\n"
            "to write sigma_pix_measured.npy first.")
    for name, kw in TUNED.items():
        print(f"    {name:9s} {kw}")
    print()

    # The cache records the kernel it was built with, so nothing downstream has
    # to infer it.  This file used to be written by the line-fit stage from the
    # shipped array, which is how the line fits and the guards came to be
    # evaluated against a kernel the caches had not been built with.
    np.save(OUT / "sigma_pix.npy", SIGMA_PIX)

    print("tuned parameters:")
    wiener = run("Wiener", _w, X_LOW, **TUNED["wiener"])
    tikh = run("Tikhonov", _t, X_LOW, **TUNED["tikhonov"])
    rl = run("R-L", _r, X_LOW, **TUNED["rl"])
    sparse = run("Sparse", _sp, X_LOW, **TUNED["sparse"])
    tv = run("Wiener + TV", _tv, wiener, **TUNED["tv"])
    mf = run("Wiener + MF", _mf, list(zip(wiener, Z)), **TUNED["mf"])

    for n, a in [("wiener", wiener), ("tikhonov", tikh), ("rl", rl),
                 ("sparse", sparse), ("tv", tv), ("mf", mf)]:
        np.save(OUT / f"{n}_cache.npy", a.astype(np.float32))

    print("\nas-published parameters (for comparison):")
    ap_w = run("Wiener", _w, X_LOW, **AS_PUBLISHED["wiener"])
    ap = {
        "Wiener": ap_w,
        "Tikhonov": run("Tikhonov", _t, X_LOW, **AS_PUBLISHED["tikhonov"]),
        "R-L": run("R-L", _r, X_LOW, **AS_PUBLISHED["rl"]),
        "Sparse": run("Sparse", _sp, X_LOW, **AS_PUBLISHED["sparse"]),
        "Wiener + TV": run("Wiener + TV", _tv, ap_w, **AS_PUBLISHED["tv"]),
    }

    nrm = lambda a: (np.asarray(a, dtype=np.float64) - HI_M) / HI_S  # noqa: E731
    rows = [stats(n, a) for n, a in [
        ("ML (SR2)", nrm(E["sr2"])),
        ("ML (SR1)", nrm(E["sr1"])),
        ("Wiener", wiener),
        ("Wiener + TV", tv),
        ("Wiener + MF", mf),
        ("Tikhonov", tikh),
        ("Cubic (LR)", X_LOW),
        ("R-L", rl),
        ("Sparse", sparse),
    ]]
    df = pd.DataFrame(rows).sort_values("MAE_scalefree").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("RETUNED — 572 held-out originals, log R=4000, 6,671 px")
    print("sorted by MAE_scalefree; std_ratio ~1 means the amplitude is preserved")
    print("=" * 78)
    print(df.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("as-published parameters on the same data, for comparison")
    print("=" * 78)
    print(pd.DataFrame([stats(n, a) for n, a in ap.items()])
          .sort_values("MAE").round(4).to_string(index=False))

    df.to_csv(OUT / "summary_retuned.csv", index=False)
    # Carry the tuner's own record through unchanged -- failed_guards, the
    # guard thresholds and the baselines are its findings, not this script's --
    # and add only what the build itself knows.
    record = dict(TUNER_RECORD)
    record.update(
        {"tuned": TUNED, "as_published": AS_PUBLISHED,
         "kernel": KERNEL_SRC,
         "params_source": PARAMS_SRC,
         "tuned_on": "cache_logR/tune_set.npz (40 spectra, disjoint from eval)",
         "objective": "MAE vs x_high masked by valid_high",
         "eval_set": "cache_logR/eval_set.npz (572 held-out originals)",
         "grid": "specsr DEFAULT_GRID, log R=4000, 1.0-5.3 um, 6671 points"})
    (OUT / "classical_params.json").write_text(json.dumps(record, indent=2))
    print(f"\nWrote {OUT}/  (caches, summary_retuned.csv, classical_params.json)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
