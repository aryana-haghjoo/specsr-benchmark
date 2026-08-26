"""Stage 2 -- cut the predictions into the three galaxy-disjoint sets.

Everything downstream reads one of these three, and which one is not a detail:

``eval_set.npz``   572 held-out originals.  Every published number.
``tune_set.npz``    40 galaxies from the *training* side.  The classical
                    parameters are chosen on this and nowhere else.
``calib_set.npz``  400 galaxies, also from the training side.  Guard 4 --
                    whether a resolvable [O III] pair survives a filter -- is
                    measured on these.

The two smaller sets exist because tuning a method on the set you then report
it on is the same error, one level up, that the group-wise split fixes at the
galaxy level.  All three draw from originals only: the paired dataset carries
21 augmented rows per galaxy, and a set built from augmented rows would measure
how well a method deconvolves a copy of a spectrum it has already seen.

Guard 4 is measured on ``calib_set`` rather than on a synthetic pair on
purpose.  A synthetic doublet was tried and discarded: it is far easier to hold
apart than a real one, and it passed settings that merge in real data.

Nothing here is random in the sense of being irreproducible -- the draws are
seeded and the seeds are part of the definition of the sets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import paths
from . import require_npz

#: Sizes and seeds of the two training-side draws.  These are *the* definition
#: of those sets: changing either changes which galaxies the classical methods
#: were tuned and guarded on, and therefore every classical number downstream.
TUNE_N, TUNE_SEED = 40, 0
CALIB_N, CALIB_SEED = 400, 1


def zscore(a):
    """Per-spectrum z-score, plus the moments needed to undo it."""
    a = np.asarray(a, dtype=np.float64)
    mu = np.nanmean(a, axis=1, keepdims=True)
    sd = np.nanstd(a, axis=1, keepdims=True)
    return (a - mu) / np.where(sd < 1e-25, 1e-25, sd), mu, sd


def training_originals(dataset: Path) -> np.ndarray:
    """Row indices of the un-augmented galaxies on the training side."""
    from specsr.data.splits import get_or_make_split_3way
    from specsr.evaluation import ALLOW_EMPTY_TEST, SEED, TRAIN_FRAC, VAL_FRAC

    train_idx, _val, _test, _path = get_or_make_split_3way(
        str(dataset), TRAIN_FRAC, VAL_FRAC, SEED, allow_empty_test=ALLOW_EMPTY_TEST)
    with np.load(str(dataset), allow_pickle=True) as d:
        is_original = np.asarray(d["is_original"], dtype=bool)
    return np.sort(np.asarray(train_idx)[is_original[np.asarray(train_idx)]])


def _subset(dataset: Path, rows: np.ndarray, keys) -> dict:
    with np.load(str(dataset), allow_pickle=True) as d:
        return {k: np.asarray(d[k])[rows] for k in keys if k in d}


def build_eval_set(predictions, out: Path) -> dict:
    """``eval_set.npz`` -- the predictions plus the normalised spectra.

    ``sigma_pix`` is deliberately **not** written.  The array of that name in
    the historical ``eval_set.npz`` was not produced by anything in either
    repository and does not describe the data -- roughly constant in nanometres
    where a spectrograph's LSF is fixed in detector pixels, and up to 2.3x too
    broad at 5 um.  Every classical method used it as its kernel, and with it
    Wiener, Tikhonov and TV merged line pairs that plain interpolation still
    resolves.  Leaving it out means the kernel can only come from
    ``specsrbench build lsf``, which measures it.
    """
    P = predictions
    x_high, hi_mean, hi_std = zscore(P["flux_high"])
    x_low, lo_mean, lo_std = zscore(P["flux_low"])
    flux_low = np.asarray(P["flux_low"], dtype=np.float64)

    data = {
        "wave": np.asarray(P["wave"], dtype=np.float64),
        "flux_low": flux_low,
        "flux_high": np.asarray(P["flux_high"], dtype=np.float64),
        "flux_high_err": np.asarray(P["flux_high_err"], dtype=np.float64),
        "valid_high": np.asarray(P["valid_high"], dtype=bool),
        # A low-resolution pixel is real if it is finite and not exactly zero;
        # the resampling writes hard zeros outside the prism's coverage.
        "valid_low": np.isfinite(flux_low) & (flux_low != 0),
        "sr1": np.asarray(P["sr1"], dtype=np.float64),
        "sr2": np.asarray(P["sr2"], dtype=np.float64),
        "z_true": np.asarray(P["z_true"], dtype=np.float64),
        "z_pred": np.asarray(P["z_pred"], dtype=np.float64),
        "x_low": x_low.astype(np.float32),
        "x_high": x_high.astype(np.float32),
        "lo_mean": lo_mean, "lo_std": lo_std,
        "hi_mean": hi_mean, "hi_std": hi_std,
        "parent_id": np.asarray(P["parent_id"]),
        "row_index": np.asarray(P["row_index"]),
    }
    return data


def main(argv=None) -> int:
    """Cut eval / tune / calib sets from the predictions and the paired dataset."""
    ap = argparse.ArgumentParser(prog="specsrbench build sets")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="paired JADES dataset (needed for the tune/calib draws)")
    ap.add_argument("--predictions", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--nproc", type=int, default=None,
                    help="unused; accepted so the CLI can pass it uniformly")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args([] if argv is None else argv)

    outdir = args.outdir or paths.sets_dir()
    pred_path = args.predictions or (outdir / "ml_predictions_val.npz")
    if args.dry_run:
        print(f"  reads  {pred_path}\n"
              f"         {args.dataset or 'specsr DEFAULT_DATASET'}\n"
              f"  writes {outdir}/eval_set.npz, tune_set.npz, calib_set.npz")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    P = require_npz(pred_path, "specsrbench build predictions")

    # ── the held-out set ──────────────────────────────────────────────────────
    data = build_eval_set(P, outdir / "eval_set.npz")
    prov = {
        "eval_set": f"{len(data['z_true'])} held-out originals, "
                    "group-wise 80/20 split",
        "grid": "specsr DEFAULT_GRID, log, R=4000, 1.0-5.3 um, 6671 points",
        "ml_source": str(pred_path.name),
        "kernel": "not included: derive it with `specsrbench build lsf`",
        "tune_draw": f"rng({TUNE_SEED}).choice(train originals, {TUNE_N})",
        "calib_draw": f"rng({CALIB_SEED}).choice(train originals, {CALIB_N})",
    }
    if "provenance" in P.files:
        prov["ml_provenance"] = str(P["provenance"])
    np.savez_compressed(outdir / "eval_set.npz", provenance=json.dumps(prov), **data)
    print(f"  eval_set.npz   {len(data['z_true']):4d} held-out originals")

    # ── the two training-side sets ────────────────────────────────────────────
    if args.dataset is None:
        try:
            from specsr.evaluation import DEFAULT_DATASET
            args.dataset = Path(DEFAULT_DATASET)
        except ImportError:
            print("  tune/calib sets skipped: specsr not importable and no --dataset")
            return 0
    if not Path(args.dataset).exists():
        print(f"  tune/calib sets skipped: {args.dataset} not present")
        return 0

    pool = training_originals(Path(args.dataset))
    print(f"  training originals available: {len(pool)}")

    tune_rows = np.sort(np.random.default_rng(TUNE_SEED).choice(
        pool, size=TUNE_N, replace=False))
    calib_rows = np.sort(np.random.default_rng(CALIB_SEED).choice(
        pool, size=CALIB_N, replace=False))

    held_out = set(np.asarray(P["row_index"]).tolist())
    for name, rows in (("tune", tune_rows), ("calib", calib_rows)):
        overlap = held_out & set(rows.tolist())
        if overlap:  # pragma: no cover - the split makes this impossible
            raise SystemExit(f"{name}_set overlaps the held-out set on "
                             f"{len(overlap)} rows; refusing to write it")

    tune = _subset(Path(args.dataset), tune_rows,
                   ["flux_low", "flux_high", "flux_high_err",
                    "valid_low", "valid_high", "z", "parent_id"])
    tune["row_index"] = tune_rows
    np.savez_compressed(outdir / "tune_set.npz", **tune)
    print(f"  tune_set.npz   {len(tune_rows):4d} galaxies  "
          f"(seed {TUNE_SEED}, disjoint from eval)")

    calib = _subset(Path(args.dataset), calib_rows,
                    ["flux_low", "flux_high", "flux_low_err", "flux_high_err",
                     "valid_low", "valid_high", "z"])
    calib["row_index"] = calib_rows
    np.savez_compressed(outdir / "calib_set.npz", **calib)
    print(f"  calib_set.npz  {len(calib_rows):4d} galaxies  "
          f"(seed {CALIB_SEED}, disjoint from eval)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
