"""Stage 1 -- run the specsr chain over the held-out split, from Hub weights.

This is the ML arm of the benchmark, and the only stage that needs ``torch``,
a GPU (optional) and network access.  It loads SR1, the redshift head and SR2
from the Hugging Face Hub, runs them over the 572 held-out galaxies of the
group-wise 80/20 split, and writes their predictions in physical units.

Why the Hub and not a local checkpoint
--------------------------------------
The predictions the *published* numbers were computed from came from three
files inside a training-run directory on one workstation.  An earlier
generation of this paper's ML arm was lost exactly that way -- ``cache/`` in
this project is a committed 229 MB of arrays that cannot be regenerated because
the checkpoints behind them no longer exist anywhere.  The weights on the Hub
are the same three files, verified byte-identical to the run directory they
came from, and are addressable from any machine.

The one trap this stage cannot check for you
--------------------------------------------
``specsr`` has shipped more than one file called ``best_sr2.pth``, trained on
different wavelength grids.  A checkpoint from the wrong grid does not fail
loudly; it produces plausible spectra that are wrong.  The Hub revision is
pinned in ``specsr.checkpoints.DEFAULT_REVISION`` for that reason, and the
provenance of whatever was actually loaded is written into the output, so the
arrays can always be traced back to the weights that made them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import paths

#: The split both papers evaluate on.  Group-wise on the parent galaxy, so all
#: 21 augmented rows of a galaxy fall on the same side: a flat row-wise split
#: leaks ~16 near-duplicate siblings of each held-out galaxy into training and
#: inflates every held-out metric.
SPLIT = "val"


def _checkpoints_from_hub(repo_id: str | None, revision: str | None):
    """``(sr1, sr1_config, zhead, sr2)`` local paths, downloading as needed."""
    from specsr.checkpoints import get_checkpoint

    kw = {}
    if repo_id:
        kw["repo_id"] = repo_id
    if revision:
        kw["revision"] = revision
    return tuple(get_checkpoint(n, **kw)
                 for n in ("sr1", "sr1_config", "zhead", "sr2"))


def main(argv=None) -> int:
    """Run SR1 -> ZHead -> SR2 over the held-out split and cache the result."""
    ap = argparse.ArgumentParser(prog="specsrbench build predictions")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="paired JADES dataset written by `specsr build`; "
                         "defaults to specsr's own DEFAULT_DATASET")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--repo-id", default=None,
                    help="Hub model repo (default: specsr's own)")
    ap.add_argument("--revision", default=None, help="Hub revision to pin to")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="load from a local directory instead of the Hub")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--nproc", type=int, default=None,
                    help="unused; accepted so the CLI can pass it uniformly")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args([] if argv is None else argv)

    out = args.out or (paths.sets_dir() / "ml_predictions_val.npz")
    if args.dry_run:
        print(f"  would run the specsr chain on split={SPLIT!r}\n"
              f"  dataset  {args.dataset or 'specsr DEFAULT_DATASET'}\n"
              f"  weights  "
              f"{'local ' + str(args.checkpoint_dir) if args.checkpoint_dir else 'Hugging Face Hub'}\n"
              f"  writes   {out}")
        return 0

    try:
        from specsr.evaluation import DEFAULT_DATASET, load_pipeline, predict
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "this stage needs the specsr package and torch:\n"
            "    pip install 'specsrbench[ml]'\n"
            f"  ({exc})")

    dataset = Path(args.dataset) if args.dataset else Path(DEFAULT_DATASET)
    if not dataset.exists():
        raise SystemExit(
            f"paired dataset not found: {dataset}\n"
            "  it is built from the raw JADES DR4 tree by specsr, and at 3.5 GB\n"
            "  is not distributable here:\n"
            "      specsr build --jades-root <JADES DR4 tree> --out <dataset>.npz")

    if args.checkpoint_dir:
        d = Path(args.checkpoint_dir)
        sr1, sr1_cfg, zhead, sr2 = (d / "best_sr1.pth", d / "config_logR.yaml",
                                    d / "best_zhead.pth", d / "best_sr2.pth")
        source = f"local {d}"
    else:
        sr1, sr1_cfg, zhead, sr2 = _checkpoints_from_hub(args.repo_id, args.revision)
        from specsr.checkpoints import DEFAULT_REPO, DEFAULT_REVISION
        source = (f"hub {args.repo_id or DEFAULT_REPO}"
                  f"@{args.revision or DEFAULT_REVISION}")

    print(f"  checkpoints: {source}")
    for name, p in (("sr1", sr1), ("sr1_config", sr1_cfg),
                    ("zhead", zhead), ("sr2", sr2)):
        print(f"    {name:11s} {p}")
    print(f"  dataset:     {dataset}")

    pipeline = load_pipeline(sr2_ckpt=str(sr2), sr1_ckpt=str(sr1),
                             sr1_config=str(sr1_cfg), zhead_ckpt=str(zhead),
                             dataset=str(dataset))
    res = predict(pipeline, SPLIT, dataset=str(dataset), batch_size=args.batch_size)

    # Record what made these arrays, beside the arrays.  A cache whose inputs
    # are not written down next to it is how a green test suite once proved the
    # paper matched a cache that had been built from different inputs than the
    # ones recorded for it.
    res["provenance"] = json.dumps({
        "checkpoints": source,
        "sr1": str(sr1), "sr1_config": str(sr1_cfg),
        "zhead": str(zhead), "sr2": str(sr2),
        "dataset": str(dataset),
        "split": f"{SPLIT} of the group-wise 80/20 split, seed 42",
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)

    n = len(res["z_true"])
    print(f"\n  cached {n} spectra -> {out}")
    if "sr2" in res:
        d1 = float(np.mean((res["sr1"] - res["flux_high"]) ** 2))
        d2 = float(np.mean((res["sr2"] - res["flux_high"]) ** 2))
        print(f"  physical-space MSE vs HR: SR1 {d1:.4e}   SR2 {d2:.4e}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
