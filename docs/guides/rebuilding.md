# Rebuilding the cache

No large files are distributed with this package. Everything the figures read
is derived data, rebuilt from the raw [JADES](https://jades-survey.github.io/)
DR4 products and the model weights published on the
[Hugging Face Hub](https://huggingface.co/aryana-haghjoo/specsr).

```bash
specsrbench build all
```

Six stages, each consuming what the one before it wrote:

| stage | reads | writes | needs |
|---|---|---|---|
| `predictions` | paired dataset, Hub weights | `ml_predictions_val.npz` | `specsr`, torch, network |
| `sets` | predictions, paired dataset | `eval_set` / `tune_set` / `calib_set` | the paired dataset |
| `lsf` | raw JADES `x1d`, line fits | `sigma_pix_measured.npy` | raw JADES, astropy |
| `tune` | tune set, calib set, kernel | `classical_params.json` | — |
| `classical` | eval set, kernel, parameters | the six classical caches | — |
| `lines` | every reconstruction | fits, S/N, `summary_final.csv` | — |

Any stage takes `--dry-run`, which prints what it would read and write and does
nothing.

## The paired dataset

Stages 1 and 2 need the paired prism/grating dataset, which `specsr` builds
from the JADES tree. At 3.5 GB it is not distributable here:

```bash
specsr build --jades-root <JADES DR4 tree> --out paired_DR4_logR.npz
specsrbench build predictions --dataset paired_DR4_logR.npz
```

## The ordering trap

`lsf` reads the line fits that `lines` writes, so the very first build of a
fresh tree runs `lines` once against whatever kernel is available, then `lsf`,
then `tune` → `classical` → `lines` again on the measured one.

`classical` **refuses** to write a cache built with a kernel it has not
measured. That refusal is what stops the first pass being mistaken for a
finished one.

## Match checkpoints by provenance, never by filename

```{danger}
More than one file called `best_sr2.pth` exists, trained on different
wavelength grids. Loading one from the wrong grid does not fail — it produces
plausible spectra that are wrong.
```

`specsrbench build predictions` pins the Hub revision and writes what it
actually loaded into the output's `provenance` field. Read that, not the
filename.

## The three sets are galaxy-disjoint

`eval_set` (572 held-out originals) carries every published number. The
classical parameters are chosen on `tune_set` (40 galaxies) and the
pair-survival guard is measured on `calib_set` (400), both drawn from the
*training* side of a group-wise split.

Tuning a method on the set you then report it on is the same error, one level
up, that the group-wise split fixes at the galaxy level. All three draw from
un-augmented originals only: the paired dataset carries 21 augmented rows per
galaxy, and a set built from those would measure how well a method deconvolves
a copy of a spectrum it has already seen.

The draws are seeded, and the seeds are part of the definition of the sets.

## Why the kernel is measured and not read

Evaluation products in this project's history carried a `sigma_pix` array,
widely used as the deconvolution kernel, that **does not describe the data**:
it is roughly constant in nanometres, where a spectrograph's line-spread
function is fixed in detector *pixels*. Measured against the paired spectra it
is up to 2.3× too broad at 5 µm. With it, Wiener, Tikhonov and TV merged line
pairs that their own input still resolves.

`specsrbench build lsf` measures the real one from the instrument: the prism
dispersion from raw `x1d` products, the effective width from
σ_eff² = σ_LR² − σ_HR² on fits to the four diagnostic lines, and the ratio of
the two. It comes out constant to a few per cent across a factor of three in
wavelength — the check that it is an instrumental LSF and not a curve fitted to
noise.

Sets built by `specsrbench build sets` carry **no** `sigma_pix` at all, so a
fresh build cannot recreate the trap.
