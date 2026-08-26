# specsrbench

[![tests](https://github.com/aryana-haghjoo/specsr-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/aryana-haghjoo/specsr-benchmark/actions/workflows/tests.yml)
[![docs](https://github.com/aryana-haghjoo/specsr-benchmark/actions/workflows/docs.yml/badge.svg)](https://aryana-haghjoo.github.io/specsr-benchmark/)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/aryana-haghjoo/specsr-benchmark)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/1346826576.svg)](https://doi.org/10.5281/zenodo.22104971)

**Benchmarking deep learning against classical deconvolution for galaxy
spectral super-resolution.**

Seven classical deconvolution methods — cubic interpolation, Wiener, Tikhonov,
Wiener + total variation, Richardson–Lucy, wavelet-sparse FISTA, and a
redshift-informed matched filter — scored against the SR2 deep-learning
pipeline of [Haghjoo et al. 2026](https://arxiv.org/abs/2603.18357) on
JWST/NIRSpec prism spectra from JADES.

The deep-learning side is a separate package,
[`specsr`](https://github.com/aryana-haghjoo/specsr), whose weights live
[on the Hub](https://huggingface.co/aryana-haghjoo/specsr). This package holds
the classical baselines, the metrics, the tuning, and the figures.

## Install

Not on PyPI yet. Install the wheel from the
[latest release](https://github.com/aryana-haghjoo/specsr-benchmark/releases/latest):

```bash
pip install https://github.com/aryana-haghjoo/specsr-benchmark/releases/download/v0.1.1/specsrbench-0.1.1-py3-none-any.whl
```

or from source:

```bash
git clone https://github.com/aryana-haghjoo/specsr-benchmark
cd specsr-benchmark
pip install -e .            # the baselines and the figures
pip install -e '.[all]'     # + the ML arm and the LSF derivation
```

`[ml]` pulls in torch and `specsr` (needed by `build predictions` and by
figure 1's toy CNN); `[lsf]` pulls in astropy (needed only to re-derive the
instrument LSF). Neither is required to draw the other five figures.

---

## What it is for

The short version of the result: **on this data, at the pixel level, no method
beats cubic interpolation.**

SR2 leads the raw mean-absolute-error table by 30%, and does it by producing a
spectrum at 0.54 of the reference's amplitude — absolute error against a noisy
reference falls when you shrink toward zero, whatever the reconstruction
quality. On a scale-free metric all nine methods land within 1.2% of each
other and SR2 ranks eighth of nine. What does survive is sharper and more
interesting than a leaderboard: SR2 exceeds the *reference's own* line signal
to noise on all four diagnostic lines while recovering only 36–53% of true line
amplitudes, at false-detection rates of 0.30 (Hβ) and 0.44 ([O II]) against
≤ 0.09 for any classical method.

Getting there took four separate metric traps, each of which had to be closed
before the benchmark meant anything. They are documented in
[`docs/GUARDS.md`](docs/GUARDS.md), and they are the part of this repository
most likely to be useful to someone benchmarking something else.

## Quick start

```bash
specsrbench paths           # where it will look for inputs
specsrbench figures all     # rebuild all six paper figures
specsrbench figures 4       # or just one
```

The figures read a cache of derived arrays. If you do not have one, build it
(next section) or point at one you do have:

```bash
export SPECSRBENCH_CACHE=/path/to/cache_logR_tuned
```

| figure | what it shows | command |
|---|---|---|
| 1 | every method on a 1D toy where truth is known | `specsrbench figures toy` |
| 2 | one held-out galaxy, all methods, [O III] inset | `specsrbench figures qualitative` |
| 3 | residual maps over all 572 spectra vs redshift | `specsrbench figures residuals` |
| 4 | global fidelity: MAE, uncertainty-normalised, RMSE | `specsrbench figures mae` |
| 5 | per-line S/N, detection, false detection, width bias | `specsrbench figures per-line-snr` |
| 6 | error against redshift, in equal-count bins | `specsrbench figures redshift` |

Figure 1 trains a small CNN inline and needs `specsrbench[ml]`; the other five
are pure numpy and matplotlib.

## Rebuilding everything from the data

No large files are distributed with this package. The chain runs from the raw
[JADES DR4](https://jades-survey.github.io/) products and the published model
weights, in six stages:

```bash
specsrbench build predictions   # specsr chain over the held-out split (Hub weights)
specsrbench build sets          # eval / tune / calib sets, galaxy-disjoint
specsrbench build lsf --jades-root <JADES DR4>   # measure the instrument LSF
specsrbench build tune          # search the classical parameters under four guards
specsrbench build classical     # the classical reconstructions
specsrbench build lines         # Gaussian line fits, S/N, the summary table
```

`specsrbench build all` runs them in order. `--dry-run` on any stage prints
what it would read and write without doing it.

What each stage needs:

| stage | needs | why |
|---|---|---|
| `predictions` | `specsr`, torch, network | runs SR1 → ZHead → SR2, weights from the Hub |
| `sets` | the paired dataset | draws the three galaxy-disjoint sets |
| `lsf` | raw JADES DR4, `astropy` | measures the prism→grating kernel |
| `tune`, `classical`, `lines` | nothing beyond the above | pure numpy/scipy |

The paired dataset (3.5 GB) is built by `specsr` from the JADES tree and is not
redistributable here:

```bash
specsr build --jades-root <JADES DR4 tree> --out paired_DR4_logR.npz
```

### The one thing to get right

**Match the checkpoint to the wavelength grid by reading its provenance, never
by its filename.** More than one file called `best_sr2.pth` exists, trained on
different grids. Loading the wrong one does not fail — it produces plausible
spectra that are wrong. `specsrbench build predictions` pins the Hub revision
and records what it loaded in the output's `provenance` field.

## Why the LSF is measured and not read

The evaluation products carry a `sigma_pix` array that was widely used as the
deconvolution kernel and **does not describe the data**: it is roughly constant
in nanometres, where a spectrograph's line-spread function is fixed in detector
*pixels*. Measured against the paired spectra it is up to 2.3× too broad at
5 µm. With it, Wiener, Tikhonov and TV merge line pairs that their own input
still resolves.

`specsrbench build lsf` measures the real one from the instrument: the prism
dispersion from the raw x1d products, the effective kernel width from
σ_eff² = σ_LR² − σ_HR² on fits to the four diagnostic lines, and the ratio of
the two. It comes out constant to a few per cent across a factor of three in
wavelength, which is the check that it is an instrumental LSF and not a curve
fitted to noise. The sets this package builds deliberately do **not** carry a
`sigma_pix`, so the kernel can only come from measuring it.

## Tuning is guarded, in four ways

Mean absolute error is not a safe objective, and no single guard catches what
the others do:

| trap | what it rewards | guard |
|---|---|---|
| smoothing | erasing every line incurs no line-shaped residual | median line S/N ≥ 0.9 × no-deconvolution baseline |
| shrinkage | scaling toward zero lowers error against a noisy reference | output std within [0.90, 1.15] of the target's |
| blurring | a unit-gain Wiener filter with snr ≤ 1 can only broaden | median FWHM bias ≤ the baseline's |
| merging | two close lines smeared into one peak look like one good line | resolvable [O III] pairs kept resolved ≥ baseline |

Each is blind to the others: line S/N is amplitude over sideband noise, so a
global rescale leaves it unchanged; the amplitude guard is width-invariant; and
all three of those are single-line statistics, so a Gaussian fitted to a
blended doublet passes them. The fourth is measured on **real** spectra — a
synthetic pair was tried and discarded as far too easy to hold apart.

`specsrbench build tune` applies all four inside the search, and
`tests/test_invariants.py` asserts them against whatever cache is present. A
method is allowed to fail guard 4 only if `classical_params.json` records that
no setting passes it; that is Tikhonov's case, and the paper says so.

## Tests

```bash
pytest                    # the full suite
pytest -m "not slow"      # skip the end-to-end figure builds
```

The suite runs without any data — tests that need a cache skip cleanly. With
one present it checks the cache's structural invariants, the four guards, and
rebuilds a slice of each classical cache from the recorded parameters to
confirm it reproduces.

## Layout

```
src/specsrbench/
  paths.py      where inputs and outputs are found
  methods.py    the nine reconstructions, their names and cache keys
  metrics.py    MAE, scale-free MAE, amplitude ratio, the guards
  classical.py  the seven classical deconvolvers, on the log R=4000 grid
  data.py       the cache loader every figure shares
  figures/      one module per paper figure
  build/        the six pipeline stages
scripts/        talk figures, which are not paper figures
tests/
```

## Documentation

Full documentation, including the API reference, is at
**<https://aryana-haghjoo.github.io/specsr-benchmark/>**.

## Citing

Paper 2 is in preparation. Until it appears, cite the archived software release
and [Haghjoo et al. 2026](https://arxiv.org/abs/2603.18357) for the models.

Every release is archived on Zenodo:

- **[10.5281/zenodo.22104971](https://doi.org/10.5281/zenodo.22104971)** — cite this one unless you
  need a specific version. It is the concept DOI and always resolves to the
  latest release.
- [10.5281/zenodo.22105099](https://doi.org/10.5281/zenodo.22105099) — v0.1.1 specifically, for
  when the exact version matters (a result you are reproducing, say).

`CITATION.cff` carries the machine-readable metadata; GitHub's "Cite this
repository" button reads it.

## License

MIT. See [LICENSE](LICENSE).
