# Tutorials

Executable notebooks that run the benchmark on a small sample, with no survey
data and no configuration.

```bash
pip install specsrbench          # enough: fetching the sample is built in
jupyter lab tutorials_for_user/01_quickstart.ipynb
```

Add the `[tutorial]` extra if you do not already have a notebook environment —
it pulls in JupyterLab and nothing else.

## The notebooks

**[`01_quickstart.ipynb`](01_quickstart.ipynb) — the benchmark in miniature.**
Load 24 held-out JADES spectra, run all six classical deconvolvers at the tuned
parameters, and score them against the grating reference. Then the part that
matters: a live demonstration that mean absolute error rewards a spectrum that
has merely been shrunk, why that reverses the headline result, and what the four
guards in `specsrbench.build.tune` are each for. Ends with how to point the
deconvolvers at a spectrum of your own.

## The data

The sample is **not committed here**. It downloads on first use from the Hub, to
`~/.cache/huggingface`:

<https://huggingface.co/datasets/aryana-haghjoo/specsr-benchmark>

That is deliberate: `specsrbench` ships as a code-only package, and the release
process fails loudly if any data file reaches the published tree. The tutorial
fetches its data the same way the ML arm fetches its weights.

To work against a local copy instead — a rebuild that is not published yet, or a
machine with no network — point `SPECSRBENCH_SAMPLE` at an `.npz`:

```bash
SPECSRBENCH_SAMPLE=/path/to/specsrbench_sample.npz jupyter lab
```

| array | contents |
|---|---|
| `x_low` | prism input (R ~ 100) on the fine grid — the input, and the do-nothing baseline |
| `x_high` | grating reference (R ~ 1000) — what every method is scored against |
| `x_high_err` | reference flux uncertainty, NaN where invalid |
| `sr2` | the SR2 deep-learning prediction, precomputed, so no torch is needed |
| `valid` | pixels where the reference is real rather than padding |
| `wave`, `sigma_pix` | the R = 4000 log grid, and the **measured** LSF in detector pixels |
| `z`, `z_pred` | spectroscopic redshift, and the redshift head's estimate |
| `mf_lines` | rest wavelengths the matched filter uses, so it runs without `specsr` |
| `params` | the tuned classical parameters, copied from the tuner's own output |
| `provenance` | what these spectra are and what produced the arrays beside them |

The 24 galaxies are the 572-spectrum evaluation set sorted by redshift and
sampled at evenly spaced ranks: held out of training by construction, spanning
z = 0.31 to 13.86, and **not** chosen for how good they look. A tutorial that
prints performance numbers off a hand-picked set of bright galaxies is quietly
claiming something the benchmark does not support.

## Two things the notebook is careful about

**These are 24 galaxies, not 572.** The ordering of the methods reproduces and
so does the lesson. The individual numbers carry a small sample's error bar and
are not the paper's.

**Absolute flux scale is not part of this problem.** Every spectrum is
per-spectrum z-scored, which is what makes an error metric comparable across
galaxies of wildly different brightness. Nothing here predicts the flux scale.

## Regenerating the sample

The sample is derived from the full evaluation set and the tuned parameters by a
maintainer script, and the selection is a pure function of that evaluation set —
the same inputs give the same 24 galaxies. `tests/test_tutorial_sample.py`
checks the result against the split, so a sample rebuilt from a different one
fails there rather than silently making the notebook's numbers optimistic.
