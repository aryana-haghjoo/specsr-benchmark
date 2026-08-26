# Tutorial

An executable notebook lives in
[`tutorials_for_user/`](https://github.com/aryana-haghjoo/specsr-benchmark/tree/main/tutorials_for_user).
It runs the whole benchmark on a small sample — no survey data, no GPU, no
configuration. The spectra download once from the Hub and everything after that
is numpy and scipy.

```bash
pip install specsrbench          # enough: fetching the sample is built in
jupyter lab tutorials_for_user/01_quickstart.ipynb
```

`huggingface_hub` is a base dependency, so the plain install can already reach
the sample. The `[tutorial]` extra adds only JupyterLab, for people who do not
have a notebook environment already.

## What it covers

**[`01_quickstart.ipynb`](https://github.com/aryana-haghjoo/specsr-benchmark/blob/main/tutorials_for_user/01_quickstart.ipynb)
— the benchmark in miniature.**

1. **The data.** 24 held-out JADES galaxies: the prism spectrum that goes in,
   the grating spectrum every method is scored against, and what the z-score
   normalisation does and does not preserve.
2. **The line-spread function.** The measured kernel in detector pixels, and why
   an LSF that is instead constant in wavelength merges line pairs the input
   still resolves.
3. **Running the baselines.** All six classical deconvolvers at the tuned
   parameters, in a few seconds, via {meth}`~specsrbench.sample.Sample.reconstruct`.
4. **Scoring.** {func}`~specsrbench.metrics.mae`,
   {func}`~specsrbench.metrics.mae_scalefree` and
   {func}`~specsrbench.metrics.std_ratio` side by side — and the reversal
   between the first two that this paper is named after.
5. **The trap, demonstrated.** Take a method that does nothing, multiply it by a
   constant, and watch MAE improve by a quarter while the scale-free metric does
   not move at all. Run live, not asserted.
6. **The four guards.** What each one catches that the others cannot.
7. **Your own spectrum.** Pointing the deconvolvers at data of your own, and the
   one check to run before trusting the output.

## The bundled data

The sample is hosted on the Hub rather than committed, and downloads on first
use to `~/.cache/huggingface`:

<https://huggingface.co/datasets/aryana-haghjoo/specsr-benchmark>

```python
from specsrbench.sample import load_sample

s = load_sample()          # downloads once, cached thereafter
print(s.summary())

recon = s.reconstruct()    # all six classical methods, tuned parameters
```

`SPECSRBENCH_SAMPLE` points at a local `.npz` instead and wins outright — no
network and no Hub account. `SPECSRBENCH_SAMPLE_REPO` and
`SPECSRBENCH_SAMPLE_REVISION` override where it is fetched from.

The 24 galaxies are the 572-spectrum evaluation set sorted by redshift and
sampled at evenly spaced ranks, so they are held out of training by construction
and span z = 0.31 to 13.86 rather than being chosen for how good they look. The
archive also carries the SR2 predictions precomputed and the matched filter's
line list, so **every method in the notebook runs without torch**.

```{admonition} These are 24 galaxies, not 572
:class: warning

The ordering of the methods reproduces the paper's and so does the lesson, but
the individual numbers carry the error bar of a small sample and are not the
paper's. Quote the paper for the paper's numbers.
```

```{admonition} Absolute flux scale is not part of this problem
:class: note

Every spectrum is per-spectrum z-scored, which is what makes an error metric
comparable across galaxies whose brightnesses differ by orders of magnitude.
Nothing in this package predicts the flux scale.
```

## Next

{doc}`guards` is the long version of step 6; {doc}`rebuilding` runs the real
benchmark on all 572 spectra.
