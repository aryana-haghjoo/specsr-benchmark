# Building the figures

The six paper figures are Python modules, one per figure, under
`specsrbench.figures`. There are no notebooks.

```bash
specsrbench figures all              # all six
specsrbench figures 4                # by paper figure number
specsrbench figures mae              # or by name
specsrbench figures all --outdir /tmp/check    # somewhere other than figures/
```

| # | name | what it shows | output |
|---|---|---|---|
| 1 | `toy` | every method on a 1D toy where the truth is known | `fig_toy_1d.pdf` |
| 2 | `qualitative` | one held-out galaxy, all methods, [O III] inset | `fig_jades_qualitative.pdf` |
| 3 | `residuals` | residual maps over all 572 spectra against redshift | `fig_residual_maps.pdf` |
| 4 | `mae` | global fidelity: MAE, uncertainty-normalised, RMSE | `fig_mae_summary.pdf` |
| 5 | `per-line-snr` | per-line S/N, detection, false detection, width bias | `fig_jades_per_line_snr.pdf` |
| 6 | `redshift` | error against redshift, in equal-count bins | `fig_redshift_mae.pdf` |

Each prints the numbers the paper quotes from it, so a figure that is wrong is
usually visible in its own output before anyone looks at the PDF.

## From Python

```python
from specsrbench import figures

path = figures.build("mae", outdir="/tmp")
```

Every figure module exposes `build(cache=None, outdir=None) -> Path`. Some take
more: figure 2 accepts `i_show` to draw a different galaxy, and figure 1 takes
`deterministic=True` and `device=`.

Several also expose the computation separately from the drawing, which is what
the tests check:

```python
from specsrbench.data import load_cache
from specsrbench.figures.fig5_per_line_snr import compute

rows = compute(load_cache())
rows["fdr"]["SR2"]        # false-detection rate per line
```

## Where the data comes from

The figures read a cache of derived arrays — by default a `cache_logR_tuned/`
directory found by walking up from the working directory. Point somewhere else
with:

```bash
export SPECSRBENCH_CACHE=/path/to/cache_logR_tuned
```

If you have no cache, {doc}`rebuilding` explains how to make one.

## Reproducibility

Five of the six figures are deterministic and were verified to render
**pixel-identical** to the published PDFs.

Figure 1 is the exception and always was. It trains a small 1D CNN inline to
make its learned panel, and cuDNN reduces convolutions in a non-deterministic
order, so that panel moves in the third or fourth decimal between runs — RMSE
0.0143 against 0.0144 when it was checked. No conclusion moves; the CNN wins
every metric on that toy by a wide margin. Pass `deterministic=True` to trade
some speed for a fixed result.

```{warning}
Rebuilding overwrites `figures/*.pdf`. The content is deterministic but the PDF
metadata is not, so a rebuild shows up as a diff even when nothing changed.
Use `--outdir` when you only want to check.
```
