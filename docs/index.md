# specsrbench

**Benchmarking deep learning against classical deconvolution for galaxy
spectral super-resolution.**

Seven classical deconvolution methods — cubic interpolation, Wiener, Tikhonov,
Wiener + total variation, Richardson–Lucy, wavelet-sparse FISTA, and a
redshift-informed matched filter — scored against the SR2 deep-learning
pipeline of [`specsr`](https://github.com/aryana-haghjoo/specsr) on 572
held-out JWST/NIRSpec prism spectra from JADES.

```{admonition} The result, stated plainly
:class: important

**At the pixel level, no method beats cubic interpolation.**

SR2 leads the raw mean-absolute-error table by 30%, and does it by producing a
spectrum at 0.54 of the reference's amplitude. On a scale-free metric all nine
methods land within 1.2% of each other and SR2 ranks *eighth of nine*.

What survives is more interesting than a leaderboard: SR2 exceeds the
reference's **own** line signal-to-noise on all four diagnostic lines while
recovering only 36–53% of true line amplitudes, at false-detection rates of
0.30 (Hβ) and 0.44 ([O II]) against ≤ 0.09 for anything classical.
```

## Quickstart

```bash
specsrbench paths           # where it will look for inputs
specsrbench figures all     # the six paper figures
specsrbench build all       # the cache they read, from JADES DR4 + the Hub
```

Or from Python:

```python
from specsrbench.data import load_cache
from specsrbench.metrics import mae, mae_scalefree, std_ratio

cache = load_cache()
truth, mask = cache.x_high, cache.valid

for key, recon in cache.arrays.items():
    print(f"{key:9s} MAE {mae(recon, truth, mask):.4f}"
          f"  scale-free {mae_scalefree(recon, truth, mask):.4f}"
          f"  amplitude {std_ratio(recon, truth, mask):.3f}")
```

Reporting `mae` without `std_ratio` beside it is how this project twice stated
a conclusion the scale-free number reverses. {doc}`guides/guards` is the long
version of why.

```{admonition} Reference
:class: seealso

Paper 2 is in preparation. For the deep-learning models it benchmarks, see

Haghjoo, A., Hemmati, S., Mobasher, B., et al.
*Learning to See Sharper: A Physics-Informed Artificial Intelligence Framework
for Super-Resolving Galaxy Spectra.*
[arXiv:2603.18357](https://arxiv.org/abs/2603.18357)
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/installation
guides/figures
guides/rebuilding
guides/guards
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```
