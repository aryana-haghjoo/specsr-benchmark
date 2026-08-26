# API reference

Leaf modules are listed explicitly rather than recursively: several names are
re-exported from their package `__init__`, and a recursive sweep documents those
objects twice, which Sphinx reports as a duplicate description.

```{eval-rst}
.. autosummary::
   :toctree: generated

   specsrbench.paths
   specsrbench.methods
   specsrbench.metrics
   specsrbench.classical
   specsrbench.data
   specsrbench.sample
   specsrbench.style
   specsrbench.cli
   specsrbench.figures
   specsrbench.figures.fig1_toy_methods
   specsrbench.figures.fig2_qualitative
   specsrbench.figures.fig3_residual_maps
   specsrbench.figures.fig4_mae_summary
   specsrbench.figures.fig5_per_line_snr
   specsrbench.figures.fig6_redshift_mae
   specsrbench.build
   specsrbench.build.predictions
   specsrbench.build.sets
   specsrbench.build.lsf
   specsrbench.build.tune
   specsrbench.build.classical_cache
   specsrbench.build.lines
```
