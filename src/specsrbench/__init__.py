"""Benchmarking deep learning against classical deconvolution for galaxy spectra.

The companion package to :mod:`specsr` (`arXiv:2603.18357
<https://arxiv.org/abs/2603.18357>`_).  Where ``specsr`` trains and runs the
super-resolution models, this package scores them against seven classical
deconvolution baselines on JWST/NIRSpec prism spectra from JADES, and draws the
figures of the resulting paper.

Nothing here reads a notebook.  Every figure and every cached array is produced
by a module in this package and reachable from the command line::

    specsrbench figures all          # the six paper figures, from the cache
    specsrbench build all            # the cache itself, from JADES + the Hub

The second command needs the raw JADES DR4 tree; the first does not.  See
:mod:`specsrbench.paths` for where each looks for its inputs.
"""
from __future__ import annotations

__version__ = "0.1.2"

__all__ = ["__version__"]
