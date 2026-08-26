# Installation

`specsrbench` needs Python 3.10 or newer. It is tested on 3.10, 3.11 and 3.12.

## From PyPI

```bash
pip install specsrbench
```

## From a release

Every release also attaches a wheel and an sdist, and each is archived on
Zenodo with its own DOI. Use these when you need a specific version pinned to a
citable archive:

```bash
pip install https://github.com/aryana-haghjoo/specsr-benchmark/releases/download/v0.1.1/specsrbench-0.1.1-py3-none-any.whl
```

## From source

```bash
git clone https://github.com/aryana-haghjoo/specsr-benchmark
cd specsr-benchmark
pip install -e .
```

## Extras

The base install carries what the classical baselines and five of the six
figures need — numpy, scipy, matplotlib, pandas, PyWavelets, scikit-image — plus
`huggingface_hub`, so that `pip install specsrbench` is enough to fetch the
{doc}`tutorial <tutorial>` sample and run the whole benchmark on it. Installing
a deep-learning stack to run a Wiener filter is a bad trade, so torch is *not* a
dependency.

| extra | pulls in | needed by |
|---|---|---|
| `tutorial` | jupyterlab | running the tutorial notebook — the *data* needs no extra |
| `ml` | torch, `specsr` | `specsrbench build predictions`, and figure 1's toy CNN |
| `lsf` | astropy | `specsrbench build lsf`, which reads raw JADES `x1d` products |
| `dev` | pytest, ruff | the test suite |
| `docs` | sphinx, furo, myst-parser | this site |
| `all` | `ml` + `lsf` + `tutorial` | |

```bash
pip install -e '.[all]'
```

## Pinned versions

`requirements.txt` pins the exact environment every cached array and figure was
verified in (Python 3.11.13). `pyproject.toml` resolves looser bounds and is
what most people want; use the pins if you need a figure to rebuild
identically.

## Checking it works

```bash
specsrbench --version
specsrbench paths          # the directories it will read and write
pytest -m "not slow"       # tests that need data skip cleanly
```

`specsrbench paths` is the first thing to run when something cannot be found:
it prints the four directories being consulted and whether each exists.
