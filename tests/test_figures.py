"""End-to-end: the figures are the product, so they must build.

Marked ``slow``; deselect with ``-m 'not slow'``.  "It ran without crashing" is
not the assertion -- each figure's printed numbers are checked against the
cache, because a figure can be drawn cleanly and be wrong.

This replaced ``test_notebooks.py`` when the six figure notebooks became
modules.  The checks it carried over are the ones that had caught real bugs:
figure 4's independently recomputed MAE table, and the flux-uncertainty
sanity bound that catches an unmasked sentinel.
"""
from __future__ import annotations

import io
import re
from contextlib import redirect_stdout

import pytest

from conftest import CACHE, mae

pytest.importorskip("matplotlib")

FIGURES = ["toy", "qualitative", "residuals", "mae", "per-line-snr", "redshift"]

#: figure name -> the file it must write.
OUTPUTS = {
    "toy": "fig_toy_1d.pdf",
    "qualitative": "fig_jades_qualitative.pdf",
    "residuals": "fig_residual_maps.pdf",
    "mae": "fig_mae_summary.pdf",
    "per-line-snr": "fig_jades_per_line_snr.pdf",
    "redshift": "fig_redshift_mae.pdf",
}


def _build(name, tmp_path):
    """Build one figure into a scratch directory, returning its stdout."""
    from specsrbench import figures

    buf = io.StringIO()
    with redirect_stdout(buf):
        out = figures.build(name, outdir=tmp_path)
    return buf.getvalue(), out


# ── structure: cheap, and they do not need the cache ──────────────────────────
def test_every_paper_figure_has_a_module():
    from specsrbench import figures

    assert set(figures.REGISTRY) == set(FIGURES)
    assert len(figures.BY_NUMBER) == 6, "the paper has six figures"
    for number, key in figures.BY_NUMBER.items():
        assert key in figures.REGISTRY, f"figure {number} maps to unknown {key!r}"


def test_figures_resolve_by_number_and_name():
    from specsrbench.figures import resolve

    assert resolve("4") == resolve("fig4") == resolve("mae") == "mae"
    with pytest.raises(KeyError):
        resolve("7")


def test_no_figure_notebooks_remain():
    """The figures are modules now; a stray notebook is a second source of truth.

    Notebooks store their outputs next to their code, so a stale number in one
    reads as data in a diff.  Every figure notebook became a module in
    ``specsrbench.figures`` on 2026-08-25; nothing should reintroduce one.

    ``tutorials_for_user/`` is deliberately exempt and separately guarded below.
    A user tutorial is not a second source of truth for a figure -- it produces
    no paper artefact and the paper does not read it.  The exemption is by
    directory rather than by filename so that a figure notebook cannot slip back
    in under a tutorial-sounding name somewhere else in the tree.
    """
    from conftest import REPO

    tutorials = REPO / "tutorials_for_user"
    stray = [p for p in REPO.rglob("*.ipynb")
             if ".ipynb_checkpoints" not in str(p) and "venv" not in str(p)
             and tutorials not in p.parents]
    assert not stray, f"notebooks reintroduced: {[str(p) for p in stray]}"


def test_the_tutorial_notebook_produces_no_paper_artefact():
    """The exemption above holds only while the tutorial stays a tutorial.

    The moment a notebook writes into ``figures/`` or imports a figure module it
    has become a second way to build a paper artefact, which is exactly what
    converting the six notebooks to modules was meant to end.
    """
    import json

    from conftest import REPO

    nb = REPO / "tutorials_for_user" / "01_quickstart.ipynb"
    if not nb.exists():
        pytest.skip("tutorial notebook not present")
    src = "\n".join("".join(c["source"])
                     for c in json.loads(nb.read_text())["cells"]
                     if c["cell_type"] == "code")
    for banned in ("specsrbench.figures", "figures_dir", "savefig"):
        assert banned not in src, (
            f"the tutorial notebook uses {banned!r}; it must not build or write "
            "a paper figure")


def test_no_figure_module_names_a_cache_directory():
    """Figures go through ``paths``/``load_cache``; none may open one by name.

    Every notebook used to carry its own repo-root walk and its own cache
    directory name.  When the cache moved from ``cache/`` to ``cache_logR/`` to
    ``cache_logR_tuned/`` that was six edits, and a figure left behind on a
    superseded cache draws a plausible picture of superseded numbers.
    """
    from conftest import REPO

    figdir = REPO / "src" / "specsrbench" / "figures"
    for path in figdir.rglob("*.py"):
        body = path.read_text()
        assert "spectra_dataset_2500" not in body, f"{path.name} reads the raw dataset"
        for name in ('"cache"', '"cache_logR"', '"cache_logR_tuned"'):
            assert name not in body, \
                f"{path.name} names a cache directory ({name}); use paths.cache_dir()"


def test_every_module_imports_without_any_data():
    """Importing must never require a cache to exist.

    Three build stages used to read ``eval_set.npz`` at module scope, so
    ``import specsrbench.build.tune`` raised on any machine without the data.
    That broke the API documentation build, and would have broken the import
    for anyone who installed the package and had not built a cache yet.

    Run in a subprocess with the paths pointed at nothing, because the modules
    are already imported in this one.
    """
    import os
    import subprocess
    import sys

    from conftest import REPO

    mods = ["specsrbench", "specsrbench.cli", "specsrbench.classical",
            "specsrbench.data", "specsrbench.figures",
            "specsrbench.build.predictions", "specsrbench.build.sets",
            "specsrbench.build.lsf", "specsrbench.build.tune",
            "specsrbench.build.classical_cache", "specsrbench.build.lines"]
    env = dict(os.environ,
               SPECSRBENCH_CACHE="/nonexistent", SPECSRBENCH_SETS="/nonexistent",
               PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([sys.executable, "-c",
                        "import " + ", ".join(mods)],
                       capture_output=True, text=True, env=env, timeout=180)
    assert r.returncode == 0, f"import failed with no data present:\n{r.stderr[-2000:]}"


# ── end to end ────────────────────────────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.parametrize("name", [f for f in FIGURES if f != "toy"])
def test_figure_builds(name, tmp_path, cache):
    text, out = _build(name, tmp_path)
    assert out.exists() and out.stat().st_size > 10_000, f"{name} wrote nothing useful"
    assert out.name == OUTPUTS[name]
    assert "572 held-out spectra" in text, f"{name} did not report the sample it used"


@pytest.mark.slow
def test_toy_figure_builds(tmp_path):
    """Figure 1 is synthetic and needs torch; it does not touch the cache."""
    pytest.importorskip("torch")
    pytest.importorskip("skimage")
    text, out = _build("toy", tmp_path)
    assert out.exists() and out.stat().st_size > 10_000
    assert "All 8 methods produced finite outputs" in text


@pytest.mark.slow
def test_figure4_table_matches_the_cache(tmp_path, reconstructions, x_high, valid):
    """Figure 4 recomputes MAE independently; it must agree with the cache.

    This is the cross-check that caught the flux-error sentinel bug.
    """
    text, _ = _build("mae", tmp_path)
    label = {"Cubic (LR)": "Cubic (LR)", "Wiener": "Wiener", "Tikhonov": "Tikhonov",
             "Wiener + TV": "TV", "R-L": "R-L", "Sparse": "Sparse",
             "Wiener + MF": "Wiener + MF", "ML (SR2)": "ML (SR2)"}
    checked = 0
    # Longest label first: "Wiener + TV" must not be matched by "Wiener".
    ordered = sorted(label.items(), key=lambda kv: -len(kv[0]))
    for line in text.splitlines():
        for shown, key in ordered:
            if line.strip().startswith(shown):
                nums = re.findall(r"-?\d+\.\d+", line)
                if not nums:
                    continue
                assert float(nums[0]) == pytest.approx(
                    mae(reconstructions[key], x_high, valid), abs=2e-3), \
                    f"figure 4 prints MAE {nums[0]} for {shown}, cache disagrees"
                checked += 1
                break
    assert checked >= 7, f"only cross-checked {checked} methods"


@pytest.mark.slow
def test_figure4_flux_uncertainty_is_sane(tmp_path, cache):
    """Regression: an unmasked sentinel drove this to 3.2e18 instead of ~0.5."""
    text, _ = _build("mae", tmp_path)
    m = re.search(r"Mean normalized flux uncertainty:\s*([0-9.eE+]+)", text)
    assert m, "figure 4 no longer reports the flux uncertainty it normalises by"
    assert 0.1 < float(m.group(1)) < 5.0, \
        f"mean normalised flux uncertainty is {m.group(1)}"


def test_figure6_bins_are_equal_count(cache):
    """Equal-count bins, not equal-width: the sample spans z = 0.3 to 13.9.

    Equal-width bins would put almost everything in the first two and leave the
    rest reporting the error of a handful of galaxies.

    The bins are half-open ``[lo, hi)`` and the top edge is the 100th
    percentile, so the single highest-redshift galaxy falls outside the last
    bin: 571 of 572 are plotted.  Asserted rather than fixed, because the
    published figure is drawn this way and one galaxy of 572 does not move any
    bin's median.
    """
    from specsrbench.data import load_cache
    from specsrbench.figures.fig6_redshift_mae import compute

    _labels, counts, _mae, _mids, _edges = compute(load_cache(CACHE))
    assert sum(counts) == 571, f"expected 571 binned (one at the top edge): {counts}"
    assert max(counts) - min(counts) <= 1, f"bins are not equal-count: {counts}"
