"""The tutorial's data and its claims.

The notebook prints performance numbers and tells the reader they are honest
held-out behaviour.  Three separate things have to be true for that, and none of
them is self-evident from the file:

1. **The galaxies are held out.**  If one had been in training, every number the
   notebook prints would be quietly optimistic and it would be making a claim it
   cannot support.  This is not hypothetical for this project family -- paper 1
   split an augmented dataset *by row*, so 99.2% of galaxies had near-duplicate
   siblings on both sides.
2. **The parameters are the tuned ones.**  The archive carries a copy of
   ``classical_params.json`` so the notebook needs no cache.  A copy is a thing
   that can drift, and this exact drift -- the tuner retuned against a corrected
   kernel while a second copy went on with the old values -- made every
   classical number in the paper wrong for a day while 106 tests passed.
3. **The kernel is the derived one.**  The evaluation set ships a ``sigma_pix``
   that does not describe the data.  A sample built from it would merge line
   pairs and hand the reader a tutorial that quietly libels the classical
   methods.

The notebook's *prose* claims are recomputed here too.  A number is only safe if
a test recomputes it, and that applies to "SR2 ranks last once the scale is
taken out" exactly as it applies to a table cell -- this project's results
prose went stale while its guarded table stayed right, twice.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO, SRC, STD_RATIO_HI, STD_RATIO_LO, mae, mae_scalefree, std_ratio

CACHE = REPO / "cache_logR_tuned"
NOTEBOOK = REPO / "tutorials_for_user" / "01_quickstart.ipynb"

# The sample is not committed -- it lives on the Hub.  Tests never download it:
# a test that needs the network is a test that fails for reasons unrelated to
# the code.  Point SPECSRBENCH_SAMPLE at one, or build it into build/.
_STAGED = REPO / "build" / "specsrbench_sample.npz"


def _sample_path() -> Path | None:
    if (env := os.environ.get("SPECSRBENCH_SAMPLE")) and Path(env).exists():
        return Path(env)
    return _STAGED if _STAGED.exists() else None


@pytest.fixture(scope="session")
def sample():
    p = _sample_path()
    if p is None:
        pytest.skip("no tutorial sample staged; "
                    "run scripts/make_tutorial_sample.py or set SPECSRBENCH_SAMPLE")
    from specsrbench.sample import load_sample
    return load_sample(p)


# ── 1. the galaxies are held out ──────────────────────────────────────────────
def test_every_sample_galaxy_is_in_the_held_out_split(sample):
    """Each sampled galaxy must come from the evaluation set, by identity."""
    if not (SRC / "eval_set.npz").exists():
        pytest.skip("cache_logR/eval_set.npz not present")
    E = np.load(SRC / "eval_set.npz", allow_pickle=True)
    held_out = set(np.asarray(E["parent_id"]).tolist())

    ids = np.asarray(sample._d["parent_id"]).tolist()
    missing = sorted(i for i in ids if i not in held_out)
    assert not missing, (
        f"{len(missing)} sample galaxy(ies) are not in the evaluation set "
        f"({missing[:5]}), so their provenance cannot be verified at all.")


def test_no_sample_galaxy_was_used_for_tuning(sample):
    """The classical parameters were chosen on tune_set; scoring on it would leak."""
    if not (SRC / "tune_set.npz").exists():
        pytest.skip("cache_logR/tune_set.npz not present")
    T = np.load(SRC / "tune_set.npz", allow_pickle=True)
    tuning = set(np.asarray(T["parent_id"]).tolist())

    leaked = sorted(set(np.asarray(sample._d["parent_id"]).tolist()) & tuning)
    assert not leaked, (
        f"{len(leaked)} sample galaxy(ies) were used to tune the classical "
        f"parameters ({leaked[:5]}); the notebook scores methods on them.")


def test_the_sample_is_not_cherry_picked(sample):
    """Evenly spaced redshift ranks, reproducible from the evaluation set alone."""
    if not (SRC / "eval_set.npz").exists():
        pytest.skip("cache_logR/eval_set.npz not present")
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from make_tutorial_sample import select

    E = np.load(SRC / "eval_set.npz", allow_pickle=True)
    z = np.asarray(E["z_true"], dtype=np.float64)
    expected = np.asarray(E["parent_id"])[select(z, sample.n_spectra)]
    assert np.array_equal(np.asarray(sample._d["parent_id"]), expected), (
        "the sample is not the selection rule's output; it was hand-edited or "
        "built from a different evaluation set")


# ── 2. the parameters have not drifted ────────────────────────────────────────
@pytest.mark.parametrize("method", ["wiener", "tikhonov", "rl", "sparse", "tv", "mf"])
def test_sample_parameters_match_the_tuner(sample, method):
    """The archive's copy must equal the tuner's own output file, block by block."""
    p = CACHE / "classical_params.json"
    if not p.exists():
        pytest.skip("cache_logR_tuned/classical_params.json not present")
    truth = json.loads(p.read_text())["tuned"][method]
    assert sample.tuned[method] == truth, (
        f"the tutorial sample's {method} parameters have drifted from "
        f"classical_params.json: {sample.tuned[method]} != {truth}")


def test_the_recorded_guard_failure_travels_with_the_sample(sample):
    """Tikhonov is retained despite failing a guard, and the notebook says so."""
    assert "tikhonov" in sample.params["failed_guards"], (
        "failed_guards no longer records Tikhonov; the notebook prints it and "
        "explains why the method is retained anyway")


# ── 3. the kernel is the derived one ──────────────────────────────────────────
def test_sample_kernel_is_the_one_the_caches_were_built_with(sample):
    if not (CACHE / "sigma_pix.npy").exists():
        pytest.skip("cache_logR_tuned/sigma_pix.npy not present")
    np.testing.assert_allclose(
        sample.sigma_pix, np.load(CACHE / "sigma_pix.npy"), rtol=0, atol=0,
        err_msg="the sample carries a different LSF than the caches were built with")


def test_sample_kernel_is_not_the_eval_set_trap(sample):
    """eval_set's own ``sigma_pix`` is up to 2.3x too broad and merges line pairs."""
    if not (SRC / "eval_set.npz").exists():
        pytest.skip("cache_logR/eval_set.npz not present")
    E = np.load(SRC / "eval_set.npz", allow_pickle=True)
    assert not np.allclose(sample.sigma_pix, np.asarray(E["sigma_pix"])), (
        "the sample ships eval_set.npz's sigma_pix, which does not describe the "
        "data; use specsrbench.classical.load_sigma_pix()")


def test_the_sample_carries_no_array_named_like_the_trap(sample):
    """A fresh build must not be able to recreate the trap by accident."""
    assert "sigma_pix" in sample._d.files
    for name in ("flux_high_err_raw", "eval_sigma_pix"):
        assert name not in sample._d.files


def test_the_sample_leaks_no_absolute_paths(sample):
    """The archive is published to a public Hub repo; a home path there is forever.

    This is not hypothetical.  The evaluation set records the SR1 config and the
    paired dataset by *absolute* path, and the first build of this sample copied
    both straight through.  ``make_public_release.sh`` greps the assembled code
    tree for exactly this pattern, but it never sees an ``.npz`` on its way to
    the Hub -- so the guard has to exist here too.
    """
    import re

    for key in ("provenance", "params"):
        text = str(sample._d[key])
        found = re.findall(r"/(?:home|Users)/[^\s'\"]*", text)
        assert not found, f"sample {key} carries absolute home path(s): {found[:3]}"


# ── 4. the notebook itself ────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def notebook():
    if not NOTEBOOK.exists():
        pytest.skip(f"{NOTEBOOK.name} not present")
    return json.loads(NOTEBOOK.read_text())


def test_notebook_has_no_stored_errors(notebook):
    """A shipped notebook whose outputs are tracebacks teaches the traceback."""
    bad = [(n, o.get("ename"))
           for n, c in enumerate(notebook["cells"]) if c["cell_type"] == "code"
           for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not bad, f"notebook cells raised: {bad}"


def test_every_notebook_code_cell_was_executed(notebook):
    """Unexecuted cells mean the outputs below them describe a different run."""
    unrun = [n for n, c in enumerate(notebook["cells"])
             if c["cell_type"] == "code" and c.get("execution_count") is None]
    assert not unrun, f"code cells {unrun} were never executed"


def test_notebook_leaks_no_local_paths(notebook):
    """An absolute home path in a shipped file leaks the machine layout.

    The needles are assembled from fragments rather than written out, because
    the release script greps the assembled tree for exactly these strings and a
    test that spells them literally fails that check itself.  It did.
    """
    body = NOTEBOOK.read_text()
    for leak in ("/" + "home/", "ahagh010" + "@"):
        assert leak not in body, f"{NOTEBOOK.name} contains {leak!r}"


def test_notebook_carries_no_download_progress_widgets(notebook):
    """Executing with a cold Hub cache bakes two progress bars into cell 2.

    They render as an empty box or a raw JSON blob depending on the viewer, and
    they carry ipywidgets state that some readers reject outright. Anyone who
    re-executes this notebook on a fresh machine will reintroduce them, so this
    is a standing check rather than a one-off cleanup.
    """
    widget = "application/vnd.jupyter.widget-view+json"
    stray = [n for n, c in enumerate(notebook["cells"]) if c["cell_type"] == "code"
             for o in c.get("outputs", []) if widget in o.get("data", {})]
    assert not stray, (
        f"cells {stray} carry ipywidget outputs (Hub download progress bars); "
        "strip them before committing")
    assert "widgets" not in notebook.get("metadata", {}), \
        "stale ipywidgets state in notebook metadata"


def test_notebook_fetches_the_sample_rather_than_a_local_file(notebook):
    """``load_sample()`` with no argument is the only form that works for a reader."""
    src = "\n".join("".join(c["source"]) for c in notebook["cells"]
                    if c["cell_type"] == "code")
    assert "load_sample()" in src, "the notebook must call load_sample() with no path"
    assert "SPECSRBENCH_SAMPLE" not in src, (
        "the notebook hardcodes the local-override env var; readers have no such file")


def test_notebook_is_reachable_from_the_docs():
    page = REPO / "docs" / "guides" / "tutorial.md"
    assert page.exists(), "docs/guides/tutorial.md is missing"
    assert "guides/tutorial" in (REPO / "docs" / "index.md").read_text(), \
        "the tutorial page is not in the docs toctree, so nothing links to it"
    assert "01_quickstart" in page.read_text()


def test_fetching_the_sample_needs_no_extra():
    """``pip install specsrbench`` must be enough to run ``load_sample()``.

    ``huggingface_hub`` began life in the ``[ml]`` and ``[tutorial]`` extras,
    which meant the plain install could not reach its own tutorial data --
    the first thing most people try.  Moving it to the base dependencies is a
    deliberate promise, and a promise made only in prose is one that regresses
    the next time someone tidies the dependency list.
    """
    # tomllib is 3.11+, and the package supports 3.10.  The promise is the
    # same on every interpreter, so checking it on the newer ones in the
    # matrix is enough; skipping is honest, failing to parse is not.
    tomllib = pytest.importorskip("tomllib")

    body = tomllib.loads((REPO / "pyproject.toml").read_text())
    base = " ".join(body["project"]["dependencies"])
    assert "huggingface_hub" in base, (
        "huggingface_hub is not a base dependency; `pip install specsrbench` "
        "can no longer fetch the tutorial sample")

    extras = body["project"]["optional-dependencies"]
    assert not any("huggingface_hub" in d for d in extras.get("tutorial", [])), \
        "the tutorial extra repeats a base dependency"


def test_the_install_instructions_do_not_demand_an_extra():
    """Every entry point tells the reader the same, correct thing."""
    import json

    nb = json.loads(NOTEBOOK.read_text()) if NOTEBOOK.exists() else {"cells": []}
    sources = {
        "README.md": (REPO / "README.md").read_text(),
        "docs/guides/tutorial.md": (REPO / "docs/guides/tutorial.md").read_text(),
        "notebook": "\n".join("".join(c["source"]) for c in nb["cells"]),
    }
    for name, body in sources.items():
        assert "pip install specsrbench" in body, f"{name} has no install line"
        assert "specsrbench[tutorial]\njupyter" not in body, (
            f"{name} still tells the reader the sample needs the tutorial extra")


def test_the_tutorial_ships():
    script = REPO / "scripts" / "make_public_release.sh"
    if not script.exists():
        pytest.skip("release script not present")
    assert "tutorials_for_user" in script.read_text(), \
        "the tutorial is not in the release manifest, so it will not ship"


# ── 5. the claims the notebook makes in prose ─────────────────────────────────
@pytest.fixture(scope="session")
def scored(sample):
    """The notebook's own table, recomputed."""
    recon = sample.reconstruct()
    return {name: (mae(a, sample.x_high, sample.valid),
                   mae_scalefree(a, sample.x_high, sample.valid),
                   std_ratio(a, sample.x_high, sample.valid))
            for name, a in recon.items()}


def test_sr2_leads_raw_mae_on_the_sample(scored):
    best = min(scored, key=lambda k: scored[k][0])
    assert best == "ML (SR2)", (
        f"the notebook says SR2 leads raw MAE; on this sample {best} does")


def test_sr2_is_last_scale_free_on_the_sample(scored):
    """The reversal is the notebook's whole argument; if it stops holding, say so."""
    worst = max(scored, key=lambda k: scored[k][1])
    assert worst == "ML (SR2)", (
        f"the notebook says SR2 ranks last once the scale is taken out; "
        f"on this sample {worst} does")


def test_sr2_falls_outside_the_amplitude_guard(scored):
    r = scored["ML (SR2)"][2]
    assert r < STD_RATIO_LO, (
        f"the notebook flags SR2 as the one method outside the amplitude band; "
        f"its std_ratio is {r:.3f}, inside [{STD_RATIO_LO}, {STD_RATIO_HI}]")


def test_tikhonov_is_the_only_other_method_near_the_edge(scored):
    """The notebook says Tikhonov sits *just inside* the upper edge, held there."""
    r = scored["Tikhonov"][2]
    assert 1.10 < r <= STD_RATIO_HI, (
        f"Tikhonov's std_ratio is {r:.3f}; the notebook describes it as sitting "
        f"just inside the upper edge of [{STD_RATIO_LO}, {STD_RATIO_HI}]")


def test_shrinking_a_do_nothing_method_improves_its_mae(sample):
    """The notebook's central demonstration, recomputed.

    If this ever stopped being true the notebook would be teaching a trap that
    does not exist -- and the four guards would have nothing to guard.
    """
    nothing = sample.x_low
    honest = mae(nothing, sample.x_high, sample.valid)
    shrunk = min(mae(g * nothing, sample.x_high, sample.valid)
                 for g in np.linspace(0.1, 1.0, 19))
    assert shrunk < honest * 0.9, (
        f"shrinking gains only {(honest - shrunk) / honest:.1%}; the notebook "
        "claims MAE meaningfully rewards it")


def test_scale_free_mae_is_invariant_under_that_rescale(sample):
    """The other half of the demonstration: the corrected metric does not move."""
    nothing = sample.x_low
    vals = [mae_scalefree(g * nothing, sample.x_high, sample.valid)
            for g in (0.3, 0.7, 1.0, 1.3)]
    assert np.ptp(vals) < 1e-9, (
        f"scale-free MAE varies by {np.ptp(vals):.2e} under a global rescale; "
        "the notebook states it is invariant by construction")
