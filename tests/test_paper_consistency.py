"""Every number the paper states must match the cache it was computed from.

Stale numbers in the manuscript have been the single most persistent failure
mode in this project: the analysis is rebuilt, the figures are regenerated, and
a sentence three sections away still quotes the old value.  These tests parse
the manuscript and check it against a fresh computation.
"""
from __future__ import annotations

import csv
import re

import numpy as np
import pytest

from conftest import CACHE, REPO, mae_scalefree, std_ratio

PAPER = REPO / "paper" / "paper.tex"


@pytest.fixture(scope="module")
def tex():
    if not PAPER.exists():
        pytest.skip("paper.tex not present")
    return PAPER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def summary():
    path = CACHE / "summary_final.csv"
    if not path.exists():
        pytest.skip("summary_final.csv not present")
    return {r["Method"]: r for r in csv.DictReader(path.open())}


# figure 4 label -> summary_final.csv label
FIG4_ROWS = {
    "Cubic (LR)": "Cubic (LR)",
    "Wiener": "Wiener",
    "Tikhonov": "Tikhonov",
    "Wiener + TV": "TV",
    "R-L": "R-L",
    "Sparse": "Sparse",
    "Wiener + MF": "Wiener + MF",
    "ML (SR2)": "ML (SR2)",
}


def test_figure4_panels_match_the_cache(tex, summary, cache):
    """Every bar of figure 4's MAE, amplitude and scale-free panels.

    These three panels replaced the manuscript's global-fidelity table on
    2026-10-02, and this test replaced the one that parsed that table.  The
    numbers a reader takes from the paper are now the ones the figure module
    computes, so they are checked against the build's own summary instead.
    """
    from specsrbench.data import load_cache
    from specsrbench.figures.fig4_mae_summary import compute

    assert "deluxetable" not in tex, "a table is back in paper.tex; guard its rows"
    rows = {r["Method"]: r for r in compute(load_cache(cache))[0]}
    for shown, key in FIG4_ROWS.items():
        got, want = rows[shown], summary[key]
        assert got["MAE"] == pytest.approx(float(want["MAE"]), abs=0.002), \
            f"{shown}: figure 4 plots MAE {got['MAE']:.4f}, cache says {want['MAE']}"
        assert got["AmpRatio"] == pytest.approx(float(want["std_ratio"]), abs=0.002), \
            f"{shown}: figure 4 plots amplitude {got['AmpRatio']:.4f}, " \
            f"cache says {want['std_ratio']}"
        assert got["MAE_scalefree"] == pytest.approx(float(want["MAE_scalefree"]), abs=0.0005), \
            f"{shown}: figure 4 plots scale-free {got['MAE_scalefree']:.4f}, " \
            f"cache says {want['MAE_scalefree']}"

    # The caption's reading of the panels.
    sf = {k: rows[k]["MAE_scalefree"] for k in FIG4_ROWS}
    order = sorted(sf, key=sf.get)
    assert order[0] == "Cubic (LR)", "cubic interpolation no longer leads scale-free"
    assert order.index("ML (SR2)") > order.index("Cubic (LR)")
    assert 0.45 <= rows["ML (SR2)"]["AmpRatio"] <= 0.55, \
        "caption says SR2 is at about half the reference's amplitude"


def test_sample_size_is_stated_correctly(tex, x_high):
    n = x_high.shape[0]
    assert f"$N = {n}$" in tex, f"paper does not state N = {n}"
    assert "1{,}187" not in tex and "1,187" not in tex, \
        "the superseded 1,187-spectrum sample size is still quoted"


def test_grid_is_described_correctly(tex, wave):
    assert f"${{{len(wave):,}}}".replace(",", "{,}") in tex or "6{,}671" in tex, \
        "paper does not state the 6,671-pixel grid"
    assert "2500-pixel" not in tex, "the superseded 2,500-pixel grid is still described"


def test_ml_amplitude_ratio_is_stated(tex, reconstructions, x_high, valid):
    """The amplitude ratio underpins the paper's central methodological argument."""
    r = std_ratio(reconstructions["ML (SR2)"], x_high, valid)
    assert f"\\gls{{sr2}} sits at ${r:.2f}$" in tex, \
        f"paper does not state the ML amplitude ratio {r:.2f}"
    assert 0.45 <= r <= 0.55, f"abstract says about half the reference's amplitude; it is {r:.2f}"


def test_scalefree_spread_claim(tex, reconstructions, x_high, valid):
    # Eight methods: SR2 and seven classical.  SR1 is an intermediate stage of
    # the pipeline and is neither cached nor reported.
    sf = {k: mae_scalefree(v, x_high, valid) for k, v in reconstructions.items()}
    vals = list(sf.values())
    spread_pct = 100 * (max(vals) - min(vals)) / min(vals)
    words = {8: "eight", 7: "seventh"}
    rank = sorted(sf, key=sf.get).index("ML (SR2)") + 1
    assert f"the {words[len(sf)]} methods lie within" in tex, \
        f"paper does not count {len(sf)} methods on the scale-free metric"
    assert f"ranks {words[rank]}, behind cubic interpolation" in tex, \
        f"paper misstates SR2's scale-free rank ({rank} of {len(sf)})"
    assert not re.search(r"Stage \\gls\{sr1\} alone", tex), \
        "the paper reports an SR1 result; only the pipeline's output is reported"
    m = re.search(r"span a total\s*\n?range of ([0-9.]+)\\%", tex) or \
        re.search(r"within \$?([0-9.]+)\\%\$? of one another", tex)
    assert m, "paper does not state the scale-free spread"
    assert float(m.group(1)) == pytest.approx(spread_pct, abs=0.4), \
        f"paper says {m.group(1)}%, computed {spread_pct:.2f}%"


def test_no_superseded_headline_numbers(tex):
    """Numbers from retracted versions of this analysis must not survive."""
    for bad, why in [
        (r"6\.6\\%", "the shrinkage-contaminated 6.6% margin"),
        (r"30\\% lower mean absolute", "the original 30% claim"),
        (r"\$0\.533", "the shrunk Wiener MAE"),
        (r"2\.49\\times", "the pre-rebuild Halpha S/N ratio"),
        (r"25\.8\\%", "the pre-rebuild amplitude recovery"),
    ]:
        assert not re.search(bad, tex), f"paper still contains {why}"


# ── Figure 2: the qualitative example ─────────────────────────────────────────
# This figure is built from a single hardcoded spectrum index, and every number
# in its caption is a property of that one spectrum.  Two cache rebuilds moved
# them all while the prose kept quoting the pre-rebuild values.
#
# The selection rule changed on 2026-08-24.  The figure used to be the
# largest-RMSE-gain spectrum, and the guard was that it really was the argmax.
# It is now chosen on a property of the *data* -- a redshift high enough that
# the [OIII] doublet is wider than the line-spread function -- because the
# previous example sat at z=2.81, where the doublet is half an LSF FWHM across
# and no method can resolve it; the figure credited SR2 for splitting a blend
# that carries no two-peak information.  So the guards are now (a) the plotted
# spectrum's doublet really is resolvable in its own low-resolution input, and
# (b) every number the paper quotes is true of whichever index the figure
# module actually plots, including where its gain ranks.


OIII_4959, OIII_5007 = 0.495891, 0.500824


def _doublet_prominence(flux, wave, z):
    """Prominence of the 4959 peak as a fraction of the window maximum.

    Zero means the two components have merged into a single peak.  A Gaussian
    fitted to a blended doublet looks much like one fitted to a resolved pair,
    which is why this is measured from peak structure and not from a fit.
    """
    from scipy.signal import find_peaks

    c49, c07 = OIII_4959 * (1 + z), OIII_5007 * (1 + z)
    sep = c07 - c49
    m = (wave >= c49 - 0.35 * sep) & (wave <= c07 + 0.35 * sep)
    w, f = wave[m], np.nan_to_num(flux[m])
    if f.size < 8 or f.max() <= 0:
        return 0.0
    pk, props = find_peaks(f, prominence=0.0)
    near = [pr for i, pr in zip(pk, props["prominences"])
            if abs(w[i] - c49) < 0.35 * sep]
    return float(max(near) / f.max()) if near else 0.0


@pytest.fixture(scope="module")
def fig2(cache, x_high, reconstructions, wave):
    """Recompute figure 2's numbers for whichever spectrum the figure plots.

    The index is imported from the figure module rather than scraped out of a
    notebook: it is a named constant now, so the paper's caption and the figure
    cannot describe different galaxies.
    """
    from specsrbench.figures.fig2_qualitative import I_SHOW

    i = int(I_SHOW)

    snr = np.load(cache / "snr.npz", allow_pickle=True)
    z = np.load(cache / "z_test.npy")
    lam5007 = OIII_5007 * (1.0 + z)
    in_grid = (lam5007 > wave.min() + 0.05) & (lam5007 < wave.max() - 0.05)
    subset = (snr["HR_OIII5007"] > 20) & in_grid

    classical_names = [k for k in reconstructions if not k.startswith("ML (")]
    rmse_all = {k: np.sqrt(np.nanmean((v - x_high) ** 2, axis=1))
                for k, v in reconstructions.items()}
    best_classical = np.min([rmse_all[k] for k in classical_names], axis=0)
    gain = 1.0 - rmse_all["ML (SR2)"] / best_classical

    idxs = np.where(subset)[0]
    assert i in idxs, (
        f"figure 2 plots index {i}, which is not in the {len(idxs)}-spectrum "
        "well-detected subset the paper describes it against")

    finite = np.isfinite(x_high[i])
    amp = {k: float(np.nanstd(v[i][finite]) / np.nanstd(x_high[i][finite]))
           for k, v in reconstructions.items()}
    return {
        "n_subset": int(subset.sum()),
        "index": i,
        "z": float(z[i]),
        "lam4959": float(OIII_4959 * (1.0 + z[i])),
        "lam5007": float(lam5007[i]),
        "sep_nm": float((OIII_5007 - OIII_4959) * (1.0 + z[i]) * 1000.0),
        "gain_pct": 100.0 * float(gain[i]),
        "median_gain_pct": 100.0 * float(np.median(gain[idxs])),
        "rank": int((gain[idxs] > gain[i]).sum()) + 1,
        "percentile": 100.0 * float((gain[idxs] < gain[i]).mean()),
        "rmse_sr2": float(rmse_all["ML (SR2)"][i]),
        "rmse_best_classical": float(best_classical[i]),
        "amp_sr2": amp["ML (SR2)"],
        "amp_classical": [amp[k] for k in classical_names],
        "prom_lr": _doublet_prominence(reconstructions["Cubic (LR)"][i], wave, z[i]),
        "prom_hr": _doublet_prominence(x_high[i], wave, z[i]),
        "snr_ha_sr2": float(snr["SR2_Halpha"][i]),
        "snr_ha_hr": float(snr["HR_Halpha"][i]),
        "snr_oiii_sr2": float(snr["SR2_OIII5007"][i]),
        "snr_oiii_hr": float(snr["HR_OIII5007"][i]),
    }


def test_fig2_doublet_is_actually_resolvable(fig2):
    """The whole point of the new selection rule.

    If the doublet is not separated in the low-resolution input, the figure is
    comparing methods on structure none of them can legitimately recover, and
    any method that "resolves" it is hallucinating.
    """
    assert fig2["prom_hr"] > 0.08, (
        f"the HR reference does not resolve the doublet at z={fig2['z']:.2f} "
        f"(4959 prominence {fig2['prom_hr']:.3f})")
    assert fig2["prom_lr"] > 0.08, (
        f"index {fig2['index']} (z={fig2['z']:.2f}, separation "
        f"{fig2['sep_nm']:.1f} nm) has an unresolvable doublet in the LR input "
        f"(4959 prominence {fig2['prom_lr']:.3f}); pick a higher-redshift "
        "spectrum or the figure credits SR2 for inventing the split")


def test_fig2_is_not_described_as_a_best_case(tex, fig2):
    """It is no longer the argmax, so it must not be called one."""
    assert not re.search(r"median-\\gls\{mae\} case", tex), \
        "paper still describes figure 2 as the median case"
    if fig2["rank"] > 1:
        assert "best case" not in tex, (
            f"paper calls figure 2 a best case, but its gain ranks "
            f"{fig2['rank']} of {fig2['n_subset']}")


def _ordinal(n):
    """English ordinal suffix, so the test does not force "33th" into the prose."""
    if 11 <= n % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def test_fig2_redshift(tex, fig2):
    # The "89th percentile of the 296" sentence left the prose on 2026-10-07;
    # test_fig2_gain_matches still checks the example is favorable, not typical.
    assert f"$z = {fig2['z']:.2f}$" in tex, \
        f"paper does not state the figure 2 redshift z = {fig2['z']:.2f}"


def test_fig2_gain_matches(tex, fig2):
    assert f"${fig2['gain_pct']:.0f}\\%$" in tex, \
        f"paper does not state figure 2's {fig2['gain_pct']:.0f}% RMSE gain"
    assert fig2["gain_pct"] > fig2["median_gain_pct"], \
        "figure 2 is no longer favorable to SR2, which the paper says it is"


def test_fig2_caption_amplitude_matches_the_cache(tex, fig2):
    assert f"${fig2['amp_sr2']:.2f}$" in tex, \
        f"paper does not state figure 2's SR2 amplitude ratio {fig2['amp_sr2']:.2f}"
    lo, hi = min(fig2["amp_classical"]), max(fig2["amp_classical"])
    assert f"${lo:.2f}$--${hi:.2f}$" in tex, \
        f"paper does not state figure 2's classical amplitude range {lo:.2f}-{hi:.2f}"


def test_fig2_superseded_numbers_are_gone(tex):
    for bad, why in [
        (r"\$z = 2\.81\$", "the z=2.81 example's redshift"),
        (r"1\.91\\,\\mu\$m", "the z=2.81 example's inset wavelength"),
        (r"\$0\.365\$", "the index-216 SR2 RMSE"),
        (r"\$0\.834\$", "the index-216 best-classical RMSE"),
        (r"\$521\$", "the index-216 SR2 Halpha S/N"),
        (r"\$231\$", "the index-216 HR Halpha S/N"),
        (r"\$285\$", "the pre-rebuild well-detected subset size"),
        (r"\$1\.148\$", "the index-99 SR2 RMSE"),
        (r"\$1\.131\$", "the index-99 best-classical RMSE"),
    ]:
        assert not re.search(bad, tex), f"paper still quotes {why}"


# ── Bibliography ──────────────────────────────────────────────────────────────
# The 2026-07-30 citation audit left references.bib and the cite keys matching
# exactly, but nothing enforced it: a methods section removed on 2026-09-07
# took its citations with it and left six entries behind, never checked
# against any record.  Either direction is a defect -- an orphan entry is
# unverified material waiting to be cited, a missing one an undefined citation.
BIB = REPO / "paper" / "references.bib"


def test_bibliography_matches_the_citations(tex):
    if not BIB.exists():
        pytest.skip("references.bib not present")
    body = re.sub(r"(?<!\\)%.*", "", tex)
    cited = {k.strip() for group in re.findall(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", body)
             for k in group.split(",")}
    defined = set(re.findall(r"^@[A-Za-z]+\{([^,\s]+),", BIB.read_text(encoding="utf-8"), re.M))
    assert not cited - defined, f"cited but not in references.bib: {sorted(cited - defined)}"
    assert not defined - cited, f"in references.bib but never cited: {sorted(defined - cited)}"


# ── the tuned parameters, as the paper states them ────────────────────────────
# classical_params.json and the cache builder once held separate copies of
# these values and diverged for a day, making every classical number in the
# paper wrong while the suite stayed green.  The manuscript is a third copy,
# and until now nothing checked it against the other two: it described the
# matched filter's fitting window as 3 LSF widths where the tuned value is 4.
PARAMS = REPO / "cache_logR_tuned" / "classical_params.json"


@pytest.fixture(scope="module")
def tuned():
    if not PARAMS.exists():
        pytest.skip("classical_params.json not present")
    import json
    return json.loads(PARAMS.read_text())["tuned"]


def test_paper_states_the_tuned_parameters(tex, tuned):
    flat = " ".join(tex.split())
    checks = [
        (rf"\$\\mathrm{{SNR}} = {tuned['wiener']['snr']:.0f}\$", "Wiener SNR"),
        (rf"\$\\lambda = {tuned['tikhonov']['lam']:.0f}\$", "Tikhonov lambda"),
        (rf"\$\\lambda = {tuned['tv']['lam']}\$", "TV lambda"),
        (rf"\${tuned['tv']['n_iter']}\$~iterations", "TV iterations"),
        (rf"\$\\lambda = {tuned['sparse']['lam']}\$", "FISTA lambda"),
        (rf"\$\\geq {tuned['mf']['detect_snr']:.0f}\\sigma\$", "MF detection threshold"),
    ]
    for pattern, what in checks:
        assert re.search(pattern, flat), \
            f"paper does not state the tuned value for {what} ({pattern})"
    # Both single-iteration methods must be described as such.
    assert tuned["rl"]["n_iter"] == 1 and tuned["sparse"]["n_iter"] == 1, \
        "Richardson-Lucy or FISTA is no longer tuned to a single iteration"
    assert flat.count("a single iteration") >= 2, \
        "paper no longer says both iterative methods run for a single iteration"
