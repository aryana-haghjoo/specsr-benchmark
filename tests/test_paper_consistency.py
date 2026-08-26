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

PAPER = REPO / "paper.tex"


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


# LaTeX label -> summary_final.csv label
TABLE_ROWS = {
    r"Wiener\,+\,\gls{mf}": "Wiener + MF",
    r"Wiener\,+\,\gls{tv}": "TV",
    "Wiener": "Wiener",
    "Cubic (LR)": "Cubic (LR)",
    "Tikhonov": "Tikhonov",
    "Richardson--Lucy": "R-L",
    r"Sparse (\gls{fista})": "Sparse",
    r"\gls{ml} (\gls{sr2})": "ML (SR2)",
}


def _table_body(tex):
    m = re.search(r"\\startdata(.*?)\\enddata", tex, re.S)
    assert m, "the global-fidelity deluxetable is missing from paper.tex"
    return m.group(1)


def test_global_table_matches_the_cache(tex, summary):
    """Every row of the global-fidelity table, against a fresh computation."""
    body = _table_body(tex)
    seen = 0
    for line in body.splitlines():
        line = line.strip().rstrip("\\").strip()
        if not line or line.startswith("%"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 4:
            continue
        label, mae_s, amp_s, sf_s = cells
        key = TABLE_ROWS.get(label)
        if key is None or key not in summary:
            continue
        seen += 1
        row = summary[key]
        assert float(mae_s) == pytest.approx(float(row["MAE"]), abs=0.002), \
            f"{label}: paper says MAE {mae_s}, cache says {row['MAE']}"
        assert float(amp_s) == pytest.approx(float(row["std_ratio"]), abs=0.02), \
            f"{label}: paper says amplitude {amp_s}, cache says {row['std_ratio']}"
        assert float(sf_s) == pytest.approx(float(row["MAE_scalefree"]), abs=0.002), \
            f"{label}: paper says scale-free {sf_s}, cache says {row['MAE_scalefree']}"
    assert seen >= 7, f"only matched {seen} table rows; the parser or table changed"


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
    """The 1.84x figure underpins the paper's central methodological argument."""
    r = std_ratio(reconstructions["ML (SR2)"], x_high, valid)
    stated = re.findall(r"([0-9]\.[0-9]{2})\\times\$ rescale|multiplied by ([0-9]\.[0-9]{2})", tex)
    flat = {float(a or b) for a, b in stated}
    assert flat, "paper never states the ML rescale factor"
    assert any(abs(v - 1.0 / r) < 0.03 for v in flat), \
        f"paper states {flat}, cache implies {1.0 / r:.2f}"


def test_scalefree_spread_claim(tex, reconstructions, x_high, valid):
    vals = [mae_scalefree(v, x_high, valid) for v in reconstructions.values()]
    spread_pct = 100 * (max(vals) - min(vals)) / min(vals)
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
                for k, v in reconstructions.items() if not k.startswith("ML (SR1)")}
    best_classical = np.min([rmse_all[k] for k in classical_names], axis=0)
    gain = 1.0 - rmse_all["ML (SR2)"] / best_classical

    idxs = np.where(subset)[0]
    assert i in idxs, (
        f"figure 2 plots index {i}, which is not in the {len(idxs)}-spectrum "
        "well-detected subset the paper describes it against")

    finite = np.isfinite(x_high[i])
    amp = {k: float(np.nanstd(v[i][finite]) / np.nanstd(x_high[i][finite]))
           for k, v in reconstructions.items() if not k.startswith("ML (SR1)")}
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


def test_fig2_rank_and_percentile_match(tex, fig2):
    rank = fig2["rank"]
    assert f"${rank}${_ordinal(rank)} of ${fig2['n_subset']}$" in tex, (
        f"paper does not state figure 2's rank of {rank}{_ordinal(rank)} of "
        f"{fig2['n_subset']}")
    assert f"${fig2['percentile']:.0f}$th percentile" in tex, \
        f"paper does not state figure 2's {fig2['percentile']:.0f}th percentile"


def test_fig2_subset_size_and_redshift(tex, fig2):
    assert f"${fig2['n_subset']}$" in tex, \
        f"paper does not state the {fig2['n_subset']}-spectrum well-detected subset"
    assert f"$z = {fig2['z']:.2f}$" in tex, \
        f"paper does not state the figure 2 redshift z = {fig2['z']:.2f}"


def test_fig2_gain_and_subset_median_match(tex, fig2):
    assert f"${fig2['gain_pct']:.0f}\\%$" in tex, \
        f"paper does not state figure 2's {fig2['gain_pct']:.0f}% RMSE gain"
    assert f"${fig2['median_gain_pct']:.0f}\\%$" in tex, \
        f"paper does not state the subset median gain of {fig2['median_gain_pct']:.0f}%"


def test_fig2_inset_wavelengths(tex, fig2):
    """The inset is centred on this example's doublet, not a stale one's."""
    m = re.search(r"doublet at \$([0-9.]+)\$ and \$([0-9.]+)\\,\\mu\$m", tex)
    assert m, "paper does not state the inset wavelengths"
    assert float(m.group(1)) == pytest.approx(fig2["lam4959"], abs=0.01), \
        f"paper says 4959 at {m.group(1)} um, cache implies {fig2['lam4959']:.3f}"
    assert float(m.group(2)) == pytest.approx(fig2["lam5007"], abs=0.01), \
        f"paper says 5007 at {m.group(2)} um, cache implies {fig2['lam5007']:.3f}"


def test_fig2_doublet_separation_matches(tex, fig2):
    assert f"${fig2['sep_nm']:.1f}\\,$nm" in tex, \
        f"paper does not state the {fig2['sep_nm']:.1f} nm doublet separation"


def test_fig2_caption_rmse_matches_the_cache(tex, fig2):
    for label, val in [("SR2", fig2["rmse_sr2"]),
                       ("best classical", fig2["rmse_best_classical"])]:
        assert f"${val:.3f}$" in tex, \
            f"paper does not state the figure 2 {label} RMSE of {val:.3f}"


def test_fig2_caption_amplitude_matches_the_cache(tex, fig2):
    assert f"${fig2['amp_sr2']:.2f}$" in tex, \
        f"paper does not state figure 2's SR2 amplitude ratio {fig2['amp_sr2']:.2f}"
    lo, hi = min(fig2["amp_classical"]), max(fig2["amp_classical"])
    assert f"${lo:.2f}$--${hi:.2f}$" in tex, \
        f"paper does not state figure 2's classical amplitude range {lo:.2f}-{hi:.2f}"


def test_fig2_line_snr_matches_the_cache(tex, fig2):
    """The S/N inflation is the honest caveat on a favourable figure."""
    for val in (fig2["snr_ha_sr2"], fig2["snr_ha_hr"],
                fig2["snr_oiii_sr2"], fig2["snr_oiii_hr"]):
        assert f"${val:.0f}$" in tex, \
            f"paper does not state figure 2's fitted S/N of {val:.0f}"


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
