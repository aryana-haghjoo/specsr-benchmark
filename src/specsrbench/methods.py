"""The nine reconstructions, their display names, and the keys they are cached under.

Three different naming schemes meet in this project and none of them can be
retired without invalidating a shipped cache:

* ``snr.npz`` is keyed by a short name -- ``LR``, ``RL``, ``MF``, ``SR2``.
* ``fit_params_cache.npz`` is keyed by the *display* name -- ``Cubic (LR)``,
  ``R-L``, ``Wiener + MF``, ``ML (SR2)``.
* The arrays themselves are separate ``.npy`` files named after the method.

Each notebook carried its own copy of all three maps, and they had drifted: three
of the six mapped ``ML (SR2)`` to the S/N key ``"ML (SR2)"``, which does not
exist in ``snr.npz``.  It never raised, because those three defined the map and
then did not use it.  One table here, with the cache's own key names checked by
``tests/test_cache_integrity.py``, is what replaces that.

Display *labels* still vary by figure -- figure 2 writes ``LR (cubic)`` and
``Wiener + TV`` where figure 3 writes ``Cubic (LR)`` and ``TV`` -- and figure 3
recolours two methods to keep them legible against its diverging colour map.
Those variations are real and are preserved by :func:`registry` overrides rather
than by nine more copies of the table.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["Method", "METHODS", "registry", "labels", "colors", "LINES",
           "line_keys", "ORDER", "NON_HR"]


@dataclass(frozen=True)
class Method:
    """One reconstruction: how to find it, what to call it, how to draw it."""

    key: str          #: canonical short name, and the ``snr.npz`` key prefix
    label: str        #: display name, and the ``fit_params_cache.npz`` prefix
    array: str | None  #: file in the cache holding it, if it has one of its own
    color: str

    @property
    def snr_key(self) -> str:
        return self.key


#: Fixed display order, and the order every bar chart and panel grid uses:
#: baseline -> linear -> non-linear/regularised -> line-list-aware -> learned,
#: with the reference last.
METHODS: tuple[Method, ...] = (
    Method("LR",       "Cubic (LR)",  "x_low.npy",         "deeppink"),
    Method("Wiener",   "Wiener",      "wiener_cache.npy",  "tomato"),
    Method("Tikhonov", "Tikhonov",    "tikhonov_cache.npy", "forestgreen"),
    Method("TV",       "TV",          "tv_cache.npy",      "teal"),
    Method("RL",       "R-L",         "rl_cache.npy",      "steelblue"),
    Method("Sparse",   "Sparse",      "sparse_cache.npy",  "goldenrod"),
    Method("MF",       "Wiener + MF", "mf_cache.npy",      "darkorchid"),
    Method("SR2",      "ML (SR2)",    None,                "darkorange"),
    Method("HR",       "HR target",   "x_high.npy",        "black"),
)

ORDER: tuple[str, ...] = tuple(m.key for m in METHODS)
NON_HR: tuple[str, ...] = tuple(k for k in ORDER if k != "HR")

#: ``(cache key, display label, rest wavelength in um)`` for the four lines the
#: paper measures.  The order is the one every per-line panel row uses.
LINES: tuple[tuple[str, str, float], ...] = (
    ("Halpha",   r"H$\alpha$",             0.6563),
    ("OIII5007", r"[O III] $\lambda$5007", 0.5007),
    ("Hbeta",    r"H$\beta$",              0.4861),
    ("OII3727",  r"[O II] $\lambda$3727",  0.3727),
)


def line_keys() -> list[str]:
    return [k for k, _, _ in LINES]


def registry(label_overrides: dict[str, str] | None = None,
             color_overrides: dict[str, str] | None = None,
             include_hr: bool = True) -> dict[str, Method]:
    """The method table, keyed by canonical name, with per-figure overrides.

    Overrides are given by canonical key, so a figure that renames ``LR`` to
    ``LR (cubic)`` cannot accidentally rename it to something the cache has
    never heard of -- the key it looks arrays up by does not change.
    """
    out: dict[str, Method] = {}
    for m in METHODS:
        if not include_hr and m.key == "HR":
            continue
        if label_overrides and m.key in label_overrides:
            m = replace(m, label=label_overrides[m.key])
        if color_overrides and m.key in color_overrides:
            m = replace(m, color=color_overrides[m.key])
        out[m.key] = m
    return out


def labels(reg: dict[str, Method]) -> list[str]:
    return [reg[k].label for k in reg]


def colors(reg: dict[str, Method]) -> list[str]:
    return [reg[k].color for k in reg]
