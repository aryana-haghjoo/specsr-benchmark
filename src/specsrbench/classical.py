"""Classical deconvolution methods, parameterised for the log constant-R grid.

The algorithms are carried over unchanged from the original linear-grid build.
What changes here is that the grid-dependent constants are arguments rather
than literals, because every one of them was sized for the old 2,500-point
linear grid where the LSF was ~3 pixels wide.  On the specsr ``DEFAULT_GRID``
(log, R=4000, 6,671 points) the same LSF spans ~25 pixels, so the published
defaults are far off.

Two fixes are structural rather than a matter of tuning:

* ``matched_filter`` took a fixed ``window_half_um``.  At 0.015 um that window
  is 1.11 sigma wide on this grid -- narrower than the line being fitted, so
  the continuum sidebands land inside the line.  The window is now set as a
  multiple of the local LSF width.
* ``wiener_deconv`` and ``tikhonov_deconv`` segment the spectrum before taking
  an FFT.  A 128-sample segment holds only 3 usable Fourier modes at
  sigma=25 px (it held 21 at sigma=3 px), so ``segment_len`` is exposed too.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pywt
from scipy.linalg import solve_banded
from scipy.signal import fftconvolve


# ── shared ────────────────────────────────────────────────────────────────────
def load_sigma_pix(out_dir=None, eval_npz=None):
    """The LSF every classical method deconvolves with, and where it came from.

    ``eval_set.npz`` ships a ``sigma_pix`` that nothing in either repo
    generates and that does not describe the blur in the data: it is roughly
    constant in *nanometres*, whereas a spectrograph's LSF is fixed in detector
    *pixels*.  Measured against the paired data it is up to 2.3x too broad at
    5 um, enough to make Wiener, Tikhonov and TV merge line pairs their own
    input still resolves.  :mod:`specsrbench.build.lsf` measures the real one.

    Every caller that deconvolves reads the kernel through this function, so
    the tuner, the cache build and the line fits cannot disagree about it --
    which is exactly how the shipped caches came to be built with one kernel
    while the parameters had been tuned against another.

    Returns ``(sigma_pix, provenance)``.  The provenance string starts with
    ``SHIPPED`` when the derived kernel is absent, and callers that write a
    cache refuse to proceed on that rather than quietly producing wrong numbers.
    """
    from . import paths

    out = Path(out_dir) if out_dir is not None else paths.cache_dir()
    derived = out / "sigma_pix_measured.npy"
    if derived.exists():
        return np.load(derived).astype(np.float64), "derived (specsrbench build lsf)"
    if eval_npz is None:
        eval_path = paths.sets_dir() / "eval_set.npz"
        if not eval_path.exists():
            raise FileNotFoundError(
                f"no kernel: neither {derived} nor {eval_path} exists.\n"
                "  run: specsrbench build lsf --jades-root <JADES DR4 tree>")
        eval_npz = np.load(eval_path, allow_pickle=True)
    if "sigma_pix" not in getattr(eval_npz, "files", ()):
        raise FileNotFoundError(
            f"no kernel: {derived} does not exist, and the evaluation set does\n"
            "  not carry one (correctly -- the array it used to ship was not\n"
            "  produced by anything and did not describe the data).  Measure it:\n"
            "      specsrbench build lsf --jades-root <JADES DR4 tree>")
    return (np.asarray(eval_npz["sigma_pix"], dtype=np.float64),
            "SHIPPED -- run `specsrbench build lsf` first; results will be wrong")


def make_kernel(sig, n_sigma=5):
    hw = max(int(np.ceil(n_sigma * sig)), 1)
    x = np.arange(-hw, hw + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sig) ** 2)
    return k / k.sum()


def _segment_starts(L, segment_len, overlap):
    step = max(segment_len - 2 * overlap, 1)
    starts = list(range(0, L - segment_len + 1, step))
    if not starts:
        return [0]
    if starts[-1] + segment_len < L:
        starts.append(L - segment_len)
    return starts


# ── 4a. Wiener ────────────────────────────────────────────────────────────────
def wiener_deconv(spec, sigma_pix_arr, snr=10.0, segment_len=128, overlap=32,
                  unit_gain=True):
    """Wiener deconvolution, normalised to unit gain at zero frequency.

    The raw MMSE filter W = H/(H^2 + 1/snr) has DC gain 1/(1 + 1/snr), so it
    rescales the whole spectrum: 0.91 at snr=10, 0.44 at snr=0.8.  Tuning it
    against MAE on z-scored spectra therefore drives snr toward zero, because
    shrinking a noisy estimate toward the mean lowers absolute error regardless
    of whether any deconvolution occurred.  That is shrinkage, not
    deconvolution, and it is invisible to any scale-free diagnostic: line S/N
    is amplitude over sideband noise, so a global rescale leaves it unchanged.

    Dividing by W(0) keeps the filter's shape -- which is what distinguishes
    Wiener from the other methods -- while removing its ability to buy MAE with
    a global rescale.  Tikhonov needs no such correction: its W(0) is already
    1, which is why it was the one linear filter that did not degenerate.
    """
    L = len(spec)
    if segment_len > L:
        segment_len, overlap = L, L // 4
    out = np.zeros(L)
    weight = np.zeros(L)
    win = np.hanning(segment_len)
    freqs = np.fft.rfftfreq(segment_len)
    for s in _segment_starts(L, segment_len, overlap):
        e = s + segment_len
        sig_loc = float(np.median(sigma_pix_arr[s:e]))
        H = np.exp(-2.0 * (np.pi * freqs * sig_loc) ** 2)
        W = H / (H**2 + 1.0 / max(snr, 1e-12))
        if unit_gain and W[0] > 0:
            W = W / W[0]
        out[s:e] += np.fft.irfft(np.fft.rfft(spec[s:e]) * W, n=segment_len) * win
        weight[s:e] += win
    return out / np.where(weight > 0, weight, 1.0)


# ── 4b. Richardson-Lucy ───────────────────────────────────────────────────────
def rl_deconv(spec, sigma_pix_arr, n_iter=30, eps=1e-6, n_seg=8):
    L = len(spec)
    floor = spec.min() - eps
    y = np.clip(spec - floor, eps, None)
    est = y.copy()
    edges = np.linspace(0, L, n_seg + 1, dtype=int)
    kernels = [make_kernel(float(np.median(sigma_pix_arr[edges[s]:edges[s + 1]])))
               for s in range(n_seg)]

    def conv_sv(arr):
        out = np.zeros_like(arr)
        for s in range(n_seg):
            sl = slice(edges[s], edges[s + 1])
            out[sl] = fftconvolve(arr, kernels[s], mode="same")[sl]
        return np.clip(out, eps, None)

    for _ in range(n_iter):
        est = np.clip(est * conv_sv(y / conv_sv(est)), eps, None)
    return est + floor


# ── 4c. Matched filter ────────────────────────────────────────────────────────
def mad_sigma(y):
    y = y[np.isfinite(y)]
    return 1.4826 * np.median(np.abs(y - np.median(y))) if len(y) >= 8 else np.nan


def matched_filter(spec, wl, z, line_rest_um, sigma_pix_arr,
                   window_nsigma=4.0, detect_snr=3.0,
                   core_nsigma=3.0, sideband_nsigma=2.0, width_scale=0.07):
    """Redshift-informed matched filter.

    ``window_nsigma`` replaces the old fixed ``window_half_um``: the fitting
    window is set to this many local LSF sigmas, so it scales with the grid
    instead of being a constant in microns.

    ``width_scale`` sets the template width as a fraction of the local LSF
    width, and is the fix for a bug the log grid exposed.  The template used to
    be the prism LSF itself, ~13.5 nm here, while a real line in the
    high-resolution reference has sigma ~0.96 nm -- 14x narrower.  Writing a
    template that wide back over the line core injects flux across a region an
    order of magnitude broader than the line, inflating the reconstruction's
    amplitude by a factor of 1.4-2.2.  The old 2,500-point grid hid this: its
    downsampled reference had broader lines and its LSF was narrower in
    microns, leaving only a ~3x mismatch.

    The window, sidebands and continuum fit stay anchored to the LSF width,
    which is the scale over which local continuum must be estimated; only the
    template itself is narrowed.
    """
    dpix_um = np.gradient(wl)
    candidates = []
    for lam0_rest in line_rest_um:
        center = lam0_rest * (1.0 + z)
        pix = int(np.argmin(np.abs(wl - center)))
        sig_lsf = float(sigma_pix_arr[pix]) * float(dpix_um[pix])
        sig_um = max(width_scale * sig_lsf, float(dpix_um[pix]))
        half = window_nsigma * sig_lsf
        if center < wl.min() + half or center > wl.max() - half:
            continue
        candidates.append((center, sig_um, half))

    if not candidates:
        return spec.copy()

    candidates.sort(key=lambda item: item[0])
    groups, current, current_hi = [], [], -np.inf
    for center, sig_um, half in candidates:
        if current and (center - half) > current_hi:
            groups.append(current)
            current = []
        current.append((center, sig_um, half))
        current_hi = max(current_hi, center + half)
    if current:
        groups.append(current)

    out = spec.copy()
    for group in groups:
        lo = min(c - h for c, _, h in group)
        hi = max(c + h for c, _, h in group)
        mask = (wl >= lo) & (wl <= hi)
        if mask.sum() < 5:
            continue
        ww, yy = wl[mask], spec[mask]

        dist = np.vstack([np.abs(ww - c) / max(sig, 1e-12) for c, sig, _ in group])
        sb = np.min(dist, axis=0) > sideband_nsigma
        if sb.sum() < 4:
            continue
        cont = np.polyval(np.polyfit(ww[sb], yy[sb], 1), ww)
        noise = mad_sigma(yy[sb])
        if not np.isfinite(noise) or noise <= 0:
            continue

        resid = yy - cont
        templates = np.column_stack([
            np.exp(-0.5 * ((ww - c) / max(sig, 1e-12)) ** 2) for c, sig, _ in group
        ])
        try:
            amps, *_ = np.linalg.lstsq(templates, resid, rcond=None)
        except np.linalg.LinAlgError:
            continue

        gram_inv = np.linalg.pinv(templates.T @ templates)
        amp_err = noise * np.sqrt(np.clip(np.diag(gram_inv), 0.0, None))
        amp_snr = np.divide(amps, amp_err, out=np.zeros_like(amps), where=amp_err > 0)
        keep = (amps > 0.0) & (amp_snr >= detect_snr)
        if not np.any(keep):
            continue

        model = templates[:, keep] @ amps[keep]
        core = np.min(dist[keep], axis=0) <= core_nsigma
        idx = np.flatnonzero(mask)[core]
        out[idx] = cont[core] + model[core]
    return out


# ── 4d. Tikhonov ──────────────────────────────────────────────────────────────
def tikhonov_deconv(spec, sigma_pix_arr, lam=0.1, segment_len=128, overlap=32):
    L = len(spec)
    if segment_len > L:
        segment_len, overlap = L, L // 4
    out = np.zeros(L)
    weight = np.zeros(L)
    win = np.hanning(segment_len)
    freqs = np.fft.rfftfreq(segment_len)
    L_sq = (2.0 * np.pi * freqs) ** 2
    for s in _segment_starts(L, segment_len, overlap):
        e = s + segment_len
        sig_loc = float(np.median(sigma_pix_arr[s:e]))
        H = np.exp(-2.0 * (np.pi * freqs * sig_loc) ** 2)
        W = H / (H**2 + lam * L_sq + 1e-30)
        out[s:e] += np.fft.irfft(np.fft.rfft(spec[s:e]) * W, n=segment_len) * win
        weight[s:e] += win
    return out / np.where(weight > 0, weight, 1.0)


# ── 4e. TV (split Bregman) ────────────────────────────────────────────────────
def _Dt_op(v, n):
    result = np.zeros(n)
    result[0] = -v[0]
    result[1:n - 1] = v[0:n - 2] - v[1:n - 1]
    result[n - 1] = v[n - 2]
    return result


def _solve_tv_x(rhs, mu, n):
    ab = np.zeros((3, n))
    ab[1, 0] = 1.0 + mu
    ab[1, 1:-1] = 1.0 + 2.0 * mu
    ab[1, -1] = 1.0 + mu
    ab[0, 1:] = -mu
    ab[2, :-1] = -mu
    return solve_banded((1, 1), ab, rhs)


def tv_denoise_1d(y, lam, mu=None, n_iter=30):
    if mu is None:
        mu = 2.0 * lam
    n = len(y)
    x = y.copy()
    d = np.zeros(n - 1)
    b = np.zeros(n - 1)
    for _ in range(n_iter):
        rhs = y + mu * _Dt_op(d - b, n)
        x = _solve_tv_x(rhs, mu, n)
        Dx = x[1:] - x[:-1]
        s = Dx + b
        d = np.sign(s) * np.maximum(np.abs(s) - lam / mu, 0.0)
        b = b + Dx - d
    return x


# ── 4f. Wavelet-sparse (FISTA) ────────────────────────────────────────────────
def _prox_wavelet_l1(x, threshold, wavelet="db4", n_levels=4):
    n = len(x)
    coeffs = pywt.wavedec(x, wavelet, level=n_levels, mode="periodization")
    new_coeffs = [coeffs[0]]
    for cD in coeffs[1:]:
        new_coeffs.append(np.sign(cD) * np.maximum(np.abs(cD) - threshold, 0.0))
    return pywt.waverec(new_coeffs, wavelet, mode="periodization")[:n]


def sparse_wavelet_deconv(spec, sigma_pix_arr, wavelet="db4", n_levels=4,
                          lam=0.05, n_iter=150, step_size=0.9, n_seg=8):
    L_spec = len(spec)
    y = spec.copy()
    edges = np.linspace(0, L_spec, n_seg + 1, dtype=int)
    kernels = [make_kernel(float(np.median(sigma_pix_arr[edges[s]:edges[s + 1]])))
               for s in range(n_seg)]

    def forward(x):
        out = np.zeros_like(x)
        for s in range(n_seg):
            sl = slice(edges[s], edges[s + 1])
            out[sl] = fftconvolve(x, kernels[s], mode="same")[sl]
        return out

    threshold = lam * step_size
    x = y.copy()
    u = x.copy()
    u_new = u
    t = 1.0
    for _ in range(n_iter):
        grad = forward(forward(x) - y)
        u_new = _prox_wavelet_l1(x - step_size * grad, threshold, wavelet, n_levels)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        x = u_new + ((t - 1.0) / t_new) * (u_new - u)
        u = u_new
        t = t_new
    return u_new
