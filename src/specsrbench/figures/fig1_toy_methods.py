"""Figure 1 -- the eight methods on a 1D toy, where the truth is known exactly.

A close doublet the LSF blends into one blob, plus a weak isolated line, at
sigma_LSF = 8 px and a peak S/N of ~12.  Every method in the paper is run on
it, scored against the truth it was generated from, and drawn.

The toy exists because on real spectra there is no ground truth: the grating
reference is itself noisy and band-limited, so "recovered the line" and
"invented a line that happens to sit there" are hard to separate.  Here they
are not.  The three numbers under each panel -- peak amplitude, the dip between
the blended pair, and fitted FWHM -- are the same quantities Figure 5 measures
on real data, against a truth that is known rather than estimated.

The parameter values in the panel titles are the classical literature defaults
at this scale, *not* the tuned values the benchmark uses.  Every classical
parameter in this project is grid-dependent, and the settings tuned for the
R = 4000 log grid are meaningless at 512 pixels.  A toy this size is also far
easier to deconvolve than a real spectrum: read the panels as an illustration
of what each method does, never as a performance claim.

Reproducibility
---------------
Seven of the eight panels are deterministic.  The eighth trains a small 1D CNN
inline, and cuDNN's convolutions reduce in a non-deterministic order on GPU, so
its row moves in the third or fourth decimal between runs (RMSE 0.0143 vs
0.0144 when this was checked).  The conclusion does not move -- the CNN wins on
every metric by a wide margin -- but this is the one figure in the paper that
is not bit-reproducible.  Pass ``deterministic=True`` to trade a little speed
for a fixed result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from .. import paths, style

__all__ = ["build", "make_toy", "METHOD_COLORS"]

N = 512
#: ``(centre, sigma, amplitude)`` of the three true lines.  The first two are
#: 50 px apart against a 8 px LSF -- blended in the observation, separable in
#: principle.
TRUE_PARAMS = ((180, 4.0, 1.0), (230, 4.0, 0.7), (360, 6.0, 0.4))
SIGMA_LSF = 8.0
NOISE_SIGMA = 0.05
OBS_SEED = 42

#: Training of the toy CNN.  Fresh batches each iteration, so there is no
#: training set to overfit and the comparison is against the same generator the
#: toy itself is drawn from.
N_ITERS = 2000
BATCH_SIZE = 64
TORCH_SEED = 0

METHOD_COLORS = {
    "Noisy observed":          "deeppink",
    "Wiener (SNR=10)":         "tomato",
    "Tikhonov (λ=0.01)":       "forestgreen",
    "TV (λ=0.05)":             "teal",
    "Richardson-Lucy (30 it)": "steelblue",
    "Wavelet Sparse":          "goldenrod",
    "Matched filter":          "darkorchid",
    "SR1-CNN (toy-trained)":   "darkorange",
}


def gaussian(x, mu, sig, amp):
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def make_toy(seed: int = OBS_SEED):
    """``(x, truth, lsf, blurred, observed)`` for the toy problem."""
    rng = np.random.default_rng(seed)
    x = np.arange(N)
    true = sum(gaussian(x, mu, s, a) for mu, s, a in TRUE_PARAMS)
    halfwin = int(np.ceil(5 * SIGMA_LSF))
    lsf_x = np.arange(-halfwin, halfwin + 1)
    lsf = np.exp(-0.5 * (lsf_x / SIGMA_LSF) ** 2)
    lsf /= lsf.sum()
    blurred = fftconvolve(true, lsf, mode="same")
    observed = blurred + rng.normal(0, NOISE_SIGMA, size=N)
    return x, true, lsf, blurred, observed


# ── the deconvolvers, at toy scale ────────────────────────────────────────────
# Deliberately separate from specsrbench.classical, which is parameterised for
# the 6,671-point log grid where the LSF spans ~25 px.  These are the textbook
# forms at 512 px; sharing code between the two would mean one of them being
# expressed in the other's units.
def _zero_phase_kernel(lsf, n):
    """Pad the LSF to length n with its peak at index 0, for the FFT."""
    k = len(lsf)
    pad = np.zeros(n)
    pad[:k] = lsf
    return np.roll(pad, -(k // 2))


def m_wiener(y, lsf, snr=10.0):
    n = len(y)
    H = np.fft.rfft(_zero_phase_kernel(lsf, n))
    W = np.conj(H) / (np.abs(H) ** 2 + 1.0 / snr)
    return np.fft.irfft(np.fft.rfft(y) * W, n=n)


def m_tikhonov(y, lsf, lam=0.01):
    """Wiener with a first-derivative penalty in place of a flat noise floor."""
    n = len(y)
    H = np.fft.rfft(_zero_phase_kernel(lsf, n))
    L_sq = (2.0 * np.pi * np.fft.rfftfreq(n)) ** 2
    W = np.conj(H) / (np.abs(H) ** 2 + lam * L_sq + 1e-30)
    return np.fft.irfft(np.fft.rfft(y) * W, n=n)


def m_tv(y, lsf, lam=0.05, step=0.5, n_iter=200):
    """min_x ||Hx - y||^2 + lam TV(x), by ISTA with a Chambolle prox."""
    from skimage.restoration import denoise_tv_chambolle
    h_flip = lsf[::-1]
    x = y.copy()
    for _ in range(n_iter):
        grad = fftconvolve(fftconvolve(x, lsf, mode="same") - y, h_flip, mode="same")
        x = denoise_tv_chambolle(x - step * grad, weight=lam * step)
    return x


def m_richardson_lucy(y, lsf, n_iter=30, eps=1e-6):
    """Multiplicative, non-negative.  The offset makes the noisy data positive."""
    floor = float(y.min()) - eps
    yp = np.clip(y - floor, eps, None)
    h_flip = lsf[::-1]
    est = yp.copy()
    for _ in range(n_iter):
        denom = np.clip(fftconvolve(est, lsf, mode="same"), eps, None)
        est = np.clip(est * fftconvolve(yp / denom, h_flip, mode="same"), eps, None)
    return est + floor


def _prox_wavelet_l1(x, threshold, wavelet="db4", n_levels=4):
    import pywt
    n = len(x)
    coeffs = pywt.wavedec(x, wavelet, level=n_levels, mode="periodization")
    new = [coeffs[0]] + [np.sign(c) * np.maximum(np.abs(c) - threshold, 0.0)
                         for c in coeffs[1:]]
    return pywt.waverec(new, wavelet, mode="periodization")[:n]


def m_sparse_wavelet(y, lsf, wavelet="db4", n_levels=4, lam=0.05,
                     n_iter=150, step_size=0.9):
    """FISTA with a wavelet-domain L1 prior: min_x ||h*x - y||^2 + lam ||Wx||_1."""
    h_flip = lsf[::-1]
    threshold = lam * step_size
    x = y.copy()
    u = x.copy()
    t = 1.0
    u_new = u
    for _ in range(n_iter):
        grad = fftconvolve(fftconvolve(x, lsf, mode="same") - y, h_flip, mode="same")
        u_new = _prox_wavelet_l1(x - step_size * grad, threshold, wavelet, n_levels)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        x = u_new + ((t - 1.0) / t_new) * (u_new - u)
        u, t = u_new, t_new
    return u_new


def m_matched_filter(y, lsf, peak_centers, peak_sigma):
    """Place each known template, fit amplitudes by inner-product projection.

    The most informed method on the panel by a wide margin: it is *told* where
    the lines are.  On real data the equivalent knowledge is the redshift.
    """
    n = len(y)
    grid = np.arange(n)
    out = np.zeros(n)
    for cen in peak_centers:
        clean = np.exp(-0.5 * ((grid - cen) / peak_sigma) ** 2)
        blurred_t = fftconvolve(clean, lsf, mode="same")
        amp = float(np.dot(y, blurred_t) / (np.dot(blurred_t, blurred_t) + 1e-30))
        out += amp * clean
    return out


# ── the learned panel ─────────────────────────────────────────────────────────
def train_toy_cnn(lsf, *, device=None, deterministic: bool = False, quiet: bool = False):
    """A small SR1-shaped 1D ResNet, trained on fresh draws from the generator."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if deterministic:
        torch.use_deterministic_algorithms(True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(TORCH_SEED)

    class ResBlock1D(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.c1 = nn.Conv1d(ch, ch, 7, padding=3)
            self.c2 = nn.Conv1d(ch, ch, 7, padding=3)

        def forward(self, x):
            return x + self.c2(F.relu(self.c1(F.relu(x))))

    class TinySR1(nn.Module):
        def __init__(self, ch=32, n_blocks=4):
            super().__init__()
            self.stem = nn.Conv1d(1, ch, 7, padding=3)
            self.blocks = nn.Sequential(*[ResBlock1D(ch) for _ in range(n_blocks)])
            self.head = nn.Conv1d(ch, 1, 7, padding=3)

        def forward(self, x):
            return self.head(self.blocks(F.relu(self.stem(x)))).squeeze(1)

    def make_batch(B, n_peak_range=(1, 5), sig_range=(3.0, 7.0),
                   amp_range=(0.2, 1.2), margin=20):
        g = np.arange(N)
        truths = np.zeros((B, N), dtype=np.float32)
        observed = np.zeros((B, N), dtype=np.float32)
        for b in range(B):
            sig_signal = np.zeros(N, dtype=np.float32)
            for _ in range(np.random.randint(n_peak_range[0], n_peak_range[1] + 1)):
                mu = np.random.uniform(margin, N - margin)
                sig = np.random.uniform(*sig_range)
                amp = np.random.uniform(*amp_range)
                sig_signal += (amp * np.exp(-0.5 * ((g - mu) / sig) ** 2)).astype(np.float32)
            truths[b] = sig_signal
            observed[b] = (fftconvolve(sig_signal, lsf, mode="same").astype(np.float32)
                           + np.random.normal(0, NOISE_SIGMA, size=N).astype(np.float32))
        return (torch.from_numpy(observed).unsqueeze(1).to(dev),
                torch.from_numpy(truths).to(dev))

    model = TinySR1().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_ITERS)

    np.random.seed(0)
    losses = []
    model.train()
    for it in range(N_ITERS):
        obs_b, tru_b = make_batch(BATCH_SIZE)
        loss = F.mse_loss(model(obs_b), tru_b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if not quiet and (it + 1) % 250 == 0:
            print(f"  iter {it + 1:>4d}/{N_ITERS}   train MSE = {np.mean(losses[-50:]):.5f}")

    n_params = sum(p.numel() for p in model.parameters())
    if not quiet:
        print(f"\nTrained TinySR1 ({n_params:,} parameters) on {N_ITERS} iters "
              f"of fresh batches, on {dev.type}.")

    def infer(y):
        model.eval()
        with torch.no_grad():
            y_t = torch.from_numpy(y.astype(np.float32)).to(dev)[None, None]
            return model(y_t).cpu().numpy().ravel()

    return infer, float(np.mean(losses[-50:]))


# ── measurements ──────────────────────────────────────────────────────────────
def fwhm_of_peak(y, near_idx, halfwin=30):
    """FWHM by counting samples above half maximum, around a known position."""
    seg = y[max(0, near_idx - halfwin):min(len(y), near_idx + halfwin + 1)]
    pk = float(seg.max())
    if pk <= 0:
        return np.nan
    above = np.where(seg >= pk / 2.0)[0]
    return float(above[-1] - above[0]) if len(above) >= 2 else np.nan


def pair_dip(y, idx_a=180, idx_b=230):
    """Depth of the valley between the blended pair, 0 = merged, 1 = resolved."""
    pk_a = float(y[max(0, idx_a - 5):idx_a + 6].max())
    pk_b = float(y[max(0, idx_b - 5):idx_b + 6].max())
    if min(pk_a, pk_b) <= 0:
        return np.nan
    return 1.0 - float(y[idx_a:idx_b + 1].min()) / min(pk_a, pk_b)


def build(cache=None, outdir: Path | None = None, *,
          deterministic: bool = False, device: str | None = None) -> Path:
    """Draw figure 1.  ``cache`` is accepted and unused -- the toy is synthetic."""
    style.use_agg()
    import matplotlib.pyplot as plt
    import pandas as pd

    style.use_paper_style()
    outdir = Path(outdir) if outdir else paths.figures_dir()
    outdir.mkdir(parents=True, exist_ok=True)

    x, true, lsf, blurred, observed = make_toy()
    print(f"True peak amplitude: {true.max():.3f}")
    print(f"Blurred peak amplitude: {blurred.max():.3f}  "
          f"(loss factor {true.max() / blurred.max():.2f}×)")
    print(f"Noise sigma: {NOISE_SIGMA},   peak SNR ~ {blurred.max() / NOISE_SIGMA:.1f}")

    infer, final_mse = train_toy_cnn(lsf, device=device, deterministic=deterministic)

    peak_centers = [p[0] for p in TRUE_PARAMS]
    peak_sigma = float(np.mean([p[1] for p in TRUE_PARAMS]))
    results = {
        "Noisy observed":          observed,
        "Wiener (SNR=10)":         m_wiener(observed, lsf, snr=10.0),
        "Tikhonov (λ=0.01)":       m_tikhonov(observed, lsf, lam=0.01),
        "TV (λ=0.05)":             m_tv(observed, lsf, lam=0.05),
        "Richardson-Lucy (30 it)": m_richardson_lucy(observed, lsf, n_iter=30),
        "Wavelet Sparse":          m_sparse_wavelet(observed, lsf),
        "Matched filter":          m_matched_filter(observed, lsf, peak_centers, peak_sigma),
        "SR1-CNN (toy-trained)":   infer(observed),
    }
    for name, arr in results.items():
        assert arr.shape == (N,), f"{name} returned shape {arr.shape}"
        assert np.isfinite(arr).all(), f"{name} contains non-finite values"
    print(f"All {len(results)} methods produced finite outputs of shape {(N,)}.")

    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True, sharey=True)
    for ax, (name, arr) in zip(axes.flat, results.items()):
        rmse = float(np.sqrt(np.mean((arr - true) ** 2)))
        ax.plot(x, true, "k-", lw=1.0, alpha=0.75, label="truth")
        ax.plot(x, arr, color=METHOD_COLORS[name], lw=1.3, label=name)
        ax.set_title(f"{name}   RMSE={rmse:.3f}")
        ax.axhline(0, color="0.7", lw=0.4, alpha=0.6, zorder=0)
        ax.set_ylim(-0.25, 1.15)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.8)

    # Light tint and a bold title on the learned panel: it is the comparison
    # the paper is about, and eight identical panels bury it.
    sr1_ax = axes.flat[list(results).index("SR1-CNN (toy-trained)")]
    sr1_ax.set_facecolor((1.00, 0.97, 0.93))
    sr1_ax.set_title(sr1_ax.get_title(), fontweight="bold")

    for ax in axes[-1]:
        ax.set_xlabel("pixel")
    for ax in axes[:, 0]:
        ax.set_ylabel("Amplitude")

    plt.tight_layout()
    out = outdir / "fig_toy_1d.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame([{
        "Method": name,
        "RMSE": round(float(np.sqrt(np.mean((arr - true) ** 2))), 4),
        "Peak A amp": round(float(arr[max(0, 180 - 5):180 + 6].max()), 3),
        "A-B dip": round(pair_dip(arr), 3),
        "Peak A FWHM (pix)": round(fwhm_of_peak(arr, 180), 1),
    } for name, arr in results.items()])
    print("True FWHM of Peak A: 2.355 × 4 = 9.4 pixels")
    print("True Peak A amplitude: 1.000")
    print("Perfect A-B dip (resolved pair): 1.000")
    print()
    print(df.to_string(index=False))
    print(f"\nfinal train MSE: {final_mse:.5f}")
    print(f"Saved → {out}")
    return out
