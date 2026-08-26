# Four ways mean absolute error lies

Every classical method in this benchmark has free parameters, and they have to
be chosen somehow. The obvious objective — minimise mean absolute error against
the high-resolution reference — is wrong in four separate ways. Each was found
by shipping it, and each cost a full rebuild.

They are written down here because none of them is specific to spectra. Any
benchmark that tunes a restoration method against a noisy reference by an
error norm has all four available to it.

The rule that came out of it: **a number is only safe if a test recomputes
it.**

---

## 1. Smoothing

**What it rewards.** A filter that erases every emission line incurs no
line-shaped residual. Against a noisy reference, the residual of a smooth
output is *smaller* than the residual of a correct one, because the correct one
also reproduces the noise it is scored against.

**Guard.** Median line signal-to-noise ≥ 0.9 × the no-deconvolution baseline.

**Where it was found.** The first tuning pass drove every regularisation
strength to its maximum and produced spectra with no lines in them at all,
scoring better than the input they came from.

## 2. Shrinkage

**What it rewards.** Scaling the whole output toward zero. If the reference is
`signal + noise` and the estimate is `k × signal`, then for `k < 1` the
absolute error falls for any noise level, regardless of whether the estimate
resolved anything.

**Guard.** Output standard deviation within [0.90, 1.15] of the target's.

**Why guard 1 cannot see it.** Line S/N is amplitude over sideband noise. A
global rescale multiplies both and leaves the ratio exactly unchanged. Every
scale-free diagnostic is blind to shrinkage by construction — which is also
why the headline comparison in this benchmark is reported on a scale-free
metric *and* an amplitude ratio, never on MAE alone.

**Where it was found.** The Wiener filter's MMSE form `W = H/(H² + 1/snr)` has
DC gain `1/(1 + 1/snr)` — 0.91 at snr = 10, 0.44 at snr = 0.8. Tuning it
against MAE drove `snr` toward zero. The filter is now normalised to unit gain
at zero frequency, which keeps its shape and removes its ability to buy MAE
with a rescale. The deep-learning model is subject to exactly the same effect
and is not immune to it: SR2's output sits at 0.54 of the reference's scale.

## 3. Blurring

**What it rewards.** Broadening lines rather than sharpening them. A unit-gain
Wiener filter with `snr ≤ 1` has its maximum at zero frequency, so it cannot
amplify any frequency — it can only smooth — while still scoring well.

**Guard.** Median line FWHM bias no larger than the baseline's.

**Why guard 2 cannot see it.** The amplitude guard is width-invariant: a line
can be twice as wide at the same standard deviation.

## 4. Merging

**What it rewards.** Smearing two close lines into one peak. This is the one
that matters physically — resolving blended doublets is most of the reason to
deconvolve a spectrum at all.

**Guard.** The fraction of resolvable [O III] pairs kept resolved must be at
least the no-deconvolution baseline's, measured on real spectra from a
galaxy-disjoint calibration set.

**Why guards 1–3 cannot see it.** All three are single-line Gaussian-fit
statistics, and a Gaussian fitted to a *blended* doublet has much the same
amplitude, signal-to-noise and width as one fitted to a separated pair. All
three pass a filter that merges the doublet.

**A synthetic pair does not work.** One was tried first and discarded: a
synthetic doublet is far easier to hold apart than a real one, and it passed
settings that merge in real data. The guard has to be measured on real spectra.

**Where it was found.** With the kernel that was in use at the time, Wiener,
Tikhonov and Wiener + TV merged line pairs that plain cubic interpolation still
resolves — 0% pair survival for all three, against 43% for doing nothing.

---

## What the guards do not fix

A guard rules a setting out; it does not make a method good. Tikhonov fails
guard 4 at every setting searched — nothing passes it and the amplitude guard
together. It is retained at its published value rather than dropped, because
dropping it would be a larger claim than the evidence supports: the failure is
of this segmented implementation on this grid, not of Tikhonov regularisation.
`classical_params.json` records the failure in its `failed_guards` field, and
the paper says so.

## Where they live

`specsrbench build tune` applies all four *inside* the search. They used to be
applied by hand afterwards, which is how a Richardson–Lucy setting of
`n_iter=1` — one iteration, i.e. barely deconvolving at all — came to be
shipped.

`tests/test_invariants.py` asserts all four against whatever cache is present,
so a rebuild that violates one fails the suite rather than reaching a figure.
