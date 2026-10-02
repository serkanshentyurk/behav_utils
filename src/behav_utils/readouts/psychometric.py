"""Psychometric curve with bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.data.arrays import TrialArrays
from behav_utils.readouts._base import X_FIT, _ro, bin_edges

PARAMS = ('mu', 'sigma', 'lapse_low', 'lapse_high')


@dataclass(frozen=True)
class PsychometricCurve:
    """Cumulative-Gaussian fit to P(B | stimulus), with binned data and a bootstrap band.

    ``ci`` is a (4, 2) array of 95% intervals in PARAMS order and ``band`` a
    (2, len(x)) array of curve percentiles; both are None when
    ``n_bootstrap=0`` or no bootstrap fit succeeded. ``success`` False means
    the point fit failed: params are NaN, ``y`` is NaN, the binned data are
    still filled.
    """

    mu: float
    sigma: float
    lapse_low: float
    lapse_high: float
    x: np.ndarray                 # (200,) evaluation grid
    y: np.ndarray                 # (200,) fitted P(B)
    bin_centres: np.ndarray       # (n_bins,)
    bin_means: np.ndarray         # (n_bins,) empirical P(B), NaN where empty
    bin_counts: np.ndarray        # (n_bins,)
    n_trials: int
    success: bool
    ci: np.ndarray | None = None       # (4, 2) lo/hi per param, PARAMS order
    band: np.ndarray | None = None     # (2, 200) lo/hi curve
    n_bootstrap: int = 0                  # successful bootstrap fits

    def __repr__(self) -> str:
        return (f'PsychometricCurve(mu={self.mu:.3f}, sigma={self.sigma:.3f}, '
                f'lapse_low={self.lapse_low:.3f}, lapse_high={self.lapse_high:.3f}, '
                f'n_trials={self.n_trials}, n_bootstrap={self.n_bootstrap}, success={self.success})')

    @property
    def params(self) -> pd.Series:
        return pd.Series([self.mu, self.sigma, self.lapse_low, self.lapse_high],
                         index=list(PARAMS), dtype=float, name='value')

    def to_rows(self) -> pd.DataFrame:
        """Tidy: one row per parameter — param, value, lo, hi."""
        ci = self.ci if self.ci is not None else np.full((4, 2), np.nan)
        return pd.DataFrame({'param': list(PARAMS), 'value': self.params.to_numpy(),
                             'lo': ci[:, 0], 'hi': ci[:, 1]})


def _binned(stim: np.ndarray, choice: np.ndarray, n_bins: int):
    edges = bin_edges(n_bins)
    centres = (edges[:-1] + edges[1:]) / 2
    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        hi = stim <= edges[i + 1] if i == n_bins - 1 else stim < edges[i + 1]
        m = (stim >= edges[i]) & hi
        counts[i] = m.sum()
        if counts[i] > 0:
            means[i] = np.mean(choice[m])
    return centres, means, counts


def compute_psychometric_curve(
    arrays: TrialArrays,
    *,
    n_bins: int = 8,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> PsychometricCurve:
    """Fit a cumulative Gaussian to one block; CIs and band from trial bootstrap.

    Args:
        arrays:      pre-filtered trials.
        n_bins:      bins for the empirical scatter only (the fit uses raw trials).
        n_bootstrap: trial resamples for the CIs (0 disables).
        seed:        bootstrap seed.
    """
    v = arrays.valid()
    keep = ~np.isnan(v.stimulus)
    stim, choice = v.stimulus[keep], v.choice[keep]
    n = int(len(stim))
    nan4 = (np.nan,) * 4
    if n == 0:
        z = np.full(n_bins, np.nan)
        return PsychometricCurve(*nan4, X_FIT, _ro(np.full(X_FIT.size, np.nan)),
                                 _ro(z), _ro(z), _ro(np.zeros(n_bins)), 0, False)

    centres, means, counts = _binned(stim, choice, n_bins)
    fit = fit_psychometric(stim, choice, x_eval=X_FIT, n_bootstrap=n_bootstrap, seed=seed)
    if not fit.get('success', False):
        return PsychometricCurve(*nan4, X_FIT, _ro(np.full(X_FIT.size, np.nan)),
                                 _ro(centres), _ro(means), _ro(counts), n, False)

    n_ok = int(fit.get('n_bootstrap_success', 0))
    ci = band = None
    if n_bootstrap > 0 and n_ok > 0:
        ci = _ro([fit.get(f'{k}_ci', (np.nan, np.nan)) for k in PARAMS])
        lo, hi = fit.get('y_fit_ci', (None, None))
        if lo is not None:
            band = _ro(np.stack([lo, hi]))
    return PsychometricCurve(
        float(fit['mu']), float(fit['sigma']), float(fit['lapse_low']), float(fit['lapse_high']),
        X_FIT, _ro(fit['y_fit']), _ro(centres), _ro(means), _ro(counts), n, True,
        ci=ci, band=band, n_bootstrap=n_ok,
    )
