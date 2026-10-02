"""Statistics derived from a cumulative-Gaussian psychometric fit."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from behav_utils.analysis.psychometry import fit_psychometric, fit_psychometric_gof
from behav_utils.data.arrays import TrialArrays
from behav_utils.stats.registry import fit, stat

PSYCHOMETRIC = ('mu', 'sigma', 'lapse_low', 'lapse_high')

# A fit with slope above this is treated as unreliable (near-chance or
# strongly biased animal): mu and sigma become NaN, lapses are kept.
SLOPE_THRESHOLD = 5.0
MU_BOUND = 0.99


@fit('psychometric', outputs=PSYCHOMETRIC)
def psychometric(a: TrialArrays, *, rng) -> tuple:
    """Cumulative-Gaussian fit: (mu, sigma, lapse_low, lapse_high).

    mu and sigma are NaN when the fit is unreliable (sigma > SLOPE_THRESHOLD
    or |mu| > MU_BOUND); the lapses are returned regardless.
    """
    v = a.valid()
    p = fit_psychometric(v.stimulus, v.choice)
    if not p.get('success', False):
        return (np.nan, np.nan, np.nan, np.nan)
    mu, sigma = p['mu'], p['sigma']
    unreliable = sigma > SLOPE_THRESHOLD or abs(mu) > MU_BOUND
    return (
        np.nan if unreliable else float(mu),
        np.nan if unreliable else float(sigma),
        float(p['lapse_low']),
        float(p['lapse_high']),
    )


@stat('pse')
def pse(a: TrialArrays, *, rng) -> float:
    """Lapse-corrected point of subjective equality: where the full curve crosses 0.5.

    No slope guard (unlike 'mu'), so shallow early-learning windows survive.
    NaN when the fit fails, the corrected curve cannot cross 0.5, or |PSE| > 1.
    """
    v = a.valid()
    p = fit_psychometric(v.stimulus, v.choice)
    if not p.get('success', False):
        return np.nan
    mu, sigma, g, lam = p['mu'], p['sigma'], p['lapse_low'], p['lapse_high']
    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0):
        return np.nan
    denom = 1.0 - g - lam
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    arg = (0.5 - g) / denom
    if not (0.0 < arg < 1.0):
        return np.nan
    x = float(norm.ppf(arg, loc=mu, scale=sigma))
    if not np.isfinite(x) or abs(x) > 1.0:
        return np.nan
    return x


@stat('psychometric_gof')
def psychometric_gof(a: TrialArrays, *, rng) -> float:
    """R² between binned choice proportions and the fitted psychometric curve."""
    v = a.valid()
    if v.n_trials < 20:
        return np.nan
    p = fit_psychometric(v.stimulus, v.choice)
    if not p.get('success', False):
        return np.nan
    return float(fit_psychometric_gof(v.stimulus, v.choice, p).get('r_squared', np.nan))
