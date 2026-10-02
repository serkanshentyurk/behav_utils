"""PSE dynamics: how the point of subjective equality moves across one block of trials.

Both fits share the cumulative-Gaussian likelihood of ``fit_psychometric`` with
sigma and the two lapses held constant across the block and only the PSE
allowed to move with trial index ``t`` (0 = first trial of the block):

    exponential   mu(t) = mu_end + (mu_start - mu_end) * exp(-t / tau)       (gradual learning)
    step          mu(t) = mu_start if t < t_switch else mu_end               (switching)

Outputs are the exponential fit's parameters plus the model comparison
against the step fit. ``pse_end`` is the asymptote (an extrapolation when the
fit is censored); ``pse_final`` is the fitted PSE at the last trial of the
block (always in-sample) — use it as the endpoint. ``pse_tau`` and ``pse_trials_to_90`` (= tau * ln 10)
are in trials; when tau exceeds the block length the trajectory had not
plateaued and ``pse_censored`` is 1. ``pse_shape_daic`` = AIC(exp) - AIC(step):
negative favours a gradual trajectory, positive a step.

The block is the caller's: pass the sessions after a switch in order
(``TrialArrays.from_sessions(sorted_block)``) so ``t`` is trials since the
switch across sessions. Order-dependent, so ``exchangeable=False``: resample
by session, never by trial.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats.registry import fit

PSE_DYNAMICS = ('pse_start', 'pse_end', 'pse_final', 'pse_tau', 'pse_trials_to_90', 'pse_censored',
                'pse_shape_daic', 'pse_step_switch')
MIN_TRIALS = 100
_EPS = 1e-10


def _nll(mu: np.ndarray, sigma: float, lo: float, hi: float, stim: np.ndarray, choice: np.ndarray) -> float:
    p = lo + (1.0 - lo - hi) * norm.cdf((stim - mu) / sigma)
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.sum(choice * np.log(p) + (1.0 - choice) * np.log(1.0 - p)))


class _Fit:
    """Minimal result holder: ``x`` = (mu_s, mu_e, log_tau, sigma, lo, hi), ``fun`` = NLL, ``k`` = free params."""

    def __init__(self, x, fun, k):
        self.x, self.fun, self.k = np.asarray(x, float), float(fun), int(k)


def _fit_exponential(t, stim, choice, n, shape=None):
    """Profile tau on a log grid, then refine from the best grid point.

    ``shape`` = (sigma, lapse_low, lapse_high) pins the psychometric shape so
    only mu_start, mu_end and tau are free (3 params); None fits all six.
    """
    pinned = shape is not None
    if pinned:
        sig, lo, hi = shape

        def nll_at(x, tau):
            mu = x[1] + (x[0] - x[1]) * np.exp(-t / tau)
            return _nll(mu, sig, lo, hi, stim, choice)
        bounds, x0 = [(-1, 1), (-1, 1)], [0.0, 0.0]
    else:
        def nll_at(x, tau):
            mu_s, mu_e, sigma, lo_, hi_ = x
            mu = mu_e + (mu_s - mu_e) * np.exp(-t / tau)
            return _nll(mu, sigma, lo_, hi_, stim, choice)
        bounds, x0 = [(-1, 1), (-1, 1), (0.01, 10), (0, 0.5), (0, 0.5)], [0.0, 0.0, 0.5, 0.05, 0.05]
    best = None
    for tau in np.geomspace(2.0, 5.0 * n, 20):
        r = minimize(nll_at, x0, args=(tau,), method='L-BFGS-B', bounds=bounds)
        if np.all(np.isfinite(r.x)) and (best is None or r.fun < best[0]):
            best = (r.fun, r.x, tau)
            x0 = list(r.x)           # warm-start the next grid point
    if best is None:
        return None
    if pinned:
        x_full = [*best[1], np.log(best[2]), sig, lo, hi]
        def obj(y):
            return nll_at([y[0], y[1]], np.exp(y[2]))
        y0 = [*best[1], np.log(best[2])]
        r = minimize(obj, y0, method='L-BFGS-B', bounds=[(-1, 1), (-1, 1), (np.log(1.0), np.log(50.0 * n))])
        if np.all(np.isfinite(r.x)) and r.fun <= best[0] + 1e-9:
            x_full = [r.x[0], r.x[1], r.x[2], sig, lo, hi]
            return _Fit(x_full, r.fun, 3)
        return _Fit(x_full, best[0], 3)

    def obj6(x):
        mu_s, mu_e, log_tau, sigma, lo_, hi_ = x
        mu = mu_e + (mu_s - mu_e) * np.exp(-t / np.exp(log_tau))
        return _nll(mu, sigma, lo_, hi_, stim, choice)
    x6 = [*best[1][:2], np.log(best[2]), *best[1][2:]]
    r = minimize(obj6, x6, method='L-BFGS-B',
                 bounds=[(-1, 1), (-1, 1), (np.log(1.0), np.log(50.0 * n)), (0.01, 10), (0, 0.5), (0, 0.5)])
    if np.all(np.isfinite(r.x)) and r.fun <= best[0] + 1e-9:
        return _Fit(r.x, r.fun, 6)
    return _Fit(x6, best[0], 6)


def _fit_step(t, stim, choice, n, shape=None):
    """Profile over the switch trial on a grid; MLE for the other params at each. Returns (nll, x, t_switch, k)."""
    grid = np.unique(np.clip(np.round(np.geomspace(1, max(2, n - 1), 25)), 1, n - 1)).astype(int)
    best = None
    for ts in grid:
        after = t >= ts
        if shape is not None:
            sig, lo, hi = shape

            def obj(x, after=after, sig=sig, lo=lo, hi=hi):
                return _nll(np.where(after, x[1], x[0]), sig, lo, hi, stim, choice)
            r = minimize(obj, [0.0, 0.0], method='L-BFGS-B', bounds=[(-1, 1), (-1, 1)])
            k = 3
        else:
            def obj(x, after=after):
                mu_s, mu_e, sigma, lo_, hi_ = x
                return _nll(np.where(after, mu_e, mu_s), sigma, lo_, hi_, stim, choice)
            r = minimize(obj, [0.0, 0.0, 0.5, 0.05, 0.05], method='L-BFGS-B',
                         bounds=[(-1, 1), (-1, 1), (0.01, 10), (0, 0.5), (0, 0.5)])
            k = 6
        if np.all(np.isfinite(r.x)) and (best is None or r.fun < best[0]):
            best = (r.fun, r.x, int(ts), k)
    return best


def fit_pse_dynamics(a: TrialArrays, shape=None) -> tuple:
    """The PSE_DYNAMICS tuple for one block. ``shape=(sigma, lapse_low, lapse_high)`` pins the
    psychometric shape so only the criterion trajectory is fitted (3 free parameters instead of 6)."""
    nan = tuple(np.nan for _ in PSE_DYNAMICS)
    v = a.valid()
    n = v.n_trials
    if n < MIN_TRIALS:
        return nan
    if shape is not None and not all(np.isfinite(shape)):
        return nan
    t = np.arange(n, dtype=float)
    stim, choice = v.stimulus, v.choice
    try:
        exp_fit = _fit_exponential(t, stim, choice, n, shape)
        step_fit = _fit_step(t, stim, choice, n, shape)
    except (ValueError, RuntimeError):
        return nan
    if exp_fit is None or step_fit is None:
        return nan
    mu_s, mu_e, log_tau, *_ = exp_fit.x
    tau = float(np.exp(log_tau))
    aic_exp = 2 * exp_fit.k + 2 * exp_fit.fun
    aic_step = 2 * step_fit[3] + 2 * step_fit[0]     # fitted params + the profiled switch trial
    final = mu_e + (mu_s - mu_e) * np.exp(-(n - 1) / tau)
    return (float(mu_s), float(mu_e), float(final), tau, tau * np.log(10.0), float(tau > n),
            float(aic_exp - aic_step), float(step_fit[2]))


@fit('pse_dynamics', outputs=PSE_DYNAMICS, exchangeable=False)
def pse_dynamics(a: TrialArrays, *, rng) -> tuple:
    """Exponential PSE trajectory over the block, compared with a step trajectory (all six params free)."""
    return fit_pse_dynamics(a)
