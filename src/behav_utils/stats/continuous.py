"""
The continuous-response family's statistics: how far, and which way, responses miss the truth.

All need ``continuous`` (a task whose response is a value on the stimulus scale, such as
``Reproduction``) and read the task's truth from ``TrialArrays.truth``. ``accuracy`` (the share within
the task's tolerance) and the reaction-time statistics apply to these tasks as well.
"""

import numpy as np

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats.registry import stat

MIN_RESPONDED = 10
_CONTINUOUS = ('continuous',)


def _truth_and_response(a: TrialArrays) -> tuple:
    """(truth, response) on responded trials that have a truth."""
    v = a.valid()
    ok = ~np.isnan(v.truth) & ~np.isnan(v.choice)
    return v.truth[ok], v.choice[ok]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y on x (NaN when x does not vary)."""
    var = np.var(x, ddof=1)
    return float(np.cov(x, y, ddof=1)[0, 1] / var) if var > 0 else np.nan


@stat('response_bias', requires=_CONTINUOUS)
def response_bias(a: TrialArrays, *, rng) -> float:
    """Mean signed error, response − truth: positive means responses land above the truth."""
    truth, response = _truth_and_response(a)
    return float(np.mean(response - truth)) if truth.size >= MIN_RESPONDED else np.nan


@stat('error_sd', requires=_CONTINUOUS)
def error_sd(a: TrialArrays, *, rng) -> float:
    """Standard deviation of the signed error: the spread of responses around the truth."""
    truth, response = _truth_and_response(a)
    return float(np.std(response - truth, ddof=1)) if truth.size >= MIN_RESPONDED else np.nan


@stat('mean_abs_error', requires=_CONTINUOUS)
def mean_abs_error(a: TrialArrays, *, rng) -> float:
    """Mean absolute error, |response − truth|."""
    truth, response = _truth_and_response(a)
    return float(np.mean(np.abs(response - truth))) if truth.size >= MIN_RESPONDED else np.nan


@stat('response_slope', requires=_CONTINUOUS)
def response_slope(a: TrialArrays, *, rng) -> float:
    """Slope of the response regressed on the truth: 1 is faithful, below 1 pulled towards the middle."""
    truth, response = _truth_and_response(a)
    return _slope(truth, response) if truth.size >= MIN_RESPONDED else np.nan


@stat('serial_bias', requires=_CONTINUOUS, exchangeable=False)
def serial_bias(a: TrialArrays, *, rng) -> float:
    """How far the error leans towards the previous trial's stimulus: the coefficient of previous −
    current stimulus when the error is regressed on it and on the current stimulus. Positive is
    attraction, negative repulsion. The current stimulus is in the regression because a pull towards
    the middle (``response_slope`` below 1) also makes the error depend on it; without it, that pull
    would show up here as a serial effect."""
    p = a.lag1_pairs()
    ok = ~np.isnan(p.truth) & ~np.isnan(p.prev_stimulus) & ~np.isnan(p.choice)
    if ok.sum() < MIN_RESPONDED:
        return np.nan
    current = p.stimulus[ok]
    design = np.column_stack([np.ones(ok.sum()), p.prev_stimulus[ok] - current, current])
    if np.linalg.matrix_rank(design) < 3:
        return np.nan
    coef = np.linalg.lstsq(design, p.choice[ok] - p.truth[ok], rcond=None)[0]
    return float(coef[1])
