"""Trial-history statistics.

Lag-1 statistics use the frozen ``prev_*`` view (``arrays.lag1_pairs()``), so
they are block-aware on pooled data and trial-exchangeable. Multi-lag
statistics (``logistic_history``, ``history_interaction_r2``) and the serial-
dependence profile shift by adjacency over the responded trials, so they bridge
session seams on pooled data and are registered ``exchangeable=False``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats._binning import DEFAULT_N_BINS, bin_index
from behav_utils.stats.registry import fit, stat

MIN_RESPONDED = 10
MIN_PAIRS = 5


# ── lag-1 ───────────────────────────────────────────────────────────────────

def _pairs(a: TrialArrays):
    """(pairs, ok): the lag-1 pairs, and whether there are enough trials to use them."""
    if a.n_responded < MIN_RESPONDED:
        return None, False
    p = a.lag1_pairs()
    return p, p.n_trials >= MIN_PAIRS


@stat('recency')
def recency(a: TrialArrays, *, rng) -> float:
    """P(choose B | prev category B) − P(choose B | prev category A)."""
    p, ok = _pairs(a)
    if not ok:
        return np.nan
    was_b, was_a = p.prev_category == 1, p.prev_category == 0
    if was_b.sum() == 0 or was_a.sum() == 0:
        return np.nan
    return float(np.mean(p.choice[was_b]) - np.mean(p.choice[was_a]))


@stat('stimulus_recency')
def stimulus_recency(a: TrialArrays, *, rng) -> float:
    """P(choose B | prev stimulus > 0) − P(choose B | prev stimulus <= 0)."""
    p, ok = _pairs(a)
    if not ok:
        return np.nan
    b_side, a_side = p.prev_stimulus > 0, p.prev_stimulus <= 0
    if b_side.sum() == 0 or a_side.sum() == 0:
        return np.nan
    return float(np.mean(p.choice[b_side]) - np.mean(p.choice[a_side]))


@stat('recency_divergence')
def recency_divergence(a: TrialArrays, *, rng) -> float:
    """stimulus_recency − recency (sensory vs categorical serial dependence)."""
    s, c = stimulus_recency(a, rng=rng), recency(a, rng=rng)
    if np.isnan(s) or np.isnan(c):
        return np.nan
    return float(s - c)


@stat('win_stay')
def win_stay(a: TrialArrays, *, rng) -> float:
    """P(repeat | rewarded) − P(repeat | unrewarded)."""
    p, ok = _pairs(a)
    if not ok:
        return np.nan
    repeat = p.choice == p.prev_choice
    won, lost = p.prev_reward == 1, p.prev_reward == 0
    if won.sum() == 0 or lost.sum() == 0:
        return np.nan
    return float(np.mean(repeat[won]) - np.mean(repeat[lost]))


@stat('win_stay_rate')
def win_stay_rate(a: TrialArrays, *, rng) -> float:
    """P(repeat | rewarded)."""
    if a.n_responded < MIN_RESPONDED:
        return np.nan
    p = a.lag1_pairs()
    won = p.prev_reward == 1
    if won.sum() == 0:
        return np.nan
    return float(np.mean(p.choice[won] == p.prev_choice[won]))


@stat('lose_shift')
def lose_shift(a: TrialArrays, *, rng) -> float:
    """P(switch | unrewarded)."""
    p, ok = _pairs(a)
    if not ok:
        return np.nan
    switch = p.choice != p.prev_choice
    lost = p.prev_reward == 0
    if lost.sum() == 0:
        return np.nan
    return float(np.mean(switch[lost]))


@stat('choice_autocorr')
def choice_autocorr(a: TrialArrays, *, rng) -> float:
    """Lag-1 correlation between choice_t and choice_{t−1}."""
    if a.n_responded < MIN_RESPONDED + 1:
        return np.nan
    p = a.lag1_pairs()
    cur, lag = p.choice, p.prev_choice
    if len(cur) < 2 or np.std(cur) == 0 or np.std(lag) == 0:
        return np.nan
    return float(np.corrcoef(cur, lag)[0, 1])


@stat('perseveration', exchangeable=False)
def perseveration(a: TrialArrays, *, rng) -> float:
    """Observed repeat rate minus the repeat rate predicted by P(B | stimulus bin) alone."""
    v = a.valid()
    if v.n_trials < 20:
        return np.nan
    p = a.lag1_pairs()
    if p.n_trials == 0:
        return np.nan
    observed = float(np.mean(p.choice == p.prev_choice))

    n_bins = DEFAULT_N_BINS
    idx = bin_index(v.stimulus, n_bins)
    p_b = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            p_b[b] = np.mean(v.choice[m])

    cur_bin = bin_index(p.stimulus, n_bins)
    prv_bin = bin_index(p.prev_stimulus, n_bins)
    pc, pp = p_b[cur_bin], p_b[prv_bin]
    keep = ~(np.isnan(pc) | np.isnan(pp))
    if keep.sum() == 0:
        return np.nan
    expected = pc[keep] * pp[keep] + (1 - pc[keep]) * (1 - pp[keep])
    return float(observed - np.mean(expected))


# ── multi-lag (adjacency over responded trials) ─────────────────────────────

N_BACK = 3
L2_PENALTY = 0.1

LOGISTIC_HISTORY = ('w_stimulus',
                    *[f'w_prev_choice_{k}' for k in range(1, N_BACK + 1)],
                    *[f'w_prev_outcome_{k}' for k in range(1, N_BACK + 1)],
                    'history_decay')


def _history_design(v: TrialArrays, n_back: int):
    """(y, X_stim, X_full) for the logistic history regressions, or None if too short."""
    c, s = v.choice, v.stimulus
    outcome = (v.choice == v.category).astype(float)
    n = len(c)
    if n < n_back + 10:
        return None
    y = c[n_back:]
    n_obs = len(y)
    X_stim = np.column_stack([np.ones(n_obs), s[n_back:]])
    H = np.zeros((n_obs, 1 + 2 * n_back))
    H[:, 0] = s[n_back:]
    for k in range(1, n_back + 1):
        H[:, k] = c[n_back - k: n - k] - 0.5
        H[:, n_back + k] = outcome[n_back - k: n - k] - 0.5
    X_full = np.column_stack([np.ones(n_obs), H])
    return y, X_stim, X_full


def _neg_ll(beta, X, y, l2=0.0):
    logits = np.clip(X @ beta, -20, 20)
    p = np.clip(1 / (1 + np.exp(-logits)), 1e-10, 1 - 1e-10)
    nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return nll + (l2 / 2) * np.sum(beta[1:] ** 2)


def _neg_ll_grad(beta, X, y, l2=0.0):
    logits = np.clip(X @ beta, -20, 20)
    p = np.clip(1 / (1 + np.exp(-logits)), 1e-10, 1 - 1e-10)
    g = -X.T @ (y - p)
    g[1:] += l2 * beta[1:]
    return g


@fit('logistic_history', outputs=LOGISTIC_HISTORY, exchangeable=False)
def logistic_history(a: TrialArrays, *, rng) -> tuple:
    """L2-regularised logistic regression of choice on stimulus + N_BACK trials of history.

    Outputs: stimulus weight, previous-choice weights 1..N_BACK, previous-
    outcome weights 1..N_BACK, and the exponential decay rate of the
    previous-choice weights across lags.
    """
    nan = tuple(np.nan for _ in LOGISTIC_HISTORY)
    v = a.valid()
    if v.n_trials < 30:
        return nan
    d = _history_design(v, N_BACK)
    if d is None:
        return nan
    y, _, X = d
    try:
        res = minimize(_neg_ll, np.zeros(X.shape[1]), args=(X, y, L2_PENALTY),
                       jac=_neg_ll_grad, method='L-BFGS-B')
    except (ValueError, RuntimeError):
        return nan
    if not res.success:
        return nan
    beta = res.x
    w_stim = float(beta[1])
    w_choice = [float(beta[1 + k]) for k in range(1, N_BACK + 1)]
    w_outcome = [float(beta[1 + N_BACK + k]) for k in range(1, N_BACK + 1)]
    mags = np.abs(w_choice)
    if N_BACK >= 2 and np.all(mags > 1e-6):
        lags = np.arange(1, N_BACK + 1, dtype=float)
        decay = float(-np.polyfit(lags, np.log(mags), 1)[0])
    else:
        decay = np.nan
    return (w_stim, *w_choice, *w_outcome, decay)


@stat('history_interaction_r2', exchangeable=False)
def history_interaction_r2(a: TrialArrays, *, rng) -> float:
    """McFadden pseudo-R² gained by adding N_BACK trials of history to a stimulus-only model."""
    v = a.valid()
    if v.n_trials < 30:
        return np.nan
    d = _history_design(v, N_BACK)
    if d is None:
        return np.nan
    y, X_stim, X_full = d
    n_obs = len(y)
    try:
        p_bar = np.clip(np.mean(y), 1e-10, 1 - 1e-10)
        ll_null = n_obs * (p_bar * np.log(p_bar) + (1 - p_bar) * np.log(1 - p_bar))
        r_s = minimize(_neg_ll, np.zeros(X_stim.shape[1]), args=(X_stim, y), method='L-BFGS-B')
        if not r_s.success:
            return np.nan
        r_f = minimize(_neg_ll, np.zeros(X_full.shape[1]), args=(X_full, y), method='L-BFGS-B')
        if not r_f.success:
            return np.nan
    except (ValueError, np.linalg.LinAlgError):
        return np.nan
    if ll_null == 0:
        return np.nan
    return float((1 - (-r_f.fun) / ll_null) - (1 - (-r_s.fun) / ll_null))


# ── serial-dependence profile ───────────────────────────────────────────────

SD_PROFILE = ('sd_slope', 'sd_curvature', 'sd_range')


@fit('sd_profile', outputs=SD_PROFILE, exchangeable=False)
def sd_profile(a: TrialArrays, *, rng) -> tuple:
    """Scalar shape features of the post-correct serial-dependence profile.

    Linear slope, quadratic curvature and range of
    :func:`behav_utils.readouts.compute_sd_profile` across previous-stimulus
    bins; NaN when fewer than 4 bins are populated (curvature needs 5).
    """
    from behav_utils.readouts.sd_profile import compute_sd_profile
    prof = compute_sd_profile(a, n_bins=DEFAULT_N_BINS)
    ok = ~np.isnan(prof.profile)
    if ok.sum() < 4:
        return (np.nan, np.nan, np.nan)
    x, y = prof.centres[ok], prof.profile[ok]
    try:
        slope = float(np.polyfit(x, y, 1)[0])
    except (ValueError, np.linalg.LinAlgError):
        slope = np.nan
    try:
        curvature = float(np.polyfit(x, y, 2)[0]) if ok.sum() >= 5 else np.nan
    except (ValueError, np.linalg.LinAlgError):
        curvature = np.nan
    return (slope, curvature, float(np.max(y) - np.min(y)))
