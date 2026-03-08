"""
Behavioural Summary Statistics

Modular design with registry pattern for easy extension.
Each stat function has signature: (choices, stimuli, categories) -> scalar or dict
Handles both single-session (n_trials,) and multi-session (n_trials, n_sessions) arrays.

Used by:
    - SBI inference pipeline (flattened vector)
    - Session feature matrix for HMM (per-session dict)
    - General behavioural analysis

To add a new stat:
    @register_stat('my_stat')
    def compute_my_stat(choices, stimuli, categories):
        ...
        return scalar_or_dict
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Union
from functools import wraps

from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.analysis.utils import cumulative_gaussian


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_N_BINS = 8  # Default number of bins for binned statistics
PSYCHOMETRIC_SLOPE_THRESHOLD = 5.0  # Slope  above this = fit failure --> NaN
LOGISTIC_L2_PENALTY = 0.1  # L2 regularisation strength for logistic history regression


# =============================================================================
# REGISTRY
# =============================================================================

SUMMARY_REGISTRY: Dict[str, Callable] = {}


def register_stat(name: str):
    """Decorator to register a summary statistic function."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        SUMMARY_REGISTRY[name] = wrapper
        return wrapper
    return decorator


def list_available_stats() -> List[str]:
    """List all registered summary statistics."""
    return list(SUMMARY_REGISTRY.keys())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _ensure_1d(arr: np.ndarray) -> np.ndarray:
    """Flatten array to 1D if needed."""
    arr = np.asarray(arr)
    if arr.ndim > 1:
        return arr.flatten()
    return arr


def _is_multisession(arr: np.ndarray) -> bool:
    """
    Check if array is multi-session (2D with n_sessions > 1).
    
    WARNING: This 2D format requires equal-length sessions (columns).
    For real data with variable-length sessions, call single-session
    functions in a loop (as build_feature_matrix does).
    """
    return arr.ndim == 2 and arr.shape[1] > 1


def _apply_per_session(func: Callable, choices: np.ndarray, stimuli: np.ndarray,
                       categories: np.ndarray) -> np.ndarray:
    """Apply a single-session function to each session of multi-session data."""
    if not _is_multisession(choices):
        return func(_ensure_1d(choices), _ensure_1d(stimuli), _ensure_1d(categories))

    n_sessions = choices.shape[1]
    results = []
    for i in range(n_sessions):
        result = func(choices[:, i], stimuli[:, i], categories[:, i])
        results.append(result)

    return np.array(results)


def _valid_trials(choices: np.ndarray) -> np.ndarray:
    """Boolean mask for trials with valid (non-NaN) choices."""
    return ~np.isnan(choices)


# =============================================================================
# CORE SUMMARY STATISTICS
# =============================================================================

@register_stat('accuracy')
def compute_accuracy(choices: np.ndarray, stimuli: np.ndarray,
                     categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Overall proportion correct.

    Returns:
        float for single-session, array of shape (n_sessions,) for multi-session
    """
    choices = np.asarray(choices)
    categories = np.asarray(categories)

    if _is_multisession(choices):
        return np.mean(choices == categories, axis=0)
    else:
        c = _ensure_1d(choices)
        cat = _ensure_1d(categories)
        valid = _valid_trials(c)
        if valid.sum() == 0:
            return np.nan
        return float(np.mean(c[valid] == cat[valid]))


@register_stat('psychometric')
def compute_psychometric_params(choices: np.ndarray, stimuli: np.ndarray,
                                categories: np.ndarray,
                                slope_threshold: float = PSYCHOMETRIC_SLOPE_THRESHOLD,
                                ) -> Dict[str, Union[float, np.ndarray]]:
    """
    Fit psychometric curve and return parameters.

    When the fit produces slope (Ïƒ) above slope_threshold, PSE and slope are
    replaced with NaN because the psychometric curve is too flat for reliable
    parameter estimation (animal near chance or strong side bias).
    Lapse parameters are preserved since they can still be meaningful.

    Returns:
        Dict with 'pse', 'slope', 'lapse_low', 'lapse_high'
        Values are floats for single-session, arrays for multi-session
    """
    nan_result = {
        'pse': np.nan, 'slope': np.nan,
        'lapse_low': np.nan, 'lapse_high': np.nan
    }

    def _fit_single(c, s, cat):
        psych = fit_psychometric(s, c)
        if psych.get('success', False):
            slope_val = psych['sigma']
            pse_val = psych['mu']

            # Flag unreliable fits: slope too large or PSE at bounds
            unreliable = (slope_val > slope_threshold or abs(pse_val) > 0.99)

            return {
                'pse': np.nan if unreliable else pse_val,
                'slope': np.nan if unreliable else slope_val,
                'lapse_low': psych['lapse_low'],
                'lapse_high': psych['lapse_high'],
            }
        else:
            return nan_result

    choices = np.asarray(choices)
    stimuli = np.asarray(stimuli)

    if _is_multisession(choices):
        n_sessions = choices.shape[1]
        results = {k: [] for k in ['pse', 'slope', 'lapse_low', 'lapse_high']}

        for i in range(n_sessions):
            single_result = _fit_single(choices[:, i], stimuli[:, i], categories[:, i])
            for k, v in single_result.items():
                results[k].append(v)

        return {k: np.array(v) for k, v in results.items()}
    else:
        return _fit_single(_ensure_1d(choices), _ensure_1d(stimuli), _ensure_1d(categories))


@register_stat('recency')
def compute_recency_index(choices: np.ndarray, stimuli: np.ndarray,
                          categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Effect of previous trial category on current choice.

    Measures: P(choose B | prev_category=B) - P(choose B | prev_category=A)

    High recency = recent trials strongly influence choice (high learning rate)
    Low recency = stable behaviour (low learning rate)
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c)
        cat = _ensure_1d(cat)
        valid = _valid_trials(c)

        if valid.sum() < 10:
            return np.nan

        prev_cat = np.roll(cat, 1)
        # Exclude first trial (no valid previous) and NaN choices
        mask = np.ones(len(c), dtype=bool)
        mask[0] = False
        mask &= valid
        # Also need previous trial to be valid
        prev_valid = np.roll(valid, 1)
        prev_valid[0] = False
        mask &= prev_valid

        if mask.sum() < 5:
            return np.nan

        c_m = c[mask]
        prev_cat_m = prev_cat[mask]

        prev_was_b = prev_cat_m == 1
        prev_was_a = prev_cat_m == 0

        if prev_was_b.sum() == 0 or prev_was_a.sum() == 0:
            return np.nan

        p_b_after_b = np.mean(c_m[prev_was_b])
        p_b_after_a = np.mean(c_m[prev_was_a])

        return p_b_after_b - p_b_after_a

    choices = np.asarray(choices)

    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('win_stay')
def compute_win_stay_index(choices: np.ndarray, stimuli: np.ndarray,
                           categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Win-stay tendency: P(repeat | rewarded) - P(repeat | unrewarded).

    Positive = exploits correct responses. Near zero = ignores feedback.
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c)
        cat = _ensure_1d(cat)
        valid = _valid_trials(c)

        if valid.sum() < 10:
            return np.nan

        rewards = (c == cat).astype(float)
        rewards[~valid] = np.nan

        prev_choice = np.roll(c, 1)
        prev_reward = np.roll(rewards, 1)

        mask = np.ones(len(c), dtype=bool)
        mask[0] = False
        mask &= valid
        prev_valid = np.roll(valid, 1)
        prev_valid[0] = False
        mask &= prev_valid

        if mask.sum() < 5:
            return np.nan

        c_m = c[mask]
        pc_m = prev_choice[mask]
        pr_m = prev_reward[mask]

        repeat = (c_m == pc_m)
        won = pr_m == 1
        lost = pr_m == 0

        if won.sum() == 0 or lost.sum() == 0:
            return np.nan

        return float(np.mean(repeat[won]) - np.mean(repeat[lost]))

    choices = np.asarray(choices)

    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('win_stay_rate')
def compute_win_stay_rate(choices: np.ndarray, stimuli: np.ndarray,
                          categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Raw win-stay rate: P(repeat choice | previous trial rewarded).
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c)
        cat = _ensure_1d(cat)
        valid = _valid_trials(c)

        if valid.sum() < 10:
            return np.nan

        rewards = (c == cat).astype(float)
        rewards[~valid] = np.nan
        prev_choice = np.roll(c, 1)
        prev_reward = np.roll(rewards, 1)

        mask = np.ones(len(c), dtype=bool)
        mask[0] = False
        mask &= valid
        prev_valid = np.roll(valid, 1)
        prev_valid[0] = False
        mask &= prev_valid

        won = prev_reward[mask] == 1
        if won.sum() == 0:
            return np.nan

        return float(np.mean(c[mask][won] == prev_choice[mask][won]))

    choices = np.asarray(choices)
    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('lose_shift')
def compute_lose_shift_index(choices: np.ndarray, stimuli: np.ndarray,
                             categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Lose-shift tendency: P(switch | unrewarded).

    High = responsive to negative feedback. Low = perseverative.
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c)
        cat = _ensure_1d(cat)
        valid = _valid_trials(c)

        if valid.sum() < 10:
            return np.nan

        rewards = (c == cat).astype(float)
        rewards[~valid] = np.nan
        prev_choice = np.roll(c, 1)
        prev_reward = np.roll(rewards, 1)

        mask = np.ones(len(c), dtype=bool)
        mask[0] = False
        mask &= valid
        prev_valid = np.roll(valid, 1)
        prev_valid[0] = False
        mask &= prev_valid

        if mask.sum() < 5:
            return np.nan

        c_m = c[mask]
        pc_m = prev_choice[mask]
        pr_m = prev_reward[mask]

        switch = (c_m != pc_m)
        lost = pr_m == 0

        if lost.sum() == 0:
            return np.nan

        return float(np.mean(switch[lost]))

    choices = np.asarray(choices)

    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('choice_autocorr')
def compute_choice_autocorrelation(choices: np.ndarray, stimuli: np.ndarray,
                                   categories: np.ndarray, lag: int = 1) -> Union[float, np.ndarray]:
    """
    Choice autocorrelation at lag 1: correlation between choice_t and choice_{t-1}.
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        valid = _valid_trials(c)
        c_clean = c[valid]

        if len(c_clean) < lag + 10:
            return np.nan

        c_current = c_clean[lag:]
        c_lagged = c_clean[:-lag]

        if np.std(c_current) == 0 or np.std(c_lagged) == 0:
            return np.nan

        return float(np.corrcoef(c_current, c_lagged)[0, 1])

    choices = np.asarray(choices)

    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('side_bias')
def compute_side_bias(choices: np.ndarray, stimuli: np.ndarray,
                      categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Overall tendency to choose B: P(choose B) - 0.5.

    Positive = biased toward B, negative = biased toward A.
    """
    choices = np.asarray(choices)

    if _is_multisession(choices):
        return np.nanmean(choices, axis=0) - 0.5
    else:
        c = _ensure_1d(choices)
        valid = _valid_trials(c)
        if valid.sum() == 0:
            return np.nan
        return float(np.nanmean(c[valid]) - 0.5)


@register_stat('stimulus_sensitivity')
def compute_stimulus_sensitivity(choices: np.ndarray, stimuli: np.ndarray,
                                 categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Correlation between stimulus value and choice.

    High = choices driven by stimulus. Low = choices independent of stimulus.
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        valid = _valid_trials(c)

        if valid.sum() < 10:
            return np.nan

        c_v, s_v = c[valid], s[valid]
        if np.std(c_v) == 0 or np.std(s_v) == 0:
            return np.nan

        return float(np.corrcoef(c_v, s_v)[0, 1])

    choices = np.asarray(choices)
    stimuli = np.asarray(stimuli)

    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


# =============================================================================
# NEW STATISTICS (Phase 1 additions)
# =============================================================================

@register_stat('choice_entropy')
def compute_choice_entropy(choices: np.ndarray, stimuli: np.ndarray,
                           categories: np.ndarray,
                           n_bins: int = DEFAULT_N_BINS) -> Union[float, np.ndarray]:
    """
    Mean entropy of choice distribution across stimulus bins.

    High entropy = random/uncertain choices. Low entropy = deterministic.
    Computed as: mean over bins of H(choice | stimulus_bin).
    Normalised by log(2) so range is [0, 1].
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        valid = _valid_trials(c)

        if valid.sum() < 10:
            return np.nan

        c_v, s_v = c[valid], s[valid]
        bin_edges = np.linspace(-1, 1, n_bins + 1)
        bin_idx = np.clip(np.digitize(s_v, bin_edges) - 1, 0, n_bins - 1)

        entropies = []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() < 3:
                continue
            p = np.mean(c_v[mask])
            p = np.clip(p, 1e-10, 1 - 1e-10)
            h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            entropies.append(h)

        if len(entropies) == 0:
            return np.nan
        return float(np.mean(entropies))

    choices = np.asarray(choices)

    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('perseveration')
def compute_perseveration(choices: np.ndarray, stimuli: np.ndarray,
                          categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Perseveration index: excess same-choice repetition beyond stimulus prediction.

    Computed as: P(same choice as previous) - P(same choice | predicted by stimulus alone).
    The stimulus prediction comes from the overall P(B|stimulus_bin).
    Positive = animal repeats choices more than stimulus alone would predict.
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        valid = _valid_trials(c)

        if valid.sum() < 20:
            return np.nan

        c_v, s_v = c[valid], s[valid]

        if len(c_v) < 10:
            return np.nan

        # Observed repetition rate (excluding first valid trial)
        observed_repeat = float(np.mean(c_v[1:] == c_v[:-1]))

        # Predicted repetition rate from stimulus alone
        # Use binned P(B|stimulus) to compute expected repetition
        n_bins = 8
        bin_edges = np.linspace(-1, 1, n_bins + 1)
        bin_idx = np.clip(np.digitize(s_v, bin_edges) - 1, 0, n_bins - 1)

        # P(B) per bin
        p_b_bin = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                p_b_bin[b] = np.mean(c_v[mask])

        # Expected repeat rate: for consecutive pairs, P(both B) + P(both A)
        expected_repeats = []
        for t in range(1, len(c_v)):
            b_curr = bin_idx[t]
            b_prev = bin_idx[t - 1]
            if np.isnan(p_b_bin[b_curr]) or np.isnan(p_b_bin[b_prev]):
                continue
            p_repeat = (p_b_bin[b_curr] * p_b_bin[b_prev] +
                        (1 - p_b_bin[b_curr]) * (1 - p_b_bin[b_prev]))
            expected_repeats.append(p_repeat)

        if len(expected_repeats) == 0:
            return np.nan

        predicted_repeat = float(np.mean(expected_repeats))
        return observed_repeat - predicted_repeat

    choices = np.asarray(choices)

    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('logistic_history')
def compute_logistic_history_weights(
    choices: np.ndarray, stimuli: np.ndarray,
    categories: np.ndarray, n_back: int = 3,
    l2_penalty: float = LOGISTIC_L2_PENALTY,
) -> Union[Dict[str, float], np.ndarray]:
    """
    L2-regularised logistic regression of current choice on stimulus + trial history.

    Regressors: current_stimulus, prev_choice(1..n_back), prev_outcome(1..n_back).
    L2 penalty prevents weight explosion from complete/near-complete separation,
    which occurs when animals discriminate well or have strong side biases.

    The penalty is applied to all weights EXCEPT the intercept:
        loss = -log_likelihood + (l2_penalty / 2) * Î£ Î²_iÂ²

    Returns dict with:
        'w_stimulus': weight of current stimulus (sensitivity)
        'w_prev_choice_1'..'w_prev_choice_n': previous choice weights (perseveration)
        'w_prev_outcome_1'..'w_prev_outcome_n': previous outcome weights (win-stay/lose-shift)
        'history_decay': exponential decay rate of prev_choice weights (if n_back >= 2)
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        cat = _ensure_1d(cat).astype(float)
        valid = _valid_trials(c)

        result_keys = ['w_stimulus']
        for k in range(1, n_back + 1):
            result_keys.extend([f'w_prev_choice_{k}', f'w_prev_outcome_{k}'])
        result_keys.append('history_decay')
        nan_result = {k: np.nan for k in result_keys}

        if valid.sum() < 30:
            return nan_result

        c_v, s_v, cat_v = c[valid], s[valid], cat[valid]
        outcomes_v = (c_v == cat_v).astype(float)

        n = len(c_v)
        if n < n_back + 10:
            return nan_result

        # Build design matrix
        # Columns: stimulus, prev_choice_1..n, prev_outcome_1..n
        n_regressors = 1 + 2 * n_back
        X = np.zeros((n - n_back, n_regressors))
        y = c_v[n_back:]

        X[:, 0] = s_v[n_back:]  # current stimulus
        for k in range(1, n_back + 1):
            # Centre previous choices (0/1 -> -0.5/0.5)
            X[:, k] = c_v[n_back - k: n - k] - 0.5
            # Centre previous outcomes
            X[:, n_back + k] = outcomes_v[n_back - k: n - k] - 0.5

        # Add intercept
        X_full = np.column_stack([np.ones(len(y)), X])
        n_params = X_full.shape[1]

        # L2-regularised logistic regression
        # Penalty on all weights except intercept (index 0)
        try:
            from scipy.optimize import minimize

            def neg_ll_l2(beta):
                logits = X_full @ beta
                logits = np.clip(logits, -20, 20)
                p = 1 / (1 + np.exp(-logits))
                p = np.clip(p, 1e-10, 1 - 1e-10)
                nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
                # L2 penalty on non-intercept weights
                penalty = (l2_penalty / 2) * np.sum(beta[1:] ** 2)
                return nll + penalty

            def neg_ll_l2_grad(beta):
                logits = X_full @ beta
                logits = np.clip(logits, -20, 20)
                p = 1 / (1 + np.exp(-logits))
                p = np.clip(p, 1e-10, 1 - 1e-10)
                grad = -X_full.T @ (y - p)
                # L2 gradient (no penalty on intercept)
                grad[1:] += l2_penalty * beta[1:]
                return grad

            beta0 = np.zeros(n_params)
            res = minimize(neg_ll_l2, beta0, jac=neg_ll_l2_grad, method='L-BFGS-B')

            if not res.success:
                return nan_result

            beta = res.x
            # beta[0] = intercept, beta[1] = stimulus, beta[2..n_back+1] = prev_choice,
            # beta[n_back+2..2*n_back+1] = prev_outcome
            result = {'w_stimulus': float(beta[1])}
            prev_choice_weights = []
            for k in range(1, n_back + 1):
                result[f'w_prev_choice_{k}'] = float(beta[1 + k])
                prev_choice_weights.append(abs(float(beta[1 + k])))
                result[f'w_prev_outcome_{k}'] = float(beta[1 + n_back + k])

            # History decay: fit exponential to |prev_choice_weights|
            if n_back >= 2 and all(w > 1e-6 for w in prev_choice_weights):
                log_weights = np.log(np.array(prev_choice_weights))
                lags = np.arange(1, n_back + 1, dtype=float)
                # Linear fit: log(w) = a - decay * lag
                if np.std(lags) > 0:
                    slope = np.polyfit(lags, log_weights, 1)[0]
                    result['history_decay'] = float(-slope)
                else:
                    result['history_decay'] = np.nan
            else:
                result['history_decay'] = np.nan

            return result

        except Exception:
            return nan_result

    choices = np.asarray(choices)

    if _is_multisession(choices):
        # For multi-session: return per-key arrays
        n_sessions = choices.shape[1]
        all_results = []
        for i in range(n_sessions):
            all_results.append(
                _compute_single(choices[:, i], stimuli[:, i], categories[:, i])
            )
        # Combine into dict of arrays
        keys = all_results[0].keys()
        return {k: np.array([r[k] for r in all_results]) for k in keys}
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('hard_easy_ratio')
def compute_hard_easy_accuracy_ratio(
    choices: np.ndarray, stimuli: np.ndarray,
    categories: np.ndarray,
    hard_threshold: float = 0.3,
) -> Union[float, np.ndarray]:
    """
    Ratio of accuracy on hard trials (near boundary) to easy trials (far from boundary).

    Hard trials: |stimulus| < hard_threshold.
    Easy trials: |stimulus| >= hard_threshold.

    Low ratio = flat performance (guessing or not using boundary).
    High ratio approaching 1.0 = good discrimination even near boundary.
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        cat = _ensure_1d(cat).astype(float)
        valid = _valid_trials(c)

        if valid.sum() < 10:
            return np.nan

        c_v, s_v, cat_v = c[valid], s[valid], cat[valid]

        hard = np.abs(s_v) < hard_threshold
        easy = np.abs(s_v) >= hard_threshold

        if hard.sum() < 3 or easy.sum() < 3:
            return np.nan

        acc_hard = np.mean(c_v[hard] == cat_v[hard])
        acc_easy = np.mean(c_v[easy] == cat_v[easy])

        if acc_easy < 0.01:
            return np.nan

        return float(acc_hard / acc_easy)

    choices = np.asarray(choices)

    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('hard_accuracy')
def compute_hard_accuracy(
    choices: np.ndarray, stimuli: np.ndarray,
    categories: np.ndarray,
    hard_threshold: float = 0.3,
) -> Union[float, np.ndarray]:
    """
    Accuracy on hard trials only (|stimulus| < hard_threshold).
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        cat = _ensure_1d(cat).astype(float)
        valid = _valid_trials(c)

        if valid.sum() < 10:
            return np.nan

        c_v, s_v, cat_v = c[valid], s[valid], cat[valid]
        hard = np.abs(s_v) < hard_threshold

        if hard.sum() < 3:
            return np.nan

        return float(np.mean(c_v[hard] == cat_v[hard]))

    choices = np.asarray(choices)
    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('easy_accuracy')
def compute_easy_accuracy(
    choices: np.ndarray, stimuli: np.ndarray,
    categories: np.ndarray,
    hard_threshold: float = 0.3,
) -> Union[float, np.ndarray]:
    """
    Accuracy on easy trials only (|stimulus| >= hard_threshold).
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        cat = _ensure_1d(cat).astype(float)
        valid = _valid_trials(c)

        if valid.sum() < 10:
            return np.nan

        c_v, s_v, cat_v = c[valid], s[valid], cat[valid]
        easy = np.abs(s_v) >= hard_threshold

        if easy.sum() < 3:
            return np.nan

        return float(np.mean(c_v[easy] == cat_v[easy]))

    choices = np.asarray(choices)
    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


# =============================================================================
# UPDATE MATRIX (Conditional psychometry)
# =============================================================================

def _fit_pse_only(current_stim: np.ndarray, current_choices: np.ndarray,
                  sigma: float, lapse_low: float, lapse_high: float) -> float:
    """
    Fit PSE only, with slope and lapses fixed from the unconditional fit.

    Uses 1D optimisation over PSE (mu) to find the horizontal shift
    that best explains the conditioned choice data under the fixed
    psychometric shape.

    Args:
        current_stim: stimulus values for trials in this bin
        current_choices: choices for trials in this bin
        sigma: fixed slope from unconditional fit
        lapse_low: fixed lower lapse from unconditional fit
        lapse_high: fixed upper lapse from unconditional fit

    Returns:
        PSE estimate, or np.nan if fit fails
    """
    from scipy.optimize import minimize_scalar

    if len(current_stim) < 5:
        return np.nan

    def neg_ll(mu):
        p = cumulative_gaussian(current_stim, mu, sigma, lapse_low, lapse_high)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(current_choices * np.log(p) + (1 - current_choices) * np.log(1 - p))

    try:
        result = minimize_scalar(neg_ll, bounds=(-1.5, 1.5), method='bounded')
        if result.success:
            return float(result.x)
    except Exception:
        pass

    return np.nan


@register_stat('conditional_psychometric')
def compute_conditional_psychometric(
    choices: np.ndarray, stimuli: np.ndarray,
    categories: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    min_trials_per_bin: int = 15,
) -> Union[Dict[str, float], np.ndarray]:
    """
    Conditional psychometric curves: fit a full cumulative Gaussian for each
    previous-stimulus bin.

    Bins trials by the previous stimulus (n_bins bins), then fits a separate
    psychometric curve (mu, sigma, lapse_low, lapse_high) within each bin.

    Returns dict with keys:
        'cond_pse_0'..'cond_pse_7': PSE per previous-stimulus bin
        'cond_slope_0'..'cond_slope_7': slope per previous-stimulus bin
        'cond_lapse_low_0'..'cond_lapse_low_7': lower lapse per bin
        'cond_lapse_high_0'..'cond_lapse_high_7': upper lapse per bin
    Total: 4 * n_bins values.
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        valid = _valid_trials(c)

        # Build NaN result
        nan_result = {}
        for b in range(n_bins):
            nan_result[f'cond_pse_{b}'] = np.nan
            nan_result[f'cond_slope_{b}'] = np.nan
            nan_result[f'cond_lapse_low_{b}'] = np.nan
            nan_result[f'cond_lapse_high_{b}'] = np.nan

        if valid.sum() < 50:
            return nan_result

        c_v, s_v = c[valid], s[valid]

        # Unconditional fit as fallback (null: prev_stim has no effect)
        uncond = fit_psychometric(s_v, c_v)
        if not uncond.get('success', False):
            return nan_result
        fallback_pse = uncond['mu']
        fallback_slope = uncond['sigma']
        fallback_ll = uncond['lapse_low']
        fallback_lh = uncond['lapse_high']

        # Bin by previous stimulus (skip first trial - no previous)
        bin_edges = np.linspace(-1, 1, n_bins + 1)
        prev_stim = s_v[:-1]
        curr_stim = s_v[1:]
        curr_choices = c_v[1:]
        prev_bin_idx = np.clip(np.digitize(prev_stim, bin_edges) - 1, 0, n_bins - 1)

        result = {}
        for b in range(n_bins):
            mask = prev_bin_idx == b
            if mask.sum() < min_trials_per_bin:
                # Not enough trials: fall back to unconditional
                result[f'cond_pse_{b}'] = fallback_pse
                result[f'cond_slope_{b}'] = fallback_slope
                result[f'cond_lapse_low_{b}'] = fallback_ll
                result[f'cond_lapse_high_{b}'] = fallback_lh
                continue

            psych = fit_psychometric(curr_stim[mask], curr_choices[mask])
            if psych.get('success', False):
                pse_val = psych['mu']
                slope_val = psych['sigma']
                unreliable = (
                    slope_val > PSYCHOMETRIC_SLOPE_THRESHOLD
                    or abs(pse_val) > 0.99
                )
                if unreliable:
                    result[f'cond_pse_{b}'] = fallback_pse
                    result[f'cond_slope_{b}'] = fallback_slope
                else:
                    result[f'cond_pse_{b}'] = pse_val
                    result[f'cond_slope_{b}'] = slope_val
                result[f'cond_lapse_low_{b}'] = psych['lapse_low']
                result[f'cond_lapse_high_{b}'] = psych['lapse_high']
            else:
                # Fit failed: fall back to unconditional
                result[f'cond_pse_{b}'] = fallback_pse
                result[f'cond_slope_{b}'] = fallback_slope
                result[f'cond_lapse_low_{b}'] = fallback_ll
                result[f'cond_lapse_high_{b}'] = fallback_lh

        return result

    choices = np.asarray(choices)
    stimuli = np.asarray(stimuli)

    if _is_multisession(choices):
        n_sessions = choices.shape[1]
        all_results = []
        for i in range(n_sessions):
            all_results.append(
                _compute_single(choices[:, i], stimuli[:, i], categories[:, i])
            )
        keys = all_results[0].keys()
        return {k: np.array([r[k] for r in all_results]) for k in keys}
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('update_matrix')
def compute_update_matrix_stat(
    choices: np.ndarray, stimuli: np.ndarray,
    categories: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    trial_filter: str = 'post_correct',
) -> Union[Dict[str, float], np.ndarray]:
    """
    Empirical update matrix via canonical psychometric-fit method.

    Uses the methodology from behav_utils.analysis.update_matrix: fits a cumulative
    Gaussian per previous-stimulus bin, then computes the difference from
    the overall post-correct psychometric curve.

    Returns dict with keys 'um_i_j' for i in [0, n_bins), j in [0, n_bins),
    where i = current stimulus bin, j = previous stimulus bin.
    Total: n_bins * n_bins values (64 for default n_bins=8).
    """
    from behav_utils.analysis.update_matrix import compute_update_matrix as canonical_um

    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        cat = _ensure_1d(cat).astype(float)
        valid = _valid_trials(c)

        # Zero result: no detectable serial dependence
        zero_result = {}
        for i in range(n_bins):
            for j in range(n_bins):
                zero_result[f'um_{i}_{j}'] = 0.0

        if valid.sum() < 50:
            return zero_result

        try:
            um, _, _ = canonical_um(
                stimuli=s[valid],
                choices=c[valid],
                categories=cat[valid],
                n_bins=n_bins,
                trial_filter=trial_filter,
            )
        except Exception:
            return zero_result

        result = {}
        for i in range(n_bins):
            for j in range(n_bins):
                val = um[i, j]
                result[f'um_{i}_{j}'] = float(val) if not np.isnan(val) else 0.0

        return result

    choices = np.asarray(choices)
    stimuli = np.asarray(stimuli)
    categories = np.asarray(categories)

    if _is_multisession(choices):
        n_sessions = choices.shape[1]
        all_results = []
        for i in range(n_sessions):
            all_results.append(
                _compute_single(choices[:, i], stimuli[:, i], categories[:, i])
            )
        keys = all_results[0].keys()
        return {k: np.array([r[k] for r in all_results]) for k in keys}
    else:
        return _compute_single(choices, stimuli, categories)


# =============================================================================
# ADDITIONAL STATISTICS (Phase 2 additions)
# =============================================================================

@register_stat('psychometric_gof')
def compute_psychometric_gof_stat(
    choices: np.ndarray, stimuli: np.ndarray,
    categories: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Psychometric curve goodness-of-fit (R²).

    Fits a cumulative Gaussian to binned choice probabilities and returns
    the R² between observed binned proportions and the fitted curve.

    Tracks learning: naive sessions have low R² (noisy choices),
    expert sessions have high R² (choices follow psychometric curve).
    """
    from behav_utils.analysis.psychometry import compute_psychometric_gof

    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        valid = _valid_trials(c)

        if valid.sum() < 20:
            return np.nan

        psych = fit_psychometric(s[valid], c[valid])
        if not psych.get('success', False):
            return np.nan

        gof = compute_psychometric_gof(s[valid], c[valid], psych)
        return gof.get('r_squared', np.nan)

    choices = np.asarray(choices)

    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('stimulus_recency')
def compute_stimulus_recency(
    choices: np.ndarray, stimuli: np.ndarray,
    categories: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Effect of previous stimulus VALUE on current choice.

    Measures: P(choose B | prev_stim > 0) - P(choose B | prev_stim <= 0)

    Unlike 'recency' (which conditions on previous CATEGORY), this
    conditions on the continuous stimulus position, which maps more
    directly onto the BE model's update mechanism (boundary belief
    is shifted by perceived stimulus position, not category label).

    Positive = previous stimulus on the B side biases current choice
    toward B (assimilative serial dependence).
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        valid = _valid_trials(c)

        if valid.sum() < 10:
            return np.nan

        c_v, s_v = c[valid], s[valid]

        if len(c_v) < 5:
            return np.nan

        # Previous stimulus values (skip first trial)
        prev_stim = s_v[:-1]
        curr_choice = c_v[1:]

        prev_b_side = prev_stim > 0
        prev_a_side = prev_stim <= 0

        if prev_b_side.sum() == 0 or prev_a_side.sum() == 0:
            return np.nan

        p_b_after_high = np.mean(curr_choice[prev_b_side])
        p_b_after_low = np.mean(curr_choice[prev_a_side])

        return p_b_after_high - p_b_after_low

    choices = np.asarray(choices)

    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)

@register_stat('recency_divergence')
def compute_recency_divergence(
    choices: np.ndarray, stimuli: np.ndarray,
    categories: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Difference between stimulus-based and category-based recency.

    recency_divergence = stimulus_recency - recency

    For uniform distributions these are highly correlated (~0 divergence).
    After distribution shift they can diverge: positive = serial dependence
    is more sensory than categorical, negative = more categorical.
    """
    def _compute_single(c, s, cat):
        stim_rec = compute_stimulus_recency(c, s, cat)
        cat_rec = compute_recency_index(c, s, cat)
        if np.isnan(stim_rec) or np.isnan(cat_rec):
            return np.nan
        return stim_rec - cat_rec

    choices = np.asarray(choices)
    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('recency_divergence')
def compute_recency_divergence(
    choices: np.ndarray, stimuli: np.ndarray,
    categories: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Difference between stimulus-based and category-based recency.

    recency_divergence = stimulus_recency - recency

    Measures whether serial dependence is driven by the sensory
    position of the previous stimulus (stimulus_recency) or by its
    abstract category label (recency).

    For uniform distributions, these are highly correlated and
    divergence is ~0. After a distribution shift (e.g., stimuli
    concentrated on one side), they can diverge because category
    and stimulus magnitude become confounded.

    Positive divergence = serial dependence is more sensory than
    categorical (consistent with perceptual-level adaptation).
    Negative = more categorical (consistent with belief-level updating).
    """
    def _compute_single(c, s, cat):
        # Compute both recency measures
        stim_rec = compute_stimulus_recency(c, s, cat)
        cat_rec = compute_recency_index(c, s, cat)

        if np.isnan(stim_rec) or np.isnan(cat_rec):
            return np.nan

        return stim_rec - cat_rec

    choices = np.asarray(choices)

    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('history_interaction_r2')
def compute_history_interaction_r2(
    choices: np.ndarray, stimuli: np.ndarray,
    categories: np.ndarray, n_back: int = 3,
) -> Union[float, np.ndarray]:
    """
    How much does trial history improve choice prediction beyond stimulus?

    Computes McFadden's pseudo-R² for two logistic models:
        1. Stimulus-only: choice ~ stimulus
        2. Full: choice ~ stimulus + prev_choices + prev_outcomes

    Returns R²_full - R²_stimulus, i.e., the additional variance explained
    by trial history. High values = history-dependent behaviour (high η),
    low values = stimulus-driven behaviour (low η, expert).

    This collapses all logistic_history weights into a single interpretable
    number that is less sensitive to the L2 penalty issue.
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        cat = _ensure_1d(cat).astype(float)
        valid = _valid_trials(c)

        if valid.sum() < 30:
            return np.nan

        c_v, s_v, cat_v = c[valid], s[valid], cat[valid]
        outcomes_v = (c_v == cat_v).astype(float)

        n = len(c_v)
        if n < n_back + 10:
            return np.nan

        y = c_v[n_back:]
        n_obs = len(y)

        # --- Stimulus-only model ---
        X_stim = np.column_stack([
            np.ones(n_obs),
            s_v[n_back:],
        ])

        # --- Full model (stimulus + history) ---
        n_regressors = 1 + 2 * n_back
        X_hist = np.zeros((n_obs, n_regressors))
        X_hist[:, 0] = s_v[n_back:]
        for k in range(1, n_back + 1):
            X_hist[:, k] = c_v[n_back - k: n - k] - 0.5
            X_hist[:, n_back + k] = outcomes_v[n_back - k: n - k] - 0.5
        X_full = np.column_stack([np.ones(n_obs), X_hist])

        try:
            from scipy.optimize import minimize

            def neg_ll(beta, X):
                logits = X @ beta
                logits = np.clip(logits, -20, 20)
                p = 1 / (1 + np.exp(-logits))
                p = np.clip(p, 1e-10, 1 - 1e-10)
                return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

            # Null model log-likelihood (intercept only)
            p_bar = np.clip(np.mean(y), 1e-10, 1 - 1e-10)
            ll_null = n_obs * (p_bar * np.log(p_bar) + (1 - p_bar) * np.log(1 - p_bar))

            # Stimulus-only model
            beta0_stim = np.zeros(X_stim.shape[1])
            res_stim = minimize(neg_ll, beta0_stim, args=(X_stim,), method='L-BFGS-B')
            if not res_stim.success:
                return np.nan
            ll_stim = -res_stim.fun

            # Full model
            beta0_full = np.zeros(X_full.shape[1])
            res_full = minimize(neg_ll, beta0_full, args=(X_full,), method='L-BFGS-B')
            if not res_full.success:
                return np.nan
            ll_full = -res_full.fun

            # McFadden's pseudo-R² difference
            if ll_null == 0:
                return np.nan
            r2_stim = 1 - (ll_stim / ll_null)
            r2_full = 1 - (ll_full / ll_null)

            return float(r2_full - r2_stim)

        except Exception:
            return np.nan

    choices = np.asarray(choices)

    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('sd_profile')
def compute_sd_profile_features(
    choices: np.ndarray, stimuli: np.ndarray,
    categories: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> Union[Dict[str, float], np.ndarray]:
    """
    Scalar features from the serial dependence profile.

    Computes the serial dependence profile (mean update matrix column values)
    and extracts three scalar features:
        - sd_slope: Linear regression slope of profile vs bin centre.
          Captures overall direction/magnitude of serial dependence.
        - sd_curvature: Quadratic coefficient from polynomial fit.
          Captures the boundary-concentrated pattern (inverted-U shape)
          characteristic of the BE model. Magnitude scales with eta.
        - sd_range: Max - min of profile values.
          Captures total serial dependence magnitude regardless of shape.

    Uses a fast raw computation (no psychometric fitting) for efficiency
    in HMM/SBI pipelines. For publication-quality update matrices, use
    the 'update_matrix' stat or behav_utils.analysis.update_matrix directly.
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        cat = _ensure_1d(cat).astype(float)
        valid = _valid_trials(c)

        nan_result = {
            'sd_slope': np.nan,
            'sd_curvature': np.nan,
            'sd_range': np.nan,
        }

        if valid.sum() < 50:
            return nan_result

        c_v, s_v, cat_v = c[valid], s[valid], cat[valid]

        # Compute rewards for post-correct filtering
        rewards = (c_v == cat_v).astype(float)

        bin_edges = np.linspace(-1, 1, n_bins + 1)
        midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Previous and current trial pairing (skip first)
        prev_stim = s_v[:-1]
        curr_stim = s_v[1:]
        curr_choices = c_v[1:]
        prev_reward = rewards[:-1]

        # Post-correct filter
        mask = prev_reward == 1

        if mask.sum() < 30:
            return nan_result

        prev_stim_m = prev_stim[mask]
        curr_stim_m = curr_stim[mask]
        curr_choices_m = curr_choices[mask]

        prev_bin = np.clip(np.digitize(prev_stim_m, bin_edges) - 1, 0, n_bins - 1)
        curr_bin = np.clip(np.digitize(curr_stim_m, bin_edges) - 1, 0, n_bins - 1)

        # Marginal P(B | current_stim_bin)
        marginal_pB = np.full(n_bins, np.nan)
        for i in range(n_bins):
            m = curr_bin == i
            if m.sum() > 0:
                marginal_pB[i] = np.mean(curr_choices_m[m])

        # Serial dependence profile: for each previous-stimulus bin,
        # mean delta across all current-stimulus bins
        profile = np.full(n_bins, np.nan)
        for j in range(n_bins):
            prev_mask = prev_bin == j
            if prev_mask.sum() < 5:
                continue

            deltas = []
            for i in range(n_bins):
                cell_mask = prev_mask & (curr_bin == i)
                if cell_mask.sum() >= 3 and not np.isnan(marginal_pB[i]):
                    cond_pB = np.mean(curr_choices_m[cell_mask])
                    deltas.append(cond_pB - marginal_pB[i])

            if len(deltas) >= 3:
                profile[j] = np.mean(deltas)

        # Extract scalar features from profile
        valid_profile = ~np.isnan(profile)
        if valid_profile.sum() < 4:
            return nan_result

        x = midpoints[valid_profile]
        y = profile[valid_profile]

        # Slope: linear regression
        try:
            coeffs_1 = np.polyfit(x, y, 1)
            sd_slope = float(coeffs_1[0])
        except Exception:
            sd_slope = np.nan

        # Curvature: quadratic fit
        try:
            if valid_profile.sum() >= 5:
                coeffs_2 = np.polyfit(x, y, 2)
                sd_curvature = float(coeffs_2[0])
            else:
                sd_curvature = np.nan
        except Exception:
            sd_curvature = np.nan

        # Range
        sd_range = float(np.max(y) - np.min(y))

        return {
            'sd_slope': sd_slope,
            'sd_curvature': sd_curvature,
            'sd_range': sd_range,
        }

    choices = np.asarray(choices)

    if _is_multisession(choices):
        n_sessions = choices.shape[1]
        all_results = []
        for i in range(n_sessions):
            all_results.append(
                _compute_single(choices[:, i], stimuli[:, i], categories[:, i])
            )
        keys = all_results[0].keys()
        return {k: np.array([r[k] for r in all_results]) for k in keys}
    else:
        return _compute_single(choices, stimuli, categories)


def compute_conditional_psychometry_full(
    choices: np.ndarray, stimuli: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    min_trials_per_bin: int = 5,
) -> Dict:
    """
    Full conditional psychometry for exploratory plotting.

    NOT a registered stat (too large for feature matrix). Call directly
    on a single session for visualisation.

    Returns dict with:
        'pB_matrix': (n_bins, n_bins) array â€” P(choose B | prev_stim_bin, curr_stim_bin)
                     Rows = previous stimulus bin, columns = current stimulus bin.
                     This is the classic update matrix heatmap.
        'counts_matrix': (n_bins, n_bins) â€” trial counts per cell
        'pse_per_bin': (n_bins,) â€” conditional PSE per previous-stimulus bin
                       (PSE-only fit with fixed shape)
        'dpse_per_bin': (n_bins,) â€” Î”PSE = conditional PSE - unconditional PSE
        'psych_curves_per_bin': dict mapping bin_idx -> {'x': array, 'y': array}
                                fitted psychometric curve per prev-stim bin
                                (for overlay plotting)
        'pse_uncond': float â€” unconditional PSE
        'sigma': float â€” unconditional slope (fixed for conditional fits)
        'lapse_low': float â€” unconditional lower lapse
        'lapse_high': float â€” unconditional upper lapse
        'bin_edges': (n_bins+1,) array
        'bin_centres': (n_bins,) array
        'success': bool
    """
    choices = _ensure_1d(np.asarray(choices, dtype=float))
    stimuli = _ensure_1d(np.asarray(stimuli, dtype=float))

    valid = ~np.isnan(choices) & ~np.isnan(stimuli)
    c_v, s_v = choices[valid], stimuli[valid]

    bin_edges = np.linspace(-1, 1, n_bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    fail_result = {
        'pB_matrix': np.full((n_bins, n_bins), np.nan),
        'counts_matrix': np.zeros((n_bins, n_bins), dtype=int),
        'pse_per_bin': np.full(n_bins, np.nan),
        'dpse_per_bin': np.full(n_bins, np.nan),
        'psych_curves_per_bin': {},
        'pse_uncond': np.nan,
        'sigma': np.nan,
        'lapse_low': np.nan,
        'lapse_high': np.nan,
        'bin_edges': bin_edges,
        'bin_centres': bin_centres,
        'success': False,
    }

    if valid.sum() < 50:
        return fail_result

    # â”€â”€ Unconditional psychometric fit â”€â”€
    psych = fit_psychometric(s_v, c_v)
    if not psych.get('success', False):
        return fail_result

    sigma = psych['sigma']
    lapse_low = psych['lapse_low']
    lapse_high = psych['lapse_high']
    pse_uncond = psych['mu']

    if sigma > PSYCHOMETRIC_SLOPE_THRESHOLD or abs(pse_uncond) > 0.99:
        return fail_result

    # â”€â”€ Raw 8Ã—8 P(B) matrix â”€â”€
    prev_stim = s_v[:-1]
    curr_stim = s_v[1:]
    curr_choices = c_v[1:]

    prev_bin_idx = np.clip(np.digitize(prev_stim, bin_edges) - 1, 0, n_bins - 1)
    curr_bin_idx = np.clip(np.digitize(curr_stim, bin_edges) - 1, 0, n_bins - 1)

    pB_matrix = np.full((n_bins, n_bins), np.nan)
    counts_matrix = np.zeros((n_bins, n_bins), dtype=int)

    for pb in range(n_bins):
        for cb in range(n_bins):
            mask = (prev_bin_idx == pb) & (curr_bin_idx == cb)
            counts_matrix[pb, cb] = int(mask.sum())
            if mask.sum() >= min_trials_per_bin:
                pB_matrix[pb, cb] = float(np.mean(curr_choices[mask]))

    # â”€â”€ Per-bin conditional PSE (fixed shape) â”€â”€
    pse_per_bin = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = prev_bin_idx == b
        if mask.sum() >= max(min_trials_per_bin, 10):
            pse_per_bin[b] = _fit_pse_only(
                curr_stim[mask], curr_choices[mask],
                sigma, lapse_low, lapse_high,
            )

    dpse_per_bin = pse_per_bin - pse_uncond

    # â”€â”€ Per-bin psychometric curves (for overlay plotting) â”€â”€
    x_eval = np.linspace(-1, 1, 100)
    psych_curves = {}
    for b in range(n_bins):
        if not np.isnan(pse_per_bin[b]):
            y_eval = cumulative_gaussian(x_eval, pse_per_bin[b],
                                         sigma, lapse_low, lapse_high)
            psych_curves[b] = {'x': x_eval, 'y': y_eval}

    return {
        'pB_matrix': pB_matrix,
        'counts_matrix': counts_matrix,
        'pse_per_bin': pse_per_bin,
        'dpse_per_bin': dpse_per_bin,
        'psych_curves_per_bin': psych_curves,
        'pse_uncond': pse_uncond,
        'sigma': sigma,
        'lapse_low': lapse_low,
        'lapse_high': lapse_high,
        'bin_edges': bin_edges,
        'bin_centres': bin_centres,
        'success': True,
    }


# =============================================================================
# BINNED STATISTICS
# =============================================================================

@register_stat('binned_accuracy')
def compute_binned_accuracy(choices: np.ndarray, stimuli: np.ndarray,
                            categories: np.ndarray,
                            n_bins: int = DEFAULT_N_BINS) -> Union[np.ndarray, np.ndarray]:
    """
    Accuracy binned by stimulus value. Shape (n_bins,) or (n_bins, n_sessions).
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c)
        s = _ensure_1d(s)
        cat = _ensure_1d(cat)

        bin_edges = np.linspace(-1, 1, n_bins + 1)
        bin_indices = np.clip(np.digitize(s, bin_edges) - 1, 0, n_bins - 1)

        binned_acc = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                binned_acc[b] = np.mean(c[mask] == cat[mask])
            else:
                binned_acc[b] = np.nan

        return binned_acc

    choices = np.asarray(choices)
    stimuli = np.asarray(stimuli)
    categories = np.asarray(categories)

    if _is_multisession(choices):
        n_sessions = choices.shape[1]
        results = []
        for i in range(n_sessions):
            results.append(_compute_single(choices[:, i], stimuli[:, i], categories[:, i]))
        return np.array(results).T
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('binned_choice_prob')
def compute_binned_choice_probability(choices: np.ndarray, stimuli: np.ndarray,
                                      categories: np.ndarray,
                                      n_bins: int = DEFAULT_N_BINS) -> Union[np.ndarray, np.ndarray]:
    """
    P(choose B) binned by stimulus value (empirical psychometric curve).
    Shape (n_bins,) or (n_bins, n_sessions).
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c)
        s = _ensure_1d(s)

        bin_edges = np.linspace(-1, 1, n_bins + 1)
        bin_indices = np.clip(np.digitize(s, bin_edges) - 1, 0, n_bins - 1)

        binned_prob = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                binned_prob[b] = np.mean(c[mask])
            else:
                binned_prob[b] = np.nan

        return binned_prob

    choices = np.asarray(choices)
    stimuli = np.asarray(stimuli)

    if _is_multisession(choices):
        n_sessions = choices.shape[1]
        results = []
        for i in range(n_sessions):
            results.append(_compute_single(choices[:, i], stimuli[:, i], categories[:, i]))
        return np.array(results).T
    else:
        return _compute_single(choices, stimuli, categories)


# =============================================================================
# MAIN INTERFACE
# =============================================================================

# Default statistics for SBI (keep backwards compatible)
DEFAULT_STATS = ['accuracy', 'psychometric', 'recency', 'win_stay', 'stimulus_sensitivity']

# Extended set for session feature matrix
FEATURE_MATRIX_STATS = [
    'accuracy', 'psychometric', 'psychometric_gof',
    'recency', 'stimulus_recency', 'recency_divergence', 'recency_divergence',
    'win_stay', 'lose_shift',
    'stimulus_sensitivity', 'side_bias', 'choice_autocorr', 'choice_entropy',
    'perseveration', 'hard_easy_ratio', 'hard_accuracy', 'easy_accuracy',
    'history_interaction_r2', 'sd_profile',
    'logistic_history',
    'update_matrix',
    'conditional_psychometric',
]


def compute_summary_stats(
    choices: np.ndarray,
    stimuli: np.ndarray,
    categories: np.ndarray,
    stat_names: Optional[List[str]] = None,
    return_dict: bool = False
) -> Union[np.ndarray, Dict]:
    """
    Compute summary statistics.

    Args:
        choices: Binary choices, shape (n_trials,) or (n_trials, n_sessions)
        stimuli: Stimulus values, same shape as choices
        categories: True categories, same shape as choices
        stat_names: List of stat names to compute. If None, uses DEFAULT_STATS
        return_dict: If True, return dict; if False, return flattened array for SBI

    Returns:
        If return_dict=True: Dict mapping stat names to values
        If return_dict=False: 1D array of all stats concatenated (suitable for SBI)
    """
    if stat_names is None:
        stat_names = DEFAULT_STATS

    for name in stat_names:
        if name not in SUMMARY_REGISTRY:
            raise ValueError(f"Unknown stat: '{name}'. Available: {list_available_stats()}")

    results = {}
    for name in stat_names:
        func = SUMMARY_REGISTRY[name]
        results[name] = func(choices, stimuli, categories)

    if return_dict:
        return results

    return flatten_stats(results)


def flatten_stats(stats_dict: Dict) -> np.ndarray:
    """
    Flatten stats dict to 1D array for SBI.

    Handles scalars, arrays, and nested dicts (like psychometric params).
    """
    flat = []

    for name, value in stats_dict.items():
        if isinstance(value, dict):
            for k, v in value.items():
                v = np.atleast_1d(v)
                flat.extend(v.flatten())
        else:
            value = np.atleast_1d(value)
            flat.extend(value.flatten())

    return np.array(flat, dtype=np.float64)

def compute_summary_stats_per_session(
    session_data: Union[List[Dict[str, np.ndarray]], np.ndarray],
    stat_names: Optional[List[str]] = None,
    return_dict: bool = False,
) -> List[Union[np.ndarray, Dict]]:
    """
    Compute summary statistics for multiple sessions.

    Auto-detects input format:
        - List of dicts (variable-length sessions): loops per session
        - 2D arrays (equal-length sessions): vectorised batch computation

    The vectorised path is faster for SBI where all simulated sessions
    have the same trial count. The loop path handles real data where
    sessions have different lengths.

    Args:
        session_data: Either:
            - List of dicts with 'choices', 'stimuli', 'categories' keys
              (as from TrialData.get_arrays() or FittingData)
            - Dict with 'choices', 'stimuli', 'categories' as 2D arrays
              of shape (n_trials, n_sessions) — equal-length sessions
        stat_names: Which stats to compute (default: DEFAULT_STATS)
        return_dict: If True return list of dicts, else list of arrays

    Returns:
        List of results, one per session
    """
    if stat_names is None:
        stat_names = DEFAULT_STATS

    # Auto-detect format
    if isinstance(session_data, dict):
        # 2D arrays — vectorised path
        choices = np.asarray(session_data['choices'])
        stimuli = np.asarray(session_data['stimuli'])
        categories = np.asarray(session_data['categories'])

        if choices.ndim == 2:
            # Equal-length: use the vectorised multi-session path
            result = compute_summary_stats(
                choices, stimuli, categories,
                stat_names=stat_names,
                return_dict=return_dict,
            )
            if return_dict:
                # Result is a dict of arrays — convert to list of dicts
                n_sessions = choices.shape[1]
                keys = list(result.keys())
                return [
                    {k: (result[k][i] if hasattr(result[k], '__len__')
                         else result[k])
                     for k in keys}
                    for i in range(n_sessions)
                ]
            else:
                return result  # Already in the right format for SBI
        else:
            # 1D — single session, wrap in list
            return [compute_summary_stats(
                choices, stimuli, categories,
                stat_names=stat_names,
                return_dict=return_dict,
            )]

    elif isinstance(session_data, list):
        # List of dicts — variable-length, loop
        results = []
        for sa in session_data:
            if isinstance(sa, dict):
                choices = sa['choices']
                stimuli = sa['stimuli']
                categories = sa['categories']
            else:
                # Assume tuple (choices, stimuli, categories)
                choices, stimuli, categories = sa[0], sa[1], sa[2]

            results.append(compute_summary_stats(
                choices, stimuli, categories,
                stat_names=stat_names,
                return_dict=return_dict,
            ))
        return results

    else:
        raise TypeError(
            f"session_data must be a list of dicts or a dict of 2D arrays, "
            f"got {type(session_data)}"
        )



def get_stat_names_expanded(stat_names: Optional[List[str]] = None) -> List[str]:
    """
    Get expanded list of stat names (for labelling flattened array).

    Handles nested stats like 'psychometric' which expands to 4 params.
    """
    if stat_names is None:
        stat_names = DEFAULT_STATS

    expanded = []
    for name in stat_names:
        if name == 'psychometric':
            expanded.extend(['pse', 'slope', 'lapse_low', 'lapse_high'])
        elif name == 'binned_accuracy':
            expanded.extend([f'binned_acc_{i}' for i in range(DEFAULT_N_BINS)])
        elif name == 'binned_choice_prob':
            expanded.extend([f'binned_prob_{i}' for i in range(DEFAULT_N_BINS)])
        elif name == 'logistic_history':
            expanded.extend(['w_stimulus', 'w_prev_choice_1', 'w_prev_outcome_1',
                             'w_prev_choice_2', 'w_prev_outcome_2',
                             'w_prev_choice_3', 'w_prev_outcome_3',
                             'history_decay'])
        elif name == 'conditional_psychometric':
            for b in range(DEFAULT_N_BINS):
                expanded.extend([
                    f'cond_pse_{b}', f'cond_slope_{b}',
                    f'cond_lapse_low_{b}', f'cond_lapse_high_{b}',
                ])
        elif name == 'update_matrix':
            for i in range(DEFAULT_N_BINS):
                for j in range(DEFAULT_N_BINS):
                    expanded.append(f'um_{i}_{j}')
        elif name == 'sd_profile':
            expanded.extend(['sd_slope', 'sd_curvature', 'sd_range'])
        else:
            expanded.append(name)

    return expanded


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_stats_for_sbi(
    choices: np.ndarray,
    stimuli: np.ndarray,
    categories: np.ndarray,
    stat_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Convenience function for SBI simulator.
    Returns flattened 1D array of summary statistics.
    """
    return compute_summary_stats(choices, stimuli, categories, stat_names, return_dict=False)


def describe_stats(stat_names: Optional[List[str]] = None) -> None:
    """Print descriptions of summary statistics."""
    if stat_names is None:
        stat_names = list_available_stats()

    print("Summary Statistics")
    print("=" * 60)

    for name in stat_names:
        if name in SUMMARY_REGISTRY:
            func = SUMMARY_REGISTRY[name]
            doc = func.__doc__ or "No description"
            first_line = doc.strip().split('\n')[0]
            print(f"\n{name}:")
            print(f"  {first_line}")


def add_custom_stat(name: str, func: Callable) -> None:
    """
    Add a custom summary statistic.

    The function must have signature:
        func(choices, stimuli, categories) -> scalar, array, or dict

    Example:
        def my_stat(choices, stimuli, categories):
            return np.mean(choices) * np.std(stimuli)

        add_custom_stat('my_stat', my_stat)
    """
    SUMMARY_REGISTRY[name] = func
    print(f"Registered custom stat: '{name}'")
