"""
Fit engines on raw arrays: the cumulative-Gaussian psychometric fit, its goodness of fit, the update
matrix and the matrix error. ``readouts`` wraps them into dataclasses, ``stats`` exposes their scalar
outputs; both import from here, nothing here imports from them.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def cumulative_gaussian(x: np.ndarray, mu: float, sigma: float,
                        lapse_low: float = 0.0, lapse_high: float = 0.0) -> np.ndarray:
    """
    Compute cumulative Gaussian psychometric function.

    Args:
        x: Stimulus values
        mu: Mean (PSE - point of subjective equality)
        sigma: Standard deviation (slope)
        lapse_low: Lower lapse rate (guess rate for category A)
        lapse_high: Upper lapse rate (lapse rate for category B)

    Returns:
        P(choose B) for each stimulus value
    """
    x = np.asarray(x, dtype=np.float64)
    return lapse_low + (1 - lapse_low - lapse_high) * norm.cdf(x, mu, sigma)


def _neg_log_likelihood_psychometric(params: List[float], stimuli: np.ndarray,
                                      choices: np.ndarray) -> float:
    """Negative log-likelihood for psychometric curve fitting."""
    mu, sigma, lapse_low, lapse_high = params

    y = cumulative_gaussian(stimuli, mu, sigma, lapse_low, lapse_high)
    eps = np.finfo(float).eps
    y = np.clip(y, eps, 1 - eps)

    log_lik = choices * np.log(y) + (1 - choices) * np.log(1 - y)
    return -np.sum(log_lik)


def _fit_psychometric_once(stimuli: np.ndarray, choices: np.ndarray,
                           x_eval: np.ndarray) -> Dict:
    """
    Single psychometric fit (helper for bootstrap).

    Returns dict with parameters or NaNs if fit fails.

    Note: We accept the optimiser result even if ``result.success`` is
    False, provided the parameters are finite.  L-BFGS-B often reports
    failure on flat likelihood surfaces (e.g. chance-level performance)
    even though it found a perfectly usable set of parameters.  Only
    truly degenerate outputs (NaN / inf) are rejected.
    """
    if len(stimuli) < 10:
        return {
            'mu': np.nan, 'sigma': np.nan,
            'lapse_low': np.nan, 'lapse_high': np.nan,
            'success': False
        }

    # Initial guess and bounds
    # p0 = [0.0, 0.3, 0.05, 0.05]
    p0 = [0.0, 1.0, 0.05, 0.05]
    bounds = [(-1.0, 1.0), (0.01, 10.0), (0.0, 0.5), (0.0, 0.5)]

    try:
        result = minimize(
            _neg_log_likelihood_psychometric,
            p0, args=(stimuli, choices),
            bounds=bounds, method='L-BFGS-B'
        )

        mu, sigma, lapse_low, lapse_high = result.x

        # Reject only if parameters are actually degenerate
        if np.any(np.isnan(result.x)) or np.any(np.isinf(result.x)):
            return {
                'mu': np.nan, 'sigma': np.nan,
                'lapse_low': np.nan, 'lapse_high': np.nan,
                'success': False
            }

        y_fit = cumulative_gaussian(x_eval, mu, sigma, lapse_low, lapse_high)

        return {
            'mu': mu,
            'sigma': sigma,
            'lapse_low': lapse_low,
            'lapse_high': lapse_high,
            'x_fit': x_eval,
            'y_fit': y_fit,
            'nll': result.fun,
            'success': True,
            'optimizer_converged': result.success,
        }
    except (ValueError, RuntimeError):
        pass

    return {
        'mu': np.nan, 'sigma': np.nan,
        'lapse_low': np.nan, 'lapse_high': np.nan,
        'success': False
    }


def fit_psychometric(stimuli: np.ndarray, choices: np.ndarray,
                     x_eval: np.ndarray | None = None,
                     n_bootstrap: int = 0, seed: int = 42) -> Dict:
    """
    Fit psychometric curve to choice data.

    Args:
        stimuli: Array of stimulus values
        choices: Array of binary choices (0 = A, 1 = B)
        x_eval: Points at which to evaluate fitted curve (default: linspace(-1, 1, 100))
        n_bootstrap: Number of bootstrap samples for confidence intervals (0 = no bootstrap)
        seed: Random seed for bootstrap

    Returns:
        Dict with:
            'mu': PSE (point of subjective equality)
            'sigma': Slope (smaller = steeper)
            'lapse_low': Lower lapse rate (floor)
            'lapse_high': Upper lapse rate (1 - ceiling)
            'x_fit': Evaluation points
            'y_fit': Fitted curve values
            'nll': Negative log-likelihood
            'success': Whether fit succeeded

        If n_bootstrap > 0, also includes:
            'mu_ci': (lower, upper) 95% CI for mu
            'sigma_ci': (lower, upper) 95% CI for sigma
            'lapse_low_ci': (lower, upper) 95% CI for lapse_low
            'lapse_high_ci': (lower, upper) 95% CI for lapse_high
            'y_fit_ci': (lower, upper) curves for 95% CI band
            'bootstrap_params': DataFrame with all bootstrap parameter values
    """
    stimuli = np.asarray(stimuli, dtype=np.float64)
    choices = np.asarray(choices, dtype=np.float64)

    # Remove NaNs
    valid = ~np.isnan(stimuli) & ~np.isnan(choices)
    stimuli = stimuli[valid]
    choices = choices[valid]

    if x_eval is None:
        x_eval = np.linspace(-1, 1, 100)

    # Fit on original data
    result = _fit_psychometric_once(stimuli, choices, x_eval)

    if not result['success']:
        return result

    # Bootstrap if requested
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        n_trials = len(stimuli)

        boot_params = {
            'mu': [], 'sigma': [], 'lapse_low': [], 'lapse_high': []
        }
        boot_curves = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            idx = rng.choice(n_trials, size=n_trials, replace=True)
            boot_stim = stimuli[idx]
            boot_choices = choices[idx]

            boot_fit = _fit_psychometric_once(boot_stim, boot_choices, x_eval)

            if boot_fit['success']:
                for key in ['mu', 'sigma', 'lapse_low', 'lapse_high']:
                    boot_params[key].append(boot_fit[key])
                boot_curves.append(boot_fit['y_fit'])

        # Compute CIs (2.5th and 97.5th percentiles)
        for key in ['mu', 'sigma', 'lapse_low', 'lapse_high']:
            values = np.array(boot_params[key])
            if len(values) > 0:
                result[f'{key}_ci'] = (np.percentile(values, 2.5), np.percentile(values, 97.5))
                result[f'{key}_se'] = np.std(values)
            else:
                result[f'{key}_ci'] = (np.nan, np.nan)
                result[f'{key}_se'] = np.nan

        # Curve CI band
        if len(boot_curves) > 0:
            boot_curves = np.array(boot_curves)
            result['y_fit_ci'] = (
                np.percentile(boot_curves, 2.5, axis=0),
                np.percentile(boot_curves, 97.5, axis=0)
            )
            lo, hi = result.get('y_fit_ci', (None, None))
            result['curve_band'] = {
                'x':      result['x_fit'],
                'median': result['y_fit'],
                'lo':     lo,
                'hi':     hi,
            }
        else:
            result['y_fit_ci'] = (None, None)

        # Store all bootstrap values for further analysis
        result['bootstrap_params'] = {k: np.array(v) for k, v in boot_params.items()}
        result['n_bootstrap_success'] = len(boot_params['mu'])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fit_psychometric_gof(stimuli: np.ndarray, choices: np.ndarray,
                             psych_params: Dict, n_bins: int = 8) -> Dict:
    """
    Compute goodness-of-fit metrics for psychometric curve.

    Args:
        stimuli: Stimulus values
        choices: Binary choices
        psych_params: Fitted psychometric parameters (from fit_psychometric)
        n_bins: Number of bins for binned metrics

    Returns:
        Dict with:
            'r_squared': RÃ‚Â² between binned data and fitted curve
            'deviance': Binomial deviance
            'deviance_explained': Fraction of null deviance explained
            'rmse': Root mean squared error (binned)
            'mae': Mean absolute error (binned)
            'log_likelihood': Log-likelihood of fit
            'aic': Akaike information criterion
            'bic': Bayesian information criterion
    """
    stimuli = np.asarray(stimuli)
    choices = np.asarray(choices)

    # Remove NaNs
    valid = ~np.isnan(stimuli) & ~np.isnan(choices)
    stimuli = stimuli[valid]
    choices = choices[valid]
    n_total = len(choices)

    if not psych_params.get('success', False) or n_total < 10:
        return {
            'r_squared': np.nan,
            'deviance': np.nan,
            'deviance_explained': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'log_likelihood': np.nan,
            'aic': np.nan,
            'bic': np.nan
        }

    mu = psych_params['mu']
    sigma = psych_params['sigma']
    lapse_low = psych_params['lapse_low']
    lapse_high = psych_params['lapse_high']

    # --- Trial-level metrics ---
    # Predicted probability for each trial
    p_pred = cumulative_gaussian(stimuli, mu, sigma, lapse_low, lapse_high)
    p_pred = np.clip(p_pred, 1e-10, 1 - 1e-10)

    # Log-likelihood
    log_lik = np.sum(choices * np.log(p_pred) + (1 - choices) * np.log(1 - p_pred))

    # Null model (just mean)
    p_null = np.mean(choices)
    p_null = np.clip(p_null, 1e-10, 1 - 1e-10)
    log_lik_null = np.sum(choices * np.log(p_null) + (1 - choices) * np.log(1 - p_null))

    # Deviance
    deviance = -2 * log_lik
    deviance_null = -2 * log_lik_null
    deviance_explained = 1 - (deviance / deviance_null) if deviance_null != 0 else np.nan

    # AIC/BIC (4 parameters: mu, sigma, lapse_low, lapse_high)
    n_params = 4
    aic = 2 * n_params - 2 * log_lik
    bic = n_params * np.log(n_total) - 2 * log_lik

    # --- Binned metrics ---
    bin_edges = np.linspace(-1, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(stimuli, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    prop_observed = np.zeros(n_bins)
    prop_predicted = np.zeros(n_bins)
    valid_bins = np.zeros(n_bins, dtype=bool)

    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            prop_observed[b] = np.mean(choices[mask])
            prop_predicted[b] = cumulative_gaussian(bin_centers[b], mu, sigma, lapse_low, lapse_high)
            valid_bins[b] = True

    # RÃ‚Â² on binned data
    if np.sum(valid_bins) > 1:
        ss_res = np.sum((prop_observed[valid_bins] - prop_predicted[valid_bins])**2)
        ss_tot = np.sum((prop_observed[valid_bins] - np.mean(prop_observed[valid_bins]))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    else:
        r_squared = np.nan

    # RMSE and MAE on binned data
    rmse = np.sqrt(np.mean((prop_observed[valid_bins] - prop_predicted[valid_bins])**2))
    mae = np.mean(np.abs(prop_observed[valid_bins] - prop_predicted[valid_bins]))

    return {
        'r_squared': r_squared,
        'deviance': deviance,
        'deviance_explained': deviance_explained,
        'rmse': rmse,
        'mae': mae,
        'log_likelihood': log_lik,
        'aic': aic,
        'bic': bic,
        'n_trials': n_total
    }



def fit_update_matrix(
    stimuli: np.ndarray,
    choices: np.ndarray,
    categories: np.ndarray,
    n_bins: int = 8,
    trial_filter: Literal['all', 'post_correct'] = 'post_correct',
    no_response: np.ndarray | None = None,
    not_blockstart: np.ndarray | None = None,
    prev_stimuli: np.ndarray | None = None,
    prev_choices: np.ndarray | None = None,
    prev_categories: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute update matrix from raw behavioural arrays.

    The update matrix captures serial dependence: how does the previous
    trial's stimulus shift the current psychometric curve?

    Args:
        stimuli: Stimulus values for each trial.
        choices: Binary choices (0=A, 1=B).
        categories: True categories (0=A, 1=B).
        n_bins: Number of bins for stimulus discretisation.
        trial_filter: 'post_correct' (only after correct) or 'all'.
        no_response: Bool array (True = no response). Inferred from NaN if None.
        not_blockstart: Bool array (True = not start of block). Auto if None.
        prev_stimuli, prev_choices, prev_categories: Frozen,
            abort-aware lag-1 arrays aligned to every trial. If prev_stimuli is
            given, the previous trial is taken from these (NOT from array
            adjacency), so the matrix is correct on a non-consecutive subset
            (e.g. opto-only or post-opto trials). If None, the previous trial is
            the immediately preceding array element via not_blockstart (the
            simulated / SBI path, unchanged).

    Returns:
        update_matrix: (n_bins, n_bins) shift in P(B)
        conditional_matrix: (n_bins, n_bins) conditional P(B) values
        info: Dict with fitting details
    """
    stimuli = np.asarray(stimuli, dtype=np.float64)
    choices = np.asarray(choices, dtype=np.float64)
    categories = np.asarray(categories, dtype=np.float64)
    n_trials = len(stimuli)

    if no_response is None:
        no_response = np.isnan(choices)
    else:
        no_response = np.asarray(no_response, dtype=bool)

    bin_edges = np.linspace(-1, 1, n_bins + 1)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    if prev_stimuli is not None:
        # SESSION PATH: previous trial from the frozen, abort-aware lag-1 view.
        # Current trial = every trial; valid pairs gated by has_prev. Correct on
        # a non-consecutive subset (opto-only / post-opto), where array adjacency
        # would otherwise give the wrong predecessor.
        prev_stimuli = np.asarray(prev_stimuli, dtype=np.float64)
        prev_choices = np.asarray(prev_choices, dtype=np.float64)
        prev_categories = np.asarray(prev_categories, dtype=np.float64)

        curr_stim = stimuli
        curr_choice = choices
        prev_bin = np.clip(np.digitize(prev_stimuli, bin_edges) - 1, 0, n_bins - 1)
        prev_reward = (prev_choices == prev_categories)   # mirrors rewards, on prev
        curr_responded = ~no_response
        prev_responded = ~np.isnan(prev_choices)

        if trial_filter == 'post_correct':
            base = prev_reward & curr_responded & prev_responded
        elif trial_filter == 'all':
            base = curr_responded & prev_responded
        else:
            raise ValueError(f"trial_filter must be 'post_correct' or 'all', got '{trial_filter}'")
    else:
        # ADJACENCY PATH: previous trial = the immediately preceding array
        # element (simulated / SBI arrays, which carry no prev_trial view).
        if not_blockstart is None:
            not_blockstart = np.ones(n_trials, dtype=bool)
            if n_trials > 0:
                not_blockstart[0] = False
        else:
            not_blockstart = np.asarray(not_blockstart, dtype=bool)

        rewards = (choices == categories).astype(float)
        rewards[np.isnan(choices)] = np.nan
        bin_indices = np.clip(np.digitize(stimuli, bin_edges) - 1, 0, n_bins - 1)

        curr_stim = stimuli[1:]
        curr_choice = choices[1:]
        prev_bin = bin_indices[:-1]
        curr_responded = ~no_response[1:]
        prev_responded = ~no_response[:-1]
        is_not_blockstart = not_blockstart[1:]

        if trial_filter == 'post_correct':
            prev_correct = rewards[:-1] == 1
            base = prev_correct & curr_responded & prev_responded & is_not_blockstart
        elif trial_filter == 'all':
            base = curr_responded & prev_responded & is_not_blockstart
        else:
            raise ValueError(f"trial_filter must be 'post_correct' or 'all', got '{trial_filter}'")

    total_stimuli = curr_stim[base]
    total_choices = curr_choice[base]
    total_psych = fit_psychometric(total_stimuli, total_choices, midpoints)

    total_curve = total_psych['y_fit'] if total_psych['success'] else np.full(n_bins, np.nan)

    conditional_matrix = np.zeros((n_bins, n_bins))
    update_matrix = np.zeros((n_bins, n_bins))
    bin_counts = np.zeros(n_bins, dtype=int)
    conditional_psychs = []

    for j in range(n_bins):
        prev_in_bin = prev_bin == j
        condition = base & prev_in_bin
        cond_stimuli = curr_stim[condition]
        cond_choices = curr_choice[condition]
        bin_counts[j] = len(cond_stimuli)

        if len(cond_stimuli) < 10:
            conditional_matrix[:, j] = np.nan
            update_matrix[:, j] = np.nan
            conditional_psychs.append(None)
        else:
            cond_psych = fit_psychometric(cond_stimuli, cond_choices, midpoints)
            conditional_psychs.append(cond_psych)
            if cond_psych['success']:
                conditional_matrix[:, j] = cond_psych['y_fit']
                update_matrix[:, j] = cond_psych['y_fit'] - total_curve
            else:
                conditional_matrix[:, j] = np.nan
                update_matrix[:, j] = np.nan

    info = {
        'total_psychometric': total_psych,
        'conditional_psychometrics': conditional_psychs,
        'bin_edges': bin_edges,
        'midpoints': midpoints,
        'bin_counts': bin_counts,
        'total_trials': len(total_stimuli),
        'trial_filter': trial_filter,
        'total_curve': total_curve,
    }
    return update_matrix, conditional_matrix, info


def matrix_error(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """Mean squared error between two matrices, ignoring NaNs."""
    diff = matrix1 - matrix2
    valid = ~np.isnan(diff)
    if np.sum(valid) == 0:
        return np.nan
    return np.mean(diff[valid] ** 2)


# =============================================================================
# SESSION-LEVEL (NO FILTERING — data must be pre-filtered)
# =============================================================================
