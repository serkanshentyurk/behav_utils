from behav_utils.analysis.utils import cumulative_gaussian

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import warnings

from scipy.optimize import minimize


def _neg_log_likelihood_psychometric(params: List[float], stimulus: np.ndarray,
                                      choices: np.ndarray) -> float:
    """Negative log-likelihood for psychometric curve fitting."""
    mu, sigma, lapse_low, lapse_high = params
    
    y = cumulative_gaussian(stimulus, mu, sigma, lapse_low, lapse_high)
    eps = np.finfo(float).eps
    y = np.clip(y, eps, 1 - eps)
    
    log_lik = choices * np.log(y) + (1 - choices) * np.log(1 - y)
    return -np.sum(log_lik)


def _fit_psychometric_once(stimulus: np.ndarray, choices: np.ndarray,
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
    if len(stimulus) < 10:
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
            p0, args=(stimulus, choices),
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


def fit_psychometric(stimulus: np.ndarray, choices: np.ndarray,
                     x_eval: Optional[np.ndarray] = None,
                     n_bootstrap: int = 0, seed: int = 42) -> Dict:
    """
    Fit psychometric curve to choice data.
    
    Args:
        stimulus: Array of stimulus values
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
    stimulus = np.asarray(stimulus, dtype=np.float64)
    choices = np.asarray(choices, dtype=np.float64)
    
    # Remove NaNs
    valid = ~np.isnan(stimulus) & ~np.isnan(choices)
    stimulus = stimulus[valid]
    choices = choices[valid]
    
    if x_eval is None:
        x_eval = np.linspace(-1, 1, 100)
    
    # Fit on original data
    result = _fit_psychometric_once(stimulus, choices, x_eval)
    
    if not result['success']:
        return result
    
    # Bootstrap if requested
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        n_trials = len(stimulus)
        
        boot_params = {
            'mu': [], 'sigma': [], 'lapse_low': [], 'lapse_high': []
        }
        boot_curves = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            idx = rng.choice(n_trials, size=n_trials, replace=True)
            boot_stim = stimulus[idx]
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
        else:
            result['y_fit_ci'] = (None, None)
        
        # Store all bootstrap values for further analysis
        result['bootstrap_params'] = {k: np.array(v) for k, v in boot_params.items()}
        result['n_bootstrap_success'] = len(boot_params['mu'])
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_stim_choices(session) -> tuple:
    """Extract (stimuli, choices) arrays from a pre-filtered SessionData."""
    arr = session.get_arrays()
    valid = ~arr['no_response']
    return arr['stimuli'][valid], arr['choices'][valid]


def _bin_data(stimuli, choices, n_bins=8):
    """Bin stimuli and compute mean P(B) per bin."""
    edges = np.linspace(-1, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (stimuli >= edges[i]) & (stimuli < edges[i + 1])
        else:
            mask = (stimuli >= edges[i]) & (stimuli <= edges[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            means[i] = np.mean(choices[mask])
    return centres, means, counts


def _bootstrap_ci(stimuli, choices, n_bootstrap, n_bins=8, seed=42):
    """Bootstrap CI for binned P(B)."""
    rng = np.random.default_rng(seed)
    n = len(stimuli)
    boot_curves = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        _, means, _ = _bin_data(stimuli[idx], choices[idx], n_bins)
        boot_curves.append(means)
    if not boot_curves:
        return None, None
    boot_arr = np.array(boot_curves)
    return (
        np.nanpercentile(boot_arr, 2.5, axis=0),
        np.nanpercentile(boot_arr, 97.5, axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session-level psychometric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_psychometric(
    data,
    mode: str = 'pooled',
    n_bins: int = 8,
    n_bootstrap: int = 0,
) -> Dict:
    """
    Compute psychometric curve result from pre-filtered sessions.

    This is the session-level entry point. It calls the low-level
    fit_psychometric(stimuli, choices) internally and packages results
    into a structured dict ready for plot_psychometric().

    Args:
        data: Pre-filtered SessionData, List[SessionData], AnimalData,
              or (stimuli, choices) tuple.
        mode: 'pooled' (default), 'overlay', or 'session_mean'.
              Only affects multi-session data.
        n_bins: Number of stimulus bins for data points.
        n_bootstrap: Bootstrap iterations for CI (0 = off).

    Returns:
        Dict with 'mode' key and mode-specific fields.
        Pass directly to plot_psychometric().
    """
    from behav_utils.analysis.psychometry import fit_psychometric
    from behav_utils.analysis.utils import cumulative_gaussian

    sessions = _resolve_to_sessions(data)
    x_fit = np.linspace(-1, 1, 200)

    if mode == 'pooled':
        return _compute_pooled(
            sessions, x_fit, n_bins, n_bootstrap,
            fit_psychometric, cumulative_gaussian)

    elif mode == 'overlay':
        return _compute_overlay(
            sessions, x_fit, n_bins,
            fit_psychometric, cumulative_gaussian)

    elif mode == 'session_mean':
        return _compute_session_mean(
            sessions, x_fit, n_bins, n_bootstrap,
            fit_psychometric, cumulative_gaussian)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'pooled', 'overlay', or 'session_mean'.")


def _resolve_to_sessions(data):
    """Normalise input to a list of SessionData."""
    from behav_utils.data.structures import SessionData, AnimalData

    if isinstance(data, tuple) and len(data) == 2:
        # (stimuli, choices) tuple — wrap in a fake structure
        return [data]
    if isinstance(data, SessionData):
        return [data]
    if isinstance(data, AnimalData):
        return list(data.sessions)
    if isinstance(data, (list, tuple)):
        return list(data)
    raise TypeError(f"Expected SessionData/List/AnimalData/(stim,ch) tuple, got {type(data).__name__}")


def _get_stim_choices(item):
    """Extract (stimuli, choices) from a session or raw tuple."""
    if isinstance(item, tuple) and len(item) == 2:
        return np.asarray(item[0]), np.asarray(item[1])
    return _extract_stim_choices(item)


def _compute_pooled(sessions, x_fit, n_bins, n_bootstrap, fit_fn, cg_fn):
    """Pool all trials, fit one curve."""
    all_stim, all_ch = [], []
    for s in sessions:
        st, ch = _get_stim_choices(s)
        if len(st) > 0:
            all_stim.append(st)
            all_ch.append(ch)

    if not all_stim:
        return {'mode': 'pooled', 'params': {}, 'n_trials': 0}

    stimuli = np.concatenate(all_stim)
    choices = np.concatenate(all_ch)
    n_trials = len(stimuli)

    params = fit_fn(stimuli, choices)
    centres, means, counts = _bin_data(stimuli, choices, n_bins)

    y_fit = None
    if params.get('success', False) and not np.isnan(params.get('mu', np.nan)):
        y_fit = cg_fn(x_fit, params['mu'], params['sigma'],
                       params['lapse_low'], params['lapse_high'])

    ci_lo, ci_hi = None, None
    if n_bootstrap > 0:
        ci_lo, ci_hi = _bootstrap_ci(stimuli, choices, n_bootstrap, n_bins)

    return {
        'mode': 'pooled',
        'params': params,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'bin_centres': centres,
        'bin_means': means,
        'bin_counts': counts,
        'n_trials': n_trials,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
    }


def _compute_overlay(sessions, x_fit, n_bins, fit_fn, cg_fn):
    """Fit per-session curves."""
    per_session = []
    for s in sessions:
        st, ch = _get_stim_choices(s)
        if len(st) < 10:
            per_session.append({'params': {}, 'n_trials': len(st)})
            continue

        params = fit_fn(st, ch)
        y_fit = None
        if params.get('success', False) and not np.isnan(params.get('mu', np.nan)):
            y_fit = cg_fn(x_fit, params['mu'], params['sigma'],
                           params['lapse_low'], params['lapse_high'])

        entry = {
            'params': params,
            'x_fit': x_fit,
            'y_fit': y_fit,
            'n_trials': len(st),
        }
        if hasattr(s, 'session_id'):
            entry['session_id'] = s.session_id
        per_session.append(entry)

    return {
        'mode': 'overlay',
        'per_session': per_session,
        'n_sessions': len(per_session),
    }


def _compute_session_mean(sessions, x_fit, n_bins, n_bootstrap, fit_fn, cg_fn):
    """Per-session binning, then mean P(B) ± SEM across sessions."""
    all_binned = []
    per_session_fits = []
    centres = None

    for s in sessions:
        st, ch = _get_stim_choices(s)
        if len(st) < 10:
            continue

        c, m, _ = _bin_data(st, ch, n_bins)
        all_binned.append(m)
        centres = c

        params = fit_fn(st, ch)
        y_fit = None
        if params.get('success', False) and not np.isnan(params.get('mu', np.nan)):
            y_fit = cg_fn(x_fit, params['mu'], params['sigma'],
                           params['lapse_low'], params['lapse_high'])
        per_session_fits.append({
            'params': params,
            'x_fit': x_fit,
            'y_fit': y_fit,
        })

    if not all_binned:
        return {'mode': 'session_mean', 'n_sessions': 0}

    arr = np.array(all_binned)
    mean_p = np.nanmean(arr, axis=0)
    n_valid = np.sum(~np.isnan(arr), axis=0)
    sem_p = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))

    # Fit to the mean curve
    valid = ~np.isnan(mean_p)
    pooled_params = fit_fn(centres[valid], mean_p[valid]) if valid.sum() >= 4 else {}
    pooled_y = None
    if pooled_params.get('success', False):
        pooled_y = cg_fn(x_fit, pooled_params['mu'], pooled_params['sigma'],
                          pooled_params['lapse_low'], pooled_params['lapse_high'])

    return {
        'mode': 'session_mean',
        'bin_centres': centres,
        'mean_p': mean_p,
        'sem_p': sem_p,
        'pooled_fit': {'params': pooled_params, 'x_fit': x_fit, 'y_fit': pooled_y},
        'per_session_fits': per_session_fits,
        'n_sessions': len(all_binned),
    }


    
def compute_psychometric_gof(stimuli: np.ndarray, choices: np.ndarray,
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
    
def compute_psych_error(psych_true: Dict, psych_fitted: Dict) -> Dict:
    """
    Compute errors between two psychometric fits.
    
    Args:
        psych_true: Psychometric fit from true model
        psych_fitted: Psychometric fit from fitted model
    
    Returns:
        Dict with errors for each psychometric parameter
    """
    errors = {}
    for key in ['mu', 'sigma', 'lapse_low', 'lapse_high']:
        if psych_true.get('success', False) and psych_fitted.get('success', False):
            errors[key] = psych_fitted[key] - psych_true[key]
            errors[f'{key}_true'] = psych_true[key]
            errors[f'{key}_fitted'] = psych_fitted[key]
        else:
            errors[key] = np.nan
            errors[f'{key}_true'] = np.nan
            errors[f'{key}_fitted'] = np.nan
    
    # Curve error (if both have fitted curves)
    if 'y_fit' in psych_true and 'y_fit' in psych_fitted:
        if psych_true['y_fit'] is not None and psych_fitted['y_fit'] is not None:
            errors['curve_mae'] = np.mean(np.abs(psych_fitted['y_fit'] - psych_true['y_fit']))
            errors['curve_max_diff'] = np.max(np.abs(psych_fitted['y_fit'] - psych_true['y_fit']))
        else:
            errors['curve_mae'] = np.nan
            errors['curve_max_diff'] = np.nan
    
    return errors