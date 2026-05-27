"""
Condition comparison for 2AFC tasks.

General-purpose comparison of two independent trial sets:
psychometric parameter differences with permutation tests,
bootstrap CIs on differences, accuracy Fisher's exact test,
and update matrix comparison.

Works for any two-condition comparison:
- Opto vs control trials (within session)
- Pre-shift vs post-shift sessions
- Hard-A vs Hard-B sessions
- Masking vs real opto
- Het vs WT animals (pooled trials)

Public API:
    compare_conditions      — Full comparison of two trial sets
    permutation_test_params — Permutation test on psychometric param diffs
    bootstrap_param_diff    — Bootstrap CI on psychometric param diffs

Usage:
    from behav_utils.analysis.comparison import compare_conditions

    result = compare_conditions(
        stim_a, choices_a, cat_a,
        stim_b, choices_b, cat_b,
    )
    # result['diffs']['mu']  → PSE difference (A - B)
    # result['perm_p']['mu'] → permutation p-value
    # result['boot_ci']['mu'] → (lower, upper) 95% CI on diff
    # result['um_rmse']       → UM RMSE between conditions
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon, mannwhitneyu

from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.analysis.update_matrix import compute_update_matrix

from behav_utils import pool_arrays

PARAM_NAMES = ('mu', 'sigma', 'lapse_low', 'lapse_high')


def _fit_params(stimuli, choices):
    """Fit psychometric and return param dict, or None if fit fails.

    Keys: mu (PSE), sigma (slope), lapse_low, lapse_high — matching
    fit_psychometric() and compute_psychometric() conventions.
    """
    pfit = fit_psychometric(stimuli, choices)
    if not pfit.get('success', False) and np.isnan(pfit.get('mu', np.nan)):
        return None
    return {
        'mu':         float(pfit['mu']),
        'sigma':      float(pfit['sigma']),
        'lapse_low':  float(pfit['lapse_low']),
        'lapse_high': float(pfit['lapse_high']),
    }


def _accuracy(choices, categories):
    """Compute accuracy, handling NaN choices."""
    valid = ~np.isnan(choices)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(choices[valid] == categories[valid]))


def permutation_test_params(
    stimuli_a: np.ndarray, choices_a: np.ndarray,
    stimuli_b: np.ndarray, choices_b: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Permutation test for psychometric parameter differences.

    Pools all trials, shuffles group labels, refits both groups,
    computes param diffs. p-value = proportion of permuted diffs
    at least as extreme as observed.

    Two-sided test: compares |observed diff| against |permuted diffs|.

    Args:
        stimuli_a, choices_a: Condition A trials
        stimuli_b, choices_b: Condition B trials
        n_permutations: Number of permutations
        seed: Random seed

    Returns:
        Dict mapping param name → p-value. NaN if fit fails.
    """
    rng = np.random.default_rng(seed)

    # Observed diffs
    params_a = _fit_params(stimuli_a, choices_a)
    params_b = _fit_params(stimuli_b, choices_b)
    if params_a is None or params_b is None:
        return {p: np.nan for p in PARAM_NAMES}

    observed = {p: params_a[p] - params_b[p] for p in params_a}

    # Pool trials
    all_stim = np.concatenate([stimuli_a, stimuli_b])
    all_choice = np.concatenate([choices_a, choices_b])
    n_a = len(stimuli_a)
    n_total = len(all_stim)

    # Permutation distribution
    perm_diffs = {p: [] for p in observed}
    for _ in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        perm_stim_a = all_stim[perm_idx[:n_a]]
        perm_choice_a = all_choice[perm_idx[:n_a]]
        perm_stim_b = all_stim[perm_idx[n_a:]]
        perm_choice_b = all_choice[perm_idx[n_a:]]

        pa = _fit_params(perm_stim_a, perm_choice_a)
        pb = _fit_params(perm_stim_b, perm_choice_b)
        if pa is None or pb is None:
            continue

        for p in observed:
            perm_diffs[p].append(pa[p] - pb[p])

    # Compute p-values (two-sided)
    p_values = {}
    for p in observed:
        perm = np.array(perm_diffs[p])
        if len(perm) < 10:
            p_values[p] = np.nan
            continue
        p_values[p] = float(np.mean(np.abs(perm) >= np.abs(observed[p])))

    return p_values


def bootstrap_param_diff(
    stimuli_a: np.ndarray, choices_a: np.ndarray,
    stimuli_b: np.ndarray, choices_b: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap CI on psychometric parameter differences (A - B).

    Resamples each group independently with replacement, refits,
    computes diff. Returns percentile CIs.

    Args:
        stimuli_a, choices_a: Condition A trials
        stimuli_b, choices_b: Condition B trials
        n_bootstrap: Number of bootstrap samples
        ci_level: CI coverage (default 0.95 → 2.5/97.5 percentiles)
        seed: Random seed

    Returns:
        Dict mapping param name → (lower, upper) CI.
    """
    rng = np.random.default_rng(seed)
    alpha = (1 - ci_level) / 2
    n_a = len(stimuli_a)
    n_b = len(stimuli_b)

    boot_diffs = {p: [] for p in PARAM_NAMES}

    for _ in range(n_bootstrap):
        idx_a = rng.choice(n_a, size=n_a, replace=True)
        idx_b = rng.choice(n_b, size=n_b, replace=True)

        pa = _fit_params(stimuli_a[idx_a], choices_a[idx_a])
        pb = _fit_params(stimuli_b[idx_b], choices_b[idx_b])
        if pa is None or pb is None:
            continue

        for p in boot_diffs:
            boot_diffs[p].append(pa[p] - pb[p])

    cis = {}
    for p in boot_diffs:
        vals = np.array(boot_diffs[p])
        if len(vals) < 10:
            cis[p] = (np.nan, np.nan)
        else:
            cis[p] = (
                float(np.percentile(vals, 100 * alpha)),
                float(np.percentile(vals, 100 * (1 - alpha))),
            )
    return cis

def _bootstrap_curve_band(
    stimuli: np.ndarray,
    choices: np.ndarray,
    x_eval: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Bootstrap percentile band around the fitted psychometric curve.

    For each bootstrap iteration, resample trials with replacement,
    refit the psychometric, evaluate the cumulative gaussian at x_eval.
    Returns lower and upper percentiles per x.

    Args:
        stimuli, choices: Trial arrays for one condition.
        x_eval: x values at which to evaluate the curve.
        n_bootstrap: Number of bootstrap iterations.
        alpha: Two-sided significance level (0.05 → 95% band).
        seed: RNG seed.

    Returns:
        {'x': x_eval, 'lo': lower band, 'hi': upper band, 'median': median curve}
    """
    from behav_utils.analysis.utils import cumulative_gaussian

    rng = np.random.default_rng(seed)
    n = len(stimuli)
    curves = np.full((n_bootstrap, len(x_eval)), np.nan)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        p = fit_psychometric(stimuli[idx], choices[idx])
        if p.get('success', False) and not np.isnan(p.get('mu', np.nan)):
            curves[i] = cumulative_gaussian(
                x_eval,
                p['mu'], p['sigma'],
                p.get('lapse_low', 0.0), p.get('lapse_high', 0.0),
            )

    return {
        'x':      x_eval,
        'lo':     np.nanpercentile(curves, 100 * alpha / 2,       axis=0),
        'hi':     np.nanpercentile(curves, 100 * (1 - alpha / 2), axis=0),
        'median': np.nanmedian(curves, axis=0),
    }

def compare_conditions(
    stimuli_a: np.ndarray, choices_a: np.ndarray, categories_a: np.ndarray,
    stimuli_b: np.ndarray, choices_b: np.ndarray, categories_b: np.ndarray,
    n_bins: int = 8,
    n_permutations: int = 1000,
    n_bootstrap: int = 1000,
    seed: int = 42,
    label_a: str = 'A',
    label_b: str = 'B',
) -> Dict:
    """
    Compare two independent trial sets across all behavioural metrics.

    Computes psychometric parameters, accuracy, and update matrices
    for each condition, then tests for differences using permutation
    tests, bootstrap CIs, and Fisher's exact test.

    This is the general-purpose comparison function. Specific
    use-cases (opto vs control, pre vs post shift) are thin
    wrappers that select trials and call this.

    Args:
        stimuli_a/b: Stimulus arrays for conditions A and B
        choices_a/b: Choice arrays (binary, may contain NaN)
        categories_a/b: Category arrays
        n_bins: Number of bins for update matrix
        n_permutations: Permutation test iterations (0 to skip)
        n_bootstrap: Bootstrap iterations (0 to skip)
        seed: Random seed
        label_a, label_b: Labels for the two conditions

    Returns dict with:
        params_a, params_b: {accuracy, mu, sigma, lapse_low, lapse_high}
            mu = PSE; sigma = psychometric slope.
        diffs: same keys, A − B.
        n_a, n_b: trial counts.

        perm_p: {mu, sigma, lapse_low, lapse_high} → permutation p-values
            (None if n_permutations == 0).
        boot_ci: {mu, sigma, lapse_low, lapse_high} → (lo, hi) CIs on diffs
            (None if n_bootstrap == 0).
        boot_band_a, boot_band_b: per-condition fit-curve bands
            {x, lo, hi, median} (None if n_bootstrap == 0).

        fisher_p: Fisher's exact test p-value for accuracy difference.
        fisher_odds: odds ratio.

        um_a, um_b: (n_bins, n_bins) update matrices.
        um_diff: um_a − um_b.
        um_rmse: element-wise RMSE of difference.
        um_corr: Pearson r between flattened UMs.

        label_a, label_b: condition labels.
    """
    # Clean NaN choices
    valid_a = ~np.isnan(choices_a.astype(float))
    valid_b = ~np.isnan(choices_b.astype(float))
    stim_a, ch_a, cat_a = stimuli_a[valid_a], choices_a[valid_a], categories_a[valid_a]
    stim_b, ch_b, cat_b = stimuli_b[valid_b], choices_b[valid_b], categories_b[valid_b]

    # ── Psychometric fits ────────────────────────────────────────────
    params_a = _fit_params(stim_a, ch_a) or {}
    params_b = _fit_params(stim_b, ch_b) or {}

    acc_a = _accuracy(ch_a, cat_a)
    acc_b = _accuracy(ch_b, cat_b)
    params_a['accuracy'] = acc_a
    params_b['accuracy'] = acc_b

    diffs = {}
    for key in ('accuracy',) + PARAM_NAMES:
        diffs[key] = params_a.get(key, np.nan) - params_b.get(key, np.nan)

    # ── Permutation test on psychometric params ──────────────────────
    perm_p = None
    if n_permutations > 0 and len(stim_a) >= 10 and len(stim_b) >= 10:
        perm_p = permutation_test_params(
            stim_a, ch_a, stim_b, ch_b,
            n_permutations=n_permutations, seed=seed,
        )

    # ── Bootstrap CI on param diffs ──────────────────────────────────
    boot_ci = None
    if n_bootstrap > 0 and len(stim_a) >= 10 and len(stim_b) >= 10:
        boot_ci = bootstrap_param_diff(
            stim_a, ch_a, stim_b, ch_b,
            n_bootstrap=n_bootstrap, seed=seed,
        )
        
    # ── Bootstrap per-condition curve bands ─────────────────────────
    boot_band_a, boot_band_b = None, None
    if n_bootstrap > 0 and len(stim_a) >= 10 and len(stim_b) >= 10:
        x_eval = np.linspace(-1, 1, 200)
        boot_band_a = _bootstrap_curve_band(
            stim_a, ch_a, x_eval,
            n_bootstrap=n_bootstrap, seed=seed,
        )
        boot_band_b = _bootstrap_curve_band(
            stim_b, ch_b, x_eval,
            n_bootstrap=n_bootstrap, seed=seed + 1,
        )
        
    # ── Fisher's exact test on accuracy ──────────────────────────────
    fisher_p, fisher_odds = np.nan, np.nan
    if len(ch_a) >= 5 and len(ch_b) >= 5:
        correct_a = int(np.sum(ch_a == cat_a))
        incorrect_a = int(np.sum(ch_a != cat_a))
        correct_b = int(np.sum(ch_b == cat_b))
        incorrect_b = int(np.sum(ch_b != cat_b))
        try:
            table = [[correct_a, incorrect_a], [correct_b, incorrect_b]]
            fisher_odds, fisher_p = fisher_exact(table, alternative='two-sided')
            fisher_p = float(fisher_p)
            fisher_odds = float(fisher_odds)
        except (ValueError, ZeroDivisionError):
            pass

    # ── Update matrices ──────────────────────────────────────────────
    um_a, _, _ = compute_update_matrix(stim_a, ch_a, cat_a, n_bins=n_bins)
    um_b, _, _ = compute_update_matrix(stim_b, ch_b, cat_b, n_bins=n_bins)

    um_diff = um_a - um_b
    valid_um = ~np.isnan(um_a) & ~np.isnan(um_b)

    if valid_um.sum() >= 4:
        um_rmse = float(np.sqrt(np.mean(um_diff[valid_um] ** 2)))
        from scipy.stats import pearsonr
        um_corr, _ = pearsonr(um_a[valid_um], um_b[valid_um])
        um_corr = float(um_corr)
    else:
        um_rmse = np.nan
        um_corr = np.nan

    return {
        'params_a': params_a,
        'params_b': params_b,
        'diffs': diffs,
        'n_a': int(len(stim_a)),
        'n_b': int(len(stim_b)),
        'perm_p': perm_p,
        'boot_ci': boot_ci,
        'boot_band_a': boot_band_a,           
        'boot_band_b': boot_band_b,           
        'fisher_p': fisher_p,
        'fisher_odds': fisher_odds,
        'um_a': um_a,
        'um_b': um_b,
        'um_diff': um_diff,
        'um_rmse': um_rmse,
        'um_corr': um_corr,
        'label_a': label_a,
        'label_b': label_b,
    }

def compute_comparison(
    sessions_a: List['SessionData'],
    sessions_b: List['SessionData'],
    n_bins: int = 8,
    n_permutations: int = 1000,
    n_bootstrap: int = 1000,
    seed: int = 42,
    label_a: str = 'A',
    label_b: str = 'B',
) -> Dict:
    """
    Compare two groups of pre-filtered sessions.

    Session-level wrapper around compare_conditions(). Pools arrays
    from each group, then runs the full comparison pipeline.

    Args:
        sessions_a: Pre-filtered sessions for condition A.
        sessions_b: Pre-filtered sessions for condition B.
        n_bins: Bins for update matrix.
        n_permutations: Permutation test iterations (0 to skip).
        n_bootstrap: Bootstrap iterations (0 to skip).
        seed: Random seed.
        label_a, label_b: Condition labels.

    Returns:
        Dict from compare_conditions() plus:
            'n_sessions_a', 'n_sessions_b': session counts
        Pass to plot_comparison() for drawing.
    """
    from behav_utils.data.filtering import pool_arrays
    from behav_utils.analysis.comparison import compare_conditions

    arr_a = pool_arrays(sessions_a)
    arr_b = pool_arrays(sessions_b)

    valid_a = ~arr_a['no_response']
    valid_b = ~arr_b['no_response']

    result = compare_conditions(
        arr_a['stimuli'][valid_a], arr_a['choices'][valid_a], arr_a['categories'][valid_a],
        arr_b['stimuli'][valid_b], arr_b['choices'][valid_b], arr_b['categories'][valid_b],
        n_bins=n_bins,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        seed=seed,
        label_a=label_a,
        label_b=label_b,
    )

    result['n_sessions_a'] = len(sessions_a)
    result['n_sessions_b'] = len(sessions_b)

    return result

def compute_per_animal_stats(animals, 
                             sessions_per_animal=None, 
                             stat_keys=('mu', 'sigma', 'lapse_low', 'lapse_high', 'accuracy')):
    rows = []
    for animal in animals:
        sessions = (sessions_per_animal or {}).get(animal.animal_id, animal.sessions)
        if not sessions:
            continue
        # Pool trials within animal
        pooled = pool_arrays(sessions)
        # Fit psychometric on pooled trials
        psych = fit_psychometric(pooled['stimuli'], pooled['choices'])
        # Accuracy
        valid = ~np.isnan(pooled['choices'])
        acc = float(np.mean(pooled['choices'][valid] == pooled['categories'][valid]))
        row = {
            'animal_id':      animal.animal_id,
            'genotype':       getattr(animal, 'genotype', None),
            'n_sessions':     len(sessions),
            'n_trials_total': int(valid.sum()),
            'mu':             psych.get('mu', np.nan),
            'sigma':          psych.get('sigma', np.nan),
            'lapse_low':      psych.get('lapse_low', np.nan),
            'lapse_high':     psych.get('lapse_high', np.nan),
            'accuracy':       acc,
        }
        rows.append({k: v for k, v in row.items() if k == 'animal_id' or k == 'genotype' or k in stat_keys or k.startswith('n_')})
    return pd.DataFrame(rows)


def compute_group_comparison(
    df_a, df_b, 
    label_a='A', label_b='B',
    paired=False, 
    stat_keys=('mu', 'sigma', 'lapse_low', 'lapse_high', 'accuracy')
    ):
    medians_a = {k: float(df_a[k].median()) for k in stat_keys if k in df_a.columns}
    medians_b = {k: float(df_b[k].median()) for k in stat_keys if k in df_b.columns}
    diffs    = {k: medians_a[k] - medians_b[k] for k in medians_a}

    p_values = {}
    for k in stat_keys:
        if k not in df_a.columns or k not in df_b.columns:
            p_values[k] = np.nan
            continue
        vals_a = df_a[k].dropna().values
        vals_b = df_b[k].dropna().values
        try:
            if paired:
                if len(vals_a) != len(vals_b):
                    p_values[k] = np.nan
                    continue
                _, p = wilcoxon(vals_a, vals_b)
            else:
                _, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
            p_values[k] = float(p)
        except ValueError:
            p_values[k] = np.nan

    return {
        'group_a':   {k: df_a[k].values for k in stat_keys if k in df_a.columns},
        'group_b':   {k: df_b[k].values for k in stat_keys if k in df_b.columns},
        'medians_a': medians_a,
        'medians_b': medians_b,
        'diffs':     diffs,
        'p_values':  p_values,
        'n_a':       len(df_a),
        'n_b':       len(df_b),
        'paired':    paired,
        'label_a':   label_a,
        'label_b':   label_b,
    }