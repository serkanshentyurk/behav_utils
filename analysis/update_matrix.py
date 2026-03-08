"""
Update Matrix Computation

Computes serial dependence (update) matrices from behavioural data.
Operates on raw arrays — no data class dependency.
"""

import numpy as np
from typing import Optional, Dict, Tuple, Literal

from behav_utils.analysis.psychometry import fit_psychometric


def compute_update_matrix(
    stimuli: np.ndarray,
    choices: np.ndarray,
    categories: np.ndarray,
    n_bins: int = 8,
    trial_filter: Literal['all', 'post_correct'] = 'post_correct',
    no_response: Optional[np.ndarray] = None,
    not_blockstart: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute update matrix from behavioural data.

    The update matrix captures serial dependence: how does the previous trial's
    stimulus shift the current psychometric curve?

    Args:
        stimuli: Stimulus values for each trial
        choices: Binary choices (0=A, 1=B)
        categories: True categories (0=A, 1=B)
        n_bins: Number of bins for stimulus discretisation
        trial_filter: 'post_correct' (only after correct trials) or 'all'
        no_response: Boolean array (True = no response).
                     If None, inferred from np.isnan(choices)
        not_blockstart: Boolean array (True = not start of block).
                        If None, inferred as [False, True, True, ...]

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

    if not_blockstart is None:
        not_blockstart = np.ones(n_trials, dtype=bool)
        if n_trials > 0:
            not_blockstart[0] = False
    else:
        not_blockstart = np.asarray(not_blockstart, dtype=bool)

    # Rewards
    rewards = (choices == categories).astype(float)
    rewards[np.isnan(choices)] = np.nan

    # Bins
    bin_edges = np.linspace(-1, 1, n_bins + 1)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.clip(np.digitize(stimuli, bin_edges) - 1, 0, n_bins - 1)

    # Selection mask
    curr_responded = ~no_response[1:]
    prev_responded = ~no_response[:-1]
    is_not_blockstart = not_blockstart[1:]

    if trial_filter == 'post_correct':
        prev_correct = rewards[:-1] == 1
        base_condition = prev_correct & curr_responded & prev_responded & is_not_blockstart
    elif trial_filter == 'all':
        base_condition = curr_responded & prev_responded & is_not_blockstart
    else:
        raise ValueError(f"trial_filter must be 'post_correct' or 'all', got '{trial_filter}'")

    # Overall psychometric
    total_stimuli = stimuli[1:][base_condition]
    total_choices = choices[1:][base_condition]
    total_psych = fit_psychometric(total_stimuli, total_choices, midpoints)

    if total_psych['success']:
        total_curve = total_psych['y_fit']
    else:
        total_curve = np.full(n_bins, np.nan)

    # Conditional psychometrics
    conditional_matrix = np.zeros((n_bins, n_bins))
    update_matrix = np.zeros((n_bins, n_bins))
    bin_counts = np.zeros(n_bins, dtype=int)
    conditional_psychs = []

    for j in range(n_bins):
        prev_in_bin = bin_indices[:-1] == j
        condition = base_condition & prev_in_bin

        cond_stimuli = stimuli[1:][condition]
        cond_choices = choices[1:][condition]
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
