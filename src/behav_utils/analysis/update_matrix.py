"""
Update-matrix fit engine on raw arrays.

``fit_update_matrix`` is the array-level fitter (used by the simulator / SBI
path, which carries no lag-1 view). The session-level readout is
``behav_utils.readouts.compute_update_matrix`` on a ``TrialArrays``.
"""

from typing import Dict, Literal, Tuple

import numpy as np

from behav_utils.analysis.psychometry import fit_psychometric


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
