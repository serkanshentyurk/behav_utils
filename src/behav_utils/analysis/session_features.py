"""
Session Feature Matrix Builder

Computes per-session feature dictionaries from a SessionData object,
suitable for SLDS/HMM epoch categorisation and general behavioural
analysis.

Combines:
    - Scalar statistics from behav_utils.stats (choice-based)
    - RT-based features
    - Session metadata

For a multi-session matrix, the workflow is inline in the notebook:

    import pandas as pd
    df = pd.DataFrame([compute_session_features(s) for s in animal.sessions])
"""

from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats import compute_stats, list_stats

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData


def compute_session_features(
    session: 'SessionData',
    stat_names: List[str] | None = None,
    hard_threshold: float = 0.3,
    fast_threshold: float = 50.0,
) -> Dict[str, float]:
    """
    Compute all features for a single session.

    No filtering here. get_arrays() no longer drops aborts, so the session
    must be pre-filtered via filter_trials (default exclude_abort=True).

    Args:
        session: SessionData with valid trials.
        stat_names: Scalar stat names to include (default: every registered stat).
        hard_threshold: |stimulus| below this counts as a "hard" trial
            for the rt_median_hard/easy split.
        fast_threshold: RT (ms) below this counts as a "fast" response
            for the proportion_fast feature.

    Returns:
        Dict {feature_name: scalar value}, including:
            - metadata (animal_id, session_id, session_idx, date, ...)
            - trial counts (n_trials_total, n_trials_valid, ...)
            - one entry per registered summary stat in stat_names
            - RT features (rt_median, rt_iqr, rt_skewness, proportion_fast,
              rt_median_hard, rt_median_easy, rt_correct_vs_error)
    """
    if stat_names is None:
        stat_names = list_stats()

    arrays = session.get_arrays()
    stimuli = arrays['stimuli']
    choices = arrays['choices']
    categories = arrays['categories']

    # ── Metadata + counts ─────────────────────────────────────────
    features: Dict[str, Union[float, str]] = {
        'animal_id':     session.metadata.animal_id,
        'session_id':    session.session_id,
        'session_idx':   session.session_idx,
        'date':          session.date, # type: ignore
        'stage':         session.stage,
        'distribution':  session.distribution,
        'n_trials_total': len(session.trials.choice),
        'n_trials_valid': int(session.trials.valid_mask.sum()),
        'n_trials_abort': int(session.trials.no_response.sum()),
        'abort_rate':    float(session.trials.no_response.mean()),
    }

    # ── Scalar stats from the registry ────────────────────────────
    features.update(compute_stats(TrialArrays.from_pooled(arrays), stat_names).to_dict())

    # ── RT features (inlined) ─────────────────────────────────────
    features.update(_compute_rt_features(
        session, stimuli, categories, choices,
        hard_threshold=hard_threshold,
        fast_threshold=fast_threshold,
    ))

    return features


# ── private helpers ─────────────────────────────────────────────

def _compute_rt_features(
    session: 'SessionData',
    stimuli: np.ndarray,
    categories: np.ndarray,
    choices: np.ndarray,
    hard_threshold: float,
    fast_threshold: float,
) -> Dict[str, float]:
    """
    Extract reaction times from session.trials.reaction_time, then compute
    RT summary features on the valid subset.

    Assumes a pre-filtered session (aborts already removed via filter_trials);
    abort / no-response RTs are set to NaN here and excluded from the valid
    subset regardless. ``rt`` aligns 1:1 with ``choices`` (both span the whole
    session, since get_arrays no longer drops aborts).
    """
    rt_full = session.trials.reaction_time.copy().astype(float)
    rt_full[session.trials.abort] = np.nan
    rt_full[session.trials.no_response] = np.nan
    rt = rt_full

    nan_keys = [
        'rt_median', 'rt_iqr', 'rt_skewness', 'proportion_fast',
        'rt_median_hard', 'rt_median_easy', 'rt_correct_vs_error',
    ]
    valid = ~np.isnan(rt) & ~np.isnan(choices)
    if valid.sum() < 10:
        return {k: np.nan for k in nan_keys}

    rt_v = rt[valid]
    s_v = stimuli[valid]
    c_v = choices[valid]
    cat_v = categories[valid]

    result = {
        'rt_median':       float(np.median(rt_v)),
        'rt_iqr':          float(np.percentile(rt_v, 75) - np.percentile(rt_v, 25)),
        'proportion_fast': float(np.mean(rt_v <= fast_threshold)),
    }

    if np.std(rt_v) > 0:
        result['rt_skewness'] = float(
            np.mean(((rt_v - np.mean(rt_v)) / np.std(rt_v)) ** 3)
        )
    else:
        result['rt_skewness'] = np.nan

    hard = np.abs(s_v) < hard_threshold
    easy = ~hard
    result['rt_median_hard'] = float(np.median(rt_v[hard])) if hard.sum() >= 3 else np.nan
    result['rt_median_easy'] = float(np.median(rt_v[easy])) if easy.sum() >= 3 else np.nan

    correct = c_v == cat_v
    error = ~correct
    if correct.sum() >= 3 and error.sum() >= 3:
        result['rt_correct_vs_error'] = float(
            np.median(rt_v[correct]) - np.median(rt_v[error])
        )
    else:
        result['rt_correct_vs_error'] = np.nan

    return result
