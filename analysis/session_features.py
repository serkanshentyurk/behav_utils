"""
Session Feature Matrix Builder

Computes per-session feature dictionaries from a SessionData object,
suitable for SLDS/HMM epoch categorisation and general behavioural
analysis.

Combines:
    - Core summary statistics from summary_stats (choice-based)
    - RT-based features
    - Session metadata

For a multi-session matrix, the workflow is inline in the notebook:

    import pandas as pd
    df = pd.DataFrame([compute_session_features(s) for s in animal.sessions])
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Union, Optional, TYPE_CHECKING

from behav_utils.analysis.summary_stats import (
    SUMMARY_REGISTRY,
    FEATURE_MATRIX_STATS,
    compute_summary_stats,
)

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData


def compute_session_features(
    session: 'SessionData',
    stat_names: Optional[List[str]] = None,
    hard_threshold: float = 0.3,
    fast_threshold: float = 50.0,
) -> Dict[str, float]:
    """
    Compute all features for a single session.

    No filtering. Data must be pre-filtered via filter_session() if needed.

    Args:
        session: SessionData with valid trials.
        stat_names: Stat names to include (default: FEATURE_MATRIX_STATS).
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
        stat_names = FEATURE_MATRIX_STATS

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

    # ── Summary stats from registry ───────────────────────────────
    stats_dict = compute_summary_stats(
        choices=choices, stimuli=stimuli, categories=categories,
        stat_names=stat_names, return_dict=True,
    )
    for stat_name, value in stats_dict.items():
        if isinstance(value, dict):
            for k, v in value.items():
                features[k] = float(v) if not isinstance(v, (str, type(None))) else v
        elif isinstance(value, np.ndarray):
            for i, v in enumerate(value):
                features[f'{stat_name}_{i}'] = float(v)
        else:
            features[stat_name] = (
                float(value) if not isinstance(value, (str, type(None))) else value
            )

    # ── RT features (inlined) ─────────────────────────────────────
    features.update(_compute_rt_features(
        session, arrays['trial_indices'],
        stimuli, categories, choices,
        hard_threshold=hard_threshold,
        fast_threshold=fast_threshold,
    ))

    return features


# ── private helpers ─────────────────────────────────────────────

def _compute_rt_features(
    session: 'SessionData',
    trial_indices: np.ndarray,
    stimuli: np.ndarray,
    categories: np.ndarray,
    choices: np.ndarray,
    hard_threshold: float,
    fast_threshold: float,
) -> Dict[str, float]:
    """
    Extract reaction times from session.trials.reaction_time, then compute
    RT summary features on the valid subset.
    """
    rt_full = session.trials.reaction_time.copy().astype(float)
    rt_full[session.trials.abort] = np.nan
    rt_full[session.trials.no_response] = np.nan
    rt = rt_full[trial_indices]

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
