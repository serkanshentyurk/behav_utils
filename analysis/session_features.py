"""
Session Feature Matrix Builder

Computes a (session x feature) DataFrame from AnimalData, suitable for
SLDS/HMM epoch categorisation and general behavioural analysis.

Combines:
    - Core summary statistics from summary_stats (choice-based)
    - RT-based features
    - Session-to-session deltas
    - Session metadata

Usage:
    from behav_utils.analysis.session_features import build_feature_matrix

    df = build_feature_matrix(animal)
    df = build_feature_matrix_multi([animal1, animal2])
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, List, Callable, Dict, Union, TYPE_CHECKING

from behav_utils.analysis.summary_stats import (
    SUMMARY_REGISTRY,
    FEATURE_MATRIX_STATS,
    compute_summary_stats,
)

if TYPE_CHECKING:
    from behav_utils.data.structures import AnimalData, SessionData, TrialData


# =============================================================================
# RT EXTRACTION
# =============================================================================

def default_rt_extractor(trials: 'TrialData') -> np.ndarray:
    """
    Default RT extractor: uses reaction_time field.
    Returns array of shape (n_trials,) with NaN for aborts/no-response.
    """
    rt = trials.reaction_time.copy().astype(float)
    rt[trials.abort] = np.nan
    rt[trials.no_response] = np.nan
    return rt


# =============================================================================
# RT FEATURES
# =============================================================================

def compute_rt_features(
    rt: np.ndarray,
    stimuli: np.ndarray,
    categories: np.ndarray,
    choices: np.ndarray,
    hard_threshold: float = 0.3,
    fast_threshold: float = 50.0,
) -> Dict[str, float]:
    """Compute RT-based features from a single session."""
    valid = ~np.isnan(rt) & ~np.isnan(choices)
    rt_valid = rt[valid]

    result = {}

    if len(rt_valid) < 10:
        for key in ['rt_median', 'rt_iqr', 'rt_skewness', 'proportion_fast',
                     'rt_median_hard', 'rt_median_easy', 'rt_correct_vs_error']:
            result[key] = np.nan
        return result

    result['rt_median'] = float(np.median(rt_valid))
    q25, q75 = np.percentile(rt_valid, [25, 75])
    result['rt_iqr'] = float(q75 - q25)

    if np.std(rt_valid) > 0:
        result['rt_skewness'] = float(
            np.mean(((rt_valid - np.mean(rt_valid)) / np.std(rt_valid)) ** 3)
        )
    else:
        result['rt_skewness'] = np.nan

    result['proportion_fast'] = float(np.mean(rt_valid <= fast_threshold))

    s_valid = stimuli[valid]
    hard = np.abs(s_valid) < hard_threshold
    easy = np.abs(s_valid) >= hard_threshold

    result['rt_median_hard'] = float(np.median(rt_valid[hard])) if hard.sum() >= 3 else np.nan
    result['rt_median_easy'] = float(np.median(rt_valid[easy])) if easy.sum() >= 3 else np.nan

    c_valid = choices[valid]
    cat_valid = categories[valid]
    correct = c_valid == cat_valid
    error = c_valid != cat_valid

    if correct.sum() >= 3 and error.sum() >= 3:
        result['rt_correct_vs_error'] = float(
            np.median(rt_valid[correct]) - np.median(rt_valid[error])
        )
    else:
        result['rt_correct_vs_error'] = np.nan

    return result


# =============================================================================
# SINGLE SESSION FEATURES
# =============================================================================

def compute_session_features(
    session: 'SessionData',
    rt_extractor: Callable = default_rt_extractor,
    stat_names: Optional[List[str]] = None,
    exclude_abort: bool = True,
    exclude_opto: bool = True,
    hard_threshold: float = 0.3,
    fast_threshold: float = 50.0,
) -> Dict[str, float]:
    """
    Compute all features for a single session.

    Returns dict of feature_name -> value (all scalars).
    """
    if stat_names is None:
        stat_names = FEATURE_MATRIX_STATS

    # Metadata
    features = {
        'animal_id': session.metadata.animal_id,
        'session_id': session.session_id,
        'session_idx': session.session_idx,
        'date': session.date,
        'stage': session.stage,
        'distribution': session.distribution,
    }

    # Get filtered arrays
    arrays = session.trials.get_arrays(
        exclude_abort=exclude_abort,
        exclude_opto=exclude_opto,
    )

    stimuli = arrays['stimuli']
    categories = arrays['categories']
    choices = arrays['choices']
    no_response = arrays['no_response']

    features['n_trials_total'] = session.trials.n_trials
    features['n_trials_valid'] = int((~no_response).sum())
    features['n_trials_abort'] = int(session.trials.abort.sum())
    features['abort_rate'] = float(session.trials.abort.mean())

    if features['n_trials_valid'] < 10:
        warnings.warn(
            f"Session {session.session_id}: only {features['n_trials_valid']} valid trials."
        )

    # Core summary stats
    stats_dict = compute_summary_stats(
        choices, stimuli, categories,
        stat_names=stat_names,
        return_dict=True,
    )

    for stat_name, value in stats_dict.items():
        if isinstance(value, dict):
            for k, v in value.items():
                features[k] = float(v) if not isinstance(v, (str, type(None))) else v
        elif isinstance(value, np.ndarray):
            for i, v in enumerate(value):
                features[f'{stat_name}_{i}'] = float(v)
        else:
            features[stat_name] = float(value) if not isinstance(value, (str, type(None))) else value

    # RT features
    rt_raw = rt_extractor(session.trials)
    trial_indices = arrays['trial_indices']
    rt_filtered = rt_raw[trial_indices]

    rt_features = compute_rt_features(
        rt_filtered, stimuli, categories, choices,
        hard_threshold=hard_threshold,
        fast_threshold=fast_threshold,
    )
    features.update(rt_features)

    return features


# =============================================================================
# FEATURE MATRIX BUILDER
# =============================================================================

def build_feature_matrix(
    animal: 'AnimalData',
    rt_extractor: Callable = default_rt_extractor,
    stat_names: Optional[List[str]] = None,
    stage: Optional[str] = None,
    exclude_abort: bool = True,
    exclude_opto: bool = True,
    min_valid_trials: int = 10,
    hard_threshold: float = 0.3,
    fast_threshold: float = 50.0,
    compute_deltas: bool = True,
) -> pd.DataFrame:
    """
    Build session x feature DataFrame for a single animal.

    Returns DataFrame with one row per session, columns = features.
    """
    sessions = animal.get_sessions(stage=stage) if stage else animal.sessions

    rows = []
    for session in sessions:
        features = compute_session_features(
            session,
            rt_extractor=rt_extractor,
            stat_names=stat_names,
            exclude_abort=exclude_abort,
            exclude_opto=exclude_opto,
            hard_threshold=hard_threshold,
            fast_threshold=fast_threshold,
        )

        if features['n_trials_valid'] < min_valid_trials:
            continue

        rows.append(features)

    if not rows:
        warnings.warn(f"No valid sessions for animal {animal.animal_id}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if compute_deltas and len(df) > 1:
        df = _add_delta_features(df)

    return df


def build_feature_matrix_multi(
    animals: List['AnimalData'],
    **kwargs,
) -> pd.DataFrame:
    """Build pooled feature matrix across multiple animals."""
    dfs = []
    for animal in animals:
        df = build_feature_matrix(animal, **kwargs)
        if len(df) > 0:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# DELTA FEATURES
# =============================================================================

DELTA_FEATURES = [
    'pse', 'slope', 'accuracy', 'side_bias', 'recency',
    'choice_entropy', 'rt_median',
]


def _add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feat in DELTA_FEATURES:
        if feat in df.columns:
            df[f'delta_{feat}'] = df[feat].diff().abs()
    return df


# =============================================================================
# METADATA & FEATURE UTILITIES
# =============================================================================

METADATA_COLUMNS = [
    'animal_id', 'session_id', 'session_idx', 'date', 'stage', 'distribution',
    'n_trials_total', 'n_trials_valid', 'n_trials_abort', 'abort_rate',
]


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric feature columns (excluding metadata and deltas)."""
    exclude = set(METADATA_COLUMNS)
    return [
        col for col in df.columns
        if col not in exclude
        and not col.startswith('delta_')
        and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]


def get_delta_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith('delta_')]


def get_numeric_features(df: pd.DataFrame, include_deltas: bool = False) -> pd.DataFrame:
    cols = get_feature_columns(df)
    if include_deltas:
        cols += get_delta_columns(df)
    return df[cols].copy()


def zscore_features(df: pd.DataFrame, include_deltas: bool = False) -> pd.DataFrame:
    df_out = df.copy()
    cols = get_feature_columns(df_out)
    if include_deltas:
        cols += get_delta_columns(df_out)
    for col in cols:
        vals = df_out[col].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() > 1:
            mu = np.nanmean(vals)
            sd = np.nanstd(vals)
            if sd > 1e-10:
                df_out[col] = (vals - mu) / sd
            else:
                df_out[col] = 0.0
    return df_out


def summarise_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = get_feature_columns(df)
    rows = []
    for col in cols:
        vals = df[col].values.astype(float)
        valid = ~np.isnan(vals)
        rows.append({
            'feature': col,
            'n_valid': int(valid.sum()),
            'n_nan': int((~valid).sum()),
            'mean': float(np.nanmean(vals)) if valid.any() else np.nan,
            'std': float(np.nanstd(vals)) if valid.any() else np.nan,
            'min': float(np.nanmin(vals)) if valid.any() else np.nan,
            'max': float(np.nanmax(vals)) if valid.any() else np.nan,
        })
    return pd.DataFrame(rows)
