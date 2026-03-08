"""
behav_utils.analysis — Behavioural Analysis Tools

Summary statistics, psychometric fitting, update matrices,
and session feature extraction for 2AFC tasks.

Usage:
    from behav_utils.analysis import compute_summary_stats, fit_psychometric
    from behav_utils.analysis import compute_update_matrix
    from behav_utils.analysis import build_feature_matrix
"""

from behav_utils.analysis.utils import cumulative_gaussian, generate_stimuli
from behav_utils.analysis.psychometry import (
    fit_psychometric,
    compute_psychometric_gof,
)
from behav_utils.analysis.summary_stats import (
    compute_summary_stats,
    compute_summary_stats_per_session,
    list_available_stats,
    register_stat,
    SUMMARY_REGISTRY,
    FEATURE_MATRIX_STATS,
    flatten_stats,
    get_stat_names_expanded,
)
from behav_utils.analysis.update_matrix import (
    compute_update_matrix,
    matrix_error,
)
from behav_utils.analysis.session_features import (
    build_feature_matrix,
    build_feature_matrix_multi,
    compute_session_features,
    summarise_features,
    get_feature_columns,
    get_numeric_features,
    zscore_features,
)

__all__ = [
    # Utils
    'cumulative_gaussian',
    'generate_stimuli',

    # Psychometry
    'fit_psychometric',
    'compute_psychometric_gof',

    # Summary stats
    'compute_summary_stats',
    'compute_summary_stats_per_session',
    'list_available_stats',
    'register_stat',
    'SUMMARY_REGISTRY',
    'FEATURE_MATRIX_STATS',
    'flatten_stats',
    'get_stat_names_expanded',

    # Update matrix
    'compute_update_matrix',
    'matrix_error',

    # Session features
    'build_feature_matrix',
    'build_feature_matrix_multi',
    'compute_session_features',
    'summarise_features',
    'get_feature_columns',
    'get_numeric_features',
    'zscore_features',
]
