"""
behav_utils.analysis — Behavioural Analysis Tools

Two levels per domain:
    Low-level:     Takes raw arrays. Usable with any data source.
    Session-level: Takes pre-filtered sessions. Returns result dicts for plotting.

Usage:
    # Low-level (arrays from any source)
    params = fit_psychometric(stimuli, choices)
    um, cond, info = compute_update_matrix(stim, ch, cat)
    result = compare_conditions(stim_a, ch_a, cat_a, stim_b, ch_b, cat_b)

    # Session-level (pre-filtered sessions → result dicts)
    psych = compute_psychometric(sessions, mode='pooled')
    um = compute_um(sessions)
    traj = compute_trajectory(sessions, ['accuracy', 'mu'])
    comp = compute_comparison(ctrl_sessions, opto_sessions)
    raster = compute_session_raster(session)
"""

from behav_utils.analysis.utils import cumulative_gaussian, generate_stimuli

# Low-level (arrays)
from behav_utils.analysis.psychometry import fit_psychometric, compute_psychometric_gof
from behav_utils.analysis.update_matrix import compute_update_matrix, matrix_error
from behav_utils.analysis.comparison import compare_conditions, permutation_test_params, bootstrap_param_diff
from behav_utils.analysis.summary_stats import (
    compute_summary_stats, compute_summary_stats_per_session,
    list_available_stats, register_stat,
    SUMMARY_REGISTRY, FEATURE_MATRIX_STATS, flatten_stats, get_stat_names_expanded,
)

# Session-level (sessions → result dicts)
from behav_utils.analysis.psychometry import compute_psychometric
from behav_utils.analysis.update_matrix import compute_um
from behav_utils.analysis.trajectory import compute_trajectory
from behav_utils.analysis.comparison import compute_comparison
from behav_utils.analysis.session_raster import compute_session_raster
from behav_utils.analysis.session_features import compute_session_features

__all__ = [
    # Utils
    'cumulative_gaussian', 'generate_stimuli',

    # Low-level
    'fit_psychometric', 'compute_psychometric_gof',
    'compute_update_matrix', 'matrix_error',
    'compare_conditions', 'permutation_test_params', 'bootstrap_param_diff',
    'compute_summary_stats', 'compute_summary_stats_per_session',
    'list_available_stats', 'register_stat',
    'SUMMARY_REGISTRY', 'FEATURE_MATRIX_STATS', 'flatten_stats', 'get_stat_names_expanded',

    # Session-level
    'compute_psychometric',
    'compute_um',
    'compute_trajectory',
    'compute_comparison',
    'compute_session_raster',
    'compute_session_features'
]

