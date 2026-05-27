"""
behav_utils — Behavioural Neuroscience Data Utilities

Config-driven library for loading, filtering, analysing, and
plotting trial-based behavioural data.

Architecture — three levels per domain:

    Low-level:     fit_psychometric(stim, ch)       — raw arrays, any source
    Session-level: compute_psychometric(sessions)   — pre-filtered sessions → result dict
    Plotting:      plot_psychometric(result)         — result dict → axes

Pipeline:
    experiment = load_experiment('config.yaml')
    sessions   = select_sessions(animal, preset='expert_uniform')
    clean      = filter_trials(sessions)

    psych = compute_psychometric(clean, mode='pooled')
    fig, ax = plt.subplots()
    plot_psychometric(psych, ax=ax)

Modules:
    behav_utils.data        — structures, loading, selection, filtering, synthetic
    behav_utils.analysis    — psychometry, update matrix, trajectory, comparison, stats
    behav_utils.plotting    — psychometric, update matrix, trajectory, comparison, session
"""

# ── Config ───────────────────────────────────────────────────────────────────
from behav_utils.config.schema import load_config, ProjectConfig

# ── Data structures ──────────────────────────────────────────────────────────
from behav_utils.data.structures import (
    ExperimentData, AnimalData, SessionData,
    SessionMetadata, TrialData, FittingData,
)

# ── Loading ──────────────────────────────────────────────────────────────────
from behav_utils.data.loading import load_experiment, load_session_csv, load_animal

# ── Session selection ────────────────────────────────────────────────────────
from behav_utils.data.selection import (
    select_sessions, SessionFilter,
    fitting_data_from_sessions,
    register_preset, list_presets, register_presets_from_config,
)

# ── Trial filtering ──────────────────────────────────────────────────────────
from behav_utils.data.filtering import (
    filter_trials, pool_arrays,
    build_mask, opto_mask,
    filter_session, filter_trial_data, get_arrays,
)

# ── Synthetic data ───────────────────────────────────────────────────────────
from behav_utils.data.synthetic import (
    generate_synthetic_animal, generate_synthetic_session,
    sample_stimuli, noisy_psychometric_simulator,
)


# ── Analysis: low-level (arrays) ─────────────────────────────────────────────
from behav_utils.analysis.psychometry import fit_psychometric, compute_psychometric_gof
from behav_utils.analysis.update_matrix import compute_update_matrix, matrix_error
from behav_utils.analysis.comparison import (
    compare_conditions, permutation_test_params, bootstrap_param_diff,
)
from behav_utils.analysis.summary_stats import (
    compute_summary_stats, compute_summary_stats_per_session,
    list_available_stats, register_stat,
)
from behav_utils.analysis.session_features import (
    compute_session_features,
)
from behav_utils.analysis.utils import cumulative_gaussian, generate_stimuli

# ── Analysis: session-level (sessions → result dicts) ────────────────────────
from behav_utils.analysis.psychometry import compute_psychometric
from behav_utils.analysis.update_matrix import compute_um
from behav_utils.analysis.trajectory import compute_trajectory
from behav_utils.analysis.comparison import compute_comparison
from behav_utils.analysis.session_raster import compute_session_raster

# ── Plotting (result dicts → axes) ───────────────────────────────────────────
from behav_utils.plotting import (
    plot_psychometric, plot_um, plot_trajectory,
    plot_comparison, plot_session_raster,
    PALETTE, COLOURS, UM_CMAP,
    apply_style, get_colour,
)

__version__ = '0.2.0'

__all__ = [
    # Config
    'load_config', 'ProjectConfig',

    # Structures
    'ExperimentData', 'AnimalData', 'SessionData',
    'SessionMetadata', 'TrialData', 'FittingData',

    # Loading
    'load_experiment', 'load_session_csv', 'load_animal',

    # Session selection
    'select_sessions', 'SessionFilter',
    'fitting_data_from_sessions',
    'register_preset', 'list_presets', 'register_presets_from_config',

    # Trial filtering
    'filter_trials', 'pool_arrays',
    'build_mask', 'opto_mask',
    'filter_session', 'filter_trial_data', 'get_arrays',

    # Synthetic
    'generate_synthetic_animal', 'generate_synthetic_session',
    'sample_stimuli', 'noisy_psychometric_simulator',


    # Analysis: low-level
    'fit_psychometric', 'compute_psychometric_gof',
    'compute_update_matrix', 'matrix_error',
    'compare_conditions', 'permutation_test_params', 'bootstrap_param_diff',
    'compute_summary_stats', 'compute_summary_stats_per_session',
    'list_available_stats', 'register_stat',
    'compute_session_features',
    'cumulative_gaussian', 'generate_stimuli',

    # Analysis: session-level
    'compute_psychometric',
    'compute_um',
    'compute_trajectory',
    'compute_comparison',
    'compute_session_raster',

    # Plotting
    'plot_psychometric', 'plot_um', 'plot_trajectory',
    'plot_comparison', 'plot_session_raster',
    'PALETTE', 'COLOURS', 'UM_CMAP',
    'apply_style', 'get_colour',
]
