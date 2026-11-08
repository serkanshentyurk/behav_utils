"""
behav_utils — analysis library for 2-AFC behavioural data.

Config-driven: distributions, session types and presets come from the YAML
you pass to ``load_experiment``; the library itself knows no experiment.

Pipeline (fixed order):

    experiment = load_experiment('config.yaml')
    sessions   = select_sessions(animal, preset='expert')     # which sessions
    clean      = filter_trials(sessions, trial_type='non_opto')  # which trials
    arrays     = TrialArrays.from_sessions(clean)              # TrialArrays

    s     = compute_stats(arrays, ['accuracy', *PSYCHOMETRIC]) # scalars  → pd.Series
    curve = compute_psychometric_curve(arrays)                 # readout  → PsychometricCurve
    plot_psychometric_curve(curve, ax=ax)                      # draw-only

    r = compute_phase_stats(clean, ['accuracy', 'mu'], per_session=True)   # pooled + per-session
    d = compute_delta_stat({'off': a, 'on': b}, ['mu'], reference='off')   # contrast with resampling

Modules:
    behav_utils.data        — structures, loading, selection, filtering, synthetic
    behav_utils.stats       — scalar statistic registry (TrialArrays → Series)
    behav_utils.readouts    — array-valued readouts as dataclasses
    behav_utils.analysis    — phase statistics, contrasts, resampling, group tests
    behav_utils.plotting    — one draw-only plot_x per compute_x
"""

# ── Analysis ─────────────────────────────────────────────────────────────────
from behav_utils.analysis.comparison import (
    DeltaStats,
    Interaction,
    compute_delta_stat,
    compute_interaction,
)
from behav_utils.analysis.group import (
    GroupComparison,
    RankTest,
    UnitBootstrap,
    collect_rows,
    compare_groups,
    rank_test,
)
from behav_utils.analysis.phase import PhaseStats, compute_phase_stats
from behav_utils.analysis.rolling import RollingStats, compute_rolling_stats

# ── Config ───────────────────────────────────────────────────────────────────
from behav_utils.config.schema import ProjectConfig, TaskConfig, load_cohorts, load_config

# ── Analysis: low-level (arrays) ─────────────────────────────────────────────
from behav_utils.data.arrays import TrialArrays, trial_key_of

# ── Loading ──────────────────────────────────────────────────────────────────
from behav_utils.data.loading import apply_session_type, load_animal, load_experiment, load_session_csv

# ── Trial filtering ──────────────────────────────────────────────────────────
from behav_utils.data.ops.filtering import (
    build_mask,
    filter_session,
    filter_trial_data,
    filter_trials,
    get_arrays,
    opto_mask,
    pool_arrays,
)

# ── Session selection ────────────────────────────────────────────────────────
from behav_utils.data.ops.selection import (
    SessionFilter,
    list_presets,
    preset_from_spec,
    register_preset,
    register_presets_from_config,
    select_sessions,
)
from behav_utils.data.ops.switches import Block, block_after, block_before, find_blocks, find_switches

# ── Data structures ──────────────────────────────────────────────────────────
from behav_utils.data.structures import (
    AnimalData,
    ExperimentData,
    SessionData,
    SessionMetadata,
    SessionSummary,
    TrialData,
)

# ── Synthetic data ───────────────────────────────────────────────────────────
from behav_utils.data.synthetic import (
    generate_stimuli,
    generate_synthetic_animal,
    generate_synthetic_session,
    noisy_psychometric_simulator,
    sample_stimuli,
)
from behav_utils.plotting import (
    COLOURS,
    PALETTE,
    UM_CMAP,
    apply_style,
    get_colour,
    plot_binned_curve,
    plot_comparison,
    plot_conditional_psychometric,
    plot_interaction,
    plot_interaction_grid,
    plot_psychometric_curve,
    plot_rolling_stats,
    plot_sd_profile,
    plot_stat_comparison,
    plot_stat_comparison_grid,
    plot_trajectory,
    plot_update_matrix,
)
from behav_utils.readouts import (
    BinnedCurve,
    ConditionalPsychometric,
    PsychometricCurve,
    SerialDependenceProfile,
    UpdateMatrix,
    compute_binned_curve,
    compute_conditional_psychometric,
    compute_psychometric_curve,
    compute_sd_profile,
    compute_update_matrix,
)
from behav_utils.readouts.fit import (
    cumulative_gaussian,
    fit_psychometric,
    fit_psychometric_gof,
    fit_update_matrix,
    matrix_error,
)
from behav_utils.stats import PSYCHOMETRIC, compute_stats, is_exchangeable, list_stats

__version__ = '0.10.0'

__all__ = [
    # Config
    'load_config', 'load_cohorts', 'ProjectConfig', 'TaskConfig',

    # Structures
    'ExperimentData', 'AnimalData', 'SessionData',
    'SessionMetadata', 'SessionSummary', 'TrialData',

    # Loading
    'load_experiment', 'load_session_csv', 'load_animal', 'apply_session_type',

    # Session selection
    'select_sessions', 'SessionFilter',
    'register_preset', 'list_presets', 'register_presets_from_config', 'preset_from_spec',
    'find_switches', 'find_blocks', 'block_before', 'block_after', 'Block',

    # Trial filtering
    'filter_trials', 'pool_arrays',
    'build_mask', 'opto_mask',
    'filter_session', 'filter_trial_data', 'get_arrays',

    # Synthetic
    'generate_synthetic_animal', 'generate_synthetic_session',
    'sample_stimuli', 'noisy_psychometric_simulator',


    # Arrays, stats, readouts
    'TrialArrays', 'trial_key_of', 'compute_stats', 'list_stats', 'is_exchangeable', 'PSYCHOMETRIC',
    'compute_psychometric_curve', 'compute_update_matrix', 'compute_conditional_psychometric',
    'compute_binned_curve', 'compute_sd_profile',
    'PsychometricCurve', 'UpdateMatrix', 'ConditionalPsychometric', 'BinnedCurve',
    'SerialDependenceProfile',
    # Analysis
    'fit_psychometric', 'fit_psychometric_gof', 'fit_update_matrix', 'matrix_error',
    'PhaseStats', 'compute_phase_stats',
    'DeltaStats', 'Interaction', 'compute_delta_stat', 'compute_interaction',
    'RollingStats', 'compute_rolling_stats',
    'collect_rows', 'compare_groups', 'rank_test', 'GroupComparison', 'RankTest', 'UnitBootstrap',
    'cumulative_gaussian', 'generate_stimuli',
    # Plotting
    'plot_psychometric_curve', 'plot_update_matrix', 'plot_conditional_psychometric',
    'plot_binned_curve', 'plot_sd_profile', 'plot_trajectory', 'plot_rolling_stats',
    'plot_comparison', 'plot_stat_comparison', 'plot_stat_comparison_grid',
    'plot_interaction', 'plot_interaction_grid',
    'PALETTE', 'COLOURS', 'UM_CMAP', 'apply_style', 'get_colour',
]
