"""
behav_utils — trial-based behavioural analysis.

Load anyone's trial data (readers), score it by its task (``TwoAFC``, ``GoNoGo``, ``Reproduction``, or
yours), describe it with statistics and readouts, compare phases and groups with resampling, and simulate
any model through the same pipeline. Config-driven: the library itself knows no experiment.

    experiment = load_experiment('config.yaml')
    sessions   = select_sessions(animal, preset='expert')          # which sessions
    clean      = filter_trials(sessions, trial_type='non_opto')     # which trials
    arrays     = TrialArrays.from_sessions(clean)

    s     = compute_stats(arrays, ['accuracy', 'mu', 'sigma'])      # scalars → pd.Series
    curve = compute_psychometric_curve(arrays)                     # readout
    plot_psychometric_curve(curve, ax=ax)                          # draw-only

    r = compute_phase_stats(clean, ['accuracy', 'mu'], per_session=True)
    d = compute_delta_stat({'off': a, 'on': b}, ['mu'], reference='off')
    experiment.save('cohort1')                                       # a store; ExperimentData.load('cohort1')

Public API: the names in ``__all__`` here (the everyday ones) and in the ``__all__`` of each sub-package
(everything public, each name with one home sub-package). Module paths below a sub-package and names
starting with ``_`` are private. Public names change only in a major release, after a minor release in
which they warn (CONTRIBUTING.md). The full list, with signatures, is ``docs/public_api.md``.

Sub-packages:
    behav_utils.config      — YAML schema and loader
    behav_utils.tasks       — tasks (what counts as correct), capabilities, stimulus distributions
    behav_utils.data        — structures, readers, discovery, loading, selection, filtering, the store
    behav_utils.stats       — the scalar statistic registry (TrialArrays → Series)
    behav_utils.readouts    — array-valued readouts as dataclasses, and the fit engines
    behav_utils.models      — behavioural models, designs, simulate() and synthetic data
    behav_utils.analysis    — phase statistics, contrasts, resampling, rolling statistics, group tests
    behav_utils.plotting    — one draw-only plot_x per readout or result
"""

from behav_utils.analysis import (
    compare_groups,
    compute_delta_stat,
    compute_interaction,
    compute_phase_stats,
    compute_rolling_stats,
)
from behav_utils.config import ConfigError, ProjectConfig, load_config
from behav_utils.data import (
    AnimalData,
    ExperimentData,
    SessionData,
    SessionFilter,
    StoreError,
    TrialArrays,
    TrialData,
    build_mask,
    discover_sessions,
    filter_trials,
    find_blocks,
    find_switches,
    list_presets,
    load_experiment,
    load_store,
    opto_mask,
    pool_arrays,
    read_manifest,
    register_preset,
    save_store,
    select_sessions,
)
from behav_utils.models import (
    IdealObserver,
    PsychometricModel,
    RandomModel,
    ReproductionModel,
    generate_synthetic_animal,
    generate_synthetic_experiment,
    generate_synthetic_session,
    make_design,
    simulate,
)
from behav_utils.plotting import (
    apply_style,
    plot_binned_curve,
    plot_conditional_psychometric,
    plot_error_distribution,
    plot_interaction,
    plot_interaction_grid,
    plot_psychometric_comparison,
    plot_psychometric_curve,
    plot_response_curve,
    plot_rolling_stats,
    plot_sd_profile,
    plot_serial_bias,
    plot_stat_comparison,
    plot_stat_comparison_grid,
    plot_trajectory,
    plot_update_matrix,
)
from behav_utils.readouts import (
    compute_binned_curve,
    compute_conditional_psychometric,
    compute_error_distribution,
    compute_psychometric_curve,
    compute_response_curve,
    compute_sd_profile,
    compute_serial_bias,
    compute_update_matrix,
    fit_psychometric,
)
from behav_utils.stats import compute_stats, list_stats
from behav_utils.tasks import GoNoGo, Outcome, Reproduction, RequirementError, Task, TwoAFC

__version__ = '1.0.1'

__all__ = [
    # Config
    'load_config', 'ProjectConfig', 'ConfigError',
    # Tasks: what counts as correct
    'Task', 'TwoAFC', 'GoNoGo', 'Reproduction', 'Outcome', 'RequirementError',
    # Data: structures, loading, discovery, selection, filtering, the store
    'ExperimentData', 'AnimalData', 'SessionData', 'TrialData', 'TrialArrays',
    'load_experiment', 'discover_sessions',
    'select_sessions', 'SessionFilter', 'register_preset', 'list_presets',
    'filter_trials', 'build_mask', 'opto_mask', 'pool_arrays', 'find_blocks', 'find_switches',
    'save_store', 'load_store', 'read_manifest', 'StoreError',
    # Statistics
    'compute_stats', 'list_stats',
    # Readouts
    'fit_psychometric', 'compute_psychometric_curve', 'compute_update_matrix', 'compute_conditional_psychometric',
    'compute_binned_curve', 'compute_sd_profile', 'compute_response_curve', 'compute_error_distribution',
    'compute_serial_bias',
    # Models and simulation
    'simulate', 'make_design', 'PsychometricModel', 'RandomModel', 'IdealObserver', 'ReproductionModel',
    'generate_synthetic_session', 'generate_synthetic_animal', 'generate_synthetic_experiment',
    # Analysis
    'compute_phase_stats', 'compute_delta_stat', 'compute_interaction', 'compute_rolling_stats', 'compare_groups',
    # Plotting
    'plot_psychometric_curve', 'plot_update_matrix', 'plot_conditional_psychometric', 'plot_binned_curve',
    'plot_sd_profile', 'plot_response_curve', 'plot_error_distribution', 'plot_serial_bias',
    'plot_trajectory', 'plot_rolling_stats', 'plot_psychometric_comparison', 'plot_stat_comparison',
    'plot_stat_comparison_grid', 'plot_interaction', 'plot_interaction_grid', 'apply_style',
]
