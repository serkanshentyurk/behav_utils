"""
behav_utils.analysis — phase-level statistics, contrasts, resampling, group tests.

    phase = filter_trials(select_sessions(animal, preset='expert'))
    r  = compute_phase_stats(phase, ['accuracy', *PSYCHOMETRIC], per_session=True)
    d  = compute_delta_stat({'off': a, 'on': b}, ['mu', 'sigma'], reference='off')
    ix = compute_interaction(d_treated, d_control, 'on_vs_off')

Scalar statistics themselves live in ``behav_utils.stats``; array-valued readouts and the two fit
engines on raw arrays (``fit_psychometric``, ``fit_update_matrix``, in ``readouts/fit.py``) in
``behav_utils.readouts``; the fit engines are re-exported here for convenience.

Files: ``phase.py`` (compute_phase_stats), ``comparison.py`` (contrasts), ``resampling.py`` (the draw
engine and draw summaries), ``group.py`` (across-animal tests), ``rolling.py``, ``session_features.py``.
"""

from behav_utils.analysis.comparison import (
    Contrast,
    DeltaStats,
    Interaction,
    PhaseSummary,
    compute_delta_stat,
    compute_interaction,
    contrast_key,
)
from behav_utils.analysis.group import (
    average_arrays,
    bootstrap_units,
    collect_rows,
    combine,
    compare_groups,
    min_achievable_p,
    paired_diff,
    rank_test,
)
from behav_utils.analysis.phase import PhaseStats, compute_phase_stats, infer_animal_id
from behav_utils.analysis.resampling import (
    DrawSummary,
    bootstrap_phase_stats,
    calculate_min_n,
    downsample,
    permute_phase_difference,
    resample_psychometric_curve,
    resample_stat_vectors,
    resample_update_matrix,
    summarise_draw_frame,
    summarise_draws,
)
from behav_utils.analysis.rolling import RollingStats, compute_rolling_stats
from behav_utils.analysis.session_features import compute_session_features
from behav_utils.data.synthetic import generate_stimuli
from behav_utils.readouts.fit import (
    cumulative_gaussian,
    fit_psychometric,
    fit_psychometric_gof,
    fit_update_matrix,
    matrix_error,
)

__all__ = [
    'cumulative_gaussian', 'generate_stimuli',
    'fit_psychometric', 'fit_psychometric_gof', 'fit_update_matrix', 'matrix_error',
    'PhaseStats', 'compute_phase_stats', 'infer_animal_id',
    'DeltaStats', 'PhaseSummary', 'Contrast', 'Interaction',
    'compute_delta_stat', 'compute_interaction', 'contrast_key',
    'bootstrap_phase_stats', 'permute_phase_difference', 'summarise_draws',
    'summarise_draw_frame', 'DrawSummary',
    'downsample', 'calculate_min_n', 'resample_stat_vectors',
    'resample_psychometric_curve', 'resample_update_matrix',
    'RollingStats', 'compute_rolling_stats',
    'collect_rows', 'compare_groups',
    'combine', 'paired_diff', 'bootstrap_units', 'rank_test', 'average_arrays', 'min_achievable_p',
    'compute_session_features',
]
