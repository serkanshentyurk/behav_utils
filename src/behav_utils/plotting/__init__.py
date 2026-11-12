"""
behav_utils.plotting — draw-only plotters.

Every plotter takes a typed result from the matching compute_ function and
computes nothing; every one is single-panel and takes ``ax``:

    compute_psychometric_curve(arrays)             → plot_psychometric_curve(curve, ax=)
    compute_update_matrix(arrays)                  → plot_update_matrix(um, ax=)
    compute_conditional_psychometric(arrays)       → plot_conditional_psychometric(cp, ax=)
    compute_binned_curve(arrays)                   → plot_binned_curve(bc, ax=)
    compute_sd_profile(arrays)                     → plot_sd_profile(sd, ax=)
    compute_phase_stats(sessions, per_session=True)→ plot_trajectory(result, stat, ax=)
    compute_rolling_stats(sessions, names)         → plot_rolling_stats(result, stat, ax=)
    compute_delta_stat({...})                      → plot_psychometric_comparison(d, ax=) / plot_stat_comparison(d, stat, ax=)
    compute_interaction(r1, r2, key)               → plot_interaction(ix, stat, ax=)

Layout is the caller's: pass ``ax``. Overlay by calling the same plotter twice
on one axes with different colours. The only exception: ``plot_stat_comparison_grid``
and ``plot_interaction_grid`` build a grid of axes and call the single-panel
plotter once per stat.
"""

from behav_utils.plotting.comparison import (
    plot_interaction,
    plot_interaction_grid,
    plot_psychometric_comparison,
    plot_stat_comparison,
    plot_stat_comparison_grid,
)
from behav_utils.plotting.continuous import plot_error_distribution, plot_response_curve, plot_serial_bias
from behav_utils.plotting.readouts import (
    plot_binned_curve,
    plot_conditional_psychometric,
    plot_psychometric_curve,
    plot_sd_profile,
    plot_update_matrix,
)
from behav_utils.plotting.styles import COLOURS, PALETTE, UM_CMAP, apply_style, get_colour
from behav_utils.plotting.trajectory import plot_rolling_stats, plot_trajectory

__all__ = [
    'plot_response_curve', 'plot_error_distribution', 'plot_serial_bias',
    'plot_psychometric_curve', 'plot_update_matrix', 'plot_conditional_psychometric',
    'plot_binned_curve', 'plot_sd_profile',
    'plot_trajectory', 'plot_rolling_stats',
    'plot_psychometric_comparison', 'plot_stat_comparison', 'plot_stat_comparison_grid',
    'plot_interaction', 'plot_interaction_grid',
    'PALETTE', 'COLOURS', 'UM_CMAP', 'apply_style', 'get_colour',
]
