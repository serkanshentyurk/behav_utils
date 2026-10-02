"""
behav_utils.plotting — draw-only plotters.

Every plotter takes a typed result from the matching compute_ function and
computes nothing:

    compute_psychometric_curve(arrays)             → plot_psychometric_curve(curve)
    compute_update_matrix(arrays)                  → plot_update_matrix(um)
    compute_phase_stats(sessions, per_session=True)→ plot_trajectory(result, stat)
    compute_delta_stat({...})                      → plot_comparison / plot_stat_comparison
    compute_interaction(r1, r2, key)               → plot_interaction

Layout is the caller's: pass ``ax``. Overlay by calling the same plotter twice
on one axes with different colours.
"""

from behav_utils.plotting.comparison import (
    plot_comparison,
    plot_interaction,
    plot_interaction_single,
    plot_stat_comparison,
    plot_stat_comparison_single,
)
from behav_utils.plotting.readouts import (
    plot_binned_curve,
    plot_conditional_psychometric,
    plot_psychometric_curve,
    plot_sd_profile,
    plot_update_matrix,
)
from behav_utils.plotting.styles import COLOURS, PALETTE, UM_CMAP, apply_style, get_colour
from behav_utils.plotting.trajectory import plot_trajectory

__all__ = [
    'plot_psychometric_curve', 'plot_update_matrix', 'plot_conditional_psychometric',
    'plot_binned_curve', 'plot_sd_profile',
    'plot_trajectory',
    'plot_comparison', 'plot_stat_comparison', 'plot_stat_comparison_single',
    'plot_interaction', 'plot_interaction_single',
    'PALETTE', 'COLOURS', 'UM_CMAP', 'apply_style', 'get_colour',
]
