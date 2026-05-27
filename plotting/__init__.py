"""
behav_utils.plotting — Behavioural data visualisation.

All plotting functions take pre-computed result dicts from the
corresponding compute_ functions. No computation inside plotting.

    compute_psychometric(sessions) → plot_psychometric(result)
    compute_um(sessions)           → plot_um(result)
    compute_trajectory(sessions)   → plot_trajectory(result)
    compute_comparison(a, b)       → plot_comparison(result)
    compute_session_raster(sess)   → plot_session_raster(result)

User controls layout via plt.subplots(). Overlay = call the same
plot function twice on the same axes with different colours.

Usage:
    from behav_utils.analysis import compute_psychometric, compute_um
    from behav_utils.plotting import plot_psychometric, plot_um, PALETTE

    apply_style()

    psych = compute_psychometric(early_sessions, mode='pooled')
    fig, ax = plt.subplots()
    plot_psychometric(psych, ax=ax, color=PALETTE[0], label='Early')
"""

from behav_utils.plotting.psychometric import plot_psychometric
from behav_utils.plotting.update_matrix import plot_um
from behav_utils.plotting.trajectory import plot_trajectory
from behav_utils.plotting.comparison import plot_comparison
from behav_utils.plotting.session import plot_session_raster
from behav_utils.plotting.styles import (
    PALETTE, COLOURS, UM_CMAP,
    apply_style, get_colour,
)

__all__ = [
    'plot_psychometric',
    'plot_um',
    'plot_trajectory',
    'plot_comparison',
    'plot_session_raster',
    'PALETTE', 'COLOURS', 'UM_CMAP',
    'apply_style', 'get_colour',
]
