"""
Trajectory Plotting

plot_trajectory(result, stat, ax=None)

Draw-only. Takes a PhaseStats from compute_phase_stats(sessions, names, per_session=True).

Usage:
    r = compute_phase_stats(sessions, ['accuracy', 'mu'], per_session=True)
    fig, axes = plt.subplots(1, 2)
    plot_trajectory(r, 'accuracy', ax=axes[0])
    plot_trajectory(r, 'mu', ax=axes[1])
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from behav_utils.analysis.phase import PhaseStats
from behav_utils.plotting.styles import (
    DEFAULT_ALPHA,
    DEFAULT_LINE_WIDTH,
    DEFAULT_MARKER_SIZE,
    PALETTE,
)

# Map dict-key (math name) → display label (literature name)
_DISPLAY_LABEL = {
    'mu':         'PSE',
    'sigma':      'slope',
    'lapse_low':  'λ_low',
    'lapse_high': 'λ_high',
    'accuracy':   'Accuracy',
}


def plot_trajectory(
    result: PhaseStats,
    stat: str,
    ax: plt.Axes | None = None,
    color: str | None = None,
    label: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    linewidth: float = DEFAULT_LINE_WIDTH,
    marker: str = 'o',
    markersize: float = DEFAULT_MARKER_SIZE,
    linestyle: str = '-',
    show_distribution_boundaries: bool = True,
    title: str = '',
) -> Tuple[plt.Figure, plt.Axes]:
    """One stat per session, in session order, from ``compute_phase_stats(..., per_session=True)``.

    Args:
        result: :class:`PhaseStats` with ``sessions`` filled.
        stat:   which stat to draw.
        show_distribution_boundaries: vertical lines where ``distribution`` changes.
    """
    if result.sessions is None:
        raise ValueError('plot_trajectory needs compute_phase_stats(..., per_session=True)')
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    else:
        fig = ax.get_figure()
    color = color or PALETTE[0]

    frame = result.sessions[result.sessions['stat'] == stat]
    x = np.asarray(frame['session'], dtype=float)
    ax.plot(x, frame['value'].to_numpy(), marker=marker, ms=markersize, ls=linestyle,
            lw=linewidth, color=color, alpha=alpha, label=label, zorder=2)

    if show_distribution_boundaries:
        dists = list(frame['distribution'])
        for i in range(1, len(dists)):
            if dists[i - 1] and dists[i] and dists[i - 1] != dists[i]:
                ax.axvline((x[i - 1] + x[i]) / 2, ls=':', color='grey', alpha=0.5, zorder=0)

    ax.set_xlabel('Session')
    ax.set_ylabel(_DISPLAY_LABEL.get(stat, stat))
    if title:
        ax.set_title(title)
    return fig, ax
