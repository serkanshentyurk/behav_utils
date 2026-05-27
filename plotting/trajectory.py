"""
Trajectory Plotting

plot_trajectory(result, stat_name, ax, **kwargs)

Takes a result dict from compute_trajectory(). Does ZERO computation.
Just draws the trajectory line.

Usage:
    result = compute_trajectory(sessions, ['accuracy', 'mu'])
    fig, axes = plt.subplots(1, 2)
    plot_trajectory(result, 'accuracy', ax=axes[0])
    plot_trajectory(result, 'mu', ax=axes[1])
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from behav_utils.plotting.styles import (
    PALETTE, COLOURS, DEFAULT_ALPHA, DEFAULT_LINE_WIDTH, DEFAULT_MARKER_SIZE,
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
    result: dict,
    stat_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    alpha: float = DEFAULT_ALPHA,
    linewidth: float = DEFAULT_LINE_WIDTH,
    marker: str = 'o',
    markersize: float = DEFAULT_MARKER_SIZE,
    linestyle: str = '-',
    show_distribution_boundaries: bool = True,
    title: str = '',
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a stat trajectory from a compute_trajectory() result.

    Args:
        result: Dict from compute_trajectory().
        stat_name: Which stat to plot. If None, uses the first stat in the result.
        ax: Matplotlib axes (creates one if None).
        color: Line colour.
        label: Legend label.
        show_distribution_boundaries: Draw vertical lines at distribution changes.
        title: Axes title.

    Returns:
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    else:
        fig = ax.get_figure()

    color = color or PALETTE[0]

    if stat_name is None:
        stat_name = result['stat_names'][0]

    values = result['values'][stat_name]
    x = np.array(result['session_indices'])

    ax.plot(x, values, marker=marker, ms=markersize, ls=linestyle,
            lw=linewidth, color=color, alpha=alpha, label=label, zorder=2)

    # Distribution boundaries
    if show_distribution_boundaries:
        per_sess = result.get('per_session', [])
        for i in range(1, len(per_sess)):
            prev_dist = per_sess[i - 1].get('distribution')
            curr_dist = per_sess[i].get('distribution')
            if prev_dist and curr_dist and prev_dist != curr_dist:
                boundary_x = (x[i - 1] + x[i]) / 2
                ax.axvline(boundary_x, ls=':', color='grey', alpha=0.5, zorder=0)

    ax.set_xlabel('Session')
    ax.set_ylabel(_DISPLAY_LABEL.get(stat_name, stat_name))

    if title:
        ax.set_title(title)

    return fig, ax
