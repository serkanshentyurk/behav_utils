"""
Stat Trajectory Plotting

Plot how summary statistics evolve across sessions for single animals
or groups of animals. Supports individual traces, mean/SEM, median/IQR.

All functions return (fig, ax) for further customisation.

Usage:
    from behav_utils.plotting.trajectory import (
        plot_stat_trajectory,
        plot_multi_animal_trajectory,
    )

    # Single animal
    fig, ax = plot_stat_trajectory(session_indices, values, ylabel='Accuracy')

    # Multi-animal (called by experiment.plot_trajectory)
    fig, ax = plot_multi_animal_trajectory(
        animals, stat='accuracy', combine='mean_sem',
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple, TYPE_CHECKING

from behav_utils.plotting.styles import (
    COLOURS, SEM_ALPHA, DEFAULT_ALPHA, DEFAULT_LINE_WIDTH,
    get_animal_colours,
)

if TYPE_CHECKING:
    from behav_utils.data.structures import AnimalData


# =============================================================================
# SINGLE TRAJECTORY
# =============================================================================

def plot_stat_trajectory(
    session_indices: np.ndarray,
    values: np.ndarray,
    title: str = '',
    ylabel: str = '',
    xlabel: str = 'Session',
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show_points: bool = True,
    marker: str = 'o',
    linewidth: float = DEFAULT_LINE_WIDTH,
    alpha: float = DEFAULT_ALPHA,
    label: Optional[str] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a single stat trajectory across sessions.

    Args:
        session_indices: x-axis values (session numbers)
        values: y-axis values (stat values)
        title: Plot title
        ylabel: Y-axis label
        xlabel: X-axis label
        color: Line colour (default: COLOURS['default'])
        ax: Existing axes (creates new figure if None)
        show_points: Whether to show markers at each session
        label: Legend label

    Returns:
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    if color is None:
        color = COLOURS['default']

    # Filter NaNs for plotting
    valid = ~np.isnan(values.astype(float))

    plot_kwargs = dict(color=color, linewidth=linewidth, alpha=alpha)
    if label is not None:
        plot_kwargs['label'] = label

    if show_points:
        ax.plot(session_indices[valid], values[valid],
                f'{marker}-', markersize=4, **plot_kwargs, **kwargs)
    else:
        ax.plot(session_indices[valid], values[valid],
                '-', **plot_kwargs, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return fig, ax


# =============================================================================
# MULTI-ANIMAL TRAJECTORY
# =============================================================================

def plot_multi_animal_trajectory(
    animals: List['AnimalData'],
    stat: str,
    stage: Optional[str] = None,
    combine: str = 'mean_sem',
    show_individual: bool = True,
    individual_alpha: float = 0.3,
    individual_linewidth: float = 0.8,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: str = 'Session',
    ax: Optional[plt.Axes] = None,
    colour_by: str = 'animal',
    manipulation_lines: bool = False,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot stat trajectories for multiple animals with group summary.

    Args:
        animals: List of AnimalData objects
        stat: Feature name to plot
        stage: Stage filter for sessions
        combine: Group summary method:
            'mean_sem' — mean line with SEM shading
            'median_iqr' — median line with IQR shading
            'mean_only' — just the mean line
            'none' — no group summary
        show_individual: Show per-animal traces
        individual_alpha: Alpha for individual traces
        title: Plot title (default: stat name)
        ylabel: Y-axis label (default: stat name)
        colour_by: 'animal' (each animal different colour) or
                   'manipulation' (colour by manipulation type)
        manipulation_lines: Draw vertical lines at manipulation sessions

    Returns:
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    if title is None:
        title = stat
    if ylabel is None:
        ylabel = stat

    # Get trajectories
    trajectories = []  # (indices, values, animal_id, animal)
    for animal in animals:
        try:
            idx, vals = animal.stat_trajectory(stat, stage=stage)
            trajectories.append((idx, vals, animal.animal_id, animal))
        except (ValueError, KeyError):
            continue

    if not trajectories:
        ax.text(0.5, 0.5, f'No data for stat "{stat}"',
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title)
        return fig, ax

    # Colour assignment
    if colour_by == 'animal':
        animal_ids = [t[2] for t in trajectories]
        colours = get_animal_colours(animal_ids)
    else:
        colours = {}

    # Plot individual traces
    if show_individual:
        for idx, vals, aid, animal in trajectories:
            color = colours.get(aid, COLOURS['default'])
            valid = ~np.isnan(vals.astype(float))
            ax.plot(idx[valid], vals[valid], '-',
                    color=color, alpha=individual_alpha,
                    linewidth=individual_linewidth)

    # Compute group summary
    if combine != 'none':
        # Align by session index (not all animals have same number of sessions)
        all_indices = set()
        for idx, _, _, _ in trajectories:
            all_indices.update(idx.astype(int))
        all_indices = sorted(all_indices)

        summary_x = []
        summary_mean = []
        summary_low = []
        summary_high = []

        for si in all_indices:
            vals_at_si = []
            for idx, vals, _, _ in trajectories:
                mask = idx.astype(int) == si
                if mask.any():
                    v = vals[mask][0]
                    if not np.isnan(v):
                        vals_at_si.append(v)

            if len(vals_at_si) < 2:
                continue

            arr = np.array(vals_at_si)
            summary_x.append(si)

            if combine in ('mean_sem', 'mean_only'):
                summary_mean.append(np.mean(arr))
                summary_low.append(np.mean(arr) - np.std(arr) / np.sqrt(len(arr)))
                summary_high.append(np.mean(arr) + np.std(arr) / np.sqrt(len(arr)))
            elif combine == 'median_iqr':
                summary_mean.append(np.median(arr))
                summary_low.append(np.percentile(arr, 25))
                summary_high.append(np.percentile(arr, 75))

        summary_x = np.array(summary_x)
        summary_mean = np.array(summary_mean)
        summary_low = np.array(summary_low)
        summary_high = np.array(summary_high)

        # Plot summary
        ax.plot(summary_x, summary_mean, '-',
                color=COLOURS['mean_line'], linewidth=2.5, zorder=10,
                label=combine.replace('_', ' '))

        if combine != 'mean_only':
            ax.fill_between(summary_x, summary_low, summary_high,
                            color=COLOURS['sem_fill'], alpha=SEM_ALPHA, zorder=5)

    # Manipulation lines
    if manipulation_lines:
        for _, _, aid, animal in trajectories:
            manip = animal.metadata.get('manipulation_session', None)
            if manip is not None:
                ax.axvline(manip, color='red', linestyle='--', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if combine != 'none':
        ax.legend(fontsize=8)

    return fig, ax


# =============================================================================
# MULTI-STAT GRID
# =============================================================================

def plot_stat_grid(
    animals: List['AnimalData'],
    stats: List[str],
    stage: Optional[str] = None,
    combine: str = 'mean_sem',
    n_cols: int = 4,
    figsize_per_panel: Tuple[float, float] = (4.0, 3.0),
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Grid of stat trajectories — one panel per stat.

    Args:
        animals: List of AnimalData
        stats: List of stat names
        stage: Stage filter
        combine: Group summary method
        n_cols: Columns in grid

    Returns:
        (fig, axes_array)
    """
    n_stats = len(stats)
    n_rows = int(np.ceil(n_stats / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, stat in enumerate(stats):
        plot_multi_animal_trajectory(
            animals, stat=stat, stage=stage,
            combine=combine, ax=axes_flat[i],
            title=stat, **kwargs,
        )

    # Hide empty panels
    for j in range(n_stats, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f'Summary Stat Trajectories ({len(animals)} animals)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    return fig, axes
