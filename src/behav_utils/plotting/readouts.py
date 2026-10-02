"""
Draw-only plotters for readout dataclasses. One ``plot_x(result, ax=None)`` per
readout; every function returns ``(fig, ax)`` and computes nothing.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from behav_utils.plotting.styles import (
    COLOURS,
    DEFAULT_ALPHA,
    DEFAULT_LINE_WIDTH,
    DEFAULT_MARKER_SIZE,
    PALETTE,
    SEM_ALPHA,
    UM_CMAP,
)
from behav_utils.readouts import (
    BinnedCurve,
    ConditionalPsychometric,
    PsychometricCurve,
    SerialDependenceProfile,
    UpdateMatrix,
)


def _axes(ax: plt.Axes | None, figsize) -> Tuple[plt.Figure, plt.Axes]:
    if ax is None:
        return plt.subplots(1, 1, figsize=figsize)
    return ax.get_figure(), ax


def _stimulus_axes(ax: plt.Axes, ylabel: str, reference: bool) -> None:
    if reference:
        ax.axhline(0.5, ls='--', color='grey', alpha=0.3, zorder=0)
        ax.axvline(0.0, ls='--', color='grey', alpha=0.3, zorder=0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel(ylabel)


def plot_psychometric_curve(
    curve: PsychometricCurve,
    ax: plt.Axes | None = None,
    *,
    color: str | None = None,
    label: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    linewidth: float = DEFAULT_LINE_WIDTH,
    linestyle: str = '-',
    show_data: bool = True,
    show_band: bool = True,
    show_params: bool = False,
    show_lapse: bool = False,
    show_reference: bool = True,
    title: str = '',
) -> Tuple[plt.Figure, plt.Axes]:
    """Fitted curve, binned data points, and the bootstrap band if present."""
    fig, ax = _axes(ax, (5, 4))
    color = color or COLOURS.get('default', PALETTE[0])

    if curve.success:
        lbl = label
        if show_params:
            lbl = f"{label or ''} (PSE={curve.mu:.2f}, \u03c3={curve.sigma:.2f})".strip()
        ax.plot(curve.x, curve.y, color=color, lw=linewidth, ls=linestyle, alpha=alpha,
                label=lbl, zorder=2)
    if show_data:
        v = ~np.isnan(curve.bin_means)
        ax.plot(curve.bin_centres[v], curve.bin_means[v], 'o', color=color,
                markersize=DEFAULT_MARKER_SIZE, alpha=alpha * 0.7, zorder=3,
                label=None if curve.success else label)
    if show_band and curve.band is not None:
        ax.fill_between(curve.x, curve.band[0], curve.band[1], color=color, alpha=SEM_ALPHA, zorder=1)
    if show_lapse and curve.success:
        ax.axhline(curve.lapse_low, color='grey', ls=':', alpha=0.4)
        ax.axhline(1 - curve.lapse_high, color='grey', ls=':', alpha=0.4)

    _stimulus_axes(ax, 'P(choose B)', show_reference)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_update_matrix(
    um: UpdateMatrix,
    ax: plt.Axes | None = None,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap=None,
    colorbar: bool = True,
    title: str = '',
    xlabel: str = 'Previous stimulus',
    ylabel: str = 'Current stimulus',
) -> Tuple[plt.Figure, plt.Axes]:
    """Heatmap of the shift matrix (symmetric colour scale unless given)."""
    fig, ax = _axes(ax, (4.5, 4))
    m = um.matrix
    if vmin is None or vmax is None:
        with np.errstate(all='ignore'):
            abs_max = np.nanmax(np.abs(m)) if np.isfinite(m).any() else 0.0
        abs_max = max(float(abs_max), 0.01)
        vmin = -abs_max if vmin is None else vmin
        vmax = abs_max if vmax is None else vmax
    im = ax.imshow(m, cmap=cmap or UM_CMAP, vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046)
    ticks = [f'{c:.1f}' for c in um.centres]
    ax.set_xticks(range(um.n_bins))
    ax.set_xticklabels(ticks, fontsize=7, rotation=45)
    ax.set_yticks(range(um.n_bins))
    ax.set_yticklabels(ticks, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_conditional_psychometric(
    cp: ConditionalPsychometric,
    ax: plt.Axes | None = None,
    *,
    param: str = 'mu',
    color: str | None = None,
    label: str | None = None,
    show_unconditional: bool = True,
    title: str = '',
) -> Tuple[plt.Figure, plt.Axes]:
    """One parameter as a function of the previous-stimulus bin; hollow markers where it fell back."""
    from behav_utils.readouts.psychometric import PARAMS
    fig, ax = _axes(ax, (5, 3.5))
    color = color or COLOURS.get('default', PALETTE[0])
    k = list(PARAMS).index(param)
    y = cp.params[:, k]
    fb = cp.fell_back
    ax.plot(cp.centres, y, '-', color=color, lw=DEFAULT_LINE_WIDTH, alpha=DEFAULT_ALPHA, label=label)
    ax.plot(cp.centres[~fb], y[~fb], 'o', color=color, markersize=DEFAULT_MARKER_SIZE)
    ax.plot(cp.centres[fb], y[fb], 'o', mfc='none', mec=color, markersize=DEFAULT_MARKER_SIZE)
    if show_unconditional and cp.success:
        ax.axhline(cp.unconditional[k], ls='--', color='grey', alpha=0.4, zorder=0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel('Previous stimulus')
    ax.set_ylabel(param)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_binned_curve(
    curve: BinnedCurve,
    ax: plt.Axes | None = None,
    *,
    color: str | None = None,
    label: str | None = None,
    show_reference: bool = True,
    title: str = '',
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = _axes(ax, (5, 4))
    color = color or COLOURS.get('default', PALETTE[0])
    v = ~np.isnan(curve.values)
    ax.plot(curve.centres[v], curve.values[v], 'o-', color=color, lw=DEFAULT_LINE_WIDTH,
            markersize=DEFAULT_MARKER_SIZE, alpha=DEFAULT_ALPHA, label=label)
    _stimulus_axes(ax, 'P(choose B)' if curve.kind == 'choice_prob' else 'P(correct)',
                   show_reference and curve.kind == 'choice_prob')
    if title:
        ax.set_title(title)
    return fig, ax


def plot_sd_profile(
    prof: SerialDependenceProfile,
    ax: plt.Axes | None = None,
    *,
    color: str | None = None,
    label: str | None = None,
    title: str = '',
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = _axes(ax, (5, 3.5))
    color = color or COLOURS.get('default', PALETTE[0])
    v = ~np.isnan(prof.profile)
    ax.axhline(0, ls='--', color='grey', alpha=0.3, zorder=0)
    ax.plot(prof.centres[v], prof.profile[v], 'o-', color=color, lw=DEFAULT_LINE_WIDTH,
            markersize=DEFAULT_MARKER_SIZE, alpha=DEFAULT_ALPHA, label=label)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel('Previous stimulus')
    ax.set_ylabel('\u0394 P(choose B)')
    if title:
        ax.set_title(title)
    return fig, ax
