"""Draw-only plots for the continuous-response family's readouts: one panel each, ``plot_x(result, ax=None)``."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from behav_utils.plotting.readouts import _axes
from behav_utils.plotting.styles import (
    COLOURS,
    DEFAULT_ALPHA,
    DEFAULT_LINE_WIDTH,
    DEFAULT_MARKER_SIZE,
    PALETTE,
)
from behav_utils.readouts.continuous import ErrorDistribution, ResponseCurve, SerialBiasCurve


def _points(ax, x, y, sem, color, label) -> None:
    shown = ~np.isnan(y)
    ax.errorbar(x[shown], y[shown], yerr=np.nan_to_num(sem[shown]), fmt='o-', color=color, lw=DEFAULT_LINE_WIDTH,
                markersize=DEFAULT_MARKER_SIZE, alpha=DEFAULT_ALPHA, label=label, capsize=0)


def plot_response_curve(curve: ResponseCurve, ax: plt.Axes | None = None, *, color: str | None = None,
                        label: str | None = None, show_identity: bool = True,
                        title: str = '') -> Tuple[plt.Figure, plt.Axes]:
    """Mean response per stimulus bin (± SEM), with the identity line a faithful reproducer would follow."""
    fig, ax = _axes(ax, (5, 4))
    if show_identity:
        ax.plot([-1, 1], [-1, 1], ls='--', color='grey', alpha=0.4, zorder=0)
    _points(ax, curve.centres, curve.values, curve.sem, color or COLOURS.get('default', PALETTE[0]), label)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Response')
    if title:
        ax.set_title(title)
    return fig, ax


def plot_error_distribution(dist: ErrorDistribution, ax: plt.Axes | None = None, *, color: str | None = None,
                            label: str | None = None, title: str = '') -> Tuple[plt.Figure, plt.Axes]:
    """Density of signed errors (response − truth), with zero marked."""
    fig, ax = _axes(ax, (5, 4))
    ax.stairs(dist.density, dist.edges, fill=True, alpha=0.4, color=color or COLOURS.get('default', PALETTE[0]),
              label=label)
    ax.axvline(0.0, ls='--', color='grey', alpha=0.4, zorder=0)
    ax.set_xlabel('Error (response − truth)')
    ax.set_ylabel('Density')
    if title:
        ax.set_title(title)
    return fig, ax


def plot_serial_bias(curve: SerialBiasCurve, ax: plt.Axes | None = None, *, color: str | None = None,
                     label: str | None = None, title: str = '') -> Tuple[plt.Figure, plt.Axes]:
    """Mean error (current stimulus's effect removed) per bin of previous − current stimulus, ± SEM."""
    fig, ax = _axes(ax, (5, 4))
    ax.axhline(0.0, ls='--', color='grey', alpha=0.4, zorder=0)
    ax.axvline(0.0, ls='--', color='grey', alpha=0.4, zorder=0)
    _points(ax, curve.centres, curve.values, curve.sem, color or COLOURS.get('default', PALETTE[0]), label)
    ax.set_xlim(-2.1, 2.1)
    ax.set_xlabel('Previous − current stimulus')
    ax.set_ylabel('Error')
    if title:
        ax.set_title(title)
    return fig, ax
