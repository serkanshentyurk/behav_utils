"""
Psychometric Curve Plotting

plot_psychometric(result, ax, **kwargs)

Takes a result dict from compute_psychometric(). Does ZERO computation.
Just draws the pre-computed curves, data points, and CI bands.

Usage:
    from behav_utils.analysis.psychometry import compute_psychometric
    from behav_utils.plotting.psychometric import plot_psychometric

    result = compute_psychometric(filtered_sessions, mode='pooled', n_bootstrap=200)
    fig, ax = plt.subplots()
    plot_psychometric(result, ax=ax, color=PALETTE[0], label='Expert')
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from behav_utils.plotting.styles import (
    PALETTE, COLOURS, DEFAULT_ALPHA, SEM_ALPHA,
    DEFAULT_LINE_WIDTH, DEFAULT_MARKER_SIZE,
)


def plot_psychometric(
    result: dict,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    alpha: float = DEFAULT_ALPHA,
    linewidth: float = DEFAULT_LINE_WIDTH,
    linestyle: str = '-',
    show_data: bool = True,
    show_ci: bool = True,
    show_params: bool = False,
    show_lapse: bool = False,
    show_reference: bool = True,
    show_individual: bool = True,
    individual_alpha: float = 0.12,
    session_colours: Optional[list] = None,
    title: str = '',
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot psychometric curve(s) from a pre-computed result dict.

    Args:
        result: Dict from compute_psychometric(). Must contain 'mode' key.
        ax: Matplotlib axes (creates one if None).
        color: Line/point colour.
        label: Legend label.
        show_data: Show binned data points.
        show_ci: Show bootstrap CI band (if available in result).
        show_params: Annotate PSE and slope on curve.
        show_lapse: Show lapse rate reference lines.
        show_reference: Show 0.5 horizontal and 0 vertical lines.
        show_individual: For session_mean, show faint per-session fits.
        session_colours: Per-session colours for overlay mode.
        title: Axes title.

    Returns:
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    else:
        fig = ax.get_figure()

    color = color or COLOURS.get('default', PALETTE[0])
    mode = result.get('mode', 'pooled')

    if mode == 'pooled':
        _draw_pooled(result, ax, color, label, alpha, linewidth, linestyle,
                     show_data, show_ci, show_params, show_lapse)

    elif mode == 'overlay':
        _draw_overlay(result, ax, color, alpha, linewidth, session_colours)

    elif mode == 'session_mean':
        _draw_session_mean(result, ax, color, label, alpha, linewidth,
                           show_data, show_ci, show_individual, individual_alpha)

    if show_reference:
        ax.axhline(0.5, ls='--', color='grey', alpha=0.3, zorder=0)
        ax.axvline(0.0, ls='--', color='grey', alpha=0.3, zorder=0)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('P(choose B)')

    if title:
        ax.set_title(title)

    return fig, ax


# ─── Mode-specific drawing ──────────────────────────────────────────────────

def _draw_pooled(result, ax, color, label, alpha, linewidth, linestyle,
                 show_data, show_ci, show_params, show_lapse):
    """Draw a single pooled psychometric curve from result dict."""
    params = result.get('params', {})
    x_fit = result.get('x_fit')
    y_fit = result.get('y_fit')

    # Fitted curve
    if y_fit is not None and x_fit is not None:
        lbl = label
        if show_params and params.get('mu') is not None:
            lbl = f"{label or ''} (PSE={params['mu']:.2f}, \u03c3={params['sigma']:.2f})".strip()
        ax.plot(x_fit, y_fit, color=color, lw=linewidth, ls=linestyle,
                alpha=alpha, label=lbl, zorder=2)

    # Binned data points
    if show_data:
        centres = result.get('bin_centres')
        means = result.get('bin_means')
        if centres is not None and means is not None:
            v = ~np.isnan(means)
            ax.plot(centres[v], means[v], 'o', color=color,
                    markersize=DEFAULT_MARKER_SIZE, alpha=alpha * 0.7,
                    zorder=3, label=label if y_fit is None else None)

    # Bootstrap CI
    if show_ci:
        ci_lo = result.get('ci_lo')
        ci_hi = result.get('ci_hi')
        centres = result.get('bin_centres')
        if ci_lo is not None and ci_hi is not None and centres is not None:
            ax.fill_between(centres, ci_lo, ci_hi, color=color,
                            alpha=SEM_ALPHA, zorder=1)

    # Lapse lines
    if show_lapse and params:
        ll = params.get('lapse_low')
        lh = params.get('lapse_high')
        if ll is not None:
            ax.axhline(ll, color='grey', ls=':', alpha=0.4)
        if lh is not None:
            ax.axhline(1 - lh, color='grey', ls=':', alpha=0.4)


def _draw_overlay(result, ax, color, alpha, linewidth, session_colours):
    """Draw per-session psychometric curves."""
    per_session = result.get('per_session', [])
    n = len(per_session)
    if session_colours is not None:
        colours = session_colours
    elif color:
        colours = [color] * n
    else:
        import matplotlib
        cmap = matplotlib.cm.get_cmap('viridis')
        colours = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for i, entry in enumerate(per_session):
        y_fit = entry.get('y_fit')
        x_fit = entry.get('x_fit')
        if y_fit is not None and x_fit is not None:
            a = 0.3 + 0.5 * (i / max(n - 1, 1))
            ax.plot(x_fit, y_fit, color=colours[i], lw=linewidth,
                    alpha=a, zorder=2)


def _draw_session_mean(result, ax, color, label, alpha, linewidth,
                       show_data, show_ci, show_individual, individual_alpha):
    """Draw mean P(B) ± SEM across sessions."""
    centres = result.get('bin_centres')
    mean_p = result.get('mean_p')
    sem_p = result.get('sem_p')

    if centres is None or mean_p is None:
        return

    # Faint individual session fits
    if show_individual:
        for fit in result.get('per_session_fits', []):
            y = fit.get('y_fit')
            x = fit.get('x_fit')
            if y is not None and x is not None:
                ax.plot(x, y, color=color, alpha=individual_alpha,
                        lw=0.8, zorder=1)

    # Mean data points
    v = ~np.isnan(mean_p)
    if show_data:
        ax.plot(centres[v], mean_p[v], 'o-', color=color,
                markersize=DEFAULT_MARKER_SIZE, lw=linewidth,
                alpha=alpha, label=label, zorder=3)

    # SEM band
    if show_ci and sem_p is not None:
        ax.fill_between(centres, mean_p - sem_p, mean_p + sem_p,
                        color=color, alpha=SEM_ALPHA, zorder=1)
