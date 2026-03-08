"""
Psychometric Curve Plotting

Single sessions, overlays, grids, and pooled curves.
All functions return (fig, ax) or (fig, axes, info) for further customisation.

Standalone functions work with raw arrays.
SessionData methods delegate here.

Usage:
    from behav_utils.plotting.psychometric import plot_psychometric

    # Raw arrays
    fig, ax, info = plot_psychometric(stimuli, choices)

    # Via data class
    fig, ax, info = session.plot_psychometric()
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Union, TYPE_CHECKING

from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.analysis.utils import cumulative_gaussian
from behav_utils.plotting.styles import (
    COLOURS, get_session_colours, DEFAULT_ALPHA,
)

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData


# =============================================================================
# SINGLE PSYCHOMETRIC
# =============================================================================

def plot_psychometric(
    stimuli: np.ndarray,
    choices: np.ndarray,
    ax: Optional[plt.Axes] = None,
    n_bins: int = 8,
    color: Optional[str] = None,
    title: str = '',
    show_params: bool = True,
    show_gof: bool = False,
    show_lapse: bool = False,
    show_ci: bool = False,
    n_bootstrap: int = 0,
    label: Optional[str] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """
    Plot psychometric curve from raw stimulus and choice arrays.

    Args:
        stimuli: Stimulus values
        choices: Binary choices (0=A, 1=B), NaN=no response
        ax: Existing axes (creates new if None)
        n_bins: Number of bins for data points
        color: Line/point colour
        title: Plot title
        show_params: Annotate PSE and slope
        show_gof: Annotate R²
        show_lapse: Show lapse rate lines
        show_ci: Show bootstrap confidence interval (requires n_bootstrap > 0)
        n_bootstrap: Number of bootstrap samples for CI
        label: Legend label for the fitted curve

    Returns:
        (fig, ax, info) where info contains fit parameters
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    if color is None:
        color = COLOURS['default']

    # Filter NaN
    valid = ~np.isnan(stimuli) & ~np.isnan(choices)
    stim = stimuli[valid]
    ch = choices[valid]

    # Binned data points
    bin_edges = np.linspace(-1, 1, n_bins + 1)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(stim, bin_edges) - 1, 0, n_bins - 1)

    p_b = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            p_b[b] = np.mean(ch[mask])

    # Scatter with size proportional to count
    sizes = np.clip(counts / max(counts.max(), 1) * 80, 10, 80)
    ax.scatter(midpoints, p_b, s=sizes, c=color, edgecolor='black',
               linewidth=0.5, zorder=5, alpha=0.8)

    # Fit
    psych = fit_psychometric(stim, ch, n_bootstrap=n_bootstrap)
    info = psych

    if psych.get('success', False):
        x_fine = np.linspace(-1.1, 1.1, 200)
        y_fit = cumulative_gaussian(
            x_fine, psych['mu'], psych['sigma'],
            psych['lapse_low'], psych['lapse_high'],
        )
        ax.plot(x_fine, y_fit, '-', color=color, linewidth=2,
                label=label, zorder=4)

        # CI band
        if show_ci and 'y_fit_ci' in psych:
            ci_lo, ci_hi = psych['y_fit_ci']
            x_ci = psych.get('x_fit', x_fine)
            ax.fill_between(x_ci, ci_lo, ci_hi, color=color, alpha=0.15)

        # Annotations
        text_parts = []
        if show_params:
            text_parts.append(f"PSE = {psych['mu']:.3f}")
            text_parts.append(f"\u03c3 = {psych['sigma']:.3f}")
        if show_lapse:
            text_parts.append(f"\u03b3 = {psych['lapse_low']:.3f}")
            text_parts.append(f"\u03bb = {psych['lapse_high']:.3f}")
        if show_gof:
            from behav_utils.analysis.psychometry import compute_psychometric_gof
            gof = compute_psychometric_gof(stim, ch, psych)
            r2 = gof.get('r_squared', np.nan)
            text_parts.append(f"R\u00b2 = {r2:.3f}")
            info['r_squared'] = r2

        if text_parts:
            text = '\n'.join(text_parts)
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                    fontsize=8, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.8, edgecolor='grey'))

        # Lapse lines
        if show_lapse:
            ax.axhline(psych['lapse_low'], color='grey', ls=':', alpha=0.4)
            ax.axhline(1 - psych['lapse_high'], color='grey', ls=':', alpha=0.4)

    # Reference lines
    ax.axhline(0.5, color='grey', ls='--', alpha=0.3)
    ax.axvline(0, color='grey', ls='--', alpha=0.3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('P(choose B)')
    if title:
        ax.set_title(title)

    return fig, ax, info


# =============================================================================
# SESSION PSYCHOMETRICS (multi-session)
# =============================================================================

def plot_session_psychometrics(
    sessions: List['SessionData'],
    mode: str = 'overlay',
    n_max: int = 20,
    ax: Optional[plt.Axes] = None,
    suptitle: Optional[str] = None,
    exclude_abort: bool = True,
    exclude_opto: bool = True,
    n_bootstrap: int = 0,
    show_ci: bool = True,
    **kwargs,
) -> Union[
    Tuple[plt.Figure, plt.Axes, List[Dict]],
    Tuple[plt.Figure, np.ndarray, List[Dict]],
]:
    """
    Plot psychometric curves for multiple sessions.

    Args:
        sessions: List of SessionData objects
        mode: 'overlay' — all on one axes, colour gradient early→late
              'grid' — one subplot per session (evenly sampled)
              'pooled' — pool all trials into one curve
        n_max: Max sessions to show in grid mode (evenly sampled)
        ax: Existing axes (overlay/pooled only)
        suptitle: Figure title

    Returns:
        (fig, ax, infos) for overlay/pooled
        (fig, axes, infos) for grid
    """
    if mode == 'overlay':
        return _plot_overlay(sessions, ax=ax, suptitle=suptitle,
                             exclude_abort=exclude_abort,
                             exclude_opto=exclude_opto, **kwargs)
    elif mode == 'grid':
        return _plot_grid(sessions, n_max=n_max, suptitle=suptitle,
                          exclude_abort=exclude_abort,
                          exclude_opto=exclude_opto, **kwargs)
    elif mode == 'pooled':
        return _plot_pooled(sessions, ax=ax, suptitle=suptitle,
                            exclude_abort=exclude_abort,
                            exclude_opto=exclude_opto,
                            n_bootstrap=n_bootstrap,
                            show_ci=show_ci, **kwargs)
    else:
        raise ValueError(f"mode must be 'overlay', 'grid', or 'pooled', got '{mode}'")


def _extract_valid_arrays(session, exclude_abort, exclude_opto):
    """Helper: get valid stimuli and choices from a session."""
    arrays = session.trials.get_arrays(
        exclude_abort=exclude_abort,
        exclude_opto=exclude_opto,
    )
    valid = ~arrays['no_response']
    return arrays['stimuli'][valid], arrays['choices'][valid]


def _plot_overlay(sessions, ax=None, suptitle=None,
                  exclude_abort=True, exclude_opto=True, **kwargs):
    """All sessions on one axes with colour gradient."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    colours = get_session_colours(len(sessions))
    infos = []

    for i, sess in enumerate(sessions):
        stim, ch = _extract_valid_arrays(sess, exclude_abort, exclude_opto)
        if len(stim) < 10:
            infos.append({'success': False})
            continue

        psych = fit_psychometric(stim, ch)
        infos.append(psych)

        if psych.get('success', False):
            x_fine = np.linspace(-1.1, 1.1, 200)
            y_fit = cumulative_gaussian(
                x_fine, psych['mu'], psych['sigma'],
                psych['lapse_low'], psych['lapse_high'],
            )
            ax.plot(x_fine, y_fit, '-', color=colours[i], linewidth=1.2,
                    alpha=0.7, label=f'S{sess.session_idx}')

    ax.axhline(0.5, color='grey', ls='--', alpha=0.3)
    ax.axvline(0, color='grey', ls='--', alpha=0.3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('P(choose B)')
    if suptitle:
        ax.set_title(suptitle)
    ax.legend(fontsize=7, ncol=2, loc='lower right')

    return fig, ax, infos


def _plot_grid(sessions, n_max=20, suptitle=None,
               exclude_abort=True, exclude_opto=True, **kwargs):
    """One subplot per session."""
    # Evenly sample if too many
    if len(sessions) > n_max:
        indices = np.linspace(0, len(sessions) - 1, n_max, dtype=int)
        sessions = [sessions[i] for i in indices]

    n = len(sessions)
    n_cols = min(5, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    axes_flat = axes.flatten()

    infos = []
    for i, sess in enumerate(sessions):
        stim, ch = _extract_valid_arrays(sess, exclude_abort, exclude_opto)
        _, _, info = plot_psychometric(
            stim, ch, ax=axes_flat[i],
            title=f'S{sess.session_idx}',
            show_params=True, show_gof=True,
            **kwargs,
        )
        infos.append(info)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()

    return fig, axes, infos


def _plot_pooled(sessions, ax=None, suptitle=None,
                 exclude_abort=True, exclude_opto=True,
                 n_bootstrap=0, **kwargs):
    """Pool all trials across sessions into one curve."""
    all_stim = []
    all_ch = []
    for sess in sessions:
        stim, ch = _extract_valid_arrays(sess, exclude_abort, exclude_opto)
        all_stim.append(stim)
        all_ch.append(ch)

    stim_pooled = np.concatenate(all_stim)
    ch_pooled = np.concatenate(all_ch)

    fig, ax, info = plot_psychometric(
        stim_pooled, ch_pooled, ax=ax,
        title=suptitle or f'Pooled ({len(sessions)} sessions)',
        n_bootstrap=n_bootstrap,
        show_params=True, show_gof=True,
        **kwargs,
    )

    return fig, ax, info
