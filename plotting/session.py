"""
Session-Level Trial Plots

Trial rasters, within-session choice/stimulus visualisation.
All functions return (fig, ax).

Usage:
    from behav_utils.plotting.session import plot_session_trials

    fig, ax = plot_session_trials(session)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional, Tuple, TYPE_CHECKING

from behav_utils.plotting.styles import COLOURS

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData


def plot_session_trials(
    session: 'SessionData',
    window: int = 0,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_rolling: bool = True,
    exclude_abort: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot trial-by-trial raster for a session.

    Shows stimulus values, choices (correct/error/abort), and
    optionally rolling performance.

    Args:
        session: SessionData object
        window: Rolling window for performance line (0 = no rolling line)
        ax: Existing axes
        title: Plot title (default: session_id)
        show_rolling: Show rolling accuracy
        exclude_abort: Whether to hide abort trials

    Returns:
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.get_figure()

    trials = session.trials
    n = trials.n_trials
    trial_idx = np.arange(n)

    stim = trials.stimulus
    correct = trials.correct
    abort = trials.abort
    no_resp = trials.no_response

    # Mask
    if exclude_abort:
        mask = ~abort
    else:
        mask = np.ones(n, dtype=bool)

    t = trial_idx[mask]
    s = stim[mask]
    c = correct[mask]
    a = abort[mask]
    nr = no_resp[mask]

    # Colour by outcome
    colours = np.full(len(t), COLOURS['correct'], dtype=object)
    colours[~c & ~a & ~nr] = COLOURS['error']
    colours[a] = COLOURS['no_response']
    colours[nr] = COLOURS['no_response']

    # Plot stimulus values coloured by outcome
    for outcome_label, colour in [
        ('Correct', COLOURS['correct']),
        ('Error', COLOURS['error']),
        ('No response', COLOURS['no_response']),
    ]:
        if outcome_label == 'Correct':
            m = c & ~a & ~nr
        elif outcome_label == 'Error':
            m = ~c & ~a & ~nr
        else:
            m = a | nr

        if m.sum() > 0:
            ax.scatter(t[m], s[m], c=colour, s=8, alpha=0.6,
                       label=outcome_label, zorder=3)

    # Boundary line
    ax.axhline(0, color='grey', ls='--', alpha=0.4, linewidth=0.8)

    # Rolling accuracy
    if show_rolling and window > 0:
        correct_full = trials.correct.astype(float).copy()
        correct_full[trials.abort | trials.no_response] = np.nan

        if np.any(~np.isnan(correct_full)):
            rolling = np.full(n, np.nan)
            for i in range(window, n):
                w = correct_full[i - window:i]
                valid_w = w[~np.isnan(w)]
                if len(valid_w) > 0:
                    rolling[i] = np.mean(valid_w)

            # Plot on secondary axis
            ax2 = ax.twinx()
            ax2.plot(trial_idx, rolling, '-', color='black', alpha=0.4,
                     linewidth=1, label=f'Rolling acc (w={window})')
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel('Rolling accuracy', fontsize=8)
            ax2.tick_params(labelsize=7)

    ax.set_xlabel('Trial')
    ax.set_ylabel('Stimulus')
    ax.set_title(title or session.session_id)
    ax.legend(fontsize=7, loc='upper right', markerscale=2)

    return fig, ax


def plot_session_comparison(
    sessions: list,
    stat: str = 'accuracy',
    labels: Optional[list] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare a stat across multiple sessions as bar chart.

    Args:
        sessions: List of SessionData
        stat: Stat name to compare
        labels: Bar labels (default: session IDs)

    Returns:
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, len(sessions) * 0.8), 4))
    else:
        fig = ax.get_figure()

    if labels is None:
        labels = [f'S{s.session_idx}' for s in sessions]

    values = []
    for sess in sessions:
        s = sess.stats([stat])
        val = s.get(stat, np.nan)
        if isinstance(val, dict):
            val = list(val.values())[0]  # Take first sub-value
        values.append(float(val) if not isinstance(val, (str, type(None))) else np.nan)

    x = np.arange(len(values))
    ax.bar(x, values, color=COLOURS['default'], edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(stat)
    ax.set_title(f'{stat} across sessions')

    return fig, ax
