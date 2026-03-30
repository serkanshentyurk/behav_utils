"""
Psychometric Curve Plotting

Single sessions, overlays, grids, pooled, and session-mean curves.
All functions return (fig, ax, info) or (fig, axes, infos) for further
customisation.

Standalone functions work with raw arrays.
SessionData methods delegate here.

Modes for multi-session plotting (plot_session_psychometrics):
    'overlay'       Each session as a separate curve, colour gradient.
    'grid'          One subplot per session.
    'pooled'        Pool all trials, single fit. Bootstrap CI resamples
                    trials (ignores session structure — use session_mean
                    if between-session variability matters).
    'session_mean'  Fit each session independently, plot mean P(B) per
                    bin ± SEM across sessions. Error reflects day-to-day
                    variability, not trial noise.
    'per_animal'    One subplot per animal, each showing session_mean
                    or pooled within that animal. Only meaningful when
                    sessions span multiple animals (ExperimentData use).

Usage:
    from behav_utils.plotting.psychometric import plot_psychometric

    fig, ax, info = plot_psychometric(stimuli, choices)
    fig, ax, info = plot_session_psychometrics(sessions, mode='session_mean')
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
        stimuli: Stimulus values (float array)
        choices: Binary choices (0=A, 1=B, NaN=no response)
        ax: Existing axes (creates new figure if None)
        n_bins: Number of bins for data points
        color: Line/point colour (default: COLOURS['default'])
        title: Plot title
        show_params: Annotate PSE and slope on plot
        show_gof: Annotate R²
        show_lapse: Show lapse rate lines and values
        show_ci: Show bootstrap confidence band (requires n_bootstrap > 0)
        n_bootstrap: Number of bootstrap resamples for CI (0 = no CI)
        label: Legend label for the fitted curve

    Returns:
        (fig, ax, info) where info is the dict from fit_psychometric
        containing 'mu', 'sigma', 'lapse_low', 'lapse_high', 'success',
        and optionally bootstrap CI fields.
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
            if ci_lo is not None:
                x_ci = psych.get('x_fit', x_fine)
                ax.fill_between(x_ci, ci_lo, ci_hi, color=color, alpha=0.15)

        # Annotations
        text_parts = []
        if show_params:
            mu_str = f"PSE = {psych['mu']:.3f}"
            if 'mu_se' in psych and not np.isnan(psych['mu_se']):
                mu_str += f" \u00b1 {psych['mu_se']:.3f}"
            text_parts.append(mu_str)

            sig_str = f"\u03c3 = {psych['sigma']:.3f}"
            if 'sigma_se' in psych and not np.isnan(psych['sigma_se']):
                sig_str += f" \u00b1 {psych['sigma_se']:.3f}"
            text_parts.append(sig_str)
        if show_lapse:
            ll_str = f"\u03b3 = {psych['lapse_low']:.3f}"
            if 'lapse_low_se' in psych and not np.isnan(psych['lapse_low_se']):
                ll_str += f" \u00b1 {psych['lapse_low_se']:.3f}"
            text_parts.append(ll_str)

            lh_str = f"\u03bb = {psych['lapse_high']:.3f}"
            if 'lapse_high_se' in psych and not np.isnan(psych['lapse_high_se']):
                lh_str += f" \u00b1 {psych['lapse_high_se']:.3f}"
            text_parts.append(lh_str)
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
    Tuple[plt.Figure, plt.Axes, Union[Dict, List[Dict]]],
    Tuple[plt.Figure, np.ndarray, List[Dict]],
]:
    """
    Plot psychometric curves for multiple sessions.

    Args:
        sessions: List of SessionData objects
        mode: How to combine sessions:
            'overlay'       All on one axes, colour gradient early→late.
            'grid'          One subplot per session (evenly sampled if >n_max).
            'pooled'        Pool all trials into one curve. Bootstrap CI
                            resamples trials; does NOT account for session
                            clustering. Use 'session_mean' if between-session
                            variability is the quantity of interest.
            'session_mean'  Fit each session independently, plot mean P(B)
                            per bin ± SEM across sessions. CI reflects
                            day-to-day variability, not trial noise.
            'per_animal'    One subplot per animal, each showing the
                            sub_mode within that animal. Useful when
                            sessions span multiple animals.
        n_max: Max sessions to show in grid mode (evenly sampled)
        ax: Existing axes (overlay/pooled/session_mean only; ignored
            for grid/per_animal)
        suptitle: Figure-level title
        exclude_abort: Exclude abort trials
        exclude_opto: Exclude opto trials
        n_bootstrap: Bootstrap samples for CI (pooled mode only)
        show_ci: Show confidence/SEM band (pooled and session_mean)

        Mode-specific kwargs (passed via **kwargs):
            show_individual: bool — show faint per-session curves
                (default False for pooled, True for session_mean)
            individual_alpha: float — alpha for individual curves (0.15)
            n_bins: int — number of stimulus bins (8)
            color: str — colour for mean/pooled curve
            show_params: bool — annotate PSE/slope (True)
            subplot_titles: list[str] — custom titles for grid/per_animal
                subplots (length must match number of subplots)
            sub_mode: str — mode within each per_animal subplot
                ('session_mean' or 'pooled', default 'session_mean')

    Returns:
        (fig, ax, info) for overlay/pooled/session_mean
        (fig, axes, infos) for grid/per_animal
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
    elif mode == 'session_mean':
        return _plot_session_mean(sessions, ax=ax, suptitle=suptitle,
                                  exclude_abort=exclude_abort,
                                  exclude_opto=exclude_opto,
                                  show_ci=show_ci, **kwargs)
    elif mode == 'per_animal':
        return _plot_per_animal(sessions, suptitle=suptitle,
                                exclude_abort=exclude_abort,
                                exclude_opto=exclude_opto,
                                show_ci=show_ci,
                                n_bootstrap=n_bootstrap, **kwargs)
    else:
        raise ValueError(
            f"mode must be 'overlay', 'grid', 'pooled', 'session_mean', "
            f"or 'per_animal', got '{mode}'"
        )


# =============================================================================
# HELPERS
# =============================================================================

def _extract_valid_arrays(session, exclude_abort, exclude_opto):
    """Helper: get valid stimuli and choices from a session."""
    arrays = session.trials.get_arrays(
        exclude_abort=exclude_abort,
        exclude_opto=exclude_opto,
    )
    valid = ~arrays['no_response']
    return arrays['stimuli'][valid], arrays['choices'][valid]


def _infer_animal_id(sessions):
    """Infer animal_id if all sessions belong to the same animal."""
    aids = set()
    for s in sessions:
        aid = s.metadata.get('animal_id', None)
        if aid:
            aids.add(aid)
    if len(aids) == 1:
        return aids.pop()
    return None


def _auto_title(suptitle, sessions, mode_label):
    """Generate a title if none provided, including animal_id when unique."""
    if suptitle is not None:
        return suptitle
    aid = _infer_animal_id(sessions)
    prefix = f'{aid} — ' if aid else ''
    return f'{prefix}{mode_label} ({len(sessions)} sessions)'


# =============================================================================
# OVERLAY
# =============================================================================

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
    ax.set_title(_auto_title(suptitle, sessions, 'Overlay'))
    ax.legend(fontsize=7, ncol=2, loc='lower right')

    return fig, ax, infos


# =============================================================================
# GRID
# =============================================================================

def _plot_grid(sessions, n_max=20, suptitle=None,
               exclude_abort=True, exclude_opto=True,
               subplot_titles=None, **kwargs):
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

        # Subplot title
        if subplot_titles is not None and i < len(subplot_titles):
            sub_title = subplot_titles[i]
        else:
            sub_title = f'S{sess.session_idx}'

        _, _, info = plot_psychometric(
            stim, ch, ax=axes_flat[i],
            title=sub_title,
            show_params=True, show_gof=True,
            **kwargs,
        )
        infos.append(info)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    title = _auto_title(suptitle, sessions, 'Grid')
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()

    return fig, axes, infos


# =============================================================================
# POOLED
# =============================================================================

def _plot_pooled(sessions, ax=None, suptitle=None,
                 exclude_abort=True, exclude_opto=True,
                 n_bootstrap=0, show_ci=False,
                 show_individual=False, individual_alpha=0.15,
                 color=None, **kwargs):
    """
    Pool all trials across sessions into one curve.

    Note on CI: the bootstrap resamples individual trials, ignoring
    session structure. This can underestimate uncertainty when sessions
    have different psychometric curves (e.g., during learning). For
    error bars reflecting between-session variability, use
    mode='session_mean' instead.

    Args:
        show_individual: If True, draw faint per-session fitted curves
            behind the pooled curve.
        individual_alpha: Alpha for individual curves.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    if color is None:
        color = COLOURS['default']

    x_fine = np.linspace(-1.1, 1.1, 200)

    # Draw individual session curves if requested
    per_session_infos = []
    all_stim = []
    all_ch = []
    for sess in sessions:
        stim, ch = _extract_valid_arrays(sess, exclude_abort, exclude_opto)
        all_stim.append(stim)
        all_ch.append(ch)

        if show_individual and len(stim) >= 10:
            psych = fit_psychometric(stim, ch)
            per_session_infos.append(psych)
            if psych.get('success', False):
                y = cumulative_gaussian(
                    x_fine, psych['mu'], psych['sigma'],
                    psych['lapse_low'], psych['lapse_high'],
                )
                ax.plot(x_fine, y, '-', color=color,
                        alpha=individual_alpha, linewidth=0.8)

    # Pool and fit
    stim_pooled = np.concatenate(all_stim)
    ch_pooled = np.concatenate(all_ch)

    _, _, info = plot_psychometric(
        stim_pooled, ch_pooled, ax=ax,
        title='',  # set below
        n_bootstrap=n_bootstrap,
        show_ci=show_ci,
        color=color,
        **kwargs,
    )

    ax.set_title(_auto_title(suptitle, sessions, 'Pooled'))

    if show_individual:
        info['per_session_fits'] = per_session_infos

    return fig, ax, info


# =============================================================================
# SESSION MEAN
# =============================================================================

def _plot_session_mean(
    sessions,
    ax=None,
    suptitle=None,
    exclude_abort=True,
    exclude_opto=True,
    show_ci=True,
    show_individual=True,
    individual_alpha=0.15,
    n_bins=8,
    color=None,
    show_params=True,
    show_lapse=False,
    min_sessions_per_bin=3,
    **kwargs,
):
    """
    Mean psychometric across sessions with between-session SEM.

    For each session independently: bin stimuli, compute P(B) per bin,
    fit psychometric curve. Across sessions: mean ± SEM of binned P(B)
    as error bars; mean ± SEM of fitted curves as shaded band.

    This captures between-session variability — the error reflects how
    consistent the psychometric function is day to day, not trial-level
    noise within pooled data.

    Args:
        sessions: List of SessionData
        ax: Existing axes (creates new if None)
        suptitle: Title (auto-generated with animal_id if None)
        show_ci: Show SEM band on fitted curve
        show_individual: Show faint per-session curves underneath
        individual_alpha: Alpha for individual curves
        n_bins: Number of stimulus bins
        color: Colour for mean curve/points
        show_params: Annotate mean ± SEM of PSE and slope
        show_lapse: Include lapse rates in annotation
        min_sessions_per_bin: Minimum sessions contributing to a bin
            for it to be plotted (default 3)

    Returns:
        (fig, ax, info) where info contains:
            'success': bool
            'mode': 'session_mean'
            'n_sessions': int
            'n_fits_successful': int
            'param_summary': dict of {param: {mean, sem, std, n, values}}
            'bin_midpoints', 'bin_mean', 'bin_sem', 'bin_n': arrays
            'per_session_fits': list of fit dicts
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    if color is None:
        color = COLOURS['default']

    bin_edges = np.linspace(-1, 1, n_bins + 1)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    x_fine = np.linspace(-1.1, 1.1, 200)

    # ── Per-session: bin P(B) and fit curve ─────────────────────────────────
    all_binned = []
    all_fits = []
    all_curves = []

    for sess in sessions:
        stim, ch = _extract_valid_arrays(sess, exclude_abort, exclude_opto)
        if len(stim) < 10:
            continue

        # Bin this session
        bin_idx = np.clip(np.digitize(stim, bin_edges) - 1, 0, n_bins - 1)
        pb = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() >= 3:
                pb[b] = np.mean(ch[mask])
        all_binned.append(pb)

        # Fit this session
        psych = fit_psychometric(stim, ch)
        all_fits.append(psych)

        if psych.get('success', False):
            y = cumulative_gaussian(
                x_fine, psych['mu'], psych['sigma'],
                psych['lapse_low'], psych['lapse_high'],
            )
            all_curves.append(y)

            if show_individual:
                ax.plot(x_fine, y, '-', color=color,
                        alpha=individual_alpha, linewidth=0.8)

    n_sessions_used = len(all_binned)
    if n_sessions_used == 0:
        ax.text(0.5, 0.5, 'No valid sessions',
                transform=ax.transAxes, ha='center', va='center')
        return fig, ax, {'success': False, 'n_sessions': 0}

    # ── Across sessions: mean ± SEM of binned P(B) ─────────────────────────
    binned_matrix = np.array(all_binned)  # (n_sessions, n_bins)
    bin_mean = np.nanmean(binned_matrix, axis=0)
    bin_n = np.sum(~np.isnan(binned_matrix), axis=0)
    bin_sem = np.where(
        bin_n > 1,
        np.nanstd(binned_matrix, axis=0, ddof=1) / np.sqrt(bin_n),
        0.0,
    )

    valid_bins = bin_n >= min_sessions_per_bin

    ax.errorbar(
        midpoints[valid_bins], bin_mean[valid_bins],
        yerr=bin_sem[valid_bins],
        fmt='o', color=color, markersize=6, capsize=3,
        elinewidth=1.5, markeredgecolor='black', markeredgewidth=0.5,
        zorder=10, label=f'Mean \u00b1 SEM (n={n_sessions_used})',
    )

    # ── Mean curve ± SEM band ──────────────────────────────────────────────
    if len(all_curves) >= 2:
        curve_matrix = np.array(all_curves)
        mean_curve = np.mean(curve_matrix, axis=0)
        sem_curve = np.std(curve_matrix, axis=0, ddof=1) / np.sqrt(len(all_curves))

        ax.plot(x_fine, mean_curve, '-', color=color, linewidth=2.5, zorder=8)

        if show_ci:
            ax.fill_between(
                x_fine,
                mean_curve - sem_curve,
                mean_curve + sem_curve,
                color=color, alpha=0.2, zorder=3,
            )
    elif len(all_curves) == 1:
        ax.plot(x_fine, all_curves[0], '-', color=color, linewidth=2.5, zorder=8)

    # ── Parameter summary ──────────────────────────────────────────────────
    good_fits = [f for f in all_fits if f.get('success', False)]
    param_summary = {}
    for key in ['mu', 'sigma', 'lapse_low', 'lapse_high']:
        vals = np.array([f[key] for f in good_fits])
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            param_summary[key] = {
                'mean': float(np.mean(vals)),
                'sem': float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                       if len(vals) > 1 else 0.0,
                'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'n': len(vals),
                'values': vals,
            }
        else:
            param_summary[key] = {
                'mean': np.nan, 'sem': np.nan, 'std': np.nan, 'n': 0,
                'values': np.array([]),
            }

    if show_params and param_summary['mu']['n'] > 0:
        text_parts = []
        mu_s = param_summary['mu']
        sigma_s = param_summary['sigma']
        text_parts.append(f"PSE = {mu_s['mean']:.3f} \u00b1 {mu_s['sem']:.3f}")
        text_parts.append(f"\u03c3 = {sigma_s['mean']:.3f} \u00b1 {sigma_s['sem']:.3f}")
        if show_lapse:
            ll_s = param_summary['lapse_low']
            lh_s = param_summary['lapse_high']
            text_parts.append(f"\u03b3 = {ll_s['mean']:.3f} \u00b1 {ll_s['sem']:.3f}")
            text_parts.append(f"\u03bb = {lh_s['mean']:.3f} \u00b1 {lh_s['sem']:.3f}")
        text_parts.append(f"n = {n_sessions_used} sessions")

        text = '\n'.join(text_parts)
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='grey'))

    # ── Formatting ─────────────────────────────────────────────────────────
    ax.axhline(0.5, color='grey', ls='--', alpha=0.3)
    ax.axvline(0, color='grey', ls='--', alpha=0.3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('P(choose B)')
    ax.set_title(_auto_title(suptitle, sessions, 'Session mean'))
    ax.legend(fontsize=8, loc='lower right')

    info = {
        'success': True,
        'mode': 'session_mean',
        'n_sessions': n_sessions_used,
        'n_fits_successful': len(good_fits),
        'param_summary': param_summary,
        'bin_midpoints': midpoints,
        'bin_mean': bin_mean,
        'bin_sem': bin_sem,
        'bin_n': bin_n,
        'per_session_fits': all_fits,
    }

    return fig, ax, info


# =============================================================================
# PER-ANIMAL SUBPLOTS
# =============================================================================

def _plot_per_animal(
    sessions,
    suptitle=None,
    exclude_abort=True,
    exclude_opto=True,
    show_ci=True,
    n_bootstrap=0,
    sub_mode='session_mean',
    n_cols=4,
    figsize_per_panel=(4.0, 3.5),
    subplot_titles=None,
    **kwargs,
):
    """
    One subplot per animal, each showing sub_mode within that animal.

    Args:
        sessions: List of SessionData (may span multiple animals)
        sub_mode: Mode for each subplot ('session_mean' or 'pooled')
        n_cols: Columns in grid
        figsize_per_panel: (width, height) per subplot
        subplot_titles: Custom titles per animal (list, same order as
            sorted animal IDs)

    Returns:
        (fig, axes, infos) — infos is a list (one per animal)
    """
    # Group sessions by animal
    by_animal = {}
    for sess in sessions:
        aid = sess.metadata.get('animal_id', 'unknown')
        by_animal.setdefault(aid, []).append(sess)

    animal_ids = sorted(by_animal.keys())
    n_animals = len(animal_ids)

    n_rows = int(np.ceil(n_animals / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols,
                 figsize_per_panel[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    infos = []
    for i, aid in enumerate(animal_ids):
        ax = axes_flat[i]
        animal_sessions = by_animal[aid]

        # Subplot title
        if subplot_titles is not None and i < len(subplot_titles):
            sub_title = subplot_titles[i]
        else:
            sub_title = f'{aid} ({len(animal_sessions)} sessions)'

        if sub_mode == 'session_mean':
            _, _, info = _plot_session_mean(
                animal_sessions, ax=ax, suptitle=sub_title,
                exclude_abort=exclude_abort, exclude_opto=exclude_opto,
                show_ci=show_ci, **kwargs,
            )
        elif sub_mode == 'pooled':
            _, _, info = _plot_pooled(
                animal_sessions, ax=ax, suptitle=sub_title,
                exclude_abort=exclude_abort, exclude_opto=exclude_opto,
                n_bootstrap=n_bootstrap, show_ci=show_ci, **kwargs,
            )
        else:
            _, _, info = _plot_session_mean(
                animal_sessions, ax=ax, suptitle=sub_title,
                exclude_abort=exclude_abort, exclude_opto=exclude_opto,
                show_ci=show_ci, **kwargs,
            )
        infos.append(info)

    # Hide unused panels
    for j in range(n_animals, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig, axes, infos


# =============================================================================
# COMPARE GROUPS OF SESSIONS
# =============================================================================

def plot_psychometric_compare(
    session_groups: Dict[str, List['SessionData']],
    mode: str = 'session_mean',
    suptitle: Optional[str] = None,
    exclude_abort: bool = True,
    exclude_opto: bool = True,
    show_ci: bool = True,
    n_bootstrap: int = 0,
    colours: Optional[Dict[str, str]] = None,
    figsize_per_panel: Tuple[float, float] = (5.0, 4.5),
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Dict]]:
    """
    Side-by-side psychometric comparison across groups of sessions.

    Each group gets its own subplot. Useful for comparing pre vs post,
    early vs late, opto vs control, different distributions, etc.

    Args:
        session_groups: Dict mapping group labels to lists of SessionData.
            Order of keys determines subplot order (use OrderedDict or
            Python 3.7+ dict insertion order).
        mode: Mode for each subplot:
            'session_mean' — mean P(B) ± between-session SEM (default)
            'pooled' — pool trials, single fit
            'overlay' — per-session curves
        suptitle: Figure-level title
        exclude_abort: Exclude abort trials
        exclude_opto: Exclude opto trials
        show_ci: Show SEM/CI band
        n_bootstrap: Bootstrap samples (pooled mode only)
        colours: Dict mapping group labels to colours. If None, uses
            a default palette.
        figsize_per_panel: (width, height) per subplot

    Returns:
        (fig, axes, infos) where:
            axes: 1D array of length n_groups
            infos: dict mapping group labels to their info dicts
    """
    labels = list(session_groups.keys())
    n_groups = len(labels)

    if colours is None:
        default_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        colours = {lab: default_palette[i % len(default_palette)]
                   for i, lab in enumerate(labels)}

    fig, axes = plt.subplots(
        1, n_groups,
        figsize=(figsize_per_panel[0] * n_groups, figsize_per_panel[1]),
        squeeze=False,
    )
    axes = axes.flatten()

    infos = {}
    for i, label in enumerate(labels):
        ax = axes[i]
        sessions = session_groups[label]
        color = colours.get(label, COLOURS['default'])

        if mode == 'session_mean':
            _, _, info = _plot_session_mean(
                sessions, ax=ax, suptitle=label,
                exclude_abort=exclude_abort, exclude_opto=exclude_opto,
                show_ci=show_ci, color=color, **kwargs,
            )
        elif mode == 'pooled':
            _, _, info = _plot_pooled(
                sessions, ax=ax, suptitle=label,
                exclude_abort=exclude_abort, exclude_opto=exclude_opto,
                n_bootstrap=n_bootstrap, show_ci=show_ci,
                color=color, **kwargs,
            )
        elif mode == 'overlay':
            _, _, info = _plot_overlay(
                sessions, ax=ax, suptitle=label,
                exclude_abort=exclude_abort, exclude_opto=exclude_opto,
                **kwargs,
            )
        else:
            raise ValueError(f"mode must be 'session_mean', 'pooled', "
                             f"or 'overlay', got '{mode}'")

        infos[label] = info

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.03)

    plt.tight_layout()
    return fig, axes, infos
