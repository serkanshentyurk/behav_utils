"""
behav_utils/plotting/comparison.py — Visualise compare_conditions output.

Pairs with behav_utils.analysis.comparison.compare_conditions.

Dict-key convention: params_a/params_b/diffs use math names
(mu, sigma, lapse_low, lapse_high, accuracy) matching fit_psychometric
and compute_psychometric. Plot labels display the literature names
("PSE", "slope") for readability.

If compare_conditions was called with n_bootstrap > 0, the result
contains per-condition fit bands (boot_band_a, boot_band_b) which
are shaded around each fit line.

If you want binned data points overlaid, pass data_a and data_b
as (stimuli, choices) tuples.

Usage:
    from behav_utils.analysis.comparison import compare_conditions
    from behav_utils.plotting.comparison import plot_comparison

    result = compare_conditions(
        stim_a, ch_a, cat_a, stim_b, ch_b, cat_b,
        label_a='opto_on', label_b='opto_off',
        n_bootstrap=1000,
    )
    plot_comparison(result, data_a=(stim_a, ch_a), data_b=(stim_b, ch_b))
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from behav_utils.analysis.utils import cumulative_gaussian


# Map dict-key (math name) → display label (literature name)
_LABEL_FOR_KEY = {
    'mu':         'PSE',
    'sigma':      'slope',
    'lapse_low':  'λ_low',
    'lapse_high': 'λ_high',
    'accuracy':   'Acc',
}


def plot_comparison(
    result: Dict,
    ax: Optional[Axes] = None,
    color_a: str = '#d62728',
    color_b: str = '#444444',
    show_stats: bool = True,
    show_band: bool = True,
    data_a: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    data_b: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    n_bins: int = 8,
    stats_keys: Tuple[str, ...] = ('mu', 'sigma', 'accuracy'),
) -> Axes:
    """
    Overlay psychometric curves for conditions A and B with key statistics.

    Args:
        result: Dict from compare_conditions.
        ax: Axis to draw on (creates 4.5x4 figure if None).
        color_a, color_b: Line colours.
        show_stats: Annotate Δ values + p-values + n in top-left.
        show_band: Shade bootstrap fit bands if present in result.
        data_a, data_b: Optional (stimuli, choices) for binned data overlay.
        n_bins: Stimulus bins for data overlay.
        stats_keys: Which diff stats to show. Default mu/sigma/accuracy;
            pass ('mu', 'sigma', 'lapse_low', 'lapse_high', 'accuracy')
            for everything.

    Returns:
        The axis used.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    label_a = result.get('label_a', 'A')
    label_b = result.get('label_b', 'B')
    params_a = result.get('params_a', {})
    params_b = result.get('params_b', {})

    # ── Bands (under the lines) ────────────────────────────────────
    if show_band:
        _plot_band(ax, result.get('boot_band_b'), color_b)
        _plot_band(ax, result.get('boot_band_a'), color_a)

    # ── Fit curves (B first so A is on top) ────────────────────────
    x = np.linspace(-1, 1, 200)
    for params, color, label in [
        (params_b, color_b, label_b),
        (params_a, color_a, label_a),
    ]:
        if 'mu' in params and not np.isnan(params['mu']):
            y = cumulative_gaussian(
                x, params['mu'], params['sigma'],
                params.get('lapse_low', 0.0), params.get('lapse_high', 0.0),
            )
            ax.plot(x, y, color=color, label=label, linewidth=2)

    # ── Optional data overlay (above curves) ───────────────────────
    _plot_data_points(ax, data_b, color_b, n_bins)
    _plot_data_points(ax, data_a, color_a, n_bins)

    # ── References + labels ───────────────────────────────────────
    ax.axhline(0.5, color='k', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.axvline(0.0, color='k', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Stimulus (distance from boundary)')
    ax.set_ylabel('P(choose B)')
    ax.legend(loc='lower right', fontsize=8)

    if show_stats:
        ax.text(
            0.03, 0.97, _stats_text(result, stats_keys),
            transform=ax.transAxes, va='top', ha='left',
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       alpha=0.7, edgecolor='none'),
        )

    return ax


# ── private helpers ────────────────────────────────────────────────

def _plot_band(ax: Axes, band: Optional[Dict], color: str):
    if band is None:
        return
    ax.fill_between(
        band['x'], band['lo'], band['hi'],
        color=color, alpha=0.20, linewidth=0,
    )


def _plot_data_points(
    ax: Axes,
    data: Optional[Tuple[np.ndarray, np.ndarray]],
    color: str,
    n_bins: int,
):
    if data is None:
        return
    stim, ch = np.asarray(data[0]), np.asarray(data[1])
    valid = ~np.isnan(ch)
    if valid.sum() == 0:
        return
    bins = np.linspace(-1, 1, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        sel = (stim[valid] >= bins[i]) & (stim[valid] < bins[i + 1])
        if i == n_bins - 1:
            sel |= stim[valid] == bins[-1]
        counts[i] = int(sel.sum())
        if counts[i] > 0:
            means[i] = float(np.mean(ch[valid][sel]))
    sizes = 8 + 0.05 * counts
    ax.scatter(
        centres, means, s=sizes, color=color, alpha=0.7, edgecolors='none',
        zorder=3,
    )


def _stats_text(result: Dict, stats_keys: Tuple[str, ...]) -> str:
    diffs = result.get('diffs', {})
    boot_ci = result.get('boot_ci')
    perm_p = result.get('perm_p')
    fisher_p = result.get('fisher_p')

    lines = []
    for key in stats_keys:
        val = diffs.get(key, np.nan)
        if np.isnan(val):
            continue
        label = _LABEL_FOR_KEY.get(key, key)
        line = f"Δ{label:<5s} = {val:+.3f}"
        if boot_ci is not None and key in boot_ci:
            lo, hi = boot_ci[key]
            if not (np.isnan(lo) or np.isnan(hi)):
                line += f" [{lo:+.3f}, {hi:+.3f}]"
        if key == 'accuracy' and fisher_p is not None and not np.isnan(fisher_p):
            line += f"  p={fisher_p:.3f}"
        elif perm_p is not None and key in perm_p:
            p = perm_p[key]
            if p is not None and not np.isnan(p):
                line += f"  p={p:.3f}"
        lines.append(line)

    lines.append(f"n_{result.get('label_a', 'A')} = {result.get('n_a', '?')}")
    lines.append(f"n_{result.get('label_b', 'B')} = {result.get('n_b', '?')}")
    return '\n'.join(lines)
