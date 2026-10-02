"""
Draw-only plotters for :class:`DeltaStats` and :class:`Interaction`.

    r = compute_delta_stat({'non_opto': ctrl, 'opto': opto}, reference='non_opto',
                           units=('trials', 'sessions'))
    plot_comparison(r, 'opto_vs_non_opto')                 # curves + Δ text
    plot_stat_comparison(r, ['mu', 'sigma', 'accuracy'])   # point + CI per phase
    ix = compute_interaction(r_opto, r_mask, 'opto_vs_non_opto')
    plot_interaction(ix)                                   # difference of differences

Intervals: one per requested unit, offset at the same x. When two units are
drawn, the trial CI is thin and grey (it ignores session scatter) and the
session CI thick and coloured (the honest interval for a between-phase
comparison). A hollow marker flags an estimate lying outside its own interval.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from behav_utils.analysis.comparison import Contrast, DeltaStats, Interaction
from behav_utils.plotting.readouts import plot_psychometric_curve

_LABEL = {'mu': 'PSE', 'sigma': 'slope', 'lapse_low': 'λ_low', 'lapse_high': 'λ_high',
          'accuracy': 'Acc'}


def _label(stat: str) -> str:
    return _LABEL.get(stat, stat)


def _pick_contrast(result: DeltaStats, contrast: str | None) -> Contrast:
    if contrast is None:
        if len(result.contrasts) != 1:
            raise KeyError(f'several contrasts present, pass one of {sorted(result.contrasts)}')
        contrast = next(iter(result.contrasts))
    if contrast not in result.contrasts:
        raise KeyError(f'no contrast {contrast!r}; available: {sorted(result.contrasts)}')
    return result.contrasts[contrast]


def _draw_intervals(ax: Axes, x: float, value: float, intervals: Dict[str, Tuple[float, float]],
                    colour: str, *, lw: float = 1.4, star_sessions: bool = False) -> bool:
    """Draw one CI per unit at x; return whether ``value`` lies outside the honest one."""
    units = [u for u, (lo, hi) in intervals.items() if np.isfinite(lo) and np.isfinite(hi)]
    multi = len(units) > 1
    outside = False
    for j, u in enumerate(units):
        lo, hi = intervals[u]
        dx = (j - (len(units) - 1) / 2) * 0.16 if multi else 0.0
        honest = (u == 'sessions') or not multi
        c = colour if honest else '0.55'
        w = lw * (1.7 if (honest and multi) else 1.0)
        ax.plot([x + dx, x + dx], [lo, hi], color=c, lw=w, solid_capstyle='butt', zorder=2)
        for cap in (lo, hi):
            ax.plot([x + dx - 0.06, x + dx + 0.06], [cap, cap], color=c, lw=w, zorder=2)
        if star_sessions and u == 'sessions' and (lo > 0 or hi < 0):
            ax.annotate('\u2217', (x + dx, hi), textcoords='offset points', xytext=(0, 3),
                        ha='center', va='bottom', fontsize=11, color=c, zorder=5)
        if honest:
            outside = value < lo or value > hi
    return outside


def _marker(ax: Axes, x: float, value: float, colour: str, outside: bool, size: float = 7, z: int = 3):
    ax.plot(x, value, marker='o', markersize=size,
            markerfacecolor='white' if outside else colour, markeredgecolor=colour,
            markeredgewidth=1.6 if outside else 0.6, zorder=z)


# ── curves ──────────────────────────────────────────────────────────────────

def _delta_text(con: Contrast, stats: Sequence[str], unit: str | None) -> str:
    t = con.table(unit if con.units else None).set_index('stat') if con.units else None
    lines = []
    for key in stats:
        if key not in con.diff.index or not np.isfinite(con.diff[key]):
            continue
        line = f'Δ{_label(key):<5s} = {con.diff[key]:+.3f}'
        if t is not None and key in t.index and np.isfinite(t.loc[key, 'ci_lo']):
            line += f" [{t.loc[key, 'ci_lo']:+.3f}, {t.loc[key, 'ci_hi']:+.3f}]"
        if con.perm_p is not None and key in con.perm_p.index and np.isfinite(con.perm_p[key]):
            line += f'  p={con.perm_p[key]:.3f}'
        lines.append(line)
    lines.append(f'n_{con.label_a} = {con.n_a}')
    lines.append(f'n_{con.label_b} = {con.n_b}')
    return '\n'.join(lines)


def plot_comparison(
    result: DeltaStats,
    contrast: str | None = None,
    ax: Axes | None = None,
    *,
    color_a: str = '#d62728',
    color_b: str = '#444444',
    show_stats: bool = True,
    show_band: bool = True,
    show_data: bool = True,
    stats_keys: Sequence[str] = ('mu', 'sigma', 'accuracy'),
    unit: str | None = None,
) -> Axes:
    """Two psychometric curves (phase over reference) with the Δ, CI and p annotated.

    Requires ``compute_delta_stat(..., curve=True)``. ``unit`` selects which
    bootstrap CI is quoted in the text (default: the result's primary unit).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    con = _pick_contrast(result, contrast)
    pa, pb = result.phases[con.label_a], result.phases[con.label_b]
    if pa.curve is None or pb.curve is None:
        raise KeyError('plot_comparison: no curve on one of the phases — call '
                       'compute_delta_stat with curve=True.')
    for ph, colour in ((pb, color_b), (pa, color_a)):
        plot_psychometric_curve(ph.curve, ax=ax, color=colour, label=ph.label,
                                show_data=show_data, show_band=show_band, show_reference=False)
    ax.axhline(0.5, color='k', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.axvline(0.0, color='k', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Stimulus (distance from boundary)')
    ax.legend(loc='lower right', fontsize=8)
    if show_stats:
        ax.text(0.03, 0.97, _delta_text(con, stats_keys, unit), transform=ax.transAxes,
                va='top', ha='left', fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    return ax


# ── per-stat point + CI ────────────────────────────────────────────────────

def _phase_order(result: DeltaStats, phase_order):
    if phase_order:
        return list(phase_order)
    return [result.reference] + [p for p in result.phases if p != result.reference]


def plot_stat_comparison_single(
    result: DeltaStats,
    stat: str,
    ax: Axes | None = None,
    *,
    phase_order: Sequence[str] | None = None,
    palette: Sequence[str] | None = None,
    show_p: bool = True,
    units: Sequence[str] | None = None,
) -> Axes:
    """One stat: a point per phase with its bootstrap CI(s); p above each non-reference phase.

    The point is the pooled estimate and the interval its sampling uncertainty
    (not a session spread). ``units`` selects which intervals to draw
    (default: the result's primary unit only). Above a non-reference phase the
    permutation p is shown when present; otherwise the bootstrap p per unit
    (``pt`` trials, ``ps`` sessions).
    """
    from behav_utils.plotting.styles import get_colour
    order = _phase_order(result, phase_order)
    colours = list(palette) if palette else [get_colour(i) for i in range(len(order))]
    units = list(units) if units else [result.primary_unit]
    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 3.0))

    xs = np.arange(len(order))
    tops = []
    for i, name in enumerate(order):
        ph = result.phases[name]
        v = float(ph.stats.get(stat, np.nan))
        if not np.isfinite(v):
            continue
        colour = colours[i % len(colours)]
        intervals = {}
        for u in units:
            if u in ph.draws and stat in ph.draws[u].columns:
                s = ph.ci(u).loc[stat]
                intervals[u] = (s['ci_lo'], s['ci_hi'])
        outside = _draw_intervals(ax, xs[i], v, intervals, colour)
        _marker(ax, xs[i], v, colour, outside)
        tops.append(max([v] + [hi for _, hi in intervals.values() if np.isfinite(hi)]))

    if show_p and tops:
        vals = [float(result.phases[n].stats.get(stat, np.nan)) for n in order]
        finite = [v for v in vals if np.isfinite(v)]
        span = (max(finite) - min(finite)) or (abs(max(finite)) or 1.0)
        top = max(tops)
        for i, name in enumerate(order):
            if name == result.reference or name not in [c.label_a for c in result.contrasts.values()]:
                continue
            con = result.contrast(name)
            pp = con.perm_p[stat] if con.perm_p is not None and stat in con.perm_p.index else np.nan
            if np.isfinite(pp):
                ax.annotate(f'p={pp:.3g}', xy=(xs[i], top + 0.12 * span), ha='center',
                            va='bottom', fontsize=7.5)
                continue
            parts = []
            for u in units:
                if u in con.difference_draws and stat in con.difference_draws[u].columns:
                    pu = con.boot(u).loc[stat, 'p']
                    if np.isfinite(pu):
                        parts.append(('s' if u == 'sessions' else 't', pu))
            for k, (tag, pu) in enumerate(parts):
                ax.annotate(f'p{tag}={pu:.2g}', xy=(xs[i], top + (0.12 + 0.13 * k) * span),
                            ha='center', va='bottom', fontsize=7.5,
                            color='0.55' if tag == 't' else 'k')
        ax.margins(y=0.22)

    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=8)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_title(_label(stat), fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    return ax


def plot_stat_comparison(
    result: DeltaStats,
    stats: Sequence[str] | None = None,
    *,
    ncols: int = 3,
    phase_order: Sequence[str] | None = None,
    palette: Sequence[str] | None = None,
    panel_size: tuple = (3.0, 3.0),
    suptitle: str | None = None,
    show_p: bool = True,
    units: Sequence[str] | None = None,
):
    """Grid of :func:`plot_stat_comparison_single`, one panel per stat. Returns ``(fig, axes)``."""
    from behav_utils.plotting.styles import get_colour
    stats = list(stats) if stats is not None else list(result.names)
    if not stats:
        raise ValueError('plot_stat_comparison: no stats to plot')
    order = _phase_order(result, phase_order)
    colours = list(palette) if palette else [get_colour(i) for i in range(len(order))]
    ncols = max(1, min(ncols, len(stats)))
    nrows = int(np.ceil(len(stats) / ncols))
    fig, axarr = plt.subplots(nrows, ncols, squeeze=False,
                              figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    flat = axarr.ravel()
    for ax, stat in zip(flat, stats):
        plot_stat_comparison_single(result, stat, ax=ax, phase_order=order, palette=colours,
                                    show_p=show_p, units=units)
    for ax in flat[len(stats):]:
        fig.delaxes(ax)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig, list(flat[:len(stats)])


# ── interaction ────────────────────────────────────────────────────────────

def plot_interaction_single(
    interaction: Interaction,
    stat: str,
    ax: Axes | None = None,
    *,
    show_p: bool = True,
    show_components: bool = True,
    colour_a: str = '#1f77b4',
    colour_b: str = '#ff7f0e',
    colour_interaction: str = '#7f2704',
    units: Sequence[str] | None = None,
) -> Axes:
    """One stat: Δ_a, Δ_b (context) and their difference with its CI and bootstrap p.

    ``units`` selects which interaction CI(s) to draw (default: the first unit
    in the result). A '∗' marks a session CI that excludes zero.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 3.0))
    units = list(units) if units else [interaction.units[0]]
    units = [u for u in units if u in interaction.draws]
    if not units:
        raise KeyError(f'no draws for {units}; available: {list(interaction.units)}')
    t = {u: interaction.table(u).set_index('stat') for u in units}
    u0 = units[0]
    if stat not in t[u0].index:
        raise KeyError(f'no stat {stat!r} in interaction; available: {list(t[u0].index)}')
    row = t[u0].loc[stat]

    if show_components:
        positions = [f'{interaction.label_a}\n{interaction.contrast_a}',
                     f'{interaction.label_b}\n{interaction.contrast_b}', 'difference']
        values = [row['delta_a'], row['delta_b'], row['interaction']]
        intervals = [{u0: (row['ci_a_lo'], row['ci_a_hi'])}, {u0: (row['ci_b_lo'], row['ci_b_hi'])},
                     {u: (t[u].loc[stat, 'ci_lo'], t[u].loc[stat, 'ci_hi']) for u in units}]
        colours = [colour_a, colour_b, colour_interaction]
    else:
        positions = ['difference']
        values = [row['interaction']]
        intervals = [{u: (t[u].loc[stat, 'ci_lo'], t[u].loc[stat, 'ci_hi']) for u in units}]
        colours = [colour_interaction]

    ax.axhline(0.0, color='0.6', lw=1, ls='--', zorder=0)
    for i, (v, iv, c) in enumerate(zip(values, intervals, colours)):
        if not np.isfinite(v):
            continue
        last = i == len(values) - 1
        outside = _draw_intervals(ax, i, v, iv, c, lw=1.8 if last else 1.3, star_sessions=last)
        _marker(ax, i, v, c, outside, size=8 if last else 6.5, z=4 if last else 3)

    if show_p:
        pairs = [('' if len(units) == 1 else ('s' if u == 'sessions' else 't'), t[u].loc[stat, 'p'])
                 for u in units if np.isfinite(t[u].loc[stat, 'p'])]
        finite = [(v, iv) for v, iv in zip(values, intervals) if np.isfinite(v)]
        if finite and pairs:
            his = [hi for _, iv in finite for _, hi in iv.values() if np.isfinite(hi)]
            los = [lo for _, iv in finite for lo, _ in iv.values() if np.isfinite(lo)]
            top = max([v for v, _ in finite] + his)
            bottom = min([v for v, _ in finite] + los)
            span = (top - bottom) or (abs(top) or 1.0)
            for k, (tag, pu) in enumerate(pairs):
                label = f'p = {pu:.3g}' if tag == '' else f'p{tag}={pu:.2g}'
                ax.annotate(label, xy=(len(values) - 1, top + (0.10 + 0.13 * k) * span),
                            ha='center', va='bottom', fontsize=8,
                            color='0.55' if tag == 't' else colour_interaction)
            ax.margins(y=0.22 + 0.13 * max(0, len(pairs) - 1))

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(positions, fontsize=8, rotation=20 if show_components else 0,
                       ha='right' if show_components else 'center')
    ax.set_xlim(-0.6, len(positions) - 0.4)
    ax.set_title(_label(stat), fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    return ax


def plot_interaction(
    interaction: Interaction,
    stats: Sequence[str] | None = None,
    *,
    ncols: int = 4,
    panel_size: tuple = (3.0, 3.0),
    suptitle: str | None = None,
    show_p: bool = True,
    show_components: bool = True,
    colour_a: str = '#1f77b4',
    colour_b: str = '#ff7f0e',
    colour_interaction: str = '#7f2704',
    units: Sequence[str] | None = None,
):
    """Grid of :func:`plot_interaction_single`, one panel per stat, each with its own y-scale."""
    stats = list(stats) if stats is not None else list(interaction.interaction.index)
    if not stats:
        raise ValueError('plot_interaction: no stats to plot')
    ncols = max(1, min(ncols, len(stats)))
    nrows = int(np.ceil(len(stats) / ncols))
    fig, axarr = plt.subplots(nrows, ncols, squeeze=False,
                              figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    flat = axarr.ravel()
    for ax, stat in zip(flat, stats):
        plot_interaction_single(interaction, stat, ax=ax, show_p=show_p,
                                show_components=show_components, colour_a=colour_a,
                                colour_b=colour_b, colour_interaction=colour_interaction, units=units)
    for ax in flat[len(stats):]:
        fig.delaxes(ax)
    if suptitle:
        fig.suptitle(f'{suptitle}', fontsize=13)
    fig.tight_layout()
    return fig, list(flat[:len(stats)])
