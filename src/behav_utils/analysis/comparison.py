"""
compute_delta_stat — differences between phases, with uncertainty.

    load_experiment → select_sessions → filter_trials → {label: phase}
                                                             ↓
                 compute_delta_stat(phases, names, reference=...)  → DeltaStats
                                                             ↓
                 compute_interaction(r1, r2, ...)                 → Interaction

Every non-reference phase is contrasted against the reference (Δ = phase −
reference). Each phase is bootstrapped ONCE per resampling unit and the draws
are stored; every contrast and interaction is a subtraction of those stored
draw frames, so a shared reference cancels exactly ((A − C) − (B − C) = A − B)
instead of contributing its variance twice.

Two engines (see ``behav_utils.analysis.resampling``): a per-trial or
per-session bootstrap for intervals, and — only where the label was randomised
per trial (opto within a session type) — a permutation null for a p-value.

Uncertainty is stored as draws and summarised on demand (``.boot(unit)``,
``.table(unit)``), so one result carries the trial-level and session-level
intervals together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from behav_utils.analysis.resampling import (
    bootstrap_phase_stats,
    permute_phase_difference,
    summarise_draw_frame,
)
from behav_utils.data.arrays import TrialArrays
from behav_utils.readouts import (
    PsychometricCurve,
    UpdateMatrix,
    compute_psychometric_curve,
    compute_update_matrix,
)
from behav_utils.readouts.psychometric import PARAMS as PSYCHOMETRIC
from behav_utils.stats import compute_stats, is_exchangeable, validate_names

__all__ = ['DeltaStats', 'PhaseSummary', 'Contrast', 'Interaction',
           'compute_delta_stat', 'compute_interaction', 'contrast_key']

UNITS = ('trials', 'sessions')
MIN_TRIALS_FOR_RESAMPLING = 10


def contrast_key(phase: str, reference: str) -> str:
    return f'{phase}_vs_{reference}'


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PhaseSummary:
    """One condition: observed stats, its bootstrap draws per unit, and optional readouts."""

    label: str
    stats: pd.Series                               # observed, indexed by stat
    n_trials: int                                  # responded trials
    n_sessions: int
    draws: Mapping[str, pd.DataFrame] = field(default_factory=dict)   # unit → (n_draws × stats)
    stats_matched: pd.Series | None = None      # median of matched-n trial draws (downsample=True)
    curve: PsychometricCurve | None = None
    update_matrix: UpdateMatrix | None = None

    @property
    def units(self) -> Tuple[str, ...]:
        return tuple(self.draws)

    def ci(self, unit: str | None = None, *, ci: float = 0.95) -> pd.DataFrame:
        """Bootstrap interval per stat for ``unit`` (default: first available). Columns ci_lo, ci_hi, p, median, n_draws."""
        unit = _pick_unit(self.draws, unit)
        if unit is None:
            return _empty_summary(self.stats.index)
        return summarise_draw_frame(self.draws[unit], ci=ci)

    def __repr__(self) -> str:
        return (f'PhaseSummary({self.label!r}, n_trials={self.n_trials}, '
                f'n_sessions={self.n_sessions}, units={list(self.draws)})')


@dataclass(frozen=True)
class Contrast:
    """Δ = a − b, with difference draws per unit and an optional permutation p."""

    label_a: str
    label_b: str
    diff: pd.Series                                # observed a − b (matched values when downsampled)
    n_a: int
    n_b: int
    n_sessions_a: int
    n_sessions_b: int
    difference_draws: Mapping[str, pd.DataFrame] = field(default_factory=dict)   # unit → draws of Δ
    perm_p: pd.Series | None = None
    um_diff: np.ndarray | None = None
    um_rmse: float = np.nan
    um_corr: float = np.nan

    @property
    def key(self) -> str:
        return contrast_key(self.label_a, self.label_b)

    @property
    def units(self) -> Tuple[str, ...]:
        return tuple(self.difference_draws)

    def boot(self, unit: str | None = None, *, ci: float = 0.95) -> pd.DataFrame:
        """Bootstrap ci_lo, ci_hi, p (two-sided vs 0), median, n_draws per stat."""
        unit = _pick_unit(self.difference_draws, unit)
        if unit is None:
            return _empty_summary(self.diff.index)
        return summarise_draw_frame(self.difference_draws[unit], ci=ci)

    def table(self, unit: str | None = None, *, ci: float = 0.95) -> pd.DataFrame:
        """One row per stat: diff, ci_lo, ci_hi, boot_p, perm_p, unit."""
        b = self.boot(unit, ci=ci)
        unit = _pick_unit(self.difference_draws, unit)
        t = pd.DataFrame({'stat': self.diff.index, 'diff': self.diff.to_numpy()})
        t['ci_lo'] = b['ci_lo'].reindex(t['stat']).to_numpy()
        t['ci_hi'] = b['ci_hi'].reindex(t['stat']).to_numpy()
        t['boot_p'] = b['p'].reindex(t['stat']).to_numpy()
        t['perm_p'] = (self.perm_p.reindex(t['stat']).to_numpy() if self.perm_p is not None else np.nan)
        t['unit'] = unit
        return t

    def __repr__(self) -> str:
        return (f'Contrast({self.key!r}, n_a={self.n_a}, n_b={self.n_b}, '
                f'units={list(self.difference_draws)}, perm={self.perm_p is not None})')


@dataclass(frozen=True)
class DeltaStats:
    """Result of :func:`compute_delta_stat`."""

    phases: Dict[str, PhaseSummary]
    contrasts: Dict[str, Contrast]       # keyed by ``contrast_key(label, reference)``
    reference: str
    names: Tuple[str, ...]
    units: Tuple[str, ...]
    downsample: bool = False
    n_matched: Mapping[str, int] = field(default_factory=dict)
    seed: int = 0

    @property
    def primary_unit(self) -> str:
        return self.units[0] if self.units else 'trials'

    def contrast(self, label: str) -> Contrast:
        """The contrast of ``label`` vs the reference."""
        return self.contrasts[contrast_key(label, self.reference)]

    def table(self, unit: str | None = None, *, ci: float = 0.95) -> pd.DataFrame:
        """All contrasts stacked; adds a ``contrast`` column."""
        frames = [c.table(unit, ci=ci).assign(contrast=k) for k, c in self.contrasts.items()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def __repr__(self) -> str:
        return (f'DeltaStats(phases={list(self.phases)}, reference={self.reference!r}, '
                f'stats={list(self.names)}, units={list(self.units)}, downsample={self.downsample})')


@dataclass(frozen=True)
class Interaction:
    """Difference of differences: (Δ_a) − (Δ_b), bootstrap only."""

    label_a: str
    label_b: str
    contrast_a: str
    contrast_b: str
    delta_a: pd.Series
    delta_b: pd.Series
    interaction: pd.Series
    draws: Mapping[str, pd.DataFrame]              # unit → draws of the interaction
    draws_a: Mapping[str, pd.DataFrame]            # unit → draws of Δ_a (context)
    draws_b: Mapping[str, pd.DataFrame]
    shared_result: bool = False

    @property
    def units(self) -> Tuple[str, ...]:
        return tuple(self.draws)

    def table(self, unit: str | None = None, *, ci: float = 0.95) -> pd.DataFrame:
        """Per stat: delta_a, ci_a_lo/hi, delta_b, ci_b_lo/hi, interaction, ci_lo, ci_hi, p, n_draws."""
        unit = _pick_unit(self.draws, unit)
        if unit is None:
            return pd.DataFrame()
        s = summarise_draw_frame(self.draws[unit], ci=ci)
        sa = summarise_draw_frame(self.draws_a[unit], ci=ci)
        sb = summarise_draw_frame(self.draws_b[unit], ci=ci)
        idx = s.index
        return pd.DataFrame({
            'stat': idx,
            'delta_a': self.delta_a.reindex(idx).to_numpy(),
            'ci_a_lo': sa['ci_lo'].reindex(idx).to_numpy(), 'ci_a_hi': sa['ci_hi'].reindex(idx).to_numpy(),
            'delta_b': self.delta_b.reindex(idx).to_numpy(),
            'ci_b_lo': sb['ci_lo'].reindex(idx).to_numpy(), 'ci_b_hi': sb['ci_hi'].reindex(idx).to_numpy(),
            'interaction': self.interaction.reindex(idx).to_numpy(),
            'ci_lo': s['ci_lo'].to_numpy(), 'ci_hi': s['ci_hi'].to_numpy(),
            'p': s['p'].to_numpy(), 'n_draws': s['n_draws'].to_numpy(),
            'unit': unit,
        })

    def __repr__(self) -> str:
        return (f'Interaction({self.label_a!r} {self.contrast_a} − {self.label_b!r} {self.contrast_b}, '
                f'units={list(self.draws)})')


def _pick_unit(draws: Mapping[str, pd.DataFrame], unit: str | None) -> str | None:
    if not draws:
        return None
    if unit is None:
        return next(iter(draws))
    if unit not in draws:
        raise KeyError(f"no draws for unit {unit!r}; available: {list(draws)}")
    return unit


def _empty_summary(index) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=list(index),
                        columns=['ci_lo', 'ci_hi', 'p', 'median', 'n_draws'])


# ─────────────────────────────────────────────────────────────────────────────
# compute_delta_stat
# ─────────────────────────────────────────────────────────────────────────────

def compute_delta_stat(
    phases,
    names: Sequence[str] = PSYCHOMETRIC,
    *,
    labels: Sequence[str] | None = None,
    reference: str | None = None,
    curve: bool = True,
    update_matrix: bool = False,
    downsample: bool = False,
    n_permutations: int = 1000,
    n_bootstrap: int = 1000,
    units: Sequence[str] = ('trials',),
    n_bins: int = 8,
    trial_filter: str = 'post_correct',
    n_repeats: int = 200,
    seed: int = 42,
) -> DeltaStats:
    """Compare N phases against a reference, with bootstrap CI and permutation p.

    Args:
        phases:         ``{label: phase}``, or a list of phases with ``labels``.
                        Each phase is the output of ``filter_trials``.
        names:          scalar stat names (default: the four psychometric params).
        reference:      label the others are contrasted against (default: first).
        curve:          also fit a :class:`PsychometricCurve` per phase (display).
        update_matrix:  also fit an :class:`UpdateMatrix` per phase and report
                        the matrix difference, RMSE and correlation per contrast
                        (descriptive only — no resampling; one matrix is nine fits).
        downsample:     match trial counts across phases before comparing
                        (matched n per unit: trials for stats and the curve,
                        post-correct pairs for the matrix). Removes precision
                        bias at the cost of power.
        n_permutations: label shuffles for the p-value (0 to skip). Floor is
                        ``1 / (n_permutations + 1)``.
        n_bootstrap:    resamples per phase per unit (0 to skip).
        units:          resampling units, e.g. ``('trials', 'sessions')``. The
                        first is the default for ``.boot()`` / ``.table()``. A
                        phase with < 2 sessions gets no session draws.
        n_bins, trial_filter: update-matrix settings.
        n_repeats:      matched-n draws for the readouts when downsampling.
        seed:           base RNG seed.

    Returns:
        :class:`DeltaStats`.

    Raises:
        ValueError: fewer than two phases, unknown reference or unit, or an
            order-dependent stat requested with trial-level resampling.
    """
    items = _as_items(phases, labels)
    order = [lab for lab, _ in items]
    reference = order[0] if reference is None else reference
    if reference not in order:
        raise ValueError(f'compute_delta_stat: reference {reference!r} not among {order}')

    names = tuple(validate_names(names))
    units = tuple(dict.fromkeys(units))
    bad_units = [u for u in units if u not in UNITS]
    if bad_units:
        raise ValueError(f'compute_delta_stat: units must be in {UNITS}, got {bad_units}')

    trial_level = n_permutations > 0 or (n_bootstrap > 0 and 'trials' in units)
    if trial_level:
        bad = [s for s in names if not is_exchangeable(s)]
        if bad:
            raise ValueError(
                f'compute_delta_stat: order-dependent stat(s) {bad} cannot be permuted or '
                f'trial-bootstrapped. Drop them, use units=("sessions",) with '
                f'n_permutations=0, or set n_permutations=n_bootstrap=0.')

    n_matched: Dict[str, int] = {}
    if downsample:
        from behav_utils.analysis.downsample import calculate_min_n
        groups = [ph for _, ph in items]
        n_matched['trials'] = calculate_min_n(groups, unit='trials')
        if update_matrix:
            n_matched['pairs'] = calculate_min_n(groups, unit='pairs')
    draw_n = n_matched.get('trials') if downsample else None

    # ── per phase ─────────────────────────────────────────────────────────
    summaries: Dict[str, PhaseSummary] = {}
    for i, (label, phase) in enumerate(items):
        arrays = TrialArrays.from_sessions(phase)
        n_trials, n_sessions = arrays.n_responded, len(phase)
        observed = compute_stats(arrays, names, rng=np.random.default_rng(seed + i))

        draws: Dict[str, pd.DataFrame] = {}
        if n_bootstrap > 0 and names:
            for u_idx, unit in enumerate(units):
                if unit == 'trials' and n_trials < MIN_TRIALS_FOR_RESAMPLING:
                    continue
                if unit == 'sessions' and n_sessions < 2:
                    continue
                u_seed = (seed + i) if u_idx == 0 else (seed + i + u_idx * 1_000_003)
                draws[unit] = bootstrap_phase_stats(
                    phase, names, n_draws=n_bootstrap, n_trials=draw_n, seed=u_seed, unit=unit)

        matched = None
        if downsample and 'trials' in draws:
            matched = draws['trials'].median(axis=0, skipna=True)

        curve_obj = um_obj = None
        if curve:
            if downsample:
                from behav_utils.analysis.downsample import resample_psychometric_curve
                curve_obj = resample_psychometric_curve(
                    phase, n_matched['trials'], n_repeats=n_repeats, seed=seed + i)
            else:
                curve_obj = compute_psychometric_curve(arrays)
        if update_matrix:
            if downsample:
                from behav_utils.analysis.downsample import resample_update_matrix
                um_obj = resample_update_matrix(
                    phase, n_matched['pairs'], n_repeats=n_repeats, n_bins=n_bins,
                    trial_filter=trial_filter, seed=seed + i)
            else:
                um_obj = compute_update_matrix(arrays, n_bins=n_bins, trial_filter=trial_filter)

        summaries[label] = PhaseSummary(label, observed, int(n_trials), int(n_sessions),
                                        draws, matched, curve_obj, um_obj)

    # ── contrasts vs reference ───────────────────────────────────────────
    ref = summaries[reference]
    contrasts: Dict[str, Contrast] = {}
    for k, label in enumerate(lab for lab in order if lab != reference):
        ph = summaries[label]
        a = ph.stats_matched if ph.stats_matched is not None else ph.stats
        b = ref.stats_matched if ref.stats_matched is not None else ref.stats
        diff = (a - b).reindex(list(names))

        diff_draws = {u: ph.draws[u] - ref.draws[u]
                      for u in units if u in ph.draws and u in ref.draws}

        perm_p = None
        if (n_permutations > 0 and names
                and ph.n_trials >= MIN_TRIALS_FOR_RESAMPLING
                and ref.n_trials >= MIN_TRIALS_FOR_RESAMPLING):
            null = permute_phase_difference(
                dict(items)[label], dict(items)[reference], names,
                n_draws=n_permutations, n_trials=draw_n, seed=seed + 1000 + k)
            perm_p = summarise_draw_frame(null, observed=diff)['p'].reindex(list(names))

        um_diff, um_rmse, um_corr = None, np.nan, np.nan
        if update_matrix and ph.update_matrix is not None and ref.update_matrix is not None:
            ma, mb = ph.update_matrix.matrix, ref.update_matrix.matrix
            um_diff = ma - mb
            usable = np.isfinite(ma) & np.isfinite(mb)
            if usable.sum() >= 4:
                from scipy.stats import pearsonr
                um_rmse = float(np.sqrt(np.mean(um_diff[usable] ** 2)))
                um_corr = float(pearsonr(ma[usable], mb[usable])[0])

        contrasts[contrast_key(label, reference)] = Contrast(
            label, reference, diff, ph.n_trials, ref.n_trials, ph.n_sessions, ref.n_sessions,
            diff_draws, perm_p, um_diff, um_rmse, um_corr)

    return DeltaStats(summaries, contrasts, reference, names, units, downsample, n_matched, seed)


def _as_items(phases, labels):
    if isinstance(phases, Mapping):
        items = list(phases.items())
    else:
        phases = list(phases)
        if labels is None:
            labels = [f'phase_{i}' for i in range(len(phases))]
        if len(labels) != len(phases):
            raise ValueError('compute_delta_stat: len(labels) != len(phases)')
        items = list(zip(labels, phases))
    if len(items) < 2:
        raise ValueError('compute_delta_stat: need at least two phases')
    return items


# ─────────────────────────────────────────────────────────────────────────────
# compute_interaction
# ─────────────────────────────────────────────────────────────────────────────

def compute_interaction(
    result_a: DeltaStats,
    result_b: DeltaStats,
    contrast: str,
    *,
    contrast_b: str | None = None,
    label_a: str = 'a',
    label_b: str = 'b',
) -> Interaction:
    """Difference of differences between two :class:`DeltaStats` results.

    Answers "is the opto effect different in *these* sessions than in *those*"
    — e.g. whether the on − off shift on sham sessions differs from
    the shift on real opto sessions, which is what separates an inactivation
    effect from the light-delivery artefact. Comparing two p-values is not a
    substitute: "significant here, not there" is not evidence they differ.

    Bootstrap only: session type was not randomised per trial, so no shuffle
    reproduces the design.

    Args:
        result_a, result_b: outputs of ``compute_delta_stat``. Pass the same
            object twice to compare two contrasts within one result (their
            shared reference draws then cancel exactly).
        contrast:   contrast key in ``result_a``, e.g. 'opto_vs_non_opto'.
        contrast_b: contrast key in ``result_b``; defaults to ``contrast``.
        label_a, label_b: names for the two results in the output.

    Raises:
        KeyError:   if either contrast is absent.
        ValueError: if the two contrasts share no bootstrap unit.
    """
    contrast_b = contrast_b or contrast
    for res, key, which in ((result_a, contrast, 'result_a'), (result_b, contrast_b, 'result_b')):
        if key not in res.contrasts:
            raise KeyError(f'compute_interaction: {which} has no contrast {key!r}; '
                           f'available: {sorted(res.contrasts)}')
    ca, cb = result_a.contrasts[contrast], result_b.contrasts[contrast_b]
    common = [u for u in ca.difference_draws if u in cb.difference_draws]
    if not common:
        raise ValueError('compute_interaction: no shared bootstrap unit — rerun '
                         'compute_delta_stat with n_bootstrap > 0 and matching units.')
    draws, draws_a, draws_b = {}, {}, {}
    for u in common:
        da, db = ca.difference_draws[u], cb.difference_draws[u]
        n = min(len(da), len(db))
        cols = [c for c in da.columns if c in db.columns]
        draws_a[u], draws_b[u] = da[cols].iloc[:n], db[cols].iloc[:n]
        draws[u] = draws_a[u] - draws_b[u]
    cols = list(draws[common[0]].columns)
    delta_a, delta_b = ca.diff.reindex(cols), cb.diff.reindex(cols)
    return Interaction(label_a, label_b, contrast, contrast_b, delta_a, delta_b,
                       delta_a - delta_b, draws, draws_a, draws_b, result_a is result_b)
