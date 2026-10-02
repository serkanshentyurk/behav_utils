"""Group-level combination, resampling and testing — pure numeric, no behavioural objects.

This is the "Tier B" layer: every function takes a tidy stat table (the
``sessions`` frame from :func:`behav_utils.analysis.statistics.compute_phase_stats`) or
plain arrays, never ``SessionData`` / ``AnimalData``. Each verb does one job, so
combining / bootstrapping / testing compose freely:

    combine(points, over='session')      # sessions → one value per animal
    paired_diff(points, by='condition', a='opto', b='nonopto')   # per-animal Δ
    bootstrap_units(delta_values)        # across-animal CI on the Δ
    rank_test(het_delta, wt_delta)       # group difference test
    average_arrays([um for animal in ...])   # element-wise mean curve / UM

Two distinct resampling levels live in this codebase and must not be confused:

  * within-unit (trial-level) — produced at extraction time by
    ``compute_delta_stat`` bootstrap draws and stored in the ``ci_*_within`` columns /
    the reps table. Answers "how precise is THIS unit's estimate".
  * across-unit (cluster) — :func:`bootstrap_units` here, resampling the
    per-unit ``value`` array. Answers "how much does the effect vary across
    animals". This is the interval you want for a group claim.

The two are different operations with different answers; this module only does
the across-unit one.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, Iterable, Sequence, Union

import numpy as np
import pandas as pd

# Columns that are values/weights/within-unit-CIs/axes/bookkeeping — never grouping
# keys. 'n_units' is combine's own output column: listing it here makes combine
# idempotent under chaining (combine(over='session') then combine(over='animal')).
_NON_KEY = {'value', 'n_trials', 'ci_lo_within', 'ci_hi_within', 'n_units'}


# ─────────────────────────────────────────────────────────────────────────────
# Combine — collapse one axis of a tidy stat table
# ─────────────────────────────────────────────────────────────────────────────
def combine(
    points: pd.DataFrame,
    *,
    over: str = 'session',
    how: str = 'mean',
    weight: Union[str, None] = None,
) -> pd.DataFrame:
    """Collapse one axis of a tidy stat table, carrying all other label columns.

    Groups by every column except ``over`` and the value/weight/CI columns, so it
    can never silently average across conditions, phases, stats or groups.

      * ``over='session'`` → one value per (animal, stat, label…)  — per-animal.
      * ``over='animal'``  → one value per (stat, label…)          — per-group.

    This is the *average* path. The trial-weighted alternative (re-pool trials,
    compute once) comes from ``compute_phase_stats(...).to_rows()`` instead — a
    deliberately different number. Within-unit CIs are dropped on output because
    they no longer apply; a group CI comes from :func:`bootstrap_units`.

    Args:
        points: tidy stat frame (columns include ``value``, ``n_trials`` and the
                axis ``over``).
        over:   axis column to collapse.
        how:    'mean' or 'median'.
        weight: None (equal weight) or 'n_trials' (trial-weighted mean; ignored
                when ``how='median'``).

    Returns:
        Collapsed frame: grouping keys + ``value``, ``n_trials`` (summed) and
        ``n_units`` (number of contributing rows).

    Raises:
        ValueError: if ``over`` is absent, already collapsed (all-<NA>), or if no
            grouping columns would remain.
    """
    if over not in points.columns:
        raise ValueError(f"combine: column {over!r} not in table; nothing to collapse over")
    if points[over].isna().all():
        raise ValueError(
            f"combine: axis {over!r} is already collapsed (all <NA>); the table "
            f"is already at that level"
        )
    if how not in ('mean', 'median'):
        raise ValueError(f"combine: how must be 'mean' or 'median', got {how!r}")

    keys = [c for c in points.columns if c not in (_NON_KEY | {over})]
    if not keys:
        raise ValueError("combine: no grouping columns remain; refusing to collapse "
                         "the whole table into a single value")

    has_w = 'n_trials' in points.columns

    def _value(g: pd.DataFrame) -> float:
        v = g['value'].to_numpy(dtype=float)
        w = g['n_trials'].to_numpy(dtype=float) if has_w else None
        m = np.isfinite(v)
        if not m.any():
            return np.nan
        if how == 'median':
            return float(np.nanmedian(v[m]))
        if weight == 'n_trials' and w is not None:
            mw = m & np.isfinite(w) & (w > 0)
            if mw.any():
                return float(np.average(v[mw], weights=w[mw]))
        return float(np.mean(v[m]))

    # Explicit groupby iteration rather than .apply(): avoids the pandas-version
    # churn around include_groups / grouping-column handling, and is transparent.
    records = []
    for key_vals, g in points.groupby(keys, dropna=False, sort=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        v = g['value'].to_numpy(dtype=float)
        n_units = int(np.isfinite(v).sum())
        w_sum = (float(np.nansum(g['n_trials'].to_numpy(dtype=float)))
                 if has_w else np.nan)
        rec = dict(zip(keys, key_vals))
        rec['value'] = _value(g)
        rec['n_trials'] = w_sum
        rec['n_units'] = n_units
        records.append(rec)

    return pd.DataFrame.from_records(records, columns=keys + ['value', 'n_trials', 'n_units'])


# ─────────────────────────────────────────────────────────────────────────────
# Paired difference — two label values → per-unit Δ
# ─────────────────────────────────────────────────────────────────────────────
def paired_diff(
    points: pd.DataFrame,
    *,
    by: str,
    a: str,
    b: str,
    keys: Sequence[str] = ('animal',),
    value: str = 'value',
) -> pd.DataFrame:
    """Per-unit difference ``value[a] - value[b]`` across label column ``by``.

    Expects one row per (unit, ``by``-level, stat) — i.e. animal-level input
    (use ``compute_phase_stats(...).to_rows()`` or :func:`combine` first). Inner-joins
    the two levels on every identifying column, so units missing ``a`` or ``b``
    are dropped — and a warning reports how many.

    Produces the Δ table used both for the opto−nonopto effect and, applied twice,
    the difference-of-differences.

    Args:
        points: tidy stat frame.
        by:     label column holding the two conditions.
        a, b:   the two levels of ``by``; difference is ``a - b``.
        keys:   the unit identifier(s) (e.g. ``('animal',)``); used only to name
                dropped units in the warning.
        value:  value column.

    Returns:
        Frame with the identifying columns, ``a`` and ``b`` (the two values) and
        ``delta``.

    Raises:
        ValueError: if ``by``, ``a`` or ``b`` is absent.
    """
    if by not in points.columns:
        raise ValueError(f"paired_diff: column {by!r} not in table")
    keys = list(keys)

    # The value column is excluded from the index regardless of its name, so
    # paired_diff chains: the Δ table it produces (value column 'delta') can be
    # fed straight back in with value='delta' for a difference-of-differences.
    drop = _NON_KEY | {by, 'session', value}
    idx = [c for c in points.columns if c not in drop]
    if not idx:
        raise ValueError("paired_diff: no identifying columns remain to pair on")

    sub = points[points[by].isin([a, b])]
    wide = sub.pivot_table(index=idx, columns=by, values=value, aggfunc='mean')
    for lvl in (a, b):
        if lvl not in wide.columns:
            raise ValueError(f"paired_diff: level {lvl!r} not found under {by!r}")

    before = len(wide)
    paired = wide.dropna(subset=[a, b])
    dropped = before - len(paired)

    out = paired.reset_index()
    out['delta'] = paired[a].to_numpy(dtype=float) - paired[b].to_numpy(dtype=float)

    if dropped:
        lost = wide[wide[[a, b]].isna().any(axis=1)].reset_index()
        key = keys[0] if keys and keys[0] in lost.columns else None
        ids = sorted(map(str, lost[key].unique())) if key else None
        msg = f"paired_diff: dropped {dropped} unit(s) missing {a!r} or {b!r}"
        if ids:
            msg += f" ({key}={ids})"
        warnings.warn(msg, stacklevel=2)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Across-unit bootstrap — the group CI
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_units(
    values: Iterable[float],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int = 0,
) -> Dict[str, float]:
    """Across-unit (cluster) bootstrap CI on a per-unit value array.

    Resamples the units (animals) with replacement — this is the interval for a
    group claim (between-animal variability), NOT the within-unit / trial-level
    interval stored in ``ci_*_within``.

    Args:
        values:    1D per-unit values (e.g. one Δ per animal). NaNs dropped.
        n_boot:    bootstrap resamples.
        ci:        central interval width (0.95 → 2.5/97.5 percentiles).
        statistic: summary applied to each resample (default mean).
        seed:      RNG seed.

    Returns:
        ``{'point', 'lo', 'hi', 'n'}``. ``lo``/``hi`` are NaN when n<2.
    """
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return {'point': np.nan, 'lo': np.nan, 'hi': np.nan, 'n': 0}
    point = float(statistic(v))
    if n < 2:
        warnings.warn("bootstrap_units: n<2; CI undefined", stacklevel=2)
        return {'point': point, 'lo': np.nan, 'hi': np.nan, 'n': n}

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = statistic(v[rng.integers(0, n, n)])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return {'point': point, 'lo': float(lo), 'hi': float(hi), 'n': n}


# ─────────────────────────────────────────────────────────────────────────────
# Group test — two per-unit arrays → statistic + p
# ─────────────────────────────────────────────────────────────────────────────
def rank_test(
    a: Iterable[float],
    b: Iterable[float],
    *,
    paired: bool = False,
    alternative: str = 'two-sided',
) -> Dict[str, Union[float, str, bool, int]]:
    """Rank test between two per-unit value arrays.

    Paired → Wilcoxon signed-rank (requires aligned, equal-length arrays; align
    with :func:`paired_diff` first). Unpaired → Mann-Whitney U. Plain arrays in,
    not frames — the project's within/between/interaction opto tests are all
    expressed through this one function plus :func:`paired_diff`.

    Power is limited at small n; with 6 paired animals a two-sided Wilcoxon p
    cannot fall below ~0.031. Treat as directional evidence.

    Returns:
        ``{'test', 'statistic', 'p', 'paired', 'alternative', ...}`` with ``n``
        (paired) or ``n_a``/``n_b`` (unpaired). Degenerate cases return NaN p.

    Raises:
        ValueError: paired with mismatched lengths.
    """
    from scipy.stats import mannwhitneyu, wilcoxon

    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)

    if paired:
        if a.shape != b.shape:
            raise ValueError(
                f"rank_test(paired=True) needs aligned equal-length arrays "
                f"(got {a.shape} vs {b.shape}); align with paired_diff first"
            )
        m = ~(np.isnan(a) | np.isnan(b))
        a, b = a[m], b[m]
        n = int(a.size)
        base = {'test': 'wilcoxon', 'paired': True, 'alternative': alternative, 'n': n}
        if n < 1 or np.all(a - b == 0):
            return {**base, 'statistic': np.nan, 'p': np.nan}
        try:
            stat, p = wilcoxon(a, b, alternative=alternative)
        except ValueError:
            stat, p = np.nan, np.nan
        return {**base, 'statistic': float(stat) if stat == stat else np.nan,
                'p': float(p) if p == p else np.nan}

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    base = {'test': 'mannwhitneyu', 'paired': False, 'alternative': alternative,
            'n_a': int(a.size), 'n_b': int(b.size)}
    if a.size < 1 or b.size < 1:
        return {**base, 'statistic': np.nan, 'p': np.nan}
    stat, p = mannwhitneyu(a, b, alternative=alternative)
    return {**base, 'statistic': float(stat), 'p': float(p)}


# ─────────────────────────────────────────────────────────────────────────────
# Average arrays — one element-wise averager for scalars, curves and matrices
# ─────────────────────────────────────────────────────────────────────────────
def average_arrays(arrays: Sequence[np.ndarray], *, ddof: int = 1) -> Dict[str, np.ndarray]:
    """Element-wise mean and SEM across a stack of same-shaped arrays.

    Shape-agnostic: scalars, 1D psychometric curves and 2D update matrices all go
    through here — the caller picks which field to stack (``y_fit`` for a curve
    band, the params vector, or the UM grid). Averaging curves point-wise and
    averaging the fitted params give different mean curves; choose deliberately.

    Element ``[i]`` of every input must correspond (same x-grid for curves, same
    bin layout for UMs). Mismatched shapes raise rather than broadcast.

    Args:
        arrays: list of ndarrays, all the same shape.
        ddof:   delta-DOF for the across-stack std (1 = sample SEM).

    Returns:
        ``{'mean', 'sem', 'n'}``; ``sem`` is all-NaN when n<2.

    Raises:
        ValueError: empty input, or shapes that do not match.
    """
    arrs = list(arrays)
    if len(arrs) == 0:
        raise ValueError("average_arrays: empty input")
    try:
        stack = np.stack([np.asarray(x, dtype=float) for x in arrs])
    except ValueError as e:
        shapes = [tuple(np.asarray(x).shape) for x in arrs]
        raise ValueError(
            f"average_arrays: inputs must share a shape (got {shapes}); for curves "
            f"this usually means animals were fit on different x-grids"
        ) from e

    n = stack.shape[0]
    mean = np.nanmean(stack, axis=0)
    if n < 2:
        warnings.warn("average_arrays: n<2; SEM undefined", stacklevel=2)
        sem = np.full_like(mean, np.nan, dtype=float)
    else:
        sem = np.nanstd(stack, axis=0, ddof=ddof) / np.sqrt(n)
    return {'mean': mean, 'sem': sem, 'n': n}


def min_achievable_p(test, *, n=None, n1=None, n2=None):
    """Smallest two-sided p-value the given rank test can return at this sample size.

    Rank tests are discrete: with few observations even a perfectly consistent effect cannot
    drop below a floor set by the number of possible rank arrangements. Report this beside a
    non-significant result to show whether the test *could* have rejected at all — a p that
    merely equals this floor is the strongest evidence the sample can give, not a meaningful
    null.

    Args:
        test: 'signed_rank' (Wilcoxon signed-rank / paired) — needs ``n``, the number of
              non-zero pairs; floor = 2 / 2**n (all pairs the same sign).
              'rank_sum'  (Mann-Whitney U / unpaired) — needs ``n1``, ``n2``;
              floor = 2 / C(n1 + n2, n1) (the most extreme rank separation).
        n:      non-zero pair count for 'signed_rank'.
        n1, n2: group sizes for 'rank_sum'.

    Returns:
        Minimum two-sided p in (0, 1], or nan if sizes are missing / too small.

    Assumes the exact (permutation) null and no ties; scipy uses the exact distribution at
    these small n, so the floor matches what :func:`rank_test` can actually return.
    """
    from math import comb
    t = str(test).lower()
    if t in ('signed_rank', 'wilcoxon', 'paired'):
        if not n or n < 1:
            return float('nan')
        return min(1.0, 2.0 / (2 ** n))
    if t in ('rank_sum', 'mannwhitney', 'mannwhitneyu', 'mwu', 'unpaired'):
        if not n1 or not n2 or n1 < 1 or n2 < 1:
            return float('nan')
        return min(1.0, 2.0 / comb(n1 + n2, n1))
    raise ValueError(f"test must be 'signed_rank' or 'rank_sum', got {test!r}")
