"""Group-level combination, resampling and testing — tidy tables and arrays in, typed results out.

Every function takes a tidy stat table (``compute_phase_stats(...).to_rows()`` or
``.sessions``, ``Contrast.table()``, accumulated with :func:`collect_rows`) or
plain arrays, never ``SessionData`` / ``AnimalData``. Each verb does one job, so
combining / bootstrapping / testing compose freely:

    rows = []
    for animal in animals:
        r = compute_phase_stats(phase_of(animal), ['mu', 'accuracy'])
        rows += collect_rows(r.to_rows().to_dict('records'), animal=animal.animal_id, group=...)
    compare_groups(rows, groups=('control', 'treated'))   # one rank test per stat
    combine(points, over='session')                        # sessions → one value per animal
    paired_diff(points, by='condition', a='on', b='off')   # per-animal Δ
    bootstrap_units(delta_values)                          # across-animal CI on the Δ
    rank_test(delta_treated, delta_control)                # group difference test
    average_arrays([um for animal in ...])                 # element-wise mean curve / UM

Two resampling levels exist and must not be confused: the within-unit
(trial- or session-level) draws that ``compute_delta_stat`` stores answer "how
precise is THIS animal's estimate"; the across-unit bootstrap here resamples
animals and answers "how much does the effect vary across animals". A group
claim needs the second.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import comb
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd

_NON_KEY = {'value', 'n_trials', 'ci_lo_within', 'ci_hi_within', 'n_units'}


__all__ = ['combine', 'paired_diff', 'bootstrap_units', 'rank_test', 'average_arrays', 'min_achievable_p',
           'collect_rows', 'compare_groups',
           'UnitBootstrap', 'RankTest', 'ArrayAverage', 'GroupComparison']


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UnitBootstrap:
    """Across-unit bootstrap of one per-unit value array (``bootstrap_units``)."""
    point: float
    lo: float
    hi: float
    n: int

    def to_rows(self) -> pd.DataFrame:
        return pd.DataFrame([{'point': self.point, 'lo': self.lo, 'hi': self.hi, 'n': self.n}])


@dataclass(frozen=True)
class RankTest:
    """One rank test (``rank_test``): Wilcoxon signed-rank (paired) or Mann-Whitney U.

    ``n`` is the number of pairs used (paired) or ``n_a + n_b`` (unpaired);
    ``min_p`` the smallest two-sided p this test could return at that size.
    """
    test: str
    paired: bool
    alternative: str
    statistic: float
    p: float
    n_a: int
    n_b: int
    min_p: float
    group_a: object = None
    group_b: object = None
    median_a: float = np.nan
    median_b: float = np.nan
    note: str = ''

    @property
    def n(self) -> int:
        return self.n_a if self.paired else self.n_a + self.n_b

    def to_rows(self) -> pd.DataFrame:
        return pd.DataFrame([{f: getattr(self, f) for f in self.__dataclass_fields__}])

    def __repr__(self) -> str:
        return (f'RankTest({self.test}, n_a={self.n_a}, n_b={self.n_b}, '
                f'statistic={self.statistic:.3g}, p={self.p:.3g}, min_p={self.min_p:.3g})')


@dataclass(frozen=True)
class ArrayAverage:
    """Element-wise mean and SEM across a stack (``average_arrays``)."""
    mean: np.ndarray
    sem: np.ndarray
    n: int


@dataclass(frozen=True)
class GroupComparison:
    """One :class:`RankTest` per stat between two groups of animals (``compare_groups``)."""
    tests: Dict[str, RankTest] = field(default_factory=dict)
    group_a: object = None
    group_b: object = None
    paired: bool = False

    @property
    def stats(self) -> List[str]:
        return list(self.tests)

    def test(self, stat: str) -> RankTest:
        return self.tests[stat]

    def table(self) -> pd.DataFrame:
        """One tidy row per stat: ``stat, group_a, group_b, n_a, n_b, median_a, median_b, statistic, p, min_p, note``."""
        cols = ['stat', 'group_a', 'group_b', 'n_a', 'n_b', 'median_a', 'median_b', 'statistic', 'p', 'min_p', 'note']
        rows = [{'stat': k, **{c: getattr(t, c) for c in cols[1:]}} for k, t in self.tests.items()]
        return pd.DataFrame(rows, columns=cols)

    to_rows = table

    def __repr__(self) -> str:
        return (f'GroupComparison({self.group_a!r} vs {self.group_b!r}, paired={self.paired}, '
                f'stats={self.stats})')

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
    # dropna=False: label columns that are legitimately empty on pooled rows
    # (``distribution`` None, ``session_type`` '') must not drop the unit.
    wide = sub.pivot_table(index=idx, columns=by, values=value, aggfunc='mean', dropna=False)
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
) -> UnitBootstrap:
    """Across-unit (cluster) bootstrap CI on a per-unit value array.

    Resamples the units (animals) with replacement — this is the interval for a
    group claim (between-animal variability), NOT the within-unit interval
    ``compute_delta_stat`` stores per animal.

    Args:
        values:    1D per-unit values (e.g. one Δ per animal). NaNs dropped.
        n_boot:    bootstrap resamples.
        ci:        central interval width (0.95 → 2.5/97.5 percentiles).
        statistic: summary applied to each resample (default mean).
        seed:      RNG seed.

    Returns:
        :class:`UnitBootstrap`; ``lo``/``hi`` are NaN when n<2.
    """
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    n = int(v.size)
    if n == 0:
        return UnitBootstrap(np.nan, np.nan, np.nan, 0)
    point = float(statistic(v))
    if n < 2:
        warnings.warn("bootstrap_units: n<2; CI undefined", stacklevel=2)
        return UnitBootstrap(point, np.nan, np.nan, n)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = statistic(v[rng.integers(0, n, n)])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return UnitBootstrap(point, float(lo), float(hi), n)


# ─────────────────────────────────────────────────────────────────────────────
# Group test — two per-unit arrays → statistic + p
# ─────────────────────────────────────────────────────────────────────────────
def rank_test(
    a: Iterable[float],
    b: Iterable[float],
    *,
    paired: bool = False,
    alternative: str = 'two-sided',
) -> RankTest:
    """Rank test between two per-unit value arrays.

    Paired → Wilcoxon signed-rank (requires aligned, equal-length arrays; align
    with :func:`paired_diff` first). Unpaired → Mann-Whitney U. Plain arrays in,
    not frames — the project's within/between/interaction opto tests are all
    expressed through this one function plus :func:`paired_diff`.

    Power is limited at small n; with 6 paired animals a two-sided Wilcoxon p
    cannot fall below ~0.031 — ``min_p`` on the result says what the floor is.

    Returns:
        :class:`RankTest`. Degenerate cases (no data, all-zero paired
        differences) have NaN ``p`` and a ``note``.

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
        med_a = float(np.median(a)) if n else np.nan
        med_b = float(np.median(b)) if n else np.nan
        floor = min_achievable_p('signed_rank', n=n)
        if n < 1 or np.all(a - b == 0):
            return RankTest('wilcoxon', True, alternative, np.nan, np.nan, n, n, floor,
                            median_a=med_a, median_b=med_b, note='no non-zero pairs')
        try:
            stat, p = wilcoxon(a, b, alternative=alternative)
            stat, p = float(stat), float(p)
        except ValueError as e:
            return RankTest('wilcoxon', True, alternative, np.nan, np.nan, n, n, floor,
                            median_a=med_a, median_b=med_b, note=str(e))
        return RankTest('wilcoxon', True, alternative, stat, p, n, n, floor,
                        median_a=med_a, median_b=med_b)

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a, n_b = int(a.size), int(b.size)
    med_a = float(np.median(a)) if n_a else np.nan
    med_b = float(np.median(b)) if n_b else np.nan
    floor = min_achievable_p('rank_sum', n1=n_a, n2=n_b)
    if n_a < 1 or n_b < 1:
        return RankTest('mannwhitneyu', False, alternative, np.nan, np.nan, n_a, n_b, floor,
                        median_a=med_a, median_b=med_b, note='a group is empty')
    stat, p = mannwhitneyu(a, b, alternative=alternative)
    return RankTest('mannwhitneyu', False, alternative, float(stat), float(p), n_a, n_b, floor,
                    median_a=med_a, median_b=med_b)


# ─────────────────────────────────────────────────────────────────────────────
# Average arrays — one element-wise averager for scalars, curves and matrices
# ─────────────────────────────────────────────────────────────────────────────
def average_arrays(arrays: Sequence[np.ndarray], *, ddof: int = 1) -> ArrayAverage:
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
        :class:`ArrayAverage`; ``sem`` is all-NaN when n<2.

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
    return ArrayAverage(mean, sem, int(n))


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


def collect_rows(
    scalars: Sequence[Mapping],
    animal: str,
    group: str | None = None,
    group_col: str = 'group',
    **labels,
) -> List[Dict]:
    """Stamp per-animal scalar rows with the animal, its group and any labels.

    Turns the rows an analysis emits for one animal — ``PhaseStats.to_rows()``
    / ``.sessions``, ``Contrast.table()``, as ``to_dict('records')`` — into
    fully-labelled rows ready to accumulate across animals.

    ``animal`` always wins: a row's own ``animal`` key is overwritten (a warning
    is issued if it held a different real id), so a frame built without the id
    can never collapse several animals into one unit.

    Args:
        scalars:   mappings with at least ``stat`` and ``value`` (extra keys
                   carried through).
        animal:    animal id.
        group:     the group this animal belongs to (cohort, condition, ...).
                   Optional: omit if you will split by explicit name-lists in
                   ``compare_groups`` instead.
        group_col: the column name to store ``group`` under. Default 'group'.
        **labels:  any extra columns to stamp on every row (session_type,
                   distribution, contrast, ...).

    Returns:
        A list of dicts, one per scalar.
    """
    rows = []
    for entry in scalars:
        entry = dict(entry)
        own = entry.get('animal')
        if own not in (None, '', 'unknown', animal):
            warnings.warn(f"collect_rows: row carries animal={own!r}, stamping {animal!r} over it",
                          stacklevel=2)
        row = {**entry, 'animal': animal, **labels}
        if group is not None:
            row[group_col] = group
        rows.append(row)
    return rows


def compare_groups(
    rows: pd.DataFrame | Sequence[Mapping[str, Any]],
    group_col: str = 'group',
    groups: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    stats: Sequence[str] | None = None,
    value_col: str = 'value',
    stat_col: str = 'stat',
    unit_col: str = 'animal',
    paired: bool = False,
    alternative: str = 'two-sided',
) -> GroupComparison:
    """Rank-test each stat across animals, split into two groups.

    One test per stat: one value per animal (rows of the same animal within a
    group are averaged), split into the two groups, rank-test. Animals with a
    non-finite value for a stat are dropped from that stat's test only.

    Grouping is specified one of two ways:

    * by column — ``group_col`` names a column already on the rows (the default,
      'group');
    * by name-lists — ``groups={label: [animal, ...]}`` assigns each animal to a
      group explicitly, needing no group column. Use this for an ad-hoc split.

    Args:
        rows:        list of dicts, or a DataFrame — accumulated ``collect_rows``
                     output across animals.
        group_col:   column holding the grouping (ignored if ``groups`` given).
        groups:      either the two group labels in (a, b) order (selecting from
                     ``group_col``), or a ``{label: [animal_ids]}`` mapping that
                     defines the split directly. If None, the two labels present
                     in ``group_col`` are used in sorted order — pass the pair
                     explicitly whenever the reference group matters.
        stats:       which stats to test; default is every stat present.
        value_col:   The column holding the values.
        stat_col:    The column holding each value's statistic name.
        unit_col:    The column identifying the unit, usually the animal, a value belongs to.
        paired:      Wilcoxon signed-rank instead of Mann-Whitney. The two
                     "groups" are then two conditions of the *same* animals: the
                     values are aligned by ``unit_col`` and an animal missing a
                     finite value on either side is dropped from that pair.
        alternative: 'two-sided' | 'less' | 'greater'.

    Returns:
        :class:`GroupComparison` — ``.test(stat)`` is a :class:`RankTest` whose
        ``min_p`` is the smallest p obtainable at these group sizes; compare a p
        against it before reading a null as no effect. ``.table()`` is tidy.

    Raises:
        ValueError: on no rows, fewer than two groups, a name-list mapping
            without exactly two groups, or ``paired=True`` with disjoint animals.
    """
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if df.empty:
        raise ValueError("compare_groups: no rows")

    # ── resolve the two groups + a per-animal → group map ─────────────────
    name_to_group = None
    if isinstance(groups, Mapping):
        if len(groups) != 2:
            raise ValueError(
                f"compare_groups: groups mapping must have exactly two entries, "
                f"got {list(groups)}")
        group_a, group_b = list(groups)
        name_to_group = {}
        for label, ids in groups.items():
            for aid in ids:
                name_to_group[aid] = label
    else:
        if group_col not in df.columns:
            raise ValueError(
                f"compare_groups: no column {group_col!r}; pass groups=... to "
                f"split by animal-name lists instead")
        present = list(pd.unique(df[group_col]))
        selected = list(groups) if groups is not None else None
        if selected is None:
            selected = sorted(present, key=str)
        selected = [g for g in selected if g in present]
        if len(selected) < 2:
            raise ValueError(
                f"compare_groups: need two groups in {group_col!r}, found {present}")
        group_a, group_b = selected[0], selected[1]

    if stats is None:
        stats = list(pd.unique(df[stat_col]))

    def _group_of(row):
        if name_to_group is not None:
            return name_to_group.get(row[unit_col])
        return row[group_col]

    tests: Dict[str, RankTest] = {}
    for stat in stats:
        sub = df[df[stat_col] == stat].copy()
        sub['_grp'] = sub.apply(_group_of, axis=1)

        def _per_unit(label, sub=sub) -> pd.Series:
            rows_g = sub[sub['_grp'] == label]
            return rows_g.groupby(unit_col)[value_col].mean().astype(float)

        pa, pb = _per_unit(group_a), _per_unit(group_b)
        if paired:
            common = pa.index.intersection(pb.index)
            if len(common) == 0:
                raise ValueError(
                    f"compare_groups(paired=True): {group_a!r} and {group_b!r} share no "
                    f"{unit_col}; a paired test needs the same animals on both sides")
            a = pa.reindex(common).to_numpy()
            b = pb.reindex(common).to_numpy()      # rank_test drops NaN pairs, aligned
        else:
            a = pa.to_numpy()
            a = a[np.isfinite(a)]
            b = pb.to_numpy()
            b = b[np.isfinite(b)]

        t = rank_test(a, b, paired=paired, alternative=alternative)
        tests[stat] = RankTest(t.test, t.paired, t.alternative, t.statistic, t.p, t.n_a, t.n_b,
                               t.min_p, group_a, group_b, t.median_a, t.median_b, t.note)
    return GroupComparison(tests, group_a, group_b, paired)
