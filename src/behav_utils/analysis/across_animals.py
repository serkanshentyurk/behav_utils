"""
Across-animal comparison — the group fold.

Everything upstream runs on one animal and emits the same tidy per-animal row:
``{animal, group, ..., stat, value}``. This module folds those rows: for each
stat, take one value per animal, split into two groups, and rank-test. That is
the ONLY level that supports a population claim — a within-animal p describes
that animal, not the group.

Grouping is arbitrary: genotype, cohort, sex, lesion, or an ad-hoc pair of
animal-name lists. Pass ``groups=(a, b)`` to fix which group is the reference;
without it the two labels are taken in sorted order.

    from behav_utils.analysis import collect_rows, compare_groups

    rows = []
    for animal in animals:
        r = compute_phase_stats(phase, ['pse']).to_rows()
        rows += collect_rows(r.to_dict('records'), animal=animal.animal_id,
                             group=animal.metadata['group'])

    # group by the stamped column, control first
    result = compare_groups(rows, groups=('control', 'lesion'))
    # or split by explicit animal-name lists, no group column needed
    result = compare_groups(rows, groups={'lesion': ['A01', 'A02'],
                                          'sham':   ['A03', 'A04']})
    result['pse_plateau']['p'], result['pse_plateau']['min_p']

The unit is the ANIMAL, never the session or trial — pooling those into the
test is pseudoreplication, inflating n above the number of animals, which is the
only n that varies with group. At 4 vs 5 animals the smallest two-sided
Mann-Whitney p is ~0.016; ``min_p`` is reported on every result so an
underpowered null is not read as a real one.

Public API:
    collect_rows      — stamp per-animal scalars with animal, group and labels
    compare_groups    — rank-test each stat across animals, split by group
"""

from typing import Dict, List, Mapping, Sequence

import numpy as np

__all__ = ['collect_rows', 'compare_groups']


def collect_rows(
    scalars: Sequence[Mapping],
    animal: str,
    group: str | None = None,
    group_col: str = 'group',
    **labels,
) -> List[Dict]:
    """Stamp an analysis's per-animal scalars with animal, group and labels.

    Turns the ``scalars`` list an analysis emits (each ``{stat, value, ...}``)
    into fully-labelled rows ready to accumulate across animals.

    Args:
        scalars:   the ``scalars`` field of a compute_* result — each a mapping
                   with at least ``stat`` and ``value`` (extra keys carried
                   through).
        animal:    animal id.
        group:     the group this animal belongs to (cohort, condition, ...).
                   Optional: omit if you will split by explicit name-lists in
                   ``compare_groups`` instead.
        group_col: the column name to store ``group`` under. Default 'group'.
        **labels:  any extra columns to stamp on every row (session_type,
                   distribution, contrast, ...).

    Returns:
        A list of dicts, one per scalar, each carrying animal, the group column
        (when ``group`` is given), the labels, and the scalar's own fields.
    """
    rows = []
    for entry in scalars:
        row = {'animal': animal, **labels}
        if group is not None:
            row[group_col] = group
        row.update(dict(entry))
        rows.append(row)
    return rows


def compare_groups(
    rows,
    group_col: str = 'group',
    groups=None,
    stats: Sequence[str] | None = None,
    value_col: str = 'value',
    stat_col: str = 'stat',
    unit_col: str = 'animal',
    paired: bool = False,
    alternative: str = 'two-sided',
) -> Dict:
    """Rank-test each stat across animals, split into two groups.

    One test per stat: one value per animal, split into the two groups,
    rank-test. Animals with a non-finite value for a stat are dropped from that
    stat's test only.

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
        value_col, stat_col, unit_col: column names.
        paired:      Wilcoxon signed-rank instead of Mann-Whitney. Requires the
                     same animals in both groups aligned by ``unit_col`` — for a
                     within-animal design only, NOT a between-group split, so
                     the default is unpaired.
        alternative: 'two-sided' | 'less' | 'greater'.

    Returns:
        ``{stat: {test, statistic, p, min_p, n_a, n_b, median_a, median_b,
        group_a, group_b, ...}}``. ``min_p`` is the smallest p obtainable at
        these group sizes — compare your p against it before reading a null as
        no effect.

    Raises:
        ValueError: on no rows, fewer than two groups, or a name-list mapping
            without exactly two groups.
    """
    import pandas as pd

    from behav_utils.analysis.group import min_achievable_p, rank_test

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

    out: Dict[str, Dict] = {}
    for stat in stats:
        sub = df[df[stat_col] == stat].copy()
        sub['_grp'] = sub.apply(_group_of, axis=1)

        def _values(label, sub=sub):
            rows_g = sub[sub['_grp'] == label]
            per_unit = rows_g.groupby(unit_col)[value_col].mean()
            v = per_unit.to_numpy(dtype=float)
            return v[np.isfinite(v)]

        a = _values(group_a)
        b = _values(group_b)
        if a.size < 1 or b.size < 1:
            out[stat] = {'test': None, 'p': np.nan, 'min_p': np.nan,
                         'n_a': int(a.size), 'n_b': int(b.size),
                         'group_a': group_a, 'group_b': group_b,
                         'median_a': float(np.median(a)) if a.size else np.nan,
                         'median_b': float(np.median(b)) if b.size else np.nan,
                         'note': 'too few animals with a finite value'}
            continue

        test = rank_test(a, b, paired=paired, alternative=alternative)
        try:
            floor = (min_achievable_p('signed_rank', n=a.size) if paired
                     else min_achievable_p('rank_sum', n1=a.size, n2=b.size))
        except Exception:
            floor = np.nan

        out[stat] = {
            'test': test.get('test'),
            'statistic': test.get('statistic'),
            'p': test.get('p'),
            'min_p': floor,
            'n_a': int(a.size), 'n_b': int(b.size),
            'group_a': group_a, 'group_b': group_b,
            'median_a': float(np.median(a)), 'median_b': float(np.median(b)),
            'alternative': alternative, 'paired': paired,
        }
    return out

