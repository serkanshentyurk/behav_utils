"""Tests for behav_utils.analysis.group (Tier B: combine / paired_diff /
bootstrap_units / rank_test / average_arrays).

These operate on plain tidy tables and arrays — no SessionData — so they build
small DataFrames inline rather than using the session fixtures. Covers the
collapse/contrast/test verbs, the chaining-idempotency regression (a second
combine must not duplicate n_units), and every guard (raise vs warn).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from behav_utils.analysis.group import (
    average_arrays,
    bootstrap_units,
    combine,
    paired_diff,
    rank_test,
)


def _table():
    """3 het + 2 wt animals x {opto, opto_off} x 3 sessions, one stat."""
    rows = []
    rng = np.random.default_rng(0)
    for gen, animals in [('het', ['A', 'B', 'C']), ('wt', ['D', 'E'])]:
        for animal in animals:
            for cond in ['opto', 'opto_off']:
                for sidx in range(3):
                    rows.append(dict(
                        animal=animal, session=sidx, stat='recency',
                        value=float(rng.normal(0.2 if cond == 'opto' else 0.3, 0.05)),
                        n_trials=int(rng.integers(80, 200)),
                        ci_lo_within=np.nan, ci_hi_within=np.nan,
                        genotype=gen, trial_type=cond,
                    ))
    return pd.DataFrame(rows)


class TestCombine:

    def test_collapses_sessions(self):
        out = combine(_table(), over='session')
        assert 'session' not in out.columns
        assert len(out) == 5 * 2                      # 5 animals x 2 conditions
        assert (out['n_units'] == 3).all()            # 3 sessions each

    def test_collapses_animals_to_group(self):
        per_animal = combine(_table(), over='session')
        per_geno = combine(per_animal, over='animal')
        assert len(per_geno) == 2 * 2                 # 2 genotypes x 2 conditions
        assert set(per_geno['n_units']) == {3, 2}     # het=3 animals, wt=2

    def test_chaining_does_not_duplicate_n_units(self):
        # Regression: n_units must be a non-key, or the second combine treats the
        # first's n_units as a grouping column AND re-adds it.
        per_animal = combine(_table(), over='session')
        per_geno = combine(per_animal, over='animal')
        assert list(per_geno.columns).count('n_units') == 1
        assert isinstance(per_geno['n_units'], pd.Series)

    def test_weighted_mean_runs_and_is_finite(self):
        out = combine(_table(), over='session', how='mean', weight='n_trials')
        assert np.isfinite(out['value']).all()

    def test_median_runs(self):
        out = combine(_table(), over='session', how='median')
        assert np.isfinite(out['value']).all()

    def test_drops_within_ci_columns(self):
        out = combine(_table(), over='session')
        assert 'ci_lo_within' not in out.columns
        assert 'ci_hi_within' not in out.columns

    def test_missing_axis_raises(self):
        out = combine(_table(), over='session')          # session now gone
        with pytest.raises(ValueError):
            combine(out, over='session')

    def test_collapsed_axis_raises(self):
        with pytest.raises(ValueError):
            combine(_table().assign(session=pd.NA), over='session')

    def test_bad_how_raises(self):
        with pytest.raises(ValueError):
            combine(_table(), over='session', how='geometric')


class TestPairedDiff:

    def test_delta_per_unit(self):
        per_animal = combine(_table(), over='session')
        delta = paired_diff(per_animal, by='trial_type', a='opto', b='opto_off')
        assert {'opto', 'opto_off', 'delta'} <= set(delta.columns)
        assert len(delta) == 5                          # one Δ per animal
        np.testing.assert_allclose(
            delta['delta'].to_numpy(),
            delta['opto'].to_numpy() - delta['opto_off'].to_numpy(),
        )

    def test_drop_warns(self):
        per_animal = combine(_table(), over='session')
        dropped = per_animal[~((per_animal['animal'] == 'A') &
                               (per_animal['trial_type'] == 'opto'))]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = paired_diff(dropped, by='trial_type', a='opto', b='opto_off')
        assert any('paired_diff' in str(w.message) for w in caught)
        assert len(out) == 4                            # A dropped

    def test_missing_level_raises(self):
        per_animal = combine(_table(), over='session')
        with pytest.raises(ValueError):
            paired_diff(per_animal, by='trial_type', a='opto', b='nonexistent')

    def test_missing_column_raises(self):
        per_animal = combine(_table(), over='session')
        with pytest.raises(ValueError):
            paired_diff(per_animal, by='not_a_column', a='opto', b='opto_off')

    def test_chains_for_difference_of_differences(self):
        # The Δ table (value column 'delta') fed back in for a second difference —
        # the interaction is paired_diff twice. Regression: the value column must
        # be excluded from the pivot index, else this silently returns empty.
        d = pd.DataFrame([
            dict(animal='A', genotype='het', stat='recency', phase='uniform', delta=0.10),
            dict(animal='A', genotype='het', stat='recency', phase='hard_a', delta=0.35),
            dict(animal='B', genotype='wt',  stat='recency', phase='uniform', delta=0.20),
            dict(animal='B', genotype='wt',  stat='recency', phase='hard_a', delta=0.22),
        ])
        out = paired_diff(d, by='phase', a='hard_a', b='uniform', value='delta')
        assert len(out) == 2
        np.testing.assert_allclose(
            out['delta'].to_numpy(), (out['hard_a'] - out['uniform']).to_numpy())


class TestBootstrapUnits:

    def test_structure(self):
        res = bootstrap_units([0.1, 0.2, 0.3, 0.4, 0.5], n_boot=500, seed=1)
        assert res['n'] == 5
        assert res['lo'] <= res['point'] <= res['hi']

    def test_drops_nan(self):
        res = bootstrap_units([0.1, np.nan, 0.3], n_boot=200, seed=1)
        assert res['n'] == 2

    def test_small_n_warns_and_nan_ci(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = bootstrap_units([0.5], n_boot=100)
        assert res['n'] == 1
        assert np.isnan(res['lo']) and np.isnan(res['hi'])
        assert any('n<2' in str(w.message) for w in caught)

    def test_empty(self):
        res = bootstrap_units([], n_boot=100)
        assert res['n'] == 0 and np.isnan(res['point'])

    def test_reproducible(self):
        a = bootstrap_units([0.1, 0.2, 0.3, 0.4], seed=7)
        b = bootstrap_units([0.1, 0.2, 0.3, 0.4], seed=7)
        assert a == b


class TestRankTest:

    def test_paired_wilcoxon(self):
        res = rank_test([0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.2, 0.3], paired=True)
        assert res['test'] == 'wilcoxon' and res['paired'] is True
        assert 0.0 <= res['p'] <= 1.0 and res['n'] == 4

    def test_unpaired_mannwhitney(self):
        res = rank_test([0.3, 0.4, 0.5], [0.1, 0.2], paired=False)
        assert res['test'] == 'mannwhitneyu' and res['paired'] is False
        assert res['n_a'] == 3 and res['n_b'] == 2 and 0.0 <= res['p'] <= 1.0

    def test_paired_misaligned_raises(self):
        with pytest.raises(ValueError):
            rank_test([1, 2, 3], [1, 2], paired=True)

    def test_paired_all_zero_diff_is_nan(self):
        res = rank_test([0.2, 0.3], [0.2, 0.3], paired=True)
        assert np.isnan(res['p'])

    def test_paired_drops_nan_pairs(self):
        res = rank_test([0.3, np.nan, 0.5, 0.6], [0.1, 0.2, 0.2, 0.3], paired=True)
        assert res['n'] == 3


class TestAverageArrays:

    def test_curve(self):
        res = average_arrays([np.random.rand(8) for _ in range(4)])
        assert res['mean'].shape == (8,) and res['sem'].shape == (8,) and res['n'] == 4

    def test_update_matrix(self):
        res = average_arrays([np.random.rand(8, 8) for _ in range(3)])
        assert res['mean'].shape == (8, 8) and res['n'] == 3

    def test_mean_correct(self):
        res = average_arrays([np.zeros(4), np.ones(4) * 2])
        np.testing.assert_allclose(res['mean'], np.ones(4))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            average_arrays([np.zeros(8), np.zeros(7)])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            average_arrays([])

    def test_single_warns_and_nan_sem(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = average_arrays([np.ones(4)])
        assert res['n'] == 1 and np.isnan(res['sem']).all()
        assert any('n<2' in str(w.message) for w in caught)
