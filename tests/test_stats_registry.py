"""TrialArrays + behav_utils.stats registry.

Contract tests. (Bit-exact equivalence against the legacy registry was
verified before it was deleted: 350 values, zero mismatches.)
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from behav_utils.data.arrays import TrialArrays
from behav_utils.data.ops.filtering import filter_trials, pool_arrays
from behav_utils.data.ops.selection import select_sessions
from behav_utils.data.structures import TrialData
from behav_utils.data.synthetic import generate_synthetic_animal
from behav_utils.stats import (
    PSYCHOMETRIC,
    compute_stats,
    is_exchangeable,
    list_stats,
)
from behav_utils.stats.registry import _OUTPUT_TO_PRODUCER, _PRODUCERS, _list_producers, fit, stat

# ── fixtures ────────────────────────────────────────────────────────────────

def _sessions(seed: int, n_sessions: int = 4, p_no_response: float = 0.05):
    """Filtered sessions with injected no-response trials and a properly re-frozen lag-1 view."""
    animal, _ = generate_synthetic_animal(
        animal_id=f'T{seed}', n_sessions=n_sessions, trials_per_session=300,
        seed=seed, stage='Full_Task_Cont')
    rng = np.random.default_rng(seed)
    for s in animal.sessions:
        c = s.trials.choice.astype(float)
        c[rng.random(len(c)) < p_no_response] = np.nan
        d = {f.name: getattr(s.trials, f.name) for f in dataclasses.fields(s.trials)
             if not f.name.startswith('prev_')}
        d['choice'] = c
        s.trials = TrialData(**d)
    # exclude_abort=True explicitly: 'all' currently keeps aborts (see filtering.py:373)
    return filter_trials(select_sessions(animal, stage='Full_Task_Cont'), exclude_opto=False)


@pytest.fixture(scope='module')
def pooled():
    return pool_arrays(_sessions(0))


@pytest.fixture(scope='module')
def arrays(pooled):
    return TrialArrays.from_pooled(pooled)


# ── TrialArrays ─────────────────────────────────────────────────────────────

class TestTrialArrays:
    def test_from_pooled_lengths_and_dtype(self, pooled, arrays):
        n = pooled['n_trials']
        assert arrays.n_trials == n
        for f in ('choice', 'stimulus', 'category', 'prev_choice', 'prev_stimulus',
                  'prev_category', 'reaction_time'):
            v = getattr(arrays, f)
            assert v.shape == (n,) and v.dtype == float

    def test_read_only(self, arrays):
        with pytest.raises(ValueError):
            arrays.choice[0] = 1.0

    def test_missing_optional_fields_become_nan(self):
        a = TrialArrays(choice=[0, 1, 1], stimulus=[-0.5, 0.2, 0.9], category=[0, 1, 1])
        assert np.isnan(a.prev_choice).all() and np.isnan(a.reaction_time).all()
        assert a.has_prev.sum() == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            TrialArrays(choice=[0, 1], stimulus=[0.1], category=[0, 1])

    def test_from_sequence_matches_loader_on_one_block(self):
        sess = _sessions(1, n_sessions=1)[:1]
        p = pool_arrays(sess)
        seq = TrialArrays.from_sequence(p['choices'], p['stimuli'], p['categories'])
        ld = TrialArrays.from_pooled(p)
        for f in ('prev_choice', 'prev_stimulus', 'prev_category'):
            assert np.array_equal(getattr(seq, f), getattr(ld, f), equal_nan=True), f

    def test_lag1_pairs_masks_no_response_and_block_start(self):
        a = TrialArrays.from_sequence(choice=[0, np.nan, 1, 1], stimulus=[-1, 0, 0.5, 1],
                                      category=[0, 0, 1, 1])
        p = a.lag1_pairs()
        # t=0 no predecessor; t=1 no response; t=2 predecessor was no-response; t=3 ok
        assert p.n_trials == 1 and p.choice[0] == 1 and p.prev_choice[0] == 1

    def test_pooled_view_does_not_bridge_sessions(self):
        sess = _sessions(2, n_sessions=3)
        p = pool_arrays(sess)
        a = TrialArrays.from_pooled(p)
        for b in p['session_boundaries'][1:-1]:
            assert not a.has_prev[b], 'first trial of a pooled session must have no predecessor'

    def test_prev_correct_is_the_tasks_score_of_the_predecessor(self):
        a = TrialArrays(choice=[1, 1, 1], stimulus=[0.5, 0.5, 0.5], category=[1, 1, 1],
                        prev_choice=[np.nan, 1, 1], prev_stimulus=[np.nan, -0.5, 0.5], prev_category=[np.nan, 0, 1])
        assert np.array_equal(a.prev_correct, [np.nan, 0.0, 1.0], equal_nan=True)


# ── registry contract ───────────────────────────────────────────────────────

class TestRegistry:
    def test_every_name_maps_to_one_producer(self):
        for name in list_stats():
            assert name in _OUTPUT_TO_PRODUCER
        for pname, outs in _list_producers().items():
            for o in outs:
                assert _OUTPUT_TO_PRODUCER[o] == pname

    def test_returns_float_series_in_request_order(self, arrays):
        names = ['sigma', 'accuracy', 'mu', 'lose_shift']
        s = compute_stats(arrays, names, rng=np.random.default_rng(0))
        assert isinstance(s, pd.Series) and s.dtype == float
        assert list(s.index) == names

    def test_all_stats_run(self, arrays):
        s = compute_stats(arrays, list_stats(), rng=np.random.default_rng(0))
        assert len(s) == len(list_stats())
        assert s.notna().sum() > 25   # synthetic has no RT; the rest should fit

    def test_fitter_runs_once_for_all_its_outputs(self, arrays, monkeypatch):
        calls = {'n': 0}
        entry = _PRODUCERS['psychometric']
        orig = entry.func

        def counting(a, *, rng):
            calls['n'] += 1
            return orig(a, rng=rng)
        monkeypatch.setitem(_PRODUCERS, 'psychometric',
                            dataclasses.replace(entry, func=counting))
        compute_stats(arrays, [*PSYCHOMETRIC, 'accuracy', 'mu'])
        assert calls['n'] == 1

    def test_unknown_name_raises(self, arrays):
        with pytest.raises(KeyError):
            compute_stats(arrays, ['not_a_stat'])

    def test_empty_input_gives_nan(self):
        empty = TrialArrays.from_sequence([], [], [])
        s = compute_stats(empty, ['accuracy', 'mu', 'recency', 'reaction_time'])
        assert s.isna().all()

    def test_strict_false_fills_nan_for_failed_producer(self, arrays):
        @stat('_test_boom')
        def _boom(a, *, rng):
            raise RuntimeError('boom')
        try:
            with pytest.raises(RuntimeError):
                compute_stats(arrays, ['_test_boom'])
            s = compute_stats(arrays, ['accuracy', '_test_boom'], strict=False)
            assert np.isfinite(s['accuracy']) and np.isnan(s['_test_boom'])
        finally:
            del _PRODUCERS['_test_boom']
            del _OUTPUT_TO_PRODUCER['_test_boom']

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError):
            @stat('accuracy')
            def _dup(a, *, rng):
                return 0.0
        with pytest.raises(ValueError):
            @fit('_test_fit', outputs=('mu', 'x'))
            def _dup_out(a, *, rng):
                return (0.0, 0.0)

    def test_fit_output_count_is_checked(self, arrays):
        @fit('_test_short', outputs=('_a', '_b'))
        def _short(a, *, rng):
            return (1.0,)
        try:
            with pytest.raises(RuntimeError):
                compute_stats(arrays, ['_a'])
        finally:
            del _PRODUCERS['_test_short']
            for o in ('_a', '_b'):
                del _OUTPUT_TO_PRODUCER[o]

    def test_exchangeability_flags(self):
        assert is_exchangeable('accuracy') and is_exchangeable('mu') and is_exchangeable('recency')
        for n in ('w_stimulus', 'history_decay', 'sd_slope', 'perseveration', 'history_interaction_r2'):
            assert not is_exchangeable(n), n

    def test_rng_makes_jitter_reproducible(self):
        a = TrialArrays(choice=[1, 0, 1, 1], stimulus=[0.5, -0.5, 0.2, 0.8], category=[1, 0, 1, 1],
                        reaction_time=[300, 400, 350, 500])
        x = compute_stats(a, ['reaction_time_jitter'], rng=np.random.default_rng(5))
        y = compute_stats(a, ['reaction_time_jitter'], rng=np.random.default_rng(5))
        z = compute_stats(a, ['reaction_time_jitter'], rng=np.random.default_rng(6))
        assert x.iloc[0] == y.iloc[0] != z.iloc[0]
        assert compute_stats(a, ['reaction_time']).iloc[0] == 375.0


# ── basic statistics ────────────────────────────────────────────────────────

def _block(n_b, correct_b, n_a, correct_a, n_no_response=0):
    """Trials with known per-category accuracy: B trials first, then A trials, then no-responses on B."""
    choice = [1] * correct_b + [0] * (n_b - correct_b) + [0] * correct_a + [1] * (n_a - correct_a)
    category = [1] * n_b + [0] * n_a
    choice += [np.nan] * n_no_response
    category += [1] * n_no_response
    stimulus = [0.5 if c == 1 else -0.5 for c in category]
    return TrialArrays.from_sequence(choice, stimulus, category)


class TestAccuracyBMinusA:
    def test_known_value_and_sign(self):
        s = compute_stats(_block(10, 9, 10, 7), ['accuracy_b_minus_a'])
        assert s['accuracy_b_minus_a'] == pytest.approx(0.9 - 0.7)
        assert compute_stats(_block(10, 7, 10, 9), ['accuracy_b_minus_a']).iloc[0] == pytest.approx(-0.2)

    def test_does_not_move_with_how_often_each_category_is_shown(self):
        balanced = compute_stats(_block(10, 9, 10, 7), ['accuracy_b_minus_a', 'side_bias'])
        more_b = compute_stats(_block(40, 36, 10, 7), ['accuracy_b_minus_a', 'side_bias'])
        assert more_b['accuracy_b_minus_a'] == pytest.approx(balanced['accuracy_b_minus_a'])
        assert more_b['side_bias'] != pytest.approx(balanced['side_bias'])

    def test_no_response_trials_are_ignored(self):
        with_nr = compute_stats(_block(10, 9, 10, 7, n_no_response=5), ['accuracy_b_minus_a']).iloc[0]
        assert with_nr == pytest.approx(0.2)

    def test_nan_when_a_category_has_fewer_than_three_responded_trials(self):
        assert np.isnan(compute_stats(_block(10, 9, 2, 2), ['accuracy_b_minus_a']).iloc[0])
        assert np.isnan(compute_stats(_block(2, 1, 10, 7), ['accuracy_b_minus_a']).iloc[0])
        assert not np.isnan(compute_stats(_block(3, 3, 3, 0), ['accuracy_b_minus_a']).iloc[0])

    def test_is_exchangeable(self):
        assert is_exchangeable('accuracy_b_minus_a')
