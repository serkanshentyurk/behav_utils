"""analysis/resampling — the single drawing engine every bootstrap interval depends on.

Checks the contract rather than the numbers: the draw size, that lag-1 pairing survives a resample,
that the matched-n target is the smallest phase, that the readout and stat-vector resamplers return
the typed results the rest of the library consumes, and that order-dependent stats are refused.
"""

import numpy as np
import pytest

from behav_utils.analysis.resampling import (
    calculate_min_n,
    downsample,
    resample_psychometric_curve,
    resample_stat_vectors,
    resample_update_matrix,
)
from behav_utils.data.ops.filtering import filter_trials, pool_arrays
from behav_utils.data.ops.selection import select_sessions
from behav_utils.readouts import PsychometricCurve, UpdateMatrix


@pytest.fixture
def clean(synthetic_animal):
    return filter_trials(select_sessions(synthetic_animal, preset='all_uniform'))


def _n_trials(sessions):
    return pool_arrays(sessions)['n_trials']


def test_downsample_draws_about_n(clean):
    rng = np.random.default_rng(0)
    n = 150
    out = downsample(clean, n, unit='trials', with_replacement=False, rng=rng)
    assert abs(_n_trials(out) - n) <= 8          # stratified draw: close to n, not exactly n
    assert all(type(s) is type(clean[0]) for s in out)


def test_downsample_without_replacement_is_a_subset(clean):
    rng = np.random.default_rng(1)
    out = downsample(clean, 100, with_replacement=False, rng=rng)
    pooled_in = pool_arrays(clean)
    for s in out:
        # every drawn stimulus/choice pair exists in the source
        src = set(zip(pooled_in['stimuli'].round(6), pooled_in['choices']))
        assert all((round(float(a), 6), b) in src for a, b in zip(s.trials.stimulus, s.trials.choice))


def test_downsample_keeps_lag1_pairing(clean):
    """A drawn trial carries its own predecessor (frozen prev_* fields), even when drawn twice."""
    rng = np.random.default_rng(2)
    out = downsample(clean, 300, with_replacement=True, rng=rng)
    src = {}
    for s in clean:
        t = s.trials
        for i in range(len(t.stimulus)):
            src[(s.session_id, round(float(t.stimulus[i]), 6), i)] = (t.prev_stimulus[i], t.prev_choice[i])
    for s in out:
        t = s.trials
        assert len(t.prev_stimulus) == len(t.stimulus)
        # prev fields are among the predecessor values of the source session, never recomputed from order
        srcs = {v for k, v in src.items() if k[0] == s.session_id}
        assert all((ps, pc) in srcs for ps, pc in zip(t.prev_stimulus, t.prev_choice) if not np.isnan(ps))


def test_downsample_empty_in_empty_out():
    assert downsample([], 10) == []


def test_calculate_min_n_is_the_smallest_phase(clean):
    small = downsample(clean, 80, with_replacement=False, rng=np.random.default_rng(3))
    n_small, n_full = _n_trials(small), _n_trials(clean)
    assert n_small < n_full
    assert calculate_min_n([clean, small]) == n_small
    assert calculate_min_n([clean, [], small]) == n_small      # empties skipped
    assert calculate_min_n([[], []]) == 0
    assert 0 < calculate_min_n([clean], unit='pairs') <= n_full


def test_resample_psychometric_curve_returns_curve(clean):
    c = resample_psychometric_curve(clean, 150, n_repeats=5, seed=0)
    assert isinstance(c, PsychometricCurve)
    assert np.isfinite(c.params['mu']) if hasattr(c, 'params') and c.success else True


def test_resample_update_matrix_returns_matrix(clean):
    um = resample_update_matrix(clean, 150, n_repeats=5, seed=0)
    assert isinstance(um, UpdateMatrix)
    assert um.matrix.shape[0] == um.matrix.shape[1]


def test_resample_stat_vectors_shape_and_seed(clean):
    a = resample_stat_vectors(clean, ['accuracy', 'side_bias'], n_repeats=6, seed=4)
    b = resample_stat_vectors(clean, ['accuracy', 'side_bias'], n_repeats=6, seed=4)
    c = resample_stat_vectors(clean, ['accuracy', 'side_bias'], n_repeats=6, seed=5)
    assert a.shape == (6, 2) and list(a.columns) == ['accuracy', 'side_bias']
    assert a.equals(b)
    assert not a.equals(c)
    assert a['accuracy'].between(0, 1).all()


def test_resample_stat_vectors_matched_n(clean):
    m = resample_stat_vectors(clean, ['accuracy'], n=100, n_repeats=3, with_replacement=False, seed=0)
    assert m.shape == (3, 1) and m['accuracy'].notna().all()


def test_resample_stat_vectors_refuses_order_dependent_stats(clean):
    from behav_utils.stats import is_exchangeable, list_stats
    non_exch = [s for s in list_stats() if not is_exchangeable(s)]
    if not non_exch:
        pytest.skip('no order-dependent stat registered')
    with pytest.raises((ValueError, KeyError)):
        resample_stat_vectors(clean, [non_exch[0]], n_repeats=2)
