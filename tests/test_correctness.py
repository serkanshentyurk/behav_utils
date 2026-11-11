"""Statistics read correctness from the task's score: TrialArrays carries it, unscored trials are left
out, sessions of different tasks are not pooled, and the update-matrix engine is given the score."""

import warnings

import numpy as np
import pytest

from behav_utils import (
    GoNoGo,
    TrialArrays,
    TwoAFC,
    compute_rolling_stats,
    compute_stats,
    pool_arrays,
    session_from_arrays,
)
from behav_utils.readouts import compute_binned_curve


def _unscored_session(seed=0, n=400):
    """2-AFC where the rig rewards at random on the boundary: a fifth of the stimuli sit exactly on it."""
    rng = np.random.default_rng(seed)
    stim = rng.choice([-0.8, -0.3, 0.0, 0.3, 0.8], n)
    choice = (rng.random(n) < 1 / (1 + np.exp(-6 * stim))).astype(float)
    return session_from_arrays(stim, choice, task=TwoAFC(on_boundary='unscored')), stim, choice


def test_trial_arrays_carry_the_tasks_score():
    session, stim, choice = _unscored_session()
    arrays = TrialArrays.from_sessions([session])
    assert arrays.task == TwoAFC(on_boundary='unscored')
    assert np.isnan(arrays.correct[stim == 0]).all()
    off = stim != 0
    np.testing.assert_array_equal(arrays.correct[off], (choice[off] == (stim[off] > 0)).astype(float))


def test_accuracy_leaves_unscored_trials_out():
    session, stim, choice = _unscored_session()
    off = stim != 0
    expected = np.mean(choice[off] == (stim[off] > 0))
    stats = compute_stats(TrialArrays.from_sessions([session]), ['accuracy', 'accuracy_b_minus_a', 'win_stay'])
    assert stats['accuracy'] == pytest.approx(expected)
    assert np.isfinite(stats['accuracy_b_minus_a']) and np.isfinite(stats['win_stay'])


def test_unscored_predecessors_count_as_neither_win_nor_loss():
    session, stim, _ = _unscored_session()
    pairs = TrialArrays.from_sessions([session]).lag1_pairs()
    after_boundary = pairs.prev_stimulus == 0
    assert after_boundary.any() and np.isnan(pairs.prev_correct[after_boundary]).all()


def test_the_logistic_history_runs_with_unscored_lags():
    session, _, _ = _unscored_session(n=600)
    stats = compute_stats(TrialArrays.from_sessions([session]), ['w_prev_outcome_1', 'w_stimulus'])
    assert np.isfinite(stats['w_prev_outcome_1']) and np.isfinite(stats['w_stimulus'])


def test_the_binned_accuracy_curve_uses_scored_trials_only():
    session, stim, _ = _unscored_session()
    curve = compute_binned_curve(TrialArrays.from_sessions([session]), kind='accuracy')
    assert np.isfinite(curve.values[curve.counts > 0]).all()
    assert curve.counts.sum() == int((stim != 0).sum())


def test_sessions_of_different_tasks_are_not_pooled():
    two_afc = session_from_arrays(np.array([-0.5, 0.5]), np.array([0.0, 1.0]))
    go_nogo = session_from_arrays(np.array([-0.5, 0.5]), np.array([np.nan, 1.0]), task=GoNoGo())
    with pytest.raises(ValueError, match='different tasks'):
        pool_arrays([two_afc, go_nogo])
    with pytest.raises(ValueError, match='different tasks'):
        TrialArrays.concat([TrialArrays.from_sessions([two_afc]), TrialArrays.from_sessions([go_nogo])])


def test_from_sequence_lets_the_task_decide():
    arrays = TrialArrays.from_sequence([0, 1, np.nan], [-0.5, 0.5, 0.5])
    np.testing.assert_array_equal(arrays.category, [0, 1, 1])
    np.testing.assert_array_equal(arrays.correct, [1, 1, np.nan])
    with pytest.raises(ValueError, match='disagree'):
        TrialArrays.from_sequence([0, 1], [-0.5, 0.5], [1, 1])
    go_nogo = TrialArrays.from_sequence([np.nan, 1], [-0.5, 0.5], task=GoNoGo())
    np.testing.assert_array_equal(go_nogo.choice, [0, 1])
    np.testing.assert_array_equal(go_nogo.correct, [1, 1])


def test_trial_data_prev_correct_is_nan_after_a_no_response():
    session = session_from_arrays(np.array([-0.5, 0.5, 0.5]), np.array([np.nan, 1.0, 1.0]))
    np.testing.assert_array_equal(session.trials.prev_correct, [np.nan, np.nan, 1.0])


def test_rolling_stats_on_a_failing_session_raise_no_future_warning(synthetic_animal):
    broken = session_from_arrays(np.array([0.5]), np.array([1.0]))
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        compute_rolling_stats([broken, *synthetic_animal.sessions[:2]], ['accuracy'], window=50)
