"""
Tests for the lag-1 (``prev_trial``) view on ``TrialData``.

Covers: alignment, the index-0 / ``has_prev`` boundary, reward & opto lag,
no-response propagation, and the key property — ``prev_trial`` is frozen on the
raw session and CARRIED (not recomputed) through filtering, so a survivor of a
``clear_flags`` filter keeps its real (possibly dropped) predecessor.
"""
import numpy as np

from behav_utils.data.ops.filtering import filter_trial_data
from behav_utils.data.structures import PrevTrial, TrialData


def _session(n: int = 6) -> TrialData:
    """Deterministic session: idx2 is a no-response; opto on idx1 and idx3."""
    return TrialData(
        trial_number=np.arange(1, n + 1),
        stimulus=np.array([-0.8, -0.2, 0.3, 0.9, -0.5, 0.1])[:n],
        choice=np.array([0.0, 1.0, np.nan, 1.0, 0.0, 1.0])[:n],
        outcome=np.array(["x"] * n, dtype=object),
        correct=np.array([1, 0, 0, 1, 1, 0], dtype=bool)[:n],
        category=np.array([0, 1, 1, 1, 0, 1])[:n],
        opto_on=np.array([False, True, False, True, False, False])[:n],
    )


def _session_with_aborts() -> TrialData:
    """Session of 5 trials with aborts at idx1 and idx4 (stim 10..14)."""
    n = 5
    return TrialData(
        trial_number=np.arange(1, n + 1),
        stimulus=np.array([10.0, 11, 12, 13, 14]),
        choice=np.array([0.0, 1, 0, 1, 1]),
        outcome=np.array(["x"] * n, dtype=object),
        correct=np.array([1, 0, 1, 0, 1], dtype=bool),
        category=np.array([0, 0, 1, 1, 1]),
        opto_on=np.array([False] * n),
        abort=np.array([False, True, False, False, True]),
    )


def _expected_prev(values: np.ndarray, abort: np.ndarray) -> np.ndarray:
    """Reference implementation of abort-aware prev: previous non-abort value."""
    n = len(values)
    out = np.full(n, np.nan)
    last = -1
    for i in range(n):
        if abort[i]:
            continue
        if last >= 0:
            out[i] = values[last]
        last = i
    return out


def test_prev_trial_returns_view():
    assert isinstance(_session().prev_trial, PrevTrial)


def test_alignment_and_boundary():
    td = _session()
    pt = td.prev_trial
    assert np.isnan(pt.stimulus[0])          # no predecessor at index 0
    assert not pt.has_prev[0]
    assert pt.has_prev[1:].all()             # every later trial has one
    assert np.allclose(pt.stimulus[1:], td.stimulus[:-1])   # prev[i] == cur[i-1]


def test_reward_and_opto_lag():
    td = _session()
    pt = td.prev_trial
    assert np.isnan(pt.opto_on[0])
    assert np.allclose(pt.opto_on[1:], td.opto_on[:-1].astype(float))
    assert np.allclose(pt.correct[1:], td.correct[:-1].astype(float))
    assert np.allclose(pt.category[1:], td.category[:-1].astype(float))


def test_previous_no_response_propagates_nan():
    td = _session()                       # idx2 is a no-response
    pt = td.prev_trial
    assert np.isnan(pt.choice[3])         # idx3's predecessor (idx2) didn't respond
    assert pt.has_prev[2]                 # the no-response trial still HAS a predecessor
    assert not np.isnan(pt.choice[2])     # ... and that predecessor (idx1) responded


def test_frozen_and_carried_through_filter():
    """Core property: filtering slices prev, it does NOT recompute on the subset."""
    td = _session()
    pt = td.prev_trial
    mask = np.array([False, True, True, False, True, True])   # drop idx0, idx3
    ft = filter_trial_data(td, mask, clear_flags=True)

    assert not ft.opto_on.any()                              # current opto cleared
    assert np.allclose(ft.prev_stimulus, pt.stimulus[mask], equal_nan=True)
    assert np.allclose(ft.prev_opto_on, pt.opto_on[mask], equal_nan=True)
    assert np.array_equal(ft.prev_has_prev, pt.has_prev[mask])

    # concretely: the survivor that was idx4 keeps its DROPPED predecessor
    # (idx3, stimulus 0.9). A recompute on the subset would wrongly give 0.3.
    surviving_positions = np.flatnonzero(mask)               # [1, 2, 4, 5]
    pos_of_idx4 = int(np.flatnonzero(surviving_positions == 4)[0])
    assert np.isclose(ft.prev_stimulus[pos_of_idx4], 0.9)


def test_recompute_on_subset_would_differ():
    """Guard: the buggy 'recompute on the filtered subset' gives a different answer."""
    td = _session()
    mask = np.array([False, True, True, False, True, True])
    ft = filter_trial_data(td, mask, clear_flags=True)
    buggy = np.full(ft.n_trials, np.nan)
    buggy[1:] = ft.stimulus[:-1]
    assert not np.allclose(ft.prev_stimulus, buggy, equal_nan=True)


def test_single_trial_session():
    pt = _session(n=1).prev_trial
    assert pt.stimulus.shape == (1,)
    assert np.isnan(pt.stimulus[0])
    assert not pt.has_prev[0]


def test_synthetic_session_has_prev_trial():
    from behav_utils.data import generate_synthetic_animal
    animal, _ = generate_synthetic_animal(
        animal_id="SYN01", n_sessions=2, trials_per_session=40
    )
    trials = animal.sessions[0].trials
    pt = trials.prev_trial
    abort = np.asarray(trials.abort, dtype=bool)
    assert pt.stimulus.shape[0] == trials.n_trials
    # Abort-aware: predecessor is the previous NON-ABORT trial, not cur[i-1].
    expected = _expected_prev(np.asarray(trials.stimulus, dtype=float), abort)
    assert np.allclose(pt.stimulus, expected, equal_nan=True)
    # has_prev is False at abort trials and at the first completed trial.
    assert not pt.has_prev[abort].any()
    first_completed = int(np.flatnonzero(~abort)[0])
    assert not pt.has_prev[first_completed]


def test_abort_is_bridged():
    """A trial after an abort takes the last COMPLETED trial as its predecessor."""
    pt = _session_with_aborts().prev_trial
    assert np.isclose(pt.stimulus[2], 10.0)   # idx2 skips abort idx1 -> idx0
    assert np.isclose(pt.stimulus[3], 12.0)   # idx3 -> idx2


def test_abort_trial_has_no_prev_view():
    """An abort is not a valid current trial: no predecessor view."""
    td = _session_with_aborts()
    pt = td.prev_trial
    ab = np.asarray(td.abort, dtype=bool)
    assert not pt.has_prev[ab].any()
    assert np.isnan(pt.stimulus[ab]).all()


def test_abort_bridged_matches_post_filter_sequence():
    """After aborts are filtered out, the carried prev matches the bridged
    sequence and prev_has_prev equals not_blockstart for the survivors."""
    td = _session_with_aborts()
    ab = np.asarray(td.abort, dtype=bool)
    ft = filter_trial_data(td, ~ab, clear_flags=True)        # drop the aborts
    # survivors are stim [10, 12, 13]; prev bridges -> [nan, 10, 12]
    assert np.allclose(ft.prev_stimulus, [np.nan, 10.0, 12.0], equal_nan=True)
    # not_blockstart for a 3-trial block is [False, True, True]
    assert list(ft.prev_has_prev) == [False, True, True]


def test_no_abort_matches_naive_lag():
    """With no aborts, prev_* is an ordinary one-trial lag (Phase-1 behaviour)."""
    td = _session()                                          # no aborts
    pt = td.prev_trial
    assert np.allclose(pt.stimulus[1:], td.stimulus[:-1], equal_nan=True)
    assert not pt.has_prev[0] and pt.has_prev[1:].all()
