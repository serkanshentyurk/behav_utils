"""
Tests for data/ops projection (get_arrays) and pooling (pool_arrays).

After the consolidation:
  - get_arrays is a PURE projection: it does NOT drop aborts and carries the
    frozen, abort-aware prev_* lag-1 view. Aborts are removed only by
    filter_trials.
  - pool_arrays is the SINGLE pooler and concatenates every session it is
    given. It performs no filtering and no session skipping.
"""

from datetime import date

import numpy as np


def _trials_with_aborts():
    """TrialData with aborts at known positions (5 trials, aborts at idx 1 & 4)."""
    from behav_utils.data.structures import TrialData
    n = 5
    return TrialData(
        trial_number=np.arange(1, n + 1),
        stimulus=np.array([10., 11, 12, 13, 14]),
        choice=np.array([0., 1, 0, 1, 1]),
        outcome=np.array([1., 0, 1, 0, 1]),
        correct=np.array([1, 0, 1, 0, 1], dtype=bool),
        category=np.array([0, 0, 1, 1, 1]),
        abort=np.array([False, True, False, False, True]),
    )


def _session(trials, idx=0):
    """Wrap a TrialData in a minimal SessionData."""
    from behav_utils.data.structures import SessionData, SessionMetadata
    return SessionData(
        session_id=f'sess_{idx:03d}', session_idx=idx, date=date(2026, 1, 1),
        metadata=SessionMetadata(fields={'stage': 'x', 'distribution': 'Uniform'}),
        trials=trials,
    )


class TestGetArraysProjection:
    """get_arrays is a pure projection — no filtering of any kind."""

    def test_keeps_aborts(self):
        """get_arrays no longer drops aborts; every trial is returned."""
        from behav_utils.data.ops.filtering import get_arrays
        arr = get_arrays(_trials_with_aborts())
        assert arr['n_trials'] == 5  # all 5 trials, aborts included

    def test_trial_indices_removed(self):
        """The old per-trial index bookkeeping was removed in the refactor."""
        from behav_utils.data.ops.filtering import get_arrays
        arr = get_arrays(_trials_with_aborts())
        assert 'trial_indices' not in arr

    def test_carries_prev_arrays(self):
        """The frozen lag-1 view is projected through, length-aligned."""
        from behav_utils.data.ops.filtering import get_arrays
        arr = get_arrays(_trials_with_aborts())
        for k in ('prev_stimuli', 'prev_choices', 'prev_correct',
                  'prev_categories', 'prev_reaction_time', 'prev_opto_on',
                  'prev_has_prev'):
            assert k in arr
        assert len(arr['prev_stimuli']) == len(arr['stimuli']) == 5

    def test_aborts_removed_only_by_filter(self):
        """get_arrays keeps everything; filter_trials is what drops aborts."""
        from behav_utils.data.ops.filtering import filter_trials, get_arrays
        sess = _session(_trials_with_aborts())
        assert get_arrays(sess.trials)['n_trials'] == 5       # aborts present
        clean = filter_trials([sess], min_trials=1)           # min_trials=1: keep short session
        assert get_arrays(clean[0].trials)['n_trials'] == 3   # 2 aborts dropped by filter


class TestPoolArrays:
    """pool_arrays only pools — no filtering, no session skipping."""

    def test_pools_all_sessions(self, synthetic_animal):
        from behav_utils.data.ops.filtering import pool_arrays
        sessions = synthetic_animal.sessions[:5]
        pooled = pool_arrays(sessions)
        assert pooled['n_sessions'] == 5
        assert pooled['n_trials'] == sum(s.trials.n_trials for s in sessions)

    def test_does_not_skip_short_sessions(self, synthetic_animal):
        """A tiny session is pooled, not skipped (no min_trials filtering)."""
        from behav_utils.data.ops.filtering import pool_arrays
        big = synthetic_animal.sessions[0]            # 300 trials
        tiny = _session(_trials_with_aborts(), idx=99)  # 5 trials
        pooled = pool_arrays([big, tiny])
        assert pooled['n_sessions'] == 2
        assert pooled['n_trials'] == big.trials.n_trials + 5

    def test_current_arrays_match_manual_concat(self, synthetic_animal):
        from behav_utils.data.ops.filtering import get_arrays, pool_arrays
        sessions = synthetic_animal.sessions[:3]
        pooled = pool_arrays(sessions)
        manual = np.concatenate([get_arrays(s.trials)['stimuli'] for s in sessions])
        assert np.allclose(pooled['stimuli'], manual)

    def test_prev_has_prev_equals_not_blockstart(self, synthetic_animal):
        """Pooled prev_has_prev is the per-session not_blockstart pattern:
        False at each session's first trial, True elsewhere."""
        from behav_utils.data.ops.filtering import pool_arrays
        sessions = synthetic_animal.sessions[:3]
        pooled = pool_arrays(sessions)
        expected = []
        for s in sessions:
            nbs = np.ones(s.trials.n_trials, dtype=bool)
            nbs[0] = False
            expected.append(nbs)
        expected = np.concatenate(expected)
        assert np.array_equal(pooled['prev_has_prev'], expected)

    def test_prev_stimulus_nan_at_each_session_start(self, synthetic_animal):
        """No pair bridges a session seam: the first trial of each session has
        no predecessor (NaN prev_stimuli)."""
        from behav_utils.data.ops.filtering import pool_arrays
        sessions = synthetic_animal.sessions[:3]
        pooled = pool_arrays(sessions)
        for start in pooled['session_boundaries'][:-1]:  # start index of each session
            assert np.isnan(pooled['prev_stimuli'][start])
