"""Tests for behav_utils.data.synthetic.session_from_arrays.

The shared arrays->SessionData constructor used by both generate_synthetic_session
and the SBI simulator; building a real SessionData gives correctly frozen prev_*.
"""

import numpy as np

from behav_utils.data.synthetic import session_from_arrays


def _data(n=120, seed=0, nan_every=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, n)
    c = (s > 0).astype(int)
    ch = c.copy().astype(float)
    if nan_every:
        ch[::nan_every] = np.nan
    return s, ch, c


class TestSessionFromArrays:

    def test_builds_valid_session(self):
        s, ch, c = _data(n=200)
        sess = session_from_arrays(s, ch, c, session_idx=2,
                                   distribution='uniform')
        a = sess.get_arrays()
        assert len(a['choices']) == 200
        assert sess.session_idx == 2
        assert np.array_equal(a['stimuli'], s)

    def test_frozen_prev_is_lag1(self):
        s, ch, c = _data(n=100, seed=1)
        a = session_from_arrays(s, ch, c).get_arrays()
        pc = a['prev_choices']
        assert np.isnan(pc[0])
        assert np.allclose(pc[1:], a['choices'][:-1], equal_nan=True)
        # prev_has_prev is the adjacency not_blockstart for one session
        assert not a['prev_has_prev'][0]
        assert np.all(a['prev_has_prev'][1:])

    def test_abort_default_all_false(self):
        s, ch, c = _data(n=50, seed=2)
        sess = session_from_arrays(s, ch, c)
        assert not np.any(sess.trials.abort)

    def test_nan_choices_preserved(self):
        s, ch, c = _data(n=60, seed=3, nan_every=10)
        sess = session_from_arrays(s, ch, c)
        assert np.isnan(sess.trials.choice[::10]).all()
        # correct is False (not NaN-propagated) at the NaN-choice trials
        assert not np.any(sess.trials.correct[::10])
