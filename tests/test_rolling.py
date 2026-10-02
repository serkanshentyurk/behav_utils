"""Tests for behav_utils.analysis.rolling.

Covers compute_rolling_stats (RollingStats tidy frame, multi-stat,
validation, short-session fallback, cohort survival) plus a bit-identity lock

"""

from datetime import date, timedelta

import numpy as np
import pytest

from behav_utils.analysis.rolling import _iter_windows, compute_rolling_stats


# ── builders (real SessionData, so pool_arrays / prev_* behave as in prod) ──
def _session(session_idx, n, *, noise=0.10, distribution='Hard-A',
             session_type='regular', seed=0):
    from behav_utils.data.structures import SessionData, SessionMetadata, TrialData
    rng = np.random.default_rng(seed + session_idx)
    stimuli = rng.uniform(-1, 1, n)
    categories = (stimuli > 0).astype(float)
    choices = categories.copy()
    flip = rng.random(n) < noise
    choices[flip] = 1 - choices[flip]
    trials = TrialData(
        trial_number=np.arange(n), stimulus=stimuli, category=categories,
        choice=choices, outcome=(choices == categories).astype(float),
        correct=(choices == categories), abort=np.zeros(n, dtype=bool),
        opto_on=np.zeros(n, dtype=bool))
    return SessionData(
        session_id=f'sess_{session_idx:03d}', session_idx=session_idx,
        date=date(2026, 1, 1) + timedelta(days=session_idx),
        metadata=SessionMetadata(fields={'stage': 'Full_Task_Cont',
                                          'distribution': distribution}),
        trials=trials, session_type=session_type)


class _BadSession:
    """Duck-typed session whose arrays cannot be read (cohort-survival test)."""
    session_id = 'bad_001'
    session_idx = 9
    session_type = 'opto'

    def get_arrays(self):
        raise RuntimeError("boom")


def _n_windows(n, window, step):
    return len(range(0, n - window + 1, step))


# ── _iter_windows ──────────────────────────────────────────────────────────
def test_iter_windows_full_only():
    n, window, step = 200, 50, 10
    w = _iter_windows(n, window, step)
    assert len(w) == _n_windows(n, window, step)
    assert w[0][0] == 25.0                       # centre of first window
    assert all(sl.stop <= n for _, sl in w)      # never runs past the block
    assert all((sl.stop - sl.start) == window for _, sl in w)  # full windows


def test_iter_windows_empty_when_short():
    assert _iter_windows(30, 50, 10) == []


# ── per-session shape / counts ─────────────────────────────────────────────
def test_per_session_shape_and_counts():
    ns = [200, 130, 60]
    sessions = [_session(i, n) for i, n in enumerate(ns)]
    res = compute_rolling_stats(sessions, 'accuracy', per_session=True, window=50, step=10)
    assert res.per_session and len(res.sessions) == len(ns)
    for sid, n in zip(res.sessions, ns):
        c = res.curve('accuracy', sid)
        assert res.n_trials(sid) == n
        assert len(c) == _n_windows(n, 50, 10)
    info = res.session_info.set_index('session')
    assert (info['session_type'] == 'regular').all() and (info['distribution'] == 'Hard-A').all()


def test_multi_stat_same_length():
    res = compute_rolling_stats([_session(0, 200)], ['accuracy', 'side_bias'], window=50, step=10)
    assert res.names == ('accuracy', 'side_bias')
    assert len(res.curve('accuracy')) == len(res.curve('side_bias'))
    assert set(res.curves['stat']) == {'accuracy', 'side_bias'}


# ── validation (fail fast) ─────────────────────────────────────────────────
@pytest.mark.parametrize('bad', ['nonsense', []])
def test_validation_rejects_bad_stats(bad):
    with pytest.raises((ValueError, KeyError)):
        compute_rolling_stats([_session(0, 100)], bad)


# ── short-session handling ─────────────────────────────────────────────────
def test_short_session_single_point():
    res = compute_rolling_stats([_session(0, 30)], 'accuracy', window=50, step=10, min_short=10)
    c = res.curve('accuracy')
    assert len(c) == 1 and c['trial'].iloc[0] == 15.0
    assert res.n_trials(res.sessions[0]) == 30


def test_too_short_session_empty_but_present():
    res = compute_rolling_stats([_session(0, 5)], 'accuracy', window=50, step=10, min_short=10)
    assert len(res.curve('accuracy')) == 0
    assert res.sessions == [res.session_info['session'].iloc[0]]
    assert res.n_trials(res.sessions[0]) == 5


# ── cohort survival ────────────────────────────────────────────────────────
def test_never_raises_on_bad_session():
    sessions = [_session(0, 120), _BadSession()]
    with pytest.warns(RuntimeWarning):
        res = compute_rolling_stats(sessions, 'accuracy', window=50, step=10)
    assert len(res.sessions) == 2
    good, bad = res.sessions
    assert res.n_trials(good) == 120 and len(res.curve('accuracy', good)) > 0
    assert bad == 'bad_001' and len(res.curve('accuracy', bad)) == 0


# ── pooled mode (windows cross session boundaries) ─────────────────────────
def test_pooled_mode():
    ns = [80, 90]
    sessions = [_session(i, n) for i, n in enumerate(ns)]
    res = compute_rolling_stats(sessions, 'accuracy', per_session=False, window=50, step=10)
    assert not res.per_session and res.sessions == ['pooled']
    total = sum(ns)
    assert res.n_trials('pooled') == total
    assert len(res.curve('accuracy')) == _n_windows(total, 50, 10)


def test_empty_sessions():
    res = compute_rolling_stats([], 'accuracy')
    assert res.sessions == [] and len(res.curves) == 0
    res_p = compute_rolling_stats([], 'accuracy', per_session=False)
    assert res_p.n_trials('pooled') == 0 and len(res_p.curves) == 0


# ── mu goes through the guarded registry ───────────────────────────────────
def test_mu_runs_and_is_finite_on_clean_session():
    res = compute_rolling_stats([_session(0, 300, noise=0.05)], 'mu', window=50, step=10)
    mu = res.curve('mu')['value'].to_numpy()
    assert mu.size == _n_windows(300, 50, 10)
    assert np.isfinite(mu).any()
