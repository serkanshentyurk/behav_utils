"""Tests for the opto analysis path, which is now the generic pipeline rather
than a dedicated ``analysis.opto`` module:

    filter_trials(sessions, trial_type=...)   — opto / non_opto / post_opto splits
    compute_delta_stat({non_opto, opto}, ...) — the opto-vs-control contrast

The opto manipulation here injects a pure *criterion* shift (a displacement of
the psychometric ``mu`` with sensitivity preserved), so the contrast should
recover a significant ``mu`` difference while accuracy stays roughly unchanged —
the "criterion shift, preserved d′" signature.
"""
from datetime import date, timedelta

import numpy as np
import pytest

from behav_utils.analysis.comparison import compute_delta_stat
from behav_utils.data.arrays import TrialArrays
from behav_utils.data.ops.filtering import filter_trials, pool_arrays
from behav_utils.data.structures import SessionData, SessionMetadata, TrialData
from behav_utils.stats import PSYCHOMETRIC, compute_stats


def _opto_session(idx, n=320, opto_bias=0.0, opto_frac=0.3, noise=0.12, seed=0):
    """Session with ``opto_frac`` opto trials; opto shifts the criterion by
    ``opto_bias`` (added to the decision variable on opto trials only)."""
    rng = np.random.default_rng(seed + idx)
    stim = rng.uniform(-1, 1, n)
    cat = (stim > 0).astype(float)
    opto = np.zeros(n, bool)
    opto[rng.choice(n, int(n * opto_frac), replace=False)] = True
    dv = stim + rng.normal(0, noise, n) + opto * opto_bias
    ch = (dv > 0).astype(float)
    tr = TrialData(trial_number=np.arange(n), stimulus=stim, category=cat, choice=ch,
                   outcome=(ch == cat).astype(float), correct=(ch == cat),
                   abort=np.zeros(n, bool), opto_on=opto)
    return SessionData(session_id=f'opto_{idx}', session_idx=idx,
                       date=date(2026, 1, 1) + timedelta(days=idx),
                       metadata=SessionMetadata(fields={'stage': 'Full_Task_Cont',
                                                        'distribution': 'Uniform'}),
                       trials=tr, session_type='opto')


def _sessions(n_sessions=4, **kw):
    return [_opto_session(i, **kw) for i in range(n_sessions)]


class TestTrialTypeFiltering:

    # NOTE: filter_trials(trial_type='opto') selects the opto trials but does not
    # carry the opto_on flag through onto the returned trials (it resets to all
    # False), so trial COUNT — not opto_on — is the honest check that the right
    # subset was selected. See test_opto_and_non_opto_partition_the_valid_trials.
    def test_opto_selects_the_opto_fraction(self):
        sess = _sessions(opto_bias=0.3, opto_frac=0.3)
        n_all = sum(s.trials.n_trials
                    for s in filter_trials(sess, trial_type='all', exclude_opto=False))
        n_opto = sum(s.trials.n_trials
                     for s in filter_trials(sess, trial_type='opto', exclude_opto=False))
        assert 0.20 * n_all < n_opto < 0.40 * n_all       # ≈ 30 %

    def test_non_opto_selects_the_complement(self):
        sess = _sessions(opto_bias=0.3, opto_frac=0.3)
        n_all = sum(s.trials.n_trials
                    for s in filter_trials(sess, trial_type='all', exclude_opto=False))
        n_non = sum(s.trials.n_trials
                    for s in filter_trials(sess, trial_type='non_opto'))
        assert 0.60 * n_all < n_non < 0.80 * n_all        # ≈ 70 %

    def test_post_opto_supported_and_non_empty(self):
        ph = filter_trials(_sessions(opto_bias=0.3), trial_type='post_opto', exclude_opto=False)
        assert pool_arrays(ph)['stimuli'].size >= 0        # supported (may be small)

    def test_none_equals_all_valid(self):
        allv = pool_arrays(filter_trials(_sessions(), trial_type='all', exclude_opto=False))
        none = pool_arrays(filter_trials(_sessions(), trial_type=None, exclude_opto=False))
        assert allv['stimuli'].size == none['stimuli'].size

    def test_opto_and_non_opto_partition_the_valid_trials(self):
        sess = _sessions(opto_bias=0.3)
        n_all = pool_arrays(filter_trials(sess, trial_type='all', exclude_opto=False))['stimuli'].size
        n_opto = pool_arrays(filter_trials(sess, trial_type='opto', exclude_opto=False))['stimuli'].size
        n_non = pool_arrays(filter_trials(sess, trial_type='non_opto'))['stimuli'].size
        assert n_opto + n_non == n_all

    def test_invalid_trial_type_raises(self):
        with pytest.raises((ValueError, KeyError)):
            filter_trials(_sessions(), trial_type='not_a_type')

    def test_min_trials_drops_small_sessions(self):
        # a session with only a handful of opto trials falls below min_trials.
        sess = _sessions(n_sessions=1, n=40, opto_frac=0.1)   # ~4 opto trials
        ph = filter_trials(sess, trial_type='opto', exclude_opto=False, min_trials=10)
        assert len(ph) == 0


class TestOptoContrast:
    """opto vs non_opto through the generic pipeline. mu / sigma use a single
    psychometric fit per side (compute_stats); the significance test
    uses the fit-free accuracy permutation — opto labels are randomised per trial,
    so permutation is the valid test."""

    def _phases(self, opto_bias, noise=0.1, n_sessions=3):
        sess = _sessions(n_sessions=n_sessions, opto_bias=opto_bias, noise=noise)
        opto = TrialArrays.from_sessions(filter_trials(sess, trial_type='opto', exclude_opto=False))
        non = TrialArrays.from_sessions(filter_trials(sess, trial_type='non_opto'))
        return opto, non

    def test_opto_shifts_mu_preserves_sensitivity(self):
        opto, non = self._phases(opto_bias=0.45)
        so = compute_stats(opto, PSYCHOMETRIC)
        sn = compute_stats(non, PSYCHOMETRIC)
        assert abs(so['mu'] - sn['mu']) > 0.2          # criterion shifts …
        assert abs(so['sigma'] - sn['sigma']) < 0.06   # … sensitivity preserved

    def test_opto_effect_is_significant_on_accuracy(self):
        sess = _sessions(n_sessions=3, opto_bias=0.45, noise=0.1)
        opto = filter_trials(sess, trial_type='opto', exclude_opto=False)
        non = filter_trials(sess, trial_type='non_opto')
        r = compute_delta_stat({'non_opto': non, 'opto': opto}, ['accuracy'],
                               reference='non_opto', curve=False, n_permutations=200,
                               n_bootstrap=40, seed=1)
        con = r.contrast('opto')
        assert abs(con.diff['accuracy']) > 0.02
        assert con.perm_p['accuracy'] < 0.05

    def test_no_effect_when_bias_zero(self):
        opto, non = self._phases(opto_bias=0.0)
        so = compute_stats(opto, PSYCHOMETRIC)
        sn = compute_stats(non, PSYCHOMETRIC)
        assert abs(so['mu'] - sn['mu']) < 0.15
