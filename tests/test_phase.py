"""Tests for ``select_sessions`` (behav_utils.data.ops.selection) — the phase /
session-selection primitive that replaced the old ``analysis.phase`` layer.

``select_sessions(animal, preset=None, **overrides)`` resolves a registered
preset and/or ad-hoc ``SessionFilter`` criteria and returns a list of matching
``SessionData``. Ad-hoc defaults exclude nothing; presets carry exclude_types
and keep opto (``exclude_opto=False``).
"""
from datetime import date, timedelta

import numpy as np
import pytest

from behav_utils.data.ops.selection import get_preset, select_sessions
from behav_utils.data.structures import AnimalData, SessionData, SessionMetadata, TrialData


def _sess(idx, dist, stype, stage='Full_Task_Cont', noise=0.08, n=300, seed=0):
    """One session; ``noise`` sets accuracy (low noise → high accuracy)."""
    rng = np.random.default_rng(seed + idx)
    stim = rng.uniform(-1, 1, n)
    cat = (stim > 0).astype(float)
    ch = ((stim + rng.normal(0, noise, n)) > 0).astype(float)
    tr = TrialData(trial_number=np.arange(n), stimulus=stim, category=cat, choice=ch,
                   outcome=(ch == cat).astype(float), correct=(ch == cat),
                   abort=np.zeros(n, bool), opto_on=np.zeros(n, bool))
    return SessionData(session_id=f'{dist}_{stype}_{idx}', session_idx=idx,
                       date=date(2026, 1, 1) + timedelta(days=idx),
                       metadata=SessionMetadata(fields={'stage': stage, 'distribution': dist}),
                       trials=tr, session_type=stype)


def _animal():
    s = [
        _sess(0, 'Uniform', 'regular', noise=0.08),   # high-acc uniform
        _sess(1, 'Uniform', 'regular', noise=0.08),
        _sess(2, 'Uniform', 'regular', noise=0.08),
        _sess(3, 'Uniform', 'regular', noise=2.0),   # low-acc uniform
        _sess(4, 'Uniform', 'regular', noise=2.0),
        _sess(5, 'Hard-A', 'opto', noise=0.12),       # hard opto
        _sess(6, 'Hard-A', 'opto', noise=0.12),
        _sess(7, 'Uniform', 'masking', noise=0.10),   # masking
        _sess(8, 'Uniform', 'regular', stage='Habituation', noise=0.10),
    ]
    return AnimalData(animal_id='SS16', sessions=s)


def _acc(sess):
    return float(np.mean(sess.trials.correct))


class TestPresetExpertUniform:

    def test_preset_resolves_to_uniform_full_task_high_acc(self):
        out = select_sessions(_animal(), 'expert_uniform', last_fraction=None)
        assert out, "expert_uniform should match the high-accuracy uniform sessions"
        for s in out:
            assert s.metadata.fields['distribution'] == 'Uniform'
            assert s.metadata.fields['stage'] == 'Full_Task_Cont'
            assert s.session_type != 'masking'
            assert _acc(s) >= 0.7

    def test_low_accuracy_sessions_excluded(self):
        out_ids = {s.session_id for s in select_sessions(_animal(), 'expert_uniform', last_fraction=None)}
        assert 'Uniform_regular_3' not in out_ids
        assert 'Uniform_regular_4' not in out_ids

    def test_override_tightening_narrows_selection(self):
        base = select_sessions(_animal(), 'expert_uniform')
        strict = select_sessions(_animal(), 'expert_uniform', min_accuracy=0.999)
        assert len(strict) <= len(base)


class TestAdHocSelection:

    def test_select_by_distribution(self):
        out = select_sessions(_animal(), distribution='Hard-A')
        assert out and all(s.metadata.fields['distribution'] == 'Hard-A' for s in out)

    def test_select_by_stage(self):
        out = select_sessions(_animal(), stage='Habituation')
        assert out and all(s.metadata.fields['stage'] == 'Habituation' for s in out)

    def test_select_by_session_type_opto(self):
        out = select_sessions(_animal(), session_type='opto')
        assert out and all(s.session_type == 'opto' for s in out)

    def test_masking_excluded_only_by_exclude_types(self):
        # ad hoc: nothing excluded
        assert any(s.session_type == 'masking' for s in select_sessions(_animal(), distribution='Uniform'))
        # exclude_types drops them; session_type selects them
        assert not any(s.session_type == 'masking'
                       for s in select_sessions(_animal(), distribution='Uniform', exclude_types=('masking',)))
        assert all(s.session_type == 'masking' for s in select_sessions(_animal(), session_type='masking'))
    def test_min_accuracy_filters_low(self):
        out = select_sessions(_animal(), distribution='Uniform', min_accuracy=0.7)
        assert out and all(_acc(s) >= 0.7 for s in out)

    def test_last_n_keeps_most_recent(self):
        out = select_sessions(_animal(), distribution='Uniform', min_accuracy=0.0,
                              last_n=2)
        assert len(out) == 2
        idxs = [s.session_idx for s in out]
        assert idxs == sorted(idxs)                       # chronological

    def test_no_match_returns_empty_list(self):
        out = select_sessions(_animal(), distribution='Nonexistent')
        assert out == []

    def test_returns_session_data_objects(self):
        out = select_sessions(_animal(), distribution='Uniform')
        assert all(isinstance(s, SessionData) for s in out)


class TestPresetErrors:

    def test_unknown_preset_raises(self):
        with pytest.raises((KeyError, ValueError)):
            select_sessions(_animal(), 'not_a_real_preset')

    def test_get_preset_expert_uniform_fields(self):
        f = get_preset('expert_uniform')
        assert f.distribution == 'Uniform'
        assert f.stage == 'Full_Task_Cont'
        assert f.min_accuracy == pytest.approx(0.7)
        assert 'masking' in f.exclude_types
