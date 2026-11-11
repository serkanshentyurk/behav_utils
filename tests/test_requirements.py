"""Requirements: statistics and readouts say what a task must provide, a mismatch names the counterpart,
and a new statistic for a new task needs no library change."""

import dataclasses
import functools

import numpy as np
import pytest

from behav_utils import (
    GoNoGo,
    TrialArrays,
    TwoAFC,
    compute_psychometric_curve,
    compute_stats,
    generate_synthetic_session,
    list_stats,
)
from behav_utils.stats import stat
from behav_utils.stats.registry import _OUTPUT_TO_PRODUCER, _PRODUCERS
from behav_utils.tasks import RequirementError, register_task, require
from behav_utils.tasks.base import _REGISTRY


@functools.cache
def _arrays(kind):
    task = GoNoGo() if kind == 'go_nogo' else TwoAFC()
    return TrialArrays.from_sessions([generate_synthetic_session(task=task, n_trials=1500, rng=0)])


def _unregister(*names):
    for name in names:
        _PRODUCERS.pop(name, None)
        _OUTPUT_TO_PRODUCER.pop(name, None)


@pytest.mark.parametrize('kind', ['two_afc', 'go_nogo'])
def test_every_statistic_computes_or_is_refused_with_a_reason(kind):
    arrays = _arrays(kind)
    applicable = list_stats(task=arrays.task)
    for name in list_stats():
        try:
            value = compute_stats(arrays, [name], rng=np.random.default_rng(0), strict=False)[name]
        except RequirementError as e:
            assert name not in applicable
            assert 'use ' in str(e) or 'no counterpart' in str(e)
        else:
            assert name in applicable and isinstance(value, float)


def test_go_nogo_counterparts_and_refusals():
    arrays = _arrays('go_nogo')
    with pytest.raises(RequirementError, match='use go_bias instead'):
        compute_stats(arrays, ['side_bias'])
    with pytest.raises(RequirementError, match='use hit_minus_cr_rate instead'):
        compute_stats(arrays, ['accuracy_b_minus_a'])
    with pytest.raises(RequirementError, match='no counterpart'):
        compute_stats(arrays, ['w_stimulus'])
    stats = compute_stats(arrays, ['go_bias', 'win_stay', 'accuracy'])
    assert stats['go_bias'] == pytest.approx(np.mean(arrays.valid().choice) - 0.5)
    with pytest.raises(RequirementError, match='use side_bias instead'):
        compute_stats(_arrays('two_afc'), ['go_bias'])


def test_with_no_names_only_what_applies_is_computed():
    applicable = list_stats(task=GoNoGo())
    assert 'go_bias' in applicable and 'side_bias' not in applicable and 'win_stay' in applicable
    assert list(compute_stats(_arrays('go_nogo'), strict=False).index) == applicable


def test_a_new_statistic_for_a_new_task_needs_no_library_change():
    @dataclasses.dataclass(frozen=True)
    class CuedTwoAFC(TwoAFC):
        name = 'cued_two_afc_for_tests'
        capabilities = TwoAFC.capabilities | {'cue_for_tests'}

    register_task(CuedTwoAFC)

    @stat('cued_rate_for_tests', requires=('cue_for_tests',))
    def cued_rate(arrays, *, rng):
        return float(np.mean(arrays.valid().choice))

    @stat('any_task_for_tests')
    def any_task(arrays, *, rng):
        return float(arrays.n_trials)

    try:
        cued = TrialArrays.from_sessions([generate_synthetic_session(task=CuedTwoAFC(), n_trials=300, rng=2)])
        assert np.isfinite(compute_stats(cued, ['cued_rate_for_tests'])['cued_rate_for_tests'])
        assert 'cued_rate_for_tests' in list_stats(task=CuedTwoAFC())
        with pytest.raises(RequirementError, match='needs cue_for_tests'):
            compute_stats(_arrays('two_afc'), ['cued_rate_for_tests'])
        for arrays in (cued, _arrays('two_afc'), _arrays('go_nogo')):
            assert compute_stats(arrays, ['any_task_for_tests'])['any_task_for_tests'] == arrays.n_trials
    finally:
        _unregister('cued_rate_for_tests', 'any_task_for_tests')
        _REGISTRY.pop('cued_two_afc_for_tests', None)


def test_a_capability_no_task_declares_is_flagged():
    @stat('typo_for_tests', requires=('two_choice',))
    def typo(arrays, *, rng):
        return 0.0

    try:
        with pytest.raises(RequirementError, match='no registered task declares two_choice'):
            compute_stats(_arrays('two_afc'), ['typo_for_tests'])
    finally:
        _unregister('typo_for_tests')


def test_a_readout_is_refused_for_a_task_without_a_stimulus_axis():
    @dataclasses.dataclass(frozen=True)
    class NoAxis(TwoAFC):
        name = 'no_axis_for_tests'
        capabilities = frozenset({'binary', 'two_choices', 'scored'})

    arrays = TrialArrays.from_sequence([0, 1, 1, 0], [-0.5, 0.5, 0.4, -0.2], task=NoAxis())
    with pytest.raises(RequirementError, match='compute_psychometric_curve needs stimulus_axis'):
        compute_psychometric_curve(arrays)


def test_require_can_be_used_directly():
    require(TwoAFC(), ('binary', 'two_choices'), 'my_readout')
    with pytest.raises(RequirementError, match='use y instead'):
        require(GoNoGo(), ('two_choices',), 'x', instead={'withhold': 'y'})
