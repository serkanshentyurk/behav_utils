"""Tasks: categories, the correct response and scoring for 2-AFC and go/no-go, and the registry."""

import dataclasses

import numpy as np
import pytest

from behav_utils import GoNoGo, Outcome, TwoAFC
from behav_utils.tasks import ResponseSpace, Task, list_tasks, register_task, task_from_dict

STIM = np.array([-0.5, 0.0, 0.5, np.nan])


class TestTwoAFC:
    def test_b_is_the_positive_side_and_the_boundary_is_a(self):
        np.testing.assert_array_equal(TwoAFC().category(STIM), [0, 0, 1, np.nan])

    def test_b_on_the_negative_side(self):
        np.testing.assert_array_equal(TwoAFC(b_side='negative').category(STIM), [1, 0, 0, np.nan])

    @pytest.mark.parametrize('rule, value', [('A', 0.0), ('B', 1.0), ('unscored', np.nan)])
    def test_on_boundary(self, rule, value):
        np.testing.assert_array_equal(TwoAFC(on_boundary=rule).category([0.0]), [value])

    def test_a_bad_setting_raises(self):
        with pytest.raises(ValueError, match='b_side'):
            TwoAFC(b_side='left')


class TestScore:
    def test_abort_then_no_response_then_unscored_then_correct_or_error(self):
        task = TwoAFC(on_boundary='unscored')
        stim = np.array([-0.5, 0.5, 0.0, 0.5, 0.5])
        response = np.array([0, 0, 1, np.nan, 1])
        abort = np.array([False, False, False, False, True])
        assert task.score(response, task.truth(stim), abort).tolist() == [
            Outcome.CORRECT, Outcome.ERROR, Outcome.UNSCORED, Outcome.NO_RESPONSE, Outcome.ABORT]


class TestGoNoGo:
    def test_go_side_and_the_boundary(self):
        np.testing.assert_array_equal(GoNoGo().truth(STIM), [0, 0, 1, np.nan])
        np.testing.assert_array_equal(GoNoGo(go_side='negative', on_boundary='go').truth(STIM), [1, 1, 0, np.nan])

    def test_a_blank_that_is_not_an_abort_is_a_withhold(self):
        out = GoNoGo().complete_responses([np.nan, 1.0, np.nan], [False, False, True])
        np.testing.assert_array_equal(out, [0.0, 1.0, np.nan])

    def test_withholding_is_correct_for_no_go_and_wrong_for_go(self):
        task = GoNoGo()
        response = task.complete_responses([np.nan, np.nan], [False, False])
        assert task.score(response, task.truth([-0.5, 0.5]), [False, False]).tolist() == [
            Outcome.CORRECT, Outcome.ERROR]


class TestRegistry:
    def test_round_trip(self):
        for task in (TwoAFC(b_side='negative', on_boundary='unscored'), GoNoGo(go_side='negative')):
            assert task_from_dict(task.to_dict()) == task

    def test_the_built_ins_are_listed(self):
        assert {'two_afc', 'go_nogo'} <= set(list_tasks())

    def test_an_unknown_type_names_the_known_ones(self):
        with pytest.raises(ValueError, match='two_afc'):
            task_from_dict({'type': 'nope'})

    def test_a_registered_task_can_be_named(self):
        @register_task
        @dataclasses.dataclass(frozen=True)
        class AlwaysB(Task):
            name = 'always_b_for_tests'

            @property
            def response_space(self):
                return ResponseSpace('binary', ('A', 'B'))

            def category(self, stimulus, context=None):
                return np.ones(len(stimulus))

            def truth(self, stimulus, context=None):
                return np.ones(len(stimulus))

        assert task_from_dict({'type': 'always_b_for_tests'}) == AlwaysB()
