"""
behav_utils.tasks — what counts as correct.

A ``Task`` decides each trial's category, its correct response and how a response is scored, for
arrays of trials. Every session carries one, built from the config's ``task:`` block (or a named
alternative in ``tasks:``), and the loader scores every trial with it; recorded correct/outcome
columns are checked against that score, never used instead of it.

    TwoAFC(b_side='positive', on_boundary='A')     # 2-AFC: B on the positive side
    GoNoGo(go_side='positive')                     # go/no-go: a blank response is a withhold
    Outcome.CORRECT, Outcome.ABORT, ...            # the codes in TrialData.outcome

Add a task with a frozen dataclass subclass of ``Task`` and ``@register_task`` (see EXTENDING.md).
"""

from behav_utils.tasks.base import Outcome, ResponseSpace, Task, list_tasks, register_task, task_from_dict
from behav_utils.tasks.categorisation import GoNoGo, TwoAFC

__all__ = ['Task', 'TwoAFC', 'GoNoGo', 'Outcome', 'ResponseSpace', 'register_task', 'task_from_dict', 'list_tasks']
