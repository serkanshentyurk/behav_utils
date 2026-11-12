"""
behav_utils.tasks — what counts as correct.

A ``Task`` decides each trial's category, its correct response and how a response is scored, for
arrays of trials. Every session carries one, built from the config's ``task:`` block (or a named
alternative in ``tasks:``), and the loader scores every trial with it; recorded correct/outcome
columns are checked against that score, never used instead of it.

    TwoAFC(b_side='positive', on_boundary='A')     # 2-AFC: B on the positive side
    GoNoGo(go_side='positive')                     # go/no-go: a blank response is a withhold
    Reproduction(tolerance=0.1)                    # continuous: correct within the tolerance of the stimulus
    Outcome.CORRECT, Outcome.ABORT, ...            # the codes in TrialData.outcome

Add a task with a frozen dataclass subclass of ``Task`` and ``@register_task`` (see EXTENDING.md).

Each task declares ``capabilities``; statistics and readouts declare what they need, and ``require``
raises ``RequirementError`` (naming any counterpart) when a task lacks it.

Stimulus distributions (``Uniform``, ``TruncatedNormal``, ``Discrete``, or a ``StimulusDistribution``
subclass) say how a design draws stimuli and what an observer believes about them.
"""

from behav_utils.tasks.base import (
    CAPABILITIES,
    Outcome,
    RequirementError,
    ResponseSpace,
    Task,
    list_tasks,
    register_task,
    require,
    task_from_dict,
)
from behav_utils.tasks.categorisation import GoNoGo, TwoAFC
from behav_utils.tasks.continuous import Reproduction
from behav_utils.tasks.distributions import Discrete, StimulusDistribution, TruncatedNormal, Uniform

__all__ = ['Task', 'TwoAFC', 'GoNoGo', 'Reproduction', 'Outcome', 'ResponseSpace', 'register_task', 'task_from_dict', 'list_tasks',
           'StimulusDistribution', 'Uniform', 'TruncatedNormal', 'Discrete',
           'CAPABILITIES', 'RequirementError', 'require']
