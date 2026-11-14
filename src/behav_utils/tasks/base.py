"""
Tasks: what a trial asks of the animal, and what counts as correct.

A ``Task`` answers, for arrays of trials, which category each stimulus belongs to, what the correct
response is and how a response is scored. The loader, the synthetic generator and (from step 6)
``simulate()`` all ask the session's task, so correctness is defined in one place: by the task and the
config that sets its parameters. Recorded correct/outcome columns are only ever checked against it.

    task = TwoAFC(b_side='positive', on_boundary='A')
    category = task.category(stimulus)                            # 0 = A, 1 = B, NaN = none
    outcome = task.score(choice, task.truth(stimulus), abort)     # Outcome codes, one per trial

Tasks are frozen dataclasses registered by name (``register_task``), so a config or a store can name
them; ``to_dict`` and ``task_from_dict`` round-trip them. Everything is vectorised over trials. The
optional ``context`` (per-trial arrays such as a rule column) is accepted now and used by no built-in
task yet.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, ClassVar, Mapping

import numpy as np


class Outcome(IntEnum):
    """
    How a trial was scored; ``TrialData.outcome`` holds these as small integers.

    Attributes:
        ERROR: Responded, and not with the correct response.
        CORRECT: Responded with the correct response.
        NO_RESPONSE: The response window passed with no response, where withholding is not an answer.
        ABORT: The trial ended before the response window.
        UNSCORED: Responded, but the task defines no correct response for this trial.
    """
    ERROR = 0          # responded, and not with the correct response
    CORRECT = 1        # responded with the correct response
    NO_RESPONSE = 2    # the response window passed with no response, where withholding is not an answer
    ABORT = 3          # the trial ended before the response window
    UNSCORED = 4       # responded, but the task defines no correct response for this trial


@dataclasses.dataclass(frozen=True)
class ResponseSpace:
    """
    What a response can be.

    Attributes:
        kind: 'binary' (0 or 1), 'categorical' (0 to K−1) or 'continuous'.
        labels: Names of the discrete values, in order; empty for a continuous response.
    """
    kind: str
    labels: tuple = ()


_REGISTRY: dict = {}


class Task(ABC):
    """
    A task: categories, the correct response and scoring, for arrays of trials.

    Subclasses are frozen dataclasses whose fields are the task's parameters (what a config sets),
    with a class-level ``name`` and the decorator ``register_task``. They implement ``category`` and
    ``truth``; ``complete_responses`` and ``score`` have defaults that suit most tasks.

    ``capabilities`` names what the task provides (see ``CAPABILITIES``); statistics and readouts
    declare what they need, and ``require`` checks one against the other. The names are open: a task
    of your own can add its own, e.g. ``capabilities = TwoAFC.capabilities | {'cue'}``.

    Attributes:
        name: The task's name in a config's ``task.type`` (class-level).
        capabilities: What the task provides, as a frozenset of capability names (class-level).
    """
    name: ClassVar[str] = ''
    capabilities: ClassVar[frozenset] = frozenset()

    @property
    @abstractmethod
    def response_space(self) -> ResponseSpace:
        """What a response can be."""

    @abstractmethod
    def category(self, stimulus, context: Mapping[str, Any] | None = None) -> np.ndarray:
        """
        The stimulus class per trial.

        Args:
            stimulus: Stimuli on the library's axis.
            context: Session-level values a task may read, such as a contingency; None for none.

        Returns:
            The class per trial as floats, NaN where a stimulus has none.
        """

    @abstractmethod
    def truth(self, stimulus, context: Mapping[str, Any] | None = None) -> np.ndarray:
        """
        The correct response per trial.

        Args:
            stimulus: Stimuli on the library's axis.
            context: Session-level values a task may read, such as a contingency; None for none.

        Returns:
            The correct response per trial as floats, NaN where none is correct.
        """

    def complete_responses(self, response, abort, context: Mapping[str, Any] | None = None) -> np.ndarray:
        """
        Responses as the task reads them.

        The default leaves them as they are; a task in which withholding is an answer, such as go/no-go, fills it in
        here.

        Args:
            response: The responses as loaded, NaN where none was recorded.
            abort: Whether each trial was aborted.
            context: Session-level values a task may read, such as a contingency; None for none.

        Returns:
            The responses the task scores.
        """
        return np.asarray(response, dtype=float).copy()

    def score(self, response, truth, abort, context: Mapping[str, Any] | None = None) -> np.ndarray:
        """
        Outcome codes for each trial.

        In order: abort, then no response, then no correct response (``UNSCORED``), then ``CORRECT`` where the response
        equals the truth and ``ERROR`` otherwise. A task with another notion of correct, such as a tolerance around a
        continuous truth, overrides it.

        Args:
            response: The responses, completed by ``complete_responses``.
            truth: The correct responses, from ``truth``.
            abort: Whether each trial was aborted.
            context: Session-level values a task may read, such as a contingency; None for none.

        Returns:
            ``Outcome`` codes, as int8.
        """
        response = np.asarray(response, dtype=float)
        truth = np.asarray(truth, dtype=float)
        out = np.full(len(response), Outcome.ERROR, dtype=np.int8)
        out[response == truth] = Outcome.CORRECT
        out[np.isnan(truth)] = Outcome.UNSCORED
        out[np.isnan(response)] = Outcome.NO_RESPONSE
        out[np.asarray(abort, dtype=bool)] = Outcome.ABORT
        return out

    @classmethod
    def parameters(cls) -> tuple:
        """
        The names a config may set for this task: its dataclass fields.

        Returns:
            The field names.
        """
        return tuple(f.name for f in dataclasses.fields(cls)) if dataclasses.is_dataclass(cls) else ()

    def to_dict(self) -> dict:
        """
        The task as a plain dict, which ``task_from_dict`` turns back into the task.

        Returns:
            ``{'type': name, **parameters}``.
        """
        return {'type': self.name, **dataclasses.asdict(self)}


# The built-in capability names and what each promises. A task may declare others of its own.
CAPABILITIES = {
    'binary': 'responses are 0 or 1',
    'two_choices': 'both responses are active choices, as in 2-AFC',
    'withhold': 'response 0 is withholding, as in go/no-go',
    'stimulus_axis': 'a signed stimulus whose category is set by its side of the boundary at 0',
    'scored': 'the task defines a correct response',
    'k_choices': 'responses are one of K > 2 categories, coded 0 to K-1',
    'continuous': 'responses are continuous values on the stimulus scale',
}


class RequirementError(ValueError):
    """A statistic or readout was asked of a task that lacks a capability it needs."""


def require(task: Task, needs, what: str, instead: Mapping[str, str] | None = None) -> None:
    """
    Raise ``RequirementError`` unless ``task`` declares every capability in ``needs``.

    ``what`` names the statistic or readout. ``instead`` maps a capability to a counterpart to suggest
    when the task has it (``{'withhold': 'go_bias'}``). The message says what is missing, what the task
    has, the counterpart or that there is none, and flags a name no registered task declares (a typo?).

    Args:
        task: The task to check.
        needs: The capability names required.
        what: The statistic or readout asking, named in the message.
        instead: Maps a capability to a counterpart to suggest when the task has it; None for none.

    Raises:
        RequirementError: The task lacks one of ``needs``.
    """
    missing = set(needs) - set(task.capabilities)
    if not missing:
        return
    described = ', '.join(f'{c} ({CAPABILITIES[c]})' if c in CAPABILITIES else c for c in sorted(missing))
    has = ', '.join(sorted(task.capabilities)) or 'no capabilities'
    parts = [f'{what} needs {described}; {task.name or type(task).__name__} has {has}']
    counterpart = next((name for cap, name in (instead or {}).items() if cap in task.capabilities), None)
    declared = set(CAPABILITIES).union(*(cls.capabilities for cls in _REGISTRY.values()))
    unknown = sorted(missing - declared)
    if counterpart:
        parts.append(f'use {counterpart} instead')
    elif unknown:
        parts.append(f"no registered task declares {', '.join(unknown)} (a typo, or a task not yet registered?)")
    else:
        parts.append('there is no counterpart for this task')
    raise RequirementError('; '.join(parts))


def register_task(cls):
    """Class decorator: make a Task subclass known by its ``name``, so configs and stores can name it."""
    if not (isinstance(cls, type) and issubclass(cls, Task)) or not cls.name:
        raise TypeError('register_task expects a Task subclass with a name')
    _REGISTRY[cls.name] = cls
    return cls


def _task_class(name: str) -> type:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown task type {name!r}; known: {', '.join(sorted(_REGISTRY))}") from None


def list_tasks() -> list:
    """
    The registered task types.

    Returns:
        The names a config's ``task.type`` may take.
    """
    return sorted(_REGISTRY)


def task_from_dict(d: Mapping[str, Any]) -> Task:
    """
    Rebuild a task from ``to_dict`` output.

    Args:
        d: ``{'type': name, **parameters}``.

    Returns:
        The task.
    """
    d = dict(d)
    return _task_class(d.pop('type'))(**d)
