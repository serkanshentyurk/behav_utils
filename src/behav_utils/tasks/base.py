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
    """How a trial was scored; ``TrialData.outcome`` holds these as small integers."""
    ERROR = 0          # responded, and not with the correct response
    CORRECT = 1        # responded with the correct response
    NO_RESPONSE = 2    # the response window passed with no response, where withholding is not an answer
    ABORT = 3          # the trial ended before the response window
    UNSCORED = 4       # responded, but the task defines no correct response for this trial


@dataclasses.dataclass(frozen=True)
class ResponseSpace:
    """What a response can be: ``kind`` is 'binary' (0/1), 'categorical' (0..K-1) or 'continuous';
    ``labels`` name the discrete values in order."""
    kind: str
    labels: tuple = ()


_REGISTRY: dict = {}


class Task(ABC):
    """
    A task: categories, the correct response and scoring, for arrays of trials.

    Subclasses are frozen dataclasses whose fields are the task's parameters (what a config sets),
    with a class-level ``name`` and the decorator ``register_task``. They implement ``category`` and
    ``truth``; ``complete_responses`` and ``score`` have defaults that suit most tasks.
    """
    name: ClassVar[str] = ''

    @property
    @abstractmethod
    def response_space(self) -> ResponseSpace:
        """What a response can be."""

    @abstractmethod
    def category(self, stimulus, context: Mapping[str, Any] | None = None) -> np.ndarray:
        """The stimulus class per trial as floats (NaN where it has none)."""

    @abstractmethod
    def truth(self, stimulus, context: Mapping[str, Any] | None = None) -> np.ndarray:
        """The correct response per trial as floats (NaN where none is correct)."""

    def complete_responses(self, response, abort) -> np.ndarray:
        """Responses as the task reads them. The default leaves them as they are; a task in which
        withholding is an answer fills it in here."""
        return np.asarray(response, dtype=float).copy()

    def score(self, response, truth, abort) -> np.ndarray:
        """Outcome codes (int8): abort, then no response, then no correct response (UNSCORED), then
        CORRECT where the response equals the truth and ERROR otherwise."""
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
        """The names a config may set for this task (its dataclass fields)."""
        return tuple(f.name for f in dataclasses.fields(cls)) if dataclasses.is_dataclass(cls) else ()

    def to_dict(self) -> dict:
        """``{'type': name, **parameters}``; ``task_from_dict`` rebuilds the task."""
        return {'type': self.name, **dataclasses.asdict(self)}


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
    """The registered task types."""
    return sorted(_REGISTRY)


def task_from_dict(d: Mapping[str, Any]) -> Task:
    """Rebuild a task from ``to_dict`` output."""
    d = dict(d)
    return _task_class(d.pop('type'))(**d)
