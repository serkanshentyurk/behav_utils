"""
Models and designs: the interface every behavioural model meets, and the run it is simulated on.

A ``Design`` is everything about a run except the responses: the task, the stimuli, the distribution
label, and which trials carry the laser, end in an abort or get no response. ``make_design`` draws one.
A model is anything with ``simulate(design, rng) -> Responses``; ``Model`` is the abstract class to
subclass, and ``TrialwiseModel`` runs the loop for models that learn from trial to trial, so they only
say how they choose and how they update. ``behav_utils.models.simulate`` turns a model and a design into
a ``SessionData`` scored by the task.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from behav_utils.tasks import Task, TwoAFC
from behav_utils.tasks.distributions import StimulusDistribution, Uniform


@dataclass(frozen=True)
class Responses:
    """What a model returns: a choice per trial (NaN for no response) and, optionally, reaction times in
    seconds."""
    choice: np.ndarray
    reaction_time: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class Design:
    """
    A run without its responses. ``stimulus`` is on the library's axis; ``laser``, ``abort`` and
    ``no_response`` are per-trial flags drawn before any model sees the run (all False if omitted);
    ``truth`` is the task's correct response per trial.
    """
    task: Task
    stimulus: np.ndarray
    label: str = ''
    laser: np.ndarray | None = None
    abort: np.ndarray | None = None
    no_response: np.ndarray | None = None

    def __post_init__(self):
        stimulus = np.asarray(self.stimulus, dtype=float).reshape(-1)
        n = len(stimulus)
        object.__setattr__(self, 'stimulus', stimulus)
        for name in ('laser', 'abort', 'no_response'):
            value = getattr(self, name)
            flags = np.zeros(n, dtype=bool) if value is None else np.asarray(value, dtype=bool).reshape(-1)
            if len(flags) != n:
                raise ValueError(f'Design: {name} has length {len(flags)}, expected {n}')
            object.__setattr__(self, name, flags)
        object.__setattr__(self, 'truth', self.task.truth(stimulus))

    @property
    def n_trials(self) -> int:
        return len(self.stimulus)


def _blank_is_a_response(task: Task) -> bool:
    """Whether the task reads a blank as a response (go/no-go: a withhold), so it has no no-response trials."""
    return not np.isnan(task.complete_responses(np.array([np.nan]), np.array([False]))[0])


def _cap_runs(flags: np.ndarray, max_run: int) -> np.ndarray:
    out = flags.copy()
    run = 0
    for i, flag in enumerate(out):
        run = run + 1 if flag else 0
        if run > max_run:
            out[i] = False
            run = 0
    return out


def make_design(
    task: Task | None = None,
    distribution: StimulusDistribution | None = None,
    n_trials: int = 300,
    *,
    label: str = '',
    laser_fraction: float = 0.0,
    max_laser_run: int | None = None,
    abort_rate: float = 0.0,
    no_response_rate: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Design:
    """
    Draw a run: ``n_trials`` stimuli from ``distribution`` (default ``Uniform()``) under ``task``
    (default ``TwoAFC()``). Each trial is independently a laser trial (``laser_fraction``; at most
    ``max_laser_run`` in a row if given), an abort (``abort_rate``) and, if not aborted, a no-response
    (``no_response_rate``). A task that reads a blank as a response (go/no-go) has no no-response trials.
    """
    task = task if task is not None else TwoAFC()
    distribution = distribution if distribution is not None else Uniform()
    rng = rng if rng is not None else np.random.default_rng()
    for name, rate in (('laser_fraction', laser_fraction), ('abort_rate', abort_rate),
                       ('no_response_rate', no_response_rate)):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f'make_design: {name} must lie in [0, 1], got {rate}')
    if max_laser_run is not None and max_laser_run < 1:
        raise ValueError(f'make_design: max_laser_run must be at least 1, got {max_laser_run}')
    if no_response_rate > 0 and _blank_is_a_response(task):
        raise ValueError(f'make_design: {task.name} reads a blank as a response (a withhold), so it has no '
                         f'no-response trials; set no_response_rate=0')
    stimulus = distribution.sample(n_trials, rng)
    laser = rng.random(n_trials) < laser_fraction
    if max_laser_run is not None:
        laser = _cap_runs(laser, max_laser_run)
    abort = rng.random(n_trials) < abort_rate
    no_response = ~abort & (rng.random(n_trials) < no_response_rate)
    return Design(task, stimulus, label, laser, abort, no_response)


class Model(ABC):
    """
    A behavioural model: ``simulate(design, rng)`` returns its ``Responses`` to a run.

    Anything with that method works with ``simulate()``; subclassing is optional. A model may look at
    the whole design, the laser flags included. ``simulate()`` blanks aborted and no-response trials
    afterwards, whatever the model returns for them.
    """

    @abstractmethod
    def simulate(self, design: Design, rng: np.random.Generator) -> Responses:
        """The model's responses to ``design``."""


class TrialwiseModel(Model):
    """
    A model that chooses trial by trial and may learn from feedback.

    Write ``p_choice``; ``update`` and ``reset`` are optional. ``simulate`` calls ``reset()``, then, for
    each trial that is neither aborted nor a no-response, draws a choice that is 1 with probability
    ``p_choice(t, stimulus, laser)`` and calls ``update(t, stimulus, choice, correct)``, ``correct``
    being the task's score of that choice (None where the task gives no score).
    """

    @abstractmethod
    def p_choice(self, t: int, stimulus: float, laser: bool) -> float:
        """The probability of response 1 on trial ``t``."""

    def update(self, t: int, stimulus: float, choice: float, correct: bool | None) -> None:
        """Learn from trial ``t``; the default learns nothing."""
        return None

    def reset(self) -> None:
        """Start of a run; the default keeps no state."""
        return None

    def simulate(self, design: Design, rng: np.random.Generator) -> Responses:
        self.reset()
        choice = np.full(design.n_trials, np.nan)
        skip = design.abort | design.no_response
        draws = rng.random(design.n_trials)
        for t in range(design.n_trials):
            if skip[t]:
                continue
            stimulus = float(design.stimulus[t])
            choice[t] = float(draws[t] < self.p_choice(t, stimulus, bool(design.laser[t])))
            truth = design.truth[t]
            self.update(t, stimulus, choice[t], None if np.isnan(truth) else bool(choice[t] == truth))
        return Responses(choice)
