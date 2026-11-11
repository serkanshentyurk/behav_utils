"""
Synthetic data: sessions, animals and experiments from any model, with every trial class.

``generate_synthetic_session`` draws a design (``make_design``), runs the model on it and adds event
times on a clock that counts seconds from midnight, like a rig's. ``generate_synthetic_animal`` strings
sessions over consecutive days; ``generate_synthetic_experiment`` builds animals with their own random
streams. The default model is ``PsychometricModel()``; any model (anything with ``simulate(design, rng)``)
replaces it. By default every trial class is present: responses, no-responses (where the task has them),
aborts and, with ``laser_fraction`` above 0, laser trials and the trials after them.

    animal = generate_synthetic_animal(PsychometricModel(sigma=0.25), n_sessions=10, laser_fraction=0.3, rng=1)
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping

import numpy as np

from behav_utils.data.structures import AnimalData, ExperimentData, SessionData
from behav_utils.data.synthetic import session_from_arrays
from behav_utils.models.base import _blank_is_a_response, make_design
from behav_utils.models.builtin import PsychometricModel
from behav_utils.models.simulate import _responses
from behav_utils.tasks import Task, TwoAFC
from behav_utils.tasks.distributions import StimulusDistribution

DEFAULT_START_DATE = date(2026, 1, 5)
DEFAULT_NO_RESPONSE_RATE = 0.02


@dataclass(frozen=True)
class Timing:
    """
    How a synthetic trial unfolds, in seconds. The session starts ``session_start`` after midnight. Each
    trial presents the stimulus ``stimulus_delay`` after it starts; a response follows after the reaction
    time (lognormal with median ``rt_median`` and log-sd ``rt_sigma``, unless the model gives its own), a
    trial without one waits out ``response_window``, and an abort ends ``abort_after`` after the stimulus.
    The next trial starts ``iti`` later.
    """
    session_start: float = 9 * 3600.0
    stimulus_delay: float = 0.5
    rt_median: float = 0.45
    rt_sigma: float = 0.35
    response_window: float = 2.0
    abort_after: float = 0.2
    iti: float = 2.0


def _events(task: Task, choice: np.ndarray, abort: np.ndarray, model_rt, timing: Timing,
            rng: np.random.Generator) -> tuple:
    """(events, reaction_time): trial start, stimulus onset and response times on the session's clock. A
    response time exists only for an overt response (a go, in a task where a blank is a withhold)."""
    n = len(choice)
    overt = ~np.isnan(choice) & ~abort
    if _blank_is_a_response(task):
        overt &= choice == 1
    drawn = timing.rt_median * np.exp(timing.rt_sigma * rng.standard_normal(n))
    rt = np.where(overt, drawn if model_rt is None else model_rt, np.nan)
    waited = np.where(abort, timing.abort_after, np.where(overt, rt, timing.response_window))
    duration = timing.stimulus_delay + waited + timing.iti
    start = timing.session_start + np.concatenate([[0.0], np.cumsum(duration[:-1])])
    stim_on = start + timing.stimulus_delay
    return {'t_trial_start': start, 't_stim_on': stim_on, 't_response': stim_on + rt}, rt


def generate_synthetic_session(
    model=None,
    *,
    task: Task | None = None,
    distribution: StimulusDistribution | None = None,
    n_trials: int = 300,
    label: str = '',
    laser_fraction: float = 0.0,
    max_laser_run: int | None = None,
    abort_rate: float = 0.05,
    no_response_rate: float | None = None,
    timing: Timing | None = None,
    animal_id: str = 'SYN01',
    stage: str = 'task',
    session_date: date | None = None,
    session_idx: int = 0,
    rng=None,
) -> SessionData:
    """
    One synthetic session: a design drawn by ``make_design``, the model's responses (``PsychometricModel()``
    if none is given), and event times on the clock ``'synthetic'``, scored by ``task`` (``TwoAFC()``).
    ``no_response_rate`` defaults to 0.02 where the task has no-response trials and to 0 where a blank is a
    withhold. ``rng`` is a seed or a ``numpy.random.Generator``.
    """
    rng = np.random.default_rng(rng)
    task = task if task is not None else TwoAFC()
    if no_response_rate is None:
        no_response_rate = 0.0 if _blank_is_a_response(task) else DEFAULT_NO_RESPONSE_RATE
    design = make_design(task, distribution, n_trials, label=label, laser_fraction=laser_fraction,
                         max_laser_run=max_laser_run, abort_rate=abort_rate, no_response_rate=no_response_rate,
                         rng=rng)
    choice, model_rt = _responses(model if model is not None else PsychometricModel(), design, rng)
    events, reaction_time = _events(task, choice, design.abort, model_rt, timing or Timing(), rng)
    return session_from_arrays(
        design.stimulus, choice, session_idx=session_idx, distribution=label or None, abort=design.abort,
        reaction_time=reaction_time, animal_id=animal_id, stage=stage, task=task, laser=design.laser,
        events=events, clock='synthetic', session_date=session_date or DEFAULT_START_DATE,
    )


def _per_session(value, n_sessions: int, name: str) -> list:
    if isinstance(value, (list, tuple)):
        if len(value) != n_sessions:
            raise ValueError(f'generate_synthetic_animal: {name} has {len(value)} entries for {n_sessions} sessions')
        return list(value)
    return [value] * n_sessions


def generate_synthetic_animal(
    model=None,
    n_sessions: int = 20,
    trials_per_session=300,
    *,
    animal_id: str = 'SYN01',
    task: Task | None = None,
    distribution=None,
    label='',
    stage: str = 'task',
    start_date: date | None = None,
    laser_fraction: float = 0.0,
    max_laser_run: int | None = None,
    abort_rate: float = 0.05,
    no_response_rate: float | None = None,
    timing: Timing | None = None,
    metadata: Mapping[str, Any] | None = None,
    rng=None,
) -> AnimalData:
    """
    ``n_sessions`` synthetic sessions on consecutive days from ``start_date``, as an ``AnimalData``.
    ``model``, ``distribution``, ``label`` and ``trials_per_session`` each take one value or a list with one
    per session (a learning curve, a switch of distribution). ``metadata`` goes to ``animal.metadata``.
    """
    rng = np.random.default_rng(rng)
    models = _per_session(model, n_sessions, 'model')
    distributions = _per_session(distribution, n_sessions, 'distribution')
    labels = _per_session(label, n_sessions, 'label')
    counts = _per_session(trials_per_session, n_sessions, 'trials_per_session')
    start = start_date or DEFAULT_START_DATE
    sessions = [
        generate_synthetic_session(
            models[i], task=task, distribution=distributions[i], n_trials=int(counts[i]), label=labels[i],
            laser_fraction=laser_fraction, max_laser_run=max_laser_run, abort_rate=abort_rate,
            no_response_rate=no_response_rate, timing=timing, animal_id=animal_id, stage=stage,
            session_date=start + timedelta(days=i), session_idx=i, rng=rng)
        for i in range(n_sessions)
    ]
    animal = AnimalData(animal_id=animal_id, sessions=sessions)
    animal.metadata.update(metadata or {})
    return animal


def generate_synthetic_experiment(animals: Mapping[str, Mapping[str, Any]], rng=None) -> ExperimentData:
    """
    Synthetic animals as an ``ExperimentData``. ``animals`` maps an animal id to the keyword arguments of
    ``generate_synthetic_animal`` (``model``, ``n_sessions``, ``metadata`` such as a group, ...). Each
    animal's random stream comes from ``rng`` and its own id, so adding or removing an animal leaves the
    others unchanged.
    """
    base = int(np.random.default_rng(rng).integers(2 ** 63))
    experiment = ExperimentData()
    for animal_id, kwargs in animals.items():
        clash = {'animal_id', 'rng'} & set(kwargs)
        if clash:
            raise ValueError(f'generate_synthetic_experiment: {animal_id}: {sorted(clash)} are set by the '
                             f'experiment (the key and its own stream)')
        stream = np.random.default_rng([base, zlib.crc32(str(animal_id).encode())])
        experiment.add_animal(generate_synthetic_animal(animal_id=str(animal_id), rng=stream, **kwargs))
    return experiment
