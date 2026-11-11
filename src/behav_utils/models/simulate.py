"""Running a model on a design: a scored SessionData, or just the choices."""

from __future__ import annotations

from datetime import date

import numpy as np

from behav_utils.data.structures import SessionData
from behav_utils.data.synthetic import session_from_arrays
from behav_utils.models.base import Design
from behav_utils.tasks import Task, TwoAFC, require


def simulate(
    model,
    design: Design,
    rng: np.random.Generator | None = None,
    *,
    session_idx: int = 0,
    animal_id: str = 'SIM',
    stage: str = 'task',
    base_date: date | None = None,
) -> SessionData:
    """
    Run ``model`` (anything with ``simulate(design, rng)``) on ``design``: a SessionData scored by the
    design's task. Aborted and no-response trials have no choice whatever the model returned; the laser
    flags become ``trials.opto_on``.
    """
    choice, reaction_time = _responses(model, design, rng if rng is not None else np.random.default_rng())
    return session_from_arrays(
        design.stimulus, choice, session_idx=session_idx, distribution=design.label or None,
        abort=design.abort, reaction_time=reaction_time, animal_id=animal_id, stage=stage,
        base_date=base_date, task=design.task, laser=design.laser,
    )


def _responses(model, design: Design, rng: np.random.Generator) -> tuple:
    """(choice, reaction_time or None): the model's responses, blanked on aborted and no-response trials."""
    if not callable(getattr(model, 'simulate', None)):
        raise TypeError(f'simulate: {type(model).__name__} has no simulate(design, rng) method')
    require(design.task, getattr(model, 'requires', frozenset()), type(model).__name__)
    responses = model.simulate(design, rng)
    choice = np.asarray(responses.choice, dtype=float).copy()
    if len(choice) != design.n_trials:
        raise ValueError(f'simulate: {type(model).__name__} returned {len(choice)} choices for '
                         f'{design.n_trials} trials')
    choice[design.abort | design.no_response] = np.nan
    rt = None if responses.reaction_time is None else np.asarray(responses.reaction_time, dtype=float)
    return choice, rt


def simulate_responses(model, stimuli, task: Task | None = None, rng: np.random.Generator | None = None,
                       laser=None) -> np.ndarray:
    """Just the model's choices to ``stimuli`` (no aborts, no session): the fast path for many runs,
    such as a training set for simulation-based inference."""
    design = Design(task if task is not None else TwoAFC(), stimuli, laser=laser)
    require(design.task, getattr(model, 'requires', frozenset()), type(model).__name__)
    return np.asarray(model.simulate(design, rng if rng is not None else np.random.default_rng()).choice,
                      dtype=float)
