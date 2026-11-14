"""
Building a SessionData from per-trial arrays.

``session_from_arrays`` is the one constructor for sessions that do not come from files: the models'
``simulate`` and the synthetic generators use it, and so can a simulator of your own. The task scores the
trials, and the lag-1 view is frozen exactly as on loading.

Synthetic sessions, animals and experiments from a model are in ``behav_utils.models``
(``generate_synthetic_session``, ``generate_synthetic_animal``, ``generate_synthetic_experiment``).
"""

from datetime import date, timedelta
from typing import Mapping, Tuple

import numpy as np

from behav_utils.data.structures import SessionData, SessionMetadata, TrialData
from behav_utils.tasks import Outcome, Task, TwoAFC

__all__ = ['session_from_arrays']


def session_from_arrays(
    stimuli: np.ndarray,
    choices: np.ndarray,
    categories: np.ndarray | None = None,
    session_idx: int = 0,
    distribution: str | None = None,
    abort: np.ndarray | None = None,
    reaction_time: np.ndarray | None = None,
    animal_id: str = 'SIM',
    stage: str = 'task',
    base_date: date | None = None,
    stim_range: Tuple[float, float] = (-1.0, 1.0),
    task: Task | None = None,
    laser: np.ndarray | None = None,
    events: Mapping[str, np.ndarray] | None = None,
    clock: str = '',
    session_date: date | None = None,
) -> SessionData:
    """Assemble a SessionData from raw per-trial arrays, scored by ``task`` (default ``TwoAFC()``).

    The one constructor for sessions that do not come from files: ``simulate`` and the synthetic
    generators use it, and so can a simulator of your own. Building a real SessionData (rather than
    pooling arrays by hand) means ``prev_*`` are frozen by ``TrialData.__post_init__`` and pooling is
    identical to loaded sessions.

    Args:
        stimuli: Per-trial stimuli on the library's axis, shape (T,).
        choices: Per-trial responses, shape (T,); NaN for aborted and no-response trials.
        categories: The task decides categories; pass None, or arrays that must agree with
            ``task.category(stimuli)`` (a check, not an override).
        session_idx: Ordinal session index (the date is ``base_date`` plus this many days, unless
            ``session_date`` is given). The session id is ``<animal_id>_<YYYYMMDD>``, as on loading.
        distribution: Optional distribution label, stored with the trials and in the metadata.
        abort: Boolean abort mask; None for no aborts.
        reaction_time: Per-trial reaction times in seconds; None for all NaN.
        animal_id: The animal the session belongs to.
        stage: The session's stage, in its metadata.
        base_date: The date of session 0; None for 1 January 2025.
        stim_range: The raw stimulus range, recorded in the metadata.
        task: The Task that scores the trials and that the session carries.
        laser: Boolean laser mask (``trials.opto_on``). None -> all False. Any laser trial makes the
            session type 'opto' (else 'regular'), the loader's rule.
        events: Event times by name (``trials.events``), seconds on ``clock``.
        clock: The clock the event times are on (``SessionData.clock``).
        session_date: The session's date, overriding ``base_date`` + ``session_idx``.

    Returns:
        The session, scored by ``task``.
    """
    stimuli = np.asarray(stimuli, dtype=float)
    choices = np.asarray(choices, dtype=float)
    categories = None if categories is None else np.asarray(categories)
    n_trials = len(stimuli)

    if abort is None:
        abort = np.zeros(n_trials, dtype=bool)
    else:
        abort = np.asarray(abort, dtype=bool)

    if reaction_time is None:
        reaction_time = np.full(n_trials, np.nan)
    else:
        reaction_time = np.asarray(reaction_time, dtype=float)

    if base_date is None:
        base_date = date(2025, 1, 1)
    if session_date is None:
        session_date = base_date + timedelta(days=session_idx)
    opto_on = np.zeros(n_trials, dtype=bool) if laser is None else np.asarray(laser, dtype=bool)

    # The task decides categories and scores the trials; given categories are only checked
    task = task if task is not None else TwoAFC()
    task_categories = task.category(stimuli)
    if categories is not None:
        given = np.asarray(categories, dtype=float)
        both = ~np.isnan(given) & ~np.isnan(task_categories)
        if np.any(given[both] != task_categories[both]):
            raise ValueError("session_from_arrays: categories disagree with the task's categories for these "
                             "stimuli; the task decides them (pass categories=None)")
    categories = task_categories
    choices = task.complete_responses(choices, abort)
    outcome = task.score(choices, task.truth(stimuli), abort)
    correct = outcome == Outcome.CORRECT

    trials = TrialData(
        trial_number=np.arange(1, n_trials + 1),
        stimulus=stimuli,
        choice=choices,
        choice_raw=choices.copy(),
        outcome=outcome,
        correct=correct,
        category=categories,
        reaction_time=reaction_time,
        abort=abort,
        opto_on=opto_on,
        events={k: np.asarray(v, dtype=float) for k, v in (events or {}).items()},
    )

    fields = {
        'animal_id': animal_id,
        'stage': stage,
        'protocol': 'Synthetic',
        'response_mapping': 'A_left_B_right',
        'stim_range_min': stim_range[0],
        'stim_range_max': stim_range[1],
    }
    if distribution is not None:
        fields['distribution'] = distribution
    metadata = SessionMetadata(fields=fields)

    session_id = f'{animal_id}_{session_date:%Y%m%d}'

    return SessionData(
        session_id=session_id,
        session_idx=session_idx,
        date=session_date,
        metadata=metadata,
        trials=trials,
        task=task,
        session_type='opto' if opto_on.any() else 'regular',
        clock=clock,
    )
