"""
Synthetic Data Generation

Generates synthetic behavioural data for testing pipelines.
Model-agnostic: takes a simulator callable that produces choices
from stimuli and categories.

The library provides stimulus generation and data structure assembly.
The user provides the simulator (model-specific).

Usage:
    from behav_utils.data.synthetic import generate_synthetic_animal

    # With a custom simulator
    def my_simulator(stimuli, categories, rng, **kwargs):
        # Your model here
        noise = rng.normal(0, 0.2, len(stimuli))
        p_b = 1 / (1 + np.exp(-(stimuli + noise) * 5))
        choices = (rng.random(len(stimuli)) < p_b).astype(float)
        return choices

    animal, info = generate_synthetic_animal(
        simulator=my_simulator,
        n_sessions=20,
        trials_per_session=300,
    )

    # Without a simulator (random choices, for pipeline testing only)
    animal, info = generate_synthetic_animal(n_sessions=10)
"""

from datetime import date, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Union,
)

import numpy as np

from behav_utils.data.structures import (
    AnimalData,
    SessionData,
    SessionMetadata,
    TrialData,
)

# =============================================================================
# STIMULUS GENERATION
# =============================================================================

def sample_stimuli(
    n_trials: int,
    distribution: str = 'uniform',
    rng: np.random.Generator | None = None,
    stim_range: Tuple[float, float] = (-1.0, 1.0),
    boundary: float = 0.0,
    **dist_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate stimuli and derive categories.

    Args:
        n_trials: Number of trials
        distribution: 'uniform', 'exponential_left', 'exponential_right',
                      or any string your preprocessing handles
        rng: Random number generator
        stim_range: (min, max) stimulus values
        boundary: Category boundary
        **dist_kwargs: Extra args (e.g., exp_rate for exponential)

    Returns:
        (stimuli, categories) — both np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()

    lo, hi = stim_range

    if distribution == 'uniform':
        stimuli = rng.uniform(lo, hi, n_trials)

    elif distribution in ('exponential_left', 'exponential_right'):
        # Exponentially weighted toward one side of the boundary
        exp_rate = dist_kwargs.get('exp_rate', 2.0)
        stimuli = np.empty(n_trials)
        for i in range(n_trials):
            if rng.random() < 0.5:
                # Below boundary
                stimuli[i] = boundary - rng.exponential(1.0 / exp_rate)
            else:
                # Above boundary
                stimuli[i] = boundary + rng.exponential(1.0 / exp_rate)
        stimuli = np.clip(stimuli, lo, hi)

        # Shift concentration
        if distribution == 'exponential_left':
            stimuli = stimuli  # already symmetric, fine
        elif distribution == 'exponential_right':
            stimuli = -stimuli  # flip

    else:
        # Default to uniform for unknown distributions
        stimuli = rng.uniform(lo, hi, n_trials)

    categories = (stimuli > boundary).astype(int)
    return stimuli, categories


# =============================================================================
# DEFAULT SIMULATOR (random choices for pipeline testing)
# =============================================================================

def random_choice_simulator(
    stimuli: np.ndarray,
    categories: np.ndarray,
    rng: np.random.Generator,
    accuracy: float = 0.75,
    **kwargs,
) -> np.ndarray:
    """
    Simple simulator: choices match category with given accuracy.
    No learning, no serial dependence — purely for pipeline testing.

    Args:
        stimuli: Stimulus values
        categories: True categories (0/1)
        rng: Random number generator
        accuracy: Probability of correct choice

    Returns:
        choices: 0/1 array
    """
    n = len(stimuli)
    correct = rng.random(n) < accuracy
    choices = np.where(correct, categories, 1 - categories).astype(float)
    return choices


def noisy_psychometric_simulator(
    stimuli: np.ndarray,
    categories: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.3,
    lapse: float = 0.05,
    **kwargs,
) -> np.ndarray:
    """
    Simulator with a noisy psychometric curve. No trial history.
    Useful for testing psychometric fitting.

    Args:
        stimuli: Stimulus values
        categories: True categories
        rng: Random number generator
        sigma: Psychometric slope (lower = steeper)
        lapse: Symmetric lapse rate

    Returns:
        choices: 0/1 array
    """
    from scipy.stats import norm
    p_b = lapse + (1 - 2 * lapse) * norm.cdf(stimuli, 0, sigma)
    choices = (rng.random(len(stimuli)) < p_b).astype(float)
    return choices


# =============================================================================
# SESSION GENERATION
# =============================================================================

def session_from_arrays(
    stimuli: np.ndarray,
    choices: np.ndarray,
    categories: np.ndarray,
    session_idx: int = 0,
    distribution: str | None = None,
    abort: np.ndarray | None = None,
    reaction_time: np.ndarray | None = None,
    animal_id: str = 'SIM',
    stage: str = 'task',
    base_date: date | None = None,
    stim_range: Tuple[float, float] = (-1.0, 1.0),
) -> SessionData:
    """Assemble a SessionData from raw per-trial arrays.

    Shared construction used by ``generate_synthetic_session`` and by the SBI
    simulator. Building a real SessionData (rather than pooling arrays by hand)
    means ``prev_*`` are frozen by ``TrialData.__post_init__`` and pooling is
    identical to real sessions.

    Args:
        stimuli, choices, categories: Per-trial arrays, shape (T,). ``choices``
            may contain NaN for abort / non-response trials.
        session_idx: Ordinal session index (sets session_id and date offset).
        distribution: Optional distribution label, stored in metadata
            (best-effort; nothing in the SBI path depends on it).
        abort: Boolean abort mask. None -> all False (the simulated case).
        reaction_time: Per-trial RT. None -> all NaN (RT is unused by stats).
        animal_id, stage, base_date, stim_range: Metadata.

    Returns:
        SessionData.
    """
    stimuli = np.asarray(stimuli, dtype=float)
    choices = np.asarray(choices, dtype=float)
    categories = np.asarray(categories)
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
    session_date = base_date + timedelta(days=session_idx)

    # Derive correct (NaN choices -> not correct)
    correct = (choices == categories)
    correct[np.isnan(choices)] = False

    # Outcome strings
    outcome = np.where(
        abort, 'Abort',
        np.where(correct, 'Correct', 'Incorrect'),
    )

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
        opto_on=np.zeros(n_trials, dtype=bool),
    )

    fields = {
        'animal_id': animal_id,
        'stage': stage,
        'protocol': 'Synthetic',
        'sound_contingency': 'Low_Left_High_Right',
        'stim_range_min': stim_range[0],
        'stim_range_max': stim_range[1],
    }
    if distribution is not None:
        fields['distribution'] = distribution
    metadata = SessionMetadata(fields=fields)

    session_id = f'{animal_id}_S{session_idx:03d}'

    return SessionData(
        session_id=session_id,
        session_idx=session_idx,
        date=session_date,
        metadata=metadata,
        trials=trials,
    )


def generate_synthetic_session(
    session_idx: int = 0,
    n_trials: int = 300,
    distribution: str = 'uniform',
    stim_range: Tuple[float, float] = (-1.0, 1.0),
    boundary: float = 0.0,
    abort_rate: float = 0.05,
    animal_id: str = 'SYN01',
    stage: str = 'task',
    base_date: date | None = None,
    rng: np.random.Generator | None = None,
    simulator: Callable | None = None,
    simulator_kwargs: Dict[str, Any] | None = None,
    **dist_kwargs,
) -> SessionData:
    """
    Generate a single synthetic session.

    Args:
        session_idx: Ordinal session index
        n_trials: Number of trials
        distribution: Stimulus distribution type
        stim_range: Stimulus range
        boundary: Category boundary
        abort_rate: Fraction of abort trials
        animal_id: Animal identifier
        stage: Training stage
        base_date: Session date (default: today + session_idx days)
        rng: Random number generator
        simulator: Callable(stimuli, categories, rng, **kwargs) → choices.
                   If None, uses random_choice_simulator.
        simulator_kwargs: Extra kwargs for simulator
        **dist_kwargs: Extra kwargs for sample_stimuli

    Returns:
        SessionData with synthetic data
    """
    if rng is None:
        rng = np.random.default_rng()
    if simulator is None:
        simulator = random_choice_simulator
    if simulator_kwargs is None:
        simulator_kwargs = {}

    # Generate stimuli
    stimuli, categories = sample_stimuli(
        n_trials, distribution=distribution, rng=rng,
        stim_range=stim_range, boundary=boundary,
        **dist_kwargs,
    )

    # Generate aborts
    abort = rng.random(n_trials) < abort_rate

    # Generate choices via simulator, mark aborts as NaN
    choices = simulator(stimuli, categories, rng, **simulator_kwargs)
    choices[abort] = np.nan

    # Synthetic RT
    rt = np.abs(rng.normal(300, 100, n_trials))
    rt[abort] = np.nan

    return session_from_arrays(
        stimuli, choices, categories,
        session_idx=session_idx,
        distribution=distribution,
        abort=abort,
        reaction_time=rt,
        animal_id=animal_id,
        stage=stage,
        base_date=base_date,
        stim_range=stim_range,
    )


# =============================================================================
# ANIMAL GENERATION
# =============================================================================

def generate_synthetic_animal(
    animal_id: str = 'SYN01',
    n_sessions: int = 20,
    trials_per_session: Union[int, List[int]] = 300,
    distribution: str = 'uniform',
    stim_range: Tuple[float, float] = (-1.0, 1.0),
    boundary: float = 0.0,
    abort_rate: float = 0.05,
    stage: str = 'task',
    seed: int = 42,
    simulator: Callable | None = None,
    simulator_kwargs: Dict[str, Any] | None = None,
    per_session_simulator_kwargs: List[Dict[str, Any]] | None = None,
    distribution_schedule: List[str] | None = None,
) -> Tuple[AnimalData, Dict]:
    """
    Generate a synthetic animal with multiple sessions.

    The simulator is called once per session. To model learning
    trajectories (e.g., decreasing noise across sessions), pass
    per_session_simulator_kwargs with session-varying parameters.

    Args:
        animal_id: Animal identifier
        n_sessions: Number of sessions
        trials_per_session: Trials per session (int or list)
        distribution: Default stimulus distribution
        stim_range: Stimulus range
        boundary: Category boundary
        abort_rate: Fraction aborts
        stage: Training stage
        seed: Random seed
        simulator: Callable(stimuli, categories, rng, **kwargs) → choices
        simulator_kwargs: Default kwargs for simulator (all sessions)
        per_session_simulator_kwargs: List of kwargs dicts, one per session.
                                      Merged with simulator_kwargs (session overrides default).
        distribution_schedule: List of distribution names, one per session.
                               Overrides `distribution` per session.

    Returns:
        (AnimalData, generation_info)
    """
    rng = np.random.default_rng(seed)

    if isinstance(trials_per_session, int):
        tps = [trials_per_session] * n_sessions
    else:
        assert len(trials_per_session) == n_sessions
        tps = trials_per_session

    if simulator_kwargs is None:
        simulator_kwargs = {}

    sessions = []
    for s_idx in range(n_sessions):
        # Per-session kwargs
        sess_kwargs = dict(simulator_kwargs)
        if per_session_simulator_kwargs is not None:
            sess_kwargs.update(per_session_simulator_kwargs[s_idx])

        # Per-session distribution
        sess_dist = distribution
        if distribution_schedule is not None:
            sess_dist = distribution_schedule[s_idx]

        session = generate_synthetic_session(
            session_idx=s_idx,
            n_trials=tps[s_idx],
            distribution=sess_dist,
            stim_range=stim_range,
            boundary=boundary,
            abort_rate=abort_rate,
            animal_id=animal_id,
            stage=stage,
            rng=rng,
            simulator=simulator,
            simulator_kwargs=sess_kwargs,
        )
        sessions.append(session)

    animal = AnimalData(animal_id=animal_id, sessions=sessions)

    info = {
        'seed': seed,
        'n_sessions': n_sessions,
        'trials_per_session': tps,
        'distribution_schedule': distribution_schedule,
        'simulator': simulator.__name__ if simulator else 'random_choice_simulator',
    }

    return animal, info
