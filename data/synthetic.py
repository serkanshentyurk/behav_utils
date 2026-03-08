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

import numpy as np
from datetime import date, timedelta
from typing import (
    Optional, List, Dict, Tuple, Callable, Union, Any,
)

from behav_utils.data.structures import (
    ExperimentData, AnimalData, SessionData, SessionMetadata, TrialData,
)


# =============================================================================
# STIMULUS GENERATION
# =============================================================================

def sample_stimuli(
    n_trials: int,
    distribution: str = 'uniform',
    rng: Optional[np.random.Generator] = None,
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

def generate_synthetic_session(
    session_idx: int = 0,
    n_trials: int = 300,
    distribution: str = 'uniform',
    stim_range: Tuple[float, float] = (-1.0, 1.0),
    boundary: float = 0.0,
    abort_rate: float = 0.05,
    animal_id: str = 'SYN01',
    stage: str = 'Full_Task_Cont',
    base_date: Optional[date] = None,
    rng: Optional[np.random.Generator] = None,
    simulator: Optional[Callable] = None,
    simulator_kwargs: Optional[Dict[str, Any]] = None,
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
    if base_date is None:
        base_date = date(2025, 1, 1)

    session_date = base_date + timedelta(days=session_idx)

    # Generate stimuli
    stimuli, categories = sample_stimuli(
        n_trials, distribution=distribution, rng=rng,
        stim_range=stim_range, boundary=boundary,
        **dist_kwargs,
    )

    # Generate aborts
    abort = rng.random(n_trials) < abort_rate

    # Generate choices via simulator
    choices = simulator(stimuli, categories, rng, **simulator_kwargs)

    # Mark aborts as NaN
    choices[abort] = np.nan

    # Derive correct
    correct = (choices == categories)
    correct[np.isnan(choices)] = False

    # Outcome strings
    outcome = np.where(
        abort, 'Abort',
        np.where(correct, 'Correct', 'Incorrect'),
    )

    # Synthetic RT
    rt = np.abs(rng.normal(300, 100, n_trials))
    rt[abort] = np.nan

    # Build TrialData
    trials = TrialData(
        trial_number=np.arange(1, n_trials + 1),
        stimulus=stimuli,
        choice=choices,
        choice_raw=choices.copy(),
        outcome=outcome,
        correct=correct,
        category=categories,
        reaction_time=rt,
        abort=abort,
        opto_on=np.zeros(n_trials, dtype=bool),
    )

    # Metadata
    metadata = SessionMetadata(fields={
        'animal_id': animal_id,
        'stage': stage,
        'protocol': 'Synthetic',
        'sound_contingency': 'Low_Left_High_Right',
        'stim_range_min': stim_range[0],
        'stim_range_max': stim_range[1],
    })

    session_id = f'{animal_id}_S{session_idx:03d}'

    return SessionData(
        session_id=session_id,
        session_idx=session_idx,
        date=session_date,
        metadata=metadata,
        trials=trials,
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
    stage: str = 'Full_Task_Cont',
    seed: int = 42,
    simulator: Optional[Callable] = None,
    simulator_kwargs: Optional[Dict[str, Any]] = None,
    per_session_simulator_kwargs: Optional[List[Dict[str, Any]]] = None,
    distribution_schedule: Optional[List[str]] = None,
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
