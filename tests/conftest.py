"""
behav_utils test fixtures.

Provides synthetic animals, sessions, and trial data for testing
without requiring real data or cluster access.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from behav_utils.data.ops.selection import register_presets_from_config

# The library ships no presets; these mirror the shape a project config would register.
register_presets_from_config({'session_presets': {
    'expert_uniform': {'stage': 'Full_Task_Cont', 'distribution': 'Uniform', 'min_accuracy': 0.70,
                       'last_fraction': 0.50, 'exclude_types': ['masking', 'washout']},
    'all_uniform': {'stage': 'Full_Task_Cont', 'distribution': 'Uniform', 'exclude_types': ['masking', 'washout']},
    'naive_uniform': {'stage': 'Full_Task_Cont', 'distribution': 'Uniform', 'first_n': 5,
                      'exclude_types': ['masking', 'washout']},
    'all_stages': {'exclude_types': ['masking', 'washout']},
}})


@pytest.fixture
def rng():
    """Deterministic random generator."""
    return np.random.default_rng(42)


def _make_trial_data(n, rng, noise=0.20, opto_frac=0.0):
    """Helper: create TrialData with correct fields."""
    from behav_utils.data.structures import TrialData

    stimuli = rng.uniform(-1, 1, n)
    categories = (stimuli > 0).astype(float)
    choices = categories.copy()
    flip = rng.random(n) < noise
    choices[flip] = 1 - choices[flip]

    opto = np.zeros(n, dtype=bool)
    if opto_frac > 0:
        opto_idx = rng.choice(n, size=int(n * opto_frac), replace=False)
        opto[opto_idx] = True

    return TrialData(
        trial_number=np.arange(n),
        stimulus=stimuli,
        category=categories,
        choice=choices,
        outcome=(choices == categories).astype(float),
        correct=(choices == categories),
        abort=np.zeros(n, dtype=bool),
        opto_on=opto,
    )


def _make_session(session_idx, base_date, trials, distribution='Uniform',
                  stage='Full_Task_Cont', masking=False, washout=False):
    """Helper: create SessionData with correct fields."""
    from behav_utils.data.structures import SessionData, SessionMetadata

    if washout:
        session_type = 'washout'
    elif masking:
        session_type = 'masking'
    elif trials.opto_on is not None and bool(np.any(trials.opto_on)):
        session_type = 'opto'
    else:
        session_type = 'regular'
    return SessionData(
        session_id=f'sess_{session_idx:03d}',
        session_idx=session_idx,
        date=base_date + timedelta(days=session_idx),
        metadata=SessionMetadata(fields={'stage': stage, 'distribution': distribution}),
        trials=trials,
        session_type=session_type,
    )


@pytest.fixture
def synthetic_trial_data(rng):
    """Create a minimal TrialData with 200 trials."""
    return _make_trial_data(200, rng, noise=0.25)


@pytest.fixture
def synthetic_opto_trial_data(rng):
    """Create TrialData with 30% opto trials."""
    return _make_trial_data(300, rng, noise=0.20, opto_frac=0.3)


@pytest.fixture
def synthetic_session(synthetic_trial_data):
    """Create a minimal SessionData."""
    return _make_session(0, date(2026, 1, 1), synthetic_trial_data)


@pytest.fixture
def synthetic_animal(rng):
    """
    Create an AnimalData with 15 sessions.

    Sessions 0-9: Uniform, no opto
    Session 10-11: Uniform, opto (masking)
    Sessions 12-14: Uniform, opto (real)
    """
    from behav_utils.data.structures import AnimalData

    sessions = []
    base_date = date(2026, 1, 1)

    for i in range(15):
        noise = 0.30 - i * 0.01
        has_opto = i >= 10
        masking = i in (10, 11)

        trials = _make_trial_data(
            300, rng, noise=max(noise, 0.10),
            opto_frac=0.3 if has_opto else 0.0,
        )
        sess = _make_session(
            i, base_date, trials,
            distribution='Uniform', masking=masking,
        )
        sessions.append(sess)

    return AnimalData(animal_id='TEST01', sessions=sessions)


@pytest.fixture
def synthetic_opto_animal(rng):
    """
    Create an AnimalData with full opto timeline:
    0-4: Uniform baseline
    5-6: Uniform masking
    7-11: Uniform opto
    12-13: Uniform washout
    14-18: Asym_Right opto
    19-23: Asym_Right recovery
    """
    from behav_utils.data.structures import AnimalData

    # (idx, distribution, opto_frac, is_masking, is_washout)
    phases = (
        [(i, 'Uniform', 0.0, False, False) for i in range(5)] +
        [(i, 'Uniform', 0.3, True, False) for i in range(5, 7)] +
        [(i, 'Uniform', 0.3, False, False) for i in range(7, 12)] +
        [(i, 'Uniform', 0.0, False, True) for i in range(12, 14)] +
        [(i, 'Asym_Right', 0.3, False, False) for i in range(14, 19)] +
        [(i, 'Asym_Right', 0.0, False, False) for i in range(19, 24)]
    )

    sessions = []
    base_date = date(2026, 3, 1)

    for idx, dist, opto_frac, is_masking, is_washout in phases:
        trials = _make_trial_data(350, rng, noise=0.20, opto_frac=opto_frac)
        sess = _make_session(
            idx, base_date, trials,
            distribution=dist, masking=is_masking, washout=is_washout,
        )
        sessions.append(sess)

    return AnimalData(animal_id='OPTO01', sessions=sessions)
