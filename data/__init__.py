"""
behav_utils.data — Data Structures, Loading, Selection, and Filtering

Hierarchical containers, config-driven loading, session selection,
trial-level filtering, and synthetic generation.

Pipeline:
    load_experiment(config)                           → ExperimentData
    select_sessions(animal, preset='expert_uniform')  → List[SessionData]
    filter_trials(sessions, mask_fn)                  → List[SessionData]
    pool_arrays(filtered_sessions)                    → dict of arrays
"""

from behav_utils.data.structures import (
    ExperimentData,
    AnimalData,
    SessionData,
    SessionMetadata,
    TrialData,
    FittingData,
)
from behav_utils.data.loading import (
    load_experiment,
    load_animal,
    load_session_csv,
)
from behav_utils.data.selection import (
    select_sessions,
    SessionFilter,
    fitting_data_from_sessions,
    register_preset,
    list_presets,
    register_presets_from_config,
)
from behav_utils.data.filtering import (
    filter_trials,
    pool_arrays,
)
from behav_utils.data.synthetic import (
    generate_synthetic_animal,
    generate_synthetic_session,
    sample_stimuli,
    random_choice_simulator,
    noisy_psychometric_simulator,
)


__version__ = '0.1.0'

__all__ = [
    # Loading
    'load_experiment',
    'load_session_csv',
    'load_animal',

    # Data structures
    'ExperimentData',
    'AnimalData',
    'SessionData',
    'SessionMetadata',
    'TrialData',
    'FittingData',

    # Selection (session-level)
    'select_sessions',
    'SessionFilter',
    'fitting_data_from_sessions',
    'register_preset',
    'list_presets',
    'register_presets_from_config',

    # Filtering (trial-level)
    'filter_trials',
    'pool_arrays',

    # Synthetic
    'generate_synthetic_animal',
    'generate_synthetic_session',
    'sample_stimuli',
    'noisy_psychometric_simulator',
]
