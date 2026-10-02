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

from behav_utils.data.loading import (
    load_animal,
    load_experiment,
    load_session_csv,
)
from behav_utils.data.ops.filtering import (
    filter_trials,
    pool_arrays,
)
from behav_utils.data.ops.selection import (
    SessionFilter,
    list_presets,
    register_preset,
    register_presets_from_config,
    select_sessions,
)
from behav_utils.data.ops.switches import Block, block_after, block_before, find_blocks, find_switches
from behav_utils.data.structures import (
    AnimalData,
    ExperimentData,
    SessionData,
    SessionMetadata,
    TrialData,
)
from behav_utils.data.synthetic import (
    generate_synthetic_animal,
    generate_synthetic_session,
    noisy_psychometric_simulator,
    random_choice_simulator,
    sample_stimuli,
)

__all__ = [
    'random_choice_simulator', 'find_switches', 'find_blocks', 'block_before', 'block_after', 'Block',
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

    # Selection (session-level)
    'select_sessions',
    'SessionFilter',
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

