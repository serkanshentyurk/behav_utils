"""
behav_utils.data — data structures, loading, selection, filtering and synthetic data.

Hierarchical containers, config-driven loading, session selection,
trial-level filtering, and synthetic generation.

Pipeline:
    load_experiment(config)                            → ExperimentData (reader → build_session)
    select_sessions(animal, preset='expert')           → List[SessionData]
    filter_trials(sessions, trial_type='non_opto')     → List[SessionData]
    TrialArrays.from_sessions(clean)                   → TrialArrays (pool_arrays underneath)
"""

from behav_utils.data.arrays import TrialArrays, trial_key_of
from behav_utils.data.discovery import discover_sessions
from behav_utils.data.loading import (
    apply_session_type,
    build_session,
    load_animal,
    load_experiment,
    load_session_csv,
)
from behav_utils.data.ops.filtering import (
    build_mask,
    filter_session,
    filter_trial_data,
    filter_trials,
    get_arrays,
    opto_mask,
    pool_arrays,
)
from behav_utils.data.ops.selection import (
    SessionFilter,
    list_presets,
    preset_from_spec,
    register_preset,
    register_presets_from_config,
    select_sessions,
)
from behav_utils.data.ops.switches import Block, block_after, block_before, find_blocks, find_switches
from behav_utils.data.readers import RawSession, csv_folder_reader, is_empty_csv, table_reader
from behav_utils.data.store import StoreError, load_store, read_manifest, save_store
from behav_utils.data.structures import (
    AnimalData,
    ExperimentData,
    SessionData,
    SessionMetadata,
    SessionSummary,
    TrialData,
)
from behav_utils.data.synthetic import session_from_arrays

__all__ = [
    # Data structures
    'ExperimentData', 'AnimalData', 'SessionData', 'SessionMetadata', 'SessionSummary', 'TrialData',

    # Arrays
    'TrialArrays', 'trial_key_of',

    # Loading: readers find and read raw tables, build_session maps them
    'load_experiment', 'load_animal', 'load_session_csv', 'apply_session_type', 'build_session',
    'RawSession', 'csv_folder_reader', 'table_reader', 'is_empty_csv',

    # Selection (session-level)
    'select_sessions', 'SessionFilter',
    'register_preset', 'list_presets', 'register_presets_from_config', 'preset_from_spec',
    'find_switches', 'find_blocks', 'block_before', 'block_after', 'Block',

    # Filtering (trial-level)
    'filter_trials', 'pool_arrays', 'build_mask', 'opto_mask',
    'filter_session', 'filter_trial_data', 'get_arrays',

    # Sessions from arrays (synthetic generators: behav_utils.models)
    'session_from_arrays',
    # Discovery: every session before loading
    'discover_sessions',
    # The store: saving and reloading, all or in part
    'save_store', 'load_store', 'read_manifest', 'StoreError',
]
