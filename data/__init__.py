"""
behav_utils — Behavioural Neuroscience Data Utilities

Config-driven library for loading, analysing, and plotting
trial-based behavioural data.

Quick start:
    from behav_utils import load_experiment

    experiment = load_experiment('config.yaml')
    animal = experiment.get_animal('SS05')

    # Stats
    session.stats(['accuracy', 'recency'])
    animal.stat_trajectory('accuracy')

    # Plotting
    session.plot_psychometric()
    animal.plot_trajectory('accuracy')
    experiment.plot_trajectory('accuracy', combine='mean_sem')
"""

"""
behav_utils.data — Data Structures and Loading

Hierarchical containers, config-driven loading, synthetic generation.
"""

from behav_utils.data.structures import (
    ExperimentData,
    AnimalData,
    SessionData,
    SessionMetadata,
    TrialData,
)
from behav_utils.data.loading import (
    load_experiment,
    load_animal,
    load_session_csv,
)
from behav_utils.data.synthetic import (
    generate_synthetic_animal,
    generate_synthetic_session,
    sample_stimuli,
    random_choice_simulator,
    noisy_psychometric_simulator,
)
from behav_utils.data.neural import NeuralData, Epoch


__version__ = '0.1.0'

__all__ = [
    # Config
    'load_config',
    'ProjectConfig',

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

    # Neural (stub)
    'NeuralData',
    'Epoch',
]
