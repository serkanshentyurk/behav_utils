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

    # Direct function access
    from behav_utils.analysis import compute_summary_stats, fit_psychometric
    from behav_utils.plotting import plot_psychometric, plot_stat_trajectory
"""

from behav_utils.config.schema import load_config, ProjectConfig
from behav_utils.data.loading import load_experiment, load_session_csv, load_animal
from behav_utils.data.structures import (
    ExperimentData,
    AnimalData,
    SessionData,
    SessionMetadata,
    TrialData,
)
from behav_utils.data.neural import NeuralData, Epoch

# Synthetic data
from behav_utils.data.synthetic import (
    generate_synthetic_animal,
    generate_synthetic_session,
    sample_stimuli,
    noisy_psychometric_simulator,
)

# Analysis (available after migration)
try:
    from behav_utils.analysis import (
        compute_summary_stats,
        fit_psychometric,
        compute_update_matrix,
        build_feature_matrix,
        list_available_stats,
    )
except ImportError:
    pass

# Plotting (available after migration)
try:
    from behav_utils.plotting.styles import apply_style, COLOURS
except ImportError:
    pass

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

    # Synthetic data
    'generate_synthetic_animal',
    'generate_synthetic_session',
    'sample_stimuli',
    'noisy_psychometric_simulator',

    # Analysis
    'compute_summary_stats',
    'fit_psychometric',
    'compute_update_matrix',
    'build_feature_matrix',
    'list_available_stats',

    # Plotting
    'apply_style',
    'COLOURS',
]
