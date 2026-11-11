"""
behav_utils.config — Project Configuration

Load a YAML config that maps your CSV columns to internal field names,
defines task parameters, and registers session filter presets.

Usage:
    from behav_utils.config import load_config

    config = load_config('config.yaml')
    experiment = load_experiment(config)

See ``behav_utils/config_presets/config_minimal.yaml`` for a starter template,
``config_presets/config_full_reference.yaml`` for every option, and
``docs/config_guide.md`` / ``docs/config_examples/`` for worked examples.
"""

from behav_utils.config.schema import (
    ClockConfig,
    ConfigError,
    ProjectConfig,
    ScoringConfig,
    TaskConfig,
    TaskSelector,
    load_cohorts,
    load_config,
)

__all__ = ['load_config', 'load_cohorts', 'ProjectConfig', 'TaskConfig', 'TaskSelector', 'ScoringConfig',
           'ClockConfig', 'ConfigError']
