"""
behav_utils.config — Project Configuration

Load a YAML config that maps your CSV columns to internal field names,
defines task parameters, and registers session filter presets.

Usage:
    from behav_utils.config import load_config

    config = load_config('config.yaml')
    experiment = load_experiment(config)

See configs/config_minimal.yaml for a starter template,
or configs/config_full_reference.yaml for all available options.
"""

from behav_utils.config.schema import load_config, ProjectConfig

__all__ = ['load_config', 'ProjectConfig']
