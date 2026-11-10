"""Session types: where they come from (a config list, a mapped column, the trials), which wins, and the
warning for values nobody declared."""

import warnings

import numpy as np
import pandas as pd
import pytest

from behav_utils import filter_trials, load_experiment

_CONFIG = """
project: {name: t}
reader:
  type: csv_folders
  data_dir: data
  behaviour_file: 'trials.csv'
  drop_last_row: false
  min_trials_per_file: 5
task:
  choice_mapping: {type: identity, no_response_value: -1}
columns:
  trial_number: {csv_name: Trial, dtype: int}
  stimulus: {csv_name: Stim, dtype: float}
  choice: {csv_name: Choice, dtype: float}
  opto_on: {csv_name: Laser, dtype: bool, optional: true, default: false}
"""

_MAPPED = "session_metadata:\n  session_type: {csv_name: SessionType, dtype: str}\n"


def _setup(tmp_path, sessions, extra=''):
    """``sessions``: {folder: (value of a SessionType column or None, laser on the first trial)}."""
    rng = np.random.default_rng(0)
    for folder, (stype, laser) in sessions.items():
        path = tmp_path / 'data' / 'A01' / folder / 'trials.csv'
        path.parent.mkdir(parents=True)
        df = pd.DataFrame({'Trial': range(1, 11), 'Stim': rng.uniform(-1, 1, 10),
                           'Choice': rng.integers(0, 2, 10), 'Laser': [laser] + [False] * 9})
        if stype is not None:
            df['SessionType'] = stype
        df.to_csv(path, index=False)
    config = tmp_path / 'config.yaml'
    config.write_text(_CONFIG + extra)
    return config


def _types(exp):
    return {s.source_id: (s.session_type, s.session_type_source) for s in exp.animals['A01'].sessions}


def _caught(text):
    """Load with every warning recorded; return those whose message contains ``text``."""
    def run(path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            exp = load_experiment(path)
        return exp, [w for w in caught if text in str(w.message)]
    return run


def test_derived_without_a_column_or_a_list(tmp_path):
    exp = load_experiment(_setup(tmp_path, {'S_2024_03_15': (None, False), 'S_2024_03_16': (None, True)}))
    assert _types(exp) == {'S_2024_03_15': ('regular', 'derived'), 'S_2024_03_16': ('opto', 'derived')}


def test_an_unmapped_session_type_column_is_ignored(tmp_path):
    exp = load_experiment(_setup(tmp_path, {'S_2024_03_15': ('sham', False)}))
    assert _types(exp) == {'S_2024_03_15': ('regular', 'derived')}


def test_a_mapped_column_sets_the_type(tmp_path):
    exp = load_experiment(_setup(tmp_path, {'S_2024_03_15': ('regular', True)}, _MAPPED))
    assert _types(exp) == {'S_2024_03_15': ('regular', 'column')}


def test_a_config_list_overrides_the_column_with_a_warning(tmp_path):
    extra = _MAPPED + "session_types:\n  sham:\n    A01: [20240315]\n"
    with pytest.warns(UserWarning, match="column says 'regular'"):
        exp = load_experiment(_setup(tmp_path, {'S_2024_03_15': ('regular', True)}, extra))
    assert _types(exp) == {'S_2024_03_15': ('sham', 'config')}


def test_a_config_list_over_a_derived_type_is_silent(tmp_path):
    extra = "session_types:\n  sham:\n    A01: [20240315]\n"
    exp, caught = _caught('session_types')(_setup(tmp_path, {'S_2024_03_15': (None, True)}, extra))
    assert caught == []
    assert _types(exp) == {'S_2024_03_15': ('sham', 'config')}


def test_two_lists_naming_one_session_warn_and_the_later_wins(tmp_path):
    extra = "session_types:\n  sham:\n    A01: [20240315]\n  washout:\n    A01: [20240315]\n"
    with pytest.warns(UserWarning, match="both 'sham' and 'washout'"):
        exp = load_experiment(_setup(tmp_path, {'S_2024_03_15': (None, False)}, extra))
    assert _types(exp) == {'S_2024_03_15': ('washout', 'config')}


def test_an_undeclared_column_value_warns_once_per_value(tmp_path):
    sessions = {'S_2024_03_15': ('washot', False), 'S_2024_03_16': ('washot', False)}
    exp, caught = _caught('neither')(_setup(tmp_path, sessions, _MAPPED))
    assert len(caught) == 1 and "'washot' (2 session(s)" in str(caught[0].message)


def test_declaring_a_type_without_dates_silences_the_warning(tmp_path):
    extra = _MAPPED + "session_types:\n  washout: {}\n"
    exp, caught = _caught('neither')(_setup(tmp_path, {'S_2024_03_15': ('washout', False)}, extra))
    assert caught == []
    assert _types(exp) == {'S_2024_03_15': ('washout', 'column')}


def test_the_source_survives_filtering_and_shows_in_the_session_table(tmp_path):
    extra = "session_types:\n  sham:\n    A01: [20240315]\n"
    exp = load_experiment(_setup(tmp_path, {'S_2024_03_15': (None, False), 'S_2024_03_16': (None, False)}, extra))
    animal = exp.animals['A01']
    assert list(animal.session_table['session_type_source']) == ['config', 'derived']
    assert [s.session_type_source for s in filter_trials(animal.sessions, min_trials=1)] == ['config', 'derived']
