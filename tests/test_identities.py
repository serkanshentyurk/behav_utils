"""Durable identities: canonical session ids, hashed trial keys, and session lists that name one session."""

import json

import numpy as np
import pandas as pd
import pytest

from behav_utils import TrialArrays, load_config, load_experiment, trial_key_of
from behav_utils.analysis import shared_trial_count
from behav_utils.config.schema import ConfigError

_CONFIG = r"""
project: {name: t}
reader:
  type: csv_folders
  data_dir: data
  behaviour_file: 'trials.csv'
  date_regex: '(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})(?:_(?P<hour>\d{2})(?P<minute>\d{2}))?'
  drop_last_row: false
  min_trials_per_file: 5
task:
  choice_mapping: {type: identity, no_response_value: -1}
columns:
  trial_number: {csv_name: Trial, dtype: int}
  stimulus: {csv_name: Stim, dtype: float}
  choice: {csv_name: Choice, dtype: float}
"""


def _write(root, folders, extra='', animals=('A01',), trials=tuple(range(1, 11))):
    """The same trials in every session folder of every animal, and the config beside them."""
    for animal in animals:
        for folder in folders:
            rng = np.random.default_rng(0)
            path = root / 'data' / animal / folder / 'trials.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'Trial': list(trials), 'Stim': rng.uniform(-1, 1, len(trials)),
                          'Choice': rng.integers(0, 2, len(trials))}).to_csv(path, index=False)
    root.mkdir(parents=True, exist_ok=True)
    config = root / 'config.yaml'
    config.write_text(_CONFIG + extra)
    return config


def _ids(exp, animal='A01'):
    return [s.session_id for s in exp.animals[animal].sessions]


def _keys(exp, animal):
    return TrialArrays.from_sessions(exp.animals[animal].sessions).trial_key


def test_ids_from_the_date_and_the_start_time(tmp_path):
    exp = load_experiment(_write(tmp_path, ['x_2024_03_15_1400', 'y_2024_03_15_0900', 'z_2024_03_14']))
    assert _ids(exp) == ['A01_20240314', 'A01_20240315T0900', 'A01_20240315T1400']
    assert [s.source_id for s in exp.animals['A01'].sessions] == ['z_2024_03_14', 'y_2024_03_15_0900',
                                                                  'x_2024_03_15_1400']


def test_same_day_sessions_without_times_are_numbered_with_a_warning(tmp_path):
    with pytest.warns(UserWarning, match='share A01_20240315'):
        exp = load_experiment(_write(tmp_path, ['b_2024_03_15', 'a_2024_03_15', 'c_2024_03_16']))
    assert _ids(exp) == ['A01_20240315_1', 'A01_20240315_2', 'A01_20240316']


def test_ignoring_a_session_never_renames_another(tmp_path):
    path = _write(tmp_path, ['a_2024_03_15', 'b_2024_03_15'], "sessions_to_ignore:\n  A01: ['20240315_1']\n")
    with pytest.warns(UserWarning):
        exp = load_experiment(path)
    assert _ids(exp) == ['A01_20240315_2']


def test_a_session_types_entry_can_name_one_session(tmp_path):
    extra = "session_types:\n  sham:\n    A01: ['20240315T0900']\n"
    exp = load_experiment(_write(tmp_path, ['x_2024_03_15_0900', 'y_2024_03_15_1400'], extra))
    assert [s.session_type for s in exp.animals['A01'].sessions] == ['sham', 'regular']


def test_a_date_entry_still_covers_the_whole_day(tmp_path):
    extra = "session_types:\n  sham:\n    A01: [20240315]\n"
    exp = load_experiment(_write(tmp_path, ['x_2024_03_15_0900', 'y_2024_03_15_1400'], extra))
    assert [s.session_type for s in exp.animals['A01'].sessions] == ['sham', 'sham']


def test_an_unquoted_suffix_entry_is_refused(tmp_path):
    with pytest.raises(ConfigError, match='Quote session entries'):
        load_config(_write(tmp_path, [], "sessions_to_ignore:\n  A01: [20240315_2]\n"))


def test_trial_keys_are_stable_across_loads_and_unique_across_animals(tmp_path):
    path = _write(tmp_path, ['s_2024_03_15', 's_2024_03_16'], animals=('A01', 'A02'))
    first, second = load_experiment(path), load_experiment(path)
    np.testing.assert_array_equal(_keys(first, 'A01'), _keys(second, 'A01'))
    assert not set(_keys(first, 'A01').tolist()) & set(_keys(first, 'A02').tolist())
    assert shared_trial_count(first.animals['A01'].sessions, first.animals['A02'].sessions) == 0


def test_ignoring_a_session_leaves_the_other_keys_alone(tmp_path):
    path = _write(tmp_path, ['s_2024_03_15', 's_2024_03_16'])
    full = load_experiment(path)
    config = load_config(path)
    config.sessions_to_ignore = {'A01': ['20240315']}
    part = load_experiment(config)
    np.testing.assert_array_equal(TrialArrays.from_sessions(full.animals['A01'].sessions[1:]).trial_key,
                                  _keys(part, 'A01'))


def test_the_experimenter_changes_the_keys_and_animal_metadata_overrides_it(tmp_path):
    path = _write(tmp_path, ['s_2024_03_15'], animals=('A01', 'A02'))
    (tmp_path / 'data' / 'animal_metadata.json').write_text(json.dumps({'A02': {'experimenter': 'KL'}}))
    plain = load_experiment(path)
    config = load_config(path)
    config.experimenter = 'SS'
    named = load_experiment(config)
    assert [a.sessions[0].experimenter for a in named.animals.values()] == ['SS', 'KL']
    assert plain.animals['A01'].sessions[0].session_hash != named.animals['A01'].sessions[0].session_hash


def test_repeated_trial_numbers_raise_when_strict_and_are_renumbered_otherwise(tmp_path):
    path = _write(tmp_path, ['s_2024_03_15'], trials=(1, 2, 3, 3, 4, 5, 6, 7, 8, 9))
    with pytest.raises(RuntimeError, match='repeat'):
        load_experiment(path)
    with pytest.warns(UserWarning, match='renumbered'):
        exp = load_experiment(path, strict=False)
    assert exp.animals['A01'].sessions[0].trials.trial_number.tolist() == list(range(1, 11))


def test_trial_numbers_beyond_the_key_range_raise(tmp_path):
    path = _write(tmp_path, ['s_2024_03_15'], trials=(1, 2, 3, 4, 5, 6, 7, 8, 9, 2 ** 20))
    with pytest.raises(RuntimeError, match='must lie in'):
        load_experiment(path)


def test_trial_key_of_puts_the_session_hash_above_the_trial_number(tmp_path):
    session = load_experiment(_write(tmp_path, ['s_2024_03_15'])).animals['A01'].sessions[0]
    keys = trial_key_of(session, [1, 2])
    assert keys.dtype == np.int64
    assert keys.tolist() == [(session.session_hash << 20) | 1, (session.session_hash << 20) | 2]
