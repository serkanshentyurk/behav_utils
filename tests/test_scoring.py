"""Scoring at load: the session's task decides what is correct and recorded columns are only checked
against it; go/no-go blanks are withholds; task_by picks a task per session."""

import warnings

import numpy as np
import pandas as pd
import pytest

from behav_utils import (
    GoNoGo,
    Outcome,
    TwoAFC,
    load_config,
    load_experiment,
)
from behav_utils.config.schema import ConfigError
from behav_utils.data import filter_trial_data, session_from_arrays

_CONFIG = """
project: {name: t}
reader:
  type: csv_folders
  data_dir: data
  behaviour_file: 'trials.csv'
  drop_last_row: false
  min_trials_per_file: 5
task:
  primary_stimulus: stimulus
  primary_choice: choice
  choice_mapping: {type: identity, no_response_value: -1}
columns:
  trial_number: {csv_name: Trial, dtype: int}
  stimulus: {csv_name: Stim, dtype: float}
  choice: {csv_name: Choice, dtype: float}
  correct: {csv_name: Correct, dtype: bool, optional: true}
  outcome: {csv_name: Outcome, dtype: str, optional: true}
session_metadata:
  stage: {csv_name: Stage, dtype: str, optional: true}
"""

STIM = [-0.5, -0.2, 0.2, 0.5, -0.4, 0.4]
CHOICE = [0, 1, 1, 1, 0, -1]                  # the last trial has no response
DERIVED = [True, False, True, True, True, False]


def _table(stim=STIM, choice=CHOICE, **columns):
    return pd.DataFrame({'Trial': range(1, len(stim) + 1), 'Stim': stim, 'Choice': choice, **columns})


def _setup(root, sessions, extra='', config=_CONFIG):
    root.mkdir(parents=True, exist_ok=True)
    for folder, df in sessions.items():
        path = root / 'data' / 'A01' / folder / 'trials.csv'
        path.parent.mkdir(parents=True)
        df.to_csv(path, index=False)
    path = root / 'config.yaml'
    path.write_text(config + extra)
    return path


def _load_recording(path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        exp = load_experiment(path)
    return exp, [str(w.message) for w in caught]


def test_correct_comes_from_the_task_not_the_recorded_column(tmp_path):
    recorded = [True, True, True, True, True, False]          # trial 2 is wrong but recorded as correct
    with pytest.warns(UserWarning, match='recorded correct column disagrees .* on 1 trial'):
        exp = load_experiment(_setup(tmp_path, {'S_2024_03_15': _table(Correct=recorded)}))
    trials = exp.animals['A01'].sessions[0].trials
    assert trials.correct.tolist() == DERIVED
    assert trials.outcome.tolist() == [1, 0, 1, 1, 1, Outcome.NO_RESPONSE]
    assert trials.correct_recorded.tolist() == recorded


def test_raise_and_ignore(tmp_path):
    recorded = [True, True, True, True, True, False]
    with pytest.raises(RuntimeError, match='disagrees'):
        load_experiment(_setup(tmp_path / 'r', {'S_2024_03_15': _table(Correct=recorded)}, 'scoring: {on_mismatch: raise}\n'))
    _, messages = _load_recording(_setup(tmp_path / 'i', {'S_2024_03_15': _table(Correct=recorded)},
                                         'scoring: {on_mismatch: ignore}\n'))
    assert not [m for m in messages if 'disagrees' in m]


def test_recorded_outcomes_are_checked_through_the_label_map(tmp_path):
    labels = ['Correct', 'Correct', 'Correct', 'Correct', 'Correct', 'Miss']      # trial 2 is mislabelled
    extra = 'scoring:\n  outcome_labels: {Correct: correct, Incorrect: error, Miss: no_response}\n'
    with pytest.warns(UserWarning, match='recorded outcome column'):
        exp = load_experiment(_setup(tmp_path, {'S_2024_03_15': _table(Outcome=labels)}, extra))
    assert exp.animals['A01'].sessions[0].trials.outcome_recorded.tolist() == labels


def test_without_a_label_map_recorded_outcomes_are_kept_but_not_checked(tmp_path):
    exp, messages = _load_recording(_setup(tmp_path, {'S_2024_03_15': _table(Outcome=['x'] * 6)}))
    assert not [m for m in messages if 'disagrees' in m]
    assert exp.animals['A01'].sessions[0].trials.outcome_recorded.tolist() == ['x'] * 6


def test_an_optional_recorded_column_missing_from_a_file_is_not_checked(tmp_path):
    exp, messages = _load_recording(_setup(tmp_path, {'S_2024_03_15': _table()}))
    assert not [m for m in messages if 'disagrees' in m]
    assert exp.animals['A01'].sessions[0].trials.correct_recorded.size == 0


def test_go_nogo_blanks_are_withholds(tmp_path):
    config = _CONFIG.replace('task:\n', 'task:\n  type: go_nogo\n', 1)
    df = _table([-0.5, 0.5, -0.5, 0.5, -0.3, 0.3], [np.nan, 1, 1, np.nan, np.nan, 1])
    session = load_experiment(_setup(tmp_path, {'S_2024_03_15': df}, config=config)).animals['A01'].sessions[0]
    assert session.task == GoNoGo()
    assert session.trials.choice.tolist() == [0, 1, 1, 0, 0, 1]
    assert session.trials.outcome.tolist() == [Outcome.CORRECT, Outcome.CORRECT, Outcome.ERROR, Outcome.ERROR,
                                               Outcome.CORRECT, Outcome.CORRECT]


def test_task_by_picks_a_task_per_session(tmp_path):
    extra = 'tasks:\n  pretraining: {type: go_nogo}\ntask_by:\n  field: stage\n  values: {GNG: pretraining}\n'
    sessions = {'S_2024_03_15': _table(Stage=['GNG'] * 6), 'S_2024_03_16': _table(Stage=['2AFC'] * 6)}
    animal = load_experiment(_setup(tmp_path, sessions, extra)).animals['A01']
    assert [(s.task_name, s.task.name) for s in animal.sessions] == [('pretraining', 'go_nogo'), ('default', 'two_afc')]
    assert list(animal.session_table['task']) == ['go_nogo', 'two_afc']


def test_task_by_must_name_a_defined_task(tmp_path):
    with pytest.raises(ConfigError, match='pretrain'):
        load_config(_setup(tmp_path, {}, 'task_by:\n  field: stage\n  values: {GNG: pretrain}\n'))


def test_new_per_trial_fields_survive_filtering(tmp_path):
    exp = load_experiment(_setup(tmp_path, {'S_2024_03_15': _table(Correct=DERIVED)}))
    kept = filter_trial_data(exp.animals['A01'].sessions[0].trials, np.array([True, False, True, False, True, True]))
    assert kept.correct_recorded.tolist() == [True, True, True, False]
    assert kept.outcome.tolist() == [1, 1, 1, Outcome.NO_RESPONSE]


def test_synthetic_sessions_are_scored_by_their_task():
    session = session_from_arrays(np.array([-0.5, 0.5, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert session.task == TwoAFC()
    assert session.trials.outcome.tolist() == [Outcome.CORRECT, Outcome.ERROR, Outcome.ERROR]
    with pytest.raises(ValueError, match='categories disagree'):
        session_from_arrays(np.array([-0.5, 0.5]), np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    go_nogo = session_from_arrays(np.array([-0.5, 0.5]), np.array([np.nan, np.nan]), task=GoNoGo())
    assert go_nogo.trials.outcome.tolist() == [Outcome.CORRECT, Outcome.ERROR]
