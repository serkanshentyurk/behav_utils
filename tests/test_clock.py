"""Event times and the clock contract: t_* columns in seconds on the recording clock, units and
overrides, the checks, and events travelling with the trials."""

import warnings

import numpy as np
import pandas as pd
import pytest

from behav_utils import filter_trials, load_config, load_experiment
from behav_utils.config.schema import ConfigError

_CONFIG = """
project: {name: t}
reader:
  type: csv_folders
  data_dir: data
  behaviour_file: 'trials.csv'
  drop_last_row: false
  min_trials_per_file: 3
task:
  primary_stimulus: stimulus
  primary_choice: choice
  choice_mapping: {type: identity, no_response_value: -1}
columns:
  trial_number: {csv_name: Trial, dtype: int}
  stimulus: {csv_name: Stim, dtype: float}
  choice: {csv_name: Choice, dtype: float}
  reaction_time: {csv_name: RT, dtype: float}
  t_trial_start: {csv_name: Start, dtype: float}
  t_response: {csv_name: Lick, dtype: float, optional: true}
"""

START = [100.0, 105.0, 110.0, 115.0]
LICK = [100.6, 105.7, 110.5, 115.4]


def _setup(root, start=START, lick=LICK, rt=(0.6, 0.7, 0.5, 0.4), extra='', config=_CONFIG):
    df = pd.DataFrame({'Trial': [1, 2, 3, 4], 'Stim': [-0.6, 0.3, -0.2, 0.7], 'Choice': [0, 1, 0, 1],
                       'RT': list(rt), 'Start': list(start)})
    if lick is not None:
        df['Lick'] = list(lick)
    path = root / 'data' / 'A01' / 'S_2024_03_15' / 'trials.csv'
    path.parent.mkdir(parents=True)
    df.to_csv(path, index=False)
    config_path = root / 'config.yaml'
    config_path.write_text(config + extra)
    return config_path


def _session(path, **kwargs):
    return load_experiment(path, **kwargs).animals['A01'].sessions[0]


def _messages(path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        session = _session(path)
    return session, [str(w.message) for w in caught]


def test_event_times_are_seconds_on_the_named_clock(tmp_path):
    session, messages = _messages(_setup(tmp_path))
    assert messages == []
    assert session.clock == 'rig'
    assert set(session.trials.events) == {'t_trial_start', 't_response'}
    np.testing.assert_array_equal(session.trials.events['t_trial_start'], START)
    assert 't_response' not in session.trials.optional_fields and 'Lick' not in session.trials.extra


def test_milliseconds_and_a_per_column_override(tmp_path):
    config = _CONFIG.replace('reaction_time: {csv_name: RT, dtype: float}',
                             'reaction_time: {csv_name: RT, dtype: float, unit: s}')
    path = _setup(tmp_path, start=[t * 1000 for t in START], lick=[t * 1000 for t in LICK],
                  extra='clock: {name: bonsai, unit: ms}\n', config=config)
    session = _session(path)
    np.testing.assert_allclose(session.trials.events['t_trial_start'], START)
    np.testing.assert_allclose(session.trials.events['t_response'], LICK)
    np.testing.assert_allclose(session.trials.reaction_time, [0.6, 0.7, 0.5, 0.4])
    assert session.clock == 'bonsai'


def test_an_optional_event_missing_from_a_file_is_nan(tmp_path):
    session = _session(_setup(tmp_path, lick=None))
    assert np.isnan(session.trials.events['t_response']).all()


def test_a_required_event_missing_from_a_file_fails_the_session(tmp_path):
    config = _CONFIG.replace('t_trial_start: {csv_name: Start, dtype: float}', 't_trial_start: {csv_name: Begin, dtype: float}')
    with pytest.raises(RuntimeError, match="missing required columns \\['Begin'\\]"):
        _session(_setup(tmp_path, config=config))


def test_a_clock_restart_warns_with_the_trial(tmp_path):
    with pytest.warns(UserWarning, match='t_trial_start goes backwards at trial 3'):
        _session(_setup(tmp_path, start=[100.0, 105.0, 2.0, 7.0], lick=None))


def test_raise_and_ignore(tmp_path):
    with pytest.raises(RuntimeError, match='goes backwards'):
        _session(_setup(tmp_path / 'r', start=[100.0, 105.0, 2.0, 7.0], lick=None,
                        extra='clock: {on_problem: raise}\n'))
    _, messages = _messages(_setup(tmp_path / 'i', start=[100.0, 105.0, 2.0, 7.0], lick=None,
                                   extra='clock: {on_problem: ignore}\n'))
    assert not [m for m in messages if 'backwards' in m]


def test_a_negative_reaction_time_warns(tmp_path):
    with pytest.warns(UserWarning, match='1 negative reaction time'):
        _session(_setup(tmp_path, rt=(0.6, -0.1, 0.5, 0.4)))


def test_events_out_of_the_declared_order_warn(tmp_path):
    lick = [100.6, 104.0, 110.5, 115.4]                     # trial 2 licks before it starts
    with pytest.warns(UserWarning, match='t_response comes before t_trial_start on 1 trial'):
        _session(_setup(tmp_path, lick=lick, extra='clock: {order: [t_trial_start, t_response]}\n'))


def test_events_travel_with_the_trials(tmp_path):
    session = _session(_setup(tmp_path))
    kept = filter_trials([session], mask_fn=lambda s: np.array([False, True, False, True]), min_trials=1)[0]
    np.testing.assert_array_equal(kept.trials.events['t_trial_start'], [105.0, 115.0])
    np.testing.assert_array_equal(kept.trials.events['t_response'], [105.7, 115.4])
    assert kept.clock == 'rig'


@pytest.mark.parametrize('columns, message', [
    ("  stimulus: {csv_name: Stim, dtype: float, unit: ms}\n", 'only event times'),
    ("  t_label: {csv_name: Label, dtype: str}\n", 'must be numeric'),
    ("  t_x: {csv_name: X, dtype: float, unit: minutes}\n", 'unit must be one of'),
])
def test_time_column_settings_are_checked(tmp_path, columns, message):
    config = _CONFIG.replace('  stimulus: {csv_name: Stim, dtype: float}\n', columns if 'stimulus' in columns else
                             '  stimulus: {csv_name: Stim, dtype: float}\n' + columns)
    with pytest.raises(ConfigError, match=message):
        load_config(_setup(tmp_path, config=config))


def test_the_order_must_name_mapped_events(tmp_path):
    with pytest.raises(ConfigError, match='t_reward'):
        load_config(_setup(tmp_path, extra='clock: {order: [t_trial_start, t_reward]}\n'))
