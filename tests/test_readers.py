"""Readers and the loader around them: the table reader against the golden values, custom readers,
ignored sessions that are never read, several sessions on one day, and the loader's strict rules."""

import json
import os
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from behav_utils import load_config, load_experiment
from behav_utils.config.schema import ConfigError, TableReaderConfig
from behav_utils.data import RawSession, table_reader

from .golden import outputs as golden

_CONFIG = r"""
project: {name: t}
reader:
  type: csv_folders
  data_dir: data
  behaviour_file: 'trials*.csv'
  date_regex: '(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})(?:_(?P<hour>\d{2})(?P<minute>\d{2}))?'
  drop_last_row: false
  min_trials_per_file: 5
task:
  choice_mapping: {type: identity, no_response_value: -1}
columns:
  trial_number: {csv_name: Trial, dtype: int}
  stimulus: {csv_name: Stim, dtype: float}
  choice: {csv_name: Choice, dtype: float}
session_metadata:
  animal_id: {csv_name: Mouse, dtype: str}
"""


def _session_csv(path, n=8, seed=0, mouse='A01'):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Trial': np.arange(1, n + 1), 'Stim': rng.uniform(-1, 1, n),
                  'Choice': rng.integers(0, 2, n), 'Mouse': mouse}).to_csv(path, index=False)


def _setup(root, sessions, config=_CONFIG):
    """Write ``{relative CSV path: _session_csv kwargs}`` under root/data and the config beside it."""
    root.mkdir(parents=True, exist_ok=True)
    for rel, kwargs in sessions.items():
        _session_csv(root / 'data' / rel, **kwargs)
    path = root / 'config.yaml'
    path.write_text(config)
    return path


def _golden_table() -> pd.DataFrame:
    frames = []
    for csv in sorted(golden.DATA_DIR.glob('*/*/trials.csv')):
        df = pd.read_csv(csv)
        df.insert(0, 'Session', csv.parent.name)
        df.insert(1, 'Date', '-'.join(csv.parent.name.split('_')[-3:]))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize('suffix', ['csv', 'parquet'])
def test_the_golden_dataset_as_one_table_gives_the_golden_values(tmp_path, suffix):
    if suffix == 'parquet':
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip('pyarrow is not installed')
    path = tmp_path / f'golden.{suffix}'
    table = _golden_table()
    if suffix == 'csv':
        table.to_csv(path, index=False)
    else:
        table.to_parquet(path)
    config = load_config(golden.CONFIG)
    config.reader = TableReaderConfig(path=str(path), animal_column='Mouse', session_column='Session',
                                      date_column='Date')
    expected = golden.read_expected()
    names = expected['stats_pooled']['stat'].unique().tolist()
    actual = golden.compute_outputs(stat_names=names, experiment=load_experiment(config))
    for name in golden.TABLES:
        assert golden.compare_table(name, expected[name], actual[name], golden.versions_match()) == [], name


def test_a_custom_reader_is_any_callable(tmp_path):
    config = load_config(_setup(tmp_path, {}))

    def reader(config):
        rng = np.random.default_rng(0)
        for i, day in enumerate((date(2024, 3, 15), date(2024, 3, 16))):
            df = pd.DataFrame({'Trial': range(1, 11), 'Stim': rng.uniform(-1, 1, 10),
                               'Choice': rng.integers(0, 2, 10), 'Mouse': 'X1'})
            yield RawSession(animal_id='X1', session_id=f's{i}', date=day, trials=df)

    exp = load_experiment(config, reader=reader)
    assert [s.session_id for s in exp.animals['X1'].sessions] == ['s0', 's1']


def test_an_ignored_session_is_never_read(tmp_path):
    path = _setup(tmp_path, {'A01/S_2024_03_15/trials.csv': {}})
    broken = tmp_path / 'data' / 'A01' / 'S_2024_03_16' / 'trials.csv'
    broken.parent.mkdir()
    broken.write_text('a,b\n1,2\n3,4,5,6\n')
    config = load_config(path)
    with pytest.raises(RuntimeError, match='S_2024_03_16'):
        load_experiment(config)                       # read, and it fails
    config.sessions_to_ignore = {'A01': ['20240316']}
    exp = load_experiment(config)                     # ignored, so never read
    assert [s.date for s in exp.animals['A01'].sessions] == [date(2024, 3, 15)]


def test_same_day_sessions_sort_by_start_time(tmp_path):
    path = _setup(tmp_path, {'A01/a_2024_03_15_1400/trials.csv': {'seed': 1},
                             'A01/b_2024_03_15_0900/trials.csv': {'seed': 2},
                             'A01/c_2024_03_14/trials.csv': {'seed': 3}})
    sessions = load_experiment(path).animals['A01'].sessions
    assert [s.session_id for s in sessions] == ['c_2024_03_14', 'b_2024_03_15_0900', 'a_2024_03_15_1400']
    assert [s.start for s in sessions] == [None, time(9, 0), time(14, 0)]
    assert [s.session_idx for s in sessions] == [0, 1, 2]


def test_same_day_sessions_without_times_sort_by_name(tmp_path):
    path = _setup(tmp_path, {'A01/b_2024_03_15/trials.csv': {}, 'A01/a_2024_03_15/trials.csv': {'seed': 1}})
    sessions = load_experiment(path).animals['A01'].sessions
    assert [s.session_id for s in sessions] == ['a_2024_03_15', 'b_2024_03_15']


def test_each_file_is_its_own_session_without_merge_files(tmp_path):
    files = {'A01/day_2024_03_15/trials_2024_03_15_0900.csv': {},
             'A01/day_2024_03_15/trials_2024_03_15_1500.csv': {'seed': 1}}
    separate = _CONFIG.replace('  drop_last_row: false\n', '  drop_last_row: false\n  merge_files: false\n')
    sessions = load_experiment(_setup(tmp_path / 'separate', files, separate)).animals['A01'].sessions
    assert [s.start for s in sessions] == [time(9, 0), time(15, 0)]
    assert len({s.session_id for s in sessions}) == 2
    merged = load_experiment(_setup(tmp_path / 'merged', files)).animals['A01'].sessions
    assert len(merged) == 1 and merged[0].n_trials == 16


def test_an_unreadable_file_raises_when_strict_and_is_skipped_otherwise(tmp_path):
    path = _setup(tmp_path, {'A01/S_2024_03_15/trials_a.csv': {}})
    (tmp_path / 'data' / 'A01' / 'S_2024_03_15' / 'trials_b.csv').write_text('a,b\n1,2\n3,4,5,6\n')
    with pytest.raises(RuntimeError, match='trials_b'):
        load_experiment(path)
    sessions = load_experiment(path, strict=False).animals['A01'].sessions
    assert len(sessions) == 1 and sessions[0].n_trials == 8


def test_a_short_single_file_is_skipped(tmp_path):
    path = _setup(tmp_path, {'A01/S_2024_03_15/trials.csv': {'n': 3}, 'A01/S_2024_03_16/trials.csv': {}})
    assert [s.date for s in load_experiment(path).animals['A01'].sessions] == [date(2024, 3, 16)]


def test_a_session_without_a_date_raises_when_strict_and_is_skipped_otherwise(tmp_path):
    path = _setup(tmp_path, {'A01/S_2024_03_15/trials.csv': {}, 'A01/undated/trials.csv': {'seed': 1}})
    with pytest.raises(RuntimeError, match='no date'):
        load_experiment(path)
    assert [s.session_id for s in load_experiment(path, strict=False).animals['A01'].sessions] == ['S_2024_03_15']


def test_an_animal_id_mismatch_raises_when_strict_and_warns_otherwise(tmp_path):
    path = _setup(tmp_path, {'A01/S_2024_03_15/trials.csv': {'mouse': 'A02'}})
    with pytest.raises(RuntimeError, match="'A02'"):
        load_experiment(path)
    with pytest.warns(UserWarning, match='A02'):
        exp = load_experiment(path, strict=False)
    assert exp.animals['A01'].sessions[0].metadata.animal_id == 'A01'


def test_relative_paths_resolve_against_the_config_file(tmp_path):
    path = _setup(tmp_path, {'A01/S_2024_03_15/trials.csv': {}})
    (tmp_path / 'meta.json').write_text(json.dumps({'A01': {'genotype': 'het'}}))
    path.write_text(path.read_text() + 'animal_metadata: meta.json\n')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    previous = os.getcwd()
    os.chdir(elsewhere)
    try:
        exp = load_experiment(path)
    finally:
        os.chdir(previous)
    assert exp.animals['A01'].metadata['genotype'] == 'het'


def test_table_reader_names_missing_columns(tmp_path):
    config = load_config(_setup(tmp_path, {}))
    reader = table_reader(pd.DataFrame({'Mouse': ['A'], 'Trial': [1]}), animal='Mouse', session='Session',
                          date='Date')
    with pytest.raises(ConfigError, match='Session'):
        load_experiment(config, reader=reader)


def test_table_reader_orders_same_day_sessions_by_start(tmp_path):
    config = load_config(_setup(tmp_path, {}))
    rng = np.random.default_rng(0)
    rows = [pd.DataFrame({'Mouse': 'A01', 'Session': name, 'Date': '2024-03-15', 'Start': start,
                          'Trial': range(1, 9), 'Stim': rng.uniform(-1, 1, 8), 'Choice': rng.integers(0, 2, 8)})
            for name, start in (('late', '15:00'), ('early', '09:30'))]
    reader = table_reader(pd.concat(rows), animal='Mouse', session='Session', date='Date', start='Start')
    sessions = load_experiment(config, reader=reader).animals['A01'].sessions
    assert [s.session_id for s in sessions] == ['early', 'late']
    assert sessions[0].start == time(9, 30)
