"""Session discovery: every session before loading, with the ids loading gives; loading a chosen subset
reads nothing else; readers may attach related files; both session tables share the identity columns."""

import warnings
from datetime import date

import pandas as pd
import pytest

from behav_utils import discover_sessions, load_config, load_experiment
from behav_utils.config.schema import TableReaderConfig
from behav_utils.data.readers import RawSession
from tests.golden import outputs as golden
from tests.test_readers import _golden_table

IDENTITY = ['session_id', 'animal_id', 'experimenter', 'date', 'start', 'source_id', 'source']


def test_discovery_gives_the_ids_and_fields_loading_gives():
    table = discover_sessions(golden.CONFIG, laser=True)
    loaded = load_experiment(golden.CONFIG).session_table
    assert list(table.columns[:7]) == IDENTITY and list(loaded.columns[:7]) == IDENTITY
    for column in ('session_id', 'animal_id', 'date', 'source_id', 'stage', 'task', 'session_type'):
        assert list(table[column]) == list(loaded[column]), column
    assert not table['ignored'].any()


def test_the_table_reader_discovers_the_same_sessions(tmp_path):
    path = tmp_path / 'golden.csv'
    _golden_table().to_csv(path, index=False)
    config = load_config(golden.CONFIG)
    config.reader = TableReaderConfig(path=str(path), animal_column='Mouse', session_column='Session',
                                      date_column='Date')
    assert list(discover_sessions(config)['session_id']) == list(discover_sessions(golden.CONFIG)['session_id'])


def test_without_laser_a_derived_type_is_left_empty():
    table = discover_sessions(golden.CONFIG)
    assert table['session_type'].isna().all() and 'laser' not in table.columns


def test_ignored_sessions_are_listed_and_not_loaded():
    config = load_config(golden.CONFIG)
    config.sessions_to_ignore = {'G01': ['20260113']}
    table = discover_sessions(config)
    assert table.loc[table.session_id == 'G01_20260113', 'ignored'].item()
    assert 'G01_20260113' not in set(load_experiment(config).session_table['session_id'])


def test_a_filtered_load_never_touches_other_sessions(tmp_path):
    good = tmp_path / 'data' / 'G01' / 'S_2024_03_15' / 'trials.csv'      # a golden G01 file
    bad = tmp_path / 'data' / 'G01' / 'S_2024_03_16' / 'trials.csv'
    for path in (good, bad):
        path.parent.mkdir(parents=True)
    pd.read_csv(sorted(golden.DATA_DIR.glob('*/*/trials.csv'))[0]).to_csv(good, index=False)
    bad.write_bytes(b'\xff\xfe\x00 not a csv \xff')
    config = load_config(golden.CONFIG)
    config.reader.data_dir = str(tmp_path / 'data')
    with pytest.raises(RuntimeError):
        load_experiment(config)
    experiment = load_experiment(config, sessions=['G01_20240315'])
    assert list(experiment.session_table['session_id']) == ['G01_20240315']


def test_an_unknown_id_raises_when_strict_and_warns_otherwise():
    with pytest.raises(ValueError, match='G09_20260101'):
        load_experiment(golden.CONFIG, sessions=['G09_20260101'])
    with pytest.warns(UserWarning, match='G09_20260101'):
        load_experiment(golden.CONFIG, sessions=['G09_20260101', 'G01_20260112'], strict=False)


def _reader(peekable, files=None):
    def read(config):
        for day in (15, 16):
            def trials():
                raise AssertionError('a trial table was read')
            yield RawSession(animal_id='X1', session_id=f's{day}', date=date(2024, 3, day), trials=trials,
                             files=files or {}, peek=None if not peekable else (lambda columns=None, nrows=None: pd.DataFrame()))
    return read


def test_peek_false_reads_no_trial_file():
    table = discover_sessions(golden.CONFIG, reader=_reader(peekable=False), peek=False)
    assert list(table['session_id']) == ['X1_20240315', 'X1_20240316']


def test_a_reader_without_peek_is_read_in_full_with_a_warning():
    def read(config):
        yield RawSession(animal_id='X1', session_id='s1', date=date(2024, 3, 15),
                         trials=pd.DataFrame({'Stage': ['task'], 'TrialNum': [1]}))
    with pytest.warns(UserWarning, match='no cheap peek'):
        table = discover_sessions(golden.CONFIG, reader=read)
    assert table['stage'].item() == 'task'


def test_related_files_become_columns():
    files = {'video': '/rig/videos/x1.avi', 'licks': '/rig/licks/x1.csv'}
    table = discover_sessions(golden.CONFIG, reader=_reader(peekable=True, files=files))
    assert list(table.columns[-2:]) == ['file_video', 'file_licks']
    assert table['file_video'].tolist() == ['/rig/videos/x1.avi'] * 2


def test_the_experiment_session_table_covers_every_animal():
    experiment = load_experiment(golden.CONFIG)
    table = experiment.session_table
    assert list(table.columns[:7]) == IDENTITY
    assert len(table) == sum(a.n_sessions for a in experiment.animals.values())
    assert set(table['animal_id']) == set(experiment.animals)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert list(table['source_id']) == list(discover_sessions(golden.CONFIG)['source_id'])
