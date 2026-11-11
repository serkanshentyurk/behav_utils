"""Tests for behav_utils.data.loading — parsers, choice conversion, column coercion,
and the CSV → SessionData → ExperimentData pipeline.

The pure parsers (parse_timespan, parse_session_datetime, convert_choice_to_category) and the
column extractor (_safe_column) are tested directly. The loaders (load_session_csv, the folder
reader's file reading, load_animal, load_experiment) are driven by small synthetic CSVs written to
tmp_path, covering the branch points: missing-required validation, category derivation (both
rules), dates, drop_last_row, the min_trials file gate, animal_metadata.json merging, and
masking/washout session_type stamping. Readers in general are in test_readers.py.
"""
import json
import pickle
import warnings
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from behav_utils.config.schema import (
    ChoiceMapping,
    ColumnMapping,
    CsvFolderReaderConfig,
    ProjectConfig,
    SessionMetadataMapping,
    TaskConfig,
)
from behav_utils.data.loading import (
    _extract_session_metadata,
    _safe_column,
    convert_choice_to_category,
    load_animal,
    load_experiment,
    load_session_csv,
    parse_timespan,
)
from behav_utils.data.readers import _read_pieces, parse_session_datetime
from behav_utils.data.structures import SessionMetadata

_REGEX = r"(\d{4})_(\d{1,2})_(\d{1,2})"


# ── helpers ──────────────────────────────────────────────────────────────────
def _minimal_config(data_dir='.', drop_last_row=False, b_side='positive',
                    no_response_value=-1, with_opto=False, **fs):
    cols = {
        'trial_number': ColumnMapping(csv_name='Trial_Number', dtype='int'),
        'stimulus': ColumnMapping(csv_name='Stim', dtype='float'),
        'choice': ColumnMapping(csv_name='Choice', dtype='float'),
    }
    if with_opto:
        cols['opto_on'] = ColumnMapping(csv_name='Opto', dtype='bool',
                                        optional=True, default=False)
    task = TaskConfig(boundary=0.0,
                      params={'b_side': b_side},
                      choice_mapping=ChoiceMapping(type='identity',
                                                   no_response_value=no_response_value))
    reader = CsvFolderReaderConfig(data_dir=str(data_dir), drop_last_row=drop_last_row,
                                   behaviour_file='trial_summary*.csv', date_regex=_REGEX, **fs)
    return ProjectConfig(name='test', columns=cols, task=task, reader=reader)


def _write_csv(path, n=11, with_opto=False, with_date=False, seed=0):
    rng = np.random.default_rng(seed)
    stim = rng.uniform(-1, 1, n)
    choice = rng.integers(0, 2, n)               # independent of category → real 'correct'
    data = {'Trial_Number': range(1, n + 1), 'Stim': stim, 'Choice': choice}
    if with_opto:
        data['Opto'] = [True] * n
    if with_date:
        data['Date'] = ['2024-03-15'] * n
    pd.DataFrame(data).to_csv(path, index=False)
    return stim, choice


# ── parse_timespan ───────────────────────────────────────────────────────────
class TestParseTimespan:
    def test_numeric_passthrough(self):
        assert parse_timespan(12.5) == 12.5
        assert parse_timespan('30') == 30.0

    def test_hms(self):
        assert parse_timespan('01:02:03') == 3723.0

    def test_hms_fractional(self):
        assert parse_timespan('00:00:01.5') == 1.5

    def test_partial_hh_mm(self):
        assert parse_timespan('00:30') == 1800.0

    def test_unparseable_returns_default(self):
        assert parse_timespan('not a time', default=-1) == -1

    def test_none_and_nan_return_default(self):
        assert parse_timespan(None, default=-1) == -1
        assert parse_timespan(float('nan'), default=-1) == -1


# ── parse_session_datetime ───────────────────────────────────────────────────
class TestParseSessionDatetime:
    def test_positional_groups(self):
        assert parse_session_datetime('SC_M1_2024_03_15', _REGEX) == (date(2024, 3, 15), None)

    def test_named_groups(self):
        rgx = r"(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})"
        assert parse_session_datetime('x_2024_03_15_y', rgx) == (date(2024, 3, 15), None)

    def test_start_time_from_named_groups(self):
        rgx = r"(\d{4})_(\d{1,2})_(\d{1,2})_(?P<hour>\d{2})(?P<minute>\d{2})"
        assert parse_session_datetime('S_2024_03_15_1430', rgx) == (date(2024, 3, 15), time(14, 30))

    def test_no_match_returns_none(self):
        assert parse_session_datetime('no_date_here', _REGEX) == (None, None)

    def test_invalid_date_returns_none(self):
        assert parse_session_datetime('2024_13_40', _REGEX) == (None, None)    # month 13


# ── convert_choice_to_category ───────────────────────────────────────────────
class TestConvertChoice:
    def test_identity_passthrough(self):
        cm = ChoiceMapping(type='identity', no_response_value=-1)
        out = convert_choice_to_category(np.array([0, 1, 1, 0]), SessionMetadata(), cm)
        np.testing.assert_array_equal(out, [0., 1., 1., 0.])

    def test_identity_masks_no_response(self):
        cm = ChoiceMapping(type='identity', no_response_value=0)
        out = convert_choice_to_category(np.array([0, 1, 0, 1]), SessionMetadata(), cm)
        assert np.isnan(out[0]) and np.isnan(out[2])
        np.testing.assert_array_equal(out[[1, 3]], [1., 1.])

    def test_identity_without_a_no_response_value_keeps_zero(self):
        cm = ChoiceMapping(type='identity', no_response_value=None)
        out = convert_choice_to_category(np.array([0.0, 1.0, np.nan, 0.0]), SessionMetadata(), cm)
        np.testing.assert_array_equal(out, [0.0, 1.0, np.nan, 0.0])

    def test_none_returns_raw_float(self):
        cm = ChoiceMapping(type='none')
        out = convert_choice_to_category(np.array([-1, 0, 1]), SessionMetadata(), cm)
        np.testing.assert_array_equal(out, [-1., 0., 1.])

    def test_spatial_applies_contingency(self):
        cm = ChoiceMapping(type='spatial_to_category', no_response_value=0,
                           contingency_field='response_mapping',
                           contingency_rules={'rule_a': {-1: 0, 1: 1}})
        meta = SessionMetadata(fields={'response_mapping': 'rule_a'})
        out = convert_choice_to_category(np.array([-1, 1, 0, 1]), meta, cm)
        assert np.isnan(out[2])
        np.testing.assert_array_equal(out[[0, 1, 3]], [0., 1., 1.])

    def test_spatial_missing_field_warns_and_returns_raw(self):
        cm = ChoiceMapping(type='spatial_to_category',
                           contingency_field='response_mapping',
                           contingency_rules={'rule_a': {-1: 0, 1: 1}})
        with pytest.warns(UserWarning):
            out = convert_choice_to_category(np.array([-1, 1]), SessionMetadata(), cm)
        np.testing.assert_array_equal(out, [-1., 1.])

    def test_spatial_unknown_contingency_warns_and_returns_raw(self):
        cm = ChoiceMapping(type='spatial_to_category',
                           contingency_field='response_mapping',
                           contingency_rules={'rule_a': {-1: 0, 1: 1}})
        meta = SessionMetadata(fields={'response_mapping': 'mystery'})
        with pytest.warns(UserWarning):
            out = convert_choice_to_category(np.array([-1, 1]), meta, cm)
        np.testing.assert_array_equal(out, [-1., 1.])

    def test_unknown_type_raises(self):
        cm = ChoiceMapping(type='bogus')
        with pytest.raises(ValueError):
            convert_choice_to_category(np.array([0, 1]), SessionMetadata(), cm)


# ── _safe_column ─────────────────────────────────────────────────────────────
class TestSafeColumn:
    def test_missing_optional_float_fills_nan(self):
        out = _safe_column(pd.DataFrame({'A': [1]}),
                           ColumnMapping(csv_name='B', dtype='float', optional=True), 3)
        assert out.shape == (3,) and np.all(np.isnan(out))

    def test_missing_optional_int_fills_default(self):
        out = _safe_column(pd.DataFrame({'A': [1]}),
                           ColumnMapping(csv_name='B', dtype='int',
                                         optional=True, default=7), 3)
        np.testing.assert_array_equal(out, [7, 7, 7])

    def test_missing_optional_str_fills_empty(self):
        out = _safe_column(pd.DataFrame({'A': [1]}),
                           ColumnMapping(csv_name='B', dtype='str', optional=True), 2)
        np.testing.assert_array_equal(out, ['', ''])

    def test_missing_required_raises(self):
        with pytest.raises(KeyError):
            _safe_column(pd.DataFrame({'A': [1]}),
                         ColumnMapping(csv_name='B', dtype='float', optional=False), 1)

    def test_value_mapping_applied(self):
        df = pd.DataFrame({'C': [-1, 1]})
        out = _safe_column(df, ColumnMapping(csv_name='C', dtype='str',
                                             mapping={-1: 'L', 1: 'R'}), 2)
        np.testing.assert_array_equal(out, ['L', 'R'])

    def test_float_coercion_non_numeric_to_nan(self):
        df = pd.DataFrame({'C': ['1.5', 'abc']})
        out = _safe_column(df, ColumnMapping(csv_name='C', dtype='float'), 2)
        assert out[0] == 1.5 and np.isnan(out[1])

    def test_bool_coercion_mixed_strings(self):
        df = pd.DataFrame({'C': ['True', 'False', '1', '0']})
        out = _safe_column(df, ColumnMapping(csv_name='C', dtype='bool'), 4)
        np.testing.assert_array_equal(out, [True, False, True, False])


# ── _extract_session_metadata ────────────────────────────────────────────────
class TestExtractSessionMetadata:
    def test_first_row_with_coercion_and_timespan(self):
        df = pd.DataFrame({'Stage': ['Full', 'Full'],
                           'Elapsed': ['00:01:00', '00:02:00'],
                           'Temp': ['21.5', '21.5']})
        meta_specs = {
            'stage': SessionMetadataMapping(csv_name='Stage', dtype='str'),
            'elapsed': SessionMetadataMapping(csv_name='Elapsed', parse_timespan=True),
            'temp': SessionMetadataMapping(csv_name='Temp', dtype='float'),
        }
        cfg = ProjectConfig(name='t', columns={'trial_number': ColumnMapping('Trial_Number')},
                            session_metadata=meta_specs,
                            task=TaskConfig())
        meta = _extract_session_metadata(df, cfg)
        assert meta.get('stage') == 'Full'
        assert meta.get('elapsed') == 60.0
        assert meta.get('temp') == 21.5

    def test_empty_df_returns_empty_metadata(self):
        cfg = ProjectConfig(name='t', columns={'trial_number': ColumnMapping('Trial_Number')},
                            task=TaskConfig())
        meta = _extract_session_metadata(pd.DataFrame(), cfg)
        assert meta.fields == {}


# ── load_session_csv ─────────────────────────────────────────────────────────
class TestLoadSessionCsv:
    def test_loads_and_derives_category_and_correct(self, tmp_path):
        cfg = _minimal_config(drop_last_row=False)
        p = tmp_path / 'trial_summary.csv'
        stim, choice = _write_csv(p, n=11)
        sess = load_session_csv(p, cfg, date=date(2024, 3, 15))
        assert sess.trials.n_trials == 11
        assert sess.date == date(2024, 3, 15)
        np.testing.assert_allclose(sess.trials.stimulus, stim)
        expected_cat = (stim > 0).astype(int)
        np.testing.assert_array_equal(sess.trials.category, expected_cat)
        np.testing.assert_array_equal(sess.trials.choice, choice.astype(float))
        # 'correct' not mapped → derived as (choice == category)
        np.testing.assert_array_equal(sess.trials.correct, choice == expected_cat)

    def test_missing_required_column_raises(self, tmp_path):
        cfg = _minimal_config(drop_last_row=False, min_trials_per_file=1)
        p = tmp_path / 'trial_summary.csv'
        pd.DataFrame({'Trial_Number': [1, 2], 'Choice': [0, 1]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match='missing required'):              # no 'Stim'
            load_session_csv(p, cfg, date=date(2024, 3, 15))

    def test_below_boundary_rule_flips_category(self, tmp_path):
        cfg = _minimal_config(drop_last_row=False, b_side='negative')
        p = tmp_path / 'trial_summary.csv'
        stim, _ = _write_csv(p, n=11)
        sess = load_session_csv(p, cfg, date=date(2024, 3, 15))
        np.testing.assert_array_equal(sess.trials.category, 1 - (stim > 0).astype(int))

    def test_drop_last_row(self, tmp_path):
        cfg = _minimal_config(drop_last_row=True)
        p = tmp_path / 'trial_summary.csv'
        _write_csv(p, n=12)
        sess = load_session_csv(p, cfg, date=date(2024, 3, 15))
        assert sess.trials.n_trials == 11           # one row dropped

    def test_no_date_raises(self, tmp_path):
        cfg = _minimal_config(drop_last_row=False)
        p = tmp_path / 'nodate.csv'                  # no date in the folder or the file name
        _write_csv(p, n=11)
        with pytest.raises(ValueError, match='no date'):
            load_session_csv(p, cfg)

    def test_date_from_the_folder_name(self, tmp_path):
        cfg = _minimal_config(drop_last_row=False)
        p = tmp_path / 'Sess_2024_03_15' / 'trial_summary.csv'
        p.parent.mkdir()
        _write_csv(p, n=11)
        assert load_session_csv(p, cfg).date == date(2024, 3, 15)

    def test_a_date_column_is_not_read(self, tmp_path):
        cfg = _minimal_config(drop_last_row=False)
        p = tmp_path / 'nodate.csv'
        _write_csv(p, n=11, with_date=True)          # a 'Date' column is no longer a fallback
        with pytest.raises(ValueError, match='no date'):
            load_session_csv(p, cfg)


# ── reading a session's files ────────────────────────────────────────────────
class TestReadPieces:
    def test_merges_and_renumbers(self, tmp_path):
        cfg = _minimal_config(drop_last_row=True, min_trials_per_file=10)
        a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
        _write_csv(a, n=12)
        _write_csv(b, n=12)
        merged = _read_pieces([a, b], cfg.reader, 'Trial_Number', True)
        assert len(merged) == 22                     # 11 + 11 after drop_last
        # Trial_Number renumbered 1..22
        np.testing.assert_array_equal(merged['Trial_Number'].to_numpy(),
                                      np.arange(1, 23))

    def test_file_below_min_trials_dropped(self, tmp_path):
        cfg = _minimal_config(drop_last_row=False, min_trials_per_file=10)
        c = tmp_path / 'c.csv'
        _write_csv(c, n=5)                           # below gate
        assert _read_pieces([c], cfg.reader, 'Trial_Number', True).empty

    def test_single_file_returned_asis(self, tmp_path):
        cfg = _minimal_config(drop_last_row=False, min_trials_per_file=10)
        a = tmp_path / 'a.csv'
        _write_csv(a, n=11)
        merged = _read_pieces([a], cfg.reader, 'Trial_Number', True)
        assert len(merged) == 11

    def test_unreadable_file_raises_when_strict_and_is_skipped_otherwise(self, tmp_path):
        cfg = _minimal_config(drop_last_row=False, min_trials_per_file=10)
        a, bad = tmp_path / 'a.csv', tmp_path / 'b.csv'
        _write_csv(a, n=11)
        bad.write_text('a,b\n1,2\n3,4,5,6\n')
        with pytest.raises(OSError, match='b.csv'):
            _read_pieces([a, bad], cfg.reader, 'Trial_Number', True)
        assert len(_read_pieces([a, bad], cfg.reader, 'Trial_Number', False)) == 11


# ── load_animal / load_experiment ────────────────────────────────────────────
class TestLoadAnimalAndExperiment:
    def _build_tree(self, root, with_opto=False):
        a1 = root / 'Animal1'
        for d in ('Sess_2024_03_15', 'Sess_2024_03_16'):
            (a1 / d).mkdir(parents=True)
            _write_csv(a1 / d / 'trial_summary.csv', n=11, with_opto=with_opto)
        return a1

    def test_load_animal_finds_sessions_and_dates(self, tmp_path):
        a1 = self._build_tree(tmp_path)
        cfg = _minimal_config(data_dir=tmp_path, drop_last_row=False)
        animal = load_animal(a1, cfg)
        assert animal.n_sessions == 2
        assert {s.date for s in animal.sessions} == {date(2024, 3, 15), date(2024, 3, 16)}

    def test_empty_files_are_skipped_as_if_absent(self, tmp_path):
        """A run stopped before writing anything leaves an empty file. Next to a real one it is ignored
        (the session loads from the real file); on its own the session is skipped, even when strict."""
        a1 = self._build_tree(tmp_path)
        (a1 / 'Sess_2024_03_15' / 'trial_summary_0.csv').write_text('')
        (a1 / 'Sess_2024_03_17').mkdir()
        (a1 / 'Sess_2024_03_17' / 'trial_summary.csv').write_text('')
        cfg = _minimal_config(data_dir=tmp_path, drop_last_row=False)
        animal = load_animal(a1, cfg)
        assert animal.n_sessions == 2
        assert {s.date for s in animal.sessions} == {date(2024, 3, 15), date(2024, 3, 16)}
        assert all(s.trials.n_trials == 11 for s in animal.sessions)

    def test_load_experiment_merges_metadata_and_masks(self, tmp_path):
        data = tmp_path / 'data'
        self._build_tree(data, with_opto=True)
        (data / 'animal_metadata.json').write_text(
            json.dumps({'Animal1': {'genotype': 'het'}}))
        cfg = _minimal_config(data_dir=data, drop_last_row=False, with_opto=True)
        cfg.session_types = {'masking': {'Animal1': ['20240315']}}

        exp = load_experiment(cfg)
        assert exp.n_animals == 1
        animal = exp.animals['Animal1']
        assert animal.n_sessions == 2
        assert animal.metadata['genotype'] == 'het'   # merged from animal_metadata.json

        by_date = {s.date: s for s in animal.sessions}
        assert by_date[date(2024, 3, 15)].session_type == 'masking'
        assert by_date[date(2024, 3, 15)].session_type == 'masking'
        # opto_on is preserved: on a masking session these are the fake-opto trials.
        assert by_date[date(2024, 3, 15)].trials.opto_on.any()
        assert by_date[date(2024, 3, 16)].trials.opto_on.any()       # untouched

    def test_load_experiment_applies_washout(self, tmp_path):
        data = tmp_path / 'data'
        self._build_tree(data, with_opto=True)
        cfg = _minimal_config(data_dir=data, drop_last_row=False, with_opto=True)
        cfg.session_types = {'washout': {'Animal1': ['20240316']}}
        exp = load_experiment(cfg)
        by_date = {s.date: s for s in exp.animals['Animal1'].sessions}
        assert by_date[date(2024, 3, 16)].session_type == 'washout'
        assert by_date[date(2024, 3, 16)].session_type == 'washout'
        assert by_date[date(2024, 3, 16)].trials.opto_on.any()        # preserved
        assert by_date[date(2024, 3, 15)].trials.opto_on.any()        # untouched

    def test_load_animal_merges_multiple_csvs_per_session(self, tmp_path):
        sess = tmp_path / 'Animal1' / 'Sess_2024_03_15'
        sess.mkdir(parents=True)
        _write_csv(sess / 'trial_summary_a.csv', n=12)
        _write_csv(sess / 'trial_summary_b.csv', n=12, seed=1)
        cfg = _minimal_config(data_dir=tmp_path, drop_last_row=True, min_trials_per_file=10)
        animal = load_animal(tmp_path / 'Animal1', cfg)
        assert animal.n_sessions == 1
        assert animal.sessions[0].trials.n_trials == 22       # 11 + 11 merged

    def test_load_experiment_missing_data_dir_raises(self, tmp_path):
        cfg = _minimal_config(data_dir=tmp_path / 'nope', drop_last_row=False)
        with pytest.raises(FileNotFoundError):
            load_experiment(cfg)


# ── stimulus normalisation, choice-encoding warning, strict loading ──────────
class TestStimulusTransform:
    def test_identity_by_default(self):
        t = TaskConfig()
        assert t.is_identity_transform
        x = np.array([-1.0, -0.2, 0.0, 0.7, np.nan])
        np.testing.assert_array_equal(t.normalise_stimulus(x), x)

    def test_linear_asymmetric_range_puts_boundary_at_zero(self):
        t = TaskConfig(boundary=0.2, stimulus_range=(-1.0, 1.0))
        np.testing.assert_allclose(t.normalise_stimulus([-1.0, 0.2, 1.0, 0.6, -0.4]),
                                   [-1.0, 0.0, 1.0, 0.5, -0.5])

    def test_log_scale(self):
        t = TaskConfig(boundary=8000.0, stimulus_range=(2000.0, 16000.0),
                       stimulus_scale='log')
        np.testing.assert_allclose(t.normalise_stimulus([2000.0, 4000.0, 8000.0, 16000.0]),
                                   [-1.0, -0.5, 0.0, 1.0])

    def test_invalid_configs_raise(self):
        with pytest.raises(ValueError):
            TaskConfig(boundary=2.0, stimulus_range=(-1.0, 1.0))
        with pytest.raises(ValueError):
            TaskConfig(stimulus_scale='sqrt')
        with pytest.raises(ValueError):
            TaskConfig(boundary=1.0, stimulus_range=(0.0, 2.0), stimulus_scale='log')

    def test_loader_applies_transform_and_keeps_raw(self, tmp_path):
        cfg = _minimal_config(drop_last_row=False, min_trials_per_file=1)
        cfg.task.boundary, cfg.task.stimulus_range, cfg.task.stimulus_scale = 8000.0, (2000.0, 16000.0), 'log'
        p = tmp_path / 'trial_summary.csv'
        raw = np.array([2000.0, 4000.0, 8000.0, 16000.0, 11313.7085])
        pd.DataFrame({'Trial_Number': range(1, 6), 'Stim': raw, 'Choice': [0, 0, 1, 1, 1]}).to_csv(p, index=False)
        sess = load_session_csv(p, cfg, date=date(2024, 3, 15))
        np.testing.assert_allclose(sess.trials.stimulus, [-1.0, -0.5, 0.0, 1.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(sess.trials.stimulus_raw, raw)
        np.testing.assert_array_equal(sess.trials.category, [0, 0, 0, 1, 1])   # category from the normalised axis
        assert 'stimuli_raw' in sess.get_arrays()

    def test_mostly_nan_choices_warn(self, tmp_path):
        cfg = _minimal_config(drop_last_row=False, no_response_value=-1)
        p = tmp_path / 'trial_summary.csv'
        pd.DataFrame({'Trial_Number': range(1, 11), 'Stim': np.linspace(-1, 1, 10),
                      'Choice': [-1] * 6 + [1] * 4}).to_csv(p, index=False)
        with pytest.warns(UserWarning, match='no-response'):
            load_session_csv(p, cfg, date=date(2024, 3, 15))


class TestStrictLoading:
    def _tree(self, tmp_path, broken=False):
        animal = tmp_path / 'A01'
        for i, d in enumerate(['2024_03_15', '2024_03_16']):
            sd = animal / d
            sd.mkdir(parents=True)
            _write_csv(sd / 'trial_summary.csv', n=11, seed=i)
        if broken:
            (animal / '2024_03_17').mkdir()
            pd.DataFrame({'Trial_Number': range(1, 12), 'Choice': [0, 1] * 5 + [0]}).to_csv(
                animal / '2024_03_17' / 'trial_summary.csv', index=False)   # missing 'Stim'
        return animal

    def test_strict_raises_on_broken_session(self, tmp_path):
        animal = self._tree(tmp_path, broken=True)
        cfg = _minimal_config(data_dir=tmp_path, drop_last_row=False)
        with pytest.raises(RuntimeError, match='2024_03_17'):
            load_animal(animal, cfg)

    def test_non_strict_skips_and_logs(self, tmp_path):
        animal = self._tree(tmp_path, broken=True)
        cfg = _minimal_config(data_dir=tmp_path, drop_last_row=False)
        a = load_animal(animal, cfg, strict=False)
        assert a.n_sessions == 2

    def test_experiment_loads_and_presets_attach_to_config(self, tmp_path):
        self._tree(tmp_path)
        cfg = _minimal_config(data_dir=tmp_path, drop_last_row=False)
        cfg.session_presets = {'first': {'first_n': 1}}
        exp = load_experiment(cfg)
        animal = exp.get_animal('A01')
        from behav_utils import select_sessions
        assert [s.session_idx for s in select_sessions(animal, preset='first', min_trials=0)] == [0]


class TestSnapshotProvenance:
    def test_save_stamps_version_and_load_warns_on_mismatch(self, tmp_path):
        import behav_utils
        from behav_utils import generate_synthetic_animal
        from behav_utils.data.structures import ExperimentData
        a = generate_synthetic_animal(animal_id='A', n_sessions=2, trials_per_session=50, stage='task', rng=0)
        exp = ExperimentData()
        exp.add_animal(a)
        p = tmp_path / 'snap.pkl'
        exp.save(p)
        assert exp.metadata['behav_utils_version'] == behav_utils.__version__
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            ExperimentData.load(p)                                   # same version: silent
        exp.metadata['behav_utils_version'] = '0.0.0'      # re-pickle by hand: save() would re-stamp
        with open(p, 'wb') as f:
            pickle.dump(exp, f)
        with pytest.warns(UserWarning, match='0.0.0'):
            ExperimentData.load(p)
