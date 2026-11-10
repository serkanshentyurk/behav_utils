"""Tests for behav_utils.config.schema — dataclass validation, YAML loading, CSV validation.

Covers the real branch points: dtype validation, the ProjectConfig._validate rules
(required trial_number, primary auto-detection, dangling references), the YAML parser's
normalisations (str-vs-dict column specs, list-of-pairs mappings, contingency-rule
coercion, session_types shapes, data_dir expandvars), and validate_csv_against_config.
"""
import glob
import os
from pathlib import Path

import pytest

from behav_utils.config.schema import (
    ColumnMapping,
    ConfigError,
    ProjectConfig,
    SessionMetadataMapping,
    TaskConfig,
    _parse_column_mapping,
    _parse_session_metadata,
    load_config,
    validate_csv_against_config,
)


def _cols(*names):
    """A column dict whose csv_names are the title-cased internal names."""
    return {n: ColumnMapping(csv_name=n.replace('_', ' ').title().replace(' ', '_'))
            for n in names}


# ── ColumnMapping / SessionMetadataMapping ───────────────────────────────────
class TestColumnMapping:
    def test_defaults(self):
        m = ColumnMapping(csv_name='Stim')
        assert m.dtype == 'float' and m.optional is False
        assert m.default is None and m.mapping is None

    def test_each_valid_dtype_constructs(self):
        for dt in ('int', 'float', 'str', 'bool'):
            assert ColumnMapping(csv_name='C', dtype=dt).dtype == dt

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError):
            ColumnMapping(csv_name='C', dtype='complex')


class TestSessionMetadataMapping:
    def test_defaults(self):
        m = SessionMetadataMapping(csv_name='Stage')
        assert m.dtype == 'str' and m.optional is True
        assert m.parse_timespan is False


# ── ProjectConfig._validate ──────────────────────────────────────────────────
class TestProjectConfigValidation:
    def test_minimal_valid_constructs(self):
        cfg = ProjectConfig(name='t', columns=_cols('trial_number'),
                            task=TaskConfig())
        assert cfg.name == 't'

    def test_missing_trial_number_raises(self):
        with pytest.raises(ValueError):
            ProjectConfig(name='t', columns={}, task=TaskConfig())

    def test_autodetects_primary_stimulus_and_choice(self):
        cfg = ProjectConfig(name='t', columns=_cols('trial_number', 'stimulus', 'choice'),
                            task=TaskConfig())
        assert cfg.task.primary_stimulus == 'stimulus'
        assert cfg.task.primary_choice == 'choice'

    def test_no_autodetect_when_columns_absent(self):
        cfg = ProjectConfig(name='t', columns=_cols('trial_number'),
                            task=TaskConfig())
        assert cfg.task.primary_stimulus is None
        assert cfg.task.primary_choice is None

    def test_primary_stimulus_referencing_missing_column_raises(self):
        with pytest.raises(ValueError):
            ProjectConfig(name='t', columns=_cols('trial_number'),
                          task=TaskConfig(primary_stimulus='ghost', primary_choice=None))


# ── ProjectConfig helpers ────────────────────────────────────────────────────
class TestProjectConfigHelpers:
    def _cfg(self):
        cols = {
            'trial_number': ColumnMapping(csv_name='Trial_Number', dtype='int'),
            'stimulus': ColumnMapping(csv_name='Stim', dtype='float'),
            'reaction_time': ColumnMapping(csv_name='RT', dtype='float', optional=True),
        }
        meta = {'stage': SessionMetadataMapping(csv_name='Stage')}
        return ProjectConfig(name='t', columns=cols, session_metadata=meta,
                             extra_columns=['Notes'],
                             task=TaskConfig())

    def test_get_csv_name(self):
        cfg = self._cfg()
        assert cfg.get_csv_name('stimulus') == 'Stim'
        assert cfg.get_csv_name('stage') == 'Stage'        # from session_metadata
        assert cfg.get_csv_name('nonexistent') is None

    def test_get_all_csv_columns(self):
        cols = set(self._cfg().get_all_csv_columns())
        assert {'Trial_Number', 'Stim', 'RT', 'Stage', 'Notes'} <= cols

    def test_required_vs_optional(self):
        cfg = self._cfg()
        assert set(cfg.required_csv_columns) == {'Trial_Number', 'Stim'}
        assert set(cfg.optional_csv_columns) == {'RT'}


# ── YAML spec parsers ────────────────────────────────────────────────────────
class TestParseColumnMapping:
    def test_string_spec_becomes_csv_name(self):
        m = _parse_column_mapping('stimulus', 'Stim_Relative')
        assert m.csv_name == 'Stim_Relative' and m.dtype == 'float'

    def test_dict_spec_reads_fields(self):
        m = _parse_column_mapping('choice', {'csv_name': 'Ch', 'dtype': 'int',
                                              'optional': True, 'default': 0})
        assert (m.csv_name, m.dtype, m.optional, m.default) == ('Ch', 'int', True, 0)

    def test_mapping_list_of_pairs_becomes_dict(self):
        m = _parse_column_mapping('choice', {'csv_name': 'Ch',
                                              'mapping': [[-1, 0], [1, 1]]})
        assert m.mapping == {-1: 0, 1: 1}

    def test_session_metadata_string_spec(self):
        m = _parse_session_metadata('stage', 'Stage')
        assert m.csv_name == 'Stage' and m.optional is True


# ── load_config ──────────────────────────────────────────────────────────────
_YAML = """
project:
  name: Test Project
  description: A test config
file_structure:
  data_dir: $TESTDATA_DIR/sub
  behaviour_file: trial_summary*.csv
task:
  boundary: 0.0
  choice_mapping:
    type: spatial_to_category
    no_response_value: 0
    contingency_field: response_mapping
    contingency_rules:
      1: {-1: 0, 1: 1}
      2: {-1: 1, 1: 0}
columns:
  trial_number: Trial_Number
  stimulus:
    csv_name: Stim
    dtype: float
  choice:
    csv_name: Choice
    dtype: int
    mapping: [[-1, 0], [1, 1]]
session_metadata:
  stage:
    csv_name: Stage
session_types:
  masking:
    Animal1: [20240315, 20240316]
    Animal2: 20240301
    Animal3: null
  washout:
    Animal1: [20240320]
"""


class TestLoadConfig:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / 'does_not_exist.yaml')

    def test_non_mapping_yaml_raises(self, tmp_path):
        p = tmp_path / 'bad.yaml'
        p.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError):
            load_config(p)

    def test_roundtrip_project_and_columns(self, tmp_path):
        os.environ['TESTDATA_DIR'] = '/tmp/somewhere'
        p = tmp_path / 'config.yaml'
        p.write_text(_YAML)
        cfg = load_config(p)
        assert cfg.name == 'Test Project'
        assert cfg.description == 'A test config'
        assert cfg.columns['stimulus'].csv_name == 'Stim'
        assert cfg.columns['choice'].dtype == 'int'
        # list-of-pairs mapping parsed to dict
        assert cfg.columns['choice'].mapping == {-1: 0, 1: 1}
        # primary fields auto-detected from column names
        assert cfg.task.primary_stimulus == 'stimulus'
        assert cfg.task.primary_choice == 'choice'

    def test_data_dir_expandvars(self, tmp_path):
        os.environ['TESTDATA_DIR'] = '/tmp/somewhere'
        p = tmp_path / 'config.yaml'
        p.write_text(_YAML)
        cfg = load_config(p)
        assert cfg.file_structure.data_dir == '/tmp/somewhere/sub'

    def test_contingency_rules_coerced(self, tmp_path):
        p = tmp_path / 'config.yaml'
        p.write_text(_YAML)
        cfg = load_config(p)
        rules = cfg.task.choice_mapping.contingency_rules
        # outer keys are strings, inner keys/values are ints
        assert rules == {'1': {-1: 0, 1: 1}, '2': {-1: 1, 1: 0}}

    def test_session_types_normalised(self, tmp_path):
        p = tmp_path / 'config.yaml'
        p.write_text(_YAML)
        cfg = load_config(p)
        assert cfg.session_types['masking']['Animal1'] == ['20240315', '20240316']  # list kept
        assert cfg.session_types['masking']['Animal2'] == ['20240301']              # scalar wrapped
        assert cfg.session_types['masking']['Animal3'] == []                        # null → empty
        assert cfg.session_types['washout']['Animal1'] == ['20240320']


# ── validate_csv_against_config ──────────────────────────────────────────────
class TestValidateCsvAgainstConfig:
    def _cfg(self):
        cols = {
            'trial_number': ColumnMapping(csv_name='Trial_Number', dtype='int'),
            'stimulus': ColumnMapping(csv_name='Stim', dtype='float'),
            'reaction_time': ColumnMapping(csv_name='RT', dtype='float', optional=True),
        }
        return ProjectConfig(name='t', columns=cols,
                             task=TaskConfig())

    def test_all_required_present(self):
        res = validate_csv_against_config(['Trial_Number', 'Stim', 'Weird'], self._cfg())
        assert set(res['matched']) == {'Trial_Number', 'Stim'}
        assert res['missing_required'] == []
        assert res['missing_optional'] == ['RT']
        assert res['unmapped'] == ['Weird']

    def test_missing_required_reported(self):
        res = validate_csv_against_config(['Trial_Number', 'Weird'], self._cfg())
        assert res['missing_required'] == ['Stim']
        assert 'RT' in res['missing_optional']
        assert res['unmapped'] == ['Weird']


# ── strict keys and required choice-mapping fields ───────────────────────────
_MINIMAL = """
project:
  name: t
task:
  choice_mapping:
    type: identity
    no_response_value: -1
columns:
  trial_number: Trial
  stimulus: Stim
  choice: Choice
"""


def _write(tmp_path, text):
    path = tmp_path / 'config.yaml'
    path.write_text(text)
    return path


class TestStrictConfig:
    def test_minimal_loads(self, tmp_path):
        cfg = load_config(_write(tmp_path, _MINIMAL))
        assert cfg.task.choice_mapping.no_response_value == -1

    @pytest.mark.parametrize('old, new, section', [
        ('task:', 'task:\n  boundry: 0.0', 'task'),
        ('project:', 'project:\n  title: x', 'project'),
        ('columns:', 'file_structure:\n  data_dri: x\ncolumns:', 'file_structure'),
        ('    no_response_value: -1', '    no_response_value: -1\n    no_responce: 0', 'task.choice_mapping'),
        ('  stimulus: Stim', '  stimulus: {csv_name: Stim, dtyp: float}', 'columns.stimulus'),
    ])
    def test_unknown_key_raises_naming_the_section(self, tmp_path, old, new, section):
        with pytest.raises(ConfigError, match=section):
            load_config(_write(tmp_path, _MINIMAL.replace(old, new, 1)))

    def test_message_suggests_the_closest_key(self, tmp_path):
        with pytest.raises(ConfigError, match="did you mean 'boundary'"):
            load_config(_write(tmp_path, _MINIMAL.replace('task:', 'task:\n  boundry: 0.0', 1)))

    def test_unknown_top_level_key_raises(self, tmp_path):
        with pytest.raises(ConfigError, match='sessions_to_ignor'):
            load_config(_write(tmp_path, _MINIMAL + 'sessions_to_ignor: {}\n'))

    def test_name_at_top_level_points_to_project(self, tmp_path):
        with pytest.raises(ConfigError, match="under 'project:'"):
            load_config(_write(tmp_path, 'name: t\n' + _MINIMAL))

    def test_removed_task_fields_are_unknown(self, tmp_path):
        with pytest.raises(ConfigError, match='inputs'):
            load_config(_write(tmp_path, _MINIMAL.replace('task:', 'task:\n  inputs: [stimulus]', 1)))

    def test_custom_block_is_never_read(self, tmp_path):
        cfg = load_config(_write(tmp_path, _MINIMAL + 'custom:\n  anything: [1, 2]\n  nested: {a: b}\n'))
        assert cfg.name == 't'

    def test_cohorts_block_is_allowed(self, tmp_path):
        load_config(_write(tmp_path, _MINIMAL + 'cohorts:\n  pilot: [A01]\n'))

    def test_config_error_is_a_value_error(self):
        assert issubclass(ConfigError, ValueError)

    def test_no_response_value_required_when_a_choice_is_mapped(self, tmp_path):
        with pytest.raises(ConfigError, match='no_response_value'):
            load_config(_write(tmp_path, _MINIMAL.replace('    no_response_value: -1\n', '')))

    def test_null_no_response_value_means_none(self, tmp_path):
        cfg = load_config(_write(tmp_path, _MINIMAL.replace('no_response_value: -1', 'no_response_value: null')))
        assert cfg.task.choice_mapping.no_response_value is None

    def test_no_response_value_not_required_without_a_choice_column(self, tmp_path):
        text = _MINIMAL.replace('  choice: Choice\n', '').replace('    no_response_value: -1\n', '')
        assert load_config(_write(tmp_path, text)).task.primary_choice is None

    def test_spatial_needs_a_contingency_field(self, tmp_path):
        text = _MINIMAL.replace('type: identity', 'type: spatial_to_category')
        with pytest.raises(ConfigError, match='contingency_field'):
            load_config(_write(tmp_path, text))

    def test_unknown_choice_mapping_type_raises(self, tmp_path):
        with pytest.raises(ConfigError, match='identity'):
            load_config(_write(tmp_path, _MINIMAL.replace('type: identity', 'type: identiy')))

    def test_category_rule_is_checked(self):
        with pytest.raises(ConfigError, match='category_rule'):
            TaskConfig(category_rule='above')


_SHIPPED = sorted(
    glob.glob(str(Path(__file__).resolve().parents[1] / 'src' / 'behav_utils' / 'config_presets' / '*.yaml'))
    + glob.glob(str(Path(__file__).resolve().parents[1] / 'docs' / 'config_examples' / '*.yaml')))


@pytest.mark.parametrize('path', _SHIPPED)
def test_shipped_configs_load(path):
    assert load_config(path).columns['trial_number'].csv_name
