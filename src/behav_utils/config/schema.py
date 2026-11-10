"""
Config Schema

Defines the structure of a behav_utils project config.
Loaded from YAML, validated on construction, used by the
loading pipeline to map CSV columns to internal field names.

Usage:
    from behav_utils.config import load_config

    config = load_config('config.yaml')
    # config.columns['stimulus'].csv_name  → 'Stim_Relative'
    # config.task.boundary                 → 0.0
    # config.reader.data_dir               → '/path/to/data'
"""

import dataclasses
import datetime
import difflib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Tuple,
    Union,
)

import numpy as np
import yaml


class ConfigError(ValueError):
    """A config that cannot be used as written: an unknown key, a missing required key, a bad value.

    A ``ValueError``, so code that caught ``ValueError`` from ``load_config`` still does.
    """

# =============================================================================
# COLUMN MAPPING
# =============================================================================

@dataclass
class ColumnMapping:
    """
    Maps one internal field name to a CSV column.

    Attributes:
        csv_name: Column name as it appears in the CSV file
        dtype: Expected data type ('int', 'float', 'str', 'bool')
        optional: If True, missing column is not an error (fills with default)
        default: Default value when column is missing or NaN
        mapping: Optional value mapping {csv_value: internal_value}
                 Applied after dtype conversion.
                 Example: {-1: 'left', 0: 'none', 1: 'right'}
    """
    csv_name: str
    dtype: str = 'float'
    optional: bool = False
    default: Any = None
    mapping: Dict[Any, Any] | None = None

    def __post_init__(self):
        valid_dtypes = {'int', 'float', 'str', 'bool'}
        if self.dtype not in valid_dtypes:
            raise ValueError(
                f"Column '{self.csv_name}': dtype must be one of "
                f"{valid_dtypes}, got '{self.dtype}'"
            )


@dataclass
class SessionMetadataMapping:
    """
    Columns constant within a session (extracted from first row).
    Same structure as ColumnMapping but semantically distinct:
    these don't vary trial-to-trial.
    """
    csv_name: str
    dtype: str = 'str'
    optional: bool = True
    default: Any = None
    parse_timespan: bool = False  # If True, parse HH:MM:SS.fff → seconds


# =============================================================================
# FILE STRUCTURE
# =============================================================================

@dataclass
class CsvFolderReaderConfig:
    """
    The ``csv_folders`` reader: data_dir / <animal> / <session folder> / <behaviour_file>.

    Every sub-directory of ``data_dir`` is an animal and every sub-directory of an animal a session
    folder; ``behaviour_file`` is a glob for the CSV(s) inside it. ``date_regex`` reads the date from a
    session's name with named groups ``year``, ``month``, ``day`` (or three positional groups first)
    and, optionally, ``hour``, ``minute``, ``second`` for the start time, which orders sessions within
    a day. With ``merge_files`` (the default) the files in one folder are pieces of one session, merged
    in name order; without it each file is a session of its own, dated by its file name (or by the
    folder when the file name has no date). ``drop_last_row`` drops each file's last row (a run that
    stops can leave it half-written); a file with fewer than ``min_trials_per_file`` rows is skipped.
    """
    TYPE: ClassVar[str] = 'csv_folders'
    data_dir: str = "."
    behaviour_file: str = "trial_summary*.csv"
    date_regex: str = r"(\d{4})_(\d{1,2})_(\d{1,2})"
    merge_files: bool = True
    drop_last_row: bool = True
    min_trials_per_file: int = 10


@dataclass
class TableReaderConfig:
    """
    The ``table`` reader: every trial of every session in one table (``.csv``, ``.tsv`` or
    ``.parquet``; Parquet needs pyarrow). ``animal_column`` and ``session_column`` identify a session,
    ``date_column`` dates it and the optional ``start_column`` orders sessions within a day.
    """
    TYPE: ClassVar[str] = 'table'
    path: str
    animal_column: str
    session_column: str
    date_column: str
    start_column: str | None = None


# =============================================================================
# TASK PARAMETERS
# =============================================================================

@dataclass
class ChoiceMapping:
    """
    Defines how raw choice values (e.g., spatial left/right) convert
    to category space (A/B).

    Attributes:
        type: Conversion type.
              'spatial_to_category' — uses contingency_field from session metadata
              'identity' — raw values are already in category space (0/1)
              'none' — no conversion, store raw values as-is

        no_response_value: Raw value meaning no response (→ NaN in category space); None when
                           no raw value does (missing cells are still no-responses)
        contingency_field: Session metadata field containing the response mapping name
        contingency_rules: {mapping_name: {raw_value: category_value}}
    """
    type: str = 'identity'
    no_response_value: Any = None
    contingency_field: str | None = None
    contingency_rules: Dict[str, Dict[Any, int]] = field(default_factory=dict)

@dataclass
class TaskConfig:
    """
    Task-level parameters (2-AFC or go/no-go categorisation).

        primary_stimulus: Which input column is THE stimulus for
                          psychometric fitting. None if there is none.
        primary_choice: Which output column is THE choice.
        boundary: Category boundary in *raw* stimulus units
        stimulus_range: Raw stimulus range (min, max); must bracket ``boundary``
        stimulus_scale: 'linear' or 'log' — the axis on which the raw stimulus
                        is mapped to the library's axis (see ``normalise_stimulus``)
        category_rule: How stimulus maps to category
        choice_mapping: How raw choice values convert to categories

    The library computes everything on a normalised stimulus axis: [-1, 1]
    with the category boundary at 0 (binning, the psychometric fit bounds, the
    hard/easy split). ``normalise_stimulus`` maps raw values onto it, piecewise
    linearly on the chosen scale: ``boundary → 0``, ``range[0] → -1``,
    ``range[1] → +1``. The default (range (-1, 1), boundary 0, linear) is the
    identity, so data already on that axis is untouched.
    """
    # Which columns are THE stimulus and THE choice
    primary_stimulus: str | None = None
    primary_choice: str | None = None

    # Category structure (only used if primary_stimulus is set)
    boundary: float = 0.0
    stimulus_range: Tuple[float, float] = (-1.0, 1.0)
    stimulus_scale: str = "linear"
    category_rule: str = "above_boundary"
    choice_mapping: ChoiceMapping = field(default_factory=ChoiceMapping)

    def __post_init__(self):
        if self.category_rule not in ('above_boundary', 'below_boundary'):
            raise ConfigError(f"task.category_rule must be 'above_boundary' or 'below_boundary', "
                              f"got {self.category_rule!r}")
        if self.stimulus_scale not in ('linear', 'log'):
            raise ConfigError(f"task.stimulus_scale must be 'linear' or 'log', got {self.stimulus_scale!r}")
        lo, hi = self.stimulus_range
        if not lo < self.boundary < hi:
            raise ConfigError(
                f"task.boundary ({self.boundary}) must lie strictly inside "
                f"task.stimulus_range ({lo}, {hi})")
        if self.stimulus_scale == 'log' and lo <= 0:
            raise ConfigError("task.stimulus_scale='log' needs a strictly positive stimulus_range")

    @property
    def is_identity_transform(self) -> bool:
        return (self.stimulus_scale == 'linear' and self.boundary == 0.0
                and tuple(self.stimulus_range) == (-1.0, 1.0))

    def normalise_stimulus(self, raw) -> np.ndarray:
        """Raw stimulus → [-1, 1] with the boundary at 0 (NaN passes through).

        Piecewise linear on the chosen scale, so an asymmetric range still puts
        the boundary at 0 and the range ends at ±1. Values outside the range map
        beyond ±1 (not clipped; the binning clips).
        """
        x = np.asarray(raw, dtype=float)
        if self.is_identity_transform:
            return x.copy()
        lo, hi, b = self.stimulus_range[0], self.stimulus_range[1], self.boundary
        if self.stimulus_scale == 'log':
            with np.errstate(invalid='ignore', divide='ignore'):
                x, lo, hi, b = np.log(x), np.log(lo), np.log(hi), np.log(b)
        out = np.where(x >= b, (x - b) / (hi - b), (x - b) / (b - lo))
        return out



# =============================================================================
# TOP-LEVEL CONFIG
# =============================================================================

@dataclass
class ProjectConfig:
    """
    Complete project configuration.

    Loaded from YAML via load_config(). Validated on construction.
    Passed to loading functions, analysis pipelines, and plotting.
    """
    name: str
    description: str = ""

    reader: Union[CsvFolderReaderConfig, TableReaderConfig] = field(default_factory=CsvFolderReaderConfig)
    animal_metadata: str | None = None      # JSON {animal_id: {...}}; see load_experiment
    experimenter: str = ''                  # whose data; an animal's metadata can override it
    task: TaskConfig = field(default_factory=TaskConfig)

    # Column mappings: internal_name → ColumnMapping
    columns: Dict[str, ColumnMapping] = field(default_factory=dict)

    # Session-level metadata columns
    session_metadata: Dict[str, SessionMetadataMapping] = field(
        default_factory=dict
    )

    # Extra columns to load but not map to specific fields
    extra_columns: List[str] = field(default_factory=list)

    # Session-type overrides, stamped onto sessions at load time:
    #   session_types: {type_name: {animal_id: ['YYYYMMDD', ...]}}
    # Type names are the project's own vocabulary (e.g. 'sham', 'washout');
    # the library only knows 'regular' and 'opto', which it derives from the data.
    session_types: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    # Named session presets (raw dicts from 'session_presets'), registered at load
    # time via behav_utils.data.ops.selection.register_presets_from_config.
    session_presets: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Sessions dropped at load time — never built, never in a snapshot.
    sessions_to_ignore: Dict[str, List[str]] = field(default_factory=dict)

    # Folder of the YAML file this config came from (None when built in code); relative paths
    # in the config resolve against it.
    source_dir: str | None = field(default=None, repr=False)

    def resolve_path(self, path: Union[str, Path]) -> Path:
        """``path`` with $ENV_VARS expanded: as given if absolute, else relative to the config
        file's folder (or the working directory for a config built in code)."""
        p = Path(os.path.expandvars(str(path)))
        if p.is_absolute() or self.source_dir is None:
            return p
        return Path(self.source_dir) / p

    def __post_init__(self):
        self._validate()

    def _validate(self):
        # trial_number is the only structurally required column
        if 'trial_number' not in self.columns:
            raise ValueError(
                "Config must define 'trial_number' column mapping."
            )

        # Auto-detect primary_stimulus/primary_choice from columns
        # if not explicitly set
        if self.task.primary_stimulus is None and 'stimulus' in self.columns:
            self.task.primary_stimulus = 'stimulus'
        if self.task.primary_choice is None and 'choice' in self.columns:
            self.task.primary_choice = 'choice'

        # If set, verify the referenced columns exist
        if self.task.primary_stimulus is not None:
            if self.task.primary_stimulus not in self.columns:
                raise ValueError(
                    f"task.primary_stimulus='{self.task.primary_stimulus}' "
                    f"but no column mapping with that name exists. "
                    f"Define it under 'columns:' or set primary_stimulus to null."
                )

        if self.task.primary_choice is not None:
            if self.task.primary_choice not in self.columns:
                raise ValueError(
                    f"task.primary_choice='{self.task.primary_choice}' "
                    f"but no column mapping with that name exists. "
                    f"Define it under 'columns:' or set primary_choice to null."
                )


    def get_csv_name(self, internal_name: str) -> str | None:
        """Get CSV column name for an internal field name."""
        if internal_name in self.columns:
            return self.columns[internal_name].csv_name
        if internal_name in self.session_metadata:
            return self.session_metadata[internal_name].csv_name
        return None

    def get_all_csv_columns(self) -> List[str]:
        """List all CSV column names referenced by this config."""
        cols = [m.csv_name for m in self.columns.values()]
        cols += [m.csv_name for m in self.session_metadata.values()]
        cols += self.extra_columns
        return cols

    @property
    def required_csv_columns(self) -> List[str]:
        """CSV columns that must be present (non-optional)."""
        return [
            m.csv_name for m in self.columns.values()
            if not m.optional
        ]

    @property
    def optional_csv_columns(self) -> List[str]:
        """CSV columns that are optional."""
        return [
            m.csv_name for m in self.columns.values()
            if m.optional
        ]


# =============================================================================
# YAML LOADING
# =============================================================================

# Every key load_config reads, per section. Anything else raises ConfigError.
_TOP_LEVEL_KEYS = (
    'project', 'reader', 'animal_metadata', 'task', 'columns', 'session_metadata', 'extra_columns',
    'session_types', 'session_presets', 'sessions_to_ignore', 'cohorts', 'custom',
)
_PROJECT_KEYS = ('name', 'description', 'experimenter')
_TASK_KEYS = ('primary_stimulus', 'primary_choice', 'boundary', 'stimulus_range', 'stimulus_scale',
              'category_rule', 'choice_mapping')
_CHOICE_MAPPING_KEYS = ('type', 'no_response_value', 'contingency_field', 'contingency_rules')
_CHOICE_MAPPING_TYPES = ('identity', 'spatial_to_category', 'none')
_COLUMN_KEYS = ('csv_name', 'dtype', 'optional', 'default', 'mapping')
_SESSION_METADATA_KEYS = ('csv_name', 'dtype', 'optional', 'default', 'parse_timespan')


def _check_keys(section: str, given: Mapping, allowed: Iterable[str]) -> None:
    """Raise ConfigError naming every key in ``given`` that ``allowed`` does not list."""
    allowed = sorted(allowed)
    unknown = [str(k) for k in given if k not in allowed]
    if not unknown:
        return
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, allowed, n=1)
        hints.append(f"{key!r} (did you mean {close[0]!r}?)" if close else repr(key))
    plural = 's' if len(unknown) > 1 else ''
    raise ConfigError(f"{section}: unknown key{plural} {', '.join(hints)}. Valid keys: {', '.join(allowed)}.")


def _parse_reader(raw: Dict[str, Any]) -> Union[CsvFolderReaderConfig, TableReaderConfig]:
    """The ``reader:`` block: ``type`` picks the reader, and each type has its own keys."""
    readers = {cls.TYPE: cls for cls in (CsvFolderReaderConfig, TableReaderConfig)}
    reader_type = raw.pop('type', CsvFolderReaderConfig.TYPE)
    if reader_type not in readers:
        raise ConfigError(f"reader.type must be one of {', '.join(readers)}, got {reader_type!r}")
    cls = readers[reader_type]
    known = [f.name for f in dataclasses.fields(cls)]
    _check_keys(f'reader ({reader_type})', raw, known)
    required = [f.name for f in dataclasses.fields(cls)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ConfigError(f"reader ({reader_type}) needs {', '.join(missing)}")
    for key in ('data_dir', 'path'):
        if isinstance(raw.get(key), str):
            raw[key] = os.path.expandvars(raw[key])
    return cls(**raw)


def _section(raw: Mapping, key: str, where: str = '') -> Dict[str, Any]:
    """``raw[key]`` as a dict ({} when absent or null); ConfigError if it is not a mapping."""
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where or key} must be a mapping, got {type(value).__name__}")
    return dict(value)

def _parse_column_mapping(name: str, spec) -> ColumnMapping:
    """Parse a single column mapping from YAML dict."""
    if isinstance(spec, str):
        return ColumnMapping(csv_name=spec)
    if not isinstance(spec, dict):
        raise ConfigError(f"columns.{name} must be a CSV column name or a mapping, got {type(spec).__name__}")
    _check_keys(f'columns.{name}', spec, _COLUMN_KEYS)

    mapping = spec.get('mapping', None)
    if isinstance(mapping, list):
        mapping = {pair[0]: pair[1] for pair in mapping}

    return ColumnMapping(
        csv_name=spec['csv_name'],
        dtype=spec.get('dtype', 'float'),
        optional=spec.get('optional', False),
        default=spec.get('default', None),
        mapping=mapping,
    )


def _parse_session_metadata(name: str, spec) -> SessionMetadataMapping:
    """Parse a session metadata mapping from YAML dict."""
    if isinstance(spec, str):
        return SessionMetadataMapping(csv_name=spec)
    if not isinstance(spec, dict):
        raise ConfigError(f"session_metadata.{name} must be a CSV column name or a mapping, "
                          f"got {type(spec).__name__}")
    _check_keys(f'session_metadata.{name}', spec, _SESSION_METADATA_KEYS)

    return SessionMetadataMapping(
        csv_name=spec['csv_name'],
        dtype=spec.get('dtype', 'str'),
        optional=spec.get('optional', True),
        default=spec.get('default', None),
        parse_timespan=spec.get('parse_timespan', False),
    )


def _session_entry(value, where: str) -> str:
    """One entry of a session list as text: a date ('YYYYMMDD') or a session id without the animal
    prefix ('20240315T0900', '20240315_2')."""
    if isinstance(value, bool) or isinstance(value, float):
        raise ConfigError(f"{where}: {value!r} is not a date or a session id; write entries as text")
    if isinstance(value, int):
        text = str(value)
        if len(text) != 8:
            raise ConfigError(f"{where}: {value} is not an 8-digit date. Quote session entries such as "
                              f"'20240315_2': unquoted, YAML reads 20240315_2 as the number 202403152.")
        return text
    if isinstance(value, datetime.date):
        return value.strftime('%Y%m%d')
    return str(value).strip()


def _normalise_session_dict(raw, where: str = 'session list') -> Dict[str, List[str]]:
    """Normalise a {animal_id: entries} mapping to {str: [str, ...]}.

    Accepts a list, a scalar, or None as the value for each animal. An entry is a date ('YYYYMMDD',
    every session that day) or a session id without the animal prefix (one session).
    """
    out: Dict[str, List[str]] = {}
    for aid, entries in (raw or {}).items():
        if entries is None:
            entries = []
        elif not isinstance(entries, list):
            entries = [entries]
        out[str(aid)] = [_session_entry(e, f'{where}.{aid}') for e in entries]
    return out


def load_cohorts(path: Union[str, Path]) -> Dict[str, List[str]]:
    """Load the ``cohorts:`` block from the project YAML.

    Cohort membership only (which animals belong to which cohort); per-animal
    attributes belong in the animal metadata, not here. Import-safe: reads the raw YAML and touches nothing else, so
    behaviour-only report scripts can call it without pulling in the fitting
    stack.

    Args:
        path: Path to the YAML config file.

    Returns:
        ``{cohort_key: [animal_id, ...]}`` (empty dict if no ``cohorts:`` block).

    Raises:
        FileNotFoundError: Config file doesn't exist.
        ValueError: ``cohorts:`` is present but not a mapping of lists.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    cohorts = (raw or {}).get('cohorts', {}) or {}
    if not isinstance(cohorts, dict):
        raise ValueError(f"'cohorts' must be a mapping, got {type(cohorts)}")
    out: Dict[str, List[str]] = {}
    for key, members in cohorts.items():
        if not isinstance(members, (list, tuple)):
            raise ValueError(
                f"cohort {key!r} must map to a list of animal ids, "
                f"got {type(members)}")
        out[str(key)] = [str(m) for m in members]
    return out


def load_config(path: Union[str, Path]) -> ProjectConfig:
    """
    Load and validate a project config from YAML.

    Args:
        path: Path to YAML config file

    Returns:
        Validated ProjectConfig

    Raises:
        FileNotFoundError: Config file doesn't exist
        ConfigError: Config is invalid: an unknown key (the message lists the valid ones and the
            closest match), a missing required key, a bad value. A ``ValueError``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"Config must be a YAML mapping, got {type(raw)}")
    misplaced = [k for k in ('name', 'description') if k in raw]
    if misplaced:
        raise ConfigError(f"{', '.join(map(repr, misplaced))} belong under 'project:', not at the top level")
    if 'file_structure' in raw:
        raise ConfigError("'file_structure' is now 'reader' (with type: csv_folders); its keys are unchanged, "
                          "plus merge_files. See docs/config_guide.md.")
    _check_keys('config', raw, _TOP_LEVEL_KEYS)

    project_raw = _section(raw, 'project')
    _check_keys('project', project_raw, _PROJECT_KEYS)

    # Reader and animal metadata
    reader = _parse_reader(_section(raw, 'reader'))
    animal_metadata = raw.get('animal_metadata')
    if animal_metadata is not None and not isinstance(animal_metadata, str):
        raise ConfigError(f"animal_metadata must be a path, got {type(animal_metadata).__name__}")
    if animal_metadata is not None:
        animal_metadata = os.path.expandvars(animal_metadata)

    # Task
    task_raw = _section(raw, 'task')
    _check_keys('task', task_raw, _TASK_KEYS)
    stim_range = task_raw.pop('stimulus_range', [-1.0, 1.0])
    if len(stim_range) != 2:
        raise ValueError(f"task.stimulus_range must be [min, max], got {stim_range}")

    # Parse choice mapping
    cm_raw = _section(task_raw, 'choice_mapping', 'task.choice_mapping')
    task_raw.pop('choice_mapping', None)
    _check_keys('task.choice_mapping', cm_raw, _CHOICE_MAPPING_KEYS)
    mapping_type = cm_raw.get('type', 'identity')
    if mapping_type not in _CHOICE_MAPPING_TYPES:
        raise ConfigError(f"task.choice_mapping.type must be one of {', '.join(_CHOICE_MAPPING_TYPES)}, "
                          f"got {mapping_type!r}")
    columns_raw = raw.get('columns') or {}
    choice_column = task_raw.get('primary_choice', 'choice' if 'choice' in columns_raw else None)
    if (choice_column is not None and mapping_type != 'none'
            and 'no_response_value' not in cm_raw):
        raise ConfigError(
            "task.choice_mapping.no_response_value is required when a choice column is mapped: the raw "
            "value that means no response, or null when no raw value does (blank cells are always "
            "no-responses). Nothing is assumed, because a default of 0 would turn every A choice in a "
            "0/1 column into a no-response.")
    if mapping_type == 'spatial_to_category' and not cm_raw.get('contingency_field'):
        raise ConfigError("task.choice_mapping.contingency_field is required for type "
                          "'spatial_to_category': the session-metadata field whose value picks the rule")
    choice_mapping = ChoiceMapping(
        type=mapping_type,
        no_response_value=cm_raw.get('no_response_value'),
        contingency_field=cm_raw.get('contingency_field'),
        contingency_rules={
            str(k): {
                (int(rk) if isinstance(rk, (int, float)) else rk): int(rv)
                for rk, rv in v.items()
            }
            for k, v in cm_raw.get('contingency_rules', {}).items()
        },
    )

    task = TaskConfig(
        primary_stimulus=task_raw.get('primary_stimulus', None),
        primary_choice=task_raw.get('primary_choice', None),
        stimulus_range=(float(stim_range[0]), float(stim_range[1])),
        choice_mapping=choice_mapping,
        **{k: task_raw[k] for k in TaskConfig.__dataclass_fields__
           if k in task_raw and k not in (
               'stimulus_range', 'choice_mapping',
               'primary_stimulus', 'primary_choice',
           )},
    )
    # Columns
    columns = {}
    for name, spec in _section(raw, 'columns').items():
        columns[name] = _parse_column_mapping(name, spec)

    # Session metadata
    session_metadata = {}
    for name, spec in _section(raw, 'session_metadata').items():
        session_metadata[name] = _parse_session_metadata(name, spec)

    # Session-type overrides: {type_name: {animal_id: ['YYYYMMDD', ...]}}
    session_types = {name: _normalise_session_dict(m or {}, f'session_types.{name}')
                     for name, m in (raw.get('session_types') or {}).items()}
    sessions_to_ignore = _normalise_session_dict(raw.get('sessions_to_ignore', {}), 'sessions_to_ignore')

    return ProjectConfig(
        name=project_raw.get('name', 'Unnamed Project'),
        description=project_raw.get('description', ''),
        experimenter=str(project_raw.get('experimenter') or ''),
        reader=reader,
        animal_metadata=animal_metadata,
        source_dir=str(path.resolve().parent),
        task=task,
        columns=columns,
        session_metadata=session_metadata,
        extra_columns=raw.get('extra_columns', []),
        session_types=session_types,
        session_presets=dict(raw.get('session_presets') or {}),
        sessions_to_ignore=sessions_to_ignore,
    )


def validate_csv_against_config(
    df_columns: List[str],
    config: ProjectConfig,
) -> Dict[str, List[str]]:
    """
    Check whether a CSV's columns match the config expectations.

    Returns dict with 'matched', 'missing_required', 'missing_optional',
    'unmapped' (CSV columns not in config).
    """
    csv_cols = set(df_columns)
    matched, missing_required, missing_optional = [], [], []

    for mapping in config.columns.values():
        if mapping.csv_name in csv_cols:
            matched.append(mapping.csv_name)
        elif mapping.optional:
            missing_optional.append(mapping.csv_name)
        else:
            missing_required.append(mapping.csv_name)

    for mapping in config.session_metadata.values():
        if mapping.csv_name in csv_cols:
            matched.append(mapping.csv_name)
        elif not mapping.optional:
            missing_required.append(mapping.csv_name)

    all_expected = set(config.get_all_csv_columns())
    unmapped = sorted(csv_cols - all_expected)

    return {
        'matched': matched,
        'missing_required': missing_required,
        'missing_optional': missing_optional,
        'unmapped': unmapped,
    }
