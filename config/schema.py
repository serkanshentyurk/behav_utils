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
    # config.file_structure.data_dir       → '/path/to/data'
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Optional, Dict, List, Tuple, Any, Union,
)


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
    mapping: Optional[Dict[Any, Any]] = None

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
class FileStructure:
    """
    How data files are organised on disk.

    Assumes: data_dir / {animal_dir} / {session_dir} / {behaviour_file}
    Patterns use Python format strings with named placeholders.
    """
    data_dir: str = "."
    animal_pattern: str = "{animal_id}"
    session_pattern: str = "{protocol}_{animal_id}_{date}"
    behaviour_file: str = "trial_summary*.csv"
    date_format: str = "{year}_{month}_{day}"
    date_regex: str = r"(\d{4})_(\d{1,2})_(\d{1,2})"


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

        no_response_value: Raw value indicating no response (→ NaN in category space)
        contingency_field: Session metadata field containing the response mapping name
        contingency_rules: {mapping_name: {raw_value: category_value}}
    """
    type: str = 'identity'
    no_response_value: Any = 0
    contingency_field: str = 'sound_contingency'
    contingency_rules: Dict[str, Dict[Any, int]] = field(default_factory=dict)

@dataclass
class TaskConfig:
    """
    Task-level parameters that define the experimental paradigm.
    """
    boundary: float = 0.0
    stimulus_range: Tuple[float, float] = (-1.0, 1.0)
    n_categories: int = 2
    category_rule: str = "above_boundary"
    choice_mapping: ChoiceMapping = field(default_factory=ChoiceMapping)


# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

@dataclass
class AnalysisConfig:
    """
    Default analysis parameters. Can be overridden per-call.
    """
    excluded_stats: List[str] = field(default_factory=list)
    hard_threshold: float = 0.3
    default_n_bins: int = 8
    min_valid_trials: int = 10
    default_stage: Optional[str] = None


# =============================================================================
# PLOTTING DEFAULTS
# =============================================================================

@dataclass
class PlottingConfig:
    """
    Default plotting parameters.
    """
    dpi: int = 100
    font_size: int = 10
    figure_width: float = 10.0
    colourmap: str = 'tab10'
    model_colours: Dict[str, str] = field(default_factory=lambda: {
        'default': 'steelblue',
    })


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

    file_structure: FileStructure = field(default_factory=FileStructure)
    task: TaskConfig = field(default_factory=TaskConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)

    # Column mappings: internal_name → ColumnMapping
    columns: Dict[str, ColumnMapping] = field(default_factory=dict)

    # Session-level metadata columns
    session_metadata: Dict[str, SessionMetadataMapping] = field(
        default_factory=dict
    )

    # Extra columns to load but not map to specific fields
    extra_columns: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Validate required columns are present."""
        required = {'trial_number', 'stimulus', 'choice', 'outcome'}
        missing = required - set(self.columns.keys())
        if missing:
            raise ValueError(
                f"Config missing required column mappings: {missing}. "
                f"These must be defined under 'columns:' in the YAML."
            )

    def get_csv_name(self, internal_name: str) -> Optional[str]:
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

def _parse_column_mapping(name: str, spec) -> ColumnMapping:
    """Parse a single column mapping from YAML dict."""
    if isinstance(spec, str):
        return ColumnMapping(csv_name=spec)

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

    return SessionMetadataMapping(
        csv_name=spec['csv_name'],
        dtype=spec.get('dtype', 'str'),
        optional=spec.get('optional', True),
        default=spec.get('default', None),
        parse_timespan=spec.get('parse_timespan', False),
    )


def load_config(path: Union[str, Path]) -> ProjectConfig:
    """
    Load and validate a project config from YAML.

    Args:
        path: Path to YAML config file

    Returns:
        Validated ProjectConfig

    Raises:
        FileNotFoundError: Config file doesn't exist
        ValueError: Config is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw)}")

    # File structure
    fs_raw = raw.get('file_structure', {})
    file_structure = FileStructure(**{
        k: fs_raw[k] for k in FileStructure.__dataclass_fields__
        if k in fs_raw
    })

    # Task
    task_raw = raw.get('task', {})
    stim_range = task_raw.pop('stimulus_range', [-1.0, 1.0])

    # Parse choice mapping
    cm_raw = task_raw.pop('choice_mapping', {})
    choice_mapping = ChoiceMapping(
        type=cm_raw.get('type', 'identity'),
        no_response_value=cm_raw.get('no_response_value', 0),
        contingency_field=cm_raw.get('contingency_field', 'sound_contingency'),
        contingency_rules={
            str(k): {
                (int(rk) if isinstance(rk, (int, float)) else rk): int(rv)
                for rk, rv in v.items()
            }
            for k, v in cm_raw.get('contingency_rules', {}).items()
        },
    )

    task = TaskConfig(
        stimulus_range=tuple(stim_range),
        choice_mapping=choice_mapping,
        **{k: task_raw[k] for k in TaskConfig.__dataclass_fields__
           if k in task_raw and k not in ('stimulus_range', 'choice_mapping')},
    )

    # Analysis
    analysis_raw = raw.get('analysis', {})
    analysis = AnalysisConfig(**{
        k: analysis_raw[k] for k in AnalysisConfig.__dataclass_fields__
        if k in analysis_raw
    })

    # Plotting
    plot_raw = raw.get('plotting', {})
    plotting = PlottingConfig(**{
        k: plot_raw[k] for k in PlottingConfig.__dataclass_fields__
        if k in plot_raw
    })

    # Columns
    columns = {}
    for name, spec in raw.get('columns', {}).items():
        columns[name] = _parse_column_mapping(name, spec)

    # Session metadata
    session_metadata = {}
    for name, spec in raw.get('session_metadata', {}).items():
        session_metadata[name] = _parse_session_metadata(name, spec)

    return ProjectConfig(
        name=raw.get('project', {}).get('name', 'Unnamed Project'),
        description=raw.get('project', {}).get('description', ''),
        file_structure=file_structure,
        task=task,
        analysis=analysis,
        plotting=plotting,
        columns=columns,
        session_metadata=session_metadata,
        extra_columns=raw.get('extra_columns', []),
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

    for name, mapping in config.columns.items():
        if mapping.csv_name in csv_cols:
            matched.append(mapping.csv_name)
        elif mapping.optional:
            missing_optional.append(mapping.csv_name)
        else:
            missing_required.append(mapping.csv_name)

    for name, mapping in config.session_metadata.items():
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
