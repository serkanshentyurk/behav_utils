"""
Config-Driven Data Loading

Loads behavioural data from CSV files into the data class hierarchy,
using a ProjectConfig to map CSV columns to internal field names.

The loader handles:
    - Directory traversal (animal/session/file structure from config)
    - Column mapping and dtype enforcement
    - Value mapping (e.g., {-1: 'left', 0: 'none', 1: 'right'})
    - TimeSpan parsing for session metadata
    - Category derivation from stimulus + boundary
    - Robust handling of truncated/malformed CSVs
    - Validation of loaded data against config

Usage:
    from behav_utils.config import load_config
    from behav_utils.data.loading import load_experiment

    config = load_config('config.yaml')
    experiment = load_experiment(config)
    # or:
    experiment = load_experiment('config.yaml')  # loads config automatically
"""

import numpy as np
import pandas as pd
import warnings
import re
import glob
from pathlib import Path
from datetime import date
from typing import Optional, List, Dict, Tuple, Union

from behav_utils.config.schema import (
    ProjectConfig, ColumnMapping, SessionMetadataMapping,
    ChoiceMapping, load_config, validate_csv_against_config,
)
from behav_utils.data.structures import (
    ExperimentData, AnimalData, SessionData, SessionMetadata, TrialData,
)


# =============================================================================
# CHOICE ENCODING CONVERSION
# =============================================================================

def convert_choice_to_category(
    choice_raw: np.ndarray,
    metadata: SessionMetadata,
    choice_mapping: ChoiceMapping,
) -> np.ndarray:
    """
    Convert raw choice values to category space using session metadata.

    Returns:
        choice_category: 0=A, 1=B, NaN=no response
    """
    n = len(choice_raw)
    choice_cat = np.full(n, np.nan)

    if choice_mapping.type == 'identity':
        choice_cat = choice_raw.astype(float).copy()
        no_resp = (choice_raw == choice_mapping.no_response_value)
        choice_cat[no_resp] = np.nan
        return choice_cat

    if choice_mapping.type == 'none':
        return choice_raw.astype(float).copy()

    if choice_mapping.type == 'spatial_to_category':
        contingency = metadata.get(choice_mapping.contingency_field, None)

        if contingency is None:
            warnings.warn(
                f"No '{choice_mapping.contingency_field}' in session metadata. "
                f"Cannot convert choice. Storing raw values."
            )
            return choice_raw.astype(float).copy()

        contingency = str(contingency)

        if contingency not in choice_mapping.contingency_rules:
            warnings.warn(
                f"Unknown contingency '{contingency}'. "
                f"Known: {list(choice_mapping.contingency_rules.keys())}. "
                f"Storing raw values."
            )
            return choice_raw.astype(float).copy()

        rules = choice_mapping.contingency_rules[contingency]
        no_resp = (choice_raw == choice_mapping.no_response_value)

        for raw_val, cat_val in rules.items():
            try:
                mask = (choice_raw == raw_val)
            except (ValueError, TypeError):
                mask = (choice_raw.astype(str) == str(raw_val))
            choice_cat[mask] = float(cat_val)

        choice_cat[no_resp] = np.nan
        return choice_cat

    raise ValueError(f"Unknown choice_mapping type: '{choice_mapping.type}'")


# =============================================================================
# TIMESPAN PARSING
# =============================================================================

def parse_timespan(val, default=None) -> Optional[float]:
    """
    Parse Bonsai TimeSpan string (HH:MM:SS.fff) to seconds.
    Returns float seconds, or default if unparseable.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    # Already numeric?
    try:
        return float(val)
    except (ValueError, TypeError):
        pass
    # Parse HH:MM:SS or HH:MM:SS.fffffff
    try:
        parts = str(val).split(':')
        h = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        s = float(parts[2]) if len(parts) > 2 else 0.0
        return h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return default


# =============================================================================
# DATE PARSING
# =============================================================================

def parse_date_from_path(path_str: str, regex: str) -> Optional[date]:
    """
    Extract date from a path string using a regex with named groups
    (year, month, day) or positional groups.
    """
    match = re.search(regex, str(path_str))
    if match:
        try:
            groups = match.groupdict()
            if groups:
                return date(
                    int(groups['year']),
                    int(groups['month']),
                    int(groups['day']),
                )
            else:
                # Positional groups
                g = match.groups()
                return date(int(g[0]), int(g[1]), int(g[2]))
        except (ValueError, IndexError, KeyError):
            pass
    return None


def parse_date_from_csv(df: pd.DataFrame) -> Optional[date]:
    """Extract date from a 'Date' column if present."""
    if 'Date' in df.columns and len(df) > 0:
        date_val = df['Date'].iloc[0]
        if pd.notna(date_val):
            try:
                return pd.to_datetime(date_val).date()
            except (ValueError, TypeError):
                pass
    return None


# =============================================================================
# COLUMN EXTRACTION
# =============================================================================

def _safe_column(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    n_rows: int,
) -> np.ndarray:
    """
    Extract and convert a single column from a DataFrame using a ColumnMapping.
    Handles missing columns, dtype conversion, and value mapping.
    """
    if mapping.csv_name not in df.columns:
        if mapping.optional:
            default = mapping.default
            if mapping.dtype == 'float':
                return np.full(n_rows, default if default is not None else np.nan)
            elif mapping.dtype == 'int':
                return np.full(n_rows, default if default is not None else 0, dtype=int)
            elif mapping.dtype == 'bool':
                return np.full(n_rows, default if default is not None else False, dtype=bool)
            elif mapping.dtype == 'str':
                return np.full(n_rows, default if default is not None else '', dtype=object)
        else:
            raise KeyError(
                f"Required column '{mapping.csv_name}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

    raw = df[mapping.csv_name].values.copy()

    # Apply value mapping first (before dtype conversion)
    if mapping.mapping is not None:
        mapped = np.empty(len(raw), dtype=object)
        for i, val in enumerate(raw):
            mapped[i] = mapping.mapping.get(val, val)
        raw = mapped

    # Dtype conversion
    try:
        if mapping.dtype == 'float':
            result = pd.to_numeric(raw, errors='coerce').astype(float)
        elif mapping.dtype == 'int':
            # Handle NaN → fill with default then convert
            numeric = pd.to_numeric(raw, errors='coerce')
            default = mapping.default if mapping.default is not None else 0
            result = np.where(np.isnan(numeric), default, numeric).astype(int)
        elif mapping.dtype == 'bool':
            if raw.dtype == bool:
                result = raw.astype(bool)
            else:
                # Handle string 'True'/'False', numeric 0/1
                result = pd.array(raw, dtype='boolean').fillna(
                    mapping.default if mapping.default is not None else False
                ).to_numpy(dtype=bool)
        elif mapping.dtype == 'str':
            result = np.array([str(v) if pd.notna(v) else '' for v in raw],
                              dtype=object)
        else:
            result = raw
    except (ValueError, TypeError) as e:
        warnings.warn(
            f"Column '{mapping.csv_name}': dtype conversion to "
            f"'{mapping.dtype}' failed: {e}. Using raw values."
        )
        result = raw

    return result


def _extract_session_metadata(
    df: pd.DataFrame,
    config: ProjectConfig,
) -> SessionMetadata:
    """
    Extract session-level metadata from the first row of a CSV.
    """
    if len(df) == 0:
        return SessionMetadata()

    row = df.iloc[0]
    fields = {}

    for name, mapping in config.session_metadata.items():
        val = row.get(mapping.csv_name, mapping.default)

        if pd.isna(val):
            val = mapping.default
        elif mapping.parse_timespan:
            val = parse_timespan(val, default=mapping.default)
        elif mapping.dtype == 'float':
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = mapping.default
        elif mapping.dtype == 'int':
            try:
                val = int(float(val))
            except (ValueError, TypeError):
                val = mapping.default
        elif mapping.dtype == 'str':
            val = str(val)

        fields[name] = val

    return SessionMetadata(fields=fields)


# =============================================================================
# SESSION LOADING
# =============================================================================

def load_session_csv(
    csv_path: Union[str, Path],
    config: ProjectConfig,
    session_idx: int = 0,
    session_date: Optional[date] = None,
) -> SessionData:
    """
    Load a single session CSV into a SessionData object.

    Args:
        csv_path: Path to CSV file
        config: Project config with column mappings
        session_idx: Ordinal index (set by caller)
        session_date: Session date (extracted from path if not provided)

    Returns:
        SessionData object
    """
    csv_path = Path(csv_path)

    # Read CSV with robust parsing
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        raise IOError(f"Failed to read {csv_path}: {e}")

    if len(df) == 0:
        warnings.warn(f"Empty CSV: {csv_path}")

    # Validate columns
    validation = validate_csv_against_config(list(df.columns), config)
    if validation['missing_required']:
        raise ValueError(
            f"CSV {csv_path.name} missing required columns: "
            f"{validation['missing_required']}"
        )

    n_rows = len(df)

    # ── Extract session metadata ────────────────────────────────────────────
    metadata = _extract_session_metadata(df, config)

    # ── Extract trial-level columns ─────────────────────────────────────────
    # Required columns
    trial_number = _safe_column(df, config.columns['trial_number'], n_rows)
    stimulus = _safe_column(df, config.columns['stimulus'], n_rows)
    choice = _safe_column(df, config.columns['choice'], n_rows)
    outcome = _safe_column(df, config.columns['outcome'], n_rows)
    correct = _safe_column(df, config.columns['correct'], n_rows)

    # Derive category from stimulus + boundary
    category = (stimulus > config.task.boundary).astype(int)
    if config.task.category_rule == 'below_boundary':
        category = 1 - category

    # Optional columns
    reaction_time = _safe_column(
        df, config.columns['reaction_time'], n_rows
    ) if 'reaction_time' in config.columns else np.full(n_rows, np.nan)

    abort = _safe_column(
        df, config.columns['abort'], n_rows
    ) if 'abort' in config.columns else np.zeros(n_rows, dtype=bool)

    opto_on = _safe_column(
        df, config.columns['opto_on'], n_rows
    ) if 'opto_on' in config.columns else np.zeros(n_rows, dtype=bool)

    # All other mapped optional columns
    optional_fields = {}
    skip_names = {
        'trial_number', 'stimulus', 'choice', 'outcome', 'correct',
        'reaction_time', 'abort', 'opto_on',
    }
    for name, mapping in config.columns.items():
        if name in skip_names:
            continue
        try:
            optional_fields[name] = _safe_column(df, mapping, n_rows)
        except KeyError:
            pass  # Optional column not present

    # Extra columns (unmapped)
    extra = {}
    for col_name in config.extra_columns:
        if col_name in df.columns:
            extra[col_name] = df[col_name].values

    # Also capture truly unmapped columns
    all_mapped = set(config.get_all_csv_columns())
    for col in df.columns:
        if col not in all_mapped and col not in extra:
            extra[col] = df[col].values

    # ── Build TrialData ─────────────────────────────────────────────────────
    # Convert choice to category space
    choice_raw = choice.copy()
    choice_category = convert_choice_to_category(
        choice_raw, metadata, config.task.choice_mapping,
    )

    trials = TrialData(
        trial_number=trial_number,
        stimulus=stimulus,
        choice=choice_category,       # category space (0/1/NaN)
        choice_raw=choice_raw,        # preserve original
        outcome=outcome,
        correct=correct,
        category=category,
        reaction_time=reaction_time,
        abort=abort,
        opto_on=opto_on,
        optional_fields=optional_fields,
        extra=extra,
    )

    # ── Resolve date ────────────────────────────────────────────────────────
    if session_date is None:
        session_date = parse_date_from_csv(df)
    if session_date is None:
        session_date = parse_date_from_path(
            str(csv_path), config.file_structure.date_regex,
        )
    if session_date is None:
        session_date = date(2000, 1, 1)  # fallback
        warnings.warn(f"Could not determine date for {csv_path.name}")

    # ── Build session ID ────────────────────────────────────────────────────
    animal_id = metadata.animal_id or csv_path.parent.parent.name
    session_id = csv_path.parent.name or f"{animal_id}_S{session_idx:03d}"

    return SessionData(
        session_id=session_id,
        session_idx=session_idx,
        date=session_date,
        metadata=metadata,
        trials=trials,
        csv_path=str(csv_path),
    )


# =============================================================================
# ANIMAL LOADING
# =============================================================================

def load_animal(
    animal_dir: Union[str, Path],
    config: ProjectConfig,
) -> AnimalData:
    """
    Load all sessions for one animal from a directory.

    Expects: animal_dir / {session_dirs} / {behaviour_file}
    """
    animal_dir = Path(animal_dir)
    animal_id = animal_dir.name

    # Find session directories
    session_dirs = sorted([
        d for d in animal_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    sessions = []
    for idx, sess_dir in enumerate(session_dirs):
        # Find behaviour CSV
        pattern = config.file_structure.behaviour_file
        csv_files = sorted(glob.glob(str(sess_dir / pattern)))

        if not csv_files:
            continue

        csv_path = csv_files[0]  # Take first match

        # Extract date from directory name
        sess_date = parse_date_from_path(
            sess_dir.name, config.file_structure.date_regex,
        )

        try:
            session = load_session_csv(
                csv_path, config,
                session_idx=len(sessions),
                session_date=sess_date,
            )
            sessions.append(session)
        except Exception as e:
            warnings.warn(f"Failed to load {csv_path}: {e}")
            continue

    return AnimalData(animal_id=animal_id, sessions=sessions)


# =============================================================================
# EXPERIMENT LOADING
# =============================================================================

def load_experiment(
    config_or_path: Union[ProjectConfig, str, Path],
) -> ExperimentData:
    """
    Load full experiment from directory structure defined in config.

    Args:
        config_or_path: ProjectConfig object or path to YAML config file.
                        If a path, loads the config automatically.

    Returns:
        ExperimentData with all animals loaded
    """
    if isinstance(config_or_path, (str, Path)):
        config = load_config(config_or_path)
    else:
        config = config_or_path

    data_dir = Path(config.file_structure.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find animal directories
    animal_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    experiment = ExperimentData(config=config)

    for animal_dir in animal_dirs:
        try:
            animal = load_animal(animal_dir, config)
            if animal.n_sessions > 0:
                experiment.add_animal(animal)
            else:
                warnings.warn(
                    f"Animal {animal.animal_id}: no valid sessions found"
                )
        except Exception as e:
            warnings.warn(f"Failed to load animal {animal_dir.name}: {e}")
            continue

    print(
        f"Loaded {experiment.n_animals} animals, "
        f"{sum(a.n_sessions for a in experiment.animals.values())} total sessions"
    )

    return experiment


# =============================================================================
# CONVENIENCE: LOAD WITH JUST A DATA DIR
# =============================================================================

def load_from_directory(
    data_dir: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    **config_overrides,
) -> ExperimentData:
    """
    Load experiment with minimal setup.

    If config_path is provided, loads that config (overriding data_dir).
    If not, looks for config.yaml in data_dir or creates a minimal default.
    """
    data_dir = Path(data_dir)

    if config_path is not None:
        config = load_config(config_path)
    elif (data_dir / 'config.yaml').exists():
        config = load_config(data_dir / 'config.yaml')
    elif (data_dir.parent / 'config.yaml').exists():
        config = load_config(data_dir.parent / 'config.yaml')
    else:
        raise FileNotFoundError(
            f"No config.yaml found in {data_dir} or parent directory. "
            f"Provide config_path explicitly or create a config.yaml."
        )

    # Override data_dir if needed
    config.file_structure.data_dir = str(data_dir)

    return load_experiment(config)
