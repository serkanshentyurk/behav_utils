"""
Loading: raw trial tables → SessionData → ExperimentData.

Two halves. A reader (``behav_utils.data.readers``) finds sessions and reads their trial tables; it
knows the data layout and nothing else. ``build_session`` maps one raw table onto the library's
structures — required columns, session metadata, column mapping, stimulus normalisation, choice
conversion, category, correctness, session type — the same way for every reader.

    experiment = load_experiment('config.yaml')                   # the reader named in the config
    experiment = load_experiment(config, reader=my_reader)        # any callable: config -> RawSession
    animal     = load_animal('/data/A01', config)                 # one animal folder
    session    = load_session_csv('/data/A01/S_2026_01_12/trials.csv', config)   # one file

Sessions in ``sessions_to_ignore`` are dropped before their tables are read. A session that fails
raises under ``strict`` (the default) and is logged and skipped otherwise. Progress goes to the
``behav_utils.data.loading`` logger.
"""

import inspect
import json
import logging
import re
import warnings
from dataclasses import replace
from datetime import date, time
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd

from behav_utils.config.schema import (
    ChoiceMapping,
    ColumnMapping,
    CsvFolderReaderConfig,
    ProjectConfig,
    load_config,
    validate_csv_against_config,
)
from behav_utils.data.arrays import TRIAL_BITS
from behav_utils.data.readers import (
    RawSession,
    Reader,
    _animal_sessions,
    _read_pieces,
    _reader_from_config,
    _trial_column,
    parse_session_datetime,
)
from behav_utils.data.structures import (
    AnimalData,
    ExperimentData,
    SessionData,
    SessionMetadata,
    TrialData,
)

log = logging.getLogger(__name__)

# Fraction of NaN (no-response) choices above which the builder warns: usually a
# mis-encoded choice column (e.g. go/no-go with the withheld response left blank).
NAN_CHOICE_WARN_FRACTION = 0.3


# =============================================================================
# CHOICE ENCODING
# =============================================================================

def _no_response_mask(choice_raw: np.ndarray, value) -> np.ndarray:
    """Raw choices equal to the configured no-response value; none when the value is None."""
    if value is None:
        return np.zeros(len(choice_raw), dtype=bool)
    mask = np.asarray(choice_raw == value)
    if mask.ndim == 0:
        mask = np.full(len(choice_raw), bool(mask))
    return mask.astype(bool)


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
        no_resp = _no_response_mask(choice_raw, choice_mapping.no_response_value)
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
            , stacklevel=2)
            return choice_raw.astype(float).copy()

        contingency = str(contingency)

        if contingency not in choice_mapping.contingency_rules:
            warnings.warn(
                f"Unknown contingency '{contingency}'. "
                f"Known: {list(choice_mapping.contingency_rules.keys())}. "
                f"Storing raw values."
            , stacklevel=2)
            return choice_raw.astype(float).copy()

        rules = choice_mapping.contingency_rules[contingency]
        no_resp = _no_response_mask(choice_raw, choice_mapping.no_response_value)

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
# PARSERS AND COLUMN EXTRACTION
# =============================================================================

def parse_timespan(val, default=None) -> float | None:
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
                # Robust bool coercion: handles 'True'/'False', 'T'/'F',
                # 1/0, 'NaN', nan, empty strings, and mixed types.
                default = mapping.default if mapping.default is not None else False
                _TRUE = {'true', '1', 'yes', 't', '1.0'}
                _FALSE = {'false', '0', 'no', 'f', '0.0'}

                def _parse_bool(val):
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        if np.isnan(val):
                            return default
                        return bool(val)
                    s = str(val).strip().lower()
                    if s in _TRUE:
                        return True
                    if s in _FALSE:
                        return False
                    return default

                result = np.array([_parse_bool(v) for v in raw], dtype=bool)
        elif mapping.dtype == 'str':
            result = np.array([str(v) if pd.notna(v) else '' for v in raw],
                              dtype=object)
        else:
            result = raw
    except (ValueError, TypeError) as e:
        warnings.warn(
            f"Column '{mapping.csv_name}': dtype conversion to "
            f"'{mapping.dtype}' failed: {e}. Using raw values."
        , stacklevel=2)
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
# BUILDING ONE SESSION
# =============================================================================

def build_session(raw: RawSession, config: ProjectConfig, *, strict: bool = True,
                  session_idx: int = 0, source_id: str | None = None) -> SessionData:
    """
    One raw session → SessionData: the half of loading that every reader shares.

    Checks the required columns, reads session metadata from the first row, maps and coerces the
    trial columns, normalises the stimulus, converts the choice and derives the category,
    correctness and session type. The animal id is the reader's; a mapped ``animal_id`` column that
    disagrees raises (``strict``) or warns and is overwritten. Trial numbers must be unique (repeats
    raise, or with ``strict=False`` warn and are renumbered from 1) and lie in 0..1,048,575, the trial
    key's range. ``AnimalData`` sets ``session_idx``; the loader passes the canonical session id in
    ``raw`` and the reader's own name as ``source_id``.

    Raises:
        ValueError: no date, no trials, a missing required column, or (strict) an animal-id mismatch.
    """
    label = f'{raw.animal_id}/{raw.session_id}'
    if raw.date is None:
        raise ValueError(f'{label}: no date (check reader.date_regex, or the table reader\'s date column)')
    df = raw.table()
    if len(df) == 0:
        raise ValueError(f'{label}: no trials')
    validation = validate_csv_against_config(list(df.columns), config)
    if validation['missing_required']:
        raise ValueError(f"{label}: missing required columns {validation['missing_required']}")

    metadata = _extract_session_metadata(df, config)
    fields = {**dict(raw.metadata), **metadata.fields}
    recorded = fields.get('animal_id')
    if recorded not in (None, '') and str(recorded) != raw.animal_id:
        message = f'{label}: the animal_id column says {recorded!r} but the reader says {raw.animal_id!r}'
        if strict:
            raise ValueError(message)
        warnings.warn(message + "; keeping the reader's id", stacklevel=2)
    fields['animal_id'] = raw.animal_id
    metadata = SessionMetadata(fields=fields)

    # ── Extract trial-level columns ─────────────────────────────────────────
    n_rows = len(df)

    # trial_number is always required
    trial_number = _safe_column(df, config.columns['trial_number'], n_rows)

    # Primary stimulus (for category derivation and psychometric analysis)
    primary_stim_name = config.task.primary_stimulus
    if primary_stim_name and primary_stim_name in config.columns:
        stimulus_raw = np.asarray(_safe_column(df, config.columns[primary_stim_name], n_rows), dtype=float)
        # Library axis: [-1, 1], boundary at 0 (identity for data already there)
        stimulus = config.task.normalise_stimulus(stimulus_raw)
        category = (stimulus > 0.0).astype(int)
        if config.task.category_rule == 'below_boundary':
            category = 1 - category
    else:
        stimulus_raw = np.full(n_rows, np.nan)
        stimulus = np.full(n_rows, np.nan)
        category = np.full(n_rows, 0, dtype=int)

    # Primary choice (for psychometric analysis)
    primary_choice_name = config.task.primary_choice
    if primary_choice_name and primary_choice_name in config.columns:
        choice_raw_arr = _safe_column(df, config.columns[primary_choice_name], n_rows)
        choice_category = convert_choice_to_category(
            choice_raw_arr, metadata, config.task.choice_mapping,
        )
        nan_frac = float(np.mean(np.isnan(choice_category))) if n_rows else 0.0
        if nan_frac > NAN_CHOICE_WARN_FRACTION:
            warnings.warn(
                f"{label}: {nan_frac:.0%} of choices are no-response (NaN). If this is a "
                f"go/no-go task, code the withheld response as 0 via columns.choice.mapping — "
                f"NaN choices are dropped from every statistic.", stacklevel=2)
    else:
        choice_raw_arr = np.full(n_rows, np.nan)
        choice_category = np.full(n_rows, np.nan)

    # Outcome and correct (optional)
    if 'outcome' in config.columns:
        outcome = _safe_column(df, config.columns['outcome'], n_rows)
    else:
        outcome = np.full(n_rows, '', dtype=object)

    if 'correct' in config.columns:
        correct = _safe_column(df, config.columns['correct'], n_rows)
    else:
        # Derive from choice and category if possible
        if not np.all(np.isnan(choice_category)):
            correct = (choice_category == category)
            correct[np.isnan(choice_category)] = False
        else:
            correct = np.full(n_rows, False, dtype=bool)

    # Standard optional columns
    if 'reaction_time' in config.columns:
        reaction_time = _safe_column(df, config.columns['reaction_time'], n_rows)
    else:
        reaction_time = np.full(n_rows, np.nan)

    if 'abort' in config.columns:
        abort = _safe_column(df, config.columns['abort'], n_rows)
    else:
        abort = np.zeros(n_rows, dtype=bool)

    if 'opto_on' in config.columns:
        opto_on = _safe_column(df, config.columns['opto_on'], n_rows)
    else:
        opto_on = np.zeros(n_rows, dtype=bool)

    if 'distribution' in config.columns:
        distribution = _safe_column(df, config.columns['distribution'], n_rows)
    else:
        distribution = np.array([])

    # All other mapped columns (inputs, outputs, extras)
    skip_names = {
        'trial_number', primary_stim_name, primary_choice_name,
        'outcome', 'correct', 'reaction_time', 'abort', 'opto_on',
        'distribution',
    }
    skip_names.discard(None)

    optional_fields = {}
    for name, mapping in config.columns.items():
        if name in skip_names:
            continue
        try:
            optional_fields[name] = _safe_column(df, mapping, n_rows)
        except KeyError:
            pass

    # Extra columns
    extra = {}
    for col_name in config.extra_columns:
        if col_name in df.columns:
            extra[col_name] = df[col_name].values

    all_mapped = set(config.get_all_csv_columns())
    for col in df.columns:
        if col not in all_mapped and col not in extra:
            extra[col] = df[col].values

    # ── Build TrialData ─────────────────────────────────────────────────────
    trials = TrialData(
        trial_number=trial_number,
        stimulus=stimulus,
        stimulus_raw=stimulus_raw,
        choice=choice_category,
        choice_raw=choice_raw_arr,
        outcome=outcome,
        correct=correct,
        category=category,
        reaction_time=reaction_time,
        abort=abort,
        opto_on=opto_on,
        distribution=distribution,
        optional_fields=optional_fields,
        extra=extra,
    )


    # ── Trial numbers: unique within the session, in the trial key's range
    numbers = np.asarray(trials.trial_number, dtype=np.int64)
    if numbers.size and (numbers.min() < 0 or numbers.max() >= 1 << TRIAL_BITS):
        raise ValueError(f'{label}: trial numbers must lie in 0..{(1 << TRIAL_BITS) - 1}')
    n_repeated = len(numbers) - len(np.unique(numbers))
    if n_repeated:
        message = f'{label}: {n_repeated} trial number(s) repeat'
        if strict:
            raise ValueError(message)
        warnings.warn(message + '; renumbered from 1', stacklevel=2)
        trials.trial_number = np.arange(1, len(numbers) + 1)

    # ── Session type: a session_type metadata field (a mapped column, or a reader's metadata),
    # else derived from the laser trials. config.session_types lists override both at load.
    recorded_type = metadata.get('session_type', None)
    if recorded_type is not None and str(recorded_type).strip().lower() not in ('', 'nan', 'none'):
        session_type, type_source = str(recorded_type).strip(), 'column'
    else:
        session_type, type_source = ('opto' if bool(np.any(opto_on)) else 'regular'), 'derived'

    return SessionData(
        session_id=raw.session_id,
        session_idx=session_idx,
        date=raw.date,
        start=raw.start,
        metadata=metadata,
        trials=trials,
        session_type=session_type,
        session_type_source=type_source,
        source_id=source_id if source_id is not None else raw.session_id,
        csv_path=raw.source,
    )


# =============================================================================
# LOADING
# =============================================================================

def _call_reader(reader: Reader, config: ProjectConfig, strict: bool) -> Iterable[RawSession]:
    """Run a reader, passing ``strict`` to readers that take it."""
    try:
        takes_strict = 'strict' in inspect.signature(reader).parameters
    except (TypeError, ValueError):
        takes_strict = False
    return reader(config, strict=strict) if takes_strict else reader(config)


def _session_order(raw: RawSession) -> tuple:
    """Chronological: date, then start time (sessions without one after those with one), then the
    reader's name."""
    return (raw.date is None, raw.date or date.min, raw.start is None, raw.start or time.min, raw.session_id)


def _canonical_ids(animal_id: str, raws: List[RawSession]) -> List[str]:
    """Session ids for one animal's sessions, in chronological order: ``<animal>_<YYYYMMDD>``, plus
    ``T<HHMM>`` when the start time is known; sessions that would still share an id are numbered
    ``_1``, ``_2`` ... in order, with a warning, since adding or removing one renumbers the rest. Ids are
    given before any session is ignored, so ignoring one never renames another. A session without a
    date keeps the reader's name (and fails to build)."""
    bases = []
    for raw in raws:
        if raw.date is None:
            bases.append(None)
            continue
        bases.append(f'{animal_id}_{raw.date:%Y%m%d}' + (f'T{raw.start:%H%M}' if raw.start is not None else ''))
    counts: Dict[str, int] = {}
    for base in bases:
        if base is not None:
            counts[base] = counts.get(base, 0) + 1
    shared = sorted(base for base, n in counts.items() if n > 1)
    if shared:
        warnings.warn(f"{animal_id}: several sessions share {', '.join(shared)}; they are numbered _1, _2, ... in "
                      f"name order, so adding or removing one renumbers the rest. Capture the start time in "
                      f"date_regex (groups hour and minute) for ids that do not depend on order.", stacklevel=4)
    ids, seen = [], {}
    for raw, base in zip(raws, bases):
        if base is None:
            ids.append(raw.session_id)
        elif counts[base] == 1:
            ids.append(base)
        else:
            seen[base] = seen.get(base, 0) + 1
            ids.append(f'{base}_{seen[base]}')
    return ids


def _session_matcher(animal_id: str, entries):
    """A test for one animal's entries in sessions_to_ignore or session_types: an 8-digit date covers
    every session that day; any other entry is a session id without the animal prefix and covers that
    one session."""
    dates, ids = set(), set()
    for entry in entries or []:
        text = str(entry).strip()
        if re.fullmatch(r'\d{8}', text):
            try:
                dates.add(date(int(text[:4]), int(text[4:6]), int(text[6:])))
            except ValueError:
                warnings.warn(f'{animal_id}: {text} is not a valid date; ignored', stacklevel=3)
        else:
            ids.add(f'{animal_id}_{text}')
    return lambda day, session_id: day in dates or session_id in ids


def _build_all(raws: Iterable[RawSession], config: ProjectConfig, strict: bool) -> Dict[str, List[SessionData]]:
    """Give each animal's sessions their ids, drop the ignored ones unread, and build the rest. A session
    with no usable rows is skipped with a warning; any other failure raises (``strict``) or is logged and
    skipped."""
    by_animal: Dict[str, List[RawSession]] = {}
    for raw in raws:
        by_animal.setdefault(raw.animal_id, []).append(raw)
    built: Dict[str, List[SessionData]] = {}
    for animal_id, animal_raws in by_animal.items():
        animal_raws = sorted(animal_raws, key=_session_order)
        ignored = _session_matcher(animal_id, config.sessions_to_ignore.get(animal_id, []))
        for raw, session_id in zip(animal_raws, _canonical_ids(animal_id, animal_raws)):
            label = f'{animal_id}/{session_id}' + (f' ({raw.session_id})' if raw.session_id != session_id else '')
            if ignored(raw.date, session_id):
                log.info('%s: in sessions_to_ignore, not read', label)
                continue
            try:
                table = raw.table()
                if len(table) == 0:
                    log.warning('%s: no usable rows; skipped', label)
                    continue
                session = build_session(replace(raw, session_id=session_id, trials=table), config,
                                        strict=strict, source_id=raw.session_id)
            except Exception as e:
                if strict:
                    raise RuntimeError(f'Failed to load {label}: {e}') from e
                log.warning('Failed to load %s: %s (skipped)', label, e)
                continue
            built.setdefault(animal_id, []).append(session)
    return built


def _assign_experimenters(animals: Iterable[AnimalData], config: ProjectConfig) -> None:
    """Stamp each session with its experimenter (the animal metadata's, else project.experimenter) and
    refuse two sessions whose hashes collide, since they would share trial keys."""
    seen: Dict[int, str] = {}
    for animal in animals:
        experimenter = str(animal.metadata.get('experimenter', config.experimenter) or '')
        for sess in animal.sessions:
            sess.experimenter = experimenter
            name = f'{animal.animal_id}/{sess.session_id}'
            other = seen.setdefault(sess.session_hash, name)
            if other != name:
                raise ValueError(f'{other} and {name} hash to the same session key; rename one of them')


def load_session_csv(csv_path: Union[str, Path], config: ProjectConfig, *,
                     date: date | None = None) -> SessionData:
    """
    Load one CSV file as one session, by the csv_folders reader's rules for a single file.

    The date is ``date`` if given, else read with ``reader.date_regex`` from the folder name, else
    from the file name. The animal is the mapped ``animal_id`` column when there is one, else the
    grandparent folder; the session id is the parent folder.

    Raises:
        ValueError: no date, no rows left, or a missing required column.
    """
    path = Path(csv_path)
    settings = config.reader if isinstance(config.reader, CsvFolderReaderConfig) else CsvFolderReaderConfig()
    start = None
    if date is None:
        date, start = parse_session_datetime(path.parent.name, settings.date_regex)
    if date is None:
        date, start = parse_session_datetime(path.stem, settings.date_regex)
    table = _read_pieces([str(path)], settings, _trial_column(config), True)
    animal_id = path.parent.parent.name
    animal_column = config.session_metadata.get('animal_id')
    if animal_column is not None and animal_column.csv_name in table.columns and len(table):
        value = table[animal_column.csv_name].iloc[0]
        if pd.notna(value) and str(value).strip():
            animal_id = str(value)
    raw = RawSession(animal_id=animal_id, session_id=path.parent.name, date=date, start=start,
                     trials=table, source=str(path))
    return build_session(raw, config)


def load_animal(animal_dir: Union[str, Path], config: ProjectConfig, *, strict: bool = True) -> AnimalData:
    """
    Load one animal folder with the csv_folders reader: every sub-directory is a session folder.

    Sessions in ``sessions_to_ignore`` are not read; a session that fails raises under ``strict``
    and is logged and skipped otherwise.
    """
    animal_dir = Path(animal_dir)
    sessions = _build_all(_animal_sessions(animal_dir, config, strict), config, strict)
    animal = AnimalData(animal_id=animal_dir.name, sessions=sessions.get(animal_dir.name, []))
    _assign_experimenters([animal], config)
    return animal


# =============================================================================
# EXPERIMENT LOADING
# =============================================================================

def apply_session_type(experiment, mapping, stype) -> None:
    """Set ``session_type = stype`` on the sessions a ``config.session_types`` entry lists
    ({animal_id: [entries]}; an entry is a date, every session that day, or a session id without the
    animal prefix); their ``session_type_source`` becomes 'config'.

    A list is the analyst's deliberate decision, so it overrides a type read from a mapped
    session_type column (warning when the two differ) and a derived one. A session listed under two
    types takes the later, with a warning.
    """
    for animal_id, date_strs in (mapping or {}).items():
        animal = experiment.animals.get(animal_id)
        if animal is None:
            continue
        listed = _session_matcher(animal_id, date_strs)
        for sess in animal.sessions:
            if not listed(sess.date, sess.session_id):
                continue
            if sess.session_type != stype and sess.session_type_source == 'column':
                warnings.warn(f"{animal_id} {sess.session_id}: the session_type column says "
                              f"'{sess.session_type}' but session_types lists it as '{stype}'; using "
                              f"'{stype}'", stacklevel=2)
            elif sess.session_type != stype and sess.session_type_source == 'config':
                warnings.warn(f"{animal_id} {sess.session_id}: listed in session_types as both "
                              f"'{sess.session_type}' and '{stype}'; using '{stype}'", stacklevel=2)
            sess.session_type = stype
            sess.session_type_source = 'config'
        animal._session_table_cache = None


def _warn_unknown_session_types(experiment: ExperimentData, config: ProjectConfig) -> None:
    """One warning per session type read from a column that is neither derived ('regular', 'opto') nor
    a name in ``config.session_types``, which is usually a typo. Listing a type there, with no dates,
    declares it."""
    known = {'regular', 'opto', *config.session_types}
    counts: Dict[str, int] = {}
    for animal in experiment.animals.values():
        for sess in animal.sessions:
            if sess.session_type_source == 'column' and sess.session_type not in known:
                counts[sess.session_type] = counts.get(sess.session_type, 0) + 1
    for value, n in sorted(counts.items()):
        warnings.warn(f"session type {value!r} ({n} session(s), from the session_type column) is neither "
                      f"'regular', 'opto' nor a name in session_types; if it is meant, list it in "
                      f"session_types, with no dates if none need stamping", stacklevel=3)


def _animal_metadata(config: ProjectConfig) -> dict:
    """Per-animal attributes, {animal_id: {...}}, from the JSON file named by ``animal_metadata``;
    without it, ``animal_metadata.json`` in the csv_folders data directory when that file exists."""
    if config.animal_metadata is not None:
        path = config.resolve_path(config.animal_metadata)
        if not path.exists():
            raise FileNotFoundError(f'animal_metadata file not found: {path}')
    elif isinstance(config.reader, CsvFolderReaderConfig):
        path = config.resolve_path(config.reader.data_dir) / 'animal_metadata.json'
        if not path.exists():
            return {}
    else:
        return {}
    with open(path) as f:
        return json.load(f)


def load_experiment(
    config_or_path: Union[ProjectConfig, str, Path],
    *,
    reader: Reader | None = None,
    strict: bool = True,
) -> ExperimentData:
    """
    Load every session a reader finds into an ExperimentData.

    Args:
        config_or_path: ProjectConfig, or the path to a YAML config.
        reader: where the trial tables come from; by default the one named in the config's
            ``reader:`` block. Any callable that takes the config and yields RawSession objects
            works; one that accepts a ``strict`` keyword is given this function's.
        strict: raise on the first session that fails to load (default); False logs and skips it.

    Sessions in ``sessions_to_ignore`` are dropped before their tables are read. Animal metadata
    comes from ``animal_metadata`` (or ``animal_metadata.json`` in the csv_folders data directory);
    then ``session_types`` are stamped and ``session_presets`` registered.
    """
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    if reader is None:
        reader = _reader_from_config(config)
    by_animal = _build_all(_call_reader(reader, config, strict), config, strict)

    experiment = ExperimentData(config=config)
    for animal_id in sorted(by_animal):
        animal = AnimalData(animal_id=animal_id, sessions=by_animal[animal_id])
        experiment.add_animal(animal)
        animal._config = config

    for animal_id, meta in _animal_metadata(config).items():
        animal = experiment.animals.get(animal_id)
        if animal is not None:
            animal.metadata.update(meta)
    _assign_experimenters(experiment.animals.values(), config)

    # Stamp session_type from the config lists. opto_on is left untouched:
    # on a sham session (light, no effective perturbation) its True trials are the
    # sham-opto trials, selectable via opto_on & the project's session type.
    for type_name, mapping in config.session_types.items():
        apply_session_type(experiment, mapping, type_name)
    _warn_unknown_session_types(experiment, config)
    if config.session_presets:
        from behav_utils.data.ops.selection import register_presets_from_config
        register_presets_from_config({'session_presets': config.session_presets})

    log.info("Loaded %d animals, %d sessions", experiment.n_animals,
             sum(a.n_sessions for a in experiment.animals.values()))

    return experiment
