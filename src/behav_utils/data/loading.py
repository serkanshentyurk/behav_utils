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
from behav_utils.tasks import Outcome

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

    raw = df[mapping.csv_name].to_numpy().copy()        # numpy, also for pandas 3 text columns

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

def _check_clock(label, trial_number, events, reaction_time, clock) -> None:
    """Event columns that go backwards across trials, negative reaction times, and trials out of
    ``clock.order``, as ``clock.on_problem`` says: one warning (or error) per kind of problem."""
    if clock.on_problem == 'ignore':
        return
    numbers = np.asarray(trial_number)
    problems = []
    for name, times in events.items():
        finite = np.flatnonzero(~np.isnan(times))
        back = finite[1:][np.diff(times[finite]) < 0]
        if back.size:
            problems.append(f"{name} goes backwards at trial {numbers[back[0]]} ({back.size} time(s)): the clock "
                            f"restarted (a file boundary?) or passed midnight")
    negative = np.flatnonzero(np.asarray(reaction_time, dtype=float) < 0)
    if negative.size:
        problems.append(f"{negative.size} negative reaction time(s) (first: trial {numbers[negative[0]]})")
    for earlier, later in zip(clock.order, clock.order[1:]):
        if earlier in events and later in events:
            bad = np.flatnonzero(events[later] < events[earlier])
            if bad.size:
                problems.append(f"{later} comes before {earlier} on {bad.size} trial(s) "
                                f"(first: trial {numbers[bad[0]]})")
    for problem in problems:
        message = f'{label}: {problem}'
        if clock.on_problem == 'raise':
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)


_OUTCOME_BY_NAME = {o.name.lower(): int(o) for o in Outcome}


def _check_recorded(label, trial_number, outcome, correct, correct_recorded, outcome_recorded, scoring) -> None:
    """Compare recorded correct/outcome columns with the task's score, as ``config.scoring`` says. The
    recorded outcome is compared only through ``scoring.outcome_labels``; labels it does not list are
    skipped."""
    if scoring.on_mismatch == 'ignore':
        return
    disagreements = []
    if len(correct_recorded):
        scored = (outcome == Outcome.CORRECT) | (outcome == Outcome.ERROR)
        disagreements.append(('correct', scored & (correct_recorded != correct)))
    if len(outcome_recorded) and scoring.outcome_labels:
        codes = np.array([_OUTCOME_BY_NAME.get(scoring.outcome_labels.get(v, ''), -1) for v in outcome_recorded])
        disagreements.append(('outcome', (codes >= 0) & (codes != outcome)))
    for what, bad in disagreements:
        if not bad.any():
            continue
        first = ', '.join(str(t) for t in np.asarray(trial_number)[bad][:5])
        message = (f"{label}: the recorded {what} column disagrees with the task's score on {int(bad.sum())} "
                   f"trial(s) (first: trial {first}); the task's score is used")
        if scoring.on_mismatch == 'raise':
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)


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

    # ── The session's task: the config's default, or the alternative that task_by picks
    task_name, task_cfg = config.task_config_for(metadata)
    task = task_cfg.make_task()
    continuous = task.response_space.kind == 'continuous'      # responses share the stimulus scale

    if 'abort' in config.columns:
        abort = np.asarray(_safe_column(df, config.columns['abort'], n_rows), dtype=bool)
    else:
        abort = np.zeros(n_rows, dtype=bool)

    # Primary stimulus, on the library's axis ([-1, 1], boundary at 0)
    primary_stim_name = task_cfg.primary_stimulus
    if primary_stim_name and primary_stim_name in config.columns:
        stimulus_raw = np.asarray(_safe_column(df, config.columns[primary_stim_name], n_rows), dtype=float)
        stimulus = (task_cfg.normalise_continuous(stimulus_raw) if continuous
                    else task_cfg.normalise_stimulus(stimulus_raw))
    else:
        stimulus_raw = np.full(n_rows, np.nan)
        stimulus = np.full(n_rows, np.nan)

    # Primary choice, as a response; the task completes it (a go/no-go blank is a withhold)
    primary_choice_name = task_cfg.primary_choice
    if primary_choice_name and primary_choice_name in config.columns:
        choice_raw_arr = _safe_column(df, config.columns[primary_choice_name], n_rows)
        responses = convert_choice_to_category(choice_raw_arr, metadata, task_cfg.choice_mapping)
        if continuous:
            responses = task_cfg.normalise_continuous(responses)
            n_outside = int(np.sum(np.abs(responses) > 1 + 1e-9))
            if n_outside:
                warnings.warn(f'{label}: {n_outside} response(s) fall outside the stimulus range after the '
                              f'transform; kept as recorded', stacklevel=2)
        choice_category = task.complete_responses(responses, abort)
        nan_frac = float(np.mean(np.isnan(choice_category))) if n_rows else 0.0
        if nan_frac > NAN_CHOICE_WARN_FRACTION:
            warnings.warn(
                f"{label}: {nan_frac:.0%} of choices are no-response (NaN), and NaN choices are dropped from "
                f"every statistic. If withholding is a response in this task, set task.type: go_nogo.",
                stacklevel=2)
    else:
        choice_raw_arr = np.full(n_rows, np.nan)
        choice_category = np.full(n_rows, np.nan)

    # ── Score: the task alone decides what is correct; recorded columns are checked against it
    category = task.category(stimulus)
    outcome = task.score(choice_category, task.truth(stimulus), abort)
    correct = outcome == Outcome.CORRECT
    recorded = {name: name in config.columns and config.columns[name].csv_name in df.columns
                for name in ('correct', 'outcome')}      # only a column this file actually has
    correct_recorded = (np.asarray(_safe_column(df, config.columns['correct'], n_rows), dtype=bool)
                        if recorded['correct'] else np.array([]))
    outcome_recorded = (np.asarray(_safe_column(df, config.columns['outcome'], n_rows)).astype(str)
                        if recorded['outcome'] else np.array([]))
    _check_recorded(label, trial_number, outcome, correct, correct_recorded, outcome_recorded, config.scoring)

    # Standard optional columns
    # ── Times: seconds on the recording clock (config.clock), never re-zeroed
    if 'reaction_time' in config.columns:
        reaction_time = config.clock.to_seconds(_safe_column(df, config.columns['reaction_time'], n_rows),
                                                config.columns['reaction_time'].unit)
    else:
        reaction_time = np.full(n_rows, np.nan)
    events = {}
    for name, mapping in config.columns.items():
        if not name.startswith('t_'):
            continue
        if mapping.csv_name in df.columns:
            events[name] = config.clock.to_seconds(_safe_column(df, mapping, n_rows), mapping.unit)
        else:                                            # optional (a required one was refused above)
            events[name] = np.full(n_rows, np.nan)          # same keys in every session
    _check_clock(label, trial_number, events, reaction_time, config.clock)

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
    skip_names.update(events)                    # event times live in TrialData.events
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
            extra[col_name] = df[col_name].to_numpy()

    all_mapped = set(config.get_all_csv_columns())
    for col in df.columns:
        if col not in all_mapped and col not in extra:
            extra[col] = df[col].to_numpy()

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
        correct_recorded=correct_recorded,
        outcome_recorded=outcome_recorded,
        reaction_time=reaction_time,
        abort=abort,
        opto_on=opto_on,
        distribution=distribution,
        optional_fields=optional_fields,
        extra=extra,
        events=events,
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
        task=task,
        task_name=task_name,
        clock=config.clock.name,
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


def _identify(raws: Iterable[RawSession], config: ProjectConfig) -> list:
    """Each session the reader found as (animal_id, raw, session_id, ignored): canonical ids given before
    anything is read, ``ignored`` when ``sessions_to_ignore`` lists the session. Animals in the order the
    reader gave them, sessions in chronological order. Discovery and loading both use it, so their ids
    cannot differ."""
    by_animal: Dict[str, List[RawSession]] = {}
    for raw in raws:
        by_animal.setdefault(raw.animal_id, []).append(raw)
    out = []
    for animal_id, animal_raws in by_animal.items():
        animal_raws = sorted(animal_raws, key=_session_order)
        ignored = _session_matcher(animal_id, config.sessions_to_ignore.get(animal_id, []))
        for raw, session_id in zip(animal_raws, _canonical_ids(animal_id, animal_raws)):
            out.append((animal_id, raw, session_id, ignored(raw.date, session_id)))
    return out


def _wanted_ids(sessions) -> set | None:
    """The session ids ``load_experiment(sessions=...)`` asks for: a table with a session_id column,
    or ids."""
    if sessions is None:
        return None
    if isinstance(sessions, pd.DataFrame):
        if 'session_id' not in sessions.columns:
            raise ValueError("load_experiment: a sessions table needs a session_id column (as discover_sessions gives)")
        return set(sessions['session_id'].astype(str))
    if isinstance(sessions, str):
        return {sessions}
    return {str(s) for s in sessions}


def _build_all(raws: Iterable[RawSession], config: ProjectConfig, strict: bool,
               wanted: set | None = None) -> Dict[str, List[SessionData]]:
    """Give each animal's sessions their ids, drop the ignored and (with ``wanted``) the unrequested ones
    unread, and build the rest. A session with no usable rows is skipped with a warning; any other
    failure raises (``strict``) or is logged and skipped. A requested id the reader does not find
    raises (``strict``) or warns."""
    built: Dict[str, List[SessionData]] = {}
    found = set()
    for animal_id, raw, session_id, ignored in _identify(raws, config):
        found.add(session_id)
        label = f'{animal_id}/{session_id}' + (f' ({raw.session_id})' if raw.session_id != session_id else '')
        if ignored:
            if wanted is not None and session_id in wanted:
                warnings.warn(f'{label} is in sessions_to_ignore; not loaded', stacklevel=3)
            log.info('%s: in sessions_to_ignore, not read', label)
            continue
        if wanted is not None and session_id not in wanted:
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
    missing = sorted((wanted or set()) - found)
    if missing:
        message = f"load_experiment: sessions not found by the reader: {', '.join(missing[:10])}"
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)
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
    sessions=None,
) -> ExperimentData:
    """
    Load every session a reader finds, or the chosen ``sessions``, into an ExperimentData.

    Args:
        config_or_path: ProjectConfig, or the path to a YAML config.
        reader: where the trial tables come from; by default the one named in the config's
            ``reader:`` block. Any callable that takes the config and yields RawSession objects
            works; one that accepts a ``strict`` keyword is given this function's.
        strict: raise on the first session that fails to load (default); False logs and skips it.
        sessions: only these sessions: a table with a ``session_id`` column (``discover_sessions``,
            filtered) or session ids. The others are never read; an id the reader does not find raises
            (``strict``) or warns, and ``sessions_to_ignore`` still applies.

    Sessions in ``sessions_to_ignore`` are dropped before their tables are read. Animal metadata
    comes from ``animal_metadata`` (or ``animal_metadata.json`` in the csv_folders data directory);
    then ``session_types`` are stamped and ``session_presets`` registered.
    """
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    if reader is None:
        reader = _reader_from_config(config)
    by_animal = _build_all(_call_reader(reader, config, strict), config, strict, _wanted_ids(sessions))

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
