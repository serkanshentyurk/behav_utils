"""
Readers: where trial tables come from.

A reader finds sessions and reads their trial tables. It knows the data layout and nothing else:
column mapping, stimulus normalisation and scoring happen afterwards in ``build_session``, the same
way for every reader.

    load_experiment(config)                       # the reader named in the config's reader: block
    load_experiment(config, reader=table_reader('all_trials.csv', animal='Mouse', session='Session',
                                                date='Date'))
    load_experiment(config, reader=my_reader)     # any callable taking the config, yielding RawSession

A reader yields one ``RawSession`` per session. Its ``trials`` is a DataFrame or a function returning
one; the loader calls the function only for sessions that are not in ``sessions_to_ignore``, so an
ignored session (often ignored because its files are broken) is never read. A reader that accepts a
``strict`` keyword is given the loader's.
"""

from __future__ import annotations

import glob
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Union

import numpy as np
import pandas as pd

from behav_utils.config.schema import ConfigError, CsvFolderReaderConfig, ProjectConfig, TableReaderConfig

log = logging.getLogger('behav_utils.data.loading')

TrialTable = Union[pd.DataFrame, Callable[[], pd.DataFrame]]


@dataclass(frozen=True)
class RawSession:
    """
    One session as a reader found it, before any column mapping.

    Attributes:
        animal_id:  the animal, as the data layout names it
        session_id: the session, unique within the animal
        date:       the session's date; None when the reader could not tell (the loader refuses it)
        trials:     the raw trial table, or a function that reads it
        start:      the time of day the session started, when known; orders sessions within a day
        metadata:   reader-supplied session metadata; config-mapped columns take precedence
        source:     where it came from (a folder, a file, a table and key), for messages
        files:      related files the reader knows about, by name ({'video': path}); discovery shows
                    them as ``file_<name>`` columns. The library defines no names.
        peek:       optional cheap look at the trials, ``peek(columns=None, nrows=None)``: the first
                    ``nrows`` rows, or only ``columns``, under the reader's own row rules. Discovery
                    uses it; without one it reads the whole table.
    """
    animal_id: str
    session_id: str
    date: date | None
    trials: TrialTable
    start: time | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None
    files: Mapping[str, Any] = field(default_factory=dict)
    peek: Callable[..., pd.DataFrame] | None = None

    def table(self) -> pd.DataFrame:
        """
        The raw trial table, read now if the reader deferred it.

        Returns:
            The table.
        """
        return self.trials() if callable(self.trials) else self.trials


Reader = Callable[..., Iterable[RawSession]]


# ── dates and times from names ───────────────────────────────────────────────

def parse_session_datetime(name: str, regex: str) -> tuple[date | None, time | None]:
    """
    The date, and the start time when the regex captures one, from a session's (or file's) name.

    ``regex`` has named groups ``year``, ``month``, ``day`` (or three positional groups, which must
    come first) and optional named groups ``hour``, ``minute``, ``second``. Returns (None, None) when
    it does not match or the date is invalid; the time is None without an ``hour`` group.
    """
    match = re.search(regex, str(name))
    if not match:
        return None, None
    groups = match.groupdict()
    try:
        if groups.get('year') is not None:
            day = date(int(groups['year']), int(groups['month']), int(groups['day']))
        else:
            g = match.groups()
            day = date(int(g[0]), int(g[1]), int(g[2]))
    except (ValueError, IndexError, KeyError, TypeError):
        return None, None
    start = None
    if groups.get('hour') is not None:
        try:
            start = time(int(groups['hour']), int(groups.get('minute') or 0), int(groups.get('second') or 0))
        except ValueError:
            start = None
    return day, start


# ── the csv_folders reader ───────────────────────────────────────────────────

def is_empty_csv(path: Union[str, Path]) -> bool:
    """
    True for a file with nothing in it, not even a header (a run stopped before writing anything).
    The reader skips such files; projects that re-read the CSVs should do the same.

    Args:
        path: The file.

    Returns:
        Whether the file is empty.
    """
    try:
        pd.read_csv(path, nrows=0)
    except pd.errors.EmptyDataError:
        return True
    except Exception:
        return False
    return False


def _read_pieces(files, settings: CsvFolderReaderConfig, trial_column: str | None,
                 strict: bool, usecols=None) -> pd.DataFrame:
    """A session's file(s) as one table: each file's last row dropped if configured, files shorter
    than ``min_trials_per_file`` skipped, the rest concatenated in name order with trial numbers
    renumbered from 1. An unreadable file raises (``strict``) or is skipped with a warning."""
    pieces = []
    for path in sorted(files):
        try:
            df = pd.read_csv(path, low_memory=False,
                             usecols=None if usecols is None else (lambda c: c in usecols))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as e:
            if strict:
                raise OSError(f'cannot read {path}: {e}') from e
            log.warning('%s: cannot read (%s); skipped', path, e)
            continue
        if settings.drop_last_row and len(df) > 1:
            df = df.iloc[:-1]
        if len(df) < settings.min_trials_per_file:
            log.warning('%s: %d rows, fewer than min_trials_per_file (%d); skipped',
                        path, len(df), settings.min_trials_per_file)
            continue
        pieces.append(df)
    if not pieces:
        return pd.DataFrame()
    if len(pieces) == 1:
        return pieces[0]
    merged = pd.concat(pieces, ignore_index=True)
    if trial_column is not None and trial_column in merged.columns:
        merged[trial_column] = range(1, len(merged) + 1)
    return merged


def _peek_pieces(files, settings: CsvFolderReaderConfig, columns=None, nrows=None) -> pd.DataFrame:
    """A cheap look at a session's file(s) under the same row rules as loading: only ``columns`` of
    every file, or the first ``nrows`` rows of the first file loading would keep (rows are counted, not
    parsed, to find it)."""
    if columns is not None:
        table = _read_pieces(files, settings, None, False, usecols=list(columns))
        return table if nrows is None else table.head(nrows)
    for path in sorted(files):
        try:
            with open(path, 'rb') as f:
                rows = sum(1 for _ in f) - 1
        except OSError:
            continue
        kept = rows - 1 if settings.drop_last_row and rows > 1 else rows
        if kept < settings.min_trials_per_file:
            continue
        try:
            return pd.read_csv(path, nrows=nrows, low_memory=False)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError):
            continue
    return pd.DataFrame()


def _peek_frame(frame: pd.DataFrame, columns=None, nrows=None) -> pd.DataFrame:
    out = frame if columns is None else frame[[c for c in columns if c in frame.columns]]
    return out if nrows is None else out.head(nrows)


def _subdirs(path: Path) -> list:
    return sorted(d for d in path.iterdir() if d.is_dir() and not d.name.startswith('.'))


def _trial_column(config: ProjectConfig) -> str | None:
    mapping = config.columns.get('trial_number')
    return mapping.csv_name if mapping is not None else None


def _folder_settings(config: ProjectConfig) -> CsvFolderReaderConfig:
    if not isinstance(config.reader, CsvFolderReaderConfig):
        raise ConfigError(f"this needs reader type 'csv_folders', not {config.reader.TYPE!r}")
    return config.reader


def _animal_sessions(animal_dir: Path, config: ProjectConfig, strict: bool = True) -> Iterator[RawSession]:
    """The sessions in one animal folder, as the csv_folders reader sees them."""
    settings = _folder_settings(config)
    trial_column = _trial_column(config)
    animal_id = animal_dir.name
    for folder in _subdirs(animal_dir):
        files = sorted(glob.glob(str(folder / settings.behaviour_file)))
        empty = [f for f in files if is_empty_csv(f)]
        if empty:
            log.warning('%s/%s: empty file(s) skipped: %s', animal_id, folder.name,
                        ', '.join(Path(f).name for f in empty))
            files = [f for f in files if f not in empty]
        if not files:
            continue
        day, start = parse_session_datetime(folder.name, settings.date_regex)
        if settings.merge_files:
            yield RawSession(animal_id=animal_id, session_id=folder.name, date=day, start=start,
                             trials=partial(_read_pieces, files, settings, trial_column, strict),
                             source=str(folder), peek=partial(_peek_pieces, files, settings))
            continue
        for path in files:
            stem = Path(path).stem
            file_day, file_start = parse_session_datetime(stem, settings.date_regex)
            yield RawSession(animal_id=animal_id, session_id=f'{folder.name}_{stem}',
                             date=file_day or day, start=file_start if file_day else None,
                             trials=partial(_read_pieces, [path], settings, trial_column, strict),
                             source=path, peek=partial(_peek_pieces, [path], settings))


def csv_folder_reader(config: ProjectConfig, *, strict: bool = True) -> Iterator[RawSession]:
    """
    The ``csv_folders`` reader.

    Every sub-directory of ``data_dir`` is an animal, and every sub-directory of an animal a session folder; the
    config's ``reader:`` block holds the settings.

    Args:
        config: The project config.
        strict: Raise on a session folder that cannot be read (the default); False skips it.

    Returns:
        The raw sessions, one at a time.
    """
    settings = _folder_settings(config)
    data_dir = config.resolve_path(settings.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')
    for animal_dir in _subdirs(data_dir):
        yield from _animal_sessions(animal_dir, config, strict)


# ── the table reader ─────────────────────────────────────────────────────────

def table_reader(source: Union[str, Path, pd.DataFrame], *, animal: str, session: str, date: str,
                 start: str | None = None) -> Reader:
    """
    A reader over one table holding every trial.

    The table is a DataFrame, or a ``.csv``, ``.tsv`` or ``.parquet`` path (relative paths resolve against the
    config file's folder; Parquet needs pyarrow).

    ``animal`` and ``session`` name the columns that identify a session, ``date`` the column that
    dates it, and ``start`` an optional column with its start time. Sessions keep their order of
    first appearance and their rows keep table order.

    Args:
        source: The table, or the path of a .csv, .tsv or .parquet file.
        animal: The column naming each row's animal.
        session: The column naming each row's session.
        date: The column holding each session's date.
        start: The column holding each session's start time; None for none.

    Returns:
        A reader, to pass as ``load_experiment(config, reader=...)``.
    """
    wanted = [c for c in (animal, session, date, start) if c is not None]

    def read(config: ProjectConfig) -> Iterator[RawSession]:
        table, name = _load_table(source, config)
        missing = [c for c in wanted if c not in table.columns]
        if missing:
            raise ConfigError(f'table reader: column(s) {missing} not in {name}; it has {list(table.columns)}')
        for key in (animal, session):
            if table[key].isna().any():
                raise ValueError(f'table reader: {name} has rows with no {key!r}')
        for (animal_id, session_id), rows in table.groupby([animal, session], sort=False):
            first = rows.iloc[0]
            rows = rows.reset_index(drop=True)
            yield RawSession(animal_id=str(animal_id), session_id=str(session_id),
                             date=_as_date(first[date]),
                             start=_as_time(first[start]) if start is not None else None,
                             trials=rows, source=f'{name} [{animal_id} / {session_id}]',
                             peek=partial(_peek_frame, rows))
    return read


def _load_table(source, config: ProjectConfig) -> tuple:
    if isinstance(source, pd.DataFrame):
        return source, 'the table'
    path = config.resolve_path(source)
    suffix = path.suffix.lower()
    if suffix == '.parquet':
        try:
            return pd.read_parquet(path), path.name
        except ImportError as e:
            raise ImportError(f'reading {path.name} needs pyarrow: pip install pyarrow') from e
    sep = '\t' if suffix in ('.tsv', '.tab') else ','
    return pd.read_csv(path, sep=sep, low_memory=False), path.name


def _as_date(value) -> date | None:
    if isinstance(value, (int, np.integer)):
        value = str(value)
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError):
        return None


def _as_time(value) -> time | None:
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, time):
        return value
    try:
        return pd.Timestamp(str(value)).time()
    except (ValueError, TypeError):
        return None


def _reader_from_config(config: ProjectConfig) -> Reader:
    """The reader the config's ``reader:`` block names."""
    settings = config.reader
    if isinstance(settings, TableReaderConfig):
        return table_reader(settings.path, animal=settings.animal_column, session=settings.session_column,
                            date=settings.date_column, start=settings.start_column)
    return csv_folder_reader
