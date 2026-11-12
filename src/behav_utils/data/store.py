"""
The store: an experiment saved as Parquet tables and JSON, and reloaded exactly, all of it or in part.

    <store>/store.json              format, schema version, library version, creation time, counts,
                                    experiment metadata
    <store>/animals.json            each animal's metadata
    <store>/manifest.parquet        one row per session: the shared session-table columns, then the task's
                                    name and spec, the session's metadata (``metadata_json``), its clock
                                    and its trial file
    <store>/trials/<file>.parquet   one per session: every per-trial field; optional, extra and event
                                    columns carry the prefixes ``opt__``, ``extra__`` and ``event__``

The lag-1 view is not stored: ``TrialData`` rebuilds it from the stored trials exactly as on loading. The
manifest is the session table the rest of the library uses (``discover_sessions``,
``experiment.session_table``), so ``read_manifest`` is filtered the same way and passed to
``load_store(path, sessions=...)``, which opens only the chosen sessions' files.

Parquet can be read by pandas, polars or R without this library; writing and reading it needs pyarrow
(``pip install 'behav_utils[store]'``). An extra column that mixes numbers and text is stored as text.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import uuid
import warnings
from dataclasses import fields
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from behav_utils.data.loading import _wanted_ids
from behav_utils.data.structures import (
    SESSION_TABLE_COLUMNS,
    AnimalData,
    ExperimentData,
    SessionData,
    SessionMetadata,
    TrialData,
)
from behav_utils.tasks import task_from_dict

__all__ = ['save_store', 'load_store', 'read_manifest', 'StoreError']

FORMAT = 'behav_utils store'
SCHEMA_VERSION = 1
_MAP_PREFIX = {'optional_fields': 'opt__', 'extra': 'extra__', 'events': 'event__'}
_LAG_VIEW = frozenset(f.name for f in fields(TrialData) if f.name.startswith('prev_'))
_MANIFEST_EXTRA = ['task_name', 'task_spec', 'metadata_json', 'clock', 'trials_file']


class StoreError(ValueError):
    """A store that cannot be written or read as asked."""


# ── the only two functions that touch Parquet ────────────────────────────────

def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise StoreError("the store needs pyarrow: pip install 'behav_utils[store]'") from None


def _write_table(table: pd.DataFrame, path: Path) -> None:
    _require_pyarrow()
    table.to_parquet(path, index=False)


def _read_table(path: Path) -> pd.DataFrame:
    _require_pyarrow()
    return pd.read_parquet(path)


# ── values ───────────────────────────────────────────────────────────────────

def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _json_safe(value):
    """Metadata values as JSON can hold them: numpy scalars as Python ones, dates and times as ISO text,
    containers recursively, anything else as its text."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    return str(value)


def _column(values: np.ndarray) -> np.ndarray:
    """A per-trial array in a form Parquet holds exactly: numbers and booleans as they are, text as text
    with missing values as nulls, and a column mixing numbers and text as text."""
    if values.dtype.kind in 'biufM':
        return values
    objects = values.astype(object)
    present = [v for v in objects if not _missing(v)]
    if all(isinstance(v, str) for v in present):
        return np.array([None if _missing(v) else v for v in objects], dtype=object)
    if all(isinstance(v, (bool, int, float, np.bool_, np.integer, np.floating)) for v in present):
        return np.array([np.nan if _missing(v) else v for v in objects], dtype=float)
    return np.array([None if _missing(v) else str(v) for v in objects], dtype=object)


def _array(series: pd.Series) -> np.ndarray:
    """A stored column back as a numpy array: text as an object array with NaN for missing, like loading."""
    values = series.to_numpy()
    if values.dtype.kind in 'biufM':
        return values
    return np.array([np.nan if _missing(v) or v is pd.NA else v for v in values], dtype=object)


# ── sessions to tables and back ──────────────────────────────────────────────

def _trials_table(trials: TrialData) -> pd.DataFrame:
    n = trials.n_trials
    columns = {}
    for f in fields(TrialData):
        if f.metadata.get('per_trial') and f.name not in _LAG_VIEW:
            values = np.asarray(getattr(trials, f.name))
            if len(values) == n:                 # an unmapped field is empty: left out, comes back empty
                columns[f.name] = _column(values)
        elif f.metadata.get('per_trial_map'):
            for key, values in getattr(trials, f.name).items():
                values = np.asarray(values)
                if len(values) == n:
                    columns[_MAP_PREFIX[f.name] + key] = _column(values)
    return pd.DataFrame(columns)


def _trials_from_table(table: pd.DataFrame) -> TrialData:
    kwargs: dict = {}
    maps: dict = {name: {} for name in _MAP_PREFIX}
    for column in table.columns:
        values = _array(table[column])
        for name, prefix in _MAP_PREFIX.items():
            if column.startswith(prefix):
                maps[name][column[len(prefix):]] = values
                break
        else:
            kwargs[column] = values
    for name in ('trial_number',):
        kwargs[name] = np.asarray(kwargs[name], dtype=np.int64)
    return TrialData(**kwargs, **maps)


def _task_spec(session: SessionData) -> str:
    return json.dumps(_json_safe(session.task.to_dict()))


def _task(spec: str):
    d = json.loads(spec)
    return task_from_dict({k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def _file_name(session_id: str, used: set) -> str:
    stem = re.sub(r'[^A-Za-z0-9._-]', '_', session_id) or 'session'
    name, k = f'{stem}.parquet', 2
    while name.lower() in used:
        name, k = f'{stem}_{k}.parquet', k + 1
    used.add(name.lower())
    return name


# ── the store ────────────────────────────────────────────────────────────────

_PICKLE_GONE = ('pickle snapshots were removed in 1.0: reload from the raw data with load_experiment, then '
                'experiment.save(path) writes a store (a directory)')


def _check_store(path: Path) -> dict:
    if path.is_file() or path.suffix == '.pkl':
        raise StoreError(f'{path} is a file, not a store; {_PICKLE_GONE}')
    info_path = path / 'store.json'
    if not info_path.is_file():
        raise StoreError(f'{path} is not a behav_utils store (no store.json)')
    info = json.loads(info_path.read_text())
    if info.get('format') != FORMAT:
        raise StoreError(f'{path}: store.json is not a behav_utils store (format {info.get("format")!r})')
    version = info.get('schema_version')
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        raise StoreError(f'{path} was written with store schema {version}; this behav_utils reads up to '
                         f'{SCHEMA_VERSION}: upgrade behav_utils to read it')
    return info


def save_store(experiment: ExperimentData, path: Union[str, Path], *, overwrite: bool = False) -> Path:
    """
    Save ``experiment`` as a store at ``path`` (a directory). The store is written to a temporary directory
    beside it and moved into place only when complete, so a failed write leaves nothing behind. An existing
    store is replaced only with ``overwrite=True``; any other existing path is refused and left untouched.
    """
    path = Path(path)
    if path.suffix == '.pkl':
        raise StoreError(f"{path}: a store is a directory, not a .pkl file; {_PICKLE_GONE}")
    if path.exists():
        if not (path / 'store.json').is_file():
            raise StoreError(f'{path} exists and is not a behav_utils store; refusing to write over it')
        if not overwrite:
            raise StoreError(f'{path} is already a store; pass overwrite=True to replace it')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f'.{path.name}.writing-{uuid.uuid4().hex[:8]}'
    try:
        (temporary / 'trials').mkdir(parents=True)
        rows, used = [], set()
        for animal_id in sorted(experiment.animals):
            animal = experiment.animals[animal_id]
            table = animal.session_table
            for session, (_, summary) in zip(animal.sessions, table.iterrows()):
                name = _file_name(session.session_id, used)
                _write_table(_trials_table(session.trials), temporary / 'trials' / name)
                row = {column: summary[column] for column in SESSION_TABLE_COLUMNS}
                row['date'] = session.date.isoformat()
                row['start'] = session.start.isoformat() if session.start is not None else None
                row.update(task_name=session.task_name, task_spec=_task_spec(session),
                           metadata_json=json.dumps(_json_safe(dict(session.metadata.fields))),
                           clock=session.clock, trials_file=f'trials/{name}')
                rows.append(row)
        _write_table(pd.DataFrame(rows, columns=SESSION_TABLE_COLUMNS + _MANIFEST_EXTRA),
                     temporary / 'manifest.parquet')
        (temporary / 'animals.json').write_text(json.dumps(
            {a: _json_safe(dict(experiment.animals[a].metadata)) for a in sorted(experiment.animals)}, indent=1))
        from behav_utils import __version__
        (temporary / 'store.json').write_text(json.dumps({
            'format': FORMAT, 'schema_version': SCHEMA_VERSION, 'library_version': __version__,
            'created': datetime.now(UTC).isoformat(timespec='seconds'),
            'n_animals': len(experiment.animals), 'n_sessions': len(rows),
            'experiment_metadata': _json_safe(dict(experiment.metadata)),
        }, indent=1))
        if path.exists():
            shutil.rmtree(path)
        temporary.rename(path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return path


def read_manifest(path: Union[str, Path]) -> pd.DataFrame:
    """The store's sessions, one row each: the shared session-table columns (dates and start times as
    ``date`` and ``time``), then ``task_name``, ``task_spec``, ``metadata_json``, ``clock`` and
    ``trials_file``. Filter it
    and pass it to ``load_store(path, sessions=...)``."""
    path = Path(path)
    _check_store(path)
    manifest = _read_table(path / 'manifest.parquet')
    manifest['date'] = [date.fromisoformat(d) for d in manifest['date']]
    manifest['start'] = [None if _missing(t) or t is pd.NA else time.fromisoformat(t) for t in manifest['start']]
    return manifest


def load_store(path: Union[str, Path], *, sessions=None, strict: bool = True) -> ExperimentData:
    """
    Load a store, or only ``sessions`` (a table with a ``session_id`` column, such as a filtered
    ``read_manifest``, or session ids): only their trial files are opened. An id the store does not have
    raises (``strict``) or warns. A store written with a newer schema than this library reads raises, and
    so does a chosen session whose trial file is missing (the store is incomplete).
    """
    path = Path(path)
    info = _check_store(path)
    manifest = read_manifest(path)
    wanted = _wanted_ids(sessions)
    if wanted is not None:
        missing = sorted(wanted - set(manifest['session_id']))
        if missing:
            message = f"load_store: sessions not in the store: {', '.join(missing[:10])}"
            if strict:
                raise StoreError(message)
            warnings.warn(message, stacklevel=2)
        manifest = manifest[manifest['session_id'].isin(wanted)]
    animal_metadata = json.loads((path / 'animals.json').read_text())
    by_animal: dict = {}
    for row in manifest.to_dict('records'):
        trials_file = path / row['trials_file']
        if not trials_file.is_file():
            raise StoreError(f"{path}: the trial file of {row['session_id']} ({row['trials_file']}) is missing; "
                             f"the store is incomplete")
        trials = _trials_from_table(_read_table(trials_file))
        by_animal.setdefault(row['animal_id'], []).append(SessionData(
            session_id=row['session_id'], session_idx=int(row['session_idx']), date=row['date'],
            metadata=SessionMetadata(fields=json.loads(row['metadata_json'])), trials=trials,
            session_type=row['session_type'], start=row['start'],
            session_type_source=row['session_type_source'],
            source_id=None if _missing(row['source_id']) else row['source_id'],
            experimenter=row['experimenter'] if isinstance(row['experimenter'], str) else '',
            task=_task(row['task_spec']), task_name=row['task_name'],
            clock=row['clock'] if isinstance(row['clock'], str) else '',
            source=None if _missing(row['source']) else row['source'],
        ))
    experiment = ExperimentData()
    experiment.metadata.update(info.get('experiment_metadata') or {})
    for animal_id in sorted(by_animal):
        animal = AnimalData(animal_id=animal_id, sessions=by_animal[animal_id])
        animal.metadata.update(animal_metadata.get(animal_id, {}))
        experiment.add_animal(animal)
    return experiment
