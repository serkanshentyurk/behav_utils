"""
Session discovery: every session a reader finds, as a table, before any trial file is read in full.

``discover_sessions(config)`` lists each session with the identity columns every session table shares
(id, animal, experimenter, date, start, source id, source), whether ``sessions_to_ignore`` lists it,
the session fields the reader can peek at cheaply, the task, the session type where it can be decided
without reading trials, and any related files the reader attached. Filter it like any DataFrame and
pass it to ``load_experiment(config, sessions=...)`` to read only those sessions.

    table = discover_sessions('config.yaml')
    chosen = table[(table.stage == 'task') & ~table.ignored]
    experiment = load_experiment('config.yaml', sessions=chosen)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from behav_utils.config.schema import ProjectConfig, load_config
from behav_utils.data.loading import (
    _animal_metadata,
    _call_reader,
    _extract_session_metadata,
    _identify,
    _reader_from_config,
    _safe_column,
    _session_matcher,
)
from behav_utils.data.readers import RawSession, Reader
from behav_utils.data.structures import _IDENTITY_COLUMNS

__all__ = ['discover_sessions']


def _look(raw: RawSession, columns, nrows, fallback: list) -> pd.DataFrame:
    """The reader's cheap peek, or the whole table (noted once in ``fallback``) when it has none."""
    if raw.peek is not None:
        return raw.peek(columns=columns, nrows=nrows)
    fallback.append(raw.source or raw.session_id)
    table = raw.table()
    if columns is not None:
        table = table[[c for c in columns if c in table.columns]]
    return table if nrows is None else table.head(nrows)


def _listed_type(config: ProjectConfig, animal_id: str, raw: RawSession, session_id: str) -> str | None:
    """The type a ``session_types`` list gives the session (the later list, if two name it)."""
    listed = None
    for type_name, mapping in config.session_types.items():
        if _session_matcher(animal_id, mapping.get(animal_id, []))(raw.date, session_id):
            listed = type_name
    return listed


def discover_sessions(
    config_or_path: Union[ProjectConfig, str, Path],
    *,
    reader: Reader | None = None,
    peek: bool = True,
    laser: bool = False,
) -> pd.DataFrame:
    """
    Every session the reader finds, one row each, without loading trials.

    The session type comes from a ``session_types`` list or a mapped session_type column; one derived
    from the trials ('opto' or 'regular') needs ``laser=True`` and is otherwise left empty rather than
    guessed. The ids are the ones ``load_experiment`` gives.

    Args:
        config_or_path: A ``ProjectConfig``, or the path of a YAML config.
        reader: Where the sessions come from; by default the reader the config's ``reader:`` block names.
        peek: Read each session's first row for its ``session_metadata`` fields. A reader that cannot
            read one row cheaply reads the whole table instead, and a warning says so.
        laser: Read the laser column, to derive 'opto' or 'regular' and to add a ``laser`` column.

    Returns:
        One row per session, with the identity columns (``session_id``, ``animal_id``, ``experimenter``,
        ``date``, ``start``, ``source_id``, ``source``), ``ignored``, ``task_name`` and ``task``,
        ``session_type`` and ``session_type_source``, the reader's metadata, the peeked
        ``session_metadata`` fields, and ``file_<name>`` for related files a reader attached.

    Raises:
        FileNotFoundError: Given a path, the config file does not exist.
        ConfigError: Given a path, the config cannot be used as written.

    Warns:
        UserWarning: The reader cannot peek, so whole tables were read (``peek=False`` avoids it).
    """
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    reader = reader if reader is not None else _reader_from_config(config)
    animal_meta = _animal_metadata(config)
    laser_mapping = config.columns.get('opto_on')
    fallback: list = []
    rows = []
    for animal_id, raw, session_id, ignored in _identify(_call_reader(reader, config, False), config):
        row = {
            'session_id': session_id, 'animal_id': animal_id,
            'experimenter': str(animal_meta.get(animal_id, {}).get('experimenter', config.experimenter) or ''),
            'date': raw.date, 'start': raw.start, 'source_id': raw.session_id, 'source': raw.source,
            'ignored': ignored,
        }
        fields = {k: v for k, v in dict(raw.metadata).items() if k != 'animal_id'}
        if peek:
            first = _look(raw, None, 1, fallback)
            fields.update({k: v for k, v in _extract_session_metadata(first, config).fields.items()
                           if k != 'animal_id'})
        task_name, task_config = config.task_config_for(fields)
        row['task_name'], row['task'] = task_name, task_config.make_task().name
        listed = _listed_type(config, animal_id, raw, session_id)
        column = fields.get('session_type')
        has_column = column is not None and str(column).strip().lower() not in ('', 'nan', 'none')
        if listed is not None:
            row['session_type'], row['session_type_source'] = listed, 'config'
        elif has_column:
            row['session_type'], row['session_type_source'] = str(column).strip(), 'column'
        else:
            row['session_type'], row['session_type_source'] = None, None
        if laser:
            any_laser = False
            if laser_mapping is not None:
                values = _look(raw, [laser_mapping.csv_name], None, fallback)
                if laser_mapping.csv_name in values.columns:
                    any_laser = bool(np.any(_safe_column(values, laser_mapping, len(values))))
            row['laser'] = any_laser
            if row['session_type'] is None:
                row['session_type'], row['session_type_source'] = ('opto' if any_laser else 'regular'), 'derived'
        row.update(fields)
        row.update({f'file_{name}': path for name, path in dict(raw.files).items()})
        rows.append(row)
    if fallback:
        warnings.warn(f'discover_sessions: the reader offers no cheap peek, so {len(set(fallback))} '
                      f'session table(s) were read in full (peek=False avoids it)', stacklevel=2)
    leading = _IDENTITY_COLUMNS + ['ignored', 'task_name', 'task', 'session_type', 'session_type_source']
    if not rows:
        return pd.DataFrame(columns=leading)
    table = pd.DataFrame(rows)
    rest = [c for c in table.columns if c not in leading]
    files = [c for c in rest if c.startswith('file_')]
    table = table[leading + [c for c in rest if c not in files] + files]
    return table.sort_values('animal_id', kind='stable').reset_index(drop=True)
