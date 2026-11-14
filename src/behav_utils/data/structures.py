"""
Core Data Structures

Hierarchical containers for behavioural data:

    ExperimentData          All animals in one project
    └── AnimalData          One animal, chronologically ordered sessions
        └── SessionData     One session
            ├── SessionMetadata   Constant within session (stage, contingency, ...)
            └── TrialData         Per-trial arrays (stimulus, choice, correct, ...)

The data classes hold data and nothing else: no filtering (``data/ops``), no
statistics (``stats``, ``analysis``), no plotting. ``SessionData.get_arrays`` is
the one convenience wrapper.

Usage:
    from behav_utils import load_experiment, select_sessions, filter_trials, TrialArrays

    experiment = load_experiment('config.yaml')
    animal = experiment.get_animal('A05')
    sessions = select_sessions(animal, preset='expert')
    clean = filter_trials(sessions, trial_type='non_opto')
    arrays = TrialArrays.from_sessions(clean)
"""

from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from behav_utils.data.arrays import session_hash
from behav_utils.tasks import Outcome, Task, TwoAFC

if TYPE_CHECKING:
    pass


# =============================================================================
# SESSION METADATA
# =============================================================================

@dataclass
class SessionMetadata:
    """
    Session-level metadata (constant within a session).
    Populated from the config's session_metadata mappings.

    The 'fields' dict holds all metadata key-value pairs.
    Common fields are exposed as properties for convenience.

    Attributes:
        fields: Every metadata field of the session, ``{name: value}``.
    """
    fields: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        One metadata field, like ``dict.get``.

        Args:
            key: The field's name.
            default: What to return when the session has no such field.

        Returns:
            The field's value, or ``default``.
        """
        return self.fields.get(key, default)

    def __getattr__(self, name: str) -> Any:
        # Allow attribute-style access to fields
        if name == 'fields' or name.startswith('_'):
            raise AttributeError(name)
        if name in self.fields:
            return self.fields[name]
        raise AttributeError(
            f"SessionMetadata has no field '{name}'. "
            f"Available: {list(self.fields.keys())}"
        )

    # The two fields the library itself reads; everything else via .get() / attribute.
    @property
    def animal_id(self) -> str:
        return self.fields.get('animal_id', '')

    @property
    def stage(self) -> str:
        return self.fields.get('stage', '')


# =============================================================================
# PREVIOUS-TRIAL (LAG-1) VIEW
# =============================================================================


def _prev_nonabort_index(abort: np.ndarray) -> np.ndarray:
    """
    For each trial ``i``, the index of the most recent NON-ABORT trial strictly
    before ``i`` within this session, or ``-1`` if there is none.

    An abort is not a trial the animal completed, so it is skipped when
    determining the previous trial: a trial that follows one or more aborts
    takes the last completed trial before them as its predecessor (the abort is
    "bridged"). This matches the sequence once aborts are filtered out, while
    opto and no-response trials — real, completed trials — are NOT skipped. The
    session is the block, so there is no predecessor before its first completed
    trial. Abort trials themselves get ``-1`` (no valid predecessor, and they
    are not a predecessor for later trials).
    """
    n = len(abort)
    prev_idx = np.full(n, -1, dtype=int)
    last = -1
    for i in range(n):
        if abort[i]:
            continue          # abort: keep -1, and do not become a predecessor
        prev_idx[i] = last
        last = i
    return prev_idx


def _gather_prev(arr: np.ndarray, prev_idx: np.ndarray) -> np.ndarray:
    """
    Gather ``arr`` at ``prev_idx`` (previous-non-abort index), as float, with
    NaN where there is no predecessor (``prev_idx < 0``).

    Float so booleans/ints carry a NaN sentinel; a NaN already present in
    ``arr`` (e.g. a no-response choice) carries through, so a NaN in the result
    means "no predecessor" or "the previous trial was a no-response".
    """
    out = np.full(len(arr), np.nan, dtype=float)
    valid = prev_idx >= 0
    if valid.any():
        out[valid] = np.asarray(arr, dtype=float)[prev_idx[valid]]
    return out


@dataclass
class PrevTrial:
    """
    Lag-1 view of the previous trial's values, aligned to the current trial.

    Built once on the raw session and carried (not recomputed) through
    filtering and pooling, so each trial keeps its *real* predecessor even
    after interleaved trials are removed. The predecessor is the previous
    NON-ABORT trial within the session (the block): aborts are skipped, since
    an abort is not a completed trial, whereas opto and no-response trials are
    real and are not skipped. Numeric fields are NaN where there is no
    predecessor or the previous trial was a no-response; ``has_prev`` flags
    trials with a real predecessor (the per-trial replacement for
    ``not_blockstart``).
    """
    stimulus: np.ndarray
    choice: np.ndarray
    correct: np.ndarray
    category: np.ndarray
    reaction_time: np.ndarray
    opto_on: np.ndarray
    has_prev: np.ndarray


# =============================================================================
# TRIAL DATA
# =============================================================================


# Marks the TrialData fields that hold one value per trial; filtering slices exactly these.
_PER_TRIAL = {'per_trial': True}
# Marks the TrialData fields that map names to per-trial arrays; filtering slices every value in them.
_PER_TRIAL_MAP = {'per_trial_map': True}


def _correct_as_float(outcome, correct, choice) -> np.ndarray:
    """The score as 1.0 (correct) / 0.0 (error) / NaN where there is none: ``correct``, blanked where
    the outcome codes give no score (no response, abort, unscored) or the choice is missing. Loaded data
    always has the two in agreement; TrialData built by hand may carry other outcome labels."""
    out = np.asarray(correct, dtype=float).copy()
    codes = np.asarray(outcome)
    if codes.dtype.kind in 'iuf' and len(codes) == len(out):
        out[(codes != Outcome.CORRECT) & (codes != Outcome.ERROR)] = np.nan
    out[np.isnan(np.asarray(choice, dtype=float))] = np.nan
    return out


def _scored(outcome) -> np.ndarray:
    """Trials with a correct response defined (outcome codes); every trial for anything else."""
    o = np.asarray(outcome)
    if o.dtype.kind in 'iuf':
        return o != Outcome.UNSCORED
    return np.ones(len(o), dtype=bool)


@dataclass
class TrialData:
    """
    Per-trial arrays for a single session, all of the same length.

    Functions in ``behav_utils.data`` (``build_mask``, ``opto_mask``, ``filter_session``, ``filter_trials``)
    build new TrialData; nothing here filters.

    Attributes:
        trial_number: 1-based trial number, as read.
        stimulus: The stimulus on the library's axis, [-1, 1] with the category boundary at 0 (the loader
            applies the config's ``task`` transform, the identity for data already on that axis).
        choice: The response in category space: 0 = A, 1 = B, NaN = no response.
        outcome: How the session's task scored the trial, as ``Outcome`` codes (int8).
        correct: ``outcome == Outcome.CORRECT``.
        category: The task's category for the stimulus: 0, 1, or NaN where it has none.
        choice_raw: The response as read, before mapping to category space.
        stimulus_raw: The stimulus as read.
        reaction_time: Response latency in seconds; NaN where missing.
        abort: Whether the trial was aborted.
        opto_on: Whether the laser was on in the trial.
        distribution: The stimulus distribution's label.
        correct_recorded: A mapped ``correct`` column, kept to check the task's score against; empty if
            unmapped.
        outcome_recorded: A mapped ``outcome`` column, likewise.
        optional_fields: Mapped optional columns, ``{name: array}``.
        extra: Columns the config does not map, ``{name: array}``.
        events: Event times (columns named ``t_*``) in float64 seconds on the session's clock,
            ``{name: array}``.
        prev_stimulus: The stimulus of the previous non-aborted trial; NaN where there is none.
        prev_choice: That trial's choice.
        prev_correct: Whether that trial was correct, as 1.0, 0.0 or NaN.
        prev_category: That trial's category.
        prev_reaction_time: That trial's reaction time.
        prev_opto_on: Whether the laser was on in that trial.
        prev_has_prev: Whether the trial has such a previous trial.

    The ``prev_*`` arrays are computed once on the raw session, skipping aborts but not laser or
    no-response trials, and carried unchanged through filtering and pooling, so a filtered trial's
    previous trial is still the one the animal saw.
    """
    # ── Required ────────────────────────────────────────────────────────────
    trial_number: np.ndarray = field(metadata=_PER_TRIAL)
    stimulus: np.ndarray = field(metadata=_PER_TRIAL)
    choice: np.ndarray = field(metadata=_PER_TRIAL)
    outcome: np.ndarray = field(metadata=_PER_TRIAL)
    correct: np.ndarray = field(metadata=_PER_TRIAL)
    category: np.ndarray = field(metadata=_PER_TRIAL)

    choice_raw: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)
    stimulus_raw: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)

    # ── Optional ────────────────────────────────────────────────────────────
    reaction_time: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)
    abort: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)
    opto_on: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)
    distribution: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)
    correct_recorded: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)
    outcome_recorded: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)

    # ── All other columns ───────────────────────────────────────────────────
    optional_fields: Dict[str, np.ndarray] = field(default_factory=dict, metadata=_PER_TRIAL_MAP)
    extra: Dict[str, np.ndarray] = field(default_factory=dict, metadata=_PER_TRIAL_MAP)
    events: Dict[str, np.ndarray] = field(default_factory=dict, metadata=_PER_TRIAL_MAP)

    # ── Derived: lag-1 (previous trial) ───────────────────────────────────────
    # Frozen on the raw session and carried (sliced) through filtering/pooling.
    # Empty on construction → computed in __post_init__; non-empty (e.g. from a
    # filter slice) → carried as-is. Exposed via the `prev_trial` property.
    prev_stimulus: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)
    prev_choice: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)
    prev_correct: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)
    prev_category: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)
    prev_reaction_time: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)
    prev_opto_on: np.ndarray = field(default_factory=lambda: np.array([]), metadata=_PER_TRIAL)
    prev_has_prev: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool), metadata=_PER_TRIAL)

    def __post_init__(self):
        """Set defaults for empty optional arrays, then build the lag-1 view."""
        n = len(self.stimulus)
        if len(self.stimulus_raw) == 0:
            self.stimulus_raw = np.asarray(self.stimulus, dtype=float).copy()
        if len(self.reaction_time) == 0:
            self.reaction_time = np.full(n, np.nan)
        if len(self.abort) == 0:
            self.abort = np.zeros(n, dtype=bool)
        if len(self.opto_on) == 0:
            self.opto_on = np.zeros(n, dtype=bool)

        # Lag-1 view: compute on the raw sequence, but ONLY if not already
        # supplied. A filter/pool slice passes these in, and they must not be
        # recomputed — otherwise prev would point at the previous *surviving*
        # trial instead of the real one.
        if len(self.prev_stimulus) == 0 and n > 0:
            # Abort-aware, block(session)-aware: the predecessor is the most
            # recent NON-ABORT trial earlier in this session. Aborts are skipped
            # (bridged) so the carried view still matches the sequence once
            # aborts are filtered out; opto / no-response trials are real and
            # are NOT skipped. With no aborts this is an ordinary one-trial lag.
            prev_idx = _prev_nonabort_index(self.abort)
            self.prev_stimulus = _gather_prev(self.stimulus, prev_idx)
            self.prev_choice = _gather_prev(self.choice, prev_idx)
            self.prev_correct = _gather_prev(_correct_as_float(self.outcome, self.correct, self.choice), prev_idx)
            self.prev_category = _gather_prev(self.category, prev_idx)
            self.prev_reaction_time = _gather_prev(self.reaction_time, prev_idx)
            self.prev_opto_on = _gather_prev(self.opto_on, prev_idx)
            self.prev_has_prev = prev_idx >= 0

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def n_trials(self) -> int:
        """Number of trials."""
        return len(self.stimulus)

    @property
    def no_response(self) -> np.ndarray:
        """Boolean mask: True where choice is NaN (animal didn't respond)."""
        return np.isnan(self.choice.astype(float))

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask: non-abort, responded trials."""
        return ~self.abort & ~self.no_response

    @property
    def prev_trial(self) -> 'PrevTrial':
        """
        Lag-1 view of the previous trial's values (see PrevTrial). Frozen on
        the raw session and carried through filtering/pooling, so it stays the
        real predecessor even after interleaved trials are removed.
        """
        return PrevTrial(
            stimulus=self.prev_stimulus,
            choice=self.prev_choice,
            correct=self.prev_correct,
            category=self.prev_category,
            reaction_time=self.prev_reaction_time,
            opto_on=self.prev_opto_on,
            has_prev=self.prev_has_prev,
        )

    # ── Field access ────────────────────────────────────────────────────────

    def get_field(self, name: str) -> np.ndarray | None:
        """
        Any per-trial field by name.

        The core fields are searched first, then ``optional_fields``, then ``extra``.

        Args:
            name: The field's name.

        Returns:
            The array, or None when there is no such field.
        """
        if hasattr(self, name) and name not in ('optional_fields', 'extra'):
            val = getattr(self, name)
            if isinstance(val, np.ndarray):
                return val
        if name in self.optional_fields:
            return self.optional_fields[name]
        if name in self.extra:
            return self.extra[name]
        return None

# =============================================================================
# SESSION DATA
# =============================================================================

@dataclass
class SessionData:
    """
    All data for a single behavioural session.

    Attributes:
        session_id: ``<animal>_<YYYYMMDD>``, plus ``T<HHMM>`` when the start time is known; sessions that would
            still share an id are all numbered ``_1``, ``_2``, ... in name order (set at load).
        session_idx: Position among the animal's sessions, 0-based and chronological.
        date: The session's date.
        metadata: ``SessionMetadata``: stage, contingency, protocol and the other session-level fields.
        trials: The per-trial arrays, ``TrialData``.
        session_type: 'regular' or 'opto', derived from the data, or a project type stamped from
            ``config.session_types`` (e.g. 'sham'); the one place a session's type is kept.
        start: Time of day the session started, when the reader knows it.
        session_type_source: Where ``session_type`` came from: 'config' (a ``session_types`` list),
            'column' (a mapped session_type field) or 'derived' (the trials).
        source_id: The reader's own name for the session (a folder name, a table value).
        experimenter: Whose session (``project.experimenter``, or the animal metadata's).
        task: The session's ``Task``, built from the config; it alone defines what is correct.
        task_name: 'default', or the ``tasks`` entry that ``task_by`` picked.
        clock: The clock the event times are on (``config.clock.name``; '' if none).
        source: Where the reader found the session (a folder, a file, a table key).
        filter_info: What filtering produced this session; None if it is unfiltered.

    To filter, ``filter_session(session)`` drops aborted and laser-on trials and takes a mask for anything
    else; *Filtering trials* in the data-structures reference shows it on synthetic data.
    """
    session_id: str
    session_idx: int
    date: date
    metadata: SessionMetadata
    trials: TrialData
    session_type: str = 'regular'
    start: time | None = None

    session_type_source: str = 'derived'    # 'config', 'column' or 'derived': where session_type came from
    source_id: str | None = None            # the reader's own name for the session (folder, table value)
    experimenter: str = ''                  # project.experimenter or the animal's; part of session_hash
    task: Task = field(default_factory=TwoAFC)   # decides what counts as correct (from the config)
    task_name: str = 'default'              # 'default', or the name in config.tasks that task_by picked
    clock: str = ''                         # the recording clock event times are on (config.clock.name)

    source: str | None = None               # where the reader found it (a folder, a file, a table key)

    # Filter provenance (None = unfiltered raw data)
    filter_info: Dict[str, Any] | None = field(default=None, repr=False)

    # Set by AnimalData after construction
    _days_since_first: float | None = field(default=None, repr=False)

    @property
    def session_hash(self) -> int:
        """43-bit hash of experimenter, animal and session id: the upper bits of this session's trial
        keys, the same on every machine and every load."""
        return session_hash(self.experimenter, str(self.metadata.get('animal_id', '') or ''), self.session_id)

    @property
    def n_trials(self) -> int:
        """Number of trials (reflects filtering if filtered)."""
        return self.trials.n_trials

    @property
    def stage(self) -> str:
        return self.metadata.stage

    @property
    def distribution(self) -> str:
        dist = self.trials.get_field('distribution')
        if dist is not None and len(dist) > 0:
            vals, counts = np.unique(dist[dist != ''], return_counts=True)
            if len(vals) > 0:
                return str(vals[counts.argmax()])
        return self.metadata.get('distribution', 'Unknown')

    @property
    def days_since_first(self) -> float | None:
        return self._days_since_first

    @property
    def is_filtered(self) -> bool:
        """Whether this session has been through filter()."""
        return self.filter_info is not None

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """
        The session's trial arrays as a dict, a projection of ``trials`` that filters nothing.

        Aborted trials included; ``get_arrays`` is called with this session, so the ``trial_keys`` are unique across
        sessions and animals.

        Returns:
            The arrays, keyed as in ``behav_utils.data.get_arrays``.
        """
        from behav_utils.data.ops.filtering import get_arrays
        return get_arrays(self.trials, session=self)

    def summary(self) -> 'SessionSummary':
        """
        Counts and accuracy over the trials this session currently holds.

        Returns:
            A ``SessionSummary``.
        """
        valid = self.trials.valid_mask
        n_valid = int(valid.sum())
        scored = valid & _scored(self.trials.outcome)
        return SessionSummary(
            session_id=self.session_id, session_idx=self.session_idx, date=self.date, start=self.start,
            stage=self.stage, distribution=self.distribution, session_type=self.session_type,
            session_type_source=self.session_type_source,
            n_trials=self.n_trials, n_valid=n_valid, n_abort=int(self.trials.abort.sum()),
            n_no_response=int(self.trials.no_response.sum()),
            accuracy=float(np.asarray(self.trials.correct, dtype=float)[scored].mean()) if scored.any() else np.nan,
            task=self.task.name,
            animal_id=str(self.metadata.get('animal_id', '') or ''),
            experimenter=self.experimenter,
            source_id=self.source_id,
            source=self.source,
            filter=self.filter_info['label'] if self.filter_info else None,
        )


@dataclass(frozen=True)
class SessionSummary:
    """
    Counts and accuracy of one session (``SessionData.summary()``).

    ``accuracy`` is fraction correct over *valid* trials (non-abort, responded):
    the one definition of session accuracy in the library, used by
    ``SessionFilter.min_accuracy`` and ``AnimalData.session_table``.

    Attributes:
        session_id: The session's id.
        session_idx: Its position among the animal's sessions.
        date: Its date.
        stage: Its stage, from the metadata.
        distribution: Its distribution label, from the metadata.
        session_type: Its type, as in ``SessionData.session_type``.
        n_trials: All trials the session holds.
        n_valid: Trials neither aborted nor without a response.
        n_abort: Aborted trials.
        n_no_response: Trials without a response.
        accuracy: The fraction correct over valid trials.
        start: Time of day the session started, when known.
        session_type_source: Where ``session_type`` came from.
        task: The name of the session's task block.
        animal_id: The animal.
        experimenter: Whose session.
        source_id: The reader's own name for the session.
        source: Where the reader found it.
        filter: What filtering produced the session; None if unfiltered.
    """
    session_id: str
    session_idx: int
    date: date
    stage: str
    distribution: str
    session_type: str
    n_trials: int
    n_valid: int
    n_abort: int
    n_no_response: int
    accuracy: float
    start: time | None = None
    session_type_source: str = 'derived'
    task: str = 'two_afc'
    animal_id: str = ''
    experimenter: str = ''
    source_id: str | None = None
    source: str | None = None
    filter: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        """
        The summary as a dict, one entry per field.

        Returns:
            ``{field: value}``.
        """
        return {f: getattr(self, f) for f in self.__dataclass_fields__}



# =============================================================================
# ANIMAL DATA
# =============================================================================

# The identity every session table shares: discovery (before loading), animal and experiment session
# tables (after), and the store's manifest.
_IDENTITY_COLUMNS = ['session_id', 'animal_id', 'experimenter', 'date', 'start', 'source_id', 'source']
SESSION_TABLE_COLUMNS = _IDENTITY_COLUMNS + ['session_idx', 'stage', 'distribution', 'session_type',
                                             'session_type_source', 'task', 'n_trials', 'n_valid', 'n_abort',
                                             'n_no_response', 'accuracy']

def _session_order(session) -> tuple:
    """Chronological order: date, then start time (sessions without one after those with one),
    then session id, so two sessions on the same day always sort the same way."""
    return (session.date, session.start is None, session.start or time.min, session.session_id)


@dataclass
class AnimalData:
    """
    All data for a single animal: its sessions, in chronological order, and its metadata.

    ``select_sessions`` picks sessions from it by preset or criteria, and ``session_table`` has one row per
    session for picking them by hand.

    Attributes:
        animal_id: The animal's id.
        sessions: Its sessions, ``SessionData``, in chronological order.
        metadata: Per-animal metadata from the config's ``animal_metadata`` file (``group``, for example).
    """
    animal_id: str
    sessions: List[SessionData]
    metadata: Dict[str, Any] = field(default_factory=dict)
    _config: Any | None = field(default=None, repr=False)

    _session_table_cache: pd.DataFrame | None = field(default=None, repr=False)

    def __post_init__(self):
        self._compute_time_axes()

    def _compute_time_axes(self):
        if not self.sessions:
            return
        self.sessions.sort(key=_session_order)
        first_date = self.sessions[0].date
        for i, sess in enumerate(self.sessions):
            sess.session_idx = i
            sess._days_since_first = (sess.date - first_date).days

    def invalidate_cache(self):
        """Call when sessions change."""
        self._session_table_cache = None

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def session_ids(self) -> list:
        """List of session IDs in chronological order."""
        return [s.session_id for s in self.sessions]

    @property
    def stages(self) -> List[str]:
        return list(dict.fromkeys(s.stage for s in self.sessions))

    @property
    def session_table(self) -> pd.DataFrame:
        """One-row-per-session summary table for selection masks."""
        if getattr(self, '_session_table_cache', None) is None:
            self._session_table_cache = self.create_session_table()
        return self._session_table_cache

    def create_session_table(self) -> pd.DataFrame:
        """
        One row per session — a tabular view for building selection masks.

        Convenience for picking *sessions* (not trials). Mask against it like any DataFrame, for example
        ``t[t.session_type == 'opto']`` for the sessions with laser trials.

        Columns (exactly these; the per-session *statistics* come from
        ``compute_phase_stats(..., per_session=True)``, not from here)::

            session_idx, session_id, date, stage, distribution, session_type,
            n_trials, n_valid, n_abort, n_no_response, accuracy

        ``session_type`` is 'regular' | 'opto' (derived from the data) or a type
        stamped from ``config.session_types``; ``session_type_source`` says which. ``accuracy`` is
        ``SessionData.summary().accuracy``. Trial-level opto masks are not a
        session property — use ``filter_trials(trial_type=...)``.

        Returns:
            The table, one row per session.
        """
        return pd.DataFrame([s.summary().to_dict() for s in self.sessions],
                            columns=SESSION_TABLE_COLUMNS)

    # ── Filtering ───────────────────────────────────────────────────────────

    def get_sessions(
        self,
        stage: Union[str, List[str]] | None = None,
        distribution: Union[str, List[str]] | None = None,
        idx: Union[int, List[int], np.ndarray] | None = None,
        idx_range: Tuple[int, int] | None = None,
        mask: Union[np.ndarray, pd.Series] | None = None,
        date_range: Tuple[date, date] | None = None,
        return_indices: bool = False,
    ) -> Union[List[SessionData], Tuple[List[SessionData], List[int]]]:
        """
        Filter sessions by criteria.

        Args:
            stage: Stage filter. A single string matches exactly;
                a list matches any (OR logic). None = no filter.
            distribution: Distribution filter. A single string matches exactly;
                a list matches any (OR logic). None = no filter.
            idx: Specific session_idx values to include (int or list).
            idx_range: (start, end) session_idx range (inclusive)
            mask: Boolean mask, length = n_sessions, positionally aligned with
                self.sessions (i.e. with session_table rows). Selects sessions.
            date_range: (start, end) date range (inclusive)
            return_indices: If True, also return session_idx values.

        Returns:
            The matching sessions, in chronological order. With ``return_indices=True``, the tuple
            ``(sessions, indices)``, where ``indices`` holds each matched session's ``session_idx``.
        """
        sessions = self.sessions

        if mask is not None:
            # Mask is positional over self.sessions (the order session_table is
            # built in), NOT keyed by session_idx. Resolve it to the concrete
            # sessions first, so it stays correct when session_idx is
            # non-contiguous and doesn't misalign with the other filters below.
            m = mask.to_numpy() if isinstance(mask, (pd.Series, pd.DataFrame)) else np.asarray(mask)
            sessions = [s for s, keep in zip(sessions, m) if keep]

        if stage is not None:
            if isinstance(stage, list):
                stage_set = set(stage)
                sessions = [s for s in sessions if s.stage in stage_set]
            else:
                sessions = [s for s in sessions if s.stage == stage]
        if distribution is not None:
            if isinstance(distribution, list):
                dist_set = set(distribution)
                sessions = [s for s in sessions if s.distribution in dist_set]
            else:
                sessions = [s for s in sessions if s.distribution == distribution]
        if idx is not None:
            if isinstance(idx, (int, np.integer)):
                idx = [int(idx)]
            idx_set = set(int(i) for i in idx)
            sessions = [s for s in sessions if s.session_idx in idx_set]
        if idx_range is not None:
            sessions = [s for s in sessions
                        if idx_range[0] <= s.session_idx <= idx_range[1]]
        if date_range is not None:
            sessions = [s for s in sessions
                        if date_range[0] <= s.date <= date_range[1]]

        if return_indices:
            return sessions, [s.session_idx for s in sessions]
        return sessions

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path], *, overwrite: bool = False) -> Path:
        """
        Save this animal as a store: a directory of Parquet tables and JSON that reloads exactly.

        Args:
            path: The store's directory.
            overwrite: Replace an existing store at ``path``.

        Returns:
            The store's path.
        """
        from behav_utils.data.store import save_store
        experiment = ExperimentData()
        experiment.add_animal(self)
        return save_store(experiment, path, overwrite=overwrite)

    @classmethod
    def load(cls, path: Union[str, Path], animal_id: str | None = None) -> 'AnimalData':
        """
        Load an animal from a store, opening only its sessions' files.

        Args:
            path: The store's directory.
            animal_id: Which animal, when the store holds several; None when it holds one.

        Returns:
            The animal.
        """
        from behav_utils.data.store import StoreError, load_store, read_manifest
        manifest = read_manifest(path)
        animals = sorted(set(manifest['animal_id']))
        if animal_id is None:
            if len(animals) != 1:
                raise StoreError(f"{path} holds {len(animals)} animals ({', '.join(animals[:10])}); pass animal_id")
            animal_id = animals[0]
        if animal_id not in animals:
            raise StoreError(f"{path} has no animal {animal_id!r} (it has {', '.join(animals[:10])})")
        experiment = load_store(path, sessions=manifest[manifest['animal_id'] == animal_id])
        return experiment.animals[animal_id]


# =============================================================================
# EXPERIMENT DATA
# =============================================================================

@dataclass
class ExperimentData:
    """
    Top-level container for all animals, with the config they were loaded with.

    ``load_experiment`` returns one, and ``generate_synthetic_experiment`` builds one with the same structure.
    It answers queries across animals: ``get_animal``, ``summary``, ``session_table``.

    Attributes:
        animals: The animals, ``{animal_id: AnimalData}``.
        metadata: Free-form metadata about the experiment.
        config: The ``ProjectConfig`` it was loaded with; None when built in code.
    """
    animals: Dict[str, AnimalData] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Any | None = field(default=None, repr=False)  # ProjectConfig

    def add_animal(self, animal: AnimalData) -> None:
        """
        Add an animal, keyed by its id.

        Args:
            animal: The animal to add.
        """
        self.animals[animal.animal_id] = animal
        # Propagate config reference
        if self.config is not None:
            animal._config = self.config

    @property
    def animal_ids(self) -> List[str]:
        return sorted(self.animals.keys())

    @property
    def n_animals(self) -> int:
        return len(self.animals)

    def get_animal(self, animal_id: str) -> AnimalData:
        """
        One animal, by its id.

        Args:
            animal_id: The animal's id.

        Returns:
            The animal.
        """
        if animal_id not in self.animals:
            raise KeyError(
                f"Animal '{animal_id}' not found. "
                f"Available: {self.animal_ids}"
            )
        return self.animals[animal_id]

    @property
    def session_table(self) -> pd.DataFrame:
        """Every animal's sessions in one table: the identity columns first (as ``discover_sessions``
        gives them), then each session's summary."""
        tables = [self.animals[a].session_table for a in sorted(self.animals)]
        tables = [t for t in tables if len(t)]
        if not tables:
            return pd.DataFrame(columns=SESSION_TABLE_COLUMNS)
        return pd.concat(tables, ignore_index=True)

    def summary(self) -> pd.DataFrame:
        """
        One-row-per-animal summary.

        Returns:
            A table with each animal's number of sessions, stages and first and last dates.
        """
        rows = []
        for aid, animal in self.animals.items():
            rows.append({
                'animal_id': aid,
                'n_sessions': animal.n_sessions,
                'stages': animal.stages,
                'date_first': animal.sessions[0].date if animal.sessions else None,
                'date_last': animal.sessions[-1].date if animal.sessions else None,
            })
        return pd.DataFrame(rows)

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, path: Union[str, Path], *, overwrite: bool = False) -> Path:
        """
        Save the experiment as a store: a directory of Parquet tables and JSON that reloads exactly.

        The store can be reloaded whole or in part (see ``save_store`` and ``load_store``). The config is not stored;
        reload it from its YAML.

        Args:
            path: The store's directory.
            overwrite: Replace an existing store at ``path``; any other existing path is refused.

        Returns:
            The store's path.
        """
        from behav_utils.data.store import save_store
        return save_store(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path: Union[str, Path], *, sessions=None, strict: bool = True) -> 'ExperimentData':
        """
        Load a store written by ``save``, whole or only the chosen sessions.

        Pickle snapshots from before 1.0 cannot be opened: reload from the raw data with ``load_experiment`` and save
        again.

        Args:
            path: The store's directory.
            sessions: Only these sessions: a filtered ``read_manifest`` or session ids; None for all.
            strict: Raise when the store does not have a requested session (the default); False warns instead.

        Returns:
            The experiment.
        """
        from behav_utils.data.store import load_store
        return load_store(path, sessions=sessions, strict=strict)
