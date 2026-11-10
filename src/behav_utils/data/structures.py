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

import pickle
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
    """
    fields: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
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


def _scored(outcome) -> np.ndarray:
    """Trials with a correct response defined (outcome codes); every trial for anything else."""
    o = np.asarray(outcome)
    if o.dtype.kind in 'iuf':
        return o != Outcome.UNSCORED
    return np.ones(len(o), dtype=bool)


@dataclass
class TrialData:
    """
    Per-trial arrays for a single session.

    Required arrays (always present after loading):
        stimulus     float   Stimulus on the library's axis: [-1, 1] with the
                             category boundary at 0 (the loader applies the
                             config's ``task`` transform; identity for data
                             already on that axis)
        stimulus_raw float   The stimulus as read from the CSV
        choice       float   Category-space choice (0=A, 1=B, NaN=no response)
        outcome      int8    How the session's task scored the trial (``behav_utils.tasks.Outcome``:
                             ERROR 0, CORRECT 1, NO_RESPONSE 2, ABORT 3, UNSCORED 4)
        correct      bool    outcome == CORRECT
        category     float   The task's category for the stimulus (0, 1, NaN)
        trial_number int     1-indexed trial number from CSV

    Optional arrays (present if configured, sensible defaults otherwise):
        reaction_time  float   Response latency (NaN if missing)
        abort          bool    Trial aborted (hardware/animal error)
        opto_on        bool    Optogenetic light delivered on this trial
        distribution   str     Stimulus distribution label
        choice_raw     str     Raw choice before category-space mapping
        correct_recorded  bool  A mapped CSV ``correct`` column, kept to check the score (empty if unmapped)
        outcome_recorded  str   A mapped CSV ``outcome`` column, likewise

    Extra arrays (unmapped CSV columns):
        optional_fields  Dict[str, np.ndarray]  — mapped optional columns
        extra            Dict[str, np.ndarray]  — unmapped CSV columns

    Filtering:
        ``behav_utils.data.ops.filtering`` (``build_mask``, ``opto_mask``,
        ``filter_trial_data``, ``filter_session``, ``filter_trials``) builds new
        TrialData; nothing here filters.
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
    optional_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

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
            self.prev_correct = _gather_prev(self.correct, prev_idx)
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
        Get any field by name.

        Checks core fields, then optional_fields, then extra.
        Returns None if not found.
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
        session_id:  Unique identifier (the session directory name)
        session_idx: Ordinal index within animal (0-based, chronological)
        date:        Session date
        start:       Time of day the session started, when the reader knows it
        metadata:    SessionMetadata (stage, contingency, protocol, ...)
        trials:      TrialData (per-trial arrays)
        session_type: Session classification. The library derives 'regular' and
                     'opto' from the data; any other value is a project type stamped
                     from ``config.session_types`` (e.g. 'sham'). Single source of truth.
        session_id:  ``<animal>_<YYYYMMDD>``, plus ``T<HHMM>`` when the start time is known and
                     ``_2``, ``_3`` for later sessions that would otherwise share it (set at load)
        source_id:   The reader's own name for the session (a folder name, a table value)
        experimenter: Whose session (``project.experimenter``, or the animal metadata's)
        task:        The session's Task, built from the config; it alone defines correctness
        task_name:   'default', or the ``tasks`` entry that ``task_by`` picked
        session_type_source: Where session_type came from: 'config' (a session_types list),
                     'column' (a mapped session_type field) or 'derived' (the trials).
        csv_path:    Path to source CSV file
        filter_info: Metadata about filtering applied (None if unfiltered)

    Filtering:
        filter_session(session)                                      # abort + opto excluded
        filter_session(session, opto_mask(session.trials, 0))        # opto trials only
        filter_session(session).summary().accuracy                   # accuracy on what is left
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

    csv_path: str | None = None

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
        Trial arrays as a dict (a pure projection of ``trials``; nothing is
        filtered here, aborts included). Delegates to
        ``behav_utils.data.ops.filtering.get_arrays`` with this session's index
        so the ``trial_keys`` are global.
        """
        from behav_utils.data.ops.filtering import get_arrays
        return get_arrays(self.trials, session=self)

    def summary(self) -> 'SessionSummary':
        """Counts and accuracy over the trials this session currently holds."""
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
            filter=self.filter_info['label'] if self.filter_info else None,
        )


@dataclass(frozen=True)
class SessionSummary:
    """Counts and accuracy of one session (``SessionData.summary()``).

    ``accuracy`` is fraction correct over *valid* trials (non-abort, responded):
    the one definition of session accuracy in the library, used by
    ``SessionFilter.min_accuracy`` and ``AnimalData.session_table``.
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
    filter: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}



# =============================================================================
# ANIMAL DATA
# =============================================================================

SESSION_TABLE_COLUMNS = ['session_idx', 'session_id', 'date', 'start', 'stage', 'distribution', 'session_type', 'session_type_source', 'task',
                         'n_trials', 'n_valid', 'n_abort', 'n_no_response', 'accuracy']

def _session_order(session) -> tuple:
    """Chronological order: date, then start time (sessions without one after those with one),
    then session id, so two sessions on the same day always sort the same way."""
    return (session.date, session.start is None, session.start or time.min, session.session_id)


@dataclass
class AnimalData:
    """
    All data for a single animal. Unit of model fitting.
    Sessions stored chronologically.
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
        """One row per session — a tabular view for building selection masks.

        Convenience for picking *sessions* (not trials). Mask against it, e.g.::

            t = animal.session_table
            opto = t[t.session_type == 'opto']                 # sessions with opto trials
            good = t[(t.distribution == 'uniform') & (t.accuracy > 0.7)]

        Columns (exactly these; the per-session *statistics* come from
        ``compute_phase_stats(..., per_session=True)``, not from here):
            session_idx, session_id, date, stage, distribution, session_type,
            n_trials, n_valid, n_abort, n_no_response, accuracy

        ``session_type`` is 'regular' | 'opto' (derived from the data) or a type
        stamped from ``config.session_types``; ``session_type_source`` says which. ``accuracy`` is
        ``SessionData.summary().accuracy``. Trial-level opto masks are not a
        session property — use ``filter_trials(trial_type=...)``.
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
            If return_indices=False (default):
                List of matching SessionData, in chronological order.
            If return_indices=True:
                (sessions, indices) where indices is a list of
                session_idx values for each matched session.
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

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AnimalData':
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# EXPERIMENT DATA
# =============================================================================

@dataclass
class ExperimentData:
    """
    Top-level container for all animals.
    Provides query API for multi-animal analysis and plotting.
    """
    animals: Dict[str, AnimalData] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Any | None = field(default=None, repr=False)  # ProjectConfig

    def add_animal(self, animal: AnimalData) -> None:
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
        if animal_id not in self.animals:
            raise KeyError(
                f"Animal '{animal_id}' not found. "
                f"Available: {self.animal_ids}"
            )
        return self.animals[animal_id]

    def summary(self) -> pd.DataFrame:
        """One-row-per-animal summary."""
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
    def save(self, path: Union[str, Path]) -> None:
        """Pickle the experiment (configs are dropped — reload from YAML).

        ``metadata['behav_utils_version']`` and ``metadata['config_name']`` are
        stamped so ``load`` can warn when a snapshot and the library disagree.
        """
        from behav_utils import __version__
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata['behav_utils_version'] = __version__
        if self.config is not None:
            self.metadata['config_name'] = getattr(self.config, 'name', None)
        # Don't pickle config objects — reload from YAML
        config_backup = self.config
        self.config = None
        animal_configs = {}
        for aid, animal in self.animals.items():
            animal_configs[aid] = getattr(animal, '_config', None)
            animal._config = None
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self.config = config_backup
        for aid, animal in self.animals.items():
            animal._config = animal_configs[aid]

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentData':
        """Unpickle; warns if the snapshot was written by another library version."""
        import warnings

        from behav_utils import __version__
        with open(path, 'rb') as f:
            exp = pickle.load(f)
        written = (exp.metadata or {}).get('behav_utils_version')
        if written != __version__:
            warnings.warn(
                f"{Path(path).name}: written by behav_utils {written or '<unknown>'}, "
                f"loaded by {__version__}; re-run load_experiment if numbers matter.",
                stacklevel=2)
        return exp
