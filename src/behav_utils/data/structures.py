"""
Core Data Structures

Hierarchical containers for behavioural data:

    ExperimentData          All animals in one project
    └── AnimalData          One animal, chronologically ordered sessions
        └── SessionData     One session
            ├── SessionMetadata   Constant within session (stage, contingency, ...)
            └── TrialData         Per-trial arrays (stimulus, choice, correct, ...)


Convention — three levels per domain:
    Low-level:     fit_psychometric(stim, ch)       — raw arrays
    Session-level: compute_psychometric_curve(arrays) — TrialArrays → PsychometricCurve
    Plotting:      plot_psychometric_curve(curve)      — dataclass → axes

Plot methods on data classes are thin wrappers that call compute_ then plot_.

Usage:
    from behav_utils import (
        load_experiment, select_sessions, filter_trials,
        TrialArrays, compute_psychometric_curve, compute_update_matrix,
        plot_psychometric_curve, plot_update_matrix, PALETTE,
    )

    experiment = load_experiment('config.yaml')
    animal = experiment.get_animal('A05')
    sessions = select_sessions(animal, preset='expert_uniform')
    clean = filter_trials(sessions)

    psych = compute_psychometric_curve(TrialArrays.from_sessions(clean), n_bootstrap=200)
    fig, ax = plt.subplots()
    plot_psychometric_curve(psych, ax=ax, color=PALETTE[0])
"""

import pickle
from dataclasses import dataclass, field
from datetime import date
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

    # Common convenience properties
    @property
    def animal_id(self) -> str:
        return self.fields.get('animal_id', '')

    @property
    def stage(self) -> str:
        return self.fields.get('stage', '')

    @property
    def sound_contingency(self) -> str:
        return self.fields.get('sound_contingency', '')

    @property
    def stim_range_min(self) -> float:
        return self.fields.get('stim_range_min', -1.0)

    @property
    def stim_range_max(self) -> float:
        return self.fields.get('stim_range_max', 1.0)


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


@dataclass
class TrialData:
    """
    Per-trial arrays for a single session.

    Required arrays (always present after loading):
        stimulus     float   Raw stimulus values (normalised to [-1, 1])
        choice       float   Category-space choice (0=A, 1=B, NaN=no response)
        outcome      str     Trial outcome label from CSV
        correct      bool    Whether choice matched category
        category     int     Correct category (0 or 1), derived from stimulus
        trial_number int     1-indexed trial number from CSV

    Optional arrays (present if configured, sensible defaults otherwise):
        reaction_time  float   Response latency (NaN if missing)
        abort          bool    Trial aborted (hardware/animal error)
        opto_on        bool    Optogenetic light delivered on this trial
        distribution   str     Stimulus distribution label
        choice_raw     str     Raw choice before category-space mapping

    Extra arrays (unmapped CSV columns):
        optional_fields  Dict[str, np.ndarray]  — mapped optional columns
        extra            Dict[str, np.ndarray]  — unmapped CSV columns

    Filtering:
        Use build_mask() or opto_mask() to create boolean masks,
        then filter(mask) to create a new TrialData with only those trials.
        Analysis and plotting functions receive pre-filtered data —
        they do NOT filter internally.
    """
    # ── Required ────────────────────────────────────────────────────────────
    trial_number: np.ndarray
    stimulus: np.ndarray
    choice: np.ndarray
    outcome: np.ndarray
    correct: np.ndarray
    category: np.ndarray

    choice_raw: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Optional ────────────────────────────────────────────────────────────
    reaction_time: np.ndarray = field(default_factory=lambda: np.array([]))
    abort: np.ndarray = field(default_factory=lambda: np.array([]))
    opto_on: np.ndarray = field(default_factory=lambda: np.array([]))
    distribution: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── All other columns ───────────────────────────────────────────────────
    optional_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    # ── Derived: lag-1 (previous trial) ───────────────────────────────────────
    # Frozen on the raw session and carried (sliced) through filtering/pooling.
    # Empty on construction → computed in __post_init__; non-empty (e.g. from a
    # filter slice) → carried as-is. Exposed via the `prev_trial` property.
    prev_stimulus: np.ndarray = field(default_factory=lambda: np.array([]))
    prev_choice: np.ndarray = field(default_factory=lambda: np.array([]))
    prev_correct: np.ndarray = field(default_factory=lambda: np.array([]))
    prev_category: np.ndarray = field(default_factory=lambda: np.array([]))
    prev_reaction_time: np.ndarray = field(default_factory=lambda: np.array([]))
    prev_opto_on: np.ndarray = field(default_factory=lambda: np.array([]))
    prev_has_prev: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    def __post_init__(self):
        """Set defaults for empty optional arrays, then build the lag-1 view."""
        n = len(self.stimulus)
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
        session_id:  Unique identifier (e.g. 'SOUND_CAT_SS14_2026_05_06')
        session_idx: Ordinal index within animal (0-based, chronological)
        date:        Session date
        metadata:    SessionMetadata (stage, contingency, protocol, ...)
        trials:      TrialData (per-trial arrays)
        session_type: Session classification. The library derives 'regular' and
                     'opto' from the data; any other value is a project type stamped
                     from ``config.session_types`` (e.g. 'sham'). Single source of truth.
        csv_path:    Path to source CSV file
        filter_info: Metadata about filtering applied (None if unfiltered)

    Filtering:
        session.filter()                              # standard: exclude abort + opto
        session.filter(session.trials.opto_mask(0))   # custom: opto trials only
        filtered_session.summary()['perf']             # accuracy on filtered trials
    """
    session_id: str
    session_idx: int
    date: date
    metadata: SessionMetadata
    trials: TrialData
    session_type: str = 'regular'

    # True when session_type came from an explicit CSV column (so config-list
    # overrides in load_experiment defer to it). Internal.
    _session_type_explicit: bool = field(default=False, repr=False)

    csv_path: str | None = None

    # Filter provenance (None = unfiltered raw data)
    filter_info: Dict[str, Any] | None = field(default=None, repr=False)

    # Set by AnimalData after construction
    _days_since_first: float | None = field(default=None, repr=False)

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
        Extract trial arrays for analysis. Aborts are excluded.

        Permitted convenience method — 9+ callers use sess.get_arrays()
        which reads more naturally than get_arrays(sess.trials).
        Delegates to filtering.get_arrays.

        No filtering is done here. Filter before calling:
            filtered = filter_session(sess, mask)
            arr = filtered.get_arrays()
        """
        from behav_utils.data.ops.filtering import get_arrays
        return get_arrays(self.trials)


    def summary(self) -> Dict[str, Any]:
        """Quick summary dict with basic counts and accuracy."""
        valid = self.trials.valid_mask
        n_valid = valid.sum()
        choices = self.trials.choice[valid].astype(float)
        cats = self.trials.category[valid]
        result = {
            'session_id': self.session_id,
            'session_idx': self.session_idx,
            'date': self.date,
            'stage': self.stage,
            'distribution': self.distribution,
            'n_trials': self.n_trials,
            'n_valid': int(n_valid),
            'n_abort': int(self.trials.abort.sum()),
            'perf': float((choices == cats).mean()) if n_valid > 0 else np.nan,
        }
        if self.filter_info:
            result['filter'] = self.filter_info['label']
        return result



# =============================================================================
# ANIMAL DATA
# =============================================================================

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

    # Cache
    _feature_matrix_cache: pd.DataFrame | None = field(
        default=None, repr=False
    )

    def __post_init__(self):
        self._compute_time_axes()

    def _compute_time_axes(self):
        if not self.sessions:
            return
        self.sessions.sort(key=lambda s: s.date)
        first_date = self.sessions[0].date
        for i, sess in enumerate(self.sessions):
            sess.session_idx = i
            sess._days_since_first = (sess.date - first_date).days

    def invalidate_cache(self):
        """Call when sessions change."""
        self._feature_matrix_cache = None

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
        if self._feature_matrix_cache is None:
            self._feature_matrix_cache = self.create_session_table()
        return self._feature_matrix_cache

    def create_session_table(self) -> pd.DataFrame:
        """One row per session — a tabular view for building selection masks.

        Convenience for picking *sessions* (not trials). Mask against it, e.g.::

            t = animal.session_table
            opto = t[t.session_type == 'opto']                 # sessions with opto trials
            manipulated = t[t.session_type != 'regular']       # any non-regular type
            good = t[(t.distribution == 'uniform') & (t.accuracy > 0.7)]

        Columns:
            session_idx, session_id, date, stage, distribution,
            n_trials, n_valid,
            session_type — 'regular' | 'opto' (derived from the data) or any
                type name stamped from ``config.session_types``,
            accuracy — fraction correct over *valid* (non-aborted) trials.

        Notes:
            - The three session types are mutually exclusive, so they live in a
                single categorical column rather than three booleans (which could
                silently disagree).
            - This is session-level only. It cannot express trial-level opto
                masks (opto-on vs opto-off *within* a session) — use
                ``behav_utils.data.ops.filtering.opto_mask`` for that.
            - 'accuracy' reuses ``SessionData.summary()['perf']`` so there is a
                single definition of session accuracy across the library.
        """
        from behav_utils.data.arrays import TrialArrays
        from behav_utils.stats import compute_stats, list_stats

        rows = []
        for sess in self.sessions:
            summ = sess.summary()
            stype = getattr(sess, 'session_type', None)
            if not stype:
                # Legacy fallback (session predates stored session_type).
                if sess.trials.opto_on.size > 0 and bool(np.any(sess.trials.opto_on)):
                    stype = 'opto'
                else:
                    stype = 'regular'

            sess_stats = compute_stats(TrialArrays.from_pooled(sess.get_arrays()), list_stats())
            dict_to_append = {'session_idx':  summ['session_idx'],
                            'session_id':   summ['session_id'],
                            'date':         summ['date'],
                            'stage':        summ['stage'],
                            'distribution': summ['distribution'],
                            'n_trials':     summ['n_trials'],
                            'n_valid':      summ['n_valid'],
                            'session_type': stype}
            dict_to_append.update(sess_stats.to_dict())
            rows.append(dict_to_append)
        return pd.DataFrame(rows)

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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
        with open(path, 'rb') as f:
            return pickle.load(f)
