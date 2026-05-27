"""
Core Data Structures

Hierarchical containers for behavioural data:

    ExperimentData          All animals in one project
    └── AnimalData          One animal, chronologically ordered sessions
        └── SessionData     One session
            ├── SessionMetadata   Constant within session (stage, contingency, ...)
            └── TrialData         Per-trial arrays (stimulus, choice, correct, ...)

    FittingData             Flat per-session arrays for SBI inference

Convention — three levels per domain:
    Low-level:     fit_psychometric(stim, ch)       — raw arrays
    Session-level: compute_psychometric(sessions)   — pre-filtered sessions → result dict
    Plotting:      plot_psychometric(result)         — result dict → axes

Plot methods on data classes are thin wrappers that call compute_ then plot_.

Usage:
    from behav_utils import (
        load_experiment, select_sessions, filter_trials,
        compute_psychometric, compute_um, plot_psychometric, plot_um, PALETTE,
    )

    experiment = load_experiment('config.yaml')
    animal = experiment.get_animal('SS05')
    sessions = select_sessions(animal, preset='expert_uniform')
    clean = filter_trials(sessions)

    psych = compute_psychometric(clean, mode='pooled', n_bootstrap=200)
    fig, ax = plt.subplots()
    plot_psychometric(psych, ax=ax, color=PALETTE[0])
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Optional, Dict, List, Tuple, Union, Any, Callable, TYPE_CHECKING,
)
from datetime import date

if TYPE_CHECKING:
    from behav_utils.config.schema import ProjectConfig


_STAT_PARENT = {
    'mu': 'psychometric',
    'sigma': 'psychometric',
    'lapse_low': 'psychometric',
    'lapse_high': 'psychometric',
    'win_stay_rate': 'win_stay',
    'lose_shift_rate': 'lose_shift',
    'w_stimulus': 'logistic_history',
    'w_prev_choice_1': 'logistic_history',
    'w_prev_choice_2': 'logistic_history',
    'w_prev_choice_3': 'logistic_history',
    'psychometric_gof': 'psychometric',
}


def _flatten_stats_dict(stats_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested stat dicts into a single-level dict.

    {'psychometric': {'mu': 0.01, 'sigma': 0.3}} → {'mu': 0.01, 'sigma': 0.3}
    {'accuracy': 0.85} → {'accuracy': 0.85}
    Arrays are left as-is (e.g. update_matrix).
    """
    flat = {}
    for key, value in stats_dict.items():
        if isinstance(value, dict):
            for k, v in value.items():
                flat[k] = v
        else:
            flat[key] = value
    return flat

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

    def __post_init__(self):
        """Set defaults for empty optional arrays."""
        n = len(self.stimulus)
        if len(self.reaction_time) == 0:
            self.reaction_time = np.full(n, np.nan)
        if len(self.abort) == 0:
            self.abort = np.zeros(n, dtype=bool)
        if len(self.opto_on) == 0:
            self.opto_on = np.zeros(n, dtype=bool)

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

    # ── Field access ────────────────────────────────────────────────────────

    def get_field(self, name: str) -> Optional[np.ndarray]:
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
        masking:     Whether this is a masking (light-only control) session
        csv_path:    Path to source CSV file
        filter_info: Metadata about filtering applied (None if unfiltered)

    Filtering:
        session.filter()                              # standard: exclude abort + opto
        session.filter(session.trials.opto_mask(0))   # custom: opto trials only
        filtered_session.stats(['accuracy'])           # stats on filtered trials
    """
    session_id: str
    session_idx: int
    date: date
    metadata: SessionMetadata
    trials: TrialData
    masking: bool = False

    csv_path: Optional[str] = None

    # Filter provenance (None = unfiltered raw data)
    filter_info: Optional[Dict[str, Any]] = field(default=None, repr=False)

    # Set by AnimalData after construction
    _days_since_first: Optional[float] = field(default=None, repr=False)

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
    def days_since_first(self) -> Optional[float]:
        return self._days_since_first

    @property
    def is_filtered(self) -> bool:
        """Whether this session has been through filter()."""
        return self.filter_info is not None

    # ── Array extraction ───────────────────────────────────────────────────

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
        from behav_utils.data.filtering import get_arrays
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
# FITTING DATA (bridge between AnimalData and SBI inference)
# =============================================================================

@dataclass
class FittingData:
    """
    Structured container for SBI model fitting.

    Provides per-session trial arrays with a consistent interface expected
    by SBIFitter and the ``build_simulator`` function.  Created via
    ``AnimalData.get_fitting_data()``.

    Attributes:
        animal_id: Identifier string.
        session_ids: Per-session identifiers.
        session_dates: Per-session dates.
        session_indices: Ordinal session indices (int array).
        stimuli: List of 1-D stimulus arrays (one per session).
        categories: List of 1-D category arrays.
        choices: List of 1-D choice arrays (NaN = no response).
        no_response: List of boolean masks (True = no response).
        not_blockstart: List of boolean masks (True = not first trial).
        n_sessions: Number of sessions.
        trials_per_session: Array of trial counts.
        time_axis: Alias for ``session_indices`` (used by SBIFitter).
    """
    animal_id: str
    session_ids: List[str]
    session_dates: List[Any]
    session_indices: np.ndarray
    stimuli: List[np.ndarray]
    categories: List[np.ndarray]
    choices: List[np.ndarray]
    no_response: List[np.ndarray]
    not_blockstart: List[np.ndarray]
    n_sessions: int
    trials_per_session: np.ndarray

    @property
    def time_axis(self) -> np.ndarray:
        """Session indices as a float array (for trajectory plotting)."""
        return self.session_indices.astype(float)

    def get_session(self, idx: int) -> Dict[str, np.ndarray]:
        """Return a single session's arrays as a dict."""
        return {
            'stimuli': self.stimuli[idx],
            'categories': self.categories[idx],
            'choices': self.choices[idx],
            'no_response': self.no_response[idx],
            'not_blockstart': self.not_blockstart[idx],
        }

    def pool(self) -> Dict[str, np.ndarray]:
        """
        Concatenate all sessions into single arrays.

        Only valid (responded) trials are included.

        Returns:
            Dict with 'stimuli', 'categories', 'choices' (1-D each).
        """
        all_stim, all_cat, all_choice = [], [], []
        for i in range(self.n_sessions):
            valid = ~self.no_response[i]
            all_stim.append(self.stimuli[i][valid])
            all_cat.append(self.categories[i][valid])
            all_choice.append(self.choices[i][valid])
        return {
            'stimuli': np.concatenate(all_stim),
            'categories': np.concatenate(all_cat),
            'choices': np.concatenate(all_choice),
        }
    @classmethod
    def from_arrays(
        cls,
        session_arrays: list,
        animal_id: str = 'synthetic',
    ) -> 'FittingData':
        """
        Create FittingData from a list of session dicts.

        Each dict must have 'stimuli', 'choices', 'categories' keys
        (1D numpy arrays). Optionally 'no_response' and 'not_blockstart'.

        Usage:
            sessions = [
                {'stimuli': stim0, 'choices': ch0, 'categories': cat0},
                {'stimuli': stim1, 'choices': ch1, 'categories': cat1},
                ...
            ]
            fd = FittingData.from_arrays(sessions, animal_id='synth_BE_00')
        """
        n = len(session_arrays)

        stim_list, cat_list, ch_list = [], [], []
        no_resp_list, nbs_list = [], []

        for s in session_arrays:
            stim = np.asarray(s['stimuli'])
            cat = np.asarray(s['categories'])
            ch = np.asarray(s['choices'], dtype=float)

            stim_list.append(stim)
            cat_list.append(cat)
            ch_list.append(ch)

            if 'no_response' in s:
                no_resp_list.append(np.asarray(s['no_response'], dtype=bool))
            else:
                no_resp_list.append(np.isnan(ch))

            if 'not_blockstart' in s:
                nbs_list.append(np.asarray(s['not_blockstart'], dtype=bool))
            else:
                nbs = np.ones(len(stim), dtype=bool)
                if len(stim) > 0:
                    nbs[0] = False
                nbs_list.append(nbs)

        return cls(
            animal_id=animal_id,
            session_ids=[f'sess_{i:03d}' for i in range(n)],
            session_dates=[None] * n,
            session_indices=np.arange(n, dtype=int),
            stimuli=stim_list,
            categories=cat_list,
            choices=ch_list,
            no_response=no_resp_list,
            not_blockstart=nbs_list,
            n_sessions=n,
            trials_per_session=np.array([len(s['stimuli']) for s in session_arrays]),
        )



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
    _config: Optional[Any] = field(default=None, repr=False)

    # Cache
    _feature_matrix_cache: Optional[pd.DataFrame] = field(
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
    def genotype(self) -> str:
        return self.metadata.get('genotype', 'unknown')

    @property
    def stages(self) -> List[str]:
        return list(dict.fromkeys(s.stage for s in self.sessions))

    # ── Filtering ───────────────────────────────────────────────────────────

    def get_sessions(
        self,
        stage: Optional[Union[str, List[str]]] = None,
        distribution: Optional[str] = None,
        idx_range: Optional[Tuple[int, int]] = None,
        date_range: Optional[Tuple[date, date]] = None,
        return_indices: bool = False,
    ) -> Union[List[SessionData], Tuple[List[SessionData], List[int]]]:
        """
        Filter sessions by criteria.

        Args:
            stage: Stage filter. A single string matches exactly;
                a list matches any (OR logic). None = no filter.
            distribution: Distribution filter (exact match)
            idx_range: (start, end) session index range (inclusive)
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

        if stage is not None:
            if isinstance(stage, list):
                stage_set = set(stage)
                sessions = [s for s in sessions if s.stage in stage_set]
            else:
                sessions = [s for s in sessions if s.stage == stage]
        if distribution is not None:
            sessions = [s for s in sessions if s.distribution == distribution]
        if idx_range is not None:
            sessions = [s for s in sessions
                        if idx_range[0] <= s.session_idx <= idx_range[1]]
        if date_range is not None:
            sessions = [s for s in sessions
                        if date_range[0] <= s.date <= date_range[1]]

        if return_indices:
            return sessions, [s.session_idx for s in sessions]
        return sessions

    def _resolve_sessions(
        self,
        sessions: Union[str, List[int]],
        stage: Optional[str] = None,
    ) -> List[SessionData]:
        """
        Resolve session selector to list of SessionData.

        Args:
            sessions: Selector string or list of indices.
                'all'      — all sessions
                'last_N'   — last N sessions (e.g. 'last_5')
                'first_N'  — first N sessions
                [0, 5, 10] — specific session indices
            stage: Filter to this training stage first.

        Returns:
            List of SessionData in chronological order.
        """
        pool = self.get_sessions(stage=stage) if stage else self.sessions

        if isinstance(sessions, str):
            if sessions == 'all':
                return pool
            elif sessions.startswith('last_'):
                n = int(sessions.split('_')[1])
                return pool[-n:]
            elif sessions.startswith('first_'):
                n = int(sessions.split('_')[1])
                return pool[:n]
            else:
                raise ValueError(f"Unknown session selector: '{sessions}'")
        elif isinstance(sessions, list):
            return [pool[i] for i in sessions if i < len(pool)]
        else:
            raise TypeError(f"sessions must be str or list, got {type(sessions)}")

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
    config: Optional[Any] = field(default=None, repr=False)  # ProjectConfig

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

    # ── Filtering ───────────────────────────────────────────────────────────

    def get_animals(
        self,
        min_sessions: int = 1,
        stage: Optional[Union[str, List[str]]] = None,
        animal_ids: Optional[List[str]] = None,
    ) -> List[AnimalData]:
        """
        Filter animals by criteria.

        Args:
            min_sessions: Minimum sessions (of given stage) required
            stage: Only count sessions of this stage
            animal_ids: Restrict to these animals

        Returns:
            List of qualifying AnimalData
        """
        result = []
        for animal in self.animals.values():
            if animal_ids is not None and animal.animal_id not in animal_ids:
                continue
            if stage is not None:
                n = len(animal.get_sessions(stage=stage))
            else:
                n = animal.n_sessions
            if n >= min_sessions:
                result.append(animal)
        return result

    def get_sessions(
        self,
        stage: Optional[Union[str, List[str]]] = None,
        min_sessions_per_animal: int = 1,
        return_indices: bool = False,
        **kwargs,
    ) -> Union[List[SessionData], Tuple[List[SessionData], List[int]]]:
        """
        Get all sessions matching criteria across all animals.

        Args:
            stage: Stage filter (str or list)
            min_sessions_per_animal: Minimum sessions per animal
            return_indices: If True, also return session_idx values.
            **kwargs: Passed to AnimalData.get_sessions (e.g. distribution)

        Returns:
            If return_indices=False: List of SessionData
            If return_indices=True: (sessions, indices)
        """
        animals = self.get_animals(
            min_sessions=min_sessions_per_animal, stage=stage,
        )
        sessions = []
        for animal in animals:
            sessions.extend(animal.get_sessions(stage=stage, **kwargs))

        if return_indices:
            return sessions, [s.session_idx for s in sessions]
        return sessions

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

    def _resolve_animals(self, animals='all', min_sessions=5, stage=None):
        """Resolve animal selector to list of AnimalData."""
        if animals == 'all':
            return self.get_animals(min_sessions=min_sessions, stage=stage)
        return [self.get_animal(aid) for aid in animals]
        
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
