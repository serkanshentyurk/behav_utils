"""
Core Data Structures

Hierarchical containers for behavioural data:

    ExperimentData
    └── AnimalData
        └── SessionData
            ├── SessionMetadata
            └── TrialData

Design principles:
    - Internal field names are standardised (stimulus, choice, outcome, etc.)
    - Config maps CSV columns to these standard names at load time
    - Each level has stats(), plot methods, and filtering
    - Plot methods are thin wrappers around standalone plotting functions
    - Always return (fig, ax) from plot methods

Usage:
    from behav_utils import load_experiment

    experiment = load_experiment('config.yaml')
    animal = experiment.get_animal('SS05')
    session = animal.sessions[10]

    # Stats at any level
    session.stats(['accuracy', 'recency'])
    animal.stat_trajectory('accuracy')

    # Plotting at any level
    session.plot_psychometric()
    animal.plot_psychometric(sessions='last_5', mode='overlay')
    experiment.plot_trajectory(stat='accuracy', combine='mean_sem')
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Optional, Dict, List, Tuple, Union, Any, Callable, TYPE_CHECKING,
)
from datetime import date

if TYPE_CHECKING:
    from behav_utils.config.schema import ProjectConfig


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

    Required arrays (always present):
        stimulus, choice, outcome, correct, trial_number

    Optional arrays (present if in config, NaN/default otherwise):
        reaction_time, abort, opto_on, distribution, ...

    Extra arrays (unmapped CSV columns):
        extra: Dict[str, np.ndarray]

    Internal encoding:
        stimulus: float (raw values from CSV)
        choice: float (raw values from CSV — preprocessing converts to
                 category space if needed)
        category: int (0 or 1, derived from stimulus + boundary)
        correct: bool
        abort: bool
    """
    # ── Required ────────────────────────────────────────────────────────────
    trial_number: np.ndarray
    stimulus: np.ndarray
    choice: np.ndarray          # category space (0=A, 1=B, NaN=no response)
    outcome: np.ndarray
    correct: np.ndarray
    category: np.ndarray        # derived from stimulus + boundary

    # Raw choice preserved for reference
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

    @property
    def n_trials(self) -> int:
        return len(self.stimulus)

    @property
    def no_response(self) -> np.ndarray:
        """Boolean mask: True where choice is NaN."""
        return np.isnan(self.choice.astype(float))

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask: non-abort, responded trials."""
        return ~self.abort & ~self.no_response

    def get_field(self, name: str) -> Optional[np.ndarray]:
        """Get any field by name — checks core, optional, then extra."""
        if hasattr(self, name) and name not in ('optional_fields', 'extra'):
            val = getattr(self, name)
            if isinstance(val, np.ndarray):
                return val
        if name in self.optional_fields:
            return self.optional_fields[name]
        if name in self.extra:
            return self.extra[name]
        return None

    def get_arrays(
        self,
        exclude_abort: bool = True,
        exclude_opto: bool = True,
        exclude_no_response: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Extract filtered arrays for analysis/modelling.

        Returns dict with: 'stimuli', 'categories', 'choices',
        'no_response', 'reaction_times', 'trial_indices'
        """
        mask = np.ones(self.n_trials, dtype=bool)

        if exclude_abort:
            mask &= ~self.abort
        if exclude_opto:
            mask &= ~self.opto_on
        if exclude_no_response:
            mask &= ~self.no_response

        choices = self.choice[mask].astype(float)

        return {
            'stimuli': self.stimulus[mask],
            'categories': self.category[mask],
            'choices': choices,
            'no_response': np.isnan(choices),
            'reaction_times': self.reaction_time[mask],
            'trial_indices': np.where(mask)[0],
        }

    # ── Stats ───────────────────────────────────────────────────────────────

    def stats(
        self,
        stat_names: Optional[List[str]] = None,
        exclude_abort: bool = True,
        exclude_opto: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute summary statistics for this session's trials.

        Args:
            stat_names: Which stats to compute (default: all registered)
            exclude_abort: Remove abort trials
            exclude_opto: Remove opto trials

        Returns:
            Dict of stat_name → value
        """
        from behav_utils.analysis.summary_stats import compute_summary_stats

        arrays = self.get_arrays(
            exclude_abort=exclude_abort,
            exclude_opto=exclude_opto,
        )
        valid = ~arrays['no_response']

        if valid.sum() < 5:
            warnings.warn(f"Only {valid.sum()} valid trials — stats may be unreliable")

        return compute_summary_stats(
            arrays['choices'], arrays['stimuli'], arrays['categories'],
            stat_names=stat_names,
            return_dict=True,
        )


# =============================================================================
# SESSION DATA
# =============================================================================

@dataclass
class SessionData:
    """
    All data for a single behavioural session.
    """
    session_id: str
    session_idx: int                # ordinal index within animal
    date: date
    metadata: SessionMetadata
    trials: TrialData

    # Source
    csv_path: Optional[str] = None

    # Set by AnimalData after construction
    _days_since_first: Optional[float] = field(default=None, repr=False)

    @property
    def n_trials(self) -> int:
        return self.trials.n_trials

    @property
    def stage(self) -> str:
        return self.metadata.stage

    @property
    def distribution(self) -> str:
        dist = self.trials.get_field('distribution')
        if dist is not None and len(dist) > 0:
            # Most common value (should be constant within session)
            vals, counts = np.unique(dist[dist != ''], return_counts=True)
            if len(vals) > 0:
                return str(vals[counts.argmax()])
        return self.metadata.get('distribution', 'Unknown')

    @property
    def days_since_first(self) -> Optional[float]:
        return self._days_since_first

    # ── Stats ───────────────────────────────────────────────────────────────

    def stats(
        self,
        stat_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute summary stats. Delegates to TrialData.stats()."""
        return self.trials.stats(stat_names=stat_names, **kwargs)

    def summary(self) -> Dict[str, Any]:
        """Quick summary dict."""
        valid = self.trials.valid_mask
        n_valid = valid.sum()
        choices = self.trials.choice[valid].astype(float)
        cats = self.trials.category[valid]
        return {
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

    # ── Plotting ────────────────────────────────────────────────────────────

    def plot_psychometric(self, ax=None, **kwargs):
        """
        Plot psychometric curve for this session.

        Returns:
            (fig, ax)
        """
        from behav_utils.plotting.psychometric import plot_psychometric

        arrays = self.trials.get_arrays()
        valid = ~arrays['no_response']
        return plot_psychometric(
            arrays['stimuli'][valid],
            arrays['choices'][valid],
            ax=ax,
            title=f'{self.session_id}',
            **kwargs,
        )

    def plot_trials(self, **kwargs):
        """
        Plot trial-by-trial raster for this session.

        Returns:
            (fig, ax)
        """
        from behav_utils.plotting.session import plot_session_trials
        return plot_session_trials(self, **kwargs)


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
    def stages(self) -> List[str]:
        return list(dict.fromkeys(s.stage for s in self.sessions))

    # ── Filtering ───────────────────────────────────────────────────────────

    def get_sessions(
        self,
        stage: Optional[str] = None,
        distribution: Optional[str] = None,
        idx_range: Optional[Tuple[int, int]] = None,
        date_range: Optional[Tuple[date, date]] = None,
    ) -> List[SessionData]:
        """Filter sessions by criteria."""
        sessions = self.sessions

        if stage is not None:
            sessions = [s for s in sessions if s.stage == stage]
        if distribution is not None:
            sessions = [s for s in sessions if s.distribution == distribution]
        if idx_range is not None:
            sessions = [s for s in sessions
                        if idx_range[0] <= s.session_idx <= idx_range[1]]
        if date_range is not None:
            sessions = [s for s in sessions
                        if date_range[0] <= s.date <= date_range[1]]
        return sessions

    def get_trial_data(
        self,
        fields: Optional[List[str]] = None,
        stage: Optional[str] = None,
        exclude_abort: bool = True,
        exclude_opto: bool = True,
        min_valid_trials: int = 10,
    ) -> Dict[str, Any]:
        """
        Extract filtered trial arrays across sessions.

        Returns a dict with per-session arrays, session metadata,
        and a time axis. This is the general-purpose bridge between
        data storage and analysis/modelling pipelines.

        Args:
            fields: Which trial fields to include.
                    Default: ['stimuli', 'categories', 'choices']
                    Can also request: 'reaction_times', 'no_response',
                    'trial_indices', or any field in TrialData.optional_fields
            stage: Filter to this training stage
            exclude_abort: Remove abort trials
            exclude_opto: Remove opto trials
            min_valid_trials: Skip sessions with fewer valid trials

        Returns:
            Dict with:
                'session_arrays': List[Dict[str, np.ndarray]] — one per session
                'session_ids': List[str]
                'session_dates': List[date]
                'session_indices': np.ndarray — ordinal indices
                'n_sessions': int
                'trials_per_session': np.ndarray
                'animal_id': str

        Example:
            data = animal.get_trial_data(
                fields=['stimuli', 'choices', 'categories', 'reaction_times'],
                stage='Full_Task_Cont',
            )
            for i, sa in enumerate(data['session_arrays']):
                print(f"Session {i}: {len(sa['stimuli'])} trials")
        """
        if fields is None:
            fields = ['stimuli', 'categories', 'choices']

        sessions = self.get_sessions(stage=stage) if stage else self.sessions

        session_arrays = []
        session_ids = []
        session_dates = []
        session_indices = []

        for sess in sessions:
            # Get base arrays (always computed)
            arrays = sess.trials.get_arrays(
                exclude_abort=exclude_abort,
                exclude_opto=exclude_opto,
            )

            # Check minimum trials
            n_valid = (~arrays['no_response']).sum()
            if n_valid < min_valid_trials:
                continue

            # Build output dict with requested fields
            out = {}
            for f in fields:
                if f in arrays:
                    out[f] = arrays[f]
                elif f == 'reaction_times' and 'reaction_times' in arrays:
                    out[f] = arrays[f]
                else:
                    # Try optional_fields and extra
                    trial_indices = arrays['trial_indices']
                    raw = sess.trials.get_field(f)
                    if raw is not None:
                        out[f] = raw[trial_indices]
                    else:
                        warnings.warn(
                            f"Field '{f}' not found in session {sess.session_id}"
                        )

            # Always include no_response for downstream filtering
            if 'no_response' not in out:
                out['no_response'] = arrays['no_response']

            session_arrays.append(out)
            session_ids.append(sess.session_id)
            session_dates.append(sess.date)
            session_indices.append(sess.session_idx)

        return {
            'session_arrays': session_arrays,
            'session_ids': session_ids,
            'session_dates': session_dates,
            'session_indices': np.array(session_indices, dtype=float),
            'n_sessions': len(session_arrays),
            'trials_per_session': np.array([
                len(sa['stimuli']) for sa in session_arrays
            ]) if session_arrays else np.array([]),
            'animal_id': self.animal_id,
        }


    # ── Stats ───────────────────────────────────────────────────────────────

    def feature_matrix(
        self,
        stage: Optional[str] = None,
        stat_names: Optional[List[str]] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Build session x feature DataFrame.

        Caches the result — call invalidate_cache() if sessions change.
        """
        from behav_utils.analysis.session_features import build_feature_matrix

        cache_key = (stage, tuple(stat_names) if stat_names else None)

        if use_cache and self._feature_matrix_cache is not None:
            return self._feature_matrix_cache

        df = build_feature_matrix(
            self, stage=stage, stat_names=stat_names, **kwargs,
        )
        if use_cache:
            self._feature_matrix_cache = df
        return df

    def stat_trajectory(
        self,
        stat_name: str,
        stage: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract one stat across sessions.

        Returns:
            (session_indices, values) — both np.ndarray
        """
        df = self.feature_matrix(stage=stage, **kwargs)
        if stat_name not in df.columns:
            raise ValueError(
                f"Stat '{stat_name}' not in feature matrix. "
                f"Available: {sorted(df.columns)}"
            )
        return (
            df['session_idx'].values,
            df[stat_name].values,
        )

    def expert_baseline(
        self,
        features: List[str],
        stage: Optional[str] = None,
        last_n: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Mean and std of last N sessions for each feature.

        Returns:
            {feature: {'mean': float, 'std': float}}
        """
        df = self.feature_matrix(stage=stage)
        if len(df) < last_n:
            last_n = len(df)

        tail = df.iloc[-last_n:]
        result = {}
        for feat in features:
            if feat in tail.columns:
                vals = tail[feat].dropna()
                result[feat] = {
                    'mean': float(vals.mean()) if len(vals) > 0 else np.nan,
                    'std': float(vals.std()) if len(vals) > 0 else np.nan,
                }
        return result

    # ── Plotting ────────────────────────────────────────────────────────────

    def plot_psychometric(
        self,
        sessions: Union[str, List[int]] = 'all',
        mode: str = 'overlay',
        stage: Optional[str] = None,
        ax=None,
        **kwargs,
    ):
        """
        Plot psychometric curves for selected sessions.

        Args:
            sessions: 'all', 'last_5', 'first_5', or list of indices
            mode: 'overlay', 'grid', 'pooled'
            stage: Filter to this stage

        Returns:
            (fig, ax) or (fig, axes) for grid mode
        """
        from behav_utils.plotting.psychometric import plot_session_psychometrics

        sess_list = self._resolve_sessions(sessions, stage)
        return plot_session_psychometrics(sess_list, mode=mode, ax=ax, **kwargs)

    def plot_trajectory(
        self,
        stat: str,
        stage: Optional[str] = None,
        ax=None,
        **kwargs,
    ):
        """
        Plot one stat across sessions.

        Returns:
            (fig, ax)
        """
        from behav_utils.plotting.trajectory import plot_stat_trajectory

        indices, values = self.stat_trajectory(stat, stage=stage)
        return plot_stat_trajectory(
            indices, values,
            title=f'{self.animal_id} — {stat}',
            ylabel=stat,
            ax=ax,
            **kwargs,
        )

    def _resolve_sessions(
        self,
        sessions: Union[str, List[int]],
        stage: Optional[str] = None,
    ) -> List[SessionData]:
        """Resolve session selector to list of SessionData."""
        pool = self.get_sessions(stage=stage) if stage else self.sessions

        if isinstance(sessions, str):
            if sessions == 'all':
                return pool
            elif sessions == 'last_5':
                return pool[-5:]
            elif sessions == 'first_5':
                return pool[:5]
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
        stage: Optional[str] = None,
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
        stage: Optional[str] = None,
        min_sessions_per_animal: int = 1,
        **kwargs,
    ) -> List[SessionData]:
        """Get all sessions matching criteria across all animals."""
        animals = self.get_animals(
            min_sessions=min_sessions_per_animal, stage=stage,
        )
        sessions = []
        for animal in animals:
            sessions.extend(animal.get_sessions(stage=stage, **kwargs))
        return sessions

    # ── Stats ───────────────────────────────────────────────────────────────

    def feature_matrix(
        self,
        stage: Optional[str] = None,
        min_sessions: int = 5,
        stat_names: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Build pooled session x feature DataFrame across all animals.
        """
        from behav_utils.analysis.session_features import build_feature_matrix

        animals = self.get_animals(min_sessions=min_sessions, stage=stage)
        dfs = []
        for animal in animals:
            df = animal.feature_matrix(stage=stage, stat_names=stat_names, **kwargs)
            if len(df) > 0:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

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

    # ── Query API: Plotting ─────────────────────────────────────────────────

    def plot_trajectory(
        self,
        stat: str,
        animals: Union[str, List[str]] = 'all',
        sessions: Optional[Dict[str, Any]] = None,
        combine: str = 'mean_sem',
        stage: Optional[str] = None,
        min_sessions: int = 5,
        ax=None,
        **kwargs,
    ):
        """
        Plot stat trajectory across sessions for multiple animals.

        Args:
            stat: Feature name to plot
            animals: 'all' or list of animal IDs
            sessions: Filter dict, e.g. {'stage': 'Full_Task_Cont'}
                      (applied via get_sessions)
            combine: 'mean_sem', 'median_iqr', 'individual', 'both'
            stage: Shorthand for sessions={'stage': ...}
            min_sessions: Minimum sessions per animal
            ax: Matplotlib axes

        Returns:
            (fig, ax)
        """
        from behav_utils.plotting.trajectory import plot_multi_animal_trajectory

        # Resolve animals
        if animals == 'all':
            animal_list = self.get_animals(
                min_sessions=min_sessions, stage=stage,
            )
        else:
            animal_list = [self.get_animal(aid) for aid in animals]

        return plot_multi_animal_trajectory(
            animal_list,
            stat=stat,
            stage=stage,
            combine=combine,
            ax=ax,
            **kwargs,
        )

    def plot_psychometric(
        self,
        animals: Union[str, List[str]] = 'all',
        sessions: str = 'last_5',
        mode: str = 'pooled',
        stage: Optional[str] = None,
        min_sessions: int = 5,
        ax=None,
        **kwargs,
    ):
        """
        Plot psychometric curves across animals.

        Args:
            animals: 'all' or list of animal IDs
            sessions: Session selector per animal
            mode: 'pooled', 'overlay', 'grid'
            stage: Stage filter

        Returns:
            (fig, ax) or (fig, axes) for grid
        """
        from behav_utils.plotting.psychometric import plot_session_psychometrics

        if animals == 'all':
            animal_list = self.get_animals(
                min_sessions=min_sessions, stage=stage,
            )
        else:
            animal_list = [self.get_animal(aid) for aid in animals]

        # Collect sessions from all selected animals
        all_sessions = []
        for animal in animal_list:
            selected = animal._resolve_sessions(sessions, stage=stage)
            all_sessions.extend(selected)

        return plot_session_psychometrics(
            all_sessions, mode=mode, ax=ax, **kwargs,
        )

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Don't pickle the config object — reload from YAML
        config_backup = self.config
        self.config = None
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self.config = config_backup

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentData':
        with open(path, 'rb') as f:
            return pickle.load(f)
