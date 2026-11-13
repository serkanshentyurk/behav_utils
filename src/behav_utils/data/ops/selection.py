"""
Session Selection

Composable, preset-driven session filtering for all downstream consumers.

Design:
    SessionFilter is an immutable dataclass describing selection criteria.
    Named presets come from the config's ``session_presets`` (looked up on the
    animal's config first, then in the process-wide registry) or from
    ``register_preset``. All analysis code calls select_sessions() — never
    rolls its own filtering.

Filtering order (each step narrows the previous):
    1. Metadata:      stage, distribution
    2. Session type:  session_type (positive selection), or else the exclusions
                      exclude_opto / exclude_types (negative selection)
    3. Index range:   after_session_idx, before_session_idx, session_indices
    4. Positional:    last_fraction, first_n, last_n
    5. Quality:       min_accuracy, max_accuracy, min_trials
    6. Positional, after quality: last_n_qualifying
    7. Custom:        custom_filter callable

Which sessions count is settled (step 2) before any position is taken (steps 4 and 6), so
``last_n=5`` with ``exclude_types=('masking',)`` gives the last five sessions that are not masking.
Up to 0.8.x the exclusions ran after the positional steps: excluded sessions used up positions
and fewer than N could remain.

Usage:
    from behav_utils import select_sessions, register_preset, SessionFilter

    register_preset('expert', SessionFilter(stage='task', min_accuracy=0.70, last_fraction=0.50))

    sessions = select_sessions(animal, 'expert')                       # preset
    sessions = select_sessions(animal, 'expert', min_accuracy=0.80)    # preset + override
    sessions = select_sessions(animal, stage='task', last_n=5)         # ad hoc
"""

import warnings
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Union,
)

import numpy as np

if TYPE_CHECKING:
    from behav_utils.data.structures import AnimalData, SessionData


# =============================================================================
# SESSION FILTER
# =============================================================================

@dataclass(frozen=True)
class SessionFilter:
    """
    Immutable session selection criteria.

    All fields are optional — None means "no constraint".
    Filters are applied in the order listed in the module docstring.

    Attributes:
        stage: Task stage (exact match, or list for OR-logic)
        distribution: Stimulus distribution (exact match, or list for OR-logic)
        session_type: Session type — 'regular', 'opto', or any project type stamped
                      from config.session_types (e.g. 'sham').
                      Accepts a string or list. When set, exclude_opto and
                      exclude_types are ignored (the caller is explicitly
                      choosing which types to include).
        min_accuracy: Minimum session accuracy (fraction correct)
        max_accuracy: Maximum session accuracy
        last_fraction: Keep only the last X fraction of sessions
                       (applied BEFORE accuracy filtering)
        first_n: Keep only the first N sessions (after the metadata, session-type
                 and exclusion filters)
        last_n: Keep only the last N sessions (likewise)
        last_n_qualifying: Keep only the last N sessions that pass the quality
                      filters (applied AFTER min_accuracy / max_accuracy /
                      min_trials, unlike last_n which is applied before them)
        after_session_idx: Only sessions with session_idx > this value
        before_session_idx: Only sessions with session_idx < this value
        session_indices: Explicit list of session_idx values to include
        min_trials: Minimum valid (non-abort, responded) trials
        exclude_opto: If True, exclude sessions with opto trials
                      (ignored when session_type is set). Applied before the
                      positional steps.
        exclude_types: Session types to drop when session_type is not set,
                      e.g. ('sham', 'washout'). The names are the project's;
                      the library derives only 'regular' and 'opto' from the data.
                      Applied before the positional steps.
        custom_filter: Callable(SessionData) -> bool for arbitrary filtering
    """
    stage: Union[str, List[str]] | None = None
    distribution: Union[str, List[str]] | None = None
    session_type: Union[str, List[str]] | None = None
    min_accuracy: float | None = None
    max_accuracy: float | None = None
    last_fraction: float | None = None
    first_n: int | None = None
    last_n: int | None = None
    last_n_qualifying: int | None = None
    after_session_idx: int | None = None
    before_session_idx: int | None = None
    session_indices: List[int] | None = None
    min_trials: int = 10
    exclude_opto: bool = False
    exclude_types: Tuple[str, ...] = ()      # session types to drop (skipped when session_type is set)
    custom_filter: Callable | None = field(default=None, hash=False)

    @staticmethod
    def _resolve_session_type(sess: 'SessionData') -> str:
        """Return the session's stored session_type.

        Falls back to deriving from opto content only for legacy sessions
        that predate the stored field. Mirrors AnimalData.session_table.
        """
        stype = getattr(sess, 'session_type', None)
        if stype:
            return stype
        if sess.trials.opto_on.size > 0 and bool(np.any(sess.trials.opto_on)):
            return 'opto'
        return 'regular'

    def apply(self, animal: 'AnimalData') -> List['SessionData']:
        """
        Apply this filter to an animal's sessions.

        Returns:
            List of matching SessionData, in chronological order (may be empty).
        """
        # ── 1. Metadata ───────────────────────────────────────────────────
        sessions = animal.get_sessions(
            stage=self.stage,
            distribution=self.distribution,
        )

        # ── 2. Session type, or the exclusions ────────────────────────────
        # session_type is a positive selection (exclude_opto / exclude_types are
        # then ignored); without it the exclusions drop sessions. Either way this
        # happens before any positional step, so excluded sessions use up no positions.
        if self.session_type is not None:
            if isinstance(self.session_type, str):
                sessions = [
                    s for s in sessions
                    if self._resolve_session_type(s) == self.session_type
                ]
            else:
                type_set = set(self.session_type)
                sessions = [
                    s for s in sessions
                    if self._resolve_session_type(s) in type_set
                ]
        else:
            if self.exclude_opto:
                sessions = [s for s in sessions if not np.any(s.trials.opto_on)]
            if self.exclude_types:
                drop = set(self.exclude_types)
                sessions = [s for s in sessions if self._resolve_session_type(s) not in drop]

        # ── 3. Index range ────────────────────────────────────────────────
        if self.after_session_idx is not None:
            sessions = [
                s for s in sessions
                if s.session_idx > self.after_session_idx
            ]
        if self.before_session_idx is not None:
            sessions = [
                s for s in sessions
                if s.session_idx < self.before_session_idx
            ]
        if self.session_indices is not None:
            idx_set = set(self.session_indices)
            sessions = [s for s in sessions if s.session_idx in idx_set]

        # ── 4. Positional ─────────────────────────────────────────────────
        if self.last_fraction is not None and len(sessions) > 0:
            n_total = len(sessions)
            start_idx = int(n_total * (1.0 - self.last_fraction))
            sessions = sessions[start_idx:]

        if self.first_n is not None:
            sessions = sessions[:self.first_n]

        if self.last_n is not None:
            sessions = sessions[-self.last_n:]

        # ── 5. Quality ────────────────────────────────────────────────────
        if self.min_accuracy is not None or self.max_accuracy is not None:
            filtered = []
            for s in sessions:
                acc = s.summary().accuracy
                if self.min_accuracy is not None and acc < self.min_accuracy:
                    continue
                if self.max_accuracy is not None and acc > self.max_accuracy:
                    continue
                filtered.append(s)
            sessions = filtered

        if self.min_trials > 0:
            sessions = [
                s for s in sessions
                if s.trials.valid_mask.sum() >= self.min_trials
            ]

        # ── 6. Positional, after quality ──────────────────────────────────
        if self.last_n_qualifying is not None:
            sessions = sessions[-self.last_n_qualifying:]

        # ── 7. Custom ─────────────────────────────────────────────────────
        if self.custom_filter is not None:
            sessions = [s for s in sessions if self.custom_filter(s)]

        return sessions

    def with_overrides(self, **kwargs) -> 'SessionFilter':
        """
        Return a new SessionFilter with specified fields replaced: ``base.with_overrides(min_accuracy=0.80)``
        is ``base`` with a stricter accuracy floor.
        """
        return replace(self, **kwargs)

    def describe(self) -> str:
        """Human-readable summary of active constraints."""
        parts = []
        if self.stage is not None:
            parts.append(f"stage={self.stage}")
        if self.distribution is not None:
            parts.append(f"distribution={self.distribution}")
        if self.session_type is not None:
            parts.append(f"session_type={self.session_type}")
        if self.last_fraction is not None:
            parts.append(f"last {self.last_fraction:.0%}")
        if self.first_n is not None:
            parts.append(f"first {self.first_n}")
        if self.last_n is not None:
            parts.append(f"last {self.last_n}")
        if self.min_accuracy is not None:
            parts.append(f"acc≥{self.min_accuracy:.0%}")
        if self.max_accuracy is not None:
            parts.append(f"acc≤{self.max_accuracy:.0%}")
        if self.after_session_idx is not None:
            parts.append(f"after idx {self.after_session_idx}")
        if self.before_session_idx is not None:
            parts.append(f"before idx {self.before_session_idx}")
        if self.min_trials > 0:
            parts.append(f"≥{self.min_trials} trials")
        if self.last_n_qualifying is not None:
            parts.append(f"last {self.last_n_qualifying} qualifying")
        if self.session_type is None:
            if self.exclude_opto:
                parts.append("no opto")
            if self.exclude_types:
                parts.append('no ' + '/'.join(self.exclude_types))
        if self.custom_filter is not None:
            parts.append("+ custom filter")
        return ', '.join(parts) if parts else '(no constraints)'


# =============================================================================
# PRESET REGISTRY
# =============================================================================

_PRESETS: Dict[str, SessionFilter] = {}


def register_preset(name: str, filt: SessionFilter) -> None:
    """
    Register a named session filter preset.

    Args:
        name: Preset key (e.g. 'expert')
        filt: SessionFilter instance

    Raises:
        TypeError: If filt is not a SessionFilter
    """
    if not isinstance(filt, SessionFilter):
        raise TypeError(
            f"Expected SessionFilter, got {type(filt).__name__}"
        )
    _PRESETS[name] = filt


def get_preset(name: str, config: Any | None = None) -> SessionFilter:
    """
    Retrieve a preset by name: from ``config.session_presets`` if a config is
    given and defines it, else from the process-wide registry.

    Raises:
        KeyError: If preset not found in either place.
    """
    spec = (getattr(config, 'session_presets', None) or {}).get(name) if config is not None else None
    if isinstance(spec, dict):
        return preset_from_spec(name, spec)
    if name not in _PRESETS:
        available = ', '.join(sorted(_PRESETS.keys())) or '(none)'
        raise KeyError(
            f"Unknown session preset '{name}'. Available: {available}"
        )
    return _PRESETS[name]


def preset_from_spec(name: str, spec: Dict[str, Any]) -> SessionFilter:
    """Build a SessionFilter from a raw ``session_presets`` entry (YAML dict).

    Unknown fields are ignored with a warning; ``exclude_types`` lists become tuples.
    """
    spec = dict(spec)
    if isinstance(spec.get('exclude_types'), list):
        spec['exclude_types'] = tuple(spec['exclude_types'])
    valid_fields = {f.name for f in SessionFilter.__dataclass_fields__.values()}
    unknown = set(spec) - valid_fields
    if unknown:
        warnings.warn(f"Preset '{name}': ignoring unknown fields {sorted(unknown)}", stacklevel=2)
    return SessionFilter(**{k: v for k, v in spec.items() if k in valid_fields})


def list_presets() -> Dict[str, str]:
    """Return {name: description} for all registered presets."""
    return {name: filt.describe() for name, filt in sorted(_PRESETS.items())}



def select_sessions(
    animal: 'AnimalData',
    preset: str | None = None,
    **overrides,
) -> List['SessionData']:
    """
    Select sessions from an animal using a preset and/or keyword criteria.

    Three ways to call it: a preset alone (``select_sessions(animal, 'expert')``), a preset with fields
    overridden (``select_sessions(animal, 'expert', min_accuracy=0.80)``), or fields alone
    (``select_sessions(animal, stage='task', last_n=5)``).

    A preset is resolved on the animal's own config (``session_presets`` of the
    experiment it was loaded with) before the process-wide registry, so two
    experiments loaded in one session never see each other's presets.

    Args:
        animal: AnimalData object
        preset: Name of a preset (optional)
        **overrides: Any SessionFilter field to set or override

    Returns:
        List of SessionData matching the criteria (may be empty)
    """
    if preset is not None:
        base = get_preset(preset, config=getattr(animal, '_config', None))
        filt = base.with_overrides(**overrides) if overrides else base
    else:
        filt = SessionFilter(**overrides)

    return filt.apply(animal)


# =============================================================================
# CONFIG-DRIVEN PRESET LOADING
# =============================================================================

def register_presets_from_config(config_raw: Dict[str, Any]) -> int:
    """
    Register the ``session_presets`` of a parsed YAML config dict in the
    process-wide registry (``load_experiment`` does this; the config copy on each
    animal takes priority in ``select_sessions``).

    Expected format::

        session_presets:
          expert:
            stage: "task"
            min_accuracy: 0.70
            last_fraction: 0.50

    Returns:
        Number of presets registered
    """
    presets_raw = config_raw.get('session_presets', {})
    if not presets_raw:
        return 0

    count = 0
    for name, spec in presets_raw.items():
        if name.startswith('_'):        # YAML anchors / shared lists, not presets
            continue
        if not isinstance(spec, dict):
            warnings.warn(f"Skipping preset '{name}': expected dict, got {type(spec)}", stacklevel=2)
            continue
        try:
            register_preset(name, preset_from_spec(name, spec))
            count += 1
        except Exception as e:
            warnings.warn(f"Failed to register preset '{name}': {e}", stacklevel=2)

    return count
