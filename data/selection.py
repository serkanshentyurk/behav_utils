"""
Session Selection

Composable, preset-driven session filtering for all downstream consumers.

Design:
    SessionFilter is an immutable dataclass describing selection criteria.
    Named presets can be registered and referenced by string key.
    All analysis code calls select_sessions() — never rolls its own filtering.

Filtering order (each step narrows the previous):
    1. Metadata:   stage, distribution
    2. Index range: after_session_idx, before_session_idx, session_indices
    3. Positional:  last_fraction, first_n, last_n
    4. Quality:     min_accuracy, max_accuracy, min_trials
    5. Opto:        exclude_opto
    6. Masking:     exclude_masking
    7. Custom:      custom_filter callable

Usage:
    from behav_utils.data.selection import select_sessions, register_preset, SessionFilter

    # Register presets (typically at startup or via config)
    register_preset('expert_uniform', SessionFilter(
        stage='Full_Task_Cont',
        distribution='Uniform',
        min_accuracy=0.70,
        last_fraction=0.50,
    ))

    # Use a preset
    sessions = select_sessions(animal, 'expert_uniform')

    # Preset with override
    sessions = select_sessions(animal, 'expert_uniform', min_accuracy=0.80)

    # Ad-hoc (no preset)
    sessions = select_sessions(animal, stage='Full_Task_Cont', last_n=5)

    # Convert to FittingData for SBI
    clean = filter_trials(sessions)
    fd = fitting_data_from_sessions(clean, animal.animal_id)
"""

import numpy as np
import warnings
from dataclasses import dataclass, field, replace
from typing import (
    Optional, List, Dict, Tuple, Callable, Union, Any, TYPE_CHECKING,
)

if TYPE_CHECKING:
    from behav_utils.data.structures import AnimalData, SessionData, FittingData


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
        distribution: Stimulus distribution (exact match)
        min_accuracy: Minimum session accuracy (fraction correct)
        max_accuracy: Maximum session accuracy
        last_fraction: Keep only the last X fraction of sessions
                       (applied BEFORE accuracy filtering)
        first_n: Keep only the first N sessions (after metadata filter)
        last_n: Keep only the last N sessions (after metadata filter)
        after_session_idx: Only sessions with session_idx > this value
        before_session_idx: Only sessions with session_idx < this value
        session_indices: Explicit list of session_idx values to include
        min_trials: Minimum valid (non-abort, responded) trials
        exclude_opto: If True, only include sessions with no opto trials
        custom_filter: Callable(SessionData) -> bool for arbitrary filtering
    """
    stage: Optional[Union[str, List[str]]] = None
    distribution: Optional[str] = None
    min_accuracy: Optional[float] = None
    max_accuracy: Optional[float] = None
    last_fraction: Optional[float] = None
    first_n: Optional[int] = None
    last_n: Optional[int] = None
    after_session_idx: Optional[int] = None
    before_session_idx: Optional[int] = None
    session_indices: Optional[List[int]] = None
    min_trials: int = 10
    exclude_opto: bool = False
    exclude_masking: bool = True
    custom_filter: Optional[Callable] = field(default=None, hash=False)

    def apply(self, animal: 'AnimalData') -> List['SessionData']:
        """
        Apply this filter to an animal's sessions.

        Returns:
            List of matching SessionData, in chronological order.

        Raises:
            ValueError: If no sessions survive filtering.
        """
        # ── 1. Metadata ───────────────────────────────────────────────────
        sessions = animal.get_sessions(
            stage=self.stage,
            distribution=self.distribution,
        )

        # ── 2. Index range ────────────────────────────────────────────────
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

        # ── 3. Positional ─────────────────────────────────────────────────
        if self.last_fraction is not None and len(sessions) > 0:
            n_total = len(sessions)
            start_idx = int(n_total * (1.0 - self.last_fraction))
            sessions = sessions[start_idx:]

        if self.first_n is not None:
            sessions = sessions[:self.first_n]

        if self.last_n is not None:
            sessions = sessions[-self.last_n:]

        # ── 4. Quality ────────────────────────────────────────────────────
        if self.min_accuracy is not None or self.max_accuracy is not None:
            filtered = []
            for s in sessions:
                acc = s.stats(['accuracy'])['accuracy']
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

        # ── 5. Opto ──────────────────────────────────────────────────────
        if self.exclude_opto:
            sessions = [
                s for s in sessions
                if not np.any(s.trials.opto_on)
            ]

        # ── 6. Masking ───────────────────────────────────────────────────
        if self.exclude_masking:
            sessions = [s for s in sessions if not getattr(s, 'masking', False)]


        # ── 7. Custom ─────────────────────────────────────────────────────
        if self.custom_filter is not None:
            sessions = [s for s in sessions if self.custom_filter(s)]

        return sessions

    def with_overrides(self, **kwargs) -> 'SessionFilter':
        """
        Return a new SessionFilter with specified fields replaced.

        Usage:
            stricter = base_filter.with_overrides(min_accuracy=0.80)
        """
        return replace(self, **kwargs)

    def describe(self) -> str:
        """Human-readable summary of active constraints."""
        parts = []
        if self.stage is not None:
            parts.append(f"stage={self.stage}")
        if self.distribution is not None:
            parts.append(f"distribution={self.distribution}")
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
        if self.exclude_opto:
            parts.append("no opto")
        if self.exclude_masking:
            parts.append("no masking")
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
        name: Preset key (e.g. 'expert_uniform')
        filt: SessionFilter instance

    Raises:
        TypeError: If filt is not a SessionFilter
    """
    if not isinstance(filt, SessionFilter):
        raise TypeError(
            f"Expected SessionFilter, got {type(filt).__name__}"
        )
    _PRESETS[name] = filt


def get_preset(name: str) -> SessionFilter:
    """
    Retrieve a registered preset by name.

    Raises:
        KeyError: If preset not found
    """
    if name not in _PRESETS:
        available = ', '.join(sorted(_PRESETS.keys())) or '(none)'
        raise KeyError(
            f"Unknown session preset '{name}'. Available: {available}"
        )
    return _PRESETS[name]


def list_presets() -> Dict[str, str]:
    """Return {name: description} for all registered presets."""
    return {name: filt.describe() for name, filt in sorted(_PRESETS.items())}


def clear_presets() -> None:
    """Remove all registered presets. Mainly for testing."""
    _PRESETS.clear()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def select_sessions(
    animal: 'AnimalData',
    preset: Optional[str] = None,
    **overrides,
) -> List['SessionData']:
    """
    Select sessions from an animal using a preset and/or keyword criteria.

    Three modes:
        1. Preset only:    select_sessions(animal, 'expert_uniform')
        2. Preset + override: select_sessions(animal, 'expert_uniform', min_accuracy=0.80)
        3. Ad-hoc:         select_sessions(animal, stage='Full_Task_Cont', last_n=5)

    Args:
        animal: AnimalData object
        preset: Name of a registered preset (optional)
        **overrides: Any SessionFilter field to set or override

    Returns:
        List of SessionData matching the criteria (may be empty)
    """
    if preset is not None:
        base = get_preset(preset)
        if overrides:
            filt = base.with_overrides(**overrides)
        else:
            filt = base
    else:
        filt = SessionFilter(**overrides)

    return filt.apply(animal)


# =============================================================================
# FITTING DATA BRIDGE
# =============================================================================

def fitting_data_from_sessions(
    sessions: List['SessionData'],
    animal_id: str,
    min_valid_trials: int = 10,
) -> 'FittingData':
    """
    Convert a list of (pre-filtered) SessionData into FittingData for SBI.

    Sessions must be pre-filtered via filter_trials() before calling.
    Internally calls sess.get_arrays() which only excludes aborts
    (always invalid). Since filter_trials clears flags on survivors,
    this is a no-op if filtering was already done.

    Pipeline:
        sessions = select_sessions(animal, 'expert_uniform')
        clean = filter_trials(sessions)
        fd = fitting_data_from_sessions(clean, animal.animal_id)

    Args:
        sessions: Pre-filtered SessionData list.
        animal_id: Animal identifier string.
        min_valid_trials: Skip sessions below this threshold.

    Returns:
        FittingData ready for SBIFitter.
    """
    from behav_utils.data.structures import FittingData

    stim_list, cat_list, choice_list = [], [], []
    no_resp_list, nbs_list = [], []
    sess_ids, sess_dates, sess_indices = [], [], []

    for sess in sessions:
        arrays = sess.get_arrays()

        n_valid = (~arrays['no_response']).sum()
        if n_valid < min_valid_trials:
            continue

        stim_list.append(arrays['stimuli'])
        cat_list.append(arrays['categories'])
        choice_list.append(arrays['choices'])
        no_resp_list.append(arrays['no_response'])

        n = len(arrays['stimuli'])
        nbs = np.ones(n, dtype=bool)
        if n > 0:
            nbs[0] = False
        nbs_list.append(nbs)

        sess_ids.append(sess.session_id)
        sess_dates.append(sess.date)
        sess_indices.append(sess.session_idx)

    return FittingData(
        animal_id=animal_id,
        session_ids=sess_ids,
        session_dates=sess_dates,
        session_indices=np.array(sess_indices, dtype=int),
        stimuli=stim_list,
        categories=cat_list,
        choices=choice_list,
        no_response=no_resp_list,
        not_blockstart=nbs_list,
        n_sessions=len(stim_list),
        trials_per_session=np.array([len(s) for s in stim_list]),
    )


# =============================================================================
# CONFIG-DRIVEN PRESET LOADING
# =============================================================================

def register_presets_from_config(config_raw: Dict[str, Any]) -> int:
    """
    Load session presets from a parsed YAML config dict.

    Expected format under 'session_presets':
        session_presets:
          expert_uniform:
            stage: "Full_Task_Cont"
            distribution: "Uniform"
            min_accuracy: 0.70
            last_fraction: 0.50
          post_shift_hard_a:
            stage: "Full_Task_Cont"
            distribution: "Hard-A"

    Args:
        config_raw: Parsed YAML dict (the full config)

    Returns:
        Number of presets registered
    """
    presets_raw = config_raw.get('session_presets', {})
    if not presets_raw:
        return 0

    count = 0
    for name, spec in presets_raw.items():
        if not isinstance(spec, dict):
            warnings.warn(f"Skipping preset '{name}': expected dict, got {type(spec)}")
            continue

        # Convert session_indices from YAML list if present
        if 'session_indices' in spec and isinstance(spec['session_indices'], list):
            spec['session_indices'] = list(spec['session_indices'])

        # Filter to only valid SessionFilter fields
        valid_fields = {f.name for f in SessionFilter.__dataclass_fields__.values()}
        filtered_spec = {k: v for k, v in spec.items() if k in valid_fields}

        unknown = set(spec.keys()) - valid_fields
        if unknown:
            warnings.warn(
                f"Preset '{name}': ignoring unknown fields {unknown}"
            )

        try:
            filt = SessionFilter(**filtered_spec)
            register_preset(name, filt)
            count += 1
        except Exception as e:
            warnings.warn(f"Failed to register preset '{name}': {e}")

    return count


# =============================================================================
# DEFAULT PRESETS
# =============================================================================
# Registered on import. Projects can override via config or explicit calls.

# ── Uniform distribution ─────────────────────────────────────────────────
register_preset('expert_uniform', SessionFilter(
    stage='Full_Task_Cont',
    distribution='Uniform',
    min_accuracy=0.70,
    last_fraction=0.50,
))

register_preset('all_uniform', SessionFilter(
    stage='Full_Task_Cont',
    distribution='Uniform',
))

register_preset('naive_uniform', SessionFilter(
    stage='Full_Task_Cont',
    distribution='Uniform',
    first_n=5,
))

# ── Hard distributions ───────────────────────────────────────────────────
register_preset('all_hard_a', SessionFilter(
    stage='Full_Task_Cont',
    distribution='Hard-A',
))

register_preset('all_hard_b', SessionFilter(
    stage='Full_Task_Cont',
    distribution='Hard-B',
))

register_preset('early_hard_a', SessionFilter(
    stage='Full_Task_Cont',
    distribution='Hard-A',
    first_n=5,
))

register_preset('early_hard_b', SessionFilter(
    stage='Full_Task_Cont',
    distribution='Hard-B',
    first_n=5,
))

register_preset('expert_hard_a', SessionFilter(
    stage='Full_Task_Cont',
    distribution='Hard-A',
    min_accuracy=0.60,
    last_fraction=0.50,
))

register_preset('expert_hard_b', SessionFilter(
    stage='Full_Task_Cont',
    distribution='Hard-B',
    min_accuracy=0.60,
    last_fraction=0.50,
))

# ── Global ───────────────────────────────────────────────────────────────
register_preset('all_stages', SessionFilter())

register_preset('all_full_task', SessionFilter(
    stage='Full_Task_Cont',
))
