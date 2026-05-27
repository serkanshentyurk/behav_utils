"""
Trial-level filtering, extraction, and pooling.

This is the SINGLE source of truth for all trial filtering.
Data classes (TrialData, SessionData) have thin wrappers that delegate here.
Analysis and plotting modules do NOT filter — they receive pre-filtered data.

Architecture:
    loading.py    → raw data from CSV
    selection.py  → which sessions (session-level)
    filtering.py  → which trials (trial-level)     ← THIS FILE
    analysis/     → receives filtered data, computes
    plotting/     → receives filtered data, draws

Pipeline:
    sessions = select_sessions(animal, preset='expert_uniform')  # session-level
    clean    = filter_trials(sessions)                           # trial-level
    plot_psychometric(clean, ax=ax)                              # no filtering

Public API:
    Mask building:
        build_mask(trials, ...)     — standard exclusions (abort, opto, no_response)
        opto_mask(trials, delta)    — trials relative to opto events

    Session filtering:
        filter_session(session, mask, label)  — new SessionData with filtered trials
        filter_trials(sessions, mask_fn, ...) — batch filter, returns List[SessionData]

    Array extraction (from pre-filtered data):
        get_arrays(trials)          — trial arrays as dict
        pool_arrays(sessions, ...)  — concatenated across sessions
"""

import numpy as np
from typing import Optional, List, Dict, Callable, Tuple, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from behav_utils.data.structures import TrialData, SessionData


# =============================================================================
# MASK BUILDING
# =============================================================================

def build_mask(
    trials: 'TrialData',
    exclude_abort: bool = True,
    exclude_opto: bool = True,
    exclude_no_response: bool = False,
) -> np.ndarray:
    """
    Build a boolean trial mask from standard exclusion flags.

    This is the standard "clean trials for analysis" mask.
    Compose with other conditions using & (AND):

        mask = build_mask(trials) & (trials.stimulus > 0)
        mask = build_mask(trials) & trials.correct

    Args:
        trials:              TrialData object.
        exclude_abort:       Remove aborted trials (default True).
        exclude_opto:        Remove opto trials (default True).
        exclude_no_response: Remove no-response trials (default False).

    Returns:
        Boolean mask of length trials.n_trials.
    """
    mask = np.ones(trials.n_trials, dtype=bool)
    if exclude_abort:
        mask &= ~trials.abort
    if exclude_opto:
        mask &= ~trials.opto_on
    if exclude_no_response:
        mask &= ~trials.no_response
    return mask


def opto_mask(
    trials: 'TrialData',
    delta: Optional[Union[int, str]] = 0,
) -> np.ndarray:
    """
    Boolean mask for trials at a fixed offset from opto events.

    Aborted trials are excluded from all masks (they are invalid data).
    Consecutive opto trials are treated as one "run" — offsets are
    measured from run boundaries, not individual trials within a run.

    Args:
        trials: TrialData object.
        delta:
            None      → all non-abort trials
            0         → opto trials
            1         → first non-abort non-opto trial after each opto run
            2         → second non-abort non-opto after each opto run
            -1        → last non-abort non-opto before each opto run
            'control' → all non-abort, non-opto trials

    Returns:
        Boolean mask of length trials.n_trials.

    Examples:
        opto_mask(trials, delta=0)          # opto trials
        opto_mask(trials, delta='control')  # interleaved controls
        opto_mask(trials, delta=1)          # post-opto (carry-over)
        opto_mask(trials, delta=-1)         # pre-opto
    """
    valid = ~trials.abort
    n = trials.n_trials

    if delta is None:
        return valid.copy()
    if delta == 'control':
        return valid & ~trials.opto_on
    if delta == 0:
        return valid & trials.opto_on

    # ── Identify opto run boundaries ─────────────────────────────────
    runs = []
    in_run = False
    start = None
    for t in range(n):
        if trials.opto_on[t]:
            if not in_run:
                start = t
                in_run = True
        else:
            if in_run:
                runs.append((start, t - 1))
                in_run = False
    if in_run:
        runs.append((start, n - 1))

    mask = np.zeros(n, dtype=bool)

    if delta > 0:
        for _, run_end in runs:
            count = 0
            for t in range(run_end + 1, n):
                if trials.opto_on[t]:
                    break
                if valid[t]:
                    count += 1
                    if count == delta:
                        mask[t] = True
                        break
    elif delta < 0:
        k = abs(delta)
        for run_start, _ in runs:
            count = 0
            for t in range(run_start - 1, -1, -1):
                if trials.opto_on[t]:
                    break
                if valid[t]:
                    count += 1
                    if count == k:
                        mask[t] = True
                        break

    return mask


# =============================================================================
# TRIAL DATA FILTERING
# =============================================================================

# Fields that get sliced when filtering TrialData
_TRIAL_ARRAY_FIELDS = [
    'trial_number', 'stimulus', 'choice', 'outcome', 'correct',
    'category', 'choice_raw', 'reaction_time', 'abort', 'opto_on',
    'distribution',
]


def filter_trial_data(
    trials: 'TrialData',
    mask: np.ndarray,
    clear_flags: bool = True,
) -> 'TrialData':
    """
    Return new TrialData containing only trials where mask is True.

    All arrays are sliced to match the mask. By default, abort and
    opto_on flags are cleared on survivors (they served their purpose
    during selection — downstream code should receive clean data).

    Args:
        trials:      TrialData to filter.
        mask:        Boolean array of length trials.n_trials.
        clear_flags: If True (default), clear abort and opto_on on survivors.

    Returns:
        New TrialData with mask.sum() trials.
    """
    from behav_utils.data.structures import TrialData as TD

    kwargs = {}
    for field_name in _TRIAL_ARRAY_FIELDS:
        arr = getattr(trials, field_name)
        if isinstance(arr, np.ndarray) and len(arr) == trials.n_trials:
            kwargs[field_name] = arr[mask]
        else:
            kwargs[field_name] = arr

    kwargs['optional_fields'] = {
        k: v[mask] if isinstance(v, np.ndarray) and len(v) == trials.n_trials else v
        for k, v in trials.optional_fields.items()
    }
    kwargs['extra'] = {
        k: v[mask] if isinstance(v, np.ndarray) and len(v) == trials.n_trials else v
        for k, v in trials.extra.items()
    }

    new = TD(**kwargs)

    if clear_flags:
        n_surviving = int(mask.sum())
        new.abort = np.zeros(n_surviving, dtype=bool)
        new.opto_on = np.zeros(n_surviving, dtype=bool)

    return new


# =============================================================================
# SESSION FILTERING
# =============================================================================

def filter_session(
    session: 'SessionData',
    mask: Optional[np.ndarray] = None,
    label: str = '',
) -> 'SessionData':
    """
    Return new SessionData with only the selected trials.

    If mask is None, applies standard exclusions (abort + opto)
    via build_mask(). Metadata, date, session_id are preserved.
    filter_info records what was done.

    Args:
        session: SessionData to filter.
        mask:    Boolean array of length n_trials.
                 If None, uses build_mask() defaults.
        label:   Human-readable description (auto-generated if empty).

    Returns:
        New SessionData with filtered trials and filter_info set.

    Examples:
        filter_session(session)                                      # standard
        filter_session(session, opto_mask(session.trials, delta=0))  # opto only
        filter_session(session, session.trials.correct, 'correct')   # correct only
    """
    from behav_utils.data.structures import SessionData as SD

    n_original = session.n_trials

    if mask is None:
        mask = build_mask(session.trials)
        label = label or 'standard (exclude abort + opto)'
    elif not label:
        label = 'custom'

    new_trials = filter_trial_data(session.trials, mask, clear_flags=True)

    filter_info = {
        'label': label,
        'n_original': n_original,
        'n_filtered': int(mask.sum()),
        'fraction_kept': float(mask.sum() / n_original) if n_original > 0 else 0.0,
        'parent_session_id': session.session_id,
    }

    return SD(
        session_id=session.session_id,
        session_idx=session.session_idx,
        date=session.date,
        metadata=session.metadata,
        trials=new_trials,
        masking=session.masking,
        csv_path=session.csv_path,
        filter_info=filter_info,
        _days_since_first=session._days_since_first,
    )


def filter_trials(
    sessions: 'List[SessionData]',
    mask_fn: Optional[Callable] = None,
    min_trials: int = 10,
    label: str = '',
) -> 'List[SessionData]':
    """
    Filter trials within each session. Returns new SessionData objects.

    Each session is independently filtered: mask_fn is called per session
    to produce a boolean mask, then filter_session creates a new
    SessionData with only the selected trials.

    Sessions with fewer than min_trials surviving trials are dropped.

    Args:
        sessions:   List of SessionData.
        mask_fn:    Callable: session → boolean mask (length n_trials).
                    If None, applies standard exclusions (abort + opto).
        min_trials: Drop sessions below this threshold.
        label:      Human-readable description.

    Returns:
        List of new SessionData with filter_info set.

    Examples:
        # Standard (exclude abort + opto)
        clean = filter_trials(sessions)

        # Opto trials only
        opto = filter_trials(sessions,
            mask_fn=lambda s: opto_mask(s.trials, delta=0),
            label='opto trials')

        # Post-opto
        post = filter_trials(sessions,
            mask_fn=lambda s: opto_mask(s.trials, delta=1),
            label='post-opto')

        # Custom: fast correct
        fast = filter_trials(sessions,
            mask_fn=lambda s: (build_mask(s.trials)
                              & s.trials.correct
                              & (s.trials.reaction_time < 0.5)),
            label='fast correct')
    """
    auto_label = label or ('standard (exclude abort + opto)' if mask_fn is None else 'custom')

    result = []
    for s in sessions:
        mask = build_mask(s.trials) if mask_fn is None else mask_fn(s)

        if mask.sum() < min_trials:
            continue

        result.append(filter_session(s, mask, label=auto_label))

    return result


# =============================================================================
# ARRAY EXTRACTION (from pre-filtered data)
# =============================================================================

def get_arrays(trials: 'TrialData') -> Dict[str, np.ndarray]:
    """
    Extract trial arrays from a TrialData object.

    Aborts are always excluded (they are invalid data, not a
    scientific choice). All other filtering should be done BEFORE
    calling this, via filter_session / filter_trials.

    On pre-filtered data (where clear_flags=True was used),
    abort is already all-False, so this is effectively a no-op filter.

    Args:
        trials: TrialData object (ideally pre-filtered).

    Returns:
        Dict with:
            stimuli        np.ndarray
            categories     np.ndarray
            choices        np.ndarray  (may contain NaN = no response)
            no_response    np.ndarray  (bool)
            reaction_times np.ndarray
            trial_indices  np.ndarray  (original indices within the TrialData)
            n_trials       int
    """
    mask = ~trials.abort
    choices = trials.choice[mask].astype(float)

    return {
        'stimuli': trials.stimulus[mask],
        'categories': trials.category[mask],
        'choices': choices,
        'no_response': np.isnan(choices),
        'reaction_times': trials.reaction_time[mask],
        'trial_indices': np.where(mask)[0],
        'n_trials': int(mask.sum()),
    }


def pool_arrays(
    sessions: 'List[SessionData]',
    min_trials: int = 0,
) -> Dict[str, Any]:
    """
    Concatenate trial arrays across sessions.

    Calls get_arrays() on each session's trials and concatenates.
    No additional filtering — filter BEFORE calling this.

    Args:
        sessions:   List of SessionData (typically pre-filtered).
        min_trials: Skip sessions with fewer trials.

    Returns:
        Dict with:
            stimuli, categories, choices, no_response, reaction_times
                — concatenated arrays
            n_trials       int   — total trial count
            n_sessions     int   — number of sessions included
            session_boundaries  list  — cumulative trial counts [0, n1, n1+n2, ...]
    """
    all_arrays = []
    boundaries = [0]

    for s in sessions:
        arr = get_arrays(s.trials)
        if arr['n_trials'] < min_trials:
            continue
        all_arrays.append(arr)
        boundaries.append(boundaries[-1] + arr['n_trials'])

    if not all_arrays:
        return {
            'stimuli': np.array([]),
            'categories': np.array([]),
            'choices': np.array([]),
            'no_response': np.array([], dtype=bool),
            'reaction_times': np.array([]),
            'n_trials': 0,
            'n_sessions': 0,
            'session_boundaries': [],
        }

    return {
        'stimuli': np.concatenate([a['stimuli'] for a in all_arrays]),
        'categories': np.concatenate([a['categories'] for a in all_arrays]),
        'choices': np.concatenate([a['choices'] for a in all_arrays]),
        'no_response': np.concatenate([a['no_response'] for a in all_arrays]),
        'reaction_times': np.concatenate([a['reaction_times'] for a in all_arrays]),
        'n_trials': sum(a['n_trials'] for a in all_arrays),
        'n_sessions': len(all_arrays),
        'session_boundaries': boundaries,
    }
