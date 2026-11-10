"""
Trial-level filtering, extraction, and pooling.

The single place that decides *which trials*. ``SessionData.get_arrays`` is the
only wrapper on the data classes; analysis and plotting never filter.

Architecture:
    data/loading.py        → raw data from CSV
    data/ops/selection.py  → which sessions (session-level)
    data/ops/filtering.py  → which trials (trial-level)     ← THIS FILE
    stats/ readouts/ analysis/ → receive filtered data, compute
    plotting/              → receive results, draw

Pipeline:
    sessions = select_sessions(animal, preset='expert')          # session-level
    clean    = filter_trials(sessions, trial_type='non_opto')    # trial-level
    compute_psychometric_curve(TrialArrays.from_sessions(clean))  # no filtering

Public API:
    Mask building:
        build_mask(trials, ...)     — standard exclusions (abort, opto, no_response)
        opto_mask(trials, delta)    — trials relative to opto events

    Session filtering:
        filter_session(session, mask, label)  — new SessionData with filtered trials
        filter_trials(sessions, trial_type=…) — batch filter, returns List[SessionData]

    Array extraction (from pre-filtered data):
        get_arrays(trials)          — trial arrays as dict
        pool_arrays(sessions)       — concatenated across sessions
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Union

import numpy as np

from behav_utils.data.arrays import trial_key_of

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData, TrialData


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
    delta: Union[int, str] | None = 0,
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
            'control' → all non-abort, non-opto trials (adjacency to an opto
                        run is NOT considered; combine with ``delta=±k`` masks
                        to exclude neighbours)

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
    'trial_number', 'stimulus', 'stimulus_raw', 'choice', 'outcome', 'correct',
    'category', 'choice_raw', 'reaction_time', 'abort', 'opto_on',
    'distribution',
    # lag-1 view — sliced like any other per-trial array so the previous
    # trial is carried (frozen on the raw session), not recomputed on the subset
    'prev_stimulus', 'prev_choice', 'prev_correct', 'prev_category',
    'prev_reaction_time', 'prev_opto_on', 'prev_has_prev',
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
        arr = getattr(trials, field_name, np.array([]))   # default: field absent on an old pickle
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
    mask: np.ndarray | None = None,
    label: str = '',
    trial_type: str | None = None,
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
    # The scientific selection, carried as labels for the tidy frames downstream
    # (``compute_phase_stats`` / ``compute_rolling_stats`` read them).
    selection = {'session_type': session.session_type}
    if trial_type is not None:
        selection['trial_type'] = trial_type
    distribution = getattr(session, 'distribution', None)
    if distribution is not None:
        selection['distribution'] = distribution
    filter_info['selection'] = selection

    return SD(
        session_id=session.session_id,
        session_idx=session.session_idx,
        date=session.date,
        start=session.start,
        metadata=session.metadata,
        trials=new_trials,
        session_type=session.session_type,
        csv_path=session.csv_path,
        filter_info=filter_info,
        _days_since_first=session._days_since_first,
    )


def filter_trials(
    sessions: 'List[SessionData]',
    mask_fn: Callable | None = None,
    min_trials: int = 10,
    label: str = '',
    exclude_abort: bool = True,
    exclude_opto: bool = True,
    trial_type: Literal['all', 'non_opto', 'opto', 'post_opto'] | None = None,
) -> 'List[SessionData]':
    """
    Filter trials within each session. Returns new SessionData objects.

    Each session is filtered independently and rebuilt via ``filter_session``;
    sessions with fewer than ``min_trials`` surviving trials are dropped. Aborts
    are always dropped by ``trial_type``; no-response trials are kept (their
    choice is NaN and every statistic ignores them).

    Three ways to say which trials, in order of preference:

      1. ``trial_type`` — the named selections (use this):
           'all'       → every non-abort trial, opto included
           'non_opto'  → non-abort, laser-off trials
           'opto'      → laser-on trials only
           'post_opto' → the first laser-off trial after each opto run
      2. ``mask_fn`` — ``session -> bool mask`` for anything else
         (``exclude_abort`` / ``exclude_opto`` are then ignored).
      3. neither — ``build_mask`` with ``exclude_abort`` / ``exclude_opto``
         (equivalent to ``trial_type='non_opto'`` by default).

    ``trial_type`` and ``mask_fn`` are exclusive; passing both raises.

    Args:
        sessions:      List of SessionData.
        mask_fn:       Callable: session -> boolean mask (length n_trials).
        min_trials:    Drop sessions below this surviving-trial count.
        label:         Human-readable description (auto-generated if empty).
        exclude_abort: (no mask_fn / trial_type only) drop aborted trials.
        exclude_opto:  (no mask_fn / trial_type only) drop laser-on trials.
        trial_type:    One of the named selections above.

    Returns:
        List of new SessionData with ``filter_info`` set.

    Examples:
        clean = filter_trials(sessions, trial_type='non_opto')
        allv  = filter_trials(sessions_opto, trial_type='all')
        opto  = filter_trials(sessions_opto, trial_type='opto')
        post  = filter_trials(sessions_opto, trial_type='post_opto')
        pre   = filter_trials(sessions_opto, mask_fn=lambda s: build_mask(s.trials) & opto_mask(s.trials, -1),
                              label='pre-opto')

    Raises:
        ValueError: unknown ``trial_type``, or both ``trial_type`` and ``mask_fn`` given.
    """
    if trial_type is not None:
        if trial_type not in ('all', 'non_opto', 'opto', 'post_opto'):
            raise ValueError(f"Invalid trial_type '{trial_type}'. Choose from: 'all', 'non_opto', 'opto', 'post_opto'.")
        if mask_fn is not None:
            raise ValueError("filter_trials: pass either trial_type or mask_fn, not both.")

        def mask_fn(s, trial_type=trial_type):
            base = build_mask(s.trials, exclude_abort=True, exclude_opto=(trial_type == 'non_opto'))
            if trial_type == 'opto':
                return base & opto_mask(s.trials, delta=0)
            if trial_type == 'post_opto':
                return base & opto_mask(s.trials, delta=1)
            return base

        auto_label = label or f'trial_type={trial_type}'
    elif mask_fn is None:
        auto_label = label or _standard_mask_label(exclude_abort, exclude_opto)
    else:
        auto_label = label or 'custom'

    result = []
    for s in sessions:
        if mask_fn is None:
            mask = build_mask(s.trials, exclude_abort=exclude_abort,
                              exclude_opto=exclude_opto)
        else:
            mask = mask_fn(s)

        if mask.sum() < min_trials:
            continue

        result.append(filter_session(s, mask, label=auto_label,
                                     trial_type=trial_type))

    return result


def _standard_mask_label(exclude_abort: bool, exclude_opto: bool) -> str:
    parts = []
    if exclude_abort:
        parts.append('abort')
    if exclude_opto:
        parts.append('opto')
    return f"standard (exclude {' + '.join(parts)})" if parts else 'all trials'


# =============================================================================
# ARRAY EXTRACTION (from pre-filtered data)
# =============================================================================

def _stimulus_raw(trials: 'TrialData') -> np.ndarray:
    raw = getattr(trials, 'stimulus_raw', None)
    return raw if raw is not None and len(raw) == trials.n_trials else trials.stimulus


def get_arrays(trials: 'TrialData', session_idx: int = 0) -> Dict[str, np.ndarray]:
    """
    Extract trial arrays from a TrialData object — a pure projection.

    NO filtering is done here. ALL filtering, INCLUDING abort removal, must be
    done BEFORE calling this, via filter_session / filter_trials (abort removal
    lives in build_mask, default ``exclude_abort=True``). Passing an unfiltered
    session will therefore include abort trials.

    Returns:
        Dict with:
            stimuli         np.ndarray
            categories      np.ndarray
            choices         np.ndarray  (may contain NaN = no response)
            no_response     np.ndarray  (bool)
            reaction_times  np.ndarray
            prev_stimuli    np.ndarray  ┐ carried lag-1 view (abort-aware,
            prev_choices    np.ndarray  │ frozen on the raw session); NaN where
            prev_correct    np.ndarray  │ there is no predecessor or the
            prev_categories np.ndarray  │ previous trial was a no-response
            prev_reaction_time np.ndarray
            prev_opto_on    np.ndarray  │
            prev_has_prev   np.ndarray  ┘ bool (per-trial not_blockstart)
            trial_keys      np.ndarray  int64 identity per row (see ``TrialArrays``);
                            requires ``session_idx`` (pass it) — else NaN-free
                            but session-local (session_idx 0)
            n_trials        int
    """
    choices = trials.choice.astype(float)
    return {
        'stimuli': trials.stimulus,
        'stimuli_raw': _stimulus_raw(trials),
        'categories': trials.category,
        'choices': choices,
        'no_response': np.isnan(choices),
        'reaction_times': trials.reaction_time,
        'prev_stimuli': trials.prev_stimulus,
        'prev_choices': trials.prev_choice,
        'prev_correct': trials.prev_correct,
        'prev_categories': trials.prev_category,
        'prev_reaction_time': trials.prev_reaction_time,
        'prev_opto_on': trials.prev_opto_on,
        'prev_has_prev': trials.prev_has_prev,
        'trial_keys': trial_key_of(session_idx, trials.trial_number),
        'n_trials': len(choices),
    }


def pool_arrays(
    sessions: 'List[SessionData]',
) -> Dict[str, Any]:
    """
    Concatenate trial arrays across sessions.

    Calls get_arrays() on each session's trials and concatenates. Pooling only —
    no filtering of any kind; filter BEFORE calling this (filter_trials drops
    short sessions).

    Args:
        sessions:   List of SessionData (must be pre-filtered).

    Returns:
        Dict with:
            stimuli, categories, choices, no_response, reaction_times
                — concatenated current-trial arrays
            prev_stimuli, prev_choices, prev_correct, prev_categories,
            prev_reaction_time, prev_opto_on, prev_has_prev
                — concatenated carried lag-1 view (prev_has_prev is the pooled
                  not_blockstart)
            trial_keys     int64 — ``session_idx * 10**6 + trial_number`` per row
            n_trials       int   — total trial count
            n_sessions     int   — number of sessions included
            session_boundaries  list  — cumulative trial counts [0, n1, n1+n2, ...]
    """
    # Current-trial arrays + the carried lag-1 view. prev_* are concatenated
    # like the rest: each session's prev_* already carry the seam (NaN /
    # has_prev=False at that session's first completed trial), so the pooled
    # prev_has_prev equals not_blockstart with no re-shift across the seam.
    array_keys = [
        'stimuli', 'stimuli_raw', 'categories', 'choices', 'no_response', 'reaction_times',
        'prev_stimuli', 'prev_choices', 'prev_correct', 'prev_categories',
        'prev_reaction_time', 'prev_opto_on', 'prev_has_prev', 'trial_keys',
    ]

    all_arrays = []
    boundaries = [0]
    for s in sessions:
        arr = get_arrays(s.trials, session_idx=s.session_idx)
        all_arrays.append(arr)
        boundaries.append(boundaries[-1] + arr['n_trials'])

    if not all_arrays:
        out = {k: np.array([]) for k in array_keys}
        out['no_response'] = np.array([], dtype=bool)
        out['prev_has_prev'] = np.array([], dtype=bool)
        out['trial_keys'] = np.array([], dtype=np.int64)
        out.update({'n_trials': 0, 'n_sessions': 0, 'session_boundaries': [0]})
        return out

    out = {k: np.concatenate([a[k] for a in all_arrays]) for k in array_keys}
    out['n_trials'] = sum(a['n_trials'] for a in all_arrays)
    out['n_sessions'] = len(all_arrays)
    out['session_boundaries'] = boundaries
    return out
