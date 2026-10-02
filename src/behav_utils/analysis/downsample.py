"""Matched-n downsampling that returns SessionData, so compute_x is unchanged.

Pipeline:  filter -> calculate_min_n -> resample_<readout>  (or downsample -> compute_x).

`downsample` is `filter_trials` with a stratified n-subset selector instead of a
deterministic mask. It works because `prev_*` are stored frozen fields on TrialData and
`filter_trial_data` slices them with the rest, so a row-subset (or a with-replacement
resample, via repeated integer indices) keeps the lag-1 pairing intact. The clean sessions
are already abort/opto-cleared, so we slice with clear_flags=False — which also lets the
same path carry repeated indices for the bootstrap (with_replacement=True).

Two units: 'trials' (responded trials, for the psychometric) and 'pairs' (post-correct
pairs, for the update matrix — the rows compute_update_matrix actually uses).
"""
from __future__ import annotations

import dataclasses
import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from behav_utils.data.arrays import TrialArrays
from behav_utils.data.ops.filtering import filter_trial_data, pool_arrays
from behav_utils.data.structures import SessionData
from behav_utils.readouts import (
    PsychometricCurve,
    UpdateMatrix,
    compute_psychometric_curve,
    compute_update_matrix,
)
from behav_utils.readouts._base import X_FIT, _ro
from behav_utils.stats import compute_stats, is_exchangeable, validate_names


def _pair_base_mask(pooled) -> np.ndarray:
    """Rows fit_update_matrix counts as post-correct pairs (mirrors its session-path base)."""
    prev_choices = np.asarray(pooled['prev_choices'], dtype=float)
    prev_categories = np.asarray(pooled['prev_categories'], dtype=float)
    no_response = np.asarray(pooled['no_response'], dtype=bool)
    has_prev = np.asarray(pooled['prev_has_prev'], dtype=bool)
    return ((prev_choices == prev_categories) & (~no_response)
            & (~np.isnan(prev_choices)) & has_prev)


def _pool_index(pooled, unit) -> np.ndarray:
    """Global pooled-row indices eligible for the unit."""
    if unit == 'trials':
        return np.where(~np.asarray(pooled['no_response'], dtype=bool))[0]
    if unit == 'pairs':
        return np.where(_pair_base_mask(pooled))[0]
    raise ValueError(f"unit must be 'trials' or 'pairs', got {unit!r}")


def _draw(pooled, n, unit, n_bins, rng, replace) -> np.ndarray:
    """Stratified draw of ~n global pooled-row indices (stratified by stimulus / prev stimulus)."""
    idx_pool = _pool_index(pooled, unit)
    total = len(idx_pool)
    if total == 0:
        return np.array([], dtype=int)

    edges = np.linspace(-1, 1, n_bins + 1)
    strat_var = 'prev_stimuli' if unit == 'pairs' else 'stimuli'
    strat = np.clip(np.digitize(np.asarray(pooled[strat_var])[idx_pool], edges) - 1, 0, n_bins - 1)

    keep = []
    for b in range(n_bins):
        in_b = idx_pool[strat == b]
        if len(in_b) == 0:
            continue
        kb = round(n * len(in_b) / total)
        if not replace:
            kb = min(kb, len(in_b))
        if kb > 0:
            keep.append(rng.choice(in_b, kb, replace=replace))
    return np.concatenate(keep) if keep else np.array([], dtype=int)


def _slice_session(session: SessionData, local_idx: np.ndarray) -> SessionData:
    """New SessionData with this session's trials at local_idx (repeats allowed)."""
    new_trials = filter_trial_data(session.trials, local_idx, clear_flags=False)
    return SessionData(
        session_id=session.session_id, session_idx=session.session_idx,
        date=session.date, metadata=session.metadata, trials=new_trials,
        session_type=session.session_type, csv_path=session.csv_path,
        filter_info={'label': 'downsampled', 'n_filtered': len(local_idx),
                     'parent_session_id': session.session_id},
        _days_since_first=session._days_since_first,
    )


def downsample(clean, n, unit='trials', with_replacement=True, n_bins=8, rng=None):
    """Subsample clean sessions to ~n of the chosen unit; returns new [SessionData].

    The matched-n draw is pooled across sessions (frozen prev_* make this safe), then split
    back per session and rebuilt, so the output is consumable by compute_psychometric_curve /
    compute_update_matrix unchanged. with_replacement=True allows a trial to be drawn more than once
    (a bootstrap resample); False is a clean subsample.
    """
    rng = rng if rng is not None else np.random.default_rng()
    pooled = pool_arrays(clean)
    if pooled['n_trials'] == 0:
        return []

    sel = _draw(pooled, n, unit, n_bins, rng, with_replacement)
    boundaries = pooled['session_boundaries']

    out = []
    for i, session in enumerate(clean):
        lo, hi = boundaries[i], boundaries[i + 1]
        in_session = sel[(sel >= lo) & (sel < hi)] - lo
        if len(in_session) == 0:
            continue
        out.append(_slice_session(session, in_session))
    return out


def calculate_min_n(phases, unit='trials') -> int:
    """Smallest unit-count across a list of clean session-lists (the matched-n target).

    Args:
        phases: list of [SessionData] (each a filtered phase/condition); empties skipped.
        unit:   'trials' (responded trials) or 'pairs' (post-correct pairs).
    """
    counts = []
    for clean in phases:
        if not clean:
            continue
        pooled = pool_arrays(clean)
        if pooled['n_trials'] == 0:
            continue
        c = len(_pool_index(pooled, unit))
        if c > 0:
            counts.append(c)
    return min(counts) if counts else 0


# ── resampled readouts: K matched-n draws → one aggregated readout ──────────

def aggregate_psychometric_curves(repeats: Sequence[PsychometricCurve], n_trials: int) -> PsychometricCurve:
    """Mean curve, mean params and a 2.5–97.5 percentile band across successful repeats."""
    ok = [r for r in repeats if r.success]
    n_bins = repeats[0].bin_centres.size if repeats else 8
    centres = repeats[0].bin_centres if repeats else _ro(np.full(n_bins, np.nan))
    if not ok:
        z = np.full(n_bins, np.nan)
        return PsychometricCurve(np.nan, np.nan, np.nan, np.nan, X_FIT,
                                 _ro(np.full(X_FIT.size, np.nan)), centres, _ro(z),
                                 _ro(np.zeros(n_bins)), n_trials, False)
    Y = np.stack([r.y for r in ok])
    P = np.stack([r.params.to_numpy() for r in ok])
    B = np.stack([r.bin_means for r in ok])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        params = np.nanmean(P, axis=0)
        ci = np.stack([np.nanpercentile(P, 2.5, axis=0), np.nanpercentile(P, 97.5, axis=0)], axis=1)
        y = np.nanmean(Y, axis=0)
        band = np.stack([np.nanpercentile(Y, 2.5, axis=0), np.nanpercentile(Y, 97.5, axis=0)])
        bin_means = np.nanmean(B, axis=0)
    return PsychometricCurve(*map(float, params), X_FIT, _ro(y), centres, _ro(bin_means),
                             _ro(np.full(n_bins, n_trials // n_bins)), n_trials, True,
                             ci=_ro(ci), band=_ro(band), n_bootstrap=len(ok))


def resample_psychometric_curve(clean, n, *, n_repeats=100, with_replacement=True,
                                n_bins=8, seed=42) -> PsychometricCurve:
    """``n_repeats`` matched-n draws of ``n`` responded trials, each fitted, then aggregated."""
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_repeats):
        ds = downsample(clean, n, unit='trials', with_replacement=with_replacement, n_bins=n_bins, rng=rng)
        reps.append(compute_psychometric_curve(TrialArrays.from_sessions(ds), n_bins=n_bins, n_bootstrap=0))
    return aggregate_psychometric_curves(reps, n)


def resample_update_matrix(clean, n, *, n_repeats=100, with_replacement=True,
                           n_bins=8, trial_filter='post_correct', seed=42) -> UpdateMatrix:
    """``n_repeats`` matched-n draws of ``n`` post-correct pairs, each fitted, cell-wise mean."""
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_repeats):
        ds = downsample(clean, n, unit='pairs', with_replacement=with_replacement, n_bins=n_bins, rng=rng)
        reps.append(compute_update_matrix(TrialArrays.from_sessions(ds), n_bins=n_bins,
                                          trial_filter=trial_filter))
    avg = UpdateMatrix.average(reps)
    return dataclasses.replace(avg, n_trials=int(n))   # n_sources = n_repeats, n_trials = matched n


def _resample_whole_sessions(clean, k, with_replacement, rng):
    """Draw ``k`` whole sessions from ``clean`` and return them as a list.

    The session is the unit: sessions are selected intact (never sliced), so
    within-session trial order and the frozen lag-1 ``prev_*`` arrays are carried
    unchanged, and a session drawn twice contributes its trials twice (the
    correct bootstrap behaviour — ``pool_arrays`` concatenates duplicates). This
    is why session resampling is valid for order-dependent stats that trial
    resampling must refuse.

    Args:
        clean:            list of SessionData (a phase/condition).
        k:                number of sessions to draw.
        with_replacement: True for a bootstrap resample; False for a subsample.
        rng:              numpy Generator.

    Returns:
        list of SessionData (length ``min(k, len(clean))`` when without
        replacement; ``k`` with replacement). Empty if ``clean`` is empty.
    """
    m = len(clean)
    if m == 0 or k <= 0:
        return []
    if with_replacement:
        idx = rng.integers(0, m, size=k)
    else:
        idx = rng.permutation(m)[:k]
    return [clean[i] for i in idx]


# ── resample-and-recompute for scalar/param summary stats ───────────────────────
def resample_stat_vectors(
    clean,
    stat_names,
    *,
    n=None,
    n_repeats=1000,
    with_replacement=True,
    unit='trials',
    seed=0,
):
    """Resample trials K times and recompute summary stats — one matrix of replicates.

    The single resample-and-recompute engine for scalar stats. It is the scalar
    analogue of :func:`resample_psychometric_curve` / :func:`resample_update_matrix`
    (which target the readouts), and serves BOTH uses via its arguments:

      * trial bootstrap   — ``with_replacement=True``,  ``n=None`` (natural count)
      * matched-n draw     — ``with_replacement=False``, ``n=target_n``

    Drawing is delegated to :func:`downsample`, so the frozen lag-1 ``prev_*``
    pairing is preserved on every resample (a repeated trial index carries its own
    predecessor). Stats registered ``exchangeable=False`` are refused: trial
    resampling is invalid for order-dependent stats and would return a
    confidently-wrong interval.

    Args:
        clean:           list of [SessionData], abort/opto-cleared (a phase/condition).
        stat_names:      scalar stat names (``list_stats()``).
        n:               trials/pairs to draw per repeat; None → the natural count
                         of ``unit`` in ``clean`` (the right default for a bootstrap).
        n_repeats:       number of resamples (rows of the returned matrix).
        with_replacement: True for a bootstrap resample, False for a clean subsample.
        unit:            'trials' (responded trials), 'pairs' (post-correct
                         pairs), or 'sessions' (whole sessions, drawn intact —
                         valid for order-dependent stats; natural n = n_sessions).
        seed:            RNG seed.

    Returns:
        ``pd.DataFrame`` of shape ``(n_repeats, len(stat_names))``, columns in
        request order. Rows where the draw was empty are NaN.

    Raises:
        ValueError: if any requested stat is not trial-exchangeable.
    """
    stat_names = validate_names(stat_names)
    # Order-dependence guard is unit-aware. Trial/pairs resampling reshuffles
    # trials, so stats that depend on trial order beyond the frozen lag-1 view
    # are refused. Session resampling keeps whole sessions intact — full trial
    # order (and every lag) is preserved — so no stat is refused there.
    if unit != 'sessions':
        bad = [s for s in stat_names if not is_exchangeable(s)]
        if bad:
            raise ValueError(
                f"trial resampling is invalid for order-dependent stat(s) {bad}; "
                f"exclude them from the bootstrap / downsample (they depend on trial "
                f"order beyond the frozen lag-1 view), or resample with unit='sessions'."
            )

    rng = np.random.default_rng(seed)
    # A separate stream for stochastic stats (e.g. reaction_time_jitter): passing
    # it to compute_stats means such a stat re-draws its noise every resample
    # (folding that uncertainty into the interval) without perturbing the
    # resampling draw sequence — so every deterministic stat's draws are
    # unchanged whether or not a stochastic stat is present.
    jitter_rng = np.random.default_rng(seed + 2654435761)
    out = np.full((n_repeats, len(stat_names)), np.nan)

    def _frame():
        return pd.DataFrame(out, columns=list(stat_names))

    if not clean:
        return _frame()

    # Natural draw size per unit. For sessions the unit is the whole session, so
    # the natural count is len(clean); trials/pairs count pooled rows.
    if unit == 'sessions':
        k = len(clean) if n is None else int(n)
        if k <= 0:
            return _frame()
    else:
        if n is None:
            pooled0 = pool_arrays(clean)
            if pooled0['n_trials'] == 0:
                return _frame()
            n = len(_pool_index(pooled0, unit))
        if n <= 0:
            return _frame()

    for r in range(n_repeats):
        if unit == 'sessions':
            drawn = _resample_whole_sessions(clean, k, with_replacement, rng)
        else:
            drawn = downsample(clean, n, unit=unit,
                               with_replacement=with_replacement, rng=rng)
        if not drawn:
            continue
        arrays = TrialArrays.from_sessions(drawn)
        if arrays.n_trials == 0:
            continue
        out[r] = compute_stats(arrays, stat_names, rng=jitter_rng, strict=False).to_numpy()

    return _frame()
