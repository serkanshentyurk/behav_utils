"""
Resampling — the single drawing engine every uncertainty interval depends on, and the summaries of
its draws.

Drawing (``downsample`` and the ``resample_*`` functions): draws trials, lag-1 pairs or whole sessions
from a phase, with or without replacement, keeping each trial's frozen ``prev_*`` predecessor so trial
history survives the draw. Order-dependent statistics are refused (``is_exchangeable``).
Summaries (``bootstrap_phase_stats``, ``permute_phase_difference``, ``summarise_draws``): percentile
intervals, bootstrap p against zero and permutation p from the draws.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
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

__all__ = ['downsample', 'calculate_min_n', 'resample_psychometric_curve', 'resample_update_matrix', 'resample_stat_vectors',
           'bootstrap_phase_stats', 'permute_phase_difference', 'DrawSummary', 'summarise_draws', 'summarise_draw_frame']

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


def bootstrap_phase_stats(
    phase,
    names: Sequence[str],
    *,
    n_draws: int = 1000,
    n_trials: int | None = None,
    seed: int = 0,
    unit: str = 'trials',
) -> pd.DataFrame:
    """Resample one phase's trials (or sessions) and recompute its statistics.

    Delegates to :func:`resample_stat_vectors`,
    the library's single resample-and-recompute engine, so the frozen lag-1
    ``prev_*`` pairing survives every draw (a repeated trial carries its own
    predecessor).

    ``unit='trials'`` (default) is a per-trial bootstrap, stratified by stimulus
    bin, and answers "how precisely is this phase's stat pinned by its trials".
    ``unit='sessions'`` resamples whole sessions with replacement instead: it
    treats the session as the independent unit, so the interval reflects
    session-to-session scatter (the right unit when the phase spans sessions that
    were not randomised per trial — e.g. an opto phase vs a sham phase). A
    matched ``n_trials`` only applies to the trial bootstrap.

    The trial draw is stratified by stimulus bin: the stimulus composition is
    held roughly fixed across draws because the stimulus set is fixed by the
    design, so the interval is conditional on it (slightly narrower than an
    unconditional bootstrap). The session draw is unstratified.

    Args:
        phase:    list of SessionData from ``filter_trials``.
        names:    scalar stat names. For ``unit='trials'`` these must be
                  trial-exchangeable; ``unit='sessions'`` accepts any stat.
        n_draws:  number of resamples.
        n_trials: trial bootstrap only — draw this many trials instead of the
                  natural count (a matched n for equal-precision contrasts).
        seed:     RNG seed.
        unit:     'trials' (default) or 'sessions'.

    Returns:
        ``pd.DataFrame`` (n_draws × names). Failed draws are NaN rows.

    Raises:
        ValueError: if ``unit='trials'`` and any stat is not trial-exchangeable.
    """

    names = validate_names(names)
    if not names:
        return pd.DataFrame(index=range(n_draws))
    draw_n = None if unit == 'sessions' else n_trials
    return resample_stat_vectors(
        phase, names, n=draw_n, n_repeats=n_draws,
        with_replacement=True, unit=unit, seed=seed,
    )


def permute_phase_difference(
    phase_a,
    phase_b,
    names: Sequence[str],
    *,
    n_draws: int = 1000,
    n_trials: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Null distribution of ``stat(a) - stat(b)`` from shuffling the labels.

    Pools both phases, reassigns the phase label at random keeping the group
    sizes fixed, and recomputes the difference — the null being "which phase a
    trial belongs to carries no information".

    Only use this where the label really was randomised per trial. Opto vs
    non-opto within a phase qualifies (the rig interleaved it). Comparing phases
    that differ by session type does not: those trials were collected on
    different days, so a shuffle would treat non-exchangeable trials as
    exchangeable and absorb every between-day difference into the null. Use
    :func:`bootstrap_phase_stats` and an interval there instead.

    Args:
        phase_a, phase_b: the two phases; the difference is ``a - b``.
        names:            scalar stat names.
        n_draws:          number of shuffles. The smallest reportable p is
                          ``1 / (n_draws + 1)``.
        n_trials:         draw this many per side instead of the natural counts.
        seed:             RNG seed.

    Returns:
        ``pd.DataFrame`` (n_draws × names) of differences under the null. A
        shuffle that produced an unfittable split is a NaN row; ``summarise_draws``
        drops NaNs rather than counting them as zero.
    """
    names = validate_names(names)
    out = pd.DataFrame(np.full((n_draws, len(names)), np.nan), columns=list(names))
    if not names:
        return out

    a, b = TrialArrays.from_sessions(phase_a), TrialArrays.from_sessions(phase_b)
    a, b = a.valid(), b.valid()
    n_a, n_b = a.n_trials, b.n_trials
    combined = TrialArrays(
        np.concatenate([a.choice, b.choice]), np.concatenate([a.stimulus, b.stimulus]),
        np.concatenate([a.category, b.category]),
        np.concatenate([a.prev_choice, b.prev_choice]), np.concatenate([a.prev_stimulus, b.prev_stimulus]),
        np.concatenate([a.prev_category, b.prev_category]),
        np.concatenate([a.reaction_time, b.reaction_time]),
    )
    take_a = n_trials if n_trials is not None else n_a
    take_b = n_trials if n_trials is not None else n_b

    rng = np.random.default_rng(seed)
    for r in range(n_draws):
        shuffled = rng.permutation(n_a + n_b)
        sa = compute_stats(combined.take(shuffled[:take_a]), names, rng=rng, strict=False)
        sb = compute_stats(combined.take(shuffled[take_a:take_a + take_b]), names, rng=rng, strict=False)
        out.iloc[r] = (sa - sb).to_numpy()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Summarising a draw distribution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DrawSummary:
    ci_lo: float
    ci_hi: float
    p: float
    median: float
    n_draws: int

    @property
    def ci(self):
        return (self.ci_lo, self.ci_hi)


def summarise_draws(
    draws,
    *,
    observed: float | None = None,
    ci: float = 0.95,
    null_value: float = 0.0,
) -> DrawSummary:
    """Percentile interval and two-sided p from a distribution of draws.

    Works for either engine, but the p means different things:

    * bootstrap draws (of a difference) — ``p`` is the proportion of draws on
      the far side of ``null_value``, doubled. Report ``observed`` as the
      estimate and the interval as its uncertainty.
    * permutation draws — pass ``observed`` (the unshuffled difference) and
      ``p`` becomes the proper permutation p-value, ``(1 + #{|draw| >=
      |observed|}) / (n + 1)``. The ``+1`` keeps it from ever being exactly
      zero; the floor is ``1 / (n + 1)``.

    Do not compare two intervals by eye to judge a difference. Two 95%
    intervals can overlap while the difference is significant. Build the
    difference distribution and summarise that instead.

    Args:
        draws:      1-D array-like of resampled values (a DataFrame column). NaNs dropped.
        observed:   the unshuffled statistic — required for a permutation p.
        ci:         interval mass, e.g. 0.95.
        null_value: value corresponding to "no effect".

    Returns:
        :class:`DrawSummary`; all NaN (except ``n_draws``) if fewer than 10 usable draws.
    """
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 10:
        return DrawSummary(np.nan, np.nan, np.nan, np.nan, int(values.size))

    tail = (1.0 - ci) / 2.0 * 100.0
    ci_lo = float(np.percentile(values, tail))
    ci_hi = float(np.percentile(values, 100.0 - tail))

    if observed is not None and np.isfinite(observed):
        p = float((1 + np.sum(np.abs(values - null_value) >= abs(observed - null_value)))
                  / (values.size + 1))
    else:
        # (1 + count) / (n + 1) keeps a finite set of draws from ever reporting
        # p = 0; the floor 2 / (n + 1) is a property of n_draws, not evidence.
        below = 1 + int(np.sum(values <= null_value))
        above = 1 + int(np.sum(values >= null_value))
        p = float(min(1.0, 2.0 * min(below, above) / (values.size + 1)))

    return DrawSummary(ci_lo, ci_hi, p, float(np.median(values)), int(values.size))


def summarise_draw_frame(draws: pd.DataFrame, *, observed: pd.Series | None = None,
                         ci: float = 0.95, null_value: float = 0.0) -> pd.DataFrame:
    """``summarise_draws`` per column → DataFrame indexed by stat with columns ci_lo, ci_hi, p, median, n_draws."""
    rows = {}
    for col in draws.columns:
        obs = None if observed is None else observed.get(col)
        s = summarise_draws(draws[col], observed=obs, ci=ci, null_value=null_value)
        rows[col] = {'ci_lo': s.ci_lo, 'ci_hi': s.ci_hi, 'p': s.p, 'median': s.median, 'n_draws': s.n_draws}
    return pd.DataFrame.from_dict(rows, orient='index')
