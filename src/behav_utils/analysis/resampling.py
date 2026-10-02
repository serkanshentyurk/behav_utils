"""
Trial-level resampling for phase comparisons.

A *phase* is a pooled trial set — whatever ``filter_trials`` returned, treated as
one mega-session. These engines turn a phase into a distribution of statistic
values, so uncertainty can be attached to any contrast built from them.

    load_experiment → select_sessions → filter_trials → phase
                                                          │
                          bootstrap_phase_stats ──────────┤
                          permute_phase_difference ───────┘
                                     ↓
                          summarise_draws

Two engines, two jobs — they are not interchangeable:

**Bootstrap** resamples trials with labels fixed. It answers "how precisely is
this quantity pinned down", and is valid wherever the trials were sampled, which
is everywhere. Use it for confidence intervals, and for any contrast between
phases that were not randomised against each other (sham sessions vs opto
sessions were recorded on different days, so no shuffle mimics the design).

**Permutation** shuffles the condition label to build a null. It answers "could
the label be irrelevant", and is only meaningful when the label was assigned in a
way the shuffle reproduces. Opto was randomised per trial by the rig, so
permuting opto/non-opto *within a phase* is well grounded. Permuting a
session-level property across trials is not.

Composition rule: bootstrap each condition once, keep the draws, and build every
contrast by subtracting draw frames. Differences of differences then work
directly, and a shared reference cancels exactly — ``(A - C) - (B - C)`` uses the
*same* ``C`` draws, so its variance drops out instead of being double-counted.

Every engine returns a ``pd.DataFrame`` of draws: one row per draw, one column
per stat, columns in request order, NaN where a draw failed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats import compute_stats, validate_names

__all__ = [
    'bootstrap_phase_stats',
    'permute_phase_difference',
    'summarise_draws',
    'DrawSummary',
]


# ─────────────────────────────────────────────────────────────────────────────
# Engines
# ─────────────────────────────────────────────────────────────────────────────

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

    Delegates to :func:`behav_utils.analysis.downsample.resample_stat_vectors`,
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
    from behav_utils.analysis.downsample import resample_stat_vectors

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
