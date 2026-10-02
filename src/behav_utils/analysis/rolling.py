"""
Rolling summary statistics over ordered trials.

    compute_rolling_stats(sessions, names, per_session=True|False) -> RollingStats

The general windowed-stat computer: it slides a fixed window along trials in
acquisition order and fits summary statistics in each window. This is the
generic 2-AFC tool — continuous stimulus, binary choice, optional opto — and it
knows nothing about distribution names, session types, or normative models.

Filtering (including opto / non-opto selection) is a prior step: pass sessions
that have already been through ``filter_trials``, per the pipeline

    load -> select_sessions -> filter_trials -> compute_rolling_stats -> plot

so this function never re-filters. The session_type label rides along on each
per-session entry as neutral metadata; assigning it meaning (opto vs sham,
group, …) is the caller's job in the plot.

μ is fitted through the stat registry (``compute_stats``), so
windows the registry judges unreliable — the curve too flat (σ above threshold)
or the PSE run to the stimulus edge (|μ| > 0.99) — come back NaN. If you want
the *raw* PSE trajectory that keeps those early, still-shallow windows (the
adaptation curve), see the project's per-session trajectory
instead; those deliberately fit raw μ.

Windowing is full-windows-only: a window is emitted only when ``window`` trials
are available, so a curve stops one window short of the block end rather than
ending on a shorter, noisier window. A session with fewer than ``window`` but at
least ``min_short`` trials contributes a single whole-session point (matching
a per-session trajectory); a shorter session contributes an empty
curve, so it still appears (annotate the empty panel) rather than vanishing.
"""

import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats import compute_stats, validate_names


def _iter_windows(n: int, window: int, step: int) -> List[Tuple[float, slice]]:
    """Full windows only: ``(centre, slice)`` for each start in ``range(0, n-window+1, step)``."""
    return [(start + window / 2.0, slice(start, start + window))
            for start in range(0, n - window + 1, step)]


def _roll(arrays: TrialArrays, names: List[str], window: int, step: int, min_short: int
          ) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling ``names`` over one ordered block. Returns ``(centres, values (n_windows × names))``.

    Full windows only; a block of ``min_short..window-1`` trials yields one
    whole-block window centred at ``n/2``; shorter blocks yield nothing. A
    window that fails to fit is NaN (``strict=False``).
    """
    n = arrays.n_trials
    if n >= window:
        windows = _iter_windows(n, window, step)
    elif n >= min_short:
        windows = [(n / 2.0, slice(0, n))]
    else:
        windows = []
    centres = np.array([c for c, _ in windows], dtype=float)
    values = np.full((len(windows), len(names)), np.nan)
    for i, (_, sl) in enumerate(windows):
        values[i] = compute_stats(arrays.take(np.arange(n)[sl]), names, strict=False).to_numpy()
    return centres, values


@dataclass(frozen=True)
class RollingStats:
    """Rolling statistics as a tidy frame.

    ``curves`` columns: ``session, session_idx, session_type, distribution,
    trial, stat, value``; ``session_info`` columns: ``session, session_idx,
    session_type, distribution, n_trials`` (a session too short for any window
    still has its row). In pooled mode there is a single session labelled
    ``'pooled'``.
    """

    curves: pd.DataFrame
    session_info: pd.DataFrame     # one row per session (present even when its curve is empty)
    names: Tuple[str, ...]
    window: int
    step: int
    per_session: bool

    def curve(self, stat: str, session=None) -> pd.DataFrame:
        """``trial, value`` rows for one stat (and one session if given)."""
        c = self.curves[self.curves['stat'] == stat]
        if session is not None:
            c = c[c['session'] == session]
        return c[['session', 'trial', 'value']].reset_index(drop=True)

    @property
    def sessions(self) -> list:
        return list(self.session_info['session'])

    def n_trials(self, session) -> int:
        return int(self.session_info.set_index('session').loc[session, 'n_trials'])

    def __repr__(self) -> str:
        return (f'RollingStats(stats={list(self.names)}, window={self.window}, step={self.step}, '
                f'per_session={self.per_session}, n_sessions={len(self.sessions)})')


CURVE_COLUMNS = ['session', 'session_idx', 'session_type', 'distribution', 'trial', 'stat', 'value']
INFO_COLUMNS = ['session', 'session_idx', 'session_type', 'distribution', 'n_trials']


def _rows(centres, values, names, session, session_idx, session_type, distribution):
    if centres.size == 0:
        return pd.DataFrame(columns=CURVE_COLUMNS)
    return pd.DataFrame({
        'session': session, 'session_idx': session_idx, 'session_type': session_type,
        'distribution': distribution,
        'trial': np.repeat(centres, len(names)), 'stat': list(names) * len(centres),
        'value': values.ravel(),
    }, columns=CURVE_COLUMNS)


def compute_rolling_stats(
    sessions,
    names,
    *,
    per_session: bool = True,
    window: int = 50,
    step: int = 10,
    min_short: int = 10,
) -> RollingStats:
    """Rolling scalar statistics over pre-filtered sessions.

    Slides a ``window``-trial window (stride ``step``) along trials in
    acquisition order and computes ``names`` in each window.

    Args:
        sessions:    pre-filtered SessionData. ``filter_trials`` must have run.
        names:       scalar stat name(s); str or list.
        per_session: True windows within each session (no window crosses a
                     boundary), one curve per session. False concatenates all
                     sessions in order and windows across the run.
        window:      window size in trials.
        step:        stride between window centres. Overlapping windows smooth
                     the curve but do NOT add independent information.
        min_short:   a block of ``min_short..window-1`` trials still gets one
                     whole-block point; below this it gets nothing.

    Returns:
        :class:`RollingStats`.
    """
    names = validate_names([names] if isinstance(names, str) else names)
    if not names:
        raise ValueError('compute_rolling_stats: names is empty.')
    sessions = list(sessions)

    if not per_session:
        n = 0
        if sessions:
            arrays = TrialArrays.from_sessions(sessions)
            n = arrays.n_trials
            c, v = _roll(arrays, names, window, step, min_short)
        else:
            c, v = np.array([]), np.zeros((0, len(names)))
        curves = _rows(c, v, names, 'pooled', None, '', None)
        info = pd.DataFrame([['pooled', None, '', None, n]], columns=INFO_COLUMNS)
        return RollingStats(curves, info, tuple(names), window, step, False)

    frames, info = [], []
    for s in sessions:
        sid = getattr(s, 'session_id', '?')
        try:
            dist = s.distribution
        except Exception:
            dist = None
        try:
            c, v = _roll(TrialArrays.from_sessions([s]), names, window, step, min_short)
        except Exception as exc:
            warnings.warn(f'compute_rolling_stats: session {sid!r} failed ({exc}); emitting empty curve.',
                          RuntimeWarning, stacklevel=2)
            c, v = np.array([]), np.zeros((0, len(names)))
        sidx, stype = getattr(s, 'session_idx', None), getattr(s, 'session_type', '')
        frames.append(_rows(c, v, names, sid, sidx, stype, dist))
        try:
            n = int(len(s.trials.choice))
        except Exception:
            n = 0
        info.append([sid, sidx, stype, dist, n])
    curves = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CURVE_COLUMNS)
    return RollingStats(curves, pd.DataFrame(info, columns=INFO_COLUMNS), tuple(names), window, step, True)
