"""Reaction-time statistics."""

from __future__ import annotations

import numpy as np

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats.registry import stat

# Recording-uncertainty jitter magnitude (ms): each trial's RT gets an
# independent U[0, RT_JITTER_MS) draw added before the median is taken.
RT_JITTER_MS = 150.0


def _recorded(a: TrialArrays) -> np.ndarray:
    rt = a.reaction_time
    return rt[~np.isnan(rt)]


@stat('reaction_time')
def reaction_time(a: TrialArrays, *, rng) -> float:
    """Median response latency (ms) over trials with a recorded RT; NaN if none."""
    rt = _recorded(a)
    return float(np.median(rt)) if rt.size else np.nan


@stat('reaction_time_jitter')
def reaction_time_jitter(a: TrialArrays, *, rng) -> float:
    """Median RT (ms) after an independent U[0, RT_JITTER_MS) draw per trial.

    Re-drawn on every call, so under bootstrap the recording slop folds into
    the interval: a condition difference barely moves while its CI widens.
    """
    rt = _recorded(a)
    if not rt.size:
        return np.nan
    return float(np.median(rt + rng.uniform(0.0, RT_JITTER_MS, size=rt.size)))
