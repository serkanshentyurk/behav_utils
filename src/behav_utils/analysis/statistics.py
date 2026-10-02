"""
compute_phase_stats — scalar statistics on a phase, pooled and (optionally) per session.

    load_experiment → select_sessions → filter_trials → phase
                                                          ↓
                                     compute_phase_stats(phase, names, per_session=...)

    r = compute_phase_stats(phase, ['accuracy', 'side_bias', *PSYCHOMETRIC])
    r.pooled['side_bias']          # trial-weighted pooled estimate (pd.Series)
    r.sessions                     # None unless per_session=True

    r = compute_phase_stats(phase, ['accuracy', 'mu'], per_session=True)
    r.sessions                     # tidy DataFrame: animal, session, stat, value, n_trials

Invariant: ``pooled`` is ALWAYS the genuine pooled fit — one fit on all trials,
trial-weighted — never the mean of the per-session values. A short, atypical
session gets equal weight in a session-mean but only its trial-share when
pooled. ``per_session`` adds the breakdown; it never changes ``pooled``.

Scope: scalar statistics only (see ``behav_utils.stats``). The psychometric
curve and the update matrix are readouts (``behav_utils.readouts``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats import compute_stats, validate_names

__all__ = ['PhaseStats', 'compute_phase_stats', 'infer_animal_id', 'SESSION_COLUMNS']

SESSION_COLUMNS = ['animal', 'session', 'session_type', 'distribution', 'stat', 'value', 'n_trials']


@dataclass(frozen=True)
class PhaseStats:
    """Scalar statistics of one phase.

    ``pooled`` is a float Series indexed by stat name; ``sessions`` is the tidy
    per-session frame (``SESSION_COLUMNS``: animal, session, session_type,
    distribution, stat, value, n_trials) or None. ``n_trials`` counts responded
    trials.
    """

    pooled: pd.Series
    sessions: pd.DataFrame | None
    animal: str
    n_sessions: int
    n_trials: int

    @property
    def names(self) -> list:
        return list(self.pooled.index)

    def to_rows(self) -> pd.DataFrame:
        """Pooled values as tidy rows (session='pooled'), same columns as ``sessions``."""
        return pd.DataFrame({
            'animal': self.animal, 'session': 'pooled', 'session_type': '', 'distribution': None,
            'stat': self.names, 'value': self.pooled.to_numpy(), 'n_trials': self.n_trials,
        }, columns=SESSION_COLUMNS)

    def __repr__(self) -> str:
        return (f'PhaseStats(animal={self.animal!r}, n_sessions={self.n_sessions}, '
                f'n_trials={self.n_trials}, stats={self.names}, '
                f'per_session={self.sessions is not None})')


def infer_animal_id(phase) -> str:
    """Animal id from the first session that carries one ('unknown' otherwise)."""
    for session in phase:
        meta = getattr(session, 'filter_info', None) or {}
        selection = meta.get('selection', {}) if isinstance(meta, dict) else {}
        if selection.get('animal_id'):
            return selection['animal_id']
        aid = getattr(session, 'animal_id', None)
        if aid:
            return aid
    return 'unknown'


def _distribution(session):
    try:
        return session.distribution
    except Exception:
        return None


def _session_label(session):
    return getattr(session, 'session_idx', getattr(session, 'session_id', None))


def compute_phase_stats(
    phase,
    names: Sequence[str],
    *,
    per_session: bool = False,
    animal_id: str | None = None,
    rng: np.random.Generator | None = None,
) -> PhaseStats:
    """Scalar statistics for a phase.

    Args:
        phase:       list of SessionData — the output of ``filter_trials``.
        names:       scalar stat names (``list_stats()``); splat ``PSYCHOMETRIC``
                     for the four psychometric parameters.
        per_session: also fit each session separately and fill ``sessions``.
        animal_id:   stamped into the frames; inferred from the phase if None.
        rng:         generator for stochastic stats (seed it for reproducibility).

    Returns:
        :class:`PhaseStats`.
    """
    names = validate_names(names)
    if animal_id is None:
        animal_id = infer_animal_id(phase)
    if rng is None:
        rng = np.random.default_rng()
    phase = list(phase)

    if phase:
        arrays = TrialArrays.from_sessions(phase)
        pooled = compute_stats(arrays, names, rng=rng)
        n_trials = arrays.n_responded
    else:
        pooled = pd.Series(np.nan, index=list(names), dtype=float, name='value')
        n_trials = 0

    sessions = None
    if per_session:
        frames = []
        for session in phase:
            a = TrialArrays.from_sessions([session])
            values = compute_stats(a, names, rng=rng)
            frames.append(pd.DataFrame({
                'animal': animal_id, 'session': _session_label(session),
                'session_type': getattr(session, 'session_type', ''),
                'distribution': _distribution(session),
                'stat': names, 'value': values.to_numpy(), 'n_trials': a.n_responded,
            }, columns=SESSION_COLUMNS))
        sessions = (pd.concat(frames, ignore_index=True) if frames
                    else pd.DataFrame(columns=SESSION_COLUMNS))

    return PhaseStats(pooled, sessions, animal_id, len(phase), int(n_trials))
