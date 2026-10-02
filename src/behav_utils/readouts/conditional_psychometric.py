"""Conditional psychometrics: one cumulative-Gaussian fit per previous-stimulus bin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.data.arrays import TrialArrays
from behav_utils.readouts._base import _ro, bin_centres, bin_index
from behav_utils.readouts.psychometric import PARAMS

MIN_TRIALS = 50
SLOPE_THRESHOLD = 5.0
MU_BOUND = 0.99


@dataclass(frozen=True)
class ConditionalPsychometric:
    """(n_bins × 4) psychometric parameters conditioned on the previous-stimulus bin.

    A bin with fewer than ``min_per_bin`` pairs, a failed fit, or an unreliable
    fit (sigma > SLOPE_THRESHOLD or |mu| > MU_BOUND) falls back to the
    unconditional parameters; ``fell_back`` records which. mu/sigma fall back
    together, lapses only on failure — matching the legacy stat exactly.
    """

    params: np.ndarray            # (n_bins, 4) in PARAMS order
    unconditional: np.ndarray     # (4,)
    n_per_bin: np.ndarray         # (n_bins,)
    fell_back: np.ndarray         # (n_bins,) bool
    n_trials: int
    success: bool

    def __repr__(self) -> str:
        return (f'ConditionalPsychometric(n_bins={self.n_bins}, n_trials={self.n_trials}, '
                f'fitted_bins={int((~self.fell_back).sum())}, success={self.success})')

    @property
    def n_bins(self) -> int:
        return int(self.params.shape[0])

    @property
    def centres(self) -> np.ndarray:
        return bin_centres(self.n_bins)

    def to_rows(self) -> pd.DataFrame:
        """Tidy: one row per (prev_bin, param)."""
        n = self.n_bins
        return pd.DataFrame({
            'prev_bin': np.repeat(np.arange(n), 4),
            'prev_centre': np.repeat(self.centres, 4),
            'param': list(PARAMS) * n,
            'value': self.params.ravel(),
            'n_pairs': np.repeat(self.n_per_bin, 4),
            'fell_back': np.repeat(self.fell_back, 4),
        })

    def to_flat(self) -> pd.Series:
        """Legacy flat names ``cond_<param>_<bin>``, for feature vectors."""
        idx = [f'cond_{p}_{b}' for b in range(self.n_bins) for p in PARAMS]
        return pd.Series(self.params.ravel(), index=idx, dtype=float)


def _nan(n_bins: int, n: int) -> ConditionalPsychometric:
    return ConditionalPsychometric(_ro(np.full((n_bins, 4), np.nan)), _ro(np.full(4, np.nan)),
                                   _ro(np.zeros(n_bins)), _ro(np.ones(n_bins, dtype=bool)),
                                   n, False)


def compute_conditional_psychometric(
    arrays: TrialArrays,
    *,
    n_bins: int = 8,
    min_per_bin: int = 15,
) -> ConditionalPsychometric:
    """Per-previous-stimulus-bin psychometric fits on one block."""
    v = arrays.valid()
    if v.n_trials < MIN_TRIALS:
        return _nan(n_bins, v.n_trials)
    uncond = fit_psychometric(v.stimulus, v.choice)
    if not uncond.get('success', False):
        return _nan(n_bins, v.n_trials)
    u = np.array([uncond[k] for k in PARAMS], dtype=float)

    p = arrays.lag1_pairs()
    prev_bin = bin_index(p.prev_stimulus, n_bins)
    params = np.tile(u, (n_bins, 1))
    n_per_bin = np.zeros(n_bins, dtype=int)
    fell_back = np.ones(n_bins, dtype=bool)
    for b in range(n_bins):
        m = prev_bin == b
        n_per_bin[b] = int(m.sum())
        if n_per_bin[b] < min_per_bin:
            continue
        fit = fit_psychometric(p.stimulus[m], p.choice[m])
        if not fit.get('success', False):
            continue
        unreliable = fit['sigma'] > SLOPE_THRESHOLD or abs(fit['mu']) > MU_BOUND
        if not unreliable:
            params[b, 0], params[b, 1] = fit['mu'], fit['sigma']
            fell_back[b] = False
        params[b, 2], params[b, 3] = fit['lapse_low'], fit['lapse_high']

    return ConditionalPsychometric(_ro(params), _ro(u), _ro(n_per_bin), fell_back, v.n_trials, True)
