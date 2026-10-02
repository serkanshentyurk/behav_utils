"""Update matrix: how the previous stimulus shifts the current psychometric curve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.data.arrays import TrialArrays
from behav_utils.readouts._base import _ro, bin_centres, bin_edges, bin_index

MIN_PAIRS_PER_BIN = 10
TrialFilter = Literal['post_correct', 'all']


@dataclass(frozen=True)
class UpdateMatrix:
    """(n_bins × n_bins) shift in P(B). Row = current-stimulus bin, column = previous-stimulus bin.

    ``matrix[i, j]``  = P(B | cur bin i, prev bin j) − P(B | cur bin i), from a
    per-previous-bin cumulative-Gaussian fit minus the unconditional fit, both
    evaluated at the bin centres. NaN columns had fewer than
    ``MIN_PAIRS_PER_BIN`` pairs or a failed fit.
    """

    matrix: np.ndarray            # (n_bins, n_bins) shift in P(B)
    conditional: np.ndarray       # (n_bins, n_bins) conditional P(B)
    unconditional: np.ndarray     # (n_bins,) unconditional curve at the centres
    n_pairs: np.ndarray           # (n_bins,) pairs per previous-stimulus bin
    n_trials: int                 # pairs entering the unconditional fit
    trial_filter: str
    n_sources: int = 1            # blocks averaged into this matrix (1 = single fit)
    coverage: np.ndarray | None = None   # (n_bins, n_bins) sources per cell, when averaged
    sem: np.ndarray | None = None        # (n_bins, n_bins), when averaged

    def __repr__(self) -> str:
        finite = int(np.isfinite(self.matrix).sum())
        return (f'UpdateMatrix(n_bins={self.n_bins}, trial_filter={self.trial_filter!r}, '
                f'n_trials={self.n_trials}, n_sources={self.n_sources}, '
                f'finite_cells={finite}/{self.matrix.size})')

    @property
    def n_bins(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def centres(self) -> np.ndarray:
        return bin_centres(self.n_bins)

    @property
    def edges(self) -> np.ndarray:
        return bin_edges(self.n_bins)

    def profile(self) -> np.ndarray:
        """(n_bins,) mean shift per previous-stimulus bin (nan-aware column mean)."""
        with np.errstate(invalid='ignore'):
            return np.nanmean(self.matrix, axis=0)

    def to_rows(self) -> pd.DataFrame:
        """Tidy: one row per cell — cur_bin, prev_bin, cur_centre, prev_centre, shift, conditional."""
        n = self.n_bins
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        c = self.centres
        return pd.DataFrame({
            'cur_bin': i.ravel(), 'prev_bin': j.ravel(),
            'cur_centre': c[i.ravel()], 'prev_centre': c[j.ravel()],
            'shift': self.matrix.ravel(), 'conditional': self.conditional.ravel(),
            'n_pairs': self.n_pairs[j.ravel()],
        })

    @classmethod
    def from_matrix(cls, matrix, *, trial_filter: str = 'post_correct') -> UpdateMatrix:
        """Wrap a raw shift matrix (e.g. a model-generated one) for plotting; other fields are NaN/0."""
        m = np.asarray(matrix, dtype=float)
        n = m.shape[0]
        return cls(_ro(m), _ro(np.full_like(m, np.nan)), _ro(np.full(n, np.nan)),
                   _ro(np.zeros(n, dtype=int)), 0, trial_filter)

    @classmethod
    def average(cls, items: Sequence[UpdateMatrix], *, min_sources: int = 1) -> UpdateMatrix:
        """Cell-wise mean over matrices (equal weight per item; the item is the unit).

        Valid because every matrix bins on the same fixed grid. Cells backed by
        fewer than ``min_sources`` finite items are NaN.
        """
        items = list(items)
        if not items:
            raise ValueError('UpdateMatrix.average: nothing to average')
        shapes = {m.matrix.shape for m in items}
        if len(shapes) != 1:
            raise ValueError(f'UpdateMatrix.average: mismatched n_bins: {shapes}')
        stack = np.stack([m.matrix for m in items])
        cond = np.stack([m.conditional for m in items])
        coverage = np.isfinite(stack).sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.nanmean(stack, axis=0)
            sem = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(np.maximum(coverage, 1))
            cond_mean = np.nanmean(cond, axis=0)
            uncond = np.nanmean(np.stack([m.unconditional for m in items]), axis=0)
        keep = coverage >= min_sources
        return cls(
            matrix=_ro(np.where(keep, mean, np.nan)),
            conditional=_ro(cond_mean),
            unconditional=_ro(uncond),
            n_pairs=_ro(np.sum([m.n_pairs for m in items], axis=0)),
            n_trials=int(sum(m.n_trials for m in items)),
            trial_filter=items[0].trial_filter,
            n_sources=len(items),
            coverage=_ro(coverage),
            sem=_ro(np.where(keep, sem, np.nan)),
        )


def _empty(n_bins: int, trial_filter: str) -> UpdateMatrix:
    nan = np.full((n_bins, n_bins), np.nan)
    return UpdateMatrix(_ro(nan), _ro(nan), _ro(np.full(n_bins, np.nan)),
                        _ro(np.zeros(n_bins)), 0, trial_filter)


def compute_update_matrix(
    arrays: TrialArrays,
    *,
    n_bins: int = 8,
    trial_filter: TrialFilter = 'post_correct',
) -> UpdateMatrix:
    """Update matrix on one block of trials, using the frozen lag-1 view.

    Args:
        arrays:       pre-filtered trials (may be a non-consecutive subset such
                      as opto-only; the predecessor comes from ``prev_*``, never
                      from array adjacency).
        n_bins:       stimulus bins on the fixed [-1, 1] grid.
        trial_filter: 'post_correct' keeps pairs whose predecessor was rewarded;
                      'all' keeps every responded pair.
    """
    if trial_filter not in ('post_correct', 'all'):
        raise ValueError(f"trial_filter must be 'post_correct' or 'all', got {trial_filter!r}")
    if arrays.n_trials == 0:
        return _empty(n_bins, trial_filter)

    base = arrays.responded & arrays.has_prev
    if trial_filter == 'post_correct':
        base &= arrays.prev_reward == 1
    centres = bin_centres(n_bins)
    prev_bin = bin_index(arrays.prev_stimulus, n_bins)

    stim, choice = arrays.stimulus[base], arrays.choice[base]
    total = fit_psychometric(stim, choice, centres)
    uncond = total['y_fit'] if total['success'] else np.full(n_bins, np.nan)

    matrix = np.full((n_bins, n_bins), np.nan)
    conditional = np.full((n_bins, n_bins), np.nan)
    n_pairs = np.zeros(n_bins, dtype=int)
    for j in range(n_bins):
        m = base & (prev_bin == j)
        n_pairs[j] = int(m.sum())
        if n_pairs[j] < MIN_PAIRS_PER_BIN:
            continue
        fitj = fit_psychometric(arrays.stimulus[m], arrays.choice[m], centres)
        if fitj['success']:
            conditional[:, j] = fitj['y_fit']
            matrix[:, j] = fitj['y_fit'] - uncond

    return UpdateMatrix(_ro(matrix), _ro(conditional), _ro(uncond), _ro(n_pairs),
                        int(len(stim)), trial_filter)
