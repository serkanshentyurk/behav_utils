"""Serial-dependence profile: raw-proportion shift per previous-stimulus bin.

The fast, fit-free counterpart of the update matrix, for pipelines that
evaluate it thousands of times (SBI, HMM). Shifts by array adjacency over the
responded trials, so it bridges block seams on pooled data — descriptive use
and single blocks only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from behav_utils.data.arrays import TrialArrays
from behav_utils.readouts._base import _ro, bin_centres, bin_index

MIN_TRIALS = 50
MIN_POST_CORRECT = 30
MIN_PER_PREV_BIN = 5
MIN_PER_CELL = 3
MIN_CELLS = 3


@dataclass(frozen=True)
class SerialDependenceProfile:
    """``profile[j]`` = mean over current bins i of P(B | prev j, cur i) − P(B | cur i), post-correct pairs."""

    centres: np.ndarray      # (n_bins,)
    profile: np.ndarray      # (n_bins,) NaN where under-sampled
    marginal: np.ndarray     # (n_bins,) P(B | cur bin) over post-correct pairs
    n_pairs: int

    @property
    def n_bins(self) -> int:
        return int(self.profile.shape[0])

    def to_rows(self) -> pd.DataFrame:
        return pd.DataFrame({'prev_bin': np.arange(self.n_bins), 'prev_centre': self.centres,
                             'shift': self.profile})


def _empty(n_bins: int) -> SerialDependenceProfile:
    nan = np.full(n_bins, np.nan)
    return SerialDependenceProfile(_ro(bin_centres(n_bins)), _ro(nan), _ro(nan), 0)


def compute_sd_profile(arrays: TrialArrays, *, n_bins: int = 8) -> SerialDependenceProfile:
    v = arrays.valid()
    if v.n_trials < MIN_TRIALS:
        return _empty(n_bins)
    reward = (v.choice == v.category).astype(float)
    post_correct = reward[:-1] == 1
    if post_correct.sum() < MIN_POST_CORRECT:
        return _empty(n_bins)
    prev_bin = bin_index(v.stimulus[:-1][post_correct], n_bins)
    cur_bin = bin_index(v.stimulus[1:][post_correct], n_bins)
    cur_choice = v.choice[1:][post_correct]

    marginal = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = cur_bin == i
        if m.sum() > 0:
            marginal[i] = np.mean(cur_choice[m])

    profile = np.full(n_bins, np.nan)
    for j in range(n_bins):
        pm = prev_bin == j
        if pm.sum() < MIN_PER_PREV_BIN:
            continue
        deltas = []
        for i in range(n_bins):
            cell = pm & (cur_bin == i)
            if cell.sum() >= MIN_PER_CELL and not np.isnan(marginal[i]):
                deltas.append(np.mean(cur_choice[cell]) - marginal[i])
        if len(deltas) >= MIN_CELLS:
            profile[j] = np.mean(deltas)
    return SerialDependenceProfile(_ro(bin_centres(n_bins)), _ro(profile), _ro(marginal),
                                   int(post_correct.sum()))
