"""Binned empirical curves along the stimulus axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from behav_utils.data.arrays import TrialArrays
from behav_utils.readouts._base import _ro, bin_centres, bin_index
from behav_utils.tasks import require

Kind = Literal['choice_prob', 'accuracy']


@dataclass(frozen=True)
class BinnedCurve:
    """Per-stimulus-bin mean of either P(B) (``choice_prob``) or P(correct) (``accuracy``)."""

    kind: str
    centres: np.ndarray      # (n_bins,)
    values: np.ndarray       # (n_bins,) NaN where the bin is empty
    counts: np.ndarray       # (n_bins,)

    @property
    def n_bins(self) -> int:
        return int(self.values.shape[0])

    def to_rows(self) -> pd.DataFrame:
        return pd.DataFrame({'bin': np.arange(self.n_bins), 'centre': self.centres,
                             'kind': self.kind, 'value': self.values, 'n': self.counts})


def compute_binned_curve(arrays: TrialArrays, *, kind: Kind = 'choice_prob',
                         n_bins: int = 8) -> BinnedCurve:
    """
    Empirical curve on one block (responded trials only).

    Args:
        arrays: Pre-filtered trials of one block.
        kind: 'choice_prob' for the proportion of B choices per stimulus bin, or 'accuracy' for the
            proportion correct among the trials the task scores.
        n_bins: Stimulus bins on the fixed [-1, 1] grid.

    Returns:
        A ``BinnedCurve``; an empty bin is NaN with a count of 0.

    Raises:
        RequirementError: The task is not binary or has no stimulus axis, or, for 'accuracy', does not
            score trials.
        ValueError: ``kind`` is neither 'choice_prob' nor 'accuracy'.
    """
    needs = ('binary', 'stimulus_axis') + (('scored',) if kind == 'accuracy' else ())
    require(arrays.task, needs, 'compute_binned_curve')
    if kind not in ('choice_prob', 'accuracy'):
        raise ValueError(f"kind must be 'choice_prob' or 'accuracy', got {kind!r}")
    v = arrays.valid()
    target = v.choice if kind == 'choice_prob' else v.correct
    scored = ~np.isnan(target)                 # accuracy: trials the task scores
    target = target[scored]
    idx = bin_index(v.stimulus[scored], n_bins)
    values = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b] > 0:
            values[b] = np.mean(target[m])
    return BinnedCurve(kind, _ro(bin_centres(n_bins)), _ro(values), _ro(counts))
