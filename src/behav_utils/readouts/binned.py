"""Binned empirical curves along the stimulus axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from behav_utils.data.arrays import TrialArrays
from behav_utils.readouts._base import _ro, bin_centres, bin_index

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
    """Empirical curve on one block (responded trials only)."""
    if kind not in ('choice_prob', 'accuracy'):
        raise ValueError(f"kind must be 'choice_prob' or 'accuracy', got {kind!r}")
    v = arrays.valid()
    target = v.choice if kind == 'choice_prob' else (v.choice == v.category).astype(float)
    idx = bin_index(v.stimulus, n_bins)
    values = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b] > 0:
            values[b] = np.mean(target[m])
    return BinnedCurve(kind, _ro(bin_centres(n_bins)), _ro(values), _ro(counts))
