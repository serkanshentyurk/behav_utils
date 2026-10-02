"""
Readouts: array-valued descriptions of a block of trials.

A readout is ``compute_x(arrays: TrialArrays, ...) -> XResult`` where
``XResult`` is a frozen dataclass with:

- ``to_rows() -> pd.DataFrame``  tidy long form, one value per row
- exactly one draw-only ``plot_x(result, ax=None)`` in ``behav_utils.plotting``

Per-session readouts are the caller's loop; where a reduction across blocks
is meaningful the dataclass has a classmethod for it (``UpdateMatrix.average``).

Readouts never import ``behav_utils.stats``; stats may import readouts.
"""

from __future__ import annotations

import numpy as np

STIMULUS_RANGE = (-1.0, 1.0)
X_FIT = np.linspace(STIMULUS_RANGE[0], STIMULUS_RANGE[1], 200)
X_FIT.setflags(write=False)


def bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(STIMULUS_RANGE[0], STIMULUS_RANGE[1], n_bins + 1)


def bin_centres(n_bins: int) -> np.ndarray:
    e = bin_edges(n_bins)
    return (e[:-1] + e[1:]) / 2


def bin_index(stimulus: np.ndarray, n_bins: int) -> np.ndarray:
    """Clipped digitize on the fixed grid — the same rule everywhere."""
    return np.clip(np.digitize(stimulus, bin_edges(n_bins)) - 1, 0, n_bins - 1)


def _ro(a) -> np.ndarray:
    """Read-only copy (dataclass fields are frozen but arrays are not). Ints/bools keep their dtype."""
    a = np.array(a)
    if a.dtype.kind not in 'iub':
        a = a.astype(float)
    a.setflags(write=False)
    return a
