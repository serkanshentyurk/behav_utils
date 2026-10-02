"""Stimulus binning shared by the stats that bin on the stimulus axis."""

from __future__ import annotations

import numpy as np

STIMULUS_RANGE = (-1.0, 1.0)
DEFAULT_N_BINS = 8
HARD_THRESHOLD = 0.3          # |stimulus| below this is a "hard" trial


def bin_edges(n_bins: int = DEFAULT_N_BINS) -> np.ndarray:
    return np.linspace(STIMULUS_RANGE[0], STIMULUS_RANGE[1], n_bins + 1)


def bin_index(stimulus: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> np.ndarray:
    """Bin index in [0, n_bins) for each stimulus (clipped at the range ends)."""
    return np.clip(np.digitize(stimulus, bin_edges(n_bins)) - 1, 0, n_bins - 1)


def bin_midpoints(n_bins: int = DEFAULT_N_BINS) -> np.ndarray:
    e = bin_edges(n_bins)
    return (e[:-1] + e[1:]) / 2
