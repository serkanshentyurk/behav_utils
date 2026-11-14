"""
The continuous-response family's readouts: the response curve, the error distribution and the
serial-bias curve, on responded trials that have a truth. Each needs ``continuous`` and draws with its
own ``plot_x`` in ``behav_utils.plotting.continuous``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from behav_utils.data.arrays import TrialArrays
from behav_utils.readouts._base import _ro, bin_centres, bin_index
from behav_utils.tasks import require

_NEEDS = ('continuous',)


def _binned(x: np.ndarray, y: np.ndarray, idx: np.ndarray, n_bins: int) -> tuple:
    """Per bin: mean of y, its SEM (NaN with fewer than two trials) and the count."""
    means, sems = np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        values = y[idx == b]
        counts[b] = values.size
        if values.size:
            means[b] = values.mean()
        if values.size > 1:
            sems[b] = values.std(ddof=1) / np.sqrt(values.size)
    return means, sems, counts


@dataclass(frozen=True)
class ResponseCurve:
    """Mean response per stimulus bin, with its SEM and count. A faithful reproducer lies on the identity
    line; a slope below it is a pull towards the middle."""

    centres: np.ndarray
    values: np.ndarray
    sem: np.ndarray
    counts: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.values.shape[0])

    def to_rows(self) -> pd.DataFrame:
        return pd.DataFrame({'bin': np.arange(self.n_bins), 'centre': self.centres, 'value': self.values,
                             'sem': self.sem, 'n': self.counts})


def compute_response_curve(arrays: TrialArrays, *, n_bins: int = 8) -> ResponseCurve:
    """
    Mean response, on the library's axis, per stimulus bin.

    Args:
        arrays: Pre-filtered trials of one block.
        n_bins: Stimulus bins on the fixed [-1, 1] grid.

    Returns:
        A ``ResponseCurve``.

    Raises:
        RequirementError: The task's responses are not continuous.
    """
    require(arrays.task, _NEEDS, 'compute_response_curve')
    v = arrays.valid()
    ok = ~np.isnan(v.truth) & ~np.isnan(v.choice)
    means, sems, counts = _binned(v.stimulus[ok], v.choice[ok], bin_index(v.stimulus[ok], n_bins), n_bins)
    return ResponseCurve(_ro(bin_centres(n_bins)), _ro(means), _ro(sems), _ro(counts))


@dataclass(frozen=True)
class ErrorDistribution:
    """Signed errors (response − truth): counts and density over symmetric bins, errors beyond the
    outer edges counted in the end bins, with the mean, SD and number of errors."""

    edges: np.ndarray
    counts: np.ndarray
    density: np.ndarray
    mean: float
    sd: float
    n: int

    def to_rows(self) -> pd.DataFrame:
        return pd.DataFrame({'bin': np.arange(len(self.counts)), 'left': self.edges[:-1], 'right': self.edges[1:],
                             'n': self.counts, 'density': self.density})


def compute_error_distribution(arrays: TrialArrays, *, n_bins: int = 20, limit: float = 1.0) -> ErrorDistribution:
    """
    The signed errors binned over [-limit, limit] on the library's axis.

    An error is the response minus the correct response, on trials with both.

    Args:
        arrays: Pre-filtered trials of one block.
        n_bins: Bins over [-limit, limit].
        limit: Half-width of the binned range; errors beyond it count in the outermost bins.

    Returns:
        An ``ErrorDistribution``, with the counts and density per bin and the errors' mean and SD.

    Raises:
        RequirementError: The task's responses are not continuous.
    """
    require(arrays.task, _NEEDS, 'compute_error_distribution')
    v = arrays.valid()
    ok = ~np.isnan(v.truth) & ~np.isnan(v.choice)
    errors = v.choice[ok] - v.truth[ok]
    edges = np.linspace(-limit, limit, n_bins + 1)
    counts = np.bincount(np.clip(np.digitize(errors, edges) - 1, 0, n_bins - 1), minlength=n_bins)
    density = counts / (errors.size * (2 * limit / n_bins)) if errors.size else np.full(n_bins, np.nan)
    return ErrorDistribution(
        _ro(edges), _ro(counts), _ro(density),
        float(errors.mean()) if errors.size else np.nan,
        float(errors.std(ddof=1)) if errors.size > 1 else np.nan,
        int(errors.size),
    )


@dataclass(frozen=True)
class SerialBiasCurve:
    """Mean error per bin of previous − current stimulus, with the current stimulus's linear effect
    removed first (as in the ``serial_bias`` statistic), so a pull towards the middle does not draw a
    false curve. Rising through zero means attraction to the previous stimulus."""

    centres: np.ndarray
    values: np.ndarray
    sem: np.ndarray
    counts: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.values.shape[0])

    def to_rows(self) -> pd.DataFrame:
        return pd.DataFrame({'bin': np.arange(self.n_bins), 'centre': self.centres, 'value': self.values,
                             'sem': self.sem, 'n': self.counts})


def compute_serial_bias(arrays: TrialArrays, *, n_bins: int = 8) -> SerialBiasCurve:
    """
    The serial-bias curve over previous − current stimulus in [-2, 2] on the library's axis.

    The errors are regressed on the difference and on the current stimulus, and the current stimulus's
    part, with the intercept, is removed, so central tendency does not show as serial bias. The curve is
    the mean of what remains per bin of the difference.

    Args:
        arrays: Pre-filtered trials of one block.
        n_bins: Bins over the difference between the previous and the current stimulus, [-2, 2].

    Returns:
        A ``SerialBiasCurve``, with the mean, SEM and count per bin.

    Raises:
        RequirementError: The task's responses are not continuous.
    """
    require(arrays.task, _NEEDS, 'compute_serial_bias')
    p = arrays.lag1_pairs()
    ok = ~np.isnan(p.truth) & ~np.isnan(p.prev_stimulus) & ~np.isnan(p.choice)
    current = p.stimulus[ok]
    delta = p.prev_stimulus[ok] - current
    errors = p.choice[ok] - p.truth[ok]
    design = np.column_stack([np.ones(ok.sum()), delta, current])
    if ok.sum() >= 3 and np.linalg.matrix_rank(design) == 3:
        coef = np.linalg.lstsq(design, errors, rcond=None)[0]
        errors = errors - coef[0] - coef[2] * current          # keep only what previous − current explains
    edges = np.linspace(-2.0, 2.0, n_bins + 1)
    idx = np.clip(np.digitize(delta, edges) - 1, 0, n_bins - 1)
    means, sems, counts = _binned(delta, errors, idx, n_bins)
    return SerialBiasCurve(_ro((edges[:-1] + edges[1:]) / 2), _ro(means), _ro(sems), _ro(counts))
