"""Memoryless performance and bias statistics (no trial history)."""

from __future__ import annotations

import numpy as np

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats._binning import DEFAULT_N_BINS, HARD_THRESHOLD, bin_index
from behav_utils.stats.registry import stat


@stat('accuracy')
def accuracy(a: TrialArrays, *, rng) -> float:
    """Proportion correct over responded trials."""
    v = a.valid()
    if v.n_trials == 0:
        return np.nan
    return float(np.mean(v.choice == v.category))


@stat('side_bias')
def side_bias(a: TrialArrays, *, rng) -> float:
    """P(choose B) − 0.5. Positive = biased toward B."""
    v = a.valid()
    if v.n_trials == 0:
        return np.nan
    return float(np.mean(v.choice) - 0.5)


@stat('stimulus_sensitivity')
def stimulus_sensitivity(a: TrialArrays, *, rng) -> float:
    """Pearson correlation between stimulus value and choice."""
    v = a.valid()
    if v.n_trials < 10:
        return np.nan
    if np.std(v.choice) == 0 or np.std(v.stimulus) == 0:
        return np.nan
    return float(np.corrcoef(v.choice, v.stimulus)[0, 1])


@stat('choice_entropy')
def choice_entropy(a: TrialArrays, *, rng) -> float:
    """Mean over stimulus bins of the binary entropy of choice, in bits ([0, 1])."""
    v = a.valid()
    if v.n_trials < 10:
        return np.nan
    idx = bin_index(v.stimulus, DEFAULT_N_BINS)
    entropies = []
    for b in range(DEFAULT_N_BINS):
        m = idx == b
        if m.sum() < 3:
            continue
        p = np.clip(np.mean(v.choice[m]), 1e-10, 1 - 1e-10)
        entropies.append(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
    return float(np.mean(entropies)) if entropies else np.nan


def _split_hard_easy(v: TrialArrays):
    hard = np.abs(v.stimulus) < HARD_THRESHOLD
    return hard, ~hard


@stat('hard_accuracy')
def hard_accuracy(a: TrialArrays, *, rng) -> float:
    """Accuracy on |stimulus| < HARD_THRESHOLD."""
    v = a.valid()
    if v.n_trials < 10:
        return np.nan
    hard, _ = _split_hard_easy(v)
    if hard.sum() < 3:
        return np.nan
    return float(np.mean(v.choice[hard] == v.category[hard]))


@stat('easy_accuracy')
def easy_accuracy(a: TrialArrays, *, rng) -> float:
    """Accuracy on |stimulus| >= HARD_THRESHOLD."""
    v = a.valid()
    if v.n_trials < 10:
        return np.nan
    _, easy = _split_hard_easy(v)
    if easy.sum() < 3:
        return np.nan
    return float(np.mean(v.choice[easy] == v.category[easy]))


@stat('hard_easy_ratio')
def hard_easy_ratio(a: TrialArrays, *, rng) -> float:
    """hard_accuracy / easy_accuracy. NaN if either side has < 3 trials or easy accuracy ≈ 0."""
    v = a.valid()
    if v.n_trials < 10:
        return np.nan
    hard, easy = _split_hard_easy(v)
    if hard.sum() < 3 or easy.sum() < 3:
        return np.nan
    acc_hard = np.mean(v.choice[hard] == v.category[hard])
    acc_easy = np.mean(v.choice[easy] == v.category[easy])
    if acc_easy < 0.01:
        return np.nan
    return float(acc_hard / acc_easy)
