"""Memoryless performance and bias statistics (no trial history)."""

from __future__ import annotations

import numpy as np

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats._binning import DEFAULT_N_BINS, HARD_THRESHOLD, bin_index
from behav_utils.stats.registry import stat


def _mean_scored(score: np.ndarray) -> float:
    """Mean of the task's 1/0/NaN score over the trials that have one (NaN if none do)."""
    scored = score[~np.isnan(score)]
    return float(np.mean(scored)) if scored.size else np.nan

@stat('accuracy', requires=('scored',))
def accuracy(a: TrialArrays, *, rng) -> float:
    """Proportion correct over responded trials."""
    v = a.valid()
    if v.n_trials == 0:
        return np.nan
    return _mean_scored(v.correct)


@stat('side_bias', requires=('two_choices',), instead={'withhold': 'go_bias'})
def side_bias(a: TrialArrays, *, rng) -> float:
    """P(choose B) − 0.5. Positive = biased toward B."""
    v = a.valid()
    if v.n_trials == 0:
        return np.nan
    return float(np.mean(v.choice) - 0.5)


@stat('accuracy_b_minus_a', requires=('two_choices', 'scored'), instead={'withhold': 'hit_minus_cr_rate'})
def accuracy_b_minus_a(a: TrialArrays, *, rng) -> float:
    """Accuracy on category-B trials minus accuracy on category-A trials. Positive = more accurate on B.

    Unlike ``side_bias``, it does not depend on how often each category is presented (bias-correcting
    trial selection, an asymmetric stimulus distribution); it does change when the two categories differ
    in difficulty. NaN if either category has fewer than 3 responded trials.
    """
    v = a.valid()
    on_b, on_a = v.category == 1, v.category == 0
    if on_b.sum() < 3 or on_a.sum() < 3:
        return np.nan
    return float(_mean_scored(v.correct[on_b]) - _mean_scored(v.correct[on_a]))


@stat('go_bias', requires=('withhold',), instead={'two_choices': 'side_bias'})
def go_bias(a: TrialArrays, *, rng) -> float:
    """Go/no-go: go rate − 0.5 over all non-aborted trials (a withhold is a choice of 0)."""
    return side_bias(a, rng=rng)


@stat('hit_minus_cr_rate', requires=('withhold', 'scored'), instead={'two_choices': 'accuracy_b_minus_a'})
def hit_minus_cr_rate(a: TrialArrays, *, rng) -> float:
    """Go/no-go: hit rate (correct on go trials) minus correct-rejection rate (correct on no-go trials)."""
    return accuracy_b_minus_a(a, rng=rng)


@stat('stimulus_sensitivity', requires=('binary', 'stimulus_axis'))
def stimulus_sensitivity(a: TrialArrays, *, rng) -> float:
    """Pearson correlation between stimulus value and choice."""
    v = a.valid()
    if v.n_trials < 10:
        return np.nan
    if np.std(v.choice) == 0 or np.std(v.stimulus) == 0:
        return np.nan
    return float(np.corrcoef(v.choice, v.stimulus)[0, 1])


@stat('choice_entropy', requires=('binary', 'stimulus_axis'))
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


@stat('hard_accuracy', requires=('scored', 'stimulus_axis'))
def hard_accuracy(a: TrialArrays, *, rng) -> float:
    """Accuracy on |stimulus| < HARD_THRESHOLD."""
    v = a.valid()
    if v.n_trials < 10:
        return np.nan
    hard, _ = _split_hard_easy(v)
    if hard.sum() < 3:
        return np.nan
    return _mean_scored(v.correct[hard])


@stat('easy_accuracy', requires=('scored', 'stimulus_axis'))
def easy_accuracy(a: TrialArrays, *, rng) -> float:
    """Accuracy on |stimulus| >= HARD_THRESHOLD."""
    v = a.valid()
    if v.n_trials < 10:
        return np.nan
    _, easy = _split_hard_easy(v)
    if easy.sum() < 3:
        return np.nan
    return _mean_scored(v.correct[easy])


@stat('hard_easy_ratio', requires=('scored', 'stimulus_axis'))
def hard_easy_ratio(a: TrialArrays, *, rng) -> float:
    """
    hard_accuracy / easy_accuracy. NaN if either side has < 3 trials or easy accuracy ≈ 0.

    Defined for this library; it follows no published measure.
    """
    v = a.valid()
    if v.n_trials < 10:
        return np.nan
    hard, easy = _split_hard_easy(v)
    if hard.sum() < 3 or easy.sum() < 3:
        return np.nan
    acc_hard = _mean_scored(v.correct[hard])
    acc_easy = _mean_scored(v.correct[easy])
    if acc_easy < 0.01:
        return np.nan
    return float(acc_hard / acc_easy)
