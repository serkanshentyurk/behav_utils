"""stats/sdt: d′ and criterion with B as the signal."""

import numpy as np

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats import SDT, compute_stats, list_stats


def _arrays(p_b_given_b, p_b_given_a, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 2, n)
    p = np.where(cat == 1, p_b_given_b, p_b_given_a)
    choice = (rng.random(n) < p).astype(float)
    stim = np.where(cat == 1, 0.5, -0.5) + rng.normal(0, 0.1, n)
    return TrialArrays.from_sequence(choice, stim, cat.astype(float))


def test_registered():
    assert set(SDT) <= set(list_stats())


def test_unbiased_observer_has_zero_criterion():
    s = compute_stats(_arrays(0.84, 0.16), list(SDT))
    assert abs(s['criterion']) < 0.1
    assert 1.6 < s['dprime'] < 2.4          # z(.84) − z(.16) ≈ 2


def test_bias_toward_a_is_positive_criterion():
    s = compute_stats(_arrays(0.6, 0.05), list(SDT))         # rarely says B
    assert s['criterion'] > 0.3
    t = compute_stats(_arrays(0.95, 0.4), list(SDT))         # readily says B
    assert t['criterion'] < -0.3


def test_sensitivity_independent_of_bias():
    a = compute_stats(_arrays(0.84, 0.16), list(SDT))
    b = compute_stats(_arrays(0.95, 0.40), list(SDT))        # same d′ ≈ 2 at a different criterion
    assert abs(a['dprime'] - b['dprime']) < 0.35


def test_too_few_trials_is_nan():
    s = compute_stats(_arrays(0.8, 0.2, n=6), list(SDT))
    assert np.isnan(s['dprime']) and np.isnan(s['criterion'])
