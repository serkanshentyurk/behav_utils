"""Signal-detection view of a 2-AFC block: sensitivity and criterion, with B as the signal."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats.registry import fit

SDT = ('dprime', 'criterion')


def _rates(a: TrialArrays):
    v = a.valid()
    is_b = v.category == 1
    n_b, n_a = int(is_b.sum()), int((~is_b).sum())
    if n_b == 0 or n_a == 0:
        return np.nan, np.nan, n_b, n_a
    # log-linear correction (Hautus 1995): keeps 0 and 1 rates finite
    hit = (float(np.sum(v.choice[is_b] == 1)) + 0.5) / (n_b + 1)
    fa = (float(np.sum(v.choice[~is_b] == 1)) + 0.5) / (n_a + 1)
    return hit, fa, n_b, n_a


@fit('sdt', outputs=SDT, exchangeable=True)
def sdt(a: TrialArrays, *, rng):
    """Equal-variance signal-detection summary with category B as the signal and "choose B" as a yes:
    hit = P(choose B | B), false alarm = P(choose B | A), both log-linear corrected.
    ``dprime`` = z(hit) − z(fa) — discriminability independent of bias; ``criterion`` c = −(z(hit) + z(fa)) / 2 —
    positive = conservative about B, i.e. biased toward A. A manipulation that moves the criterion without
    moving d′ shifts the decision rule, not the evidence."""
    hit, fa, n_b, n_a = _rates(a)
    if not np.isfinite(hit) or min(n_b, n_a) < 5:
        return np.nan, np.nan
    zh, zf = norm.ppf(hit), norm.ppf(fa)
    return float(zh - zf), float(-(zh + zf) / 2)
