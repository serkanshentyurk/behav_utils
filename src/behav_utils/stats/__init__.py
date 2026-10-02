"""
Scalar statistics on trial data.

    from behav_utils.stats import compute_stats, PSYCHOMETRIC
    from behav_utils.data.arrays import TrialArrays

    arrays = TrialArrays.from_sessions(sessions)
    s = compute_stats(arrays, [*PSYCHOMETRIC, 'accuracy', 'side_bias'])
    s['mu'], s['accuracy']          # a float pd.Series, request order

Every statistic is a float. Multi-output fits (a psychometric fit, the
logistic history regression, the serial-dependence profile) expose each
output as its own name and run once per call. Array-valued readouts (update
matrix, conditional psychometrics, binned curves) are not statistics and live
in ``behav_utils.readouts``.

Importing this package registers all built-in producers.
"""

from behav_utils.stats import basic, dynamics, history, psychometric, rt, sdt  # noqa: F401  (registration)
from behav_utils.stats.dynamics import PSE_DYNAMICS
from behav_utils.stats.history import LOGISTIC_HISTORY, SD_PROFILE
from behav_utils.stats.psychometric import PSYCHOMETRIC
from behav_utils.stats.registry import (
    compute_stats,
    fit,
    is_exchangeable,
    list_producers,
    list_stats,
    producer_of,
    stat,
    validate_names,
)
from behav_utils.stats.sdt import SDT

__all__ = [
    'compute_stats', 'stat', 'fit',
    'list_stats', 'list_producers', 'producer_of', 'is_exchangeable', 'validate_names',
    'PSYCHOMETRIC', 'LOGISTIC_HISTORY', 'SD_PROFILE', 'PSE_DYNAMICS', 'SDT',
]
