"""
Scalar statistics on trial data.

``compute_stats(arrays, names)`` returns a float ``pd.Series`` in request order, for example
``compute_stats(arrays, [*PSYCHOMETRIC, 'accuracy'])``; the statistics reference lists every name.

Every statistic is a float. Multi-output fits (a psychometric fit, the
logistic history regression, the serial-dependence profile) expose each
output as its own name and run once per call. Array-valued readouts (update
matrix, conditional psychometrics, binned curves) are not statistics and live
in ``behav_utils.readouts``.

Importing this package registers all built-in producers.
"""

from behav_utils.stats import (  # noqa: F401  (registration)
    basic,
    continuous,
    dynamics,
    history,
    psychometric,
    rt,
    sdt,
)
from behav_utils.stats.dynamics import PSE_DYNAMICS, fit_pse_dynamics
from behav_utils.stats.history import LOGISTIC_HISTORY, SD_PROFILE
from behav_utils.stats.psychometric import PSYCHOMETRIC
from behav_utils.stats.registry import (
    compute_stats,
    fit,
    is_exchangeable,
    list_stats,
    stat,
)
from behav_utils.stats.sdt import SDT

__all__ = [
    'compute_stats', 'stat', 'fit', 'list_stats', 'is_exchangeable', 'fit_pse_dynamics',
    'PSYCHOMETRIC', 'LOGISTIC_HISTORY', 'SD_PROFILE', 'PSE_DYNAMICS', 'SDT',
]
