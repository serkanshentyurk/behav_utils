"""
Array-valued readouts of a block of trials (see ``_base`` for the contract).

    from behav_utils.readouts import compute_update_matrix, compute_psychometric_curve
    um = compute_update_matrix(TrialArrays.from_sessions(sessions))
    um.matrix, um.profile(), um.to_rows()

The fit engines on raw arrays (``fit_psychometric``, ``fit_update_matrix``) live in ``fit.py``;
``stats`` and the readouts both import them, and ``fit.py`` imports nothing from the package.
"""

from behav_utils.readouts.binned import BinnedCurve, compute_binned_curve
from behav_utils.readouts.conditional_psychometric import (
    ConditionalPsychometric,
    compute_conditional_psychometric,
)
from behav_utils.readouts.fit import (
    cumulative_gaussian,
    fit_psychometric,
    fit_psychometric_gof,
    fit_update_matrix,
    matrix_error,
)
from behav_utils.readouts.psychometric import PARAMS, PsychometricCurve, compute_psychometric_curve
from behav_utils.readouts.sd_profile import SerialDependenceProfile, compute_sd_profile
from behav_utils.readouts.update_matrix import UpdateMatrix, compute_update_matrix

__all__ = [
    'fit_psychometric', 'fit_psychometric_gof', 'fit_update_matrix', 'matrix_error', 'cumulative_gaussian',
    'UpdateMatrix', 'compute_update_matrix',
    'PsychometricCurve', 'compute_psychometric_curve', 'PARAMS',
    'ConditionalPsychometric', 'compute_conditional_psychometric',
    'BinnedCurve', 'compute_binned_curve',
    'SerialDependenceProfile', 'compute_sd_profile',
]
