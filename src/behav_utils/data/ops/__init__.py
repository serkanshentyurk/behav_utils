"""
behav_utils.data.ops — Data operations (reshaping the data).

Operations that take SessionData (or raw arrays) and return a reorganised
view: session selection, trial filtering, pooling, downsampling. The data
classes themselves live in behav_utils.data.structures.
"""

from behav_utils.data.ops.filtering import (
    filter_trials,
    pool_arrays,
)
from behav_utils.data.ops.selection import (
    SessionFilter,
    list_presets,
    register_preset,
    register_presets_from_config,
    select_sessions,
)

__all__ = [
    'select_sessions', 'SessionFilter',
    'register_preset', 'list_presets', 'register_presets_from_config',
    'filter_trials', 'pool_arrays',
]
