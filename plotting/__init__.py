"""
behav_utils.plotting — Behavioural Data Visualisation

Psychometric curves, trial rasters, stat trajectories, update matrices.
All functions return (fig, ax) for further customisation.

Usage:
    from behav_utils.plotting import plot_psychometric, plot_stat_trajectory
    from behav_utils.plotting.styles import COLOURS, apply_style
"""

from behav_utils.plotting.styles import (
    COLOURS, UM_CMAP, apply_style,
    get_animal_colours, get_session_colours, get_bin_colours,
)
from behav_utils.plotting.trajectory import (
    plot_stat_trajectory,
    plot_multi_animal_trajectory,
    plot_stat_grid,
)

# These are populated after migration script runs:
try:
    from behav_utils.plotting.psychometric import (
        plot_psychometric,
        plot_session_psychometrics,
    )
except ImportError:
    pass

try:
    from behav_utils.plotting.session import plot_session_trials
except ImportError:
    pass

try:
    from behav_utils.plotting.update_matrix import (
        plot_update_matrix,
        plot_conditional_psychometrics,
    )
except ImportError:
    pass

__all__ = [
    # Styles
    'COLOURS', 'UM_CMAP', 'apply_style',
    'get_animal_colours', 'get_session_colours', 'get_bin_colours',

    # Trajectories (always available)
    'plot_stat_trajectory',
    'plot_multi_animal_trajectory',
    'plot_stat_grid',

    # Psychometric (available after migration)
    'plot_psychometric',
    'plot_session_psychometrics',

    # Session (available after migration)
    'plot_session_trials',

    # Update matrix (available after migration)
    'plot_update_matrix',
    'plot_conditional_psychometrics',
]
