"""
Session Trajectory Computation

compute_trajectory(sessions, stat_names)

Takes pre-filtered sessions, computes per-session stats.
Returns structured result ready for plot_trajectory().

Usage:
    from behav_utils.analysis.trajectory import compute_trajectory
    from behav_utils.plotting.trajectory import plot_trajectory

    sessions = filter_trials(select_sessions(animal, 'expert_uniform'))
    result = compute_trajectory(sessions, stat_names=['accuracy', 'pse'])
    fig, ax = plt.subplots()
    plot_trajectory(result, stat_name='accuracy', ax=ax)
"""

import numpy as np
from typing import Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData, AnimalData


def compute_trajectory(
    data,
    stat_names: Union[str, List[str]],
) -> Dict:
    """
    Compute per-session summary stats across a list of pre-filtered sessions.

    Args:
        data: Pre-filtered List[SessionData] or AnimalData.
        stat_names: Stat name(s) to compute. String or list of strings.

    Returns:
        Dict with:
            'stat_names': list of stat names computed
            'session_indices': list of session_idx values
            'session_ids': list of session_id values
            'per_session': list of dicts, each with stat values + metadata
            'n_sessions': int
            'values': dict of {stat_name: array of values} for convenience

        Pass to plot_trajectory() for drawing.
    """
    from behav_utils.data.structures import SessionData, AnimalData

    if isinstance(stat_names, str):
        stat_names = [stat_names]

    # Resolve to session list
    if isinstance(data, AnimalData):
        sessions = list(data.sessions)
    elif isinstance(data, (list, tuple)):
        sessions = list(data)
    else:
        raise TypeError(
            f"Expected List[SessionData] or AnimalData, got {type(data).__name__}")

    from behav_utils.analysis.summary_stats import compute_summary_stats
    from behav_utils.data.structures import _flatten_stats_dict

    per_session = []
    for sess in sessions:
        entry = {
            'session_idx': sess.session_idx,
            'session_id': sess.session_id,
            'date': sess.date,
            'stage': getattr(sess, 'stage', None),
            'distribution': getattr(sess, 'distribution', None),
        }
        arrays = sess.get_arrays()
        stats = compute_summary_stats(
            choices=arrays['choices'],
            stimuli=arrays['stimuli'],
            categories=arrays['categories'],
            stat_names=stat_names,
            return_dict=True,
        )
        entry.update(_flatten_stats_dict(stats))
        per_session.append(entry)

    # Build convenience arrays
    values = {}
    for sn in stat_names:
        values[sn] = np.array([
            e.get(sn, np.nan) for e in per_session], dtype=float)

    return {
        'stat_names': stat_names,
        'session_indices': [e['session_idx'] for e in per_session],
        'session_ids': [e['session_id'] for e in per_session],
        'per_session': per_session,
        'n_sessions': len(per_session),
        'values': values,
    }
