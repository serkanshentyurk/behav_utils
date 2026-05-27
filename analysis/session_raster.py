"""
Session Raster — Trial-level data extraction for raster plotting.

    compute_session_raster(session) → result dict → plot_session_raster(result)

Usage:
    from behav_utils.analysis.session_raster import compute_session_raster
    from behav_utils.plotting.session import plot_session_raster

    raster = compute_session_raster(filtered_session)
    fig, ax = plot_session_raster(raster, window=20)
"""

import numpy as np
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData


def compute_session_raster(session: 'SessionData') -> Dict:
    """
    Extract trial-by-trial data from a pre-filtered session for raster plotting.

    Does zero filtering — expects a pre-filtered session. Calls
    session.get_arrays() which only excludes aborts (always invalid).

    Args:
        session: Pre-filtered SessionData.

    Returns:
        Dict with:
            'stimuli':      np.ndarray — stimulus values per trial
            'choices':       np.ndarray — choice per trial (float, may contain NaN)
            'categories':    np.ndarray — correct category per trial
            'correct':       np.ndarray (bool) — whether each trial was correct
            'no_response':   np.ndarray (bool) — whether each trial had no response
            'n_trials':      int — total number of trials
            'session_id':    str — session identifier
            'session_idx':   int — session index within animal

        Pass to plot_session_raster() for drawing.
    """
    arrays = session.get_arrays()
    choices = arrays['choices']
    categories = arrays['categories']
    no_resp = arrays['no_response']

    valid = ~no_resp
    correct = np.full(len(choices), False)
    correct[valid] = (choices[valid] == categories[valid])

    return {
        'stimuli': arrays['stimuli'],
        'choices': choices,
        'categories': categories,
        'correct': correct,
        'no_response': no_resp,
        'n_trials': len(choices),
        'session_id': session.session_id,
        'session_idx': session.session_idx,
    }
