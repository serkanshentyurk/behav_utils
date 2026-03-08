"""
Shared Plotting Styles

Colours, themes, and defaults used across all behav_utils plots.
Import these to keep a consistent look.

Usage:
    from behav_utils.plotting.styles import COLOURS, apply_style, UM_CMAP
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# =============================================================================
# COLOUR PALETTES
# =============================================================================

COLOURS = {
    # Model colours
    'BE': '#4682B4',           # steelblue
    'SC': '#FF8C00',           # darkorange

    # Phase colours
    'naive': '#E74C3C',        # red
    'expert': '#2ECC71',       # green
    'post_shift': '#F39C12',   # amber
    're_adapted': '#3498DB',   # blue

    # State colours
    'inference': '#3498DB',    # blue
    'updating': '#E74C3C',     # red
    'unknown': '#95A5A6',      # grey

    # Manipulation type colours
    'distribution_shift': '#4682B4',
    'rule_flip': '#FF8C00',
    'range_change': '#2ECC71',

    # General
    'correct': '#2ECC71',
    'error': '#E74C3C',
    'no_response': '#95A5A6',
    'mean_line': '#2C3E50',    # dark blue-grey
    'sem_fill': '#2C3E50',
    'default': '#4682B4',
}


# =============================================================================
# COLOURMAPS
# =============================================================================

# Update matrix diverging colourmap (orange ← white → purple)
UM_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'update_matrix',
    [(253/255, 120/255, 6/255), (1, 1, 1), (120/255, 0/255, 220/255)]
)

# Session progression (dark → light)
SESSION_CMAP = plt.cm.viridis

# Previous-stimulus bin colourmap (coolwarm)
BIN_CMAP = plt.cm.coolwarm


# =============================================================================
# STYLE DEFAULTS
# =============================================================================

DEFAULT_DPI = 100
DEFAULT_FONT_SIZE = 10
DEFAULT_FIGURE_WIDTH = 10.0
DEFAULT_LINE_WIDTH = 1.5
DEFAULT_MARKER_SIZE = 5
DEFAULT_ALPHA = 0.7
SEM_ALPHA = 0.15


def apply_style(dpi=DEFAULT_DPI, font_size=DEFAULT_FONT_SIZE):
    """Apply default behav_utils style to matplotlib."""
    plt.rcParams.update({
        'figure.dpi': dpi,
        'font.size': font_size,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
    })


def get_animal_colours(animal_ids, cmap_name='tab10'):
    """Generate consistent colour mapping for a list of animal IDs."""
    cmap = plt.cm.get_cmap(cmap_name)
    return {
        aid: cmap(i % cmap.N)
        for i, aid in enumerate(sorted(animal_ids))
    }


def get_session_colours(n_sessions, cmap=SESSION_CMAP):
    """Generate colour gradient for sessions (early=dark, late=light)."""
    return [cmap(i / max(n_sessions - 1, 1)) for i in range(n_sessions)]


def get_bin_colours(n_bins, cmap=BIN_CMAP):
    """Generate colours for stimulus bins."""
    return [cmap(i / max(n_bins - 1, 1)) for i in range(n_bins)]
