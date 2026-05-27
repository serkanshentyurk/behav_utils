"""
Shared Plotting Styles

Colours, palettes, themes, and defaults for all behav_utils plots.

Usage:
    from behav_utils.plotting.styles import PALETTE, UM_CMAP, apply_style
    apply_style()

    # Use indexed colours for consistent group comparisons
    for i, (label, group) in enumerate(groups.items()):
        plot_psychometric(group, ax=ax, color=PALETTE[i], label=label)
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# =============================================================================
# COLOUR PALETTES
# =============================================================================

# ── Main palette (indexed for sequential use) ────────────────────────
# Use PALETTE[0], PALETTE[1], ... for consistent colouring across plots.
PALETTE = [
    '#1f77b4',   # 0  blue
    '#ff7f0e',   # 1  orange
    '#2ca02c',   # 2  green
    '#d62728',   # 3  red
    '#9467bd',   # 4  purple
    '#8c564b',   # 5  brown
    '#e377c2',   # 6  pink
    '#7f7f7f',   # 7  grey
    '#bcbd22',   # 8  olive
    '#17becf',   # 9  teal
]

# ── Named colours (semantic use) ─────────────────────────────────────
COLOURS = {
    # Models
    'BE': '#4682B4',
    'SC': '#FF8C00',

    # Learning phases
    'naive':       '#E74C3C',
    'expert':      '#2ECC71',
    'post_shift':  '#F39C12',
    're_adapted':  '#3498DB',

    # SLDS states
    'inference':  '#3498DB',
    'updating':   '#E74C3C',
    'unknown':    '#95A5A6',

    # Manipulation types
    'distribution_shift': '#4682B4',
    'rule_flip':          '#FF8C00',
    'range_change':       '#2ECC71',

    # Trial outcomes
    'correct':     '#2ECC71',
    'error':       '#E74C3C',
    'no_response': '#95A5A6',

    # Plotting defaults
    'mean_line':  '#2C3E50',
    'sem_fill':   '#2C3E50',
    'default':    '#4682B4',
}


# =============================================================================
# COLOURMAPS
# =============================================================================

UM_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'update_matrix',
    [(253/255, 120/255, 6/255), (1, 1, 1), (120/255, 0/255, 220/255)],
)

SESSION_CMAP = plt.cm.viridis
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


# =============================================================================
# COLOUR HELPERS
# =============================================================================

def get_colour(index_or_name):
    """
    Resolve a colour from index (int) or name (str).

    Usage:
        get_colour(0)         → PALETTE[0]  (blue)
        get_colour('BE')      → COLOURS['BE']  (steelblue)
        get_colour('#ff0000') → '#ff0000'  (passthrough)
    """
    if isinstance(index_or_name, int):
        return PALETTE[index_or_name % len(PALETTE)]
    if index_or_name in COLOURS:
        return COLOURS[index_or_name]
    return index_or_name  # passthrough hex/named colour