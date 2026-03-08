import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional


def cumulative_gaussian(x: np.ndarray, mu: float, sigma: float,
                        lapse_low: float = 0.0, lapse_high: float = 0.0) -> np.ndarray:
    """
    Compute cumulative Gaussian psychometric function.
    
    Args:
        x: Stimulus values
        mu: Mean (PSE - point of subjective equality)
        sigma: Standard deviation (slope)
        lapse_low: Lower lapse rate (guess rate for category A)
        lapse_high: Upper lapse rate (lapse rate for category B)
    
    Returns:
        P(choose B) for each stimulus value
    """
    x = np.asarray(x, dtype=np.float64)
    return lapse_low + (1 - lapse_low - lapse_high) * norm.cdf(x, mu, sigma)


def generate_stimuli(
    n_trials: int = 300,
    boundary: float = 0.0,
    x_min: float = -1.0,
    x_max: float = 1.0,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """
    Generate random stimuli and corresponding categories.
    
    Args:
        n_trials: Number of trials
        boundary: Category boundary location (default 0)
        x_min: Minimum stimulus value (default -1)
        x_max: Maximum stimulus value (default 1)
        seed: Random seed (ignored if rng provided)
        rng: Random number generator (created if None)
    
    Returns:
        stimuli: Array of stimulus values (uniform distribution)
        categories: Array of true categories (0 = A, 1 = B)
        rng: Random number generator (for continued use)
    
    Example:
        stimuli, categories, rng = generate_stimuli(n_trials=300, seed=42)
        choices, rewards = model.simulate_session(stimuli, categories, rng=rng)
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    
    stimuli = rng.uniform(x_min, x_max, n_trials)
    categories = (stimuli > boundary).astype(int)
    
    return stimuli, categories, rng