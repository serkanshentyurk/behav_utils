"""
Tests for behav_utils/analysis/psychometry.py.

Covers the fit_psychometric engine (session-level readouts live in test_readouts).
"""

import numpy as np


class TestFitPsychometric:
    """fit_psychometric on raw arrays returns mu/sigma keys."""

    def test_returns_mu_sigma_keys(self, rng):
        """Post-rename: returns mu, sigma, lapse_low, lapse_high — NOT pse, slope."""
        from behav_utils.analysis.psychometry import fit_psychometric

        n = 500
        stim = rng.uniform(-1, 1, n)
        ch = (stim > 0).astype(float) + (rng.random(n) < 0.1) * (rng.random(n) > 0.5)
        ch = np.clip(ch, 0, 1)

        result = fit_psychometric(stim, ch)

        # Naming convention (mu/sigma everywhere, no pse/slope)
        assert 'mu' in result
        assert 'sigma' in result
        assert 'lapse_low' in result
        assert 'lapse_high' in result
        assert 'pse' not in result, "should use 'mu' not 'pse'"
        assert 'slope' not in result, "should use 'sigma' not 'slope'"

    def test_recovers_known_boundary(self, rng):
        """Synthetic data with known boundary should be recovered."""
        from behav_utils.analysis.psychometry import fit_psychometric

        # Generate data with boundary at 0.2
        n = 2000
        stim = rng.uniform(-1, 1, n)
        true_mu = 0.2
        true_sigma = 0.15
        from scipy.stats import norm
        p_right = norm.cdf((stim - true_mu) / true_sigma)
        ch = (rng.random(n) < p_right).astype(float)

        result = fit_psychometric(stim, ch)

        # mu within ±0.05 of truth
        assert abs(result['mu'] - true_mu) < 0.05, f"mu={result['mu']}, true={true_mu}"
        # sigma within reasonable range
        assert 0.05 < result['sigma'] < 0.5

    def test_returns_success_flag(self, rng):
        """Fit returns a success flag."""
        from behav_utils.analysis.psychometry import fit_psychometric

        n = 300
        stim = rng.uniform(-1, 1, n)
        ch = (stim > 0).astype(float)
        result = fit_psychometric(stim, ch)
        assert 'success' in result

    def test_lapses_in_unit_interval(self, rng):
        """Lapse rates must be valid probabilities."""
        from behav_utils.analysis.psychometry import fit_psychometric

        n = 500
        stim = rng.uniform(-1, 1, n)
        ch = (stim > 0).astype(float) + (rng.random(n) < 0.05)
        ch = np.clip(ch, 0, 1)
        result = fit_psychometric(stim, ch)

        if result.get('success', False):
            assert 0 <= result['lapse_low'] <= 1
            assert 0 <= result['lapse_high'] <= 1
