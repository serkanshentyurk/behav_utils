"""Tests for behav_utils.analysis.update_matrix."""

import numpy as np
import pytest

from behav_utils.analysis.update_matrix import fit_update_matrix, matrix_error
from behav_utils.data.arrays import TrialArrays
from behav_utils.data.ops.filtering import filter_trials, pool_arrays
from behav_utils.readouts import compute_update_matrix


class TestFitUpdateMatrix:
    """Tests for update matrix computation."""

    def test_shape(self, rng):
        """Output should be (n_bins, n_bins)."""
        n = 500
        stimuli = rng.uniform(-1, 1, n)
        categories = (stimuli > 0).astype(float)
        choices = categories.copy()

        um, _, _ = fit_update_matrix(stimuli, choices, categories, n_bins=8)
        assert um.shape == (8, 8)

    def test_symmetric_data_symmetric_um(self, rng):
        """Symmetric behaviour should give roughly symmetric UM."""
        n = 5000
        stimuli = rng.uniform(-1, 1, n)
        categories = (stimuli > 0).astype(float)
        choices = categories.copy()  # perfect observer

        um, _, _ = fit_update_matrix(stimuli, choices, categories, n_bins=8)
        # UM should be close to zero for a perfect observer
        assert np.nanmax(np.abs(um)) < 0.05

    def test_different_n_bins(self, rng):
        """Should work with different bin counts."""
        n = 500
        stimuli = rng.uniform(-1, 1, n)
        categories = (stimuli > 0).astype(float)
        choices = rng.choice([0.0, 1.0], n)

        for n_bins in [4, 6, 8, 10]:
            um, _, _ = fit_update_matrix(
                stimuli, choices, categories, n_bins=n_bins)
            assert um.shape == (n_bins, n_bins)


class TestMatrixError:
    """Tests for matrix_error (MSE between UMs)."""

    def test_identical_matrices(self):
        """Error between identical matrices should be zero."""
        um = np.random.default_rng(42).uniform(-0.1, 0.1, (8, 8))
        err = matrix_error(um, um)
        assert err == pytest.approx(0.0)

    def test_nan_handling(self):
        """NaN cells should be excluded from error."""
        um1 = np.ones((8, 8)) * 0.1
        um2 = np.ones((8, 8)) * 0.2
        um1[0, 0] = np.nan
        um2[0, 0] = np.nan

        err = matrix_error(um1, um2)
        assert not np.isnan(err)
        assert err == pytest.approx(0.01)  # (0.1)^2

    def test_commutative(self):
        """matrix_error(a, b) == matrix_error(b, a)."""
        rng = np.random.default_rng(42)
        um1 = rng.uniform(-0.1, 0.1, (8, 8))
        um2 = rng.uniform(-0.1, 0.1, (8, 8))
        assert matrix_error(um1, um2) == pytest.approx(matrix_error(um2, um1))


class TestUpdateMatrixReadout:
    def test_pooled_shape(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        um = compute_update_matrix(TrialArrays.from_sessions(clean))
        assert um.matrix.shape == (8, 8) and um.n_trials > 0

    def test_prev_equals_adjacency_on_opto_free(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        p = pool_arrays(clean)
        um_adj, _, _ = fit_update_matrix(
            p['stimuli'], p['choices'], p['categories'],
            no_response=p['no_response'], not_blockstart=p['prev_has_prev'])
        um = compute_update_matrix(TrialArrays.from_pooled(p)).matrix
        assert np.allclose(um, um_adj, equal_nan=True)


class TestUpdateMatrixOpto:
    def _opto_sessions(self, animal):
        return [s for s in animal.sessions if np.asarray(s.trials.opto_on).any()]

    def test_opto_subset_uses_frozen_prev(self, synthetic_opto_animal):
        """On an opto-only subset the predecessor is the true one, not subset adjacency."""
        opto = self._opto_sessions(synthetic_opto_animal)
        trials_opto = filter_trials(opto, mask_fn=lambda s: s.trials.opto_on == 1)
        p = pool_arrays(trials_opto)
        um = compute_update_matrix(TrialArrays.from_pooled(p)).matrix
        um_wrong, _, _ = fit_update_matrix(
            p['stimuli'], p['choices'], p['categories'],
            no_response=p['no_response'], not_blockstart=p['prev_has_prev'])
        assert np.isfinite(um).any(), 'UM all-NaN: not enough opto trials to test'
        assert not np.allclose(um, um_wrong, equal_nan=True)

    def test_post_opto_mask_runs(self, synthetic_opto_animal):
        opto = self._opto_sessions(synthetic_opto_animal)
        post = filter_trials(opto, mask_fn=lambda s: s.trials.prev_opto_on == 1)
        assert compute_update_matrix(TrialArrays.from_sessions(post)).matrix.shape == (8, 8)
