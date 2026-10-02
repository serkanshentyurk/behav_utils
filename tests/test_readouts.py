"""behav_utils.readouts — contract tests. (Bit-exact equivalence with the legacy
compute_um / compute_psychometric / average_um was verified before deletion.)
"""

import dataclasses

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from behav_utils.data.arrays import TrialArrays
from behav_utils.data.ops.filtering import filter_trials
from behav_utils.data.ops.selection import select_sessions
from behav_utils.data.structures import TrialData
from behav_utils.data.synthetic import generate_synthetic_animal
from behav_utils.plotting.readouts import (
    plot_binned_curve,
    plot_conditional_psychometric,
    plot_psychometric_curve,
    plot_sd_profile,
    plot_update_matrix,
)
from behav_utils.readouts import (
    PARAMS,
    UpdateMatrix,
    compute_binned_curve,
    compute_conditional_psychometric,
    compute_psychometric_curve,
    compute_sd_profile,
    compute_update_matrix,
)


def _eq(a, b) -> bool:
    return np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


def _sessions(seed: int, n_sessions: int = 6, p_no_response: float = 0.05):
    animal, _ = generate_synthetic_animal(
        animal_id=f'T{seed}', n_sessions=n_sessions, trials_per_session=400,
        seed=seed, stage='Full_Task_Cont')
    rng = np.random.default_rng(seed)
    for s in animal.sessions:
        c = s.trials.choice.astype(float)
        c[rng.random(len(c)) < p_no_response] = np.nan
        d = {f.name: getattr(s.trials, f.name) for f in dataclasses.fields(s.trials)
             if not f.name.startswith('prev_')}
        d['choice'] = c
        s.trials = TrialData(**d)
    return filter_trials(select_sessions(animal, stage='Full_Task_Cont'), exclude_opto=False)


@pytest.fixture(scope='module')
def sess():
    return _sessions(0)


@pytest.fixture(scope='module')
def arrays(sess):
    return TrialArrays.from_sessions(sess)


@pytest.fixture(scope='module')
def empty():
    return TrialArrays.from_sequence([], [], [])


# ── contract ────────────────────────────────────────────────────────────────

class TestContract:
    def test_update_matrix_shape_and_rows(self, arrays):
        um = compute_update_matrix(arrays)
        assert um.matrix.shape == (8, 8) == um.conditional.shape
        assert um.n_pairs.dtype.kind == 'i' and um.n_pairs.sum() == um.n_trials
        rows = um.to_rows()
        assert len(rows) == 64 and set(rows.columns) >= {'cur_bin', 'prev_bin', 'shift', 'conditional'}
        assert um.profile().shape == (8,)

    def test_update_matrix_is_read_only(self, arrays):
        um = compute_update_matrix(arrays)
        with pytest.raises(ValueError):
            um.matrix[0, 0] = 0.0

    def test_update_matrix_uses_prev_view_not_adjacency(self, arrays):
        """A 1-in-3 subset must still find each trial's true predecessor."""
        sub = arrays.take(np.arange(arrays.n_trials) % 3 == 0)
        um = compute_update_matrix(sub)
        assert um.n_trials == int((sub.responded & sub.has_prev & (sub.prev_reward == 1)).sum())

    def test_update_matrix_bad_filter(self, arrays):
        with pytest.raises(ValueError):
            compute_update_matrix(arrays, trial_filter='bogus')

    def test_average_min_sources(self, arrays):
        a = compute_update_matrix(arrays)
        b = compute_update_matrix(arrays.take(np.arange(arrays.n_trials) < 300))   # sparse
        avg = UpdateMatrix.average([a, b], min_sources=2)
        assert avg.n_sources == 2 and avg.coverage.max() <= 2
        assert np.isnan(avg.matrix[avg.coverage < 2]).all()
        with pytest.raises(ValueError):
            UpdateMatrix.average([])

    def test_psychometric_curve_with_and_without_band(self, arrays):
        c0 = compute_psychometric_curve(arrays, n_bootstrap=0)
        c1 = compute_psychometric_curve(arrays, n_bootstrap=30, seed=1)
        assert c0.success and c0.ci is None and c0.band is None
        assert c1.ci.shape == (4, 2) and c1.band.shape == (2, c1.x.size) and c1.n_bootstrap > 0
        assert (c1.ci[:, 0] <= c1.params.to_numpy()).all() and (c1.params.to_numpy() <= c1.ci[:, 1]).all()
        assert list(c1.to_rows()['param']) == list(PARAMS)
        assert list(c0.params.index) == list(PARAMS)

    def test_psychometric_bootstrap_is_seeded(self, arrays):
        a = compute_psychometric_curve(arrays, n_bootstrap=20, seed=7)
        b = compute_psychometric_curve(arrays, n_bootstrap=20, seed=7)
        assert _eq(a.ci, b.ci)

    def test_conditional_psychometric(self, arrays):
        cp = compute_conditional_psychometric(arrays)
        assert cp.success and cp.params.shape == (8, 4)
        assert cp.fell_back.dtype == bool and cp.n_per_bin.sum() == arrays.lag1_pairs().n_trials
        assert len(cp.to_rows()) == 32 and len(cp.to_flat()) == 32
        assert cp.to_flat().index[0] == 'cond_mu_0'

    def test_binned_curve_drops_no_response(self, arrays):
        acc = compute_binned_curve(arrays, kind='accuracy')
        pb = compute_binned_curve(arrays, kind='choice_prob')
        assert acc.counts.sum() == pb.counts.sum() == arrays.n_responded
        assert np.isfinite(pb.values).all()
        with pytest.raises(ValueError):
            compute_binned_curve(arrays, kind='bogus')

    def test_sd_profile(self, arrays):
        prof = compute_sd_profile(arrays)
        assert prof.profile.shape == (8,) and prof.n_pairs > 0
        assert len(prof.to_rows()) == 8

    def test_empty_input_everywhere(self, empty):
        for f in (compute_update_matrix, compute_psychometric_curve,
                  compute_conditional_psychometric, compute_binned_curve, compute_sd_profile):
            r = f(empty)
            assert isinstance(r.to_rows(), pd.DataFrame)
        assert not compute_psychometric_curve(empty).success

    def test_plotters_draw(self, arrays, empty):
        plot_update_matrix(compute_update_matrix(arrays))
        plot_update_matrix(compute_update_matrix(empty))
        plot_psychometric_curve(compute_psychometric_curve(arrays, n_bootstrap=10))
        plot_psychometric_curve(compute_psychometric_curve(empty))
        plot_conditional_psychometric(compute_conditional_psychometric(arrays))
        plot_binned_curve(compute_binned_curve(arrays))
        plot_sd_profile(compute_sd_profile(arrays))
        plt.close('all')
