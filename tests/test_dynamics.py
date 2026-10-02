"""behav_utils.stats.dynamics — the pse_dynamics fit (exponential vs step PSE trajectory)."""

import numpy as np
import pytest
from scipy.stats import norm

from behav_utils.data.arrays import TrialArrays
from behav_utils.stats import PSE_DYNAMICS, compute_stats, is_exchangeable, list_producers


def _block(kind, n=1500, tau=120.0, mu_s=-0.4, mu_e=0.15, t_switch=200, sigma=0.25, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, n)
    cat = (s > 0).astype(float)
    t = np.arange(n)
    mu = mu_e + (mu_s - mu_e) * np.exp(-t / tau) if kind == 'exp' else np.where(t < t_switch, mu_s, mu_e)
    p = 0.03 + 0.94 * norm.cdf((s - mu) / sigma)
    c = (rng.random(n) < p).astype(float)
    return TrialArrays.from_sequence(c, s, cat)


def test_registered_and_order_dependent():
    assert list_producers()['pse_dynamics'] == PSE_DYNAMICS
    for name in PSE_DYNAMICS:
        assert not is_exchangeable(name)


def test_exponential_block_recovers_shape():
    """τ from one block is loosely identified at mouse-like noise (a single draw can
    sit at 2–4× the truth with a *better* likelihood), so assert on the median."""
    rs = [compute_stats(_block('exp', seed=k), PSE_DYNAMICS) for k in range(5)]
    taus = np.array([r['pse_tau'] for r in rs])
    assert 60 < np.median(taus) < 240                        # within a factor of two
    assert all(r['pse_start'] < r['pse_final'] for r in rs)  # moves in the right direction
    assert all(r['pse_censored'] == 0.0 for r in rs)
    assert all(r['pse_trials_to_90'] == pytest.approx(r['pse_tau'] * np.log(10.0)) for r in rs)
    assert np.median([r['pse_shape_daic'] for r in rs]) < 0  # gradual favoured on balance


def test_step_block_is_flagged_as_step():
    r = compute_stats(_block('step'), PSE_DYNAMICS)
    assert r['pse_shape_daic'] > 0                           # step favoured
    assert abs(r['pse_step_switch'] - 200) < 40


def test_flat_block_stays_flat_in_sample():
    r = compute_stats(_block('exp', tau=1e6, mu_s=0.05, mu_e=0.05), PSE_DYNAMICS)
    assert abs(r['pse_final'] - 0.05) < 0.15                 # in-sample endpoint is flat
    assert abs(r['pse_final'] - r['pse_start']) < 0.2        # no real trajectory
    assert np.isfinite(r['pse_tau'])


def test_short_block_is_nan():
    assert compute_stats(_block('exp', n=60), PSE_DYNAMICS).isna().all()


def test_trial_type_all_drops_aborts():
    """The 'all' filter must drop aborted trials like every other trial_type (bug fix)."""
    from behav_utils.data.ops.filtering import filter_trials
    from behav_utils.data.synthetic import generate_synthetic_animal
    animal, _ = generate_synthetic_animal(animal_id='T', n_sessions=1, trials_per_session=300, seed=0)
    s = animal.sessions[0]
    n_abort = int(np.asarray(s.trials.abort).sum())
    assert n_abort > 0, 'synthetic animal should contain aborts for this test'
    kept = filter_trials([s], trial_type='all')[0]
    assert len(kept.trials.choice) == len(s.trials.choice) - n_abort
    assert not np.asarray(kept.trials.abort).any()
