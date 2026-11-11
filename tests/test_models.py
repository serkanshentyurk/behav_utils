"""Models and designs: distributions, make_design's trial classes, simulate() scoring, parameter recovery,
the ideal observer's normative PSE, and models written here running with no library change."""

import numpy as np
import pytest

from behav_utils import (
    IdealObserver,
    PsychometricModel,
    RandomModel,
    TrialArrays,
    compute_stats,
    filter_trials,
    make_design,
    normative_pse,
    simulate,
)
from behav_utils.models import Design, Responses, TrialwiseModel, simulate_responses
from behav_utils.tasks import Discrete, GoNoGo, Outcome, TruncatedNormal, TwoAFC, Uniform


def _stats(session, names):
    return compute_stats(TrialArrays.from_sessions([session]), names)


class TestDistributions:
    @pytest.mark.parametrize('dist', [Uniform(), TruncatedNormal(-0.3, 0.4), Discrete([-0.5, 0.1, 0.6], [1, 2, 1])])
    def test_samples_stay_in_the_support_and_the_grid_sums_to_one(self, dist):
        sample = dist.sample(5000, np.random.default_rng(0))
        low, high = dist.support
        assert sample.min() >= low and sample.max() <= high
        assert dist.grid()[1].sum() == pytest.approx(1.0)

    def test_discrete_draws_follow_the_weights(self):
        sample = Discrete([-0.5, 0.1, 0.6], [1, 2, 1]).sample(20000, np.random.default_rng(1))
        assert np.mean(sample == 0.1) == pytest.approx(0.5, abs=0.02)

    def test_bad_settings_raise(self):
        with pytest.raises(ValueError):
            Uniform(1, -1)
        with pytest.raises(ValueError):
            TruncatedNormal(sd=0)


class TestMakeDesign:
    def test_the_rates(self):
        d = make_design(n_trials=20000, laser_fraction=0.3, abort_rate=0.05, no_response_rate=0.1,
                        rng=np.random.default_rng(0))
        assert d.laser.mean() == pytest.approx(0.3, abs=0.02)
        assert d.abort.mean() == pytest.approx(0.05, abs=0.01)
        assert d.no_response[~d.abort].mean() == pytest.approx(0.1, abs=0.01)
        assert not (d.abort & d.no_response).any()

    def test_the_laser_run_cap(self):
        d = make_design(n_trials=5000, laser_fraction=0.6, max_laser_run=1, rng=np.random.default_rng(0))
        assert not (d.laser[1:] & d.laser[:-1]).any()

    def test_go_nogo_has_no_no_response_trials(self):
        with pytest.raises(ValueError, match='withhold'):
            make_design(GoNoGo(), no_response_rate=0.1)

    def test_a_rate_outside_zero_one_raises(self):
        with pytest.raises(ValueError, match='abort_rate'):
            make_design(abort_rate=1.5)


class TestSimulate:
    def test_sessions_are_scored_by_the_task_and_carry_every_flag(self):
        rng = np.random.default_rng(0)
        d = make_design(n_trials=2000, laser_fraction=0.3, abort_rate=0.05, no_response_rate=0.05, rng=rng)
        trials = simulate(PsychometricModel(), d, rng).trials
        assert np.isnan(trials.choice[d.abort | d.no_response]).all()
        assert (trials.outcome[d.abort] == Outcome.ABORT).all()
        assert (trials.outcome[d.no_response] == Outcome.NO_RESPONSE).all()
        responded = ~(d.abort | d.no_response)
        np.testing.assert_array_equal(trials.correct[responded], trials.choice[responded] == d.truth[responded])
        np.testing.assert_array_equal(trials.opto_on, d.laser)

    def test_the_same_seed_gives_the_same_session(self):
        d = make_design(n_trials=300, rng=np.random.default_rng(3))
        a = simulate(PsychometricModel(), d, np.random.default_rng(7)).trials.choice
        b = simulate(PsychometricModel(), d, np.random.default_rng(7)).trials.choice
        np.testing.assert_array_equal(a, b)

    def test_simulate_responses_returns_only_choices(self):
        out = simulate_responses(PsychometricModel(), np.linspace(-1, 1, 50), rng=np.random.default_rng(0))
        assert out.shape == (50,) and set(np.unique(out)) <= {0.0, 1.0}


class TestBuiltIns:
    def test_psychometric_parameters_come_back(self):
        rng = np.random.default_rng(0)
        session = simulate(PsychometricModel(mu=0.1, sigma=0.25), make_design(n_trials=20000, rng=rng), rng)
        stats = _stats(session, ['mu', 'sigma'])
        assert stats['mu'] == pytest.approx(0.1, abs=0.03) and stats['sigma'] == pytest.approx(0.25, abs=0.03)

    def test_a_laser_override_shows_on_laser_trials(self):
        rng = np.random.default_rng(1)
        design = make_design(n_trials=30000, laser_fraction=0.5, rng=rng)
        session = simulate(PsychometricModel(sigma=0.2, laser={'sigma': 0.6}), design, rng)
        on = compute_stats(TrialArrays.from_sessions(filter_trials([session], trial_type='opto')), ['sigma'])
        off = compute_stats(TrialArrays.from_sessions(filter_trials([session], trial_type='non_opto')), ['sigma'])
        assert on['sigma'] == pytest.approx(0.6, abs=0.06) and off['sigma'] == pytest.approx(0.2, abs=0.03)

    def test_an_unknown_laser_override_raises(self):
        with pytest.raises(ValueError, match='slope'):
            PsychometricModel(laser={'slope': 1.0})

    def test_random_model_ignores_the_stimulus(self):
        rng = np.random.default_rng(2)
        session = simulate(RandomModel(p=0.7), make_design(n_trials=10000, rng=rng), rng)
        assert np.nanmean(session.trials.choice) == pytest.approx(0.7, abs=0.02)

    def test_the_normative_pse_is_zero_for_a_symmetric_prior(self):
        assert normative_pse(Uniform(), 0.3) == pytest.approx(0.0, abs=1e-6)

    def test_the_ideal_observer_sits_at_its_normative_pse(self):
        prior = TruncatedNormal(mean=-0.3, sd=0.4)
        expected = normative_pse(prior, 0.3)
        rng = np.random.default_rng(4)
        session = simulate(IdealObserver(noise=0.3, distribution=prior), make_design(distribution=prior,
                                                                                   n_trials=30000, rng=rng), rng)
        assert abs(expected) > 0.05
        assert _stats(session, ['mu'])['mu'] == pytest.approx(expected, abs=0.03)


class WinStayLoseShift(TrialwiseModel):
    """Repeats a correct response and switches after an error; follows the stimulus on the first trial."""

    def __init__(self, p_follow=0.9):
        self.p_follow = p_follow

    def reset(self):
        self.last = None

    def p_choice(self, t, stimulus, laser):
        if self.last is None:
            return float(stimulus > 0)
        choice, correct = self.last
        repeat = choice if correct else 1 - choice
        return self.p_follow if repeat == 1 else 1 - self.p_follow

    def update(self, t, stimulus, choice, correct):
        self.last = (choice, correct)


class AlwaysB:
    """Not a Model subclass: anything with simulate() works."""

    def simulate(self, design, rng):
        return Responses(np.ones(design.n_trials))


class TestModelsWrittenElsewhere:
    def test_a_learning_model_runs_with_no_library_change(self):
        rng = np.random.default_rng(5)
        session = simulate(WinStayLoseShift(), make_design(n_trials=4000, abort_rate=0.05, rng=rng), rng)
        stats = _stats(session, ['win_stay_rate', 'win_stay'])
        assert stats['win_stay_rate'] == pytest.approx(0.9, abs=0.03)    # P(repeat | correct)
        assert stats['win_stay'] == pytest.approx(0.8, abs=0.04)         # minus P(repeat | error)

    def test_any_object_with_simulate_runs(self):
        rng = np.random.default_rng(6)
        design = make_design(n_trials=1000, abort_rate=0.1, rng=rng)
        session = simulate(AlwaysB(), design, rng)
        assert np.isnan(session.trials.choice[design.abort]).all()
        assert _stats(session, ['accuracy'])['accuracy'] == pytest.approx(np.mean(design.truth[~design.abort]))

    def test_something_without_simulate_is_refused(self):
        with pytest.raises(TypeError, match='simulate'):
            simulate(object(), Design(TwoAFC(), [0.1]))
