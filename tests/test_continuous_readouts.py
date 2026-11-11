"""The continuous-response family (step 10b): readouts that agree with their statistics, a serial-bias curve
not fooled by a pull towards the middle, the ideal observer's posterior mean and its normative slope."""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from behav_utils import (  # noqa: E402
    GoNoGo,
    IdealObserver,
    Reproduction,
    ReproductionModel,
    TrialArrays,
    compute_error_distribution,
    compute_response_curve,
    compute_serial_bias,
    compute_stats,
    generate_synthetic_session,
    make_design,
    normative_slope,
    plot_error_distribution,
    plot_response_curve,
    plot_serial_bias,
    simulate,
)
from behav_utils.models import TrialwiseModel  # noqa: E402
from behav_utils.tasks import RequirementError, Task, TruncatedNormal, Uniform  # noqa: E402


def _arrays(model, n=6000, distribution=None, seed=0):
    return TrialArrays.from_sessions([generate_synthetic_session(model, task=Reproduction(), distribution=distribution,
                                                                 n_trials=n, rng=seed)])


def _weighted_slope(curve):
    ok = curve.counts > 0
    return np.polyfit(curve.centres[ok], curve.values[ok], 1, w=np.sqrt(curve.counts[ok]))[0]


class Attracted(TrialwiseModel):
    requires = frozenset({'continuous'})

    def __init__(self, w):
        self.w = w

    def reset(self):
        self.previous = None

    def respond(self, trial, rng):
        pull = 0.0 if self.previous is None else self.w * (self.previous - trial.stimulus)
        return 0.8 * trial.stimulus + pull + rng.normal(0.0, 0.05)

    def update(self, trial):
        self.previous = trial.stimulus


def test_the_response_curve_follows_the_response_slope():
    arrays = _arrays(ReproductionModel(bias=0.1, slope=0.8, noise=0.05), distribution=Uniform(-1, 1))
    curve = compute_response_curve(arrays)
    assert _weighted_slope(curve) == pytest.approx(compute_stats(arrays, ['response_slope'])['response_slope'], abs=0.02)
    assert curve.to_rows().shape == (8, 5) and int(curve.counts.sum()) == arrays.n_responded


def test_the_error_distribution_matches_its_statistics():
    arrays = _arrays(ReproductionModel(bias=0.1, slope=0.8, noise=0.05))
    dist = compute_error_distribution(arrays)
    stats = compute_stats(arrays, ['response_bias', 'error_sd'])
    assert dist.mean == pytest.approx(stats['response_bias']) and dist.sd == pytest.approx(stats['error_sd'])
    assert int(dist.counts.sum()) == dist.n == arrays.n_responded
    assert np.sum(dist.density * np.diff(dist.edges)) == pytest.approx(1.0)


def test_the_serial_bias_curve_is_flat_without_attraction_and_rises_with_it():
    flat = compute_serial_bias(_arrays(ReproductionModel(slope=0.8, noise=0.05), distribution=Uniform(-0.8, 0.8)))
    assert abs(_weighted_slope(flat)) < 0.02
    drawn = _arrays(Attracted(0.15), distribution=Uniform(-0.8, 0.8), seed=1)
    assert _weighted_slope(compute_serial_bias(drawn)) == pytest.approx(
        compute_stats(drawn, ['serial_bias'])['serial_bias'], abs=0.02)


def test_the_readouts_are_refused_on_a_binary_task():
    two_afc = TrialArrays.from_sessions([generate_synthetic_session(n_trials=200, rng=2)])
    for compute in (compute_response_curve, compute_error_distribution, compute_serial_bias):
        with pytest.raises(RequirementError, match='needs continuous'):
            compute(two_afc)


def test_each_plot_draws_one_panel():
    arrays = _arrays(ReproductionModel(), n=500)
    for plot, result in ((plot_response_curve, compute_response_curve(arrays)),
                         (plot_error_distribution, compute_error_distribution(arrays)),
                         (plot_serial_bias, compute_serial_bias(arrays))):
        fig, ax = plot(result)
        assert len(fig.axes) == 1 and (ax.lines or ax.patches or ax.collections)
        plt.close(fig)


def test_the_ideal_observer_pulls_towards_the_middle_as_theory_says():
    prior = TruncatedNormal(mean=0.0, sd=0.25)
    theory = 0.25 ** 2 / (0.25 ** 2 + 0.15 ** 2)
    assert normative_slope(prior, 0.15) == pytest.approx(theory, abs=0.002)
    session = generate_synthetic_session(IdealObserver(noise=0.15, distribution=prior), task=Reproduction(),
                                         distribution=prior, n_trials=20000, abort_rate=0, no_response_rate=0, rng=3)
    assert compute_stats(TrialArrays.from_sessions([session]), ['response_slope'])['response_slope'] == \
        pytest.approx(theory, abs=0.02)


def test_the_ideal_observer_still_refuses_a_task_it_cannot_serve():
    class NoAxis(GoNoGo):
        name = 'no_axis_go_nogo_for_tests'
        capabilities = frozenset({'binary', 'withhold', 'scored'})

    with pytest.raises(RequirementError, match='IdealObserver needs continuous, or binary and stimulus_axis'):
        simulate(IdealObserver(), make_design(NoAxis(), n_trials=20, rng=np.random.default_rng(4)))
    assert issubclass(NoAxis, Task)
