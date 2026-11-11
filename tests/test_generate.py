"""Synthetic data from any model: every trial class, event times that pass the clock checks, canonical ids
and dates, per-session settings, experiments with independent streams, go/no-go, and reproducibility."""

import warnings

import numpy as np
import pytest

from behav_utils import (
    GoNoGo,
    PsychometricModel,
    TrialArrays,
    compute_stats,
    filter_trials,
    generate_synthetic_animal,
    generate_synthetic_experiment,
    generate_synthetic_session,
)
from behav_utils.config import ClockConfig
from behav_utils.data.loading import _check_clock
from behav_utils.models import Responses, Timing
from behav_utils.tasks import Discrete, Uniform


def test_the_default_session_has_every_trial_class():
    session = generate_synthetic_session(n_trials=2000, laser_fraction=0.3, rng=0)
    t = session.trials
    assert t.abort.any() and (np.isnan(t.choice) & ~t.abort).any() and t.opto_on.any()
    assert set(np.unique(t.choice[~np.isnan(t.choice)])) == {0.0, 1.0}
    assert filter_trials([session], trial_type='post_opto', min_trials=1)[0].n_trials > 0
    assert session.session_type == 'opto'


def test_event_times_follow_the_trials_and_pass_the_clock_checks():
    session = generate_synthetic_session(n_trials=500, rng=1)
    events, rt = session.trials.events, session.trials.reaction_time
    assert session.clock == 'synthetic' and events['t_trial_start'][0] == Timing().session_start
    has_rt = ~np.isnan(rt)
    np.testing.assert_allclose(events['t_response'][has_rt] - events['t_stim_on'][has_rt], rt[has_rt])
    assert np.isnan(events['t_response'][np.isnan(session.trials.choice)]).all()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _check_clock('synthetic', session.trials.trial_number, events, rt,
                     ClockConfig(order=['t_trial_start', 't_stim_on', 't_response']))


def test_a_reaction_time_from_the_model_is_used():
    class Slow:
        def simulate(self, design, rng):
            return Responses(np.ones(design.n_trials), reaction_time=np.full(design.n_trials, 1.5))

    trials = generate_synthetic_session(Slow(), n_trials=50, abort_rate=0, no_response_rate=0, rng=2).trials
    np.testing.assert_allclose(trials.reaction_time, 1.5)


def test_ids_dates_and_per_session_settings():
    animal = generate_synthetic_animal(
        [PsychometricModel(sigma=0.8 - 0.1 * i) for i in range(4)], n_sessions=4,
        trials_per_session=[100, 200, 300, 400], distribution=[Uniform()] * 2 + [Discrete([-0.2, 0.2])] * 2,
        label=['uniform', 'uniform', 'narrow', 'narrow'], rng=3)
    assert animal.session_ids == ['SYN01_20260105', 'SYN01_20260106', 'SYN01_20260107', 'SYN01_20260108']
    assert [s.n_trials for s in animal.sessions] == [100, 200, 300, 400]
    assert set(np.unique(animal.sessions[3].trials.stimulus)) == {-0.2, 0.2}
    assert [s.distribution for s in animal.sessions] == ['uniform', 'uniform', 'narrow', 'narrow']


def test_a_list_of_the_wrong_length_raises():
    with pytest.raises(ValueError, match='model has 2 entries for 3 sessions'):
        generate_synthetic_animal([PsychometricModel()] * 2, n_sessions=3)


def test_an_experiment_has_groups_and_independent_streams():
    spec = {'A01': {'model': PsychometricModel(sigma=0.3), 'n_sessions': 2, 'metadata': {'group': 'control'}},
            'A02': {'model': PsychometricModel(sigma=0.5), 'n_sessions': 2, 'metadata': {'group': 'treated'}}}
    experiment = generate_synthetic_experiment(spec, rng=4)
    assert experiment.animals['A02'].metadata['group'] == 'treated'
    bigger = generate_synthetic_experiment({'A00': {'n_sessions': 1}, **spec}, rng=4)
    for animal_id in spec:
        np.testing.assert_array_equal(experiment.animals[animal_id].sessions[0].trials.choice,
                                      bigger.animals[animal_id].sessions[0].trials.choice)


def test_go_nogo_sessions_have_withholds_not_no_responses():
    trials = generate_synthetic_session(task=GoNoGo(), n_trials=600, rng=5).trials
    assert not np.isnan(trials.choice[~trials.abort]).any()
    withheld = ~trials.abort & (trials.choice == 0)
    assert withheld.any() and np.isnan(trials.events['t_response'][withheld]).all()


def test_the_same_seed_gives_the_same_animal():
    a = generate_synthetic_animal(n_sessions=2, rng=6)
    b = generate_synthetic_animal(n_sessions=2, rng=6)
    for x, y in zip(a.sessions, b.sessions):
        np.testing.assert_array_equal(x.trials.choice, y.trials.choice)
        np.testing.assert_array_equal(x.trials.events['t_response'], y.trials.events['t_response'])


def test_a_laser_effect_reaches_the_opto_split():
    animal = generate_synthetic_animal(PsychometricModel(sigma=0.2, laser={'sigma': 0.6}), n_sessions=10,
                                       trials_per_session=600, laser_fraction=0.4, rng=7)

    def sigma(kind):
        return compute_stats(TrialArrays.from_sessions(filter_trials(animal.sessions, trial_type=kind)), ['sigma'])['sigma']

    assert sigma('opto') > sigma('non_opto') + 0.2
