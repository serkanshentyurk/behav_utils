"""The continuous-response family (step 10a): the Reproduction task, one linear map for stimulus and
response, TrialArrays.truth, the error statistics, and the ReproductionModel's parameters recovered."""

import numpy as np
import pandas as pd
import pytest

from behav_utils import (
    Outcome,
    Reproduction,
    ReproductionModel,
    TrialArrays,
    compute_psychometric_curve,
    compute_stats,
    filter_trials,
    generate_synthetic_animal,
    generate_synthetic_session,
    list_stats,
    load_config,
    load_experiment,
)
from behav_utils.config.schema import ConfigError
from behav_utils.data import session_from_arrays
from behav_utils.models import TrialwiseModel
from behav_utils.tasks import RequirementError, Uniform

CONTINUOUS_STATS = ['response_bias', 'error_sd', 'mean_abs_error', 'response_slope', 'serial_bias']


def _config(tmp_path, stim, resp, task_block):
    rows = pd.DataFrame({'Animal': 'R1', 'Session': 'r1_s1', 'Date': '2026-03-02', 'TrialNum': range(1, len(stim) + 1),
                         'Pos': stim, 'Poke': resp})
    rows.to_csv(tmp_path / 'trials.csv', index=False)
    path = tmp_path / 'config.yaml'
    path.write_text(f"""
project: {{name: reproduction}}
reader: {{type: table, path: trials.csv, animal_column: Animal, session_column: Session, date_column: Date}}
task:
  type: reproduction
  {task_block}
  primary_stimulus: stimulus
  primary_choice: choice
  choice_mapping: {{type: identity, no_response_value: null}}
columns:
  trial_number: {{csv_name: TrialNum, dtype: int}}
  stimulus: {{csv_name: Pos, dtype: float}}
  choice: {{csv_name: Poke, dtype: float}}
""")
    return path


def _stats(session, names):
    return compute_stats(TrialArrays.from_sessions([session]), names)


def test_a_linear_range_maps_stimulus_and_response_alike(tmp_path):
    stim = np.linspace(5, 90, 40)                                 # every poke stays on the line
    resp = stim + np.tile([-4.0, 6.0], 20)                      # 4 cm and 6 cm misses on a 0-100 cm line
    path = _config(tmp_path, stim, resp, 'tolerance: 0.1\n  stimulus_range: [0, 100]')
    trials = load_experiment(path).animals['R1'].sessions[0].trials
    np.testing.assert_allclose(trials.stimulus, (2 * stim - 100) / 100)
    np.testing.assert_allclose(trials.choice, (2 * resp - 100) / 100)
    assert trials.correct.tolist() == [True, False] * 20          # a tolerance of 0.1 on the axis is 5 cm


def test_a_log_range_maps_stimulus_and_response_alike(tmp_path):
    stim = np.geomspace(2500, 12000, 30)
    path = _config(tmp_path, stim, stim * 1.02, 'tolerance: 0.05\n  stimulus_range: [2000, 16000]\n  stimulus_scale: log')
    trials = load_experiment(path).animals['R1'].sessions[0].trials
    expected = (2 * np.log(stim) - np.log(2000 * 16000)) / np.log(16000 / 2000)
    np.testing.assert_allclose(trials.stimulus, expected)
    np.testing.assert_allclose(trials.choice - trials.stimulus, 2 * np.log(1.02) / np.log(8))


def test_a_category_boundary_or_side_mapping_is_refused(tmp_path):
    with pytest.raises(ConfigError, match='no category boundary'):
        load_config(_config(tmp_path, [1.0], [1.0], 'stimulus_range: [0, 100]\n  boundary: 30'))
    path = _config(tmp_path, [1.0], [1.0], '')
    path.write_text(path.read_text().replace('{type: identity, no_response_value: null}',
                                             '{type: spatial_to_category, no_response_value: null, contingency_field: rule}'))
    with pytest.raises(ConfigError, match="choice_mapping type must be 'identity'"):
        load_config(path)


def test_a_response_outside_the_range_is_kept_with_a_warning(tmp_path):
    stim = np.linspace(10, 90, 20)
    resp = stim.copy()
    resp[3] = 120.0
    with pytest.warns(UserWarning, match='1 response'):
        trials = load_experiment(_config(tmp_path, stim, resp, 'stimulus_range: [0, 100]')).animals['R1'].sessions[0].trials
    assert trials.choice[3] == pytest.approx(1.4)


def test_the_tolerance_includes_its_boundary():
    session = session_from_arrays(np.array([0.0, 0.0, 0.5]), np.array([0.25, 0.26, 0.25]), task=Reproduction(tolerance=0.25))
    assert session.trials.outcome.tolist() == [Outcome.CORRECT, Outcome.ERROR, Outcome.CORRECT]
    with pytest.raises(ValueError, match='positive'):
        Reproduction(tolerance=0)


def test_the_model_parameters_come_back():
    inner = Uniform(-0.8, 0.8)                                    # away from the ends, so nothing is clipped
    tilted = generate_synthetic_session(ReproductionModel(bias=0.1, slope=0.8, noise=0.05), task=Reproduction(),
                                        distribution=inner, n_trials=6000, rng=0)
    stats = _stats(tilted, ['response_bias', 'response_slope', 'serial_bias'])
    assert stats['response_bias'] == pytest.approx(0.1, abs=0.005)
    assert stats['response_slope'] == pytest.approx(0.8, abs=0.01)
    assert stats['serial_bias'] == pytest.approx(0.0, abs=0.02)
    faithful = generate_synthetic_session(ReproductionModel(noise=0.05), task=Reproduction(), distribution=inner,
                                          n_trials=6000, rng=1)
    stats = _stats(faithful, ['error_sd', 'mean_abs_error'])
    assert stats['error_sd'] == pytest.approx(0.05, abs=0.002)
    assert stats['mean_abs_error'] == pytest.approx(0.05 * np.sqrt(2 / np.pi), abs=0.002)


class Attracted(TrialwiseModel):
    """Pulled towards the middle (slope 0.8) and towards the previous stimulus by ``w``."""
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


def test_serial_bias_finds_attraction_without_mistaking_the_pull_to_the_middle():
    session = generate_synthetic_session(Attracted(0.15), task=Reproduction(), distribution=Uniform(-0.8, 0.8),
                                         n_trials=6000, rng=2)
    assert _stats(session, ['serial_bias'])['serial_bias'] == pytest.approx(0.15, abs=0.02)


def test_statistics_and_readouts_stay_in_their_family():
    reproduction = TrialArrays.from_sessions([generate_synthetic_session(ReproductionModel(), task=Reproduction(),
                                                                         n_trials=400, rng=3)])
    two_afc = TrialArrays.from_sessions([generate_synthetic_session(n_trials=400, rng=3)])
    applicable = list_stats(task=Reproduction())
    assert set(CONTINUOUS_STATS) <= set(applicable) and 'accuracy' in applicable and 'reaction_time' in applicable
    assert 'side_bias' not in applicable and 'mu' not in applicable
    assert np.isfinite(compute_stats(reproduction, CONTINUOUS_STATS + ['accuracy'])).all()
    with pytest.raises(RequirementError, match='response_bias needs continuous'):
        compute_stats(two_afc, ['response_bias'])
    with pytest.raises(RequirementError, match='side_bias needs'):
        compute_stats(reproduction, ['side_bias'])
    with pytest.raises(RequirementError, match='compute_psychometric_curve needs'):
        compute_psychometric_curve(reproduction)


def test_truth_is_carried_per_trial():
    two_afc = TrialArrays.from_sessions([generate_synthetic_session(n_trials=100, rng=4)])
    np.testing.assert_array_equal(two_afc.truth, two_afc.category)
    reproduction = TrialArrays.from_sessions([generate_synthetic_session(ReproductionModel(), task=Reproduction(),
                                                                         n_trials=100, rng=4)])
    np.testing.assert_array_equal(reproduction.truth, reproduction.stimulus)
    part = reproduction.take(np.arange(10))
    np.testing.assert_array_equal(part.truth, reproduction.stimulus[:10])
    np.testing.assert_array_equal(TrialArrays.concat([part, part]).truth, np.tile(reproduction.stimulus[:10], 2))


def test_a_laser_override_reaches_the_opto_split():
    animal = generate_synthetic_animal(ReproductionModel(noise=0.05, laser={'noise': 0.2}), n_sessions=4,
                                       trials_per_session=800, task=Reproduction(), distribution=Uniform(-0.5, 0.5),
                                       laser_fraction=0.4, rng=5)

    def spread(kind):
        return compute_stats(TrialArrays.from_sessions(filter_trials(animal.sessions, trial_type=kind)),
                             ['error_sd'])['error_sd']

    assert spread('opto') == pytest.approx(0.2, abs=0.02) and spread('non_opto') == pytest.approx(0.05, abs=0.005)
