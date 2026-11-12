"""Generality check before the 1.0 freeze: a three-choice task and a continuous-estimation task, with
their models, all defined here, run through loading, scoring, discovery, pooling, statistics, readouts,
simulate() and the generators with no library change."""

import dataclasses
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from behav_utils import (
    PsychometricModel,
    TrialArrays,
    compute_psychometric_curve,
    compute_stats,
    discover_sessions,
    generate_synthetic_animal,
    generate_synthetic_session,
    list_stats,
    load_experiment,
    make_design,
    simulate,
)
from behav_utils.models import Model, Responses, TrialwiseModel, normative_pse
from behav_utils.tasks import Outcome, RequirementError, ResponseSpace, Task, Uniform, register_task


@register_task
@dataclasses.dataclass(frozen=True)
class ThreeChoice(Task):
    """Three categories on the stimulus axis: 0 below ``low``, 2 above ``high``, 1 between."""
    name: ClassVar[str] = 'three_choice_for_tests'
    capabilities: ClassVar[frozenset] = frozenset({'k_choices', 'scored'})
    low: float = -1 / 3
    high: float = 1 / 3

    @property
    def response_space(self):
        return ResponseSpace('categorical', ('low', 'middle', 'high'))

    def category(self, stimulus, context=None):
        s = np.asarray(stimulus, dtype=float)
        out = np.where(s < self.low, 0.0, np.where(s > self.high, 2.0, 1.0))
        return np.where(np.isnan(s), np.nan, out)

    def truth(self, stimulus, context=None):
        return self.category(stimulus, context)


@register_task
@dataclasses.dataclass(frozen=True)
class Reproduction(Task):
    """Reproduce the stimulus: correct within ``tolerance`` of it."""
    name: ClassVar[str] = 'reproduction_for_tests'
    capabilities: ClassVar[frozenset] = frozenset({'continuous', 'scored'})
    tolerance: float = 0.1

    @property
    def response_space(self):
        return ResponseSpace('continuous')

    def category(self, stimulus, context=None):
        return np.full(len(np.asarray(stimulus)), np.nan)

    def truth(self, stimulus, context=None):
        return np.asarray(stimulus, dtype=float)

    def score(self, response, truth, abort, context=None):
        out = super().score(response, truth, abort, context)
        scored = (out == Outcome.CORRECT) | (out == Outcome.ERROR)
        close = np.abs(np.asarray(response, dtype=float) - np.asarray(truth, dtype=float)) <= self.tolerance
        out[scored & close] = Outcome.CORRECT
        out[scored & ~close] = Outcome.ERROR
        return out


class SoftmaxThreeChoice(Model):
    """Chooses among the three categories with probabilities falling off with distance from each centre."""
    requires = frozenset({'k_choices'})

    def simulate(self, design, rng):
        centres = np.array([-2 / 3, 0.0, 2 / 3])
        logits = -((design.stimulus[:, None] - centres[None, :]) ** 2) / 0.05
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        draws = rng.random(design.n_trials)[:, None]
        return Responses((draws > np.cumsum(p, axis=1)).sum(axis=1).astype(float))


class NoisyReproduction(Model):
    """Reproduces the stimulus with Gaussian noise."""
    requires = frozenset({'continuous'})

    def simulate(self, design, rng):
        return Responses(design.stimulus + rng.normal(0.0, 0.05, design.n_trials))


class BiasedLearner(TrialwiseModel):
    """Starts with a bias and learns it away from each trial's error (uses the truth in update)."""
    requires = frozenset({'continuous'})

    def reset(self):
        self.bias = 0.5

    def respond(self, trial, rng):
        return trial.stimulus + self.bias + rng.normal(0.0, 0.02)

    def update(self, trial):
        self.bias -= 0.2 * (trial.choice - trial.truth)


def _table_config(tmp_path, task_block, rows):
    table = tmp_path / 'trials.csv'
    rows.to_csv(table, index=False)
    config = tmp_path / 'config.yaml'
    config.write_text(f"""
project: {{name: generality}}
reader: {{type: table, path: trials.csv, animal_column: Animal, session_column: Session, date_column: Date}}
task:
  {task_block}
  primary_stimulus: stimulus
  primary_choice: choice
  choice_mapping: {{type: identity, no_response_value: null}}
columns:
  trial_number: {{csv_name: TrialNum, dtype: int}}
  stimulus: {{csv_name: Stim, dtype: float}}
  choice: {{csv_name: Resp, dtype: float}}
""")
    return config


def _trials(responses, seed=0, n=200):
    rng = np.random.default_rng(seed)
    stim = rng.uniform(-1, 1, n)
    return pd.DataFrame({'Animal': 'K1', 'Session': 'k1_s1', 'Date': '2026-02-02', 'TrialNum': range(1, n + 1),
                         'Stim': stim, 'Resp': responses(stim, rng)})


def test_a_three_choice_task_loads_scores_and_is_discovered(tmp_path):
    rows = _trials(lambda s, rng: np.where(s < -1 / 3, 0, np.where(s > 1 / 3, 2, 1)).astype(float))
    rows.loc[::10, 'Resp'] = (rows.loc[::10, 'Resp'] + 1) % 3            # every tenth trial wrong
    config = _table_config(tmp_path, 'type: three_choice_for_tests', rows)
    session = load_experiment(config).animals['K1'].sessions[0]
    assert session.task == ThreeChoice()
    assert set(np.unique(session.trials.outcome)) == {Outcome.ERROR, Outcome.CORRECT}
    assert compute_stats(TrialArrays.from_sessions([session]), ['accuracy'])['accuracy'] == pytest.approx(0.9)
    assert discover_sessions(config)['task'].item() == 'three_choice_for_tests'


def test_binary_statistics_and_readouts_are_refused_on_three_choices(tmp_path):
    arrays = TrialArrays.from_sessions([generate_synthetic_session(SoftmaxThreeChoice(), task=ThreeChoice(),
                                                                   n_trials=600, rng=0)])
    assert set(np.unique(arrays.valid().choice)) == {0.0, 1.0, 2.0}
    applicable = list_stats(task=ThreeChoice())
    assert 'accuracy' in applicable and 'side_bias' not in applicable and 'mu' not in applicable
    assert compute_stats(arrays)['accuracy'] > 0.6
    with pytest.raises(RequirementError, match='side_bias needs'):
        compute_stats(arrays, ['side_bias'])
    with pytest.raises(RequirementError, match='compute_psychometric_curve needs'):
        compute_psychometric_curve(arrays)


def test_an_estimation_task_is_scored_by_its_own_rule(tmp_path):
    config = _table_config(tmp_path, 'type: reproduction_for_tests\n  tolerance: 0.1',
                           _trials(lambda s, rng: np.clip(s + rng.normal(0, 0.05, len(s)), -1, 1)))
    trials = load_experiment(config).animals['K1'].sessions[0].trials
    within = np.abs(trials.choice - trials.stimulus) <= 0.1
    np.testing.assert_array_equal(trials.correct, within)


def test_estimation_models_simulate_and_a_learner_uses_the_truth():
    design = make_design(Reproduction(), Uniform(), 400, rng=np.random.default_rng(1))
    session = simulate(NoisyReproduction(), design, np.random.default_rng(2))
    assert compute_stats(TrialArrays.from_sessions([session]), ['accuracy'])['accuracy'] > 0.9
    learner = simulate(BiasedLearner(), design, np.random.default_rng(3)).trials
    errors = np.abs(learner.choice - learner.stimulus)
    assert errors[:5].mean() > 0.2 and errors[-50:].mean() < 0.05
    animal = generate_synthetic_animal(NoisyReproduction(), n_sessions=2, task=Reproduction(), rng=4)
    assert animal.n_sessions == 2 and animal.sessions[0].task == Reproduction()


def test_binary_models_are_refused_on_other_tasks():
    design = make_design(Reproduction(), Uniform(), 50, rng=np.random.default_rng(5))
    with pytest.raises(RequirementError, match='PsychometricModel needs'):
        simulate(PsychometricModel(), design)
    with pytest.raises(RequirementError, match='normative_pse needs'):
        normative_pse(Uniform(), 0.3, task=ThreeChoice())


def test_three_choice_sessions_pool_and_keep_their_task():
    animal = generate_synthetic_animal(SoftmaxThreeChoice(), n_sessions=3, trials_per_session=200,
                                       task=ThreeChoice(), rng=6)
    arrays = TrialArrays.from_sessions(animal.sessions)
    assert arrays.task == ThreeChoice() and arrays.n_trials == sum(s.n_trials for s in animal.sessions)
