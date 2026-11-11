"""
behav_utils.models — behavioural models and simulation.

A model is anything with ``simulate(design, rng) -> Responses``; a ``Design`` is a run without its
responses (task, stimuli, laser, abort and no-response flags), drawn by ``make_design``.
``simulate(model, design)`` returns a ``SessionData`` scored by the task, so synthetic data goes through
the same statistics as real data; ``simulate_responses`` returns only the choices.

    design = make_design(TwoAFC(), Uniform(), 400, laser_fraction=0.3, abort_rate=0.05)
    session = simulate(PsychometricModel(mu=0.1, sigma=0.25), design, rng)

``TrialwiseModel`` runs the trial loop for models that learn (write ``p_choice`` or ``respond``, and
``update``, each taking a ``TrialInfo``);
``IdealObserver`` and ``normative_pse`` give the Bayesian benchmark for a stimulus distribution.
``generate_synthetic_session`` / ``_animal`` / ``_experiment`` produce synthetic data with every trial
class and event times, from the built-in ``PsychometricModel`` or any model.
"""

from behav_utils.models.base import Design, Model, Responses, TrialInfo, TrialwiseModel, make_design
from behav_utils.models.builtin import IdealObserver, PsychometricModel, RandomModel, normative_pse
from behav_utils.models.continuous import ReproductionModel
from behav_utils.models.generate import (
    Timing,
    generate_synthetic_animal,
    generate_synthetic_experiment,
    generate_synthetic_session,
)
from behav_utils.models.simulate import simulate, simulate_responses

__all__ = ['Model', 'TrialwiseModel', 'TrialInfo', 'Responses', 'Design', 'make_design', 'simulate', 'simulate_responses',
           'PsychometricModel', 'RandomModel', 'IdealObserver', 'normative_pse', 'ReproductionModel',
           'generate_synthetic_session', 'generate_synthetic_animal', 'generate_synthetic_experiment', 'Timing']
