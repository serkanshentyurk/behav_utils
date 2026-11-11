"""
behav_utils.models — behavioural models and simulation.

A model is anything with ``simulate(design, rng) -> Responses``; a ``Design`` is a run without its
responses (task, stimuli, laser, abort and no-response flags), drawn by ``make_design``.
``simulate(model, design)`` returns a ``SessionData`` scored by the task, so synthetic data goes through
the same statistics as real data; ``simulate_responses`` returns only the choices.

    design = make_design(TwoAFC(), Uniform(), 400, laser_fraction=0.3, abort_rate=0.05)
    session = simulate(PsychometricModel(mu=0.1, sigma=0.25), design, rng)

``TrialwiseModel`` runs the trial loop for models that learn (write ``p_choice``, and ``update``);
``IdealObserver`` and ``normative_pse`` give the Bayesian benchmark for a stimulus distribution.
"""

from behav_utils.models.base import Design, Model, Responses, TrialwiseModel, make_design
from behav_utils.models.builtin import IdealObserver, PsychometricModel, RandomModel, normative_pse
from behav_utils.models.simulate import simulate, simulate_responses

__all__ = ['Model', 'TrialwiseModel', 'Responses', 'Design', 'make_design', 'simulate', 'simulate_responses',
           'PsychometricModel', 'RandomModel', 'IdealObserver', 'normative_pse']
