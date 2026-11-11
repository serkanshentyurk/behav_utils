"""
Built-in models: a psychometric observer, a random responder, and the Bayesian ideal observer with its
normative PSE.

Each takes ``laser``, a dict of parameter values that replace its own on laser trials (for example
``{'sigma': 0.6}``), so an opto pipeline can be checked against a known effect.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import ClassVar, Mapping

import numpy as np

from behav_utils.models.base import Design, Model, Responses
from behav_utils.readouts.fit import cumulative_gaussian
from behav_utils.tasks import Task, TwoAFC, require
from behav_utils.tasks.distributions import StimulusDistribution, Uniform


def _check_laser(model, names: tuple) -> None:
    unknown = sorted(set(model.laser or {}) - set(names))
    if unknown:
        raise ValueError(f'{type(model).__name__}: laser overrides {unknown}, which it does not have '
                         f'(it has {", ".join(names)})')


def _per_trial(model, design: Design, names: tuple) -> dict:
    """Each parameter as a per-trial array: the model's value, or its laser override on laser trials."""
    overrides = model.laser or {}
    return {name: np.where(design.laser, overrides.get(name, getattr(model, name)),
                           getattr(model, name)).astype(float)
            for name in names}


@dataclass(frozen=True)
class PsychometricModel(Model):
    """
    Choices from the psychometric curve the library fits (``fit_psychometric``):
    P(response 1 | s) = lapse_low + (1 - lapse_low - lapse_high) * Phi((s - mu) / sigma).
    Simulate, then fit, and the parameters come back.
    """
    mu: float = 0.0
    sigma: float = 0.3
    lapse_low: float = 0.02
    lapse_high: float = 0.02
    laser: Mapping[str, float] | None = None
    requires: ClassVar[frozenset] = frozenset({'binary', 'stimulus_axis'})

    _NAMES = ('mu', 'sigma', 'lapse_low', 'lapse_high')

    def __post_init__(self):
        _check_laser(self, self._NAMES)

    def simulate(self, design: Design, rng: np.random.Generator) -> Responses:
        p = _per_trial(self, design, self._NAMES)
        prob = cumulative_gaussian(design.stimulus, p['mu'], p['sigma'], p['lapse_low'], p['lapse_high'])
        return Responses((rng.random(design.n_trials) < prob).astype(float))


@dataclass(frozen=True)
class RandomModel(Model):
    """Response 1 with probability ``p`` on every trial, whatever the stimulus."""
    p: float = 0.5
    laser: Mapping[str, float] | None = None
    requires: ClassVar[frozenset] = frozenset({'binary'})

    def __post_init__(self):
        _check_laser(self, ('p',))

    def simulate(self, design: Design, rng: np.random.Generator) -> Responses:
        prob = _per_trial(self, design, ('p',))['p']
        return Responses((rng.random(design.n_trials) < prob).astype(float))


def _evidence_difference(x: np.ndarray, noise: np.ndarray, distribution: StimulusDistribution,
                         task: Task) -> np.ndarray:
    """For each percept ``x`` (with its noise sd): the prior-weighted likelihood of category 1 minus that
    of category 0, under ``distribution`` and the task's categories. Positive means response 1."""
    points, weights = distribution.grid()
    category = task.category(points)
    w1 = np.where(category == 1, weights, 0.0)
    w0 = np.where(category == 0, weights, 0.0)
    out = np.empty(len(x))
    for start in range(0, len(x), 2000):
        chunk = slice(start, start + 2000)
        like = np.exp(-0.5 * ((x[chunk, None] - points[None, :]) / noise[chunk, None]) ** 2)
        out[chunk] = like @ w1 - like @ w0
    return out


@dataclass(frozen=True)
class IdealObserver(Model):
    """
    The Bayesian observer of a binary task. It perceives each stimulus with Gaussian noise (sd
    ``noise``) and gives the response whose category has the higher posterior, under its own prior
    ``distribution`` and the design's task; with probability ``lapse`` it responds at random. Its prior
    can differ from the distribution the design draws from.
    """
    noise: float = 0.3
    distribution: StimulusDistribution = field(default_factory=Uniform)
    lapse: float = 0.0
    laser: Mapping[str, float] | None = None
    requires: ClassVar[frozenset] = frozenset({'binary', 'stimulus_axis'})

    def __post_init__(self):
        _check_laser(self, ('noise', 'lapse'))
        if not self.noise > 0:
            raise ValueError(f'IdealObserver: noise must be positive, got {self.noise}')

    def simulate(self, design: Design, rng: np.random.Generator) -> Responses:
        p = _per_trial(self, design, ('noise', 'lapse'))
        percept = design.stimulus + rng.normal(0.0, 1.0, design.n_trials) * p['noise']
        choice = (_evidence_difference(percept, p['noise'], self.distribution, design.task) > 0).astype(float)
        lapsed = rng.random(design.n_trials) < p['lapse']
        choice[lapsed] = (rng.random(int(lapsed.sum())) < 0.5).astype(float)
        return Responses(choice)


def normative_pse(distribution: StimulusDistribution, noise: float, task: Task | None = None) -> float:
    """
    The ideal observer's point of subjective equality: the percept at which both categories are equally
    probable under ``distribution`` with sensory noise ``noise``. With a likelihood that rises with the
    stimulus, the observer answers 1 above it, so it is also the PSE a psychometric fit to its choices
    estimates. NaN, with a warning, if no such point lies near the support.
    """
    task = task if task is not None else TwoAFC()
    require(task, ('binary', 'stimulus_axis'), 'normative_pse')
    low, high = distribution.support
    xs = np.linspace(low - 3 * noise, high + 3 * noise, 4001)
    d = _evidence_difference(xs, np.full(len(xs), float(noise)), distribution, task)
    crossings = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
    if crossings.size == 0:
        warnings.warn('normative_pse: the posterior never crosses 0.5 near the support', stacklevel=2)
        return np.nan
    i = crossings[np.argmin(np.abs(xs[crossings]))]
    lo, hi = xs[i], xs[i + 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        d_mid = _evidence_difference(np.array([mid]), np.array([float(noise)]), distribution, task)[0]
        if np.sign(d_mid) == np.sign(d[i]):
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))
