"""The continuous-response family's built-in model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping

import numpy as np

from behav_utils.models.base import Design, Model, Responses
from behav_utils.models.builtin import _check_laser, _per_trial


@dataclass(frozen=True)
class ReproductionModel(Model):
    """
    A reproducer: response = slope × stimulus + bias + Gaussian noise (sd ``noise``), clipped to the axis
    [-1, 1]; with probability ``lapse`` the response is a uniformly random location instead. A slope
    below 1 pulls responses towards the middle. ``laser`` overrides any of the four on laser trials.
    Simulate, then compute ``response_bias``, ``response_slope`` and ``error_sd``, and they come back.
    """
    bias: float = 0.0
    slope: float = 1.0
    noise: float = 0.1
    lapse: float = 0.0
    laser: Mapping[str, float] | None = None
    requires: ClassVar[frozenset] = frozenset({'continuous'})

    _NAMES = ('bias', 'slope', 'noise', 'lapse')

    def __post_init__(self):
        _check_laser(self, self._NAMES)
        if self.noise < 0:
            raise ValueError(f'ReproductionModel: noise must be non-negative, got {self.noise}')
        if not 0.0 <= self.lapse <= 1.0:
            raise ValueError(f'ReproductionModel: lapse must lie in [0, 1], got {self.lapse}')

    def simulate(self, design: Design, rng: np.random.Generator) -> Responses:
        p = _per_trial(self, design, self._NAMES)
        n = design.n_trials
        response = p['slope'] * design.stimulus + p['bias'] + p['noise'] * rng.standard_normal(n)
        lapsed = rng.random(n) < p['lapse']
        response = np.where(lapsed, rng.uniform(-1.0, 1.0, n), response)
        return Responses(np.clip(response, -1.0, 1.0))
