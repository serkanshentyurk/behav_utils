"""
Stimulus distributions on the library's axis ([-1, 1], boundary at 0).

A distribution says how a design draws stimuli, and what an observer believes about them. Each has
``sample(n, rng)``, ``pdf(x)`` and ``support`` (low, high); ``grid()`` turns it into points and weights
that sum to 1, which an observer sums over. Subclass ``StimulusDistribution`` for others (a project's
own distributions, say); ``Discrete`` covers any finite set of stimuli with weights.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class StimulusDistribution(ABC):
    """A distribution of stimuli: ``sample``, ``pdf`` and ``support``; ``grid`` comes with it."""

    @property
    @abstractmethod
    def support(self) -> tuple:
        """(low, high): where the stimuli lie."""

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` stimuli."""

    @abstractmethod
    def pdf(self, x) -> np.ndarray:
        """The density at ``x`` (the probability mass, for a discrete distribution)."""

    def grid(self, n: int = 2000) -> tuple:
        """(points, weights): the distribution at the midpoints of ``n`` equal cells over its support,
        weights summing to 1. Midpoints keep a symmetric distribution symmetric about the boundary,
        which is never itself a point."""
        low, high = self.support
        points = low + (np.arange(n) + 0.5) * (high - low) / n
        weights = np.asarray(self.pdf(points), dtype=float)
        if not weights.sum() > 0:
            raise ValueError(f'{self!r}: the density is zero everywhere on its support')
        return points, weights / weights.sum()


def _check_interval(low: float, high: float, what: str) -> None:
    if not low < high:
        raise ValueError(f'{what}: low must be below high, got [{low}, {high}]')


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class Uniform(StimulusDistribution):
    """Uniform on [low, high]."""
    low: float = -1.0
    high: float = 1.0

    def __post_init__(self):
        _check_interval(self.low, self.high, 'Uniform')

    @property
    def support(self) -> tuple:
        return (self.low, self.high)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, n)

    def pdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where((x >= self.low) & (x <= self.high), 1.0 / (self.high - self.low), 0.0)


@dataclass(frozen=True)
class TruncatedNormal(StimulusDistribution):
    """A normal (``mean``, ``sd``) restricted to [low, high]."""
    mean: float = 0.0
    sd: float = 0.3
    low: float = -1.0
    high: float = 1.0

    def __post_init__(self):
        _check_interval(self.low, self.high, 'TruncatedNormal')
        if not self.sd > 0:
            raise ValueError(f'TruncatedNormal: sd must be positive, got {self.sd}')
        if self._mass() < 1e-6:
            raise ValueError(f'TruncatedNormal: almost none of N({self.mean}, {self.sd}) lies in '
                             f'[{self.low}, {self.high}]')

    @property
    def support(self) -> tuple:
        return (self.low, self.high)

    def _mass(self) -> float:
        return _normal_cdf((self.high - self.mean) / self.sd) - _normal_cdf((self.low - self.mean) / self.sd)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty(0)
        while out.size < n:
            draw = rng.normal(self.mean, self.sd, max(2 * (n - out.size), 16))
            out = np.concatenate([out, draw[(draw >= self.low) & (draw <= self.high)]])
        return out[:n]

    def pdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        density = (np.exp(-0.5 * ((x - self.mean) / self.sd) ** 2)
                   / (self.sd * math.sqrt(2.0 * math.pi) * self._mass()))
        return np.where((x >= self.low) & (x <= self.high), density, 0.0)


@dataclass(frozen=True)
class Discrete(StimulusDistribution):
    """A finite set of stimuli ``values`` drawn with ``weights`` (equal if omitted; normalised)."""
    values: tuple
    weights: tuple | None = None

    def __post_init__(self):
        values = tuple(float(v) for v in self.values)
        weights = tuple(float(w) for w in self.weights) if self.weights is not None else (1.0,) * len(values)
        if not values or len(weights) != len(values) or min(weights) < 0 or not sum(weights) > 0:
            raise ValueError('Discrete: values must be non-empty, with as many non-negative weights '
                             'summing above 0')
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'weights', tuple(w / sum(weights) for w in weights))

    @property
    def support(self) -> tuple:
        return (min(self.values), max(self.values))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(np.asarray(self.values), size=n, p=np.asarray(self.weights))

    def pdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros(x.shape)
        for value, weight in zip(self.values, self.weights):
            out[np.isclose(x, value)] += weight
        return out

    def grid(self, n: int = 2000) -> tuple:
        return np.asarray(self.values), np.asarray(self.weights)
