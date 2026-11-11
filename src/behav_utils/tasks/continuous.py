"""Continuous-response tasks: the response is a value on the stimulus scale, scored against the truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np

from behav_utils.tasks.base import Outcome, ResponseSpace, Task, register_task


@register_task
@dataclass(frozen=True)
class Reproduction(Task):
    """
    Reproduce the stimulus. The correct response is the stimulus itself, and a response is correct when
    it lies within ``tolerance`` of it, the boundary included, on the library's axis: a rat that sees a
    dot at x on a line and pokes at y is correct when |y − x| ≤ tolerance. There are no categories
    (``category`` is NaN). Its statistics are in ``behav_utils.stats.continuous``.
    """
    name: ClassVar[str] = 'reproduction'
    capabilities: ClassVar[frozenset] = frozenset({'continuous', 'scored'})
    tolerance: float = 0.1

    def __post_init__(self):
        if not self.tolerance > 0:
            raise ValueError(f'tolerance must be positive, got {self.tolerance!r}')

    @property
    def response_space(self) -> ResponseSpace:
        return ResponseSpace('continuous')

    def category(self, stimulus, context: Mapping[str, Any] | None = None) -> np.ndarray:
        return np.full(np.shape(np.asarray(stimulus, dtype=float)), np.nan)

    def truth(self, stimulus, context: Mapping[str, Any] | None = None) -> np.ndarray:
        return np.asarray(stimulus, dtype=float).copy()

    def score(self, response, truth, abort, context: Mapping[str, Any] | None = None) -> np.ndarray:
        out = super().score(response, truth, abort, context)
        scored = (out == Outcome.CORRECT) | (out == Outcome.ERROR)
        with np.errstate(invalid='ignore'):
            close = np.abs(np.asarray(response, dtype=float) - np.asarray(truth, dtype=float)) <= self.tolerance
        out[scored & close] = Outcome.CORRECT
        out[scored & ~close] = Outcome.ERROR
        return out
