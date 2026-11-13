"""Two-alternative forced choice and go/no-go: categorisation on the library's stimulus axis
([-1, 1] with the category boundary at 0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np

from behav_utils.tasks.base import ResponseSpace, Task, register_task

_SIDES = ('positive', 'negative')


def _side(stimulus, side: str, on_boundary: float) -> np.ndarray:
    """1.0 for stimuli on ``side`` of the boundary, 0.0 on the other side, ``on_boundary`` exactly at it,
    NaN where the stimulus is NaN."""
    s = np.asarray(stimulus, dtype=float)
    out = (s > 0).astype(float) if side == 'positive' else (s < 0).astype(float)
    out[s == 0] = on_boundary
    out[np.isnan(s)] = np.nan
    return out


def _check(value, allowed, what: str) -> None:
    if value not in allowed:
        raise ValueError(f"{what} must be one of {', '.join(map(repr, allowed))}, got {value!r}")


@register_task
@dataclass(frozen=True)
class TwoAFC(Task):
    """
    Two-alternative forced choice.

    Category B is the stimuli on ``b_side`` of the boundary (default the positive side), A the rest,
    and the correct response is the category. A stimulus exactly on the boundary counts as
    ``on_boundary``: 'A' (the default), 'B', or 'unscored' (no correct response, as on a rig that
    rewards at random there).

    Attributes:
        b_side: The side of the boundary that is category B, 'positive' or 'negative'.
        on_boundary: What a stimulus exactly on the boundary counts as: 'A', 'B' or 'unscored'.
    """
    name: ClassVar[str] = 'two_afc'
    capabilities: ClassVar[frozenset] = frozenset({'binary', 'two_choices', 'stimulus_axis', 'scored'})
    b_side: str = 'positive'
    on_boundary: str = 'A'

    def __post_init__(self):
        _check(self.b_side, _SIDES, 'b_side')
        _check(self.on_boundary, ('A', 'B', 'unscored'), 'on_boundary')

    @property
    def response_space(self) -> ResponseSpace:
        return ResponseSpace('binary', ('A', 'B'))

    def category(self, stimulus, context: Mapping[str, Any] | None = None) -> np.ndarray:
        return _side(stimulus, self.b_side, {'A': 0.0, 'B': 1.0, 'unscored': np.nan}[self.on_boundary])

    def truth(self, stimulus, context: Mapping[str, Any] | None = None) -> np.ndarray:
        return self.category(stimulus, context)


@register_task
@dataclass(frozen=True)
class GoNoGo(Task):
    """
    Go/no-go.

    The correct response is go (1) to stimuli on ``go_side`` of the boundary and no-go (0,
    withholding) to the rest. A blank response that is not an abort is a withhold, so it is scored, not
    lost. A stimulus exactly on the boundary counts as ``on_boundary``: 'no_go' (the default), 'go' or
    'unscored'.

    Attributes:
        go_side: The side of the boundary where go is correct, 'positive' or 'negative'.
        on_boundary: What a stimulus exactly on the boundary counts as: 'no_go', 'go' or 'unscored'.
    """
    name: ClassVar[str] = 'go_nogo'
    capabilities: ClassVar[frozenset] = frozenset({'binary', 'withhold', 'stimulus_axis', 'scored'})
    go_side: str = 'positive'
    on_boundary: str = 'no_go'

    def __post_init__(self):
        _check(self.go_side, _SIDES, 'go_side')
        _check(self.on_boundary, ('no_go', 'go', 'unscored'), 'on_boundary')

    @property
    def response_space(self) -> ResponseSpace:
        return ResponseSpace('binary', ('no-go', 'go'))

    def category(self, stimulus, context: Mapping[str, Any] | None = None) -> np.ndarray:
        return _side(stimulus, self.go_side, {'no_go': 0.0, 'go': 1.0, 'unscored': np.nan}[self.on_boundary])

    def truth(self, stimulus, context: Mapping[str, Any] | None = None) -> np.ndarray:
        return self.category(stimulus, context)

    def complete_responses(self, response, abort, context: Mapping[str, Any] | None = None) -> np.ndarray:
        out = np.asarray(response, dtype=float).copy()
        out[np.isnan(out) & ~np.asarray(abort, dtype=bool)] = 0.0
        return out
