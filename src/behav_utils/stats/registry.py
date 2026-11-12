"""
Statistic registry.

Contract
--------
A **statistic** is a float computed from a :class:`TrialArrays`. Every
producer has the same signature::

    def f(arrays: TrialArrays, *, rng: np.random.Generator) -> float
    def g(arrays: TrialArrays, *, rng: np.random.Generator) -> tuple[float, ...]

``rng`` is always passed; deterministic producers ignore it.

Two decorators register producers:

- ``@stat(name, exchangeable=True, requires=(), instead=None)`` — one scalar.
- ``@fit(name, outputs=(...), exchangeable=True)`` — one fitter yielding
  several named scalars (a psychometric fit gives ``mu, sigma, lapse_low,
  lapse_high``). Each output is a first-class stat name; the fitter runs once
  per :func:`compute_stats` call however many of its outputs are requested.

``compute_stats(arrays, names)`` returns a float ``pd.Series`` indexed by the
requested names, in request order. Nothing else is ever returned: no dicts,
no arrays, no family names. Array-valued readouts (update matrix, curves)
live in ``behav_utils.readouts``, not here.

``requires`` names the task capabilities a producer needs (``behav_utils.tasks.CAPABILITIES``, or
names of your own); none means it runs on every task. ``compute_stats`` refuses an explicitly
requested statistic whose needs the arrays' task lacks, with ``RequirementError``, suggesting the
``instead`` counterpart (``{capability: name}``) for a task that has that capability.
``compute_stats(arrays)`` and ``list_stats(task=...)`` give only the statistics that apply.

``exchangeable`` records whether trial-level resampling is valid for the
producer: True for memoryless and lag-1 statistics (they use the frozen
``prev_*`` view), False for anything that depends on trial order beyond
lag 1 (multi-lag regressions, run-length measures). Resamplers consult
:func:`is_exchangeable` and refuse to trial-resample a non-exchangeable stat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from behav_utils.data.arrays import TrialArrays
from behav_utils.tasks import require

Producer = Callable[..., object]


@dataclass(frozen=True)
class _Entry:
    name: str                 # producer name ('psychometric', 'accuracy')
    func: Producer
    outputs: Tuple[str, ...]  # scalar names this producer yields, in order
    exchangeable: bool
    requires: frozenset = frozenset()        # task capabilities it needs
    instead: Mapping[str, str] | None = None  # capability -> counterpart, for the refusal message


_PRODUCERS: Dict[str, _Entry] = {}       # producer name -> entry
_OUTPUT_TO_PRODUCER: Dict[str, str] = {}  # scalar name  -> producer name


def _register(entry: _Entry) -> None:
    if entry.name in _PRODUCERS:
        raise ValueError(f"stat producer '{entry.name}' is already registered")
    for out in entry.outputs:
        if out in _OUTPUT_TO_PRODUCER:
            raise ValueError(
                f"stat name '{out}' is already produced by "
                f"'{_OUTPUT_TO_PRODUCER[out]}'; cannot also register it under '{entry.name}'")
    _PRODUCERS[entry.name] = entry
    for out in entry.outputs:
        _OUTPUT_TO_PRODUCER[out] = entry.name


def stat(name: str, *, exchangeable: bool = True, requires: Iterable[str] = (),
         instead: Mapping[str, str] | None = None):
    """Register a single-scalar statistic ``f(arrays, *, rng) -> float``. ``requires``: the task
    capabilities it needs (none: every task); ``instead``: {capability: counterpart} for refusals."""
    def decorator(func: Producer) -> Producer:
        _register(_Entry(name=name, func=func, outputs=(name,), exchangeable=exchangeable,
                         requires=frozenset(requires), instead=dict(instead or {})))
        return func
    return decorator


def fit(name: str, *, outputs: Sequence[str], exchangeable: bool = True, requires: Iterable[str] = (),
        instead: Mapping[str, str] | None = None):
    """Register a multi-output fitter ``g(arrays, *, rng) -> tuple`` of len(outputs); ``requires`` and
    ``instead`` as for ``stat``."""
    outputs = tuple(outputs)
    if not outputs:
        raise ValueError(f"fit '{name}': outputs must be non-empty")

    def decorator(func: Producer) -> Producer:
        _register(_Entry(name=name, func=func, outputs=outputs, exchangeable=exchangeable,
                         requires=frozenset(requires), instead=dict(instead or {})))
        return func
    return decorator


# ── queries ─────────────────────────────────────────────────────────────────

def list_stats(task=None) -> List[str]:
    """Every scalar name that can be requested from :func:`compute_stats`; with ``task``, only those
    whose needs the task provides."""
    if task is None:
        return list(_OUTPUT_TO_PRODUCER)
    return [n for n, p in _OUTPUT_TO_PRODUCER.items() if _PRODUCERS[p].requires <= task.capabilities]


def _list_producers() -> Dict[str, Tuple[str, ...]]:
    """Producer name -> the scalar names it yields."""
    return {k: v.outputs for k, v in _PRODUCERS.items()}


def _producer_of(name: str) -> str:
    """Which producer yields scalar ``name``."""
    try:
        return _OUTPUT_TO_PRODUCER[name]
    except KeyError:
        raise KeyError(f"unknown stat '{name}'. Known: {list_stats()}") from None


def is_exchangeable(name: str) -> bool:
    """Whether trial-level resampling is valid for scalar ``name``."""
    return _PRODUCERS[_producer_of(name)].exchangeable


def _validate_names(names: Iterable[str]) -> List[str]:
    """Return ``names`` as a list, raising on the first unknown one."""
    names = list(names)
    for n in names:
        _producer_of(n)
    return names


# ── compute ─────────────────────────────────────────────────────────────────

def compute_stats(
    arrays: TrialArrays,
    names: Sequence[str] | None = None,
    *,
    rng: np.random.Generator | None = None,
    strict: bool = True,
) -> pd.Series:
    """Compute scalar statistics on one block of trials.

    Args:
        arrays: the trials (pre-filtered).
        names:  scalar stat names. Multi-output fitters run once regardless of
                how many of their outputs appear here. None: every statistic that
                applies to ``arrays.task``. A requested name whose needs the task
                lacks raises ``RequirementError``, whatever ``strict`` says.
        rng:    generator handed to every producer (stochastic ones draw from
                it; pass a seeded one for reproducibility). A fresh default
                generator is used when None.
        strict: True raises on a producer error. False fills that producer's
                outputs with NaN and continues — the resampling-loop path,
                where a failed refit is a discarded draw, not a crash.

    Returns:
        float ``pd.Series`` indexed by ``names`` in request order.
    """
    task = arrays.task
    if names is None:
        names = list_stats(task=task)
    names = _validate_names(names)
    for n in names:
        entry = _PRODUCERS[_OUTPUT_TO_PRODUCER[n]]
        if not entry.requires <= task.capabilities:
            require(task, entry.requires, n, entry.instead)
    if rng is None:
        rng = np.random.default_rng()

    wanted: Dict[str, None] = {}  # ordered set of producers to run
    for n in names:
        wanted.setdefault(_OUTPUT_TO_PRODUCER[n], None)

    values: Dict[str, float] = {}
    for pname in wanted:
        entry = _PRODUCERS[pname]
        try:
            out = entry.func(arrays, rng=rng)
        except Exception:
            if strict:
                raise
            for o in entry.outputs:
                values[o] = np.nan
            continue

        if len(entry.outputs) == 1:
            values[entry.outputs[0]] = float(out)
        else:
            out = tuple(out)
            if len(out) != len(entry.outputs):
                raise RuntimeError(
                    f"fit '{pname}' returned {len(out)} values, declared {len(entry.outputs)}")
            for o, v in zip(entry.outputs, out):
                values[o] = float(v)

    return pd.Series([values[n] for n in names], index=list(names), dtype=float, name='value')
