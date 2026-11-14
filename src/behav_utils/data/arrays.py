"""
TrialArrays — the one input type every statistic consumes.

A frozen, validated bundle of aligned 1-D float arrays for a block of trials:
the current trial (choice, stimulus, category, reaction_time) and its frozen
lag-1 predecessor (prev_choice, prev_stimulus, prev_category). ``prev_*`` are
NaN where a trial has no usable predecessor — the first trial of a block, or a
predecessor that was a no-response — so history statistics never bridge a
session seam or an abort on pooled data.

Construction:

    TrialArrays.from_sessions(sessions)    # the pipeline path (pool_arrays)
    TrialArrays.from_pooled(pool_arrays(sessions))
    TrialArrays.from_sequence(choice, stimulus)             # one contiguous
                                                            # block, e.g. a
                                                            # simulator output

``trial_key`` is an optional int64 identity per row: the session's 43-bit ``session_hash``
(experimenter, animal and session id) above the trial number's 20 bits, so it is the same on every
machine and every load and unique across animals. Resampling uses it to tell trials apart, and stream
alignment will key on it. ``from_sequence`` leaves it ``None``; ``trial_key_of`` builds it.

Nothing here filters: pass already-filtered sessions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np

from behav_utils.tasks import Outcome, Task, TwoAFC

TRIAL_BITS = 20      # trial numbers 0 .. 1,048,575
SESSION_BITS = 43    # 43 + 20 = 63 bits: keys stay positive int64


def session_hash(experimenter: str, animal_id: str, session_id: str) -> int:
    """A session's identity as a 43-bit integer: blake2b of 'experimenter/animal/session', the same on
    every machine and every load (Python's own ``hash`` is salted per process)."""
    digest = hashlib.blake2b(f'{experimenter}/{animal_id}/{session_id}'.encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'big') >> (64 - SESSION_BITS)


def trial_key_of(session, trial_number) -> np.ndarray:
    """
    The int64 trial keys of one session's trials: its session hash above the trial number's 20 bits.

    Args:
        session: The session; its experimenter, animal and session id are hashed.
        trial_number: Trial numbers, each between 0 and 1,048,575.

    Returns:
        One key per trial number.
    """
    numbers = np.asarray(trial_number, dtype=np.int64)
    if numbers.size and (numbers.min() < 0 or numbers.max() >= 1 << TRIAL_BITS):
        raise ValueError(f'trial numbers must lie in 0..{(1 << TRIAL_BITS) - 1}, '
                         f'got {numbers.min()}..{numbers.max()}')
    return (np.int64(session.session_hash) << np.int64(TRIAL_BITS)) | numbers


def _as_float_1d(x, n: int | None = None, name: str = '') -> np.ndarray:
    """Coerce to a 1-D float array; ``None`` becomes an all-NaN array of length n."""
    if x is None:
        if n is None:
            raise ValueError(f'{name}: cannot infer length for a missing array')
        return np.full(n, np.nan, dtype=float)
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _score(task: Task, response: np.ndarray, stimulus: np.ndarray) -> np.ndarray:
    """The task's score as 1.0 (correct) / 0.0 (error) / NaN (no response, or no correct response)."""
    codes = task.score(response, task.truth(stimulus), np.zeros(len(response), dtype=bool))
    return np.where(codes == Outcome.CORRECT, 1.0, np.where(codes == Outcome.ERROR, 0.0, np.nan))


@dataclass(frozen=True)
class TrialArrays:
    """Aligned per-trial arrays for one block of (pre-filtered) trials.

    All fields are 1-D float arrays of equal length. ``choice`` is NaN on
    no-response trials; ``prev_*`` are NaN where there is no usable predecessor.
    ``trial_key`` (int64, or None) identifies each row across resampling.

    ``correct`` and ``prev_correct`` are the task's score: 1.0 correct, 0.0 error,
    NaN where the task gives none (no response, abort, unscored). ``task`` is the
    Task that scored them; every statistic reads correctness from here, never
    from ``choice == category``. Built without a score (by hand), both are
    derived from the task (``TwoAFC()`` unless given). ``truth`` is the task's correct
    response per trial (the category in 2-AFC, the stimulus in reproduction), so error
    statistics never recompute it.

    Attributes:
        choice: The response in category space; NaN on no-response trials.
        stimulus: The stimulus on the library's axis.
        category: The task's category for the stimulus.
        prev_choice: The previous trial's choice; NaN where there is no usable predecessor.
        prev_stimulus: The previous trial's stimulus, likewise.
        prev_category: The previous trial's category, likewise.
        reaction_time: Response latency in seconds.
        trial_key: The row's identity across resampling (int64), or None.
        correct: The task's score: 1.0 correct, 0.0 error, NaN where the task gives none.
        prev_correct: The previous trial's score, likewise.
        task: The ``Task`` that scored the trials.
        truth: The task's correct response per trial.
    """

    choice: np.ndarray
    stimulus: np.ndarray
    category: np.ndarray
    prev_choice: np.ndarray = field(default=None)          # type: ignore[assignment]
    prev_stimulus: np.ndarray = field(default=None)        # type: ignore[assignment]
    prev_category: np.ndarray = field(default=None)        # type: ignore[assignment]
    reaction_time: np.ndarray = field(default=None)        # type: ignore[assignment]
    trial_key: np.ndarray | None = field(default=None)
    correct: np.ndarray | None = field(default=None)
    prev_correct: np.ndarray | None = field(default=None)
    task: Task | None = field(default=None)
    truth: np.ndarray | None = field(default=None)

    def __post_init__(self):
        choice = _as_float_1d(self.choice, name='choice')
        n = len(choice)
        fields = {
            'choice': choice,
            'stimulus': _as_float_1d(self.stimulus, n, 'stimulus'),
            'category': _as_float_1d(self.category, n, 'category'),
            'prev_choice': _as_float_1d(self.prev_choice, n, 'prev_choice'),
            'prev_stimulus': _as_float_1d(self.prev_stimulus, n, 'prev_stimulus'),
            'prev_category': _as_float_1d(self.prev_category, n, 'prev_category'),
            'reaction_time': _as_float_1d(self.reaction_time, n, 'reaction_time'),
        }
        for k, v in fields.items():
            if len(v) != n:
                raise ValueError(f'TrialArrays: {k} has length {len(v)}, expected {n}')
            v.setflags(write=False)
            object.__setattr__(self, k, v)
        if self.trial_key is not None:
            key = np.asarray(self.trial_key, dtype=np.int64).reshape(-1)
            if len(key) != n:
                raise ValueError(f'TrialArrays: trial_key has length {len(key)}, expected {n}')
            key.setflags(write=False)
            object.__setattr__(self, 'trial_key', key)
        task = self.task if self.task is not None else TwoAFC()
        object.__setattr__(self, 'task', task)
        scores = {
            'correct': (_score(task, fields['choice'], fields['stimulus']) if self.correct is None
                        else _as_float_1d(self.correct, n, 'correct')),
            'prev_correct': (_score(task, fields['prev_choice'], fields['prev_stimulus']) if self.prev_correct is None
                             else _as_float_1d(self.prev_correct, n, 'prev_correct')),
        }
        scores['truth'] = (task.truth(fields['stimulus']) if self.truth is None
                           else _as_float_1d(self.truth, n, 'truth'))
        for k, v in scores.items():
            v = np.asarray(v, dtype=float)
            if len(v) != n:
                raise ValueError(f'TrialArrays: {k} has length {len(v)}, expected {n}')
            v.setflags(write=False)
            object.__setattr__(self, k, v)

    # ── constructors ────────────────────────────────────────────────────────

    @classmethod
    def from_pooled(cls, pooled: Mapping[str, np.ndarray]) -> TrialArrays:
        """
        Arrays from the dict that ``pool_arrays`` or ``get_arrays`` returns.

        Args:
            pooled: That dict.

        Returns:
            The arrays.
        """
        return cls(
            choice=pooled['choices'],
            stimulus=pooled['stimuli'],
            category=pooled['categories'],
            prev_choice=pooled.get('prev_choices'),
            prev_stimulus=pooled.get('prev_stimuli'),
            prev_category=pooled.get('prev_categories'),
            reaction_time=pooled.get('reaction_times'),
            trial_key=pooled.get('trial_keys'),
            correct=pooled.get('correct'),
            prev_correct=pooled.get('prev_correct'),
            task=pooled.get('task'),
            truth=pooled.get('truth'),
        )

    @classmethod
    def from_sessions(cls, sessions: Sequence) -> TrialArrays:
        """
        Arrays from pre-filtered sessions, pooled with ``pool_arrays``.

        Args:
            sessions: The sessions, already selected and filtered.

        Returns:
            The arrays, with the previous-trial view carried from each raw session and a ``trial_key`` per row.
        """
        from behav_utils.data.ops.filtering import pool_arrays
        return cls.from_pooled(pool_arrays(list(sessions)))

    @classmethod
    def from_sequence(
        cls,
        choice,
        stimulus,
        category=None,
        reaction_time=None,
        *,
        task: Task | None = None,
    ) -> TrialArrays:
        """
        From one contiguous block with no carried lag-1 view.

        The task (``TwoAFC()`` unless given) decides the categories and scores the
        trials, exactly as on loading: a go/no-go blank is read as a withhold.
        ``category``, if given, is only checked against the task's.

        The predecessor of trial t is trial t-1 by adjacency, with no
        predecessor at t = 0. A no-response predecessor carries its NaN choice
        through (so ``has_prev`` is False there) while its stimulus and
        category are still recorded — the same rule the loader applies when it
        freezes the lag-1 view on a raw session, so this matches
        ``from_sessions`` exactly on one abort-free block.

        Args:
            choice: Responses, NaN where there is none.
            stimulus: Stimuli on the library's axis.
            category: Categories to check against the task's; None to take the task's.
            reaction_time: Reaction times in seconds; None for all NaN.
            task: The task that decides categories and scores the trials; None for ``TwoAFC()``.

        Returns:
            The arrays, with no ``trial_key`` and the previous trial taken from array order.
        """
        task = task if task is not None else TwoAFC()
        s = _as_float_1d(stimulus, name='stimulus')
        n = len(s)
        c = task.complete_responses(_as_float_1d(choice, n, 'choice'), np.zeros(n, dtype=bool))
        cat = task.category(s)
        if category is not None:
            given = _as_float_1d(category, n, 'category')
            both = ~np.isnan(given) & ~np.isnan(cat)
            if np.any(given[both] != cat[both]):
                raise ValueError("TrialArrays.from_sequence: categories disagree with the task's for these "
                                 "stimuli; the task decides them (pass category=None)")
        prev_c = np.full(n, np.nan)
        prev_s = np.full(n, np.nan)
        prev_cat = np.full(n, np.nan)
        if n > 1:
            prev_c[1:] = c[:-1]
            prev_s[1:] = s[:-1]
            prev_cat[1:] = cat[:-1]
        return cls(c, s, cat, prev_c, prev_s, prev_cat, reaction_time, task=task)

    def with_trial_key(self, trial_key) -> TrialArrays:
        """
        A copy with ``trial_key`` set.

        Args:
            trial_key: One int64 key per row.

        Returns:
            The copy.
        """
        return replace(self, trial_key=trial_key)

    # ── views ───────────────────────────────────────────────────────────────

    @property
    def n_trials(self) -> int:
        return int(len(self.choice))

    @property
    def responded(self) -> np.ndarray:
        """Boolean mask of trials with a recorded choice."""
        return ~np.isnan(self.choice)

    @property
    def n_responded(self) -> int:
        return int(self.responded.sum())

    @property
    def has_prev(self) -> np.ndarray:
        """Boolean mask of trials with a usable (responded, same-block) predecessor."""
        return ~np.isnan(self.prev_choice)

    def valid(self) -> TrialArrays:
        """
        Responded trials only.

        The ``prev_*`` view is kept as it was, not re-shifted.

        Returns:
            The subset.
        """
        return self.take(self.responded)

    def take(self, mask_or_index) -> TrialArrays:
        """
        A subset of rows; ``prev_*`` and ``trial_key`` travel with them.

        Args:
            mask_or_index: A boolean mask, or integer positions.

        Returns:
            The subset.
        """
        idx = np.asarray(mask_or_index)
        return TrialArrays(
            self.choice[idx], self.stimulus[idx], self.category[idx],
            self.prev_choice[idx], self.prev_stimulus[idx], self.prev_category[idx],
            self.reaction_time[idx],
            None if self.trial_key is None else self.trial_key[idx],
            correct=self.correct[idx], prev_correct=self.prev_correct[idx], task=self.task,
            truth=self.truth[idx],
        )

    @staticmethod
    def concat(items: Sequence[TrialArrays]) -> TrialArrays:
        """
        Rows of several arrays, one after another.

        ``trial_key`` is kept only if every item carries one.

        Args:
            items: The arrays to join.

        Returns:
            The joined arrays.

        Raises:
            ValueError: There is nothing to join, or the items were scored by different tasks.
        """
        items = list(items)
        if not items:
            raise ValueError('TrialArrays.concat: nothing to concatenate')
        tasks = {a.task for a in items}
        if len(tasks) > 1:
            raise ValueError(f"TrialArrays.concat: the items were scored by different tasks "
                             f"({', '.join(sorted(map(repr, tasks)))}); keep them apart")
        cat = lambda f: np.concatenate([getattr(a, f) for a in items])  # noqa: E731
        keys = (np.concatenate([a.trial_key for a in items])
                if all(a.trial_key is not None for a in items) else None)
        return TrialArrays(cat('choice'), cat('stimulus'), cat('category'), cat('prev_choice'),
                           cat('prev_stimulus'), cat('prev_category'), cat('reaction_time'), keys,
                           correct=cat('correct'), prev_correct=cat('prev_correct'), task=items[0].task,
                           truth=cat('truth'))

    def lag1_pairs(self) -> TrialArrays:
        """
        Trials with a responded choice and a usable predecessor.

        The subset every lag-1 history statistic is computed on.

        Returns:
            The subset.
        """
        return self.take(self.responded & self.has_prev)

    def __len__(self) -> int:
        return self.n_trials

    def __repr__(self) -> str:
        return (f'TrialArrays(n_trials={self.n_trials}, n_responded={self.n_responded}, '
                f'n_lag1_pairs={int((self.responded & self.has_prev).sum())})')
