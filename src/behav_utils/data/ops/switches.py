"""Distribution blocks and switches in an ordered session list.

    blocks   = find_blocks(sessions)                 # runs of consecutive sessions with one distribution
    switches = find_switches(sessions, min_block_trials=1000)   # boundaries between qualifying blocks
    before, after = block_before(sessions, sw), block_after(sessions, sw)

Pass an already-selected, chronologically ordered list; one distribution per
session is assumed. A short excursion (a block below ``min_block_trials``)
is folded into its neighbours so it neither counts as a switch nor breaks
the surrounding blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

__all__ = ['Block', 'find_blocks', 'find_switches', 'block_before', 'block_after']


@dataclass(frozen=True)
class Block:
    """
    A run of consecutive sessions on one distribution, as ``find_blocks`` returns it.

    Attributes:
        distribution: The sessions' distribution label.
        start: Position of the first session in the list given to ``find_blocks``.
        stop: Position after the last session (exclusive).
        trial_start: Trials in the sessions before the block.
        n_trials: Trials in the block.
    """
    distribution: str
    start: int            # position in the input list (inclusive)
    stop: int             # position in the input list (exclusive)
    trial_start: int      # cumulative trials at the first session
    n_trials: int

    @property
    def n_sessions(self) -> int:
        return self.stop - self.start


def _n(s) -> int:
    return int(len(s.trials.choice))


def find_blocks(sessions: Sequence, *, min_block_trials: int = 0) -> List[Block]:
    """Maximal runs of consecutive sessions sharing ``session.distribution``.

    With ``min_block_trials`` > 0, a run shorter than that is merged into the
    preceding block of the same distribution if the run it interrupts resumes
    afterwards (an excursion), and otherwise dropped from the block list; the
    sessions themselves are untouched.

    Args:
        sessions: Sessions in order, typically one animal's.
        min_block_trials: The fewest trials a run needs to count as a block of its own; 0 keeps every run.

    Returns:
        The blocks in order, each with its ``distribution``, its ``start`` and ``stop`` positions in
        ``sessions`` (``stop`` exclusive), ``trial_start`` (the trials before it) and ``n_trials``.
    """
    sessions = list(sessions)
    raw: List[Block] = []
    cum = 0
    k = 0
    while k < len(sessions):
        d = getattr(sessions[k], 'distribution', None)
        j, n = k, 0
        while j < len(sessions) and getattr(sessions[j], 'distribution', None) == d:
            n += _n(sessions[j])
            j += 1
        raw.append(Block(d, k, j, cum, n))
        cum += n
        k = j
    if min_block_trials <= 0:
        return raw
    # absorb short excursions: A(long) b(short) A(...) -> one A block spanning all three
    out: List[Block] = []
    i = 0
    while i < len(raw):
        b = raw[i]
        if b.n_trials < min_block_trials:
            if out and i + 1 < len(raw) and raw[i + 1].distribution == out[-1].distribution:
                prev = out.pop()
                nxt = raw[i + 1]
                merged = Block(prev.distribution, prev.start, nxt.stop, prev.trial_start,
                               prev.n_trials + b.n_trials + nxt.n_trials)
                out.append(merged)
                i += 2
                continue
            i += 1        # short block with no same-distribution neighbour to rejoin: dropped
            continue
        out.append(b)
        i += 1
    # merged blocks may now be adjacent to a same-distribution block: coalesce
    coalesced: List[Block] = []
    for b in out:
        if coalesced and coalesced[-1].distribution == b.distribution and coalesced[-1].stop >= b.start:
            p = coalesced.pop()
            coalesced.append(Block(p.distribution, p.start, b.stop, p.trial_start, p.n_trials + b.n_trials))
        else:
            coalesced.append(b)
    return coalesced


def find_switches(sessions: Sequence, *, min_block_trials: int = 0) -> List[Dict]:
    """Boundaries between consecutive qualifying blocks.

    Args:
        sessions: Sessions in order, typically one animal's.
        min_block_trials: As in ``find_blocks``, the fewest trials a run needs to count as a block.

    Returns:
        One dict per switch, with ``switch_idx``, ``session_idx`` (the first session after the switch),
        ``order`` (its position in ``sessions``), ``trial_index`` (the trials before the boundary),
        ``from_distribution``, ``to_distribution``, ``n_trials_before``, ``n_trials_after`` and
        ``n_sessions_after``.
    """
    sessions = list(sessions)
    blocks = find_blocks(sessions, min_block_trials=min_block_trials)
    out: List[Dict] = []
    for a, b in zip(blocks[:-1], blocks[1:]):
        if a.distribution == b.distribution:
            continue
        s = sessions[b.start]
        out.append({'switch_idx': len(out), 'session_idx': getattr(s, 'session_idx', b.start), 'order': b.start,
                    'trial_index': b.trial_start, 'from_distribution': a.distribution,
                    'to_distribution': b.distribution, 'n_trials_before': a.n_trials,
                    'n_trials_after': b.n_trials, 'n_sessions_after': b.n_sessions,
                    '_before': (a.start, a.stop), '_after': (b.start, b.stop)})
    return out


def block_before(sessions: Sequence, switch: Dict) -> list:
    """
    The ordered sessions of the block that ends at ``switch``.

    Args:
        sessions: The sessions given to ``find_switches``.
        switch: One of its switches.

    Returns:
        The sessions.
    """
    a, b = switch['_before']
    return list(sessions)[a:b]


def block_after(sessions: Sequence, switch: Dict) -> list:
    """
    The ordered sessions of the block that starts at ``switch``.

    Args:
        sessions: The sessions given to ``find_switches``.
        switch: One of its switches.

    Returns:
        The sessions.
    """
    a, b = switch['_after']
    return list(sessions)[a:b]
