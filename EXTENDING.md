# Extending behav_utils

Adding a statistic, a readout with its plot, a task, a model, or a reader. Each uses public names only; anything starting with
`_` can change without notice.

## A statistic

A statistic is a function of one block of trials (`TrialArrays`) that returns a float. Once registered it works
everywhere: `compute_stats`, phase statistics, contrasts, bootstraps and rolling windows.

```python
import numpy as np
from behav_utils.stats import stat

@stat('repeat_rate', exchangeable=False, requires=('binary',))
def repeat_rate(a, *, rng) -> float:
    """Fraction of trials whose choice repeats the previous trial's, where both have a response."""
    ok = ~np.isnan(a.choice) & ~np.isnan(a.prev_choice)
    return float(np.mean(a.choice[ok] == a.prev_choice[ok])) if ok.any() else np.nan
```

- Return NaN, never raise, when the data cannot give an answer.
- `requires=` names the task capabilities the statistic needs (`behav_utils.tasks.CAPABILITIES`, or names of your
  own); leave it out and the statistic runs on every task. On a task without them, `compute_stats` refuses it
  with `RequirementError`; `instead={'withhold': 'go_bias'}` names the counterpart to suggest there.
- `exchangeable=False` when the value depends on trial order (anything using `prev_*`): trial-level
  resampling then refuses it, because shuffling trials would break the lag-1 pairs it reads.
- Several outputs from one fit: `@fit('name', outputs=('a', 'b'))` on a function returning a tuple; each output
  becomes its own statistic and the fit runs once per call.
- Regenerate the reference: `python docs/gen_stats_reference.py`.

## A readout and its plot

A readout is anything array-valued (a curve, a matrix). `compute_x(arrays, ...)` returns a frozen dataclass with
`to_rows()`; `plot_x(result, ax=None)` draws it on one panel and computes nothing.

```python
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RTByStimulus:
    """Median reaction time per stimulus bin."""
    centres: np.ndarray
    median_rt: np.ndarray
    n: np.ndarray

    def to_rows(self) -> pd.DataFrame:
        return pd.DataFrame({'stimulus': self.centres, 'median_rt': self.median_rt, 'n': self.n})


def compute_rt_by_stimulus(a, n_bins: int = 8) -> RTByStimulus:
    edges = np.linspace(-1, 1, n_bins + 1)
    idx = np.clip(np.digitize(a.stimulus, edges) - 1, 0, n_bins - 1)
    rt = [a.reaction_time[(idx == i) & ~np.isnan(a.reaction_time)] for i in range(n_bins)]
    return RTByStimulus(centres=(edges[:-1] + edges[1:]) / 2,
                        median_rt=np.array([np.median(r) if len(r) else np.nan for r in rt]),
                        n=np.array([len(r) for r in rt]))


def plot_rt_by_stimulus(result: RTByStimulus, ax=None):
    """Draw only: one panel, no computation, no figure-level styling."""
    ax = ax if ax is not None else plt.gca()
    ax.plot(result.centres, result.median_rt, marker='o')
    ax.set_xlabel('stimulus')
    ax.set_ylabel('median RT (s)')
    return ax
```

In the library, the compute function goes in `behav_utils/readouts/`, the plot in `behav_utils/plotting/`, and both
names in their sub-package's `__all__`.

## A task

A task decides each trial's category, its correct response and how a response is scored; every session carries
one, and nothing else defines correctness. Subclass `Task` as a frozen dataclass whose fields are the settings a
config may give, name it, and register it:

```python
from dataclasses import dataclass

import numpy as np
from behav_utils.tasks import ResponseSpace, Task, register_task


@register_task
@dataclass(frozen=True)
class EasyOnly(Task):
    """2-AFC that scores only easy trials: |stimulus| >= threshold; the rest are UNSCORED."""
    name = 'easy_only'
    capabilities = frozenset({'binary', 'two_choices', 'stimulus_axis', 'scored'})
    threshold: float = 0.3

    @property
    def response_space(self):
        return ResponseSpace('binary', ('A', 'B'))

    def category(self, stimulus, context=None):
        s = np.asarray(stimulus, dtype=float)
        return np.where(np.isnan(s), np.nan, (s > 0).astype(float))

    def truth(self, stimulus, context=None):
        s = np.asarray(stimulus, dtype=float)
        return np.where(np.abs(s) >= self.threshold, self.category(s), np.nan)
```

A config then says `task: {type: easy_only, threshold: 0.4}`; unknown settings are refused. `capabilities` says
what the task provides; statistics and readouts check it, so a task that declares none gets only the statistics
that need nothing. `score` (abort, then
no response, then no correct response, then correct or error) and `complete_responses` (how blanks are read) have
defaults that suit most tasks; `GoNoGo` overrides `complete_responses` to read a blank as a withhold.

## A statistic for your own task

A task can declare a capability of its own, and a statistic can require it. Neither needs a change to the
library:

```python
from dataclasses import dataclass

import numpy as np
from behav_utils import TwoAFC
from behav_utils.stats import stat
from behav_utils.tasks import register_task


@register_task
@dataclass(frozen=True)
class CuedTwoAFC(TwoAFC):
    """2-AFC in which some trials carry a cue; the statistics below only make sense for it."""
    name = 'cued_two_afc'
    capabilities = TwoAFC.capabilities | {'cue'}


@stat('cued_go_rate', requires=('cue',))
def cued_go_rate(arrays, *, rng):
    """Response-1 rate over responded trials, for the cued task only."""
    return float(np.mean(arrays.valid().choice)) if arrays.n_responded else np.nan
```

`compute_stats(arrays, ['cued_go_rate'])` runs on sessions of `CuedTwoAFC` and is refused, with the reason, on any
other task; `list_stats(task=CuedTwoAFC())` includes it. A required name that no registered task declares is
flagged in the refusal, which catches a typo.

## A model

A model is anything with `simulate(design, rng)` returning `Responses`. `TrialwiseModel` runs the trial loop for
models that learn, so a learner only says how it chooses and how it updates:

```python
from behav_utils import make_design, simulate
from behav_utils.models import TrialwiseModel


class WinStayLoseShift(TrialwiseModel):
    """Repeats a correct response and switches after an error; follows the stimulus on the first trial."""

    def __init__(self, p_follow=0.9):
        self.p_follow = p_follow

    def reset(self):
        self.last = None

    def p_choice(self, t, stimulus, laser):
        if self.last is None:
            return float(stimulus > 0)
        choice, correct = self.last
        repeat = choice if correct else 1 - choice
        return self.p_follow if repeat == 1 else 1 - self.p_follow

    def update(self, t, stimulus, choice, correct):
        self.last = (choice, correct)


session = simulate(WinStayLoseShift(), make_design(n_trials=400))
```

The session is scored by the design's task and goes through every statistic like loaded data. `correct` is the
task's score of the choice (None where the task gives none). Aborts, no-responses and laser trials come from the
design (`make_design(..., abort_rate=, no_response_rate=, laser_fraction=)`); `simulate()` blanks aborted and
no-response trials whatever the model returns. A model that is not a `TrialwiseModel` writes `simulate` itself
and may use the whole design: stimuli, `truth`, the laser flags.

`generate_synthetic_animal(WinStayLoseShift(), n_sessions=10, laser_fraction=0.3, rng=0)` makes a whole animal
from it, with every trial class and event times.

## A reader

A reader finds sessions and reads their raw trial tables; column mapping, stimulus normalisation and scoring
happen afterwards in `build_session`, the same way for every reader. Any function that takes the config and
yields `RawSession` objects is a reader.

```python
from datetime import date

import pandas as pd
from behav_utils import load_experiment
from behav_utils.data import RawSession


def one_file_per_animal(config):
    """Each animal's sessions in one CSV with a 'Session' and a 'Date' column."""
    for path in sorted(config.resolve_path('exports').glob('*.csv')):
        table = pd.read_csv(path)
        for session_id, rows in table.groupby('Session', sort=False):
            yield RawSession(animal_id=path.stem, session_id=str(session_id),
                             date=date.fromisoformat(rows['Date'].iloc[0]),
                             trials=rows.reset_index(drop=True), source=str(path))


experiment = load_experiment('config.yaml', reader=one_file_per_animal)
```

- `trials` may be a DataFrame or a function returning one. With a function, the loader reads the table only if
  the session is not in `sessions_to_ignore`, which matters when sessions are ignored because their files are
  broken.
- `start` (a `datetime.time`) orders several sessions on one day; `metadata` adds session metadata (columns
  mapped in the config win); `source` names where the session came from, for messages.
- A reader that accepts a `strict` keyword is given `load_experiment`'s, so it can raise or skip on unreadable
  input the same way the built-in readers do.
- `RawSession.date` must be set: a session without one raises under `strict` and is skipped otherwise.
