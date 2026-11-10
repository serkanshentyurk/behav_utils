# Extending behav_utils

Adding a statistic, a readout with its plot, or a reader. Each uses public names only; anything starting with
`_` can change without notice. Tasks and models join this guide when they land (steps 4 and 6 of the 1.0
work).

## A statistic

A statistic is a function of one block of trials (`TrialArrays`) that returns a float. Once registered it works
everywhere: `compute_stats`, phase statistics, contrasts, bootstraps and rolling windows.

```python
import numpy as np
from behav_utils.stats import stat

@stat('repeat_rate', exchangeable=False)
def repeat_rate(a, *, rng) -> float:
    """Fraction of trials whose choice repeats the previous trial's, where both have a response."""
    ok = ~np.isnan(a.choice) & ~np.isnan(a.prev_choice)
    return float(np.mean(a.choice[ok] == a.prev_choice[ok])) if ok.any() else np.nan
```

- Return NaN, never raise, when the data cannot give an answer.
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
