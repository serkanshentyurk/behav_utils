# Find distribution blocks and switches

Designs that change the stimulus distribution partway through training need the runs of sessions under each
distribution, and the points where it changed. The distribution's label comes from the data (a mapped column, or
the config), and `find_blocks` and `find_switches` read it from consecutive sessions.

```python
from behav_utils import PsychometricModel, generate_synthetic_animal, select_sessions
from behav_utils.tasks import TruncatedNormal, Uniform

distributions = [Uniform()] * 3 + [TruncatedNormal(mean=0.3, sd=0.3)] * 3 + [Uniform()] * 3
labels = ['uniform'] * 3 + ['skewed'] * 3 + ['uniform'] * 3
animal = generate_synthetic_animal(PsychometricModel(), n_sessions=9, trials_per_session=200,
                                   distribution=distributions, label=labels, rng=3)
sessions = select_sessions(animal, min_trials=0)
```

```python
import pandas as pd

from behav_utils import find_blocks, find_switches

find_blocks(sessions)
```

A block's `start` and `stop` are positions in the list of sessions (`stop` exclusive), so `sessions[b.start:b.stop]`
is that block. The switches are the boundaries between blocks:

```python
columns = ['switch_idx', 'session_idx', 'from_distribution', 'to_distribution', 'n_trials_before', 'n_trials_after']
pd.DataFrame(find_switches(sessions))[columns]
```

To compare the sessions just before and just after each switch:

```python
from behav_utils import compute_delta_stat, filter_trials

for switch in find_switches(sessions):
    i = switch['order']
    before, after = filter_trials(sessions[i - 2:i]), filter_trials(sessions[i:i + 2])
    d = compute_delta_stat({'before': before, 'after': after}, ['mu'], reference='before', curve=False,
                           n_bootstrap=200, units=('sessions',))
    print(switch['from_distribution'], '->', switch['to_distribution'], d.contrast('after').table('sessions').round(3).to_dict('records'))
```

`min_block_trials` drops short runs, such as a single session on another distribution, from the blocks; the
sessions themselves are untouched. Before and after a switch are different sessions, so sessions are the unit (see
[Compare conditions that come in different sessions](session-conditions.md)).
