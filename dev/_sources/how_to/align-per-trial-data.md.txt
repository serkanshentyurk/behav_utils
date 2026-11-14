# Align another per-trial dataset

Neural recordings, video and pupil data are often kept per trial outside the behaviour. To relate them to a
statistic, the other dataset has to be cut to exactly the trials the statistic used, in the same order. Every row of
`TrialArrays` carries a `trial_key` for this (see [Identities and time](../concepts/identity-and-time.md)).

```python
import numpy as np
import pandas as pd

from behav_utils import PsychometricModel, TrialArrays, filter_trials, generate_synthetic_animal, select_sessions

animal = generate_synthetic_animal(PsychometricModel(), n_sessions=3, trials_per_session=200, laser_fraction=0.3,
                                   rng=7)
sessions = select_sessions(animal, min_trials=0)
```

Suppose another recording has one row per trial, identified by session and trial number:

```python
from behav_utils.data import trial_key_of

rng = np.random.default_rng(0)
recording = pd.concat([
    pd.DataFrame({'session_id': s.session_id, 'trial_number': s.trials.trial_number,
                  'pupil': rng.normal(3, 0.5, s.trials.n_trials)})
    for s in sessions])
by_session = {s.session_id: s for s in sessions}
recording['trial_key'] = np.concatenate([
    trial_key_of(by_session[session_id], group['trial_number'].to_numpy())
    for session_id, group in recording.groupby('session_id', sort=False)])
```

The behaviour, filtered as for the statistic, carries the same keys:

```python
arrays = TrialArrays.from_sessions(filter_trials(sessions, trial_type='non_opto'))
aligned = recording.set_index('trial_key').loc[arrays.trial_key]
len(arrays.trial_key), len(aligned), bool((aligned['trial_number'].to_numpy() > 0).all())
```

`aligned` now has one row per trial in `arrays`, in the same order, so a per-trial value can be set beside the
choices or masked the same way. The keys travel through `take`, `valid`, `lag1_pairs` and every resample, so the
alignment holds for whatever subset a statistic ends up using.
