# Select sessions by stage and performance

Analyses usually start from a defined set of sessions: those of one stage, after a training criterion was met, or
the last few of a block. `select_sessions` takes these criteria, and a preset names a set used more than once.

```python
from behav_utils import (PsychometricModel, SessionFilter, generate_synthetic_animal, register_preset,
                         select_sessions)

learning = [PsychometricModel(sigma=sigma) for sigma in (0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.25, 0.25)]
animal = generate_synthetic_animal(learning, n_sessions=8, trials_per_session=250, rng=2)
```

Criteria combine; each is optional:

```python
good = select_sessions(animal, stage='task', min_accuracy=0.8)
last_good = select_sessions(animal, min_accuracy=0.8, last_n_qualifying=3)
[s.session_idx for s in good], [s.session_idx for s in last_good]
```

A preset is a named set of criteria. Register it in code, or list it under `session_presets` in the config, where
`load_experiment` registers it; a call can still override a field:

```python
register_preset('expert', SessionFilter(stage='task', min_accuracy=0.8, last_n_qualifying=3))
expert = select_sessions(animal, 'expert')
last_two = select_sessions(animal, 'expert', last_n_qualifying=2)
len(expert), len(last_two)
```

To pick by hand, `animal.session_table` has one row per session, with its date, type and accuracy:

```python
animal.session_table[['session_idx', 'date', 'session_type', 'accuracy']]
```

`SessionFilter` lists every criterion and the order in which they apply; which sessions count is settled before any
position is taken, so `last_n_qualifying=3` means the last three sessions that pass the other criteria.
