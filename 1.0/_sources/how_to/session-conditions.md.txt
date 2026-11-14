# Compare conditions that come in different sessions

Stages of training, stimulus distributions and drug days all change between sessions rather than within them. The
sessions are then the independent observations, so the uncertainty has to come from resampling sessions.

```python
from behav_utils import PsychometricModel, filter_trials, generate_synthetic_animal, select_sessions
from behav_utils.tasks import TruncatedNormal, Uniform

distributions = [Uniform()] * 4 + [TruncatedNormal(mean=0.3, sd=0.3)] * 4
labels = ['uniform'] * 4 + ['skewed'] * 4
animal = generate_synthetic_animal(PsychometricModel(sigma=0.25), n_sessions=8, trials_per_session=250,
                                   distribution=distributions, label=labels, rng=1)
sessions = select_sessions(animal, min_trials=0)
uniform = filter_trials([s for s in sessions if s.distribution == 'uniform'])
skewed = filter_trials([s for s in sessions if s.distribution == 'skewed'])
```

```python
from behav_utils import compute_delta_stat

d = compute_delta_stat({'uniform': uniform, 'skewed': skewed}, ['mu', 'accuracy'], reference='uniform',
                       curve=False, n_bootstrap=200, units=('sessions',))
d.contrast('skewed').table('sessions').round(3)
```

A permutation p is refused here, because shuffling labels between trials assumes the label was assigned per trial;
with whole sessions in each condition it was not. Trials can still be named as a unit, but their interval would
ignore that trials within a session are not independent, so it would be too narrow. With few sessions per
condition the session interval is wide; that is the honest state of the evidence.

[Find distribution blocks and switches](blocks-and-switches.md) labels the sessions by block when the condition
alternates.
