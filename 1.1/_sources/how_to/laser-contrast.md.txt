# Compare laser and control trials

When the laser is switched on in some trials of a session, the question is how those trials differ from comparable
trials without it. This needs a reference that excludes the laser trials, and a unit of resampling that matches how
the laser was assigned.

```python
from behav_utils import PsychometricModel, generate_synthetic_animal, select_sessions

model = PsychometricModel(sigma=0.25, laser={'sigma': 0.4})
animal = generate_synthetic_animal(model, n_sessions=6, trials_per_session=300, laser_fraction=0.3, rng=0)
sessions = select_sessions(animal, min_trials=0)
```

`filter_trials` names the usual selections; `trial_type='non_opto'` is every trial without the laser, aborts dropped.
For a reference of interleaved control trials only, pass a mask. `opto_mask(trials, delta='control')` keeps the
non-laser trials but does not look at their neighbours, so the trial right after a laser run, which may still
carry an effect, is better excluded:

```python
from behav_utils import build_mask, filter_trials, opto_mask

laser = filter_trials(sessions, trial_type='opto')
control = filter_trials(sessions, label='control', mask_fn=lambda s: build_mask(s.trials, exclude_opto=False)
                        & opto_mask(s.trials, delta='control') & ~opto_mask(s.trials, delta=1))
```

The laser was assigned trial by trial within the same sessions, so trials are a valid unit, and so is a permutation
of the labels (see [Resampling units](../concepts/resampling.md)):

```python
from behav_utils import compute_delta_stat

d = compute_delta_stat({'control': control, 'laser': laser}, ['sigma', 'accuracy'], reference='control',
                       curve=False, n_bootstrap=200, n_permutations=200)
d.contrast('laser').table().round(3)
```

Two things to watch. The reference must not contain the laser trials, or the trials it is compared with: a phase
that shares trials with the reference dilutes the difference, and the library warns, or refuses a permutation.
And the trials just after a laser run (`trial_type='post_opto'`) are a phase of their own for asking whether the
effect outlasts the laser.
