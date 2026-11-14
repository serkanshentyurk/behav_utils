# Compare groups of animals

Claims about a manipulation across animals rest on one value per animal: for each animal, compute the statistic,
then compare the animals, not the pooled trials.

```python
from behav_utils import PsychometricModel, generate_synthetic_experiment

groups = {'C1': 'control', 'C2': 'control', 'C3': 'control', 'C4': 'control',
          'T1': 'treated', 'T2': 'treated', 'T3': 'treated', 'T4': 'treated'}
experiment = generate_synthetic_experiment(
    {animal_id: {'model': PsychometricModel(sigma=0.25 if group == 'control' else 0.4), 'n_sessions': 4,
                 'trials_per_session': 250, 'laser_fraction': 0.3, 'metadata': {'group': group}}
     for animal_id, group in groups.items()}, rng=6)
```

One row per animal and statistic, stamped with the animal and its group:

```python
from behav_utils import compute_phase_stats, filter_trials, select_sessions
from behav_utils.analysis import collect_rows

rows = []
for animal in experiment.animals.values():
    phase = filter_trials(select_sessions(animal, min_trials=0))
    stats = compute_phase_stats(phase, ['sigma', 'accuracy']).to_rows().to_dict('records')
    rows += collect_rows(stats, animal=animal.animal_id, group=animal.metadata['group'])
```

Then a rank test per statistic between the groups:

```python
from behav_utils import compare_groups

comparison = compare_groups(rows, groups=('control', 'treated'))
comparison.table().round(3)
```

`min_p` is the smallest p the test can give with these group sizes: with four animals per group it is about 0.03,
and with three it is 0.1. If it is above your threshold, a non-significant result says nothing about the effect.
For two conditions in the same animals, such as laser and no laser, stamp each row with its condition and pass
`paired=True`: the values are then matched by animal and a signed-rank test is used.
