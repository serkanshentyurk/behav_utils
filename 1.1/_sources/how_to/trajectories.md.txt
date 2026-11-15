# Follow a statistic over sessions and within them

Learning curves, and changes within a session, need a statistic computed per session or per window of trials rather
than once for all of them.

```python
from behav_utils import PsychometricModel, filter_trials, generate_synthetic_animal, select_sessions

learning = [PsychometricModel(sigma=sigma) for sigma in (0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.25, 0.25)]
animal = generate_synthetic_animal(learning, n_sessions=8, trials_per_session=250, rng=4)
sessions = filter_trials(select_sessions(animal, min_trials=0))
```

Per session, `compute_phase_stats` with `per_session=True` gives one row per session and statistic, beside the
pooled values, and `plot_trajectory` draws one statistic in session order:

```python
import matplotlib.pyplot as plt

from behav_utils import compute_phase_stats, plot_trajectory

r = compute_phase_stats(sessions, ['accuracy', 'sigma'], per_session=True)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
plot_trajectory(r, 'accuracy', ax=axes[0])
plot_trajectory(r, 'sigma', ax=axes[1])
plt.tight_layout()
```

Within sessions, `compute_rolling_stats` slides a window of trials along each session; windows never cross from one
session into the next:

```python
from behav_utils import compute_rolling_stats, plot_rolling_stats

rolling = compute_rolling_stats(sessions, ['accuracy'], window=60, step=15)
plot_rolling_stats(rolling, 'accuracy');
```

A per-session value is noisier than the pooled one, and a fitted parameter on a few hundred trials more so; the
pooled value of a phase is the fit to all its trials, never the mean of these per-session values (see
[Resampling units](../concepts/resampling.md)).
