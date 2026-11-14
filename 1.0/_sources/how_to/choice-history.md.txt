# Measure choice history

Many studies ask how the previous trial shapes the current choice: a pull towards the last choice or the last
stimulus, staying after a reward, switching after an error. The library has several measures of this, from simple
conditional rates to a logistic regression on past trials and the update matrix.

The animal here is a small learner whose criterion follows the outcomes of recent choices (the
[models tutorial](../tutorials/models.ipynb) explains it), so it has a history effect to find:

```python
from scipy.stats import norm

from behav_utils import TrialArrays, filter_trials, generate_synthetic_animal, select_sessions
from behav_utils.models import TrialwiseModel


class Learner(TrialwiseModel):
    def reset(self):
        self.criterion = 0.0

    def p_choice(self, trial):
        return float(norm.cdf((trial.stimulus - self.criterion) / 0.2))

    def update(self, trial):
        towards_repeat = -0.1 if trial.choice == 1 else 0.1
        self.criterion = 0.9 * (self.criterion + (towards_repeat if trial.correct else -towards_repeat))


animal = generate_synthetic_animal(Learner(), n_sessions=6, trials_per_session=400, rng=5)
arrays = TrialArrays.from_sessions(filter_trials(select_sessions(animal, min_trials=0)))
```

The conditional measures and the regression weights:

```python
from behav_utils import compute_stats
from behav_utils.stats import LOGISTIC_HISTORY

compute_stats(arrays, ['recency', 'stimulus_recency', 'win_stay', 'lose_shift', *LOGISTIC_HISTORY]).round(3)
```

The update matrix shows how P(B) shifts with the previous stimulus, for each current stimulus, after rewarded
trials:

```python
from behav_utils import compute_update_matrix, plot_update_matrix

plot_update_matrix(compute_update_matrix(arrays));
```

The previous trial is carried with each trial from the raw session, so these measures stay correct after filtering
or pooling (see [Identities and time](../concepts/identity-and-time.md)). Some history statistics depend on the
order of trials and cannot be resampled by trial; the [statistics reference](../stats_reference.md) marks them,
and a contrast of history between conditions uses sessions as the unit.
