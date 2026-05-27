# Data Structures Reference

## Overview

behav_utils organises behavioural data in a hierarchy that mirrors experimental structure:

```
ExperimentData                    All animals in a project
  └── AnimalData                  One animal, all its sessions
        └── SessionData           One behavioural session
              ├── SessionMetadata  Task parameters (constant within session)
              └── TrialData        Trial-by-trial arrays
```

Separately, `FittingData` provides a flat per-session array format for SBI inference.

---

## TrialData

The lowest level — per-trial arrays for a single session.

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `trial_number` | int array | Original trial numbers from CSV |
| `stimulus` | float array | Stimulus values |
| `choice` | float array | Category-space choice: 0=A, 1=B, NaN=no response |
| `choice_raw` | float array | Raw choice from CSV (before conversion) |
| `outcome` | str array | Trial outcome ('Correct', 'Incorrect', 'Abort') |
| `correct` | bool array | Whether choice matched category |
| `category` | int array | Derived from stimulus: 0=A, 1=B |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `reaction_time` | float array | Response latency (NaN for aborts) |
| `abort` | bool array | Whether animal broke fixation |
| `opto_on` | bool array | Whether optogenetics was active |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `.n_trials` | int | Total trial count |
| `.no_response` | bool array | True where `choice` is NaN |
| `.valid_mask` | bool array | `~abort & ~no_response` |

### `get_arrays()`

Extract analysis-ready arrays. **No kwargs** — always excludes aborts (invalid data), returns everything else. All scientific filtering (opto, no-response, custom) happens upstream via module-level functions.

```python
arrays = session.trials.get_arrays()
```

Returns dict:

| Key | Description |
|-----|-------------|
| `'stimuli'` | Stimulus values |
| `'categories'` | True categories 0/1 |
| `'choices'` | Choices 0/1/NaN |
| `'no_response'` | Boolean mask for NaN choices |
| `'reaction_times'` | RT values |
| `'trial_indices'` | Original indices for back-mapping |

---

## SessionData

One behavioural session — metadata + trial data.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | str | Unique identifier |
| `session_idx` | int | Ordinal position within animal (0-based) |
| `date` | datetime.date | Session date |
| `metadata` | SessionMetadata | Task parameters |
| `trials` | TrialData | Trial-by-trial arrays |
| `masking` | bool | Whether this is a masking (sham) session |
| `filter_info` | dict or None | Metadata about filtering applied |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `.n_trials` | int | Total trials (reflects filtering) |
| `.stage` | str | Training stage (from metadata) |
| `.distribution` | str | Stimulus distribution |
| `.is_filtered` | bool | Whether filtering has been applied |

### Filtering trials

Filtering uses module-level functions from `behav_utils.data.filtering`:

```python
from behav_utils.data.filtering import filter_session, opto_mask, filter_trials

# Standard filter: drop aborts and opto trials
clean = filter_session(session)

# Opto trials only
opto_only = filter_session(session, opto_mask(session.trials, 0))

# Control trials (non-opto, not adjacent to opto)
ctrl = filter_session(session, opto_mask(session.trials, 'control'))

# Filter list of sessions at once
clean_list = filter_trials(sessions)
```

`opto_mask(trials, delta)`:
- `delta=0` — opto trials themselves
- `delta=1` — first trial after each opto trial
- `delta=-1` — trial before each opto trial
- `delta='control'` — non-opto trials (not adjacent to opto)

---

## SessionMetadata

Task parameters constant within a session. Access by attribute or `.get()`:

```python
session.metadata.stage
session.metadata.get('sound_contingency')
session.metadata.fields  # raw dict
```

---

## AnimalData

One animal — all sessions in chronological order.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `animal_id` | str | Animal identifier |
| `sessions` | list[SessionData] | Chronologically ordered sessions |
| `metadata` | dict | Animal-level metadata (genotype, etc.) |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `.n_sessions` | int | Session count |
| `.session_ids` | list[str] | All session IDs |

### Working with an animal

```python
from behav_utils.data.selection import select_sessions
from behav_utils.data.filtering import filter_trials
from behav_utils.analysis.psychometry import compute_psychometric
from behav_utils.plotting.psychometric import plot_psychometric
from behav_utils.plotting.styles import PALETTE

# 1. Select sessions
sessions = select_sessions(animal, preset='expert_uniform')

# 2. Filter trials
clean = filter_trials(sessions)

# 3. Analyse
result = compute_psychometric(clean, mode='pooled', n_bootstrap=200)

# 4. Plot
fig, ax = plt.subplots()
plot_psychometric(result, ax=ax, color=PALETTE[0])
```

---

## ExperimentData

All animals in one project.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `animals` | dict[str, AnimalData] | animal_id → AnimalData |
| `config` | ProjectConfig or None | Loaded config |

### Methods

```python
from behav_utils.data.loading import load_experiment

experiment = load_experiment('config.yaml')
animal = experiment.get_animal('SS05')
all_animals = experiment.get_animals(min_sessions=10)
```

---

## FittingData

Flat per-session arrays for SBI inference. Built from pre-filtered sessions:

```python
from behav_utils.data.selection import select_sessions
from behav_utils.data.filtering import filter_trials
from behav_utils.data.fitting_data import fitting_data_from_sessions

sessions = select_sessions(animal, preset='expert_uniform')
clean = filter_trials(sessions)
fd = fitting_data_from_sessions(clean, animal.animal_id)

# fd.stimuli       — list of arrays, one per session
# fd.choices       — list of arrays
# fd.n_sessions    — int
# fd.animal_id     — str
```

---

## Pipeline pattern

Every notebook follows the same four steps:

```python
from behav_utils.data.loading import load_experiment
from behav_utils.data.selection import select_sessions
from behav_utils.data.filtering import filter_trials
from behav_utils.analysis.psychometry import compute_psychometric
from behav_utils.analysis.update_matrix import compute_um
from behav_utils.analysis.trajectory import compute_trajectory
from behav_utils.plotting.psychometric import plot_psychometric
from behav_utils.plotting.update_matrix import plot_um
from behav_utils.plotting.trajectory import plot_trajectory
from behav_utils.plotting.styles import PALETTE, apply_style

apply_style()

# 1. LOAD
experiment = load_experiment('config.yaml')
animal = experiment.get_animal('SS05')

# 2. FILTER (session-level + trial-level)
sessions = select_sessions(animal, preset='expert_uniform')
clean = filter_trials(sessions)

# 3. COMPUTE
psych = compute_psychometric(clean, mode='pooled', n_bootstrap=200)
um = compute_um(clean)
traj = compute_trajectory(clean, ['accuracy', 'mu'])

# 4. PLOT
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_psychometric(psych, ax=axes[0], color=PALETTE[0])
plot_um(um, ax=axes[1])
plot_trajectory(traj, 'accuracy', ax=axes[2])
```

### Comparing two conditions

```python
from behav_utils.data.filtering import filter_session, opto_mask
from behav_utils.analysis.comparison import compute_comparison
from behav_utils.plotting.comparison import plot_comparison

ctrl = [filter_session(s, opto_mask(s.trials, 'control')) for s in sessions]
opto = [filter_session(s, opto_mask(s.trials, 0))         for s in sessions]

# Full statistical comparison
comp = compute_comparison(ctrl, opto, label_a='Control', label_b='Opto',
                            n_bootstrap=1000, n_permutations=1000)
plot_comparison(comp)
```

Result dict keys:

| Key | Description |
|-----|-------------|
| `params_a`, `params_b` | Fitted psychometric params (mu, sigma, lapse_low, lapse_high) for each condition |
| `diffs` | params_a − params_b for each key |
| `perm_p` | Permutation p-values per param key |
| `boot_ci` | Bootstrap CIs per param key |
| `boot_band_a`, `boot_band_b` | Bootstrap psychometric bands: dict with `x`, `lo`, `hi`, `median` |
| `um_rmse` | Update matrix RMSE between conditions |

Note: bootstrap band keys are `lo` and `hi`, not `lower`/`upper`.

### Group-level claims (across animals)

For across-animal comparisons (e.g. HET vs WT, or paired opto effects), use the
group-level functions — never pool trials across animals.

```python
from behav_utils.analysis.comparison import (
    compute_per_animal_stats, compute_group_comparison
)

# Unpaired (cohort comparison: HET vs WT)
df_het = compute_per_animal_stats(het_animals)
df_wt  = compute_per_animal_stats(wt_animals)
result = compute_group_comparison(df_het, df_wt,
                                    label_a='HET', label_b='WT', paired=False)
# result['p_values']['mu'] is the Mann-Whitney p

# Paired (condition comparison: opto-on vs opto-off within same animals)
df_on  = compute_per_animal_stats(animals, sessions_per_animal=sessions_on)
df_off = compute_per_animal_stats(animals, sessions_per_animal=sessions_off)
result = compute_group_comparison(df_on, df_off,
                                    label_a='opto_on', label_b='opto_off', paired=True)
# result['p_values']['mu'] is the Wilcoxon p
```

---

## Psychometric naming convention

Psychometric fit parameters use math names everywhere in code:

| Code key | Display label | Meaning |
|----------|---------------|---------|
| `mu` | PSE | Point of subjective equality (boundary) |
| `sigma` | slope | Noise width (smaller = steeper psychometric) |
| `lapse_low` | λ_low | Lower asymptote lapse |
| `lapse_high` | λ_high | Upper asymptote lapse |
| `accuracy` | Accuracy | Overall correctness |

Plot functions automatically translate `mu`/`sigma` to `PSE`/`slope` for y-axis labels. You write `'mu'` in code; the plot shows "PSE".

```python
# Trajectory of PSE across sessions
traj = compute_trajectory(clean, ['accuracy', 'mu', 'sigma'])
plot_trajectory(traj, 'mu')   # y-axis label: "PSE"
plot_trajectory(traj, 'sigma') # y-axis label: "slope"
```

---

## Synthetic Data

Generate test data without real experiments:

```python
from behav_utils.data.synthetic import (
    generate_synthetic_animal, noisy_psychometric_simulator,
)

# Built-in simulator
animal, info = generate_synthetic_animal(
    animal_id='SYN01',
    n_sessions=20,
    trials_per_session=200,
    simulator=noisy_psychometric_simulator,
    simulator_kwargs={'sigma': 0.3, 'lapse': 0.05},
)

# Learning trajectory: parameters change across sessions
animal, info = generate_synthetic_animal(
    animal_id='LEARN01',
    n_sessions=20,
    simulator=noisy_psychometric_simulator,
    per_session_simulator_kwargs=[
        {'sigma': 0.8 - i * 0.03, 'lapse': max(0.01, 0.15 - i * 0.006)}
        for i in range(20)
    ],
)

# Custom simulator: any (stimuli, categories, rng, **kwargs) -> choices callable
def my_simulator(stimuli, categories, rng, **kwargs):
    ...
    return choices

animal, info = generate_synthetic_animal(
    simulator=my_simulator,
    simulator_kwargs={'my_param': 0.5},
)
```