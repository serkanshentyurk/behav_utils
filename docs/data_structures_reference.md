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

`TrialArrays` (`behav_utils.data.arrays`) is the flat, pooled view every statistic and readout computes from; `TrialArrays.from_sessions(clean)` builds it (`pool_arrays` is the raw dict underneath).

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
| `session_type` | str | `'regular'` / `'opto'` (derived) or a type stamped from `config.session_types` |
| `filter_info` | dict or None | Metadata about filtering applied |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `.n_trials` | int | Total trials (reflects filtering) |
| `.stage` | str | Training stage (from metadata) |
| `.distribution` | str | Stimulus distribution |
| `.is_filtered` | bool | Whether filtering has been applied |

### Filtering trials

Filtering uses module-level functions from `behav_utils.data.ops.filtering`:

```python
from behav_utils.data.ops.filtering import filter_session, opto_mask, filter_trials

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
| `metadata` | dict | Animal-level metadata, merged from `animal_metadata.json` (group, sex, …) |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `.n_sessions` | int | Session count |
| `.session_ids` | list[str] | All session IDs |

### Working with an animal

```python
from behav_utils import select_sessions, filter_trials, TrialArrays
from behav_utils import compute_psychometric_curve, plot_psychometric_curve, PALETTE

# 1. Select sessions (presets come from your config's session_presets)
sessions = select_sessions(animal, preset='expert')

# 2. Filter trials (abort / no-response / opto exclusions)
clean = filter_trials(sessions)

# 3. Compute a readout on the pooled arrays
curve = compute_psychometric_curve(TrialArrays.from_sessions(clean), n_bootstrap=200)

# 4. Draw
fig, ax = plt.subplots()
plot_psychometric_curve(curve, ax=ax, color=PALETTE[0])
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
animal = experiment.get_animal('A05')
all_animals = experiment.get_animals(min_sessions=10)
```

---

## Pipeline pattern

Every analysis follows the same four steps:

```python
from behav_utils import (load_experiment, select_sessions, filter_trials, TrialArrays,
                         compute_stats, PSYCHOMETRIC,
                         compute_psychometric_curve, compute_update_matrix, compute_phase_stats,
                         plot_psychometric_curve, plot_update_matrix, plot_trajectory,
                         PALETTE, apply_style)

apply_style()

# 1. LOAD
experiment = load_experiment('config.yaml')
animal = experiment.get_animal('A05')

# 2. SELECT sessions, FILTER trials
sessions = select_sessions(animal, preset='expert')
clean = filter_trials(sessions)
arrays = TrialArrays.from_sessions(clean)

# 3. COMPUTE — scalars, readouts, per-session trajectory
scalars = compute_stats(arrays, ['accuracy', *PSYCHOMETRIC])          # pd.Series
curve   = compute_psychometric_curve(arrays, n_bootstrap=200)         # PsychometricCurve
um      = compute_update_matrix(arrays)                               # UpdateMatrix
traj    = compute_phase_stats(clean, ['accuracy', 'mu'], per_session=True)   # PhaseStats

# 4. PLOT — draw-only, one axes each
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_psychometric_curve(curve, ax=axes[0], color=PALETTE[0])
plot_update_matrix(um, ax=axes[1])
plot_trajectory(traj, 'accuracy', ax=axes[2])
```

### Comparing two conditions

```python
from behav_utils import filter_trials, compute_delta_stat, compute_interaction, PSYCHOMETRIC
from behav_utils.plotting import plot_comparison, plot_stat_comparison, plot_interaction

off = filter_trials(sessions, trial_type='non_opto')
on  = filter_trials(sessions, trial_type='opto')

d = compute_delta_stat({'off': off, 'on': on}, stats=['accuracy', *PSYCHOMETRIC], reference='off',
                       n_bootstrap=1000, n_permutations=1000, resample_units=('trials', 'sessions'))
plot_comparison(d)                        # psychometric curves of both conditions
plot_stat_comparison(d, ['mu', 'sigma'])  # Δ with intervals per stat
```

`compute_delta_stat` returns a `DeltaStats`:

| field | what it holds |
|---|---|
| `phases` | `{label: PhaseSummary}` — observed `stats` (Series), `n_trials`, `n_sessions`, bootstrap `draws` per unit, optional `curve` / `update_matrix` |
| `contrasts` | `{key: Contrast}` — one per non-reference phase: `diff` (a − b), `difference_draws` per unit, `perm_p` (trial permutation, when valid), `um_diff` / `um_rmse` / `um_corr` |
| `reference` | the reference label |

`Contrast.boot(unit)` gives `ci_lo, ci_hi, p, median, n_draws` per stat; `Contrast.table(unit)` one
tidy row per stat (`diff, ci_lo, ci_hi, boot_p, perm_p, unit`). Units: `'trials'` (valid only when the
condition was randomised per trial) and `'sessions'` (always valid). `compute_interaction(d1, d2, key)`
gives the delta of deltas as an `Interaction`, with `plot_interaction`.

### Group-level claims (across animals)

For across-animal comparisons (two groups, or paired on/off effects), use the
group-level functions — never pool trials across animals.

`compute_phase_stats` turns each animal's sessions into one pooled row per stat;
`across_animals` / `group` test the per-animal values. The *animal* is the unit.

```python
from behav_utils.analysis import collect_rows, compare_groups, paired_diff, bootstrap_units
from behav_utils import compute_phase_stats, PSYCHOMETRIC

# One pooled row per animal, stamped with its group
rows = []
for a in animals:
    r = compute_phase_stats(sessions_of(a), [*PSYCHOMETRIC, 'accuracy'])
    rows += collect_rows(r.to_rows().to_dict('records'), animal=a.animal_id,
                         group=a.metadata['group'])

# Unpaired (between-group): one rank test per stat, control first
res = compare_groups(rows, groups=('control', 'treated'))
res['mu']['p'], res['mu']['min_p']        # min_p = floor at these group sizes

# Paired (on vs off within the same animals): per-animal Δ, then an across-animal CI
points = pd.DataFrame(rows_on + rows_off)                 # rows stamped condition='on' / 'off'
delta = paired_diff(points, by='condition', a='on', b='off')
bootstrap_units(delta[delta.stat == 'mu'].delta.values)
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
# Per-session trajectory of PSE
r = compute_phase_stats(clean, ['accuracy', 'mu', 'sigma'], per_session=True)
plot_trajectory(r, 'mu')      # y-axis label: "PSE"
plot_trajectory(r, 'sigma')   # y-axis label: "slope"
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