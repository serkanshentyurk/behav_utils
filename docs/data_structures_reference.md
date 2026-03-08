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

Each level provides filtering, stat computation, and plotting. Higher levels delegate to lower levels with selection and combination logic.

---

## TrialData

The lowest level — per-trial arrays for a single session.

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `trial_number` | int array | Original trial numbers from CSV |
| `stimulus` | float array | Stimulus values (raw from CSV) |
| `choice` | float array | Category-space choice: 0=A, 1=B, NaN=no response |
| `choice_raw` | float array | Raw choice from CSV (before conversion) |
| `outcome` | str array | Trial outcome (e.g., 'Correct', 'Incorrect', 'Abort') |
| `correct` | bool array | Whether choice matched category |
| `category` | int array | Derived from stimulus: 0=A (below boundary), 1=B (above) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `reaction_time` | float array | Response latency (NaN for aborts/no-response) |
| `abort` | bool array | Whether animal broke fixation |
| `opto_on` | bool array | Whether optogenetics was active |
| `distribution` | str array | Stimulus distribution name |

Additional mapped columns from the config are in `optional_fields` (dict). Unmapped CSV columns are in `extra` (dict).

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `.n_trials` | int | Total trial count |
| `.no_response` | bool array | True where `choice` is NaN |
| `.valid_mask` | bool array | `~abort & ~no_response` |

### Key Methods

#### `get_arrays(exclude_abort=True, exclude_opto=True, exclude_no_response=False)`

The standard interface for extracting analysis-ready arrays. Applies filtering, returns a dict:

```python
arrays = session.trials.get_arrays(exclude_abort=True, exclude_opto=True)
```

Returns:

| Key | Description |
|-----|-------------|
| `'stimuli'` | Stimulus values (filtered) |
| `'categories'` | True categories 0/1 (filtered) |
| `'choices'` | Choices 0/1/NaN (filtered) |
| `'no_response'` | Boolean mask for NaN choices |
| `'reaction_times'` | RT values (filtered) |
| `'trial_indices'` | Original indices for back-mapping |

#### `get_field(name)`

Access any field by name — checks core fields, `optional_fields`, then `extra`:

```python
opto_mask = session.trials.get_field('opto_mask')
```

#### `stats(stat_names=None, exclude_abort=True, exclude_opto=True)`

Compute summary statistics:

```python
s = session.trials.stats(['accuracy', 'recency'])
# {'accuracy': 0.82, 'recency': 0.15}
```

---

## SessionData

One behavioural session — contains metadata and trial data.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | str | Unique identifier |
| `session_idx` | int | Ordinal position within animal's timeline |
| `date` | datetime.date | When the session was run |
| `metadata` | SessionMetadata | Task parameters |
| `trials` | TrialData | Trial-by-trial data |
| `csv_path` | str or None | Source file path |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `.n_trials` | int | Total trials |
| `.stage` | str | Training stage (from metadata) |
| `.distribution` | str | Primary stimulus distribution |
| `.days_since_first` | float | Calendar days from animal's first session |

### Methods

```python
# Summary statistics
stats = session.stats(['accuracy', 'recency', 'psychometric'])

# Quick summary dict
info = session.summary()
# {'session_id': '...', 'perf': 0.82, 'n_valid': 245, ...}

# Plotting (returns (fig, ax, info))
fig, ax, info = session.plot_psychometric()

# Trial raster (returns (fig, ax))
fig, ax = session.plot_trials(window=20)
```

---

## SessionMetadata

Task parameters that are constant within a session. Populated from the config's `session_metadata` mappings.

All fields are stored in a `fields` dict. Access by attribute or `.get()`:

```python
session.metadata.stage                    # attribute access
session.metadata.get('sound_contingency') # dict access with default
session.metadata.fields                   # raw dict
```

Common fields (exposed as properties): `animal_id`, `stage`, `sound_contingency`, `stim_range_min`, `stim_range_max`.

---

## AnimalData

All data for one animal. Sessions stored chronologically. This is the unit of model fitting and longitudinal analysis.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `animal_id` | str | e.g., 'SS05' |
| `sessions` | List[SessionData] | Chronological list |
| `metadata` | dict | Optional animal-level info |

### Filtering

```python
# By stage
task_sessions = animal.get_sessions(stage='Full_Task_Cont')

# By date range
recent = animal.get_sessions(date_range=(date(2026, 1, 1), date(2026, 3, 1)))

# By index range
first_ten = animal.get_sessions(idx_range=(0, 9))
```

### Stats and Features

```python
# Feature matrix (cached after first call)
df = animal.feature_matrix(stage='Full_Task_Cont')
# DataFrame: one row per session, columns = all summary stats

# Single stat trajectory
indices, values = animal.stat_trajectory('accuracy')

# Expert baseline (last N sessions)
baseline = animal.expert_baseline(['accuracy', 'recency'], last_n=5)
# {'accuracy': {'mean': 0.85, 'std': 0.03}, 'recency': {'mean': 0.08, 'std': 0.02}}

# Flexible trial-level extraction
data = animal.get_trial_data(
    fields=['stimuli', 'choices', 'categories', 'reaction_times'],
    stage='Full_Task_Cont',
)
# Returns dict with 'session_arrays' (list of dicts), 'session_ids', etc.
```

### Plotting

```python
# Psychometric curves
fig, ax, infos = animal.plot_psychometric(sessions='last_5', mode='overlay')
fig, axes, infos = animal.plot_psychometric(sessions='all', mode='grid')
fig, ax, info = animal.plot_psychometric(sessions='last_5', mode='pooled')

# Stat trajectory
fig, ax = animal.plot_trajectory('accuracy', stage='Full_Task_Cont')
```

#### Session Selectors

The `sessions` argument accepts:
- `'all'` — all sessions (optionally filtered by `stage`)
- `'last_5'`, `'last_10'`, etc. — last N sessions
- `'first_5'`, `'first_3'`, etc. — first N sessions
- `[0, 5, 10, -1]` — specific indices

### Persistence

```python
animal.save('animal_SS05.pkl')
animal = AnimalData.load('animal_SS05.pkl')
```

---

## ExperimentData

Top-level container for all animals. Provides the query API for multi-animal analysis.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `animals` | Dict[str, AnimalData] | Animals keyed by ID |
| `metadata` | dict | Optional experiment-level info |
| `config` | ProjectConfig or None | The config used to load the data |

### Filtering

```python
# Get one animal
animal = experiment.get_animal('SS05')

# Filter animals
good = experiment.get_animals(min_sessions=10, stage='Full_Task_Cont')
specific = experiment.get_animals(animal_ids=['SS05', 'SS08'])

# Get all sessions matching criteria
sessions = experiment.get_sessions(stage='Full_Task_Cont', min_sessions_per_animal=10)
```

### Stats

```python
# Pooled feature matrix across all animals
df = experiment.feature_matrix(stage='Full_Task_Cont', min_sessions=10)

# Summary table
experiment.summary()
# DataFrame: animal_id, n_sessions, stages, date_first, date_last
```

### Query API: Plotting

```python
# Stat trajectory across all animals
fig, ax = experiment.plot_trajectory(
    stat='accuracy',
    animals='all',              # or ['SS05', 'SS08']
    stage='Full_Task_Cont',
    combine='mean_sem',         # 'mean_sem', 'median_iqr', 'individual', 'none'
    min_sessions=10,
)

# Psychometric curves across animals
fig, ax, info = experiment.plot_psychometric(
    animals='all',
    sessions='last_5',
    mode='pooled',
    stage='Full_Task_Cont',
)
```

### Persistence

```python
experiment.save('experiment.pkl')
experiment = ExperimentData.load('experiment.pkl')
```

---

## Data Flow

```
YAML config
    │
    ▼
load_experiment('config.yaml')
    │
    ├── Scans data_dir for animal directories
    │   └── For each animal:
    │       ├── Scans for session directories
    │       │   └── For each session:
    │       │       ├── Reads CSV
    │       │       ├── Maps columns via config
    │       │       ├── Converts choice to category space
    │       │       ├── Derives category from stimulus
    │       │       └── Builds SessionData
    │       └── Builds AnimalData (chronological)
    └── Builds ExperimentData
            │
            ├── experiment.get_animal('SS05')
            │       │
            │       ├── animal.sessions[10].stats(...)
            │       ├── animal.feature_matrix(...)
            │       ├── animal.plot_trajectory(...)
            │       └── animal.get_trial_data(...)
            │
            └── experiment.plot_trajectory(...)
```

---

## Synthetic Data

Generate test data without real experiments:

```python
from behav_utils import generate_synthetic_animal
from behav_utils.data.synthetic import noisy_psychometric_simulator

# Built-in simulator
animal, info = generate_synthetic_animal(
    n_sessions=20,
    simulator=noisy_psychometric_simulator,
    simulator_kwargs={'sigma': 0.3},
)

# Custom simulator
def my_simulator(stimuli, categories, rng, **kwargs):
    # Return choices as 0/1 array
    ...
    return choices

animal, info = generate_synthetic_animal(
    simulator=my_simulator,
    per_session_simulator_kwargs=[
        {'learning_rate': 0.5 - i * 0.02} for i in range(20)
    ],
)
```

The simulator callable signature is: `(stimuli, categories, rng, **kwargs) → choices`. This lets you plug in any model without the library knowing about it.

---

## Neural Data (Stub)

Interface defined for future implementation:

```python
from behav_utils.data.neural import NeuralData, Epoch

# Structure: (n_neurons, n_trials, n_timepoints)
neural = NeuralData.from_arrays(
    traces=np.random.randn(200, 250, 90),
    trial_indices=np.arange(250),
    neuron_types=np.array(['excitatory'] * 180 + ['inhibitory'] * 20),
    epochs=[
        Epoch('stimulus', start_idx=0, end_idx=30),
        Epoch('delay', start_idx=30, end_idx=60),
        Epoch('choice', start_idx=60, end_idx=90),
    ],
)

neural.get_epoch('stimulus')              # (200, 250, 30)
neural.get_neurons_by_type('inhibitory')  # (20, 250, 90)
```

Suite2P and CaImAn loaders are planned but not yet implemented.
