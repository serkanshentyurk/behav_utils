# Behavioural Utils Package

A config-driven Python library for loading, analysing, and plotting trial-based behavioural neuroscience data.

Built for 2-AFC (two-alternative forced choice) tasks but designed to generalise across projects, labs, and task variants. One YAML config file maps your CSV columns to standard internal names — no code changes needed when you move to a new experiment.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

---

## Features

- **Config-driven loading** — define column mappings in YAML, load any CSV format without touching library code
- **Hierarchical data classes** — `ExperimentData → AnimalData → SessionData → TrialData` with methods at every level
- **20+ summary statistics** — accuracy, psychometric parameters, recency, serial dependence, logistic history weights, and more, all via a registry pattern
- **Plotting at every level** — psychometric curves, trial rasters, stat trajectories, update matrices. All functions return `(fig, ax)` for customisation
- **Query API** — `experiment.plot_trajectory('accuracy', combine='mean_sem')` works across animals with filtering and aggregation
- **Synthetic data** — model-agnostic generation. Bring your own simulator or use built-in psychometric/random simulators for pipeline testing
- **Neural data stub** — interface defined for trial-aligned calcium imaging and electrophysiology (implementation pending)

## Installation

Clone the repository and install:

```bash
git clone https://github.com/serkanshentyurk/behav_utils.git
cd behav_utils
pip install -e .
```

Or use directly by adding to your Python path:

```bash
# In your project
sys.path.insert(0, '/path/to/behav_utils')
```

### Dependencies

```
numpy
pandas
matplotlib
scipy
pyyaml
```

## Quick Start

### 1. Create a config file

```yaml
# config.yaml
project:
  name: "My Experiment"

file_structure:
  data_dir: "/path/to/data"
  behaviour_file: "trial_summary*.csv"
  date_regex: "(?P<year>\\d{4})_(?P<month>\\d{1,2})_(?P<day>\\d{1,2})"

task:
  boundary: 0.0
  stimulus_range: [-1.0, 1.0]
  choice_mapping:
    type: "spatial_to_category"
    no_response_value: 0
    contingency_field: "sound_contingency"
    contingency_rules:
      Low_Left_High_Right:
        -1: 0
        1: 1

columns:
  trial_number:
    csv_name: "Trial_Number"
    dtype: int
  stimulus:
    csv_name: "Stim_Relative"
    dtype: float
  choice:
    csv_name: "Choice"
    dtype: int
  outcome:
    csv_name: "Trial_Outcome"
    dtype: str
  correct:
    csv_name: "Correct"
    dtype: bool

session_metadata:
  stage:
    csv_name: "Stage"
    dtype: str

analysis:
  default_stage: "Full_Task_Cont"
```

### 2. Load and explore

```python
from behav_utils import load_experiment, apply_style

apply_style()
experiment = load_experiment('config.yaml')

# Pick an animal
animal = experiment.get_animal('SS05')

# Session-level stats
session = animal.sessions[-1]
stats = session.stats(['accuracy', 'recency', 'psychometric'])
print(f"Accuracy: {stats['accuracy']:.3f}")

# Plot psychometric curve
fig, ax, info = session.plot_psychometric()
```

### 3. Analyse across sessions

```python
# Accuracy trajectory for one animal
fig, ax = animal.plot_trajectory('accuracy')

# Feature matrix (all stats × all sessions)
df = animal.feature_matrix()

# Expert baseline (last 5 sessions)
baseline = animal.expert_baseline(['accuracy', 'recency'], last_n=5)

# Group-level: all animals, mean ± SEM
fig, ax = experiment.plot_trajectory(
    'accuracy', combine='mean_sem', stage='Full_Task_Cont',
)
```

### 4. Use without real data (synthetic)

```python
from behav_utils import generate_synthetic_animal
from behav_utils.data.synthetic import noisy_psychometric_simulator

animal, info = generate_synthetic_animal(
    n_sessions=20,
    trials_per_session=300,
    simulator=noisy_psychometric_simulator,
    simulator_kwargs={'sigma': 0.3, 'lapse': 0.05},
)

# Everything works the same as with real data
fig, ax, info = animal.sessions[10].plot_psychometric()
df = animal.feature_matrix()
```

### 5. Bring your own model

```python
def my_model_simulator(stimuli, categories, rng, **kwargs):
    """Your model goes here."""
    noise = rng.normal(0, kwargs.get('sigma', 0.2), len(stimuli))
    p_b = 1 / (1 + np.exp(-(stimuli + noise) * 5))
    choices = (rng.random(len(stimuli)) < p_b).astype(float)
    return choices

animal, info = generate_synthetic_animal(
    simulator=my_model_simulator,
    simulator_kwargs={'sigma': 0.3},
    per_session_simulator_kwargs=[
        {'sigma': 0.5 - i * 0.02} for i in range(20)  # learning trajectory
    ],
)
```

## Package Structure

```
behav_utils/
├── config/
│   └── schema.py          # Config dataclass, YAML loading, validation
├── data/
│   ├── structures.py      # ExperimentData, AnimalData, SessionData, TrialData
│   ├── loading.py         # Config-driven CSV loading
│   ├── synthetic.py       # Synthetic data generation
│   └── neural.py          # Neural data container (stub)
├── analysis/
│   ├── utils.py           # cumulative_gaussian, generate_stimuli
│   ├── psychometry.py     # Psychometric curve fitting
│   ├── summary_stats.py   # Registry of 20+ summary statistics
│   ├── update_matrix.py   # Serial dependence matrices
│   └── session_features.py # Session-level feature matrix builder
├── plotting/
│   ├── styles.py          # Colours, themes, defaults
│   ├── psychometric.py    # Psychometric curve plots
│   ├── session.py         # Trial rasters
│   ├── trajectory.py      # Stat trajectories across sessions
│   └── update_matrix.py   # Update matrix heatmaps
└── configs/
    └── sound_categorisation.yaml  # Example config
```

## Documentation

- **[Data Structures Reference](docs/data_structures.md)** — class hierarchy, fields, methods, usage patterns
- **[Summary Statistics Reference](docs/summary_stats.md)** — all registered stats with formulas and interpretation
- **[Example Notebook](notebooks/example_workflow.ipynb)** — full workflow from loading to analysis to plotting

## Design Principles

1. **Config over code** — column mappings in YAML, not hardcoded strings
2. **Methods at every level** — `session.stats()`, `animal.plot_trajectory()`, `experiment.plot_trajectory()`
3. **Dual access** — data class methods call standalone functions. Use whichever feels natural
4. **Always return `(fig, ax)`** — every plot function returns the figure and axes for customisation
5. **Raw arrays underneath** — all analysis functions work on plain numpy arrays. Data classes are convenience, not requirement
6. **Extension by registration** — add new stats with `@register_stat('my_stat')`

## Adding a New Summary Statistic

```python
from behav_utils.analysis.summary_stats import register_stat

@register_stat('my_custom_stat')
def compute_my_stat(choices, stimuli, categories):
    """
    Your stat here. Receives filtered arrays (no NaN choices).
    Return a scalar or a dict of scalars.
    """
    valid = ~np.isnan(choices)
    return float(np.mean(choices[valid] == categories[valid]))
```

The stat is immediately available everywhere — `session.stats(['my_custom_stat'])`, feature matrices, SBI pipelines.

## Status

This library is under active development. The core data loading, analysis, and plotting modules are stable and tested. Neural data support is stubbed but not yet implemented.

## Licence

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Citation

If you use this library in published work, please cite:

```
behav_utils: Config-driven behavioural data analysis for neuroscience
https://github.com/yourusername/behav_utils
```
