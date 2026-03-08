# Behaviour Utils

**Config-driven behavioural data analysis for neuroscience.**

If you run trial-based behavioural experiments — 2-AFC discrimination, go/no-go, forced choice — you've written the same data loading, psychometric fitting, and plotting code for every project. Column names change, file structures differ, choice encodings vary, but the underlying analysis is the same: stimuli go in, choices come out, and you want to know how the animal's behaviour evolves across sessions.

behav_utils solves this with a single YAML config file. Map your CSV columns once, and the entire analysis pipeline works — summary statistics, psychometric curves, serial dependence matrices, learning trajectories, multi-animal comparisons. Move to a new experiment or a new lab, write a new config, and everything keeps working.

## What It Does

```
Your CSV files          →  config.yaml  →  behav_utils  →  Analysis & Plots
(any column names,         (column map,     (loading,       (psychometric curves,
 any file structure,        task params,      stats,          trajectories,
 any choice encoding)       analysis          plotting)       feature matrices,
                            defaults)                         update matrices)
```

### Data hierarchy

```
ExperimentData                    ← all animals
  └── AnimalData                  ← one animal, all sessions
        └── SessionData           ← one session
              ├── SessionMetadata ← task parameters
              └── TrialData       ← trial-by-trial arrays
                    ├── stimulus, choice, outcome, correct, category
                    ├── reaction_time, abort, opto_on, ...
                    └── any additional columns you define
```

Every level has **stats**, **plotting**, and **filtering** methods. Plot a psychometric curve for one session, overlay curves across an animal's learning, or show group-mean accuracy trajectories across 40 animals — same API, different level.

### Key features

- **Config-driven** — one YAML file maps your CSV columns. No library code changes between projects.
- **Stats at every level** — `session.stats()`, `animal.stat_trajectory()`, `experiment.feature_matrix()`
- **20+ summary statistics** — accuracy, psychometric parameters, recency, serial dependence, history regression, choice entropy, and more. All via a registry pattern — add your own with `@register_stat`.
- **Plotting at every level** — psychometric curves (single, overlay, grid, pooled with bootstrap CIs), trial rasters, stat trajectories, update matrices. All return `(fig, ax)` for customisation.
- **Query API** — `experiment.plot_trajectory('accuracy', combine='mean_sem', stage='Full_Task_Cont')` handles animal selection, session filtering, and aggregation in one call.
- **Model-agnostic synthetic data** — bring your own simulator or use built-in ones. Test your pipeline without real data.
- **Flexible column access** — any column in your CSV is accessible. Define custom column groups for your specific model inputs.
- **Neural data ready** — interface defined for trial-aligned calcium imaging and electrophysiology.

---

## Installation

```bash
git clone https://github.com/yourusername/behav_utils.git
cd behav_utils
pip install -e .
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `pyyaml`

---

## Quick Start

### 1. Write a config

Your CSV has columns called `Stim_Relative`, `Choice`, `Trial_Outcome`. Your behav_utils config maps them:

```yaml
# config.yaml
project:
  name: "My Experiment"

file_structure:
  data_dir: "/path/to/data"
  behaviour_file: "trial_summary*.csv"

task:
  boundary: 0.0
  stimulus_range: [-1.0, 1.0]
  choice_mapping:
    type: "spatial_to_category"
    no_response_value: 0
    contingency_field: "sound_contingency"
    contingency_rules:
      Standard:
        -1: 0    # left → category A
        1: 1     # right → category B

columns:
  trial_number: { csv_name: "Trial_Number", dtype: int }
  stimulus:     { csv_name: "Stim_Relative", dtype: float }
  choice:       { csv_name: "Choice", dtype: int }
  outcome:      { csv_name: "Trial_Outcome", dtype: str }
  correct:      { csv_name: "Correct", dtype: bool }
  reaction_time: { csv_name: "Response_Latency", dtype: float, optional: true }

session_metadata:
  stage: { csv_name: "Stage", dtype: str }

analysis:
  default_stage: "Full_Task_Cont"
```

See the [Configuration Guide](docs/config_guide.md) for all options and examples for different experiment types.

### 2. Load and explore

```python
from behav_utils import load_experiment, apply_style
apply_style()

experiment = load_experiment('config.yaml')
animal = experiment.get_animal('SS05')
session = animal.sessions[-1]

# Stats
stats = session.stats(['accuracy', 'recency', 'psychometric'])
print(f"Accuracy: {stats['accuracy']:.3f}")
print(f"Recency: {stats['recency']:.3f}")

# Psychometric curve
fig, ax, info = session.plot_psychometric(show_params=True)
```

### 3. Learning trajectories

```python
# One animal
fig, ax = animal.plot_trajectory('accuracy')

# All animals, mean ± SEM
fig, ax = experiment.plot_trajectory(
    'accuracy', combine='mean_sem', stage='Full_Task_Cont',
)
```

### 4. Feature matrix

```python
# All stats × all sessions for one animal
df = animal.feature_matrix()

# Pooled across all animals
df = experiment.feature_matrix(min_sessions=10)
```

### 5. Access any column

```python
# Standard arrays
arrays = session.trials.get_arrays()
stimuli = arrays['stimuli']
choices = arrays['choices']

# Any column from your CSV
rt = session.trials.get_field('reaction_time')
location = session.trials.get_field('location_x')

# Multiple fields across sessions (for model fitting)
data = animal.get_trial_data(
    fields=['stimuli', 'choices', 'categories', 'reaction_times'],
    stage='Full_Task_Cont',
)
# Returns per-session arrays ready for your model
```

### 6. Without real data

```python
from behav_utils import generate_synthetic_animal
from behav_utils.data.synthetic import noisy_psychometric_simulator

animal, info = generate_synthetic_animal(
    n_sessions=20,
    simulator=noisy_psychometric_simulator,
    simulator_kwargs={'sigma': 0.3, 'lapse': 0.05},
)

# Everything works the same
fig, ax, info = animal.sessions[10].plot_psychometric()
df = animal.feature_matrix()
```

### 7. Custom model as simulator

```python
def my_model(stimuli, categories, rng, learning_rate=0.1, **kwargs):
    """Your model here — returns choices as 0/1 array."""
    ...
    return choices

animal, info = generate_synthetic_animal(
    simulator=my_model,
    per_session_simulator_kwargs=[
        {'learning_rate': 0.5 - i * 0.02} for i in range(20)
    ],
)
```

### 8. Custom summary statistic

```python
from behav_utils.analysis.summary_stats import register_stat

@register_stat('my_metric')
def compute_my_metric(choices, stimuli, categories):
    valid = ~np.isnan(choices)
    return float(np.mean(choices[valid] == categories[valid]))

# Immediately available everywhere
session.stats(['my_metric'])
df = animal.feature_matrix()  # includes my_metric
```

---

## Package Structure

```
behav_utils/
├── config/
│   └── schema.py              # Config dataclass, YAML loading, validation
├── data/
│   ├── structures.py          # ExperimentData, AnimalData, SessionData, TrialData
│   ├── loading.py             # Config-driven CSV loading
│   ├── synthetic.py           # Synthetic data generation
│   └── neural.py              # Neural data container (stub)
├── analysis/
│   ├── utils.py               # cumulative_gaussian, generate_stimuli
│   ├── psychometry.py         # Psychometric curve fitting + bootstrap
│   ├── summary_stats.py       # Registry of 20+ summary statistics
│   ├── update_matrix.py       # Serial dependence matrices
│   └── session_features.py    # Session-level feature matrix builder
├── plotting/
│   ├── styles.py              # Colours, themes, defaults
│   ├── psychometric.py        # Psychometric curves (single, overlay, grid, pooled)
│   ├── session.py             # Trial rasters
│   ├── trajectory.py          # Stat trajectories across sessions/animals
│   └── update_matrix.py       # Update matrix heatmaps + profiles
└── configs/
    └── sound_categorisation.yaml  # Example config
```

## Documentation

| Document | What it covers |
|----------|---------------|
| [Configuration Guide](docs/config_guide.md) | How to write a config. Column mapping, choice encoding, file structure. Three worked examples for different experiment types. |
| [Data Structures Reference](docs/data_structures.md) | Class hierarchy, every field and method, data flow diagram. |
| [Summary Statistics Reference](docs/summary_stats.md) | All 20+ stats with formulas, interpretation, and usage. |
| [Example Notebook](notebooks/example_workflow.ipynb) | Full workflow on synthetic data. Anyone can run it. |

## Design Principles

1. **Config over code** — column names in YAML, not scattered through analysis scripts
2. **Methods at every level** — session, animal, and experiment all have `.stats()` and `.plot_*()`
3. **Dual access** — `session.plot_psychometric()` calls `plot_psychometric(stimuli, choices)` internally. Use either.
4. **Always return `(fig, ax)`** — every plot function. No exceptions.
5. **Raw arrays underneath** — all analysis functions take plain numpy arrays. Data classes are convenience, not lock-in.
6. **Register to extend** — `@register_stat('name')` and your stat works everywhere instantly.

## Status

Core modules (loading, analysis, plotting) are stable and tested. Neural data support is stubbed. Not yet pip-published but installable from source.



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
