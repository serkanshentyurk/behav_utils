# behav_utils

Config-driven library for loading, filtering, analysing, and plotting trial-based behavioural data.

## Architecture

Every analysis domain has three levels:

| Level | Purpose | Naming | Example |
|:------|:--------|:-------|:--------|
| **Low-level** | Raw arrays → computed result | `fit_X` / `compute_X` | `fit_psychometric(stim, ch)` |
| **Session-level** | Pre-filtered sessions → result dict | `compute_X` | `compute_psychometric(sessions)` |
| **Plotting** | Result dict → axes | `plot_X` | `plot_psychometric(result)` |

Low-level functions are always available for direct use with arbitrary arrays (model output, synthetic data, etc.). Session-level functions handle extraction and pooling. Plotting functions do zero computation.

## Pipeline

```python
from behav_utils import (
    load_experiment, select_sessions, filter_trials,
    compute_psychometric, compute_um, compute_trajectory, compute_comparison,
    plot_psychometric, plot_um, plot_trajectory, plot_comparison,
    PALETTE, apply_style,
)
apply_style()

# 1. Load
experiment = load_experiment('config.yaml')
animal = experiment.get_animal('SS05')

# 2. Select sessions (session-level)
sessions = select_sessions(animal, preset='expert_uniform')

# 3. Filter trials (trial-level)
clean = filter_trials(sessions)

# 4. Analyse (returns result dicts)
psych = compute_psychometric(clean, mode='pooled', n_bootstrap=200)
um = compute_um(clean)
traj = compute_trajectory(clean, ['accuracy', 'mu'])

# 5. Plot (draws result dicts)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_psychometric(psych, ax=axes[0])
plot_um(um, ax=axes[1])
plot_trajectory(traj, 'accuracy', ax=axes[2])
```

### Comparing two conditions

```python
ctrl = filter_trials(sessions)
opto = filter_trials(sessions, lambda s: s.trials.opto_mask(0))

# Option A: compute_comparison (full statistical comparison)
comp = compute_comparison(ctrl, opto, label_a='Control', label_b='Opto')
fig, ax = plt.subplots()
plot_comparison(comp, ax=ax, metric='psychometric')

# Option B: compute individually and overlay
ctrl_psych = compute_psychometric(ctrl)
opto_psych = compute_psychometric(opto)
fig, ax = plt.subplots()
plot_psychometric(ctrl_psych, ax=ax, color=PALETTE[0], label='Control')
plot_psychometric(opto_psych, ax=ax, color=PALETTE[1], label='Opto')
ax.legend()
```

### Using low-level functions directly

```python
from behav_utils.analysis import fit_psychometric, compute_update_matrix, compare_conditions

# With model-generated arrays
params = fit_psychometric(model_stimuli, model_choices)
um, cond, info = compute_update_matrix(stim, choices, categories)
comp = compare_conditions(stim_a, ch_a, cat_a, stim_b, ch_b, cat_b)
```

---

## Package Structure

```
behav_utils/
├── config/
│   └── schema.py            # YAML config loading, ProjectConfig dataclass
├── data/
│   ├── structures.py        # TrialData, SessionData, AnimalData, ExperimentData, FittingData
│   ├── loading.py           # CSV → data classes
│   ├── selection.py         # SessionFilter, presets, fitting_data_from_sessions
│   ├── filtering.py         # Trial-level filtering (single source of truth)
│   ├── synthetic.py         # Synthetic data generation
│   └── neural.py            # NeuralData, Epoch (imaging stub)
├── analysis/
│   ├── psychometry.py       # fit_psychometric (low), compute_psychometric (session)
│   ├── update_matrix.py     # compute_update_matrix (low), compute_um (session)
│   ├── trajectory.py        # compute_trajectory (session)
│   ├── comparison.py        # compare_conditions (low), compute_comparison (session)
│   ├── session_raster.py    # compute_session_raster (session)
│   ├── summary_stats.py     # 25+ registered stats, compute_summary_stats (low)
│   ├── session_features.py  # compute_session_features, build_feature_matrix
│   └── utils.py             # cumulative_gaussian, generate_stimuli
└── plotting/
    ├── psychometric.py      # plot_psychometric (result dict)
    ├── update_matrix.py     # plot_um (result dict)
    ├── trajectory.py        # plot_trajectory (result dict)
    ├── comparison.py        # plot_comparison (result dict)
    ├── session.py           # plot_session_raster (result dict)
    └── styles.py            # PALETTE, COLOURS, UM_CMAP, apply_style
```

---

## Data Pipeline

### Loading

| Function | Purpose |
|:---------|:--------|
| `load_experiment(config_or_path)` | Load all animals → `ExperimentData` |
| `load_animal(animal_dir, config)` | Load one animal → `AnimalData` |
| `load_session_csv(path, config)` | Load one CSV → `SessionData` |

### Session Selection

| Function | Purpose |
|:---------|:--------|
| `select_sessions(animal, preset=, **overrides)` | Filter sessions by preset or custom criteria |
| `fitting_data_from_sessions(sessions, animal_id)` | Pre-filtered sessions → `FittingData` for SBI |
| `register_preset(name, filter)` | Register a named preset |
| `list_presets()` | Show available presets |

### Trial Filtering

All filtering logic lives in `filtering.py`. Data classes have thin wrappers.

| Function | Purpose |
|:---------|:--------|
| `filter_trials(sessions, mask_fn)` | Batch-filter trials across sessions |
| `filter_session(session, mask, label)` | Filter one session's trials |
| `pool_arrays(sessions)` | Concatenate arrays across sessions |
| `build_mask(trials, ...)` | Build boolean exclusion mask |
| `opto_mask(trials, delta)` | Mask relative to opto events |
| `get_arrays(trials)` | Extract arrays (aborts always excluded) |

### Session Filter Presets

| Preset | Filters |
|:-------|:--------|
| `expert_uniform` | stage=Full_Task_Cont, distribution=Uniform, min_accuracy=0.70, last 50% |
| `all_uniform` | stage=Full_Task_Cont, distribution=Uniform |
| `naive_uniform` | stage=Full_Task_Cont, distribution=Uniform, first 5 |
| `all_hard_a` | stage=Full_Task_Cont, distribution=Hard-A |
| `expert_hard_a` | stage=Full_Task_Cont, distribution=Hard-A, min_accuracy=0.60, last 50% |
| `all_full_task` | stage=Full_Task_Cont |
| `all_stages` | No filter |

---

## Analysis Reference

### Low-level (raw arrays)

| Function | Input | Output |
|:---------|:------|:-------|
| `fit_psychometric(stimuli, choices)` | 1D arrays | Dict: mu, sigma, lapse_low, lapse_high, success, x_fit, y_fit |
| `compute_update_matrix(stim, ch, cat)` | 1D arrays | (um, conditional_matrix, info) |
| `compare_conditions(stim_a, ch_a, cat_a, stim_b, ch_b, cat_b)` | 1D arrays × 2 | Dict: params, diffs, p-values, CIs, UMs |
| `compute_summary_stats(ch, stim, cat, stat_names)` | 1D arrays | Dict: stat_name → value |
| `compute_session_features(session)` | SessionData | Dict: feature_name → value |
| `matrix_error(um_a, um_b)` | 2D arrays | float (RMSE) |
| `permutation_test_params(...)` | 1D arrays × 2 | Dict: param → p-value |
| `bootstrap_param_diff(...)` | 1D arrays × 2 | Dict: param → (lo, hi) |

### Session-level (sessions → result dict)

| Function | Input | Output |
|:---------|:------|:-------|
| `compute_psychometric(sessions, mode)` | List[SessionData] | Dict with mode-specific psychometric results |
| `compute_um(sessions)` | List[SessionData] | Dict with um, conditional_matrix, info |
| `compute_trajectory(sessions, stat_names)` | List[SessionData] | Dict with per-session stat values |
| `compute_comparison(sessions_a, sessions_b)` | List × 2 | Dict with diffs, p-values, CIs, UMs |
| `compute_session_raster(session)` | SessionData | Dict with trial-by-trial arrays |
| `build_feature_matrix(animal)` | AnimalData | DataFrame: sessions × features |

---

## Plotting Reference

All plotting functions take result dicts from the corresponding `compute_` function. No computation inside plotting.

| Function | Input | Draws |
|:---------|:------|:------|
| `plot_psychometric(result, ax)` | From `compute_psychometric` | Psychometric curve(s) with data points and CI |
| `plot_um(result, ax)` | From `compute_um` (or raw ndarray) | Update matrix heatmap |
| `plot_trajectory(result, stat_name, ax)` | From `compute_trajectory` | Per-session stat line |
| `plot_comparison(result, ax, metric)` | From `compute_comparison` | Psychometric overlay, accuracy bars, or UM comparison |
| `plot_session_raster(result, ax)` | From `compute_session_raster` | Trial-by-trial raster |

### Styles

| Item | Purpose |
|:-----|:--------|
| `PALETTE` | Indexed colour list for consistent group comparisons |
| `COLOURS` | Named colour dict (BE, SC, default, etc.) |
| `UM_CMAP` | Diverging colourmap for update matrices |
| `apply_style()` | Apply default matplotlib style |
| `get_colour(index_or_name)` | Resolve int→PALETTE, str→COLOURS |

---

## Synthetic Data

| Function | Purpose |
|:---------|:--------|
| `generate_synthetic_animal(animal_id, n_sessions, simulator, simulator_kwargs)` | Full synthetic animal; returns `(AnimalData, info)` |
| `generate_synthetic_session(n_trials, simulator, simulator_kwargs)` | Single session |
| `sample_stimuli(n, distribution='uniform', rng=)` | Draw `(stimuli, categories)` arrays |
| `noisy_psychometric_simulator(stimuli, categories, rng, sigma, lapse)` | Sigmoidal choice simulator |
| `random_choice_simulator(stimuli, categories, rng, accuracy)` | Fixed-accuracy choice simulator |

All simulators follow the signature: `(stimuli, categories, rng, **kwargs) -> choices`.
Parameters are passed via `simulator_kwargs` when using `generate_synthetic_animal`.
