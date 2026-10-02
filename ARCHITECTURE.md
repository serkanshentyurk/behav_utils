# behav_utils — architecture

## Layers

Dependencies point downward only. A layer never imports from one above it, and the library never
imports from a project.

```
plotting     draw-only: plot_x(result, ax=None) for every compute_x
analysis     phase-level: compute_phase_stats, contrasts (DeltaStats), resampling, rolling, group tests
stats        scalar statistics registry: compute_stats(TrialArrays, names) -> pd.Series
readouts     array-valued readouts: curves, matrices, profiles as frozen dataclasses
data         structures, CSV loading, select_sessions, filter_trials, pool_arrays, TrialArrays, synthetic
config       YAML schema and loader
```

`stats` may import `readouts` (the `sd_profile` fit uses the profile readout); `readouts` never imports
`stats`. Two fit engines on raw arrays live in `analysis` for historical reasons and are used by both:
`analysis.psychometry.fit_psychometric` and `analysis.update_matrix.fit_update_matrix`.

## The pipeline

```
load_experiment(config)          -> ExperimentData {animal_id: AnimalData [SessionData ...]}
select_sessions(animal, ...)     -> list[SessionData]        (stage, distribution, preset, session_type, exclude_types)
filter_trials(sessions, ...)     -> list[SessionData]        (trial_type: all | non_opto | opto | post_opto; aborts dropped)
TrialArrays.from_sessions(...)   -> TrialArrays              (aligned float arrays + frozen lag-1 view)
compute_x(...)                   -> typed result
plot_x(result, ax)               -> the axes
```

Selection and filtering are the only places that decide *which* trials; every compute takes what it is
given. A compute that filters internally is a bug.

## Contracts

### TrialArrays (`data/arrays.py`)
Frozen dataclass of 1-D float arrays of equal length: `choice, stimulus, category, prev_choice,
prev_stimulus, prev_category, reaction_time`. `choice` is NaN on no-response trials; `prev_*` are NaN
where the trial has no usable predecessor (first trial of a session, or the predecessor was an abort or
no-response), so pooled data never bridge a session seam. Constructors: `from_sessions`, `from_pooled`,
`from_sequence` (simulator path, adjacency). `valid()`, `lag1_pairs()`, `take(mask)`.

### Statistics (`stats/`)
A statistic is `f(arrays: TrialArrays, *, rng) -> float`, registered with `@stat(name, exchangeable=)`.
A multi-output fit is `g(arrays, *, rng) -> tuple`, registered with `@fit(name, outputs=(...))`; each
output is a first-class stat name and the fitter runs once per call. `compute_stats(arrays, names)`
returns a float `pd.Series` in request order; `strict=False` fills a failed producer with NaN (the
resampling path). `exchangeable=False` marks order-dependent statistics (multi-lag regressions,
`pse_dynamics`): resamplers refuse to trial-resample them and require the session unit. There are no
family names — `PSYCHOMETRIC` is a tuple you splat.

### Readouts (`readouts/`)
`compute_x(arrays, ...) -> XResult`, a frozen dataclass with read-only arrays, a short `repr`, and
`to_rows()` (tidy). Per-session readouts are the caller's loop; reductions across blocks are classmethods
(`UpdateMatrix.average(items, min_sources)`). `PsychometricCurve` carries its bootstrap band because a
curve without uncertainty is not useful; everything else leaves uncertainty to `analysis.resampling`.

### Phase results (`analysis/`)
- `compute_phase_stats(phase, names, per_session=False) -> PhaseStats(pooled: Series, sessions: DataFrame|None, ...)`.
  `pooled` is always the genuine pooled fit, never a mean of per-session values.
- `compute_delta_stat(phases, names, reference=, units=, ...) -> DeltaStats` with `.phases[label]`
  (`PhaseSummary`: stats, bootstrap draws per unit, optional curve/UM) and `.contrasts[key]` (`Contrast`:
  `diff`, `difference_draws`, `perm_p`, `.boot(unit)`, `.table(unit)`). Each phase is bootstrapped once
  per unit and every contrast is a subtraction of stored draw frames, so a shared reference cancels exactly.
- `compute_interaction(r1, r2, key) -> Interaction` (difference of differences, bootstrap only).
- Resampling units: `trials` (stratified by stimulus bin; valid when the label was randomised per trial)
  and `sessions` (whole sessions; the honest unit when the condition varies by session). A permutation p
  is computed only for within-session contrasts. `summarise_draws` returns a `DrawSummary`.
- `compute_rolling_stats(sessions, names, per_session=True) -> RollingStats` (tidy `curves` +
  `session_info`; windows never cross a session boundary).
- Group tests (`across_animals`, `group`) operate on tidy per-animal rows: `collect_rows`,
  `compare_groups`, `rank_test`, `paired_diff`, `min_achievable_p`.

### Plotting (`plotting/`)
One `plot_x(result, ax=None) -> (fig, ax)` per result type; single panel; layout is the caller's. Two
CI units are drawn side by side when both are present (trial CI thin grey, session CI thick coloured).

### Config (`config/`)
A YAML file maps CSV columns and session metadata to the structures, names cohorts, defines
`session_presets` (registered at load time; a preset may carry `exclude_types`) and `session_types`
(`{type_name: {animal_id: [YYYYMMDD, ...]}}`, stamped onto sessions). The library derives only
`regular` and `opto` from the data; every other session type is the project's vocabulary.

## What is deliberately not here
- Anything about a specific task: distribution names, normative observers, session-type semantics,
  genotypes, cohort logic, report layouts. Those belong to the project that uses the library.
- Model fitting (SBI, grid search) — project code.

## Invariants worth knowing when changing things
- `pool_arrays` and the loader's lag-1 freezing define `prev_*`; `TrialArrays.from_sequence` reproduces
  the loader's rule exactly (tested).
- Stat values are pinned by the project's reference tables; changing a stat's definition is a versioned
  change.
- `downsample.py` is the single drawing engine for bootstrap; changing it changes every interval.

## Tests
`tests/` has its own fixtures (`conftest.py`, synthetic animals, test-local presets). Run `pytest tests -q`
from `behav_utils/`. No config file or data is required.
