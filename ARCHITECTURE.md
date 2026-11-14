# behav_utils — architecture

## Layers

Dependencies point downward only. A layer never imports from one above it, and the library never
imports from a project.

```
plotting     draw-only: plot_x(result, ax=None) for every compute_x
analysis     phase-level: compute_phase_stats, contrasts (DeltaStats), resampling, rolling, group tests
models       behavioural models, designs, simulate() -> a SessionData scored by the task; synthetic data
stats        scalar statistics registry: compute_stats(TrialArrays, names) -> pd.Series
readouts     array-valued readouts: curves, matrices, profiles as frozen dataclasses
data         structures, readers, discovery, loading, the store, select_sessions, filter_trials, TrialArrays
config       YAML schema and loader
tasks        what counts as correct: Task, TwoAFC, GoNoGo, Outcome; stimulus distributions (numpy only)
```

`stats` may import `readouts` (the `sd_profile` fit uses the profile readout); `readouts` never imports
`stats`. The two fit engines on raw arrays, `fit_psychometric` and `fit_update_matrix`, live in
`readouts/fit.py`, which imports nothing from the package; `stats`, `readouts` and `analysis` all use them.

One job per file:

```
tasks/       base.py  categorisation.py (binary)  continuous.py  distributions.py
data/        structures.py  arrays.py  readers.py  loading.py  discovery.py  store.py  synthetic.py
             ops/{selection,filtering,switches}.py
stats/       registry.py  basic.py  psychometric.py  history.py  dynamics.py  rt.py  sdt.py  continuous.py
readouts/    fit.py  psychometric.py  update_matrix.py  conditional_psychometric.py  binned.py  sd_profile.py
             continuous.py (response curve, error distribution, serial-bias curve)
models/      base.py (Model, TrialwiseModel, Design, make_design)  builtin.py  simulate.py  generate.py
             continuous.py (ReproductionModel)
analysis/    phase.py (compute_phase_stats)  comparison.py (contrasts)  resampling.py (draw engine + summaries)
             group.py (across-animal combine / resample / test)  rolling.py
plotting/    readouts.py  comparison.py  trajectory.py  styles.py  continuous.py
```

## The pipeline
load → `select_sessions` → `filter_trials` → `TrialArrays` → `compute_x` → `plot_x`, explained in
[The pipeline](docs/concepts/pipeline.md). Selection and filtering are the only places that decide *which*
trials; every compute takes what it is given. A compute that filters internally is a bug.

## Contracts

### Public API
Public = the names in `behav_utils.__all__` and in each sub-package's `__all__` (`config`, `data`, `stats`,
`readouts`, `analysis`, `plotting`). Every name has one home sub-package, which `tests/test_public_api.py`
enforces; the top level re-exports the common ones. Deep module paths and `_`-prefixed names are private,
including helpers shared across modules (`_validate_names`). The list, with signatures, is generated into
`docs/public_api.md`.

### TrialArrays (`data/arrays.py`)
Why `trial_key` exists and how it is built: [Identities and time](docs/concepts/identity-and-time.md).
Frozen dataclass of 1-D float arrays of equal length: `choice, stimulus, category, prev_choice,
prev_stimulus, prev_category, reaction_time`, plus an optional int64 `trial_key`
(built by `from_sessions` from each session's hash of experimenter, animal and session id in the upper 43 bits and the trial number in the low 20, the same on every load and machine and unique across animals; `None` from
`from_sequence`). `choice`
is NaN on no-response trials; `prev_*` are NaN where the trial has no usable predecessor (first trial
of a session, or the predecessor was an abort or no-response), so pooled data never bridge a session
seam. `trial_key` travels with the rows through `take`, `valid`, `lag1_pairs`, `concat` and every
resample (a trial drawn twice carries its key twice), so an external per-trial dataset can be masked
to exactly the rows a statistic used. Constructors: `from_sessions`, `from_pooled`, `from_sequence`
(simulator path, adjacency). `valid()`, `lag1_pairs()`, `take(mask)`, `concat([...])`,
`with_trial_key(keys)`.

### The stimulus axis
Explained in [The stimulus axis](docs/concepts/stimulus-axis.md). Everything downstream of the loader
assumes it: binning (`readouts/_base.py`, the one grid), the psychometric fit bounds and the hard/easy split.
`TaskConfig.normalise_stimulus` maps raw units onto it at load; `TrialData.stimulus_raw` keeps the raw value,
and `category` is derived from the normalised one.

### The store (`data/store.py`)

`save_store` / `read_manifest` / `load_store` (and `ExperimentData.save` / `.load` over them) persist an experiment
as Parquet tables and JSON: a manifest that is the shared session table, one trial file per session, and JSON for
the store's version and the animals' metadata. Only `_write_table` and `_read_table` touch Parquet. Writes are
atomic (a temporary directory moved into place), and a store records its schema version so later versions can
migrate it. Pickle is not used anywhere in the library.

### Task families
Explained in [Tasks, families and capabilities](docs/concepts/tasks.md). Behaviour that differs by family
keys on the task's response space or capabilities, never on its class; statistics, readouts and models declare
the capabilities they need and `require` checks them. The binary family's modules keep their original names,
the continuous family's are the `continuous.py` modules, and `TaskConfig.normalise_continuous` maps a
continuous task's stimulus and response together.

### The clock (`config/`, `data/loading.py`)
Explained in [Identities and time](docs/concepts/identity-and-time.md). Units are converted once, at load,
with the checks there; `TrialData.events` is sliced with the trials like `optional_fields` and `extra`
(fields marked `per_trial_map`).

### Statistics (`stats/`)
A statistic is `f(arrays: TrialArrays, *, rng) -> float`, registered with `@stat(name, exchangeable=, requires=,
instead=)`. `requires` names the task capabilities it needs (`Task.capabilities`; open names); `compute_stats`
refuses a requested statistic the arrays' task cannot support with `RequirementError` naming the `instead`
counterpart, and readouts call `require` first.
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
- Resampling units (`trials`, stratified by stimulus bin; `sessions`, whole sessions) and when a
  permutation p is allowed: [Resampling units](docs/concepts/resampling.md). `summarise_draws` returns a
  `DrawSummary`.
- `compute_rolling_stats(sessions, names, per_session=True) -> RollingStats` (tidy `curves` +
  `session_info`; windows never cross a session boundary).
- Group level (`analysis/group.py`) operates on tidy per-animal rows, never on sessions: `collect_rows`
  (stamp the animal — the explicit id always wins), `combine`, `paired_diff` (aligned by unit),
  `bootstrap_units -> UnitBootstrap`, `rank_test -> RankTest` (with `min_p`), `compare_groups ->
  GroupComparison` (one `RankTest` per stat; `paired=True` aligns by animal), `average_arrays ->
  ArrayAverage`, `min_achievable_p`. `PhaseStats.animal` is read from `session.metadata.animal_id`.

### Plotting (`plotting/`)
One `plot_x(result, ax=None) -> (fig, ax)` per result type; single panel; layout is the caller's. Two
CI units are drawn side by side when both are present (trial CI thin grey, session CI thick coloured).
The documented exception: `plot_stat_comparison_grid` and `plot_interaction_grid` lay out one panel per
stat and call the single-panel plotter in each — they draw nothing themselves.

### Config (`config/`)
A YAML file maps CSV columns and session metadata to the structures, declares the task (boundary,
stimulus range and scale, choice encoding), names cohorts, defines `session_presets` (resolved on the
animal's own config first, then the process-wide registry; a preset may carry `exclude_types`) and
`session_types` (`{type_name: {animal_id: [YYYYMMDD, ...]}}`, stamped onto sessions). The library
derives only `regular` and `opto` from the data; every other session type is the project's
vocabulary. Loading is strict by default: a session that fails to load raises rather than shifting
every later session index. `ExperimentData.save` stamps the library version; `load` warns on a mismatch.
Every key `load_config` reads is checked: an unknown key raises `ConfigError` (a `ValueError`) naming the
section, the valid keys and the closest match, and a top-level `custom:` block is left for the project and
never read. `task.choice_mapping.no_response_value` is required whenever a choice column is mapped (`null`
when no raw value means no response). Loading has two halves: a reader (`data/readers.py`; chosen by the
`reader:` block or passed to `load_experiment`) finds sessions and reads raw tables, and `build_session` maps
each onto `SessionData` the same way for every reader. Relative paths in a config resolve against its file.

### Data classes (`data/structures.py`)
Hold data only. `SessionData.summary() -> SessionSummary` (counts, one definition of session
accuracy) and `AnimalData.session_table` (one row per session built from it, for picking sessions)
are the only computations; the statistics of a session come from `compute_phase_stats(per_session=True)`.

## What is deliberately not here
Listed in [Concepts](docs/concepts/index.md): anything specific to one experiment, model fitting, and neural
data beyond the trial key.

## Invariants worth knowing when changing things
- `pool_arrays` and the loader's lag-1 freezing define `prev_*`; `TrialArrays.from_sequence` reproduces
  the loader's rule exactly (tested).
- Stat values are pinned by the project's reference tables; changing a stat's definition is a versioned
  change.
- The drawing half of `analysis/resampling.py` (`downsample`, `resample_*`) is the single engine for bootstrap; changing it changes every interval.

## Tests
`tests/` has its own fixtures (`conftest.py`, synthetic animals, test-local presets). Run `pytest tests -q`
from `behav_utils/`. No config file or data is required. `tests/test_golden.py` pins every number on a frozen
dataset (`tests/golden/`); `tests/test_public_api.py` checks the public names and that `docs/public_api.md`
is current.
