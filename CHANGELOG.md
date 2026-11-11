# Changelog

## 1.0.0.dev0 — unreleased (branch `v1`)

The road to 1.0, on one branch with no compatibility shims. Number-affecting changes are marked ▲.

### Step 1: public API guard and tooling
- Golden values (`tests/golden/`): every statistic, per-session statistic, readout, laser-vs-off contrast and
  rolling statistic on a frozen two-animal dataset, pinned from 0.10.0. `tests/test_golden.py` fails when a
  number moves (closed-form values to 1e-12; fitted values to a relative 1e-6 with the same numpy, scipy and
  pandas, a relative 1e-3 otherwise, with an absolute floor of 1e-4).
- Public API: a name in `behav_utils.__all__` or in a sub-package's `__all__`. Every name has one home:
  `behav_utils.analysis` no longer re-exports the fit engines or `generate_stimuli`, and `behav_utils.data`
  now lists everything it owns. The generated `docs/public_api.md` lists each name with its signature;
  `tests/test_public_api.py` checks the lists and that the file is current.
- Top level gains `session_from_arrays`, `is_empty_csv`, `fit_pse_dynamics` and `ConfigError`.
- Private now: `_infer_animal_id`, `_contrast_key`, `_calculate_min_n`, `_validate_names`, `_producer_of`,
  `_list_producers` (no longer listed anywhere).
- Config:
  - An unknown key in any section `load_config` reads raises `ConfigError` (a `ValueError`) naming the
    section, the valid keys and the closest match; it used to be ignored. A top-level `custom:` block is
    reserved for the project and never read; `name` and `description` belong under `project:`.
  - ▲ `task.choice_mapping.no_response_value` is required whenever a choice column is mapped (`null` when no
    raw value means no response). The old default of 0 turned every A choice in a 0/1 column into a
    no-response; the shipped minimal preset did exactly that. `ChoiceMapping()` built in code now defaults
    to `None`.
  - `contingency_field` has no default (was `'sound_contingency'`) and is required for
    `spatial_to_category`; `category_rule` must be `above_boundary` or `below_boundary`.
  - Removed, never read: `task.inputs`, `task.outputs`, `task.n_categories`, `file_structure.animal_pattern`,
    `session_pattern`, `date_format`. The shipped presets moved `name`/`description` under `project:` (they
    were ignored at the top level) and lost `processed_subdir`, which was never read.
- Python ≥ 3.11; CI on 3.11 and 3.14. The version has one source, `behav_utils.__version__`.

### Step 2: reader hook and builder
- Loading is split. A reader finds sessions and reads their raw tables (`RawSession`); `build_session` maps a
  raw table onto `SessionData` the same way for every reader. Built in: `csv_folder_reader` (the folder layout)
  and `table_reader` (one table for all sessions: a DataFrame, CSV, TSV or Parquet). Any callable taking the
  config and yielding `RawSession` works, passed as `load_experiment(config, reader=...)`.
- Config: a `reader:` block with a `type` replaces `file_structure` (a leftover `file_structure` raises with the
  new name). `csv_folders` keeps the old keys and adds `merge_files`; `table` takes `path`, `animal_column`,
  `session_column`, `date_column` and an optional `start_column`. A new top-level `animal_metadata:` path.
  Relative paths resolve against the config file's folder, not the working directory.
- Several sessions on one day: `SessionData.start` (time of day) comes from `date_regex` groups `hour`,
  `minute`, `second` or the table's start column; sessions sort by date, start, then id; `merge_files: false`
  makes each file in a folder its own session. The session table gains a `start` column.
- Sessions in `sessions_to_ignore` are no longer read at all.
- ▲ Stricter loading: a file that cannot be parsed raises under `strict` (it was dropped silently in a
  multi-file session) and is skipped with a warning otherwise; `min_trials_per_file` applies to single-file
  sessions too, so a session with only a short file is skipped; the date comes from the reader only (no `Date`
  column fallback, no 1 Jan 2000 default), and a session without one raises under `strict`; a mapped
  `animal_id` column that disagrees with the reader raises under `strict` (warns and is overwritten
  otherwise). `metadata.animal_id` is now always set, to the reader's id.
- `load_session_csv(path, config, *, date=None)` loses `session_idx` and `df`. `parse_date_from_path` and
  `parse_date_from_csv` are gone; `behav_utils.data.readers.parse_session_datetime` reads a date and time.
- `EXTENDING.md`: adding a statistic, a readout with its plot, and a reader.

### Step 2b: where a session's type comes from
- ▲ A `session_types` list now overrides a type read from the CSV (it used to be the other way round), with a
  warning when the two differ; a session listed under two types takes the later, with a warning.
- A CSV column sets the type only when `session_type` is mapped under `session_metadata`; columns that happen to
  be named `session_type`, `Session_Type` or `SessionType` are no longer read on their own.
- `SessionData.session_type_source` (`'config'`, `'column'` or `'derived'`) replaces the private
  `_session_type_explicit`, and the session table gains `session_type_source` (and `start`, from step 2).
- A type read from the column that is neither `regular`, `opto` nor a name in `session_types` warns once per
  value; listing it in `session_types` with no dates declares it.

### Step 3: durable identities
- Session ids are `<animal>_<YYYYMMDD>`, plus `T<HHMM>` when the start time is known; sessions that would still
  share an id are numbered `_1`, `_2` in name order, with a warning. They are given before any session is
  ignored. The reader's own name moves to `SessionData.source_id`. Result tables that print session ids
  (rolling statistics, per-session rows) show the new ids.
- `TrialArrays.trial_key` is the session's 43-bit hash of experimenter, animal and session id above the trial
  number's 20 bits: the same on every load and machine, unaffected by which other sessions are loaded, and
  unique across animals. It used to be the session's position × 10⁶ + the trial number, so two animals
  shared keys and ignoring a session shifted the rest. `trial_key_of(session, trial_number)` takes the
  session (was its position); `get_arrays(trials, session=None)` likewise. `SessionData.session_hash` gives
  the upper part. Sessions whose hashes collide raise at load.
- Trial numbers must be unique within a session (repeats raise under `strict`, otherwise warn and are
  renumbered from 1) and below 1,048,576.
- `project.experimenter` (default empty), overridable per animal in the animal metadata; `SessionData.experimenter`.
- Entries in `sessions_to_ignore` and `session_types` are a date (every session that day) or a session id without
  the animal prefix (one session). A number that is not an 8-digit date is refused: an unquoted `20240315_2`
  reaches the config as 202403152.

### Step 4a: tasks and scoring
- New `behav_utils.tasks`: `Task` (category, truth, `complete_responses`, `score`, `to_dict`), `TwoAFC(b_side,
  on_boundary)`, `GoNoGo(go_side, on_boundary)`, `Outcome` codes (`ERROR` 0, `CORRECT` 1, `NO_RESPONSE` 2,
  `ABORT` 3, `UNSCORED` 4), `register_task`, `task_from_dict`, `list_tasks`. `Task`, `TwoAFC`, `GoNoGo`
  and `Outcome` are also top level.
- What counts as correct is decided by the session's task alone, built from the config. `task:` gains `type`
  (default `two_afc`) and the type's own settings, checked against it; `tasks:` holds named alternatives (the
  default with keys replaced) and `task_by: {field, values}` picks one per session by a metadata field;
  `SessionData.task` and `task_name` say which, and the session table gains `task`.
- `TrialData.outcome` holds the outcome codes and `TrialData.correct` is always `outcome == CORRECT`. A mapped
  `correct` or `outcome` column is kept as `correct_recorded` / `outcome_recorded` and checked against the
  score (`scoring: {on_mismatch: warn | raise | ignore, outcome_labels}`); it is never used instead of it.
  The shipped go/no-go example mapped `correct: Hit`, so every correct rejection was marked wrong; it now
  checks the recorded outcome through labels.
- Go/no-go: a blank response that is not an abort is a withhold (0), so no `mapping: {.nan: 0}` is needed.
- `category_rule` is replaced by `b_side` (two_afc) and `go_side` (go_nogo); a leftover key raises with the
  new names. ▲ A stimulus exactly on the boundary now follows `on_boundary` (default A) on either side; under
  `category_rule: below_boundary` it used to count as B. ▲ A NaN stimulus has category NaN (it was A).
- `TrialData.category` is float (0, 1 or NaN). `SessionSummary.accuracy`, and so `min_accuracy` presets, use
  the task's score and leave unscored trials out.
- `filter_trial_data` slices every field marked per-trial in `TrialData`, so new fields come along without a
  list to maintain.
- `session_from_arrays(..., task=None)` scores with the task (`TwoAFC()` by default); `categories` is now
  optional, and if given must agree with the task's.
- The golden fixture's recorded `Correct`/`Outcome` for one trial (stimulus rounded to exactly 0) came from the
  unrounded stimulus; the new check flagged it, and the row and `make_fixture.py` are corrected. No pinned
  statistic reads those columns, so no golden value changes.
- `docs/public_api.md` lists an enum by its members: its call signature prints differently across Python
  versions.

### Step 4b: statistics read the task's score
- `TrialArrays` carries the task's score: `correct` and `prev_correct` (1 correct, 0 error, NaN where the task
  gives none) and the `task`. They travel through `take`, `valid`, `lag1_pairs` and `concat`; built by hand,
  they are derived from the task. `TrialArrays.prev_reward` is removed (reward is what a rig delivers; this is
  the score).
- Every statistic and readout that needs correctness reads it: `accuracy`, `hard_accuracy`, `easy_accuracy`,
  `hard_easy_ratio`, `accuracy_b_minus_a`, `win_stay`, `lose_shift`, `win_stay_lose_shift`, the logistic
  history (an unscored lag's outcome term is 0), the update matrix, the serial-dependence profile and the
  binned accuracy curve. Trials without a score are left out. Every 2-AFC number is unchanged (golden values).
- `fit_update_matrix(stimuli, choices, *, correct, ..., prev_correct=None)`: the engine is given the score
  instead of `categories` / `prev_categories`, keyword-only so an old call fails rather than reading
  categories as scores.
- `pool_arrays` returns the sessions' `task` and refuses sessions of different tasks; `TrialArrays.concat`
  likewise.
- `TrialArrays.from_sequence(choice, stimulus, category=None, reaction_time=None, *, task=None)`: the task
  decides categories and the score (a go/no-go blank is a withhold); given categories are only checked.
- ▲ `TrialData.prev_correct` is NaN after a predecessor without a score (it was 0 after a no-response).
- `compute_rolling_stats` no longer passes empty frames to `pd.concat` (a pandas FutureWarning).

### Step 5: event times and the clock contract
- A column mapped under a name starting with `t_` is an event time, stored in the new `TrialData.events` (by
  name) and kept out of `optional_fields` and `extra`. Event times and `reaction_time` are converted to float64
  seconds and stay on the recording clock (never re-zeroed). An optional event missing from a file is NaN; a
  required one fails the session.
- New `clock:` block (`ClockConfig`): `name` (default `rig`, carried as `SessionData.clock`), `unit` (`s`, `ms`,
  `us`), `on_problem` (`warn`, `raise`, `ignore`) and an optional within-trial `order`. A column may set its own
  `unit`; only `t_*` columns and `reaction_time` take one, and event columns must be numeric.
- Each session is checked for an event column going backwards across trials, negative reaction times and trials
  out of `clock.order`, one warning (or error) per kind of problem.
- `filter_trial_data` slices every value of the per-trial maps (`optional_fields`, `extra`, `events`), marked in
  `TrialData`, so they follow the trials through filtering and downsampling.
- `behav_utils.config` also exports `ClockConfig`, `ScoringConfig` and `TaskSelector`.

### Step 6a: models and simulate()
- Stimulus distributions in `behav_utils.tasks`: `StimulusDistribution` (`sample`, `pdf`, `support`, `grid`),
  `Uniform`, `TruncatedNormal`, `Discrete`.
- New `behav_utils.models`: `Model` (anything with `simulate(design, rng) -> Responses`), `TrialwiseModel` (the
  trial loop for learning models: `p_choice`, `update`, `reset`), `Responses`, `Design` and `make_design` (task,
  stimuli, label, and laser, abort and no-response flags drawn from rates, with an optional cap on laser runs).
- Built-in models: `PsychometricModel` (the curve `fit_psychometric` fits, so simulating and fitting recovers
  the parameters), `RandomModel`, `IdealObserver` (Bayesian, with its own prior), and `normative_pse`. Each
  takes `laser={...}` overrides for laser trials.
- `simulate(model, design, rng)` returns a `SessionData` scored by the design's task, aborted and no-response
  trials blanked and laser flags in `trials.opto_on`; `simulate_responses` returns only the choices.
  `simulate`, `make_design` and the built-in models are also top level.
- `session_from_arrays(..., laser=None)` sets `trials.opto_on`.

### Step 6b: synthetic data from any model
- `generate_synthetic_session`, `generate_synthetic_animal` and the new `generate_synthetic_experiment` live in
  `behav_utils.models` (also top level) and take any model (`PsychometricModel()` by default). Every trial class
  is generated: responses, no-responses (0.02 by default where the task has them, none for go/no-go), aborts
  (0.05), and with `laser_fraction` laser trials and the trials after them. `model`, `distribution`, `label` and
  `trials_per_session` take one value or one per session; `metadata` goes to the animal; each experiment animal
  has its own random stream, so adding one leaves the others unchanged. One `rng` argument (a seed or a
  Generator) replaces `seed`; `generate_synthetic_animal` returns an `AnimalData` (it returned `(animal, info)`).
- Event times `t_trial_start`, `t_stim_on`, `t_response` on a clock named `synthetic` that counts seconds from
  midnight (`Timing`: inter-trial interval, stimulus delay, lognormal reaction times unless the model gives its
  own); a withhold has no response time.
- `session_from_arrays` gains `events`, `clock` and `session_date`, uses the loader's session ids
  (`<animal>_<YYYYMMDD>`, was `<animal>_S000`) and its session-type rule ('opto' with any laser trial).
- Removed: `sample_stimuli`, `generate_stimuli`, `random_choice_simulator`, `noisy_psychometric_simulator` and the
  old simulator-based generators. A simulator of the old shape becomes a model with `simulate(design, rng)`.

### Step 7: requirements on statistics and readouts
- Tasks declare `capabilities` (`TwoAFC`: `binary`, `two_choices`, `stimulus_axis`, `scored`; `GoNoGo`: `binary`,
  `withhold`, `stimulus_axis`, `scored`); the names are open, so a task of your own can add its own.
  `behav_utils.tasks` gains `CAPABILITIES`, `RequirementError` (a `ValueError`) and `require`.
- `@stat` and `@fit` take `requires=` (none: every task) and `instead={capability: counterpart}`.
  `compute_stats` refuses an explicitly requested statistic the arrays' task cannot support, naming what is
  missing, what the task has and the counterpart (or that there is none), and flagging a required name no task
  declares. `compute_stats(arrays)` (no names) and `list_stats(task=...)` give only what applies.
- On go/no-go: `side_bias` and `accuracy_b_minus_a` are refused in favour of the new `go_bias` and
  `hit_minus_cr_rate`; `logistic_history` and `history_interaction_r2` are refused (no counterpart); every other
  statistic applies. Readouts call `require` first (all built-ins need `binary` and `stimulus_axis`).
- The stats reference lists each statistic's needs and the statistics each built-in task refuses, replacing
  the go/no-go reading table.

### Step 8: session discovery and a shared session table
- `discover_sessions(config, *, reader=None, peek=True, laser=False)` lists every session a reader finds without
  loading trials: identity columns, `ignored`, the task, the session type where decidable (all with `laser=True`),
  the peeked `session_metadata` fields and any `file_<name>` columns. Ids are the ones loading gives (both use
  one id step).
- `load_experiment(config, sessions=...)` reads only the given sessions (a table with `session_id`, or ids); an
  unknown id raises under strict loading and warns otherwise.
- `RawSession` gains `files` (named related files) and `peek(columns, nrows)` (a cheap look under the reader's row
  rules); the CSV-folder and table readers implement `peek`.
- `animal.session_table` starts with the identity columns `session_id`, `animal_id`, `experimenter`, `date`,
  `start`, `source_id`, `source` (▲ column order); new `experiment.session_table` stacks every animal's.

## 0.10.0 — 2026-11-07

### Added
- `accuracy_b_minus_a`: accuracy on category-B trials minus accuracy on category-A trials. Unlike
  `side_bias`, it does not move with how often each category is presented (bias-correcting trial
  selection, asymmetric stimulus distributions); it does move when the categories differ in difficulty.
  No existing number changes.

### Docs
- Stats reference regenerated. Its go/no-go table gives `side_bias` as P(choose B) − 0.5 (it said
  P(choose B)) and reads the new statistic as hit rate minus correct-rejection rate.

## 0.9.1 — 2026-11-04

### Fixed
- `load_animal` skips empty CSV files (nothing in them, not even a header: a run stopped before writing
  anything) with a warning, as if absent; a session with only empty files is skipped. In a session with
  several files an empty one raised `EmptyDataError` and, under `strict=True`, stopped the whole load.
  `is_empty_csv(path)` is the check, for projects that re-read the CSVs.

## 0.9.0 — 2026-11-02

Number-affecting changes are marked ▲.

### Changed ▲
- `select_sessions` / `SessionFilter`: `exclude_opto` and `exclude_types` now apply before `first_n`, `last_n`,
  `last_fraction` and `last_n_qualifying`. Before, excluded sessions still took up positions, so `last_n=5`
  could return fewer than 5.

### Changed
- `compute_delta_stat` raises if a permutation is requested and a phase shares trials with the reference
  (e.g. `post_opto` inside `non_opto`); without a permutation it warns. `permute_phase_difference` raises too.

### Added
- `shared_trial_count(phase_a, phase_b)`: how many trials two phases share.

### Fixed
- `pool_arrays([])` returns `session_boundaries == [0]` (was `[]`).
- CI ran the stats-reference check twice.
- Tests: the comparison tests gave every phase the same trial numbers; each phase now has its own.

### Docs
- Install and test commands without inline comments; LLM_CONTEXT notes the selection order and the
  shared-trial guard.

## 0.8.0 — 2026-10-31

Handover release. Number-affecting defaults are marked ▲.

### Fixed
- `infer_animal_id` never found the id the loader writes (`session.metadata.animal_id`), so
  `PhaseStats.animal` was `'unknown'` and `collect_rows` let that overwrite the explicit `animal=`:
  every animal collapsed into one unit in `compare_groups` (the example notebook's group section
  reported n = 1 vs 1). Both fixed; `collect_rows(animal=...)` now always wins and warns on a conflict.
- `paired_diff` dropped every pooled row (`distribution=None` is a pivot index key; pandas' default
  `dropna` discarded it). `dropna=False`.
- `compare_groups(paired=True)` dropped non-finite values per group independently, so pairs could
  misalign silently; values are now aligned by `unit_col` and disjoint animal sets raise.
- `filter_trials`: `trial_type` silently discarded a passed `mask_fn` (now raises); `trial_type` with
  `exclude_abort=False` kept aborts (now every `trial_type` drops aborts).
- Docs: `data_structures_reference.md` rewritten against the real API (`compute_delta_stat` kwargs,
  non-existent `get_animals` / `trials.get_arrays` / `'trial_indices'`, `get_arrays` "excludes aborts",
  `opto_mask('control')` adjacency) and executed by `docs/run_doc_examples.py`; stale module docstrings
  in `structures`, `selection`, `filtering`, `group`, `config`, `data.ops`; README install pin; the
  go/no-go and orientation example configs (dead `analysis:` section, raw-unit stimuli).

### Changed ▲
- `compute_delta_stat(n_permutations=0)` by default; `n_permutations > 0` raises unless every phase
  shares a session with the reference. Calls that passed `n_permutations` explicitly on within-session
  contrasts are unchanged; calls that relied on the default lose a p they should not have had.
- `rank_test -> RankTest`, `bootstrap_units -> UnitBootstrap`, `average_arrays -> ArrayAverage`,
  `compare_groups -> GroupComparison` (`.test(stat)`, `.table()`): frozen dataclasses instead of dicts
  (rule 2). Values identical; access is by attribute.
- `SessionData.summary() -> SessionSummary` (`.accuracy`, formerly `['perf']`).
- `AnimalData.session_table` is counts + accuracy only (`SESSION_TABLE_COLUMNS`); it no longer runs
  every registered statistic (including the `pse_dynamics` curve fit) on every session, which cost
  ~0.7 s per session and was a data→stats import.
- Plotting: `plot_stat_comparison` / `plot_interaction` are now the single-panel plotters (formerly
  `*_single`); the grids are `plot_stat_comparison_grid` / `plot_interaction_grid`, the one documented
  exception to "layout is the caller's".
- Loading: `print` → `logging` (`behav_utils.data.loading`); `load_animal` / `load_experiment` take
  `strict=True` (a broken session raises instead of being skipped with a warning).
- Session presets resolve on the animal's own config (`ExperimentData.config.session_presets`) before
  the process-wide registry; `preset_from_spec` exported.
- `SessionMetadata` loses the project-vocabulary properties `sound_contingency`, `stim_range_min/max`
  (`.get()` still works).

### Added
- Stimulus normalisation at load: `task.stimulus_scale` (`linear` | `log`) alongside `boundary` and
  `stimulus_range`; `TaskConfig.normalise_stimulus` maps raw units onto `[-1, 1]` with the boundary at
  0 (identity for the defaults). `TrialData.stimulus_raw` keeps the CSV value; `pool_arrays` adds
  `stimuli_raw`. `stimulus_range` was parsed and never used before: configs in Hz or degrees produced
  silently wrong bins.
- `TrialArrays.trial_key` (int64 `session_idx * 10**6 + trial_number`), carried through `take`,
  `valid`, `lag1_pairs`, `concat` and every resample; `trial_key_of`, `with_trial_key`,
  `TrialArrays.concat`. The hook for aligning neural (or any per-trial) data.
- Go/no-go support documented: the two config differences (`docs/config_guide.md`), a table of what
  each statistic measures (`docs/stats_reference.md`), the loader's NaN-choice warning.
- `plot_rolling_stats(result, stat, ax=None)` for `RollingStats`.
- `ExperimentData.save` stamps `behav_utils_version` / `config_name`; `load` warns on a version mismatch.
- `docs/run_doc_examples.py` (run in CI).

### Removed
- `analysis/session_features.py` (`compute_session_features`): a dict result, mislabelled
  `n_trials_abort` / `abort_rate` (they counted no-response), duplicating `compute_phase_stats(per_session=True)`
  and `stats/rt.py`.
- The second copy of the stimulus-binning helpers (`stats/_binning.py` now re-exports `readouts/_base.py`).

## 0.7.1 — 2026-10
- `SessionFilter.last_n_qualifying`: the last N sessions that pass the quality filters
  (`min_accuracy`, `max_accuracy`, `min_trials`). `last_n` is unchanged and still applied before
  them, so "last 5 sessions with accuracy > 0.7" is now expressible as a preset.
- No numbers change.

## 0.7.0 — 2026-10
- `analysis/` has one job per file: `statistics.py` → `phase.py`; `downsample.py` merged into
  `resampling.py` (draw engine + draw summaries); `across_animals.py` merged into `group.py`;
  `psychometry.py`, `update_matrix.py` and `utils.py` → `readouts/fit.py` (the two fit engines and
  `cumulative_gaussian`), `generate_stimuli` → `data/synthetic.py`. Every public name is unchanged and
  still exported from `behav_utils` and `behav_utils.analysis`; only the submodule paths moved.
- `docs/config_guide.md` keeps the reference; the three complete example configs are files under
  `docs/config_examples/`.
- No numbers change.

## 0.6.0 — 2026-10
- New fitter `sdt` → `dprime`, `criterion` (B as signal, log-linear correction; exchangeable).
- The library lives in its own repository again (github.com/serkanshentyurk/behav_utils) with its own CI
  (`ruff`, the suite on 3.10 and 3.12, the example notebook, the generated stats reference) and a `v*` tag
  release workflow; packaging metadata (classifiers, keywords, URLs).

## 0.5.0 — 2026-10
- `compute_stat` → `compute_phase_stats` (one name per level: `compute_stats` on arrays,
  `compute_phase_stats` on sessions). No alias.
- `apply_session_type` is public (was `_apply_session_type`).
- Removed, no consumer: `analysis/session_raster.py`, `plotting/session.py`,
  `plotting/session_stats.py`, `compare_genotypes`, `AnimalData.genotype` (read
  `animal.metadata['genotype']`), `AnalysisConfig`/`PlottingConfig` (the `analysis:` /
  `plotting:` config sections were parsed and never read), the legacy
  `*_sessions` config keys, the experiment-specific config presets, the empty example
  notebook.
- `compare_groups` no longer orders `('wt', 'het')` first; pass `groups=(a, b)` to fix the
  reference group.
- Docstrings and docs use neutral vocabulary; `docs/data_structures_reference.md` examples
  rewritten against the 0.3+ API; `docs/config_guide.md` now documents `cohorts`, `session_types`,
  `session_presets` and `sessions_to_ignore` (the removed `analysis:`/`plotting:` sections are gone);
  `docs/stats_reference.md` is generated from the registry by `docs/gen_stats_reference.py`.
- The shipped presets (`config_minimal.yaml`, `config_full_reference.yaml`) are experiment-neutral.
- `generate_synthetic_animal` / `generate_synthetic_session` default `stage` is `'task'` (was a
  project stage name); pass `stage=` explicitly if you select by stage afterwards.

## 0.4.0 — 2026-09
- Library is task-agnostic: `session_types` and `session_presets` come from the config; hardcoded
  presets, `masking`/`washout`/ALM vocabulary and the Hard-A/B adaptation module removed.
  `SessionFilter.exclude_types` replaces the three exclusion flags. `SessionData.masking/.washout` removed.
- `find_switches` (generic distribution-switch detection).
- `pse_dynamics` fit producer (exponential vs step PSE trajectory), optional pinned shape.
- `UpdateMatrix.from_matrix`.
- `filter_trials(trial_type='all')` drops abort trials (bug fix; docstring and code disagreed).
- src/ layout; own test suite in `tests/`.

## 0.3.0 — 2026-09
- Typed stat registry: `TrialArrays` + `compute_stats -> pd.Series`; `@stat`/`@fit`; no family names.
- Readout dataclasses: `PsychometricCurve`, `UpdateMatrix`, `ConditionalPsychometric`, `BinnedCurve`,
  `SerialDependenceProfile`, each with one draw-only plotter.
- `compute_phase_stats -> PhaseStats`, `compute_delta_stat -> DeltaStats`, `compute_interaction -> Interaction`,
  `compute_rolling_stats -> RollingStats`; resampling engines return DataFrames of draws.
- Removed: `summary_stats`, `stats_table`, `trajectory`, `compute_psychometric`, `compute_um`,
  `average_um`, `plot_um`, `plot_psychometric`, `compare_phases`.
- Bit-exact with 0.2 for every scalar statistic and every bootstrap draw (verified before deletion).

## 0.2.0 — 2026-07
- Baseline before the refactor.
