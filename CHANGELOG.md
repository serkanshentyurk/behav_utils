# Changelog

Plans for later releases are in [ROADMAP.md](ROADMAP.md).

## 1.0.1 — 2026-11-13

Documentation only; the library behaves exactly as in 1.0.0.

- A documentation site, built with Sphinx and published at <https://serkanshentyurk.github.io/behav_utils/>,
  one version per minor release: the 2-AFC tutorial (now `docs/tutorials/two_afc.ipynb`), the API by
  sub-package from the docstrings, the statistics, configuration and data-structures references, and the
  contributor pages.
- Every example on the site runs in CI: each page's Python blocks top to bottom, and its YAML blocks as configs.
- Docstrings reformatted so the API pages render, with the code in them moved to the pages. Three were wrong:
  `get_arrays` and `pool_arrays` did not list every key they return, and `fit_psychometric_gof` showed R² garbled.
- The config guide's *Validation* section uses public functions only.

## 1.0.0 — 2026-11-12

The first stable release: from here, public names change only in a major release, after a minor release in which
they warn (CONTRIBUTING.md). behav_utils is now task- and model-agnostic: readers bring anyone's trial data in, the
session's task alone decides what is correct, any model simulates through the same pipeline, and every statistic,
readout and model says what a task must provide. Every 2-AFC statistic, readout, contrast and rolling statistic on
the golden dataset is unchanged from 0.10.0 except where marked ▲.

### Breaking changes from 0.10

Config
- An unknown key in any section raises `ConfigError` (a `ValueError`) naming the valid keys and the closest match;
  it used to be ignored. A top-level `custom:` block is reserved for the project; `name` and `description` belong
  under `project:`. Keys that were never read are gone (`task.inputs`, `task.outputs`, `task.n_categories`,
  `file_structure.animal_pattern`, `session_pattern`, `date_format`).
- ▲ `task.choice_mapping.no_response_value` is required whenever a choice column is mapped (`null` when no raw
  value means no response). The old default of 0 turned every A choice in a 0/1 column into a no-response.
- `reader:` (with a `type`) replaces `file_structure`; `contingency_field` has no default; `category_rule` is
  replaced by `b_side` (two_afc) and `go_side` (go_nogo). Leftover keys raise with the new names. Relative paths
  resolve against the config file's folder.

Loading and identities
- ▲ Stricter loading: an unparseable file raises under `strict` (it was dropped silently in a multi-file
  session); `min_trials_per_file` applies to single-file sessions too; the date comes from the reader only (no
  `Date`-column fallback, no 1 Jan 2000 default); a mapped `animal_id` that disagrees with the reader raises.
  Sessions in `sessions_to_ignore` are not read at all.
- ▲ A `session_types` list overrides a recorded type (it was the other way round); a column sets the type only
  when `session_type` is mapped.
- Session ids are `<animal>_<YYYYMMDD>`, plus `T<HHMM>` with a start time and `_1`, `_2` where they would clash;
  the reader's own name is `SessionData.source_id`. `TrialArrays.trial_key` is a hash of experimenter, animal and
  session above the trial number: stable across loads and unique across animals (it was the session's position
  × 10⁶ + the trial number). Trial numbers must be unique within a session and below 1,048,576.
- `animal.session_table` starts with the identity columns (`session_id`, `animal_id`, `experimenter`, `date`,
  `start`, `source_id`, `source`). `SessionData.csv_path` is renamed `source`.
- `load_session_csv(path, config, *, date=None)` loses `session_idx` and `df`; `parse_date_from_path` and
  `parse_date_from_csv` are gone.

Scoring
- ▲ The session's task decides what is correct. `TrialData.outcome` holds `Outcome` codes and `TrialData.correct`
  is always `outcome == CORRECT`; a mapped `correct` or `outcome` column is kept as `correct_recorded` /
  `outcome_recorded` and checked, never used instead.
- ▲ A stimulus exactly on the boundary follows `on_boundary` (default A) on either side (under `category_rule:
  below_boundary` it counted as B); a NaN stimulus has category NaN (it was A); `TrialData.prev_correct` is NaN
  after a predecessor without a score (it was 0 after a no-response).
- `TrialArrays.prev_reward` is removed (use `prev_correct`). `fit_update_matrix` takes the score as the keyword
  `correct=` (and `prev_correct=`) instead of `categories`.
- On go/no-go, statistics that need two active choices are refused: `side_bias` → `go_bias`, `accuracy_b_minus_a`
  → `hit_minus_cr_rate`; `logistic_history` and `history_interaction_r2` have no go/no-go version.

Models and synthetic data
- The simulator functions (`sample_stimuli`, `generate_stimuli`, `random_choice_simulator`,
  `noisy_psychometric_simulator`) and the old generators are gone: a simulator becomes a model with
  `simulate(design, rng)`. `generate_synthetic_animal` lives in `behav_utils.models`, takes `rng` instead of
  `seed` and returns an `AnimalData` (it returned `(animal, info)`). `session_from_arrays` gives the loader's
  session ids.

Saving
- ▲ Pickle is gone. `ExperimentData.save` / `.load` and `AnimalData.save` / `.load` write and read a store (a
  directory of Parquet and JSON); a `.pkl` path raises with directions. `save` no longer stamps the library
  version into the experiment's metadata (the store records it).

Public API
- The top level holds the 71 everyday names; everything else imports from its package (for example
  `from behav_utils.stats import PSYCHOMETRIC`, `from behav_utils.data import session_from_arrays, trial_key_of`,
  `from behav_utils.models import normative_pse`). `docs/public_api.md` lists every name and its home.
- Renamed: `plot_comparison` → `plot_psychometric_comparison`. Private now: `apply_session_type` and six helpers
  (`_infer_animal_id`, `_contrast_key`, `_calculate_min_n`, `_validate_names`, `_producer_of`, `_list_producers`).
  Removed: `readouts.PARAMS` (it was `stats.PSYCHOMETRIC`).
- `plot_psychometric_comparison`, `plot_stat_comparison` and `plot_interaction` return `(fig, ax)` like every
  single-panel plot (they returned `ax`).
- Python ≥ 3.11.

### Added

Data and loading
- Readers: `csv_folder_reader` (several sessions a day; `merge_files`), `table_reader` (one table: a DataFrame,
  CSV, TSV or Parquet), or any function yielding `RawSession` (with optional `files` and a cheap `peek`);
  `build_session` maps any of them onto `SessionData`.
- `discover_sessions(config)`: every session before loading, with the ids loading gives;
  `load_experiment(config, sessions=...)` reads only those chosen. `experiment.session_table` stacks every animal's.
- `project.experimenter` and per-animal metadata (`animal_metadata:`); `SessionData.session_type_source`.
- Event times: columns named `t_*`, in seconds on one recording clock (`clock:` block, `ClockConfig`), stored in
  `TrialData.events`, checked at load (times going backwards, negative reaction times, within-trial order).

Tasks
- `behav_utils.tasks`: `Task`, `TwoAFC`, `GoNoGo`, `Reproduction`, `Outcome`, `ResponseSpace`, a registry
  (`register_task`, `task_from_dict`, `list_tasks`), stimulus distributions (`Uniform`, `TruncatedNormal`,
  `Discrete`, `StimulusDistribution`) and capabilities (`CAPABILITIES`, `require`, `RequirementError`).
- Config: `task: {type, ...}` with the type's own settings, `tasks:` alternatives picked per session by `task_by:`,
  and `scoring:` (what happens when a recorded column disagrees).
- The continuous-response family: `Reproduction(tolerance)`; stimulus and response share one linear map onto
  [-1, 1]; `TrialArrays.truth`.

Statistics and readouts
- `@stat` / `@fit` take `requires=` and `instead=`; `compute_stats(arrays)` and `list_stats(task=...)` give what
  applies. `TrialArrays` carries `correct`, `prev_correct`, `truth` and `task`.
- Go/no-go: `go_bias`, `hit_minus_cr_rate`. Continuous: `response_bias`, `error_sd`, `mean_abs_error`,
  `response_slope`, `serial_bias`, and the readouts `compute_response_curve`, `compute_error_distribution`,
  `compute_serial_bias` with their plots.

Models
- `behav_utils.models`: `Model`, `TrialwiseModel` (hooks take a `TrialInfo`; `p_choice` or `respond`, `update`),
  `Responses`, `Design`, `make_design`, `simulate`, `simulate_responses`; built-ins `PsychometricModel`,
  `RandomModel`, `IdealObserver` (both families), `ReproductionModel`, with `laser={...}` overrides; `normative_pse`,
  `normative_slope`.
- Synthetic data from any model with every trial class and event times: `generate_synthetic_session`,
  `generate_synthetic_animal`, `generate_synthetic_experiment`, `Timing`.

Saving
- The store: `save_store`, `read_manifest`, `load_store`, `StoreError`. Its manifest is the session table, so it is
  filtered like discovery; writes are atomic; partial loads open only the chosen sessions' files; the schema
  version is checked. Needs pyarrow (`pip install 'behav_utils[store]'`).

Tooling and documentation
- Golden values on two frozen fixtures (2-AFC from 0.10.0; continuous from this release), exact on the
  reference machine and with set tolerances across library versions; a generated, tested
  `docs/public_api.md`; CI on Python 3.11 and 3.14 with real Parquet; `EXTENDING.md` (a statistic, a readout, a
  task, a statistic for your own task, a model, a reader); the doc-example runner skips blocks marked
  `# requires: <module>` where it is missing.

### Fixed
- The shipped minimal preset mapped no response to 0 in a 0/1 column, turning every A choice into a no-response.
- The shipped go/no-go example mapped `correct: Hit`, marking every correct rejection wrong.
- Under pandas 3, loaded text columns were pandas `StringArray`s, which filtering did not slice.
- `compute_rolling_stats` passed empty frames to `pd.concat` (a pandas FutureWarning).

### Deferred
- 1.1: streams (pupil, imaging, spikes aligned to trials by trial key and the shared clock), with companion files
  linked to sessions.
- 1.x: the K-choice family; per-trial context from loaded columns; stimuli with several features; reaction time
  derived from two events; event times in `TrialArrays`; clocks that wrap at midnight; adding sessions to an existing
  store; NWB export.

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
