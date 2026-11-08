# Changelog

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
