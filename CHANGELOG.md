# Changelog

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
