# LLM_CONTEXT — behav_utils

Read this first if you are an AI assistant (or a new person) asked to work on `behav_utils`. It says what
the library is, the rules that are not negotiable, where things live, and the mistakes people make.

## What it is
A Python ≥ 3.10 library for 2-AFC rodent behaviour: CSV → structures → session selection → trial
filtering → statistics / readouts → contrasts with uncertainty → plots. Task-agnostic: it reads a YAML
config and knows nothing about any experiment's distributions, session types or presets. It is pip
installable from `src/` layout; the package is `behav_utils`, the tests are in `behav_utils/tests/`.

## Non-negotiable rules
1. Pipeline order: `load → select_sessions → filter_trials → compute_x → plot_x`. A compute never selects
   or filters. A plot never computes.
2. Every compute returns a typed result: `pd.Series` / `DataFrame`, or a frozen dataclass with
   `to_rows()`. Never a dict whose shape depends on an argument. If you need a new result, add a dataclass.
3. Statistics are floats. A statistic is `f(arrays: TrialArrays, *, rng) -> float` registered with
   `@stat`; multi-output fits use `@fit(outputs=...)` and expose each output as its own name. No family
   names, no dict-valued stats. Array-valued things are readouts in `behav_utils.readouts`.
4. `exchangeable=False` on any statistic that depends on trial order beyond lag 1. Resamplers enforce it.
5. Project vocabulary stays out. If a change needs the words `masking`, `Hard-A`, `expert_uniform`,
   `genotype` or a normative observer, it belongs in the project package, not here.
6. Dependencies point downward: `plotting → analysis → stats/readouts → data → config`.
7. Deliver complete files, not snippets. Run the tests. British English.

## Where things are
```
src/behav_utils/
  config/schema.py, loader          YAML → ProjectConfig; session_presets, session_types
  data/structures.py                TrialData, SessionData, AnimalData, ExperimentData (lag-1 view frozen at load)
  data/loading.py                   CSV loading, session-type stamping, preset registration
  data/ops/selection.py             SessionFilter, select_sessions, presets (exclude_types, session_type)
  data/ops/filtering.py             filter_trials (trial_type), build_mask, pool_arrays
  data/ops/switches.py              find_switches
  data/arrays.py                    TrialArrays (the one input type for stats/readouts)
  data/synthetic.py                 generate_synthetic_animal / session, simulators
  stats/registry.py                 @stat, @fit, compute_stats, list_stats, is_exchangeable
  stats/basic.py history.py psychometric.py rt.py dynamics.py    the producers
  readouts/                         UpdateMatrix, PsychometricCurve, ConditionalPsychometric, BinnedCurve, SerialDependenceProfile
  analysis/statistics.py            compute_phase_stats -> PhaseStats
  analysis/comparison.py            compute_delta_stat -> DeltaStats; compute_interaction -> Interaction
  analysis/resampling.py            bootstrap_phase_stats, permute_phase_difference, summarise_draws
  analysis/downsample.py            the single resampling engine (draws), resample_* readouts
  analysis/rolling.py               compute_rolling_stats -> RollingStats
  analysis/across_animals.py group.py   collect_rows, compare_groups, rank_test, paired_diff
  analysis/psychometry.py update_matrix.py   fit engines on raw arrays
  plotting/                         one plot_x per result type; styles.py
docs/                               config_guide, data_structures_reference, stats_reference
tests/                              conftest (synthetic fixtures, test-local presets) + one file per module
```

## How to do common things
- Add a statistic: write `f(arrays, *, rng) -> float` in the right `stats/*.py`, decorate with
  `@stat('name', exchangeable=...)`, add a test in `tests/test_stats_registry.py`, regenerate
  `docs/stats_reference.md` (see CONTRIBUTING). It is then available everywhere by name.
- Add a readout: dataclass + `compute_x(arrays, ...)` in `readouts/`, one `plot_x` in
  `plotting/readouts.py`, tests in `tests/test_readouts.py`.
- Add a session type: nothing in the library — projects put it in `config.yaml: session_types`.
- Change how draws are made: `analysis/downsample.py` only, and expect every downstream interval to change.

## Traps
- `filter_trials(trial_type='all')` keeps laser trials but drops aborts (all trial types drop aborts).
- `compute_phase_stats(...).pooled` is a real pooled fit; never average per-session values to get it.
- Permutation p is only valid when the label was randomised per trial. Session-level conditions get
  bootstrap intervals only.
- With few animals the rank test has a hard floor (`min_achievable_p`); per-animal consistency is the
  evidence, the p is not.
- A `PsychometricCurve` with `n_bootstrap=1000` is slow; pass `n_bootstrap=0` inside loops.
- `pse_dynamics` (τ of a within-session PSE trajectory) is loosely identified below ~1000 trials at
  mouse-like sigma; report with session-unit CIs across animals.

## Testing and style
`pytest tests -q` from `behav_utils/`; `ruff check .` (config in `pyproject.toml`). Line length 110.
