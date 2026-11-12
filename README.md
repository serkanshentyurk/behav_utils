# behav_utils

Analysis library for trial-based behavioural experiments. It loads trials from your files through a YAML config,
scores each trial by its task (two-alternative forced choice, go/no-go, reproducing a value, or a task of your
own), computes statistics and readouts with resampling-based uncertainty, compares phases and groups, simulates any
behavioural model through the same pipeline, and saves experiments in an open format.

It knows nothing about any particular experiment: stimulus units, tasks, distributions, session types and presets
all come from the config you give it. Every computation runs on one stimulus axis, [-1, 1], that the loader maps
your raw stimulus onto (the category boundary, or for continuous tasks the range's midpoint, at 0).

## What it assumes

- One row per trial. The default reader takes CSV files laid out as `data_dir/<animal>/<session>/<file>.csv`
  (several files in a session folder merged in name order; the date, and optionally the start time, from the
  folder name). One table holding every session, or a reader of your own, also work.
- Session metadata (stage, ...) is taken from each file's first row.
- One scalar stimulus per trial.
- One response per trial: 0 or 1 in binary tasks (1 for category B, or go), a value on the stimulus scale in
  continuous ones, NaN for no response. The config names the raw value that means no response.
- What counts as correct is the session's task, set in the config; recorded correct or outcome columns are only
  checked against it.
- Optional per-trial columns: reaction time, abort, laser (opto), distribution label, event times (`t_*`, in
  seconds on one recording clock per session), and any others you map or keep.

## Install

A pinned release:

```bash
pip install "behav_utils @ git+https://github.com/serkanshentyurk/behav_utils.git@v1.0.0"
```

Saving and reloading experiments uses Parquet, which needs pyarrow, the `store` extra:

```bash
pip install "behav_utils[store] @ git+https://github.com/serkanshentyurk/behav_utils.git@v1.0.0"
```

To work on the library, an editable checkout with the development tools: `pip install -e "<path to behav_utils>[dev]"`.
Requires Python ≥ 3.11; numpy, pandas, scipy, matplotlib, pyyaml.

## A five-minute tour

No data needed: a synthetic animal stands in for `load_experiment('config.yaml')`.

```python
from behav_utils import (PsychometricModel, TrialArrays, compute_delta_stat, compute_psychometric_curve,
                         compute_stats, filter_trials, generate_synthetic_animal, plot_psychometric_curve,
                         select_sessions)

# Ten sessions of a psychometric observer whose slope flattens on laser trials
animal = generate_synthetic_animal(PsychometricModel(sigma=0.25, laser={'sigma': 0.45}), n_sessions=10,
                                   laser_fraction=0.3, rng=0)

sessions = select_sessions(animal, min_trials=0)               # which sessions (presets come from the config)
off = filter_trials(sessions, trial_type='non_opto')           # which trials: aborts and laser trials out
on = filter_trials(sessions, trial_type='opto')

arrays = TrialArrays.from_sessions(off)                        # one aligned bundle, scored by the task
compute_stats(arrays, ['accuracy', 'mu', 'sigma', 'win_stay'])  # a pd.Series

curve = compute_psychometric_curve(arrays)                     # fit and bootstrap band
plot_psychometric_curve(curve)                                 # draw-only, one panel

d = compute_delta_stat({'off': off, 'on': on}, ['sigma', 'accuracy'], reference='off', n_permutations=1000)
d.contrast('on').table()                                       # Δ, bootstrap CI and permutation p per statistic

animal.save('animal_store')                                    # a store: Parquet and JSON (needs pyarrow)
```

## Task families

| family | built-in tasks | statistics | readouts | models |
|---|---|---|---|---|
| binary | `TwoAFC`, `GoNoGo` | psychometric fit, accuracy, history, SDT, reaction times, ... | psychometric curve, update matrix, conditional psychometric, binned curve, serial-dependence profile | `PsychometricModel`, `RandomModel`, `IdealObserver` |
| continuous | `Reproduction` | bias, error spread, mean absolute error, slope, serial bias | response curve, error distribution, serial-bias curve | `ReproductionModel`, `IdealObserver` |

Statistics, readouts and models declare what a task must provide; asking for one a task cannot support raises an
error naming the counterpart, if there is one ([docs/stats_reference.md](docs/stats_reference.md)). A task, a
statistic, a readout, a model or a reader of your own plugs in without changing the library
([EXTENDING.md](EXTENDING.md)).

## What's in it

| package | what | everyday names (top level) |
|---|---|---|
| `behav_utils.config` | YAML schema and loader | `load_config`, `ProjectConfig`, `ConfigError` |
| `behav_utils.tasks` | what counts as correct; capabilities; stimulus distributions | `TwoAFC`, `GoNoGo`, `Reproduction`, `Outcome`, `RequirementError` |
| `behav_utils.data` | structures, readers, discovery, loading, selection, filtering, the store | `load_experiment`, `discover_sessions`, `select_sessions`, `filter_trials`, `TrialArrays`, `save_store`, `load_store` |
| `behav_utils.stats` | the scalar statistic registry | `compute_stats`, `list_stats` |
| `behav_utils.readouts` | array-valued readouts and the fit engines | `compute_psychometric_curve`, `compute_update_matrix`, `compute_response_curve`, ... |
| `behav_utils.models` | models, designs, simulation, synthetic data | `simulate`, `make_design`, `PsychometricModel`, `IdealObserver`, `generate_synthetic_animal` |
| `behav_utils.analysis` | phase statistics, contrasts, resampling, rolling statistics, group tests | `compute_phase_stats`, `compute_delta_stat`, `compute_rolling_stats`, `compare_groups` |
| `behav_utils.plotting` | one draw-only `plot_x` per readout or result | `plot_psychometric_curve`, `plot_stat_comparison`, `plot_trajectory`, ... |

Worked example, no data needed: [notebooks/example_workflow.ipynb](notebooks/example_workflow.ipynb). Full details:
[ARCHITECTURE.md](ARCHITECTURE.md) (design and contracts), [docs/config_guide.md](docs/config_guide.md),
[docs/data_structures_reference.md](docs/data_structures_reference.md), [docs/stats_reference.md](docs/stats_reference.md).

## Public API and stability

Public means a name in `behav_utils.__all__` (the everyday ones) or in the `__all__` of a sub-package (`config`,
`tasks`, `data`, `stats`, `readouts`, `models`, `analysis`, `plotting`); each name has one home sub-package.
[docs/public_api.md](docs/public_api.md) lists them all with their signatures; it is generated, and a test fails
when it is out of date. Module paths below a sub-package and names starting with `_` are private.

From 1.0, versions follow semantic versioning: a public name is renamed or removed only in a major release, after
at least one minor release in which it warns (`DeprecationWarning`); a change that moves a number arrives only in
a minor or major release, never a patch. Details in [CONTRIBUTING.md](CONTRIBUTING.md).

## The three rules

1. **Pipeline order is fixed:** `load → select_sessions → filter_trials → compute_x → plot_x`. Compute functions take
   pre-filtered sessions or a `TrialArrays`; they never select or filter themselves.
2. **Every compute returns a typed result**: a `pd.Series`/`DataFrame` or a frozen dataclass with `to_rows()`. No
   nested dicts, no shape that depends on a `mode` argument.
3. **Every plotter is draw-only and single-panel:** `plot_x(result, ax=None)` takes the matching result, computes
   nothing and returns `(fig, ax)`; layout is the caller's. The only exceptions are the two `*_grid` helpers, which
   lay out one panel per statistic and return `(fig, axes)`.

## Go/no-go

Set `task: {type: go_nogo, go_side: positive}`. The task scores a lick to a go stimulus and a withhold to a no-go
stimulus as correct, and a blank response is a withhold. See *Go/no-go* in [docs/config_guide.md](docs/config_guide.md)
and the statistics by task in [docs/stats_reference.md](docs/stats_reference.md).

## Continuous responses

A task whose answer is a value (a dot at x on a line, a poke at y) is `task: {type: reproduction, tolerance: 0.1}`.
Stimulus and response share one linear map onto [-1, 1], so errors and the tolerance mean the same everywhere on
the line. See *Continuous responses* in [docs/config_guide.md](docs/config_guide.md).

## Recordings alongside behaviour

Not in 1.0. What the library provides now is `TrialArrays.trial_key`, an int64 identity per row (the session's hash
above the trial number; stable across loads, unique across animals) that survives every `take`, `valid` and
resample, and event times in seconds on each session's recording clock. Aligning pupil, imaging or spike data to
trials through these is planned for 1.1.

## Tests

From the repository folder; no config or data needed:

```bash
pytest tests -q
python docs/run_doc_examples.py
```

The second checks that every code block in `docs/data_structures_reference.md` still runs (a block marked
`# requires: pyarrow` is skipped where pyarrow is missing). After changing a statistic, regenerate its reference
page with `python docs/gen_stats_reference.py`.

`tests/test_golden.py` recomputes every statistic, readout, contrast and rolling statistic on two frozen datasets
and compares them with pinned values: 2-AFC values pinned from 0.10.0, continuous ones from 1.0
([tests/golden/README.md](tests/golden/README.md)). A change that moves a number fails it.

## Versioning and maintenance

`behav_utils.__version__` is the single source of the version (pyproject reads it). See
[CHANGELOG.md](CHANGELOG.md). A `v*` tag builds the package, installs it in a clean environment, runs the tests and
publishes a GitHub release with that version's CHANGELOG section.

This repository is the canonical one; projects that take the library on should fork it and pin their fork's tags
(see *Maintenance* in [CONTRIBUTING.md](CONTRIBUTING.md)).
