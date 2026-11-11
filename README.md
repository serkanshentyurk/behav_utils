# behav_utils

Analysis library for two-alternative forced-choice (2-AFC) and go/no-go categorisation data.
Loads sessions from CSV via a YAML config, selects and filters trials, computes psychometric and
trial-history statistics with resampling-based uncertainty, and draws the standard figures.

It knows nothing about any particular experiment: stimulus units, distributions, session types and
presets all come from the config you give it. Every computation runs on one normalised stimulus
axis ([-1, 1], category boundary at 0) that the loader maps your raw stimulus onto.

## What it assumes

As of the `v1` branch, with the 1.0 work in progress (see the CHANGELOG).

- One row per trial. The default reader takes CSV files laid out as `data_dir/<animal>/<session>/<file>.csv`
  (several files in a session folder merged in name order; the date, and optionally the start time, from the
  folder name). One table holding every session, or a reader of your own, also work.
- Session metadata (animal id, stage, ...) is taken from each file's first row.
- One scalar stimulus per trial, mapped onto [-1, 1] with the category boundary at 0.
- A binary choice: 1 for the positive side (B), 0 for A, NaN for no response. The config names the raw
  value that means no response.
- What counts as correct is the session's task (2-AFC by default, or go/no-go), set in the config; recorded
  correct or outcome columns are only checked against it. In 2-AFC a stimulus exactly on the boundary
  counts as A unless the config says otherwise.
- Optional per-trial columns: reaction time, abort, laser (opto), distribution label, and any others you
  map or keep as extra columns.
- Go/no-go is a task type (below).
- Times (event columns named `t_*`, reaction time) are seconds on one recording clock per session, never
  re-zeroed; the config names the clock and the unit the files use.

## Install

A pinned release:

```bash
pip install "behav_utils @ git+https://github.com/serkanshentyurk/behav_utils.git@v0.10.0"
```

Or your own checkout, editable, to work on the library:

```bash
pip install -e <path to behav_utils>
```

Requires Python ≥ 3.11; numpy, pandas, scipy, matplotlib, pyyaml.

## Sixty-second tour

```python
from behav_utils import (load_experiment, select_sessions, filter_trials, TrialArrays,
                         compute_stats, PSYCHOMETRIC, compute_psychometric_curve,
                         compute_phase_stats, compute_delta_stat, plot_psychometric_curve)

experiment = load_experiment('config.yaml')          # {animal_id: AnimalData}
animal     = experiment.animals['A01']

sessions = select_sessions(animal, preset='expert')           # presets defined in the config
phase    = filter_trials(sessions, trial_type='non_opto')     # drops aborts and laser trials

arrays = TrialArrays.from_sessions(phase)                     # one aligned bundle of arrays
s = compute_stats(arrays, [*PSYCHOMETRIC, 'accuracy', 'win_stay'])   # pd.Series of floats
s['mu'], s['sigma']

curve = compute_psychometric_curve(arrays)                    # fit + bootstrap band
plot_psychometric_curve(curve)                                # one axes, nothing computed

r = compute_phase_stats(phase, ['accuracy', 'mu'], per_session=True) # pooled Series + tidy per-session frame
d = compute_delta_stat({'off': phase, 'on': filter_trials(sessions, trial_type='opto')},
                       ['mu', 'accuracy'], reference='off',
                       n_permutations=1000)                   # same sessions, per-trial label → permutation p is valid
d.contrast('on').table()
```

## What's in it

| package                  | what                                                                                     | entry points                                                                                                                                                                                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `behav_utils.config`   | YAML schema and loader; the raw → normalised stimulus transform                         | `load_config`, `load_cohorts`, `TaskConfig.normalise_stimulus`                                                                                                                                                                                    |
| `behav_utils.data`     | data structures, CSV loading, session selection, trial filtering, synthetic data         | `load_experiment`, `select_sessions`, `filter_trials`, `pool_arrays`, `TrialArrays` (with `trial_key`), `generate_synthetic_animal`, `find_switches`                                                                                    |
| `behav_utils.stats`    | scalar statistics registry                                                               | `compute_stats`, `list_stats`, `PSYCHOMETRIC`, `PSE_DYNAMICS`                                                                                                                                                                                   |
| `behav_utils.readouts` | array-valued readouts as dataclasses, and the two fit engines on raw arrays (`fit.py`) | `compute_psychometric_curve`, `compute_update_matrix`, `compute_conditional_psychometric`, `compute_binned_curve`, `compute_sd_profile`, `fit_psychometric`, `fit_update_matrix`                                                          |
| `behav_utils.analysis` | phase statistics, contrasts, resampling, rolling, group tests                            | `compute_phase_stats`, `compute_delta_stat`, `compute_interaction`, `bootstrap_phase_stats`, `permute_phase_difference`, `compute_rolling_stats`, `collect_rows`, `compare_groups`, `paired_diff`, `bootstrap_units`, `rank_test` |
| `behav_utils.plotting` | one single-panel, draw-only `plot_x` per result type                                   | `plot_psychometric_curve`, `plot_update_matrix`, `plot_comparison`, `plot_stat_comparison`, `plot_interaction`, `plot_trajectory`, `plot_rolling_stats` (+ `*_grid` layout helpers)                                                     |

Worked example, no data needed: [notebooks/example_workflow.ipynb](notebooks/example_workflow.ipynb).
Full details: [ARCHITECTURE.md](ARCHITECTURE.md) (design and contracts), [docs/config_guide.md](docs/config_guide.md),
[docs/data_structures_reference.md](docs/data_structures_reference.md), [docs/stats_reference.md](docs/stats_reference.md)
(every statistic), [LLM_CONTEXT.md](LLM_CONTEXT.md) (orientation for an AI assistant), [CONTRIBUTING.md](CONTRIBUTING.md).

## Public API

Public means a name in `behav_utils.__all__` or in the `__all__` of one of the sub-packages (`config`, `data`,
`stats`, `readouts`, `analysis`, `plotting`). Each name has one home sub-package, and the most used ones are
re-exported at the top level. [docs/public_api.md](docs/public_api.md) lists them all with their signatures; it
is generated, and a test fails when it is out of date. Module paths below a sub-package
(`behav_utils.readouts.fit`) and names starting with `_` are private.

## The three rules

1. **Pipeline order is fixed:** `load → select_sessions → filter_trials → compute_x → plot_x`. Compute functions take
   pre-filtered sessions or a `TrialArrays`; they never select or filter themselves.
2. **Every compute returns a typed result** — a `pd.Series`/`DataFrame` or a frozen dataclass with `to_rows()`. No nested
   dicts, no shape that depends on a `mode` argument.
3. **Every plotter is draw-only and single-panel:** `plot_x(result, ax=None)` takes the matching result and computes
   nothing; layout is the caller's. The only exception is the pair of `*_grid` helpers, which lay out one panel per stat.

## Go/no-go

Set `task: {type: go_nogo, go_side: positive}`. The task scores a lick to a go stimulus and a withhold to a
no-go stimulus as correct, and a blank response is a withhold. See *Go/no-go* in
[docs/config_guide.md](docs/config_guide.md) and the reading table at the top of
[docs/stats_reference.md](docs/stats_reference.md).

## Neural data

Out of scope here, by design. What the library provides for it is `TrialArrays.trial_key` — an int64
identity per row (the session's hash above the trial number; stable across loads, unique across animals)
that survives every `take`, `valid` and
resample — so a per-trial neural tensor keyed the same way can be masked to exactly the trials a
statistic used. Alignment (sync tables, binning spikes or frames per trial) belongs in the project.

## Tests

From the repository folder; no config or data needed:

```bash
pytest tests -q
python docs/run_doc_examples.py
```

The second checks that every code block in `docs/data_structures_reference.md` still runs. After changing
a statistic, regenerate its reference page:

```bash
python docs/gen_stats_reference.py
```

`tests/test_golden.py` recomputes every statistic, readout, contrast and rolling statistic on a frozen dataset
and compares them with values pinned from 0.10.0 ([tests/golden/README.md](tests/golden/README.md)); a change
that moves a number fails it.

## Versioning and maintenance

Semantic versioning from 1.0: a public name changes only after a release that warns about it
(`DeprecationWarning`), and is removed only in the next major version. Before 1.0 the API moves freely on the
`v1` branch. `behav_utils.__version__` is the single source of the version (pyproject reads it) and is stamped
into snapshots and result metadata by projects that use it. See [CHANGELOG.md](CHANGELOG.md). A `v*` tag
builds, tests and publishes a GitHub release automatically.

This repository is the canonical one; projects that take the library on should fork it and pin their
fork's tags (see *Maintenance* in [CONTRIBUTING.md](CONTRIBUTING.md)).
