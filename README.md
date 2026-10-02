# behav_utils

Analysis library for two-alternative forced-choice (2-AFC) behavioural data.
Loads sessions from CSV via a YAML config, selects and filters trials, computes psychometric and
trial-history statistics with resampling-based uncertainty, and draws the standard figures.

It knows nothing about any particular experiment: distributions, session types and presets all
come from the config you give it.

## Install

```bash
pip install "behav_utils @ git+https://github.com/serkanshentyurk/behav_utils.git@v0.7.0"   # from its own repo, pinned
pip install -e path/to/behav_utils                                                # editable, for development
```

Requires Python ≥ 3.10; numpy, pandas, scipy, matplotlib, pyyaml.

## Sixty-second tour

```python
from behav_utils import (load_experiment, select_sessions, filter_trials, TrialArrays,
                         compute_stats, PSYCHOMETRIC, compute_psychometric_curve,
                         compute_phase_stats, compute_delta_stat, plot_psychometric_curve)

experiment = load_experiment('config.yaml')          # {animal_id: AnimalData}
animal     = experiment.animals['A01']

sessions = select_sessions(animal, preset='expert_uniform')   # presets defined in the config
phase    = filter_trials(sessions, trial_type='non_opto')     # drops aborts and laser trials

arrays = TrialArrays.from_sessions(phase)                     # one aligned bundle of arrays
s = compute_stats(arrays, [*PSYCHOMETRIC, 'accuracy', 'win_stay'])   # pd.Series of floats
s['mu'], s['sigma']

curve = compute_psychometric_curve(arrays)                    # fit + bootstrap band
plot_psychometric_curve(curve)                                # one axes, nothing computed

r = compute_phase_stats(phase, ['accuracy', 'mu'], per_session=True) # pooled Series + tidy per-session frame
d = compute_delta_stat({'off': phase, 'on': filter_trials(sessions, trial_type='opto')},
                       ['mu', 'accuracy'], reference='off')  # bootstrap CI + permutation p
d.contrast('on').table()
```

## What's in it

| package | what | entry points |
|---|---|---|
| `behav_utils.config` | YAML schema and loader | `load_config`, `load_cohorts` |
| `behav_utils.data` | data structures, CSV loading, session selection, trial filtering, synthetic data | `load_experiment`, `select_sessions`, `filter_trials`, `pool_arrays`, `TrialArrays`, `generate_synthetic_animal`, `find_switches` |
| `behav_utils.stats` | scalar statistics registry | `compute_stats`, `list_stats`, `PSYCHOMETRIC`, `PSE_DYNAMICS` |
| `behav_utils.readouts` | array-valued readouts as dataclasses, and the two fit engines on raw arrays (`fit.py`) | `compute_psychometric_curve`, `compute_update_matrix`, `compute_conditional_psychometric`, `compute_binned_curve`, `compute_sd_profile`, `fit_psychometric`, `fit_update_matrix` |
| `behav_utils.analysis` | phase statistics, contrasts, resampling, rolling, group tests | `compute_phase_stats`, `compute_delta_stat`, `compute_interaction`, `bootstrap_phase_stats`, `permute_phase_difference`, `compute_rolling_stats`, `collect_rows`, `compare_groups` |
| `behav_utils.plotting` | one draw-only `plot_x` per `compute_x` | `plot_psychometric_curve`, `plot_update_matrix`, `plot_comparison`, `plot_stat_comparison`, `plot_interaction`, `plot_trajectory` |

Worked example, no data needed: [notebooks/example_workflow.ipynb](notebooks/example_workflow.ipynb).
Full details: [ARCHITECTURE.md](ARCHITECTURE.md) (design and contracts), [docs/config_guide.md](docs/config_guide.md),
[docs/data_structures_reference.md](docs/data_structures_reference.md), [docs/stats_reference.md](docs/stats_reference.md)
(every statistic), [LLM_CONTEXT.md](LLM_CONTEXT.md) (orientation for an AI assistant), [CONTRIBUTING.md](CONTRIBUTING.md).

## The three rules

1. **Pipeline order is fixed:** `load → select_sessions → filter_trials → compute_x → plot_x`. Compute functions take
   pre-filtered sessions or a `TrialArrays`; they never select or filter themselves.
2. **Every compute returns a typed result** — a `pd.Series`/`DataFrame` or a frozen dataclass with `to_rows()`. No nested
   dicts, no shape that depends on a `mode` argument.
3. **Every plotter is draw-only:** `plot_x(result, ax=None)` takes the matching result and computes nothing.

## Tests

```bash
pytest tests -q          # from behav_utils/; no config or data needed
```

## Versioning

Semantic. `behav_utils.__version__` is stamped into snapshots and result metadata by projects that use it.
See [CHANGELOG.md](CHANGELOG.md).
