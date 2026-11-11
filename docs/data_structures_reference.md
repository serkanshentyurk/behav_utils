# Data Structures Reference

## Overview

behav_utils organises behavioural data in a hierarchy that mirrors experimental structure:

```
ExperimentData                    All animals in a project
  └── AnimalData                  One animal, all its sessions
        └── SessionData           One behavioural session
              ├── SessionMetadata  Task parameters (constant within session)
              └── TrialData        Trial-by-trial arrays
```

`TrialArrays` (`behav_utils.data.arrays`) is the flat, pooled view every statistic and readout
computes from; `TrialArrays.from_sessions(clean)` builds it (`pool_arrays` is the raw dict
underneath). The data classes hold data only: filtering lives in `behav_utils.data.ops`,
statistics in `behav_utils.stats` / `behav_utils.analysis`.

Every code block below runs against the current API (`sessions`, `animal`, `clean`, `experiment`
are whatever the pipeline produced; `docs/run_doc_examples.py` executes them on synthetic data).

---

## TrialData

The lowest level — per-trial arrays for a single session.

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `trial_number` | int array | Original trial numbers from CSV |
| `stimulus` | float array | Stimulus on the library's axis: [-1, 1], boundary at 0 (see `docs/config_guide.md`, *The stimulus axis*) |
| `stimulus_raw` | float array | Stimulus as read from the CSV |
| `choice` | float array | Category-space choice: 0=A, 1=B, NaN=no response |
| `choice_raw` | float array | Raw choice from CSV (before conversion) |
| `outcome` | int8 array | how the session's task scored the trial: `Outcome.ERROR` 0, `CORRECT` 1, `NO_RESPONSE` 2, `ABORT` 3, `UNSCORED` 4 |
| `correct` | bool array | `outcome == CORRECT`: the task decides it |
| `category` | float array | the task's category for the stimulus: 0=A, 1=B, NaN where it has none |
| `correct_recorded`, `outcome_recorded` | arrays (empty if unmapped) | mapped CSV `correct` / `outcome` columns, kept to check the score |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `reaction_time` | float array | Response latency (NaN if missing) |
| `abort` | bool array | Trial aborted |
| `opto_on` | bool array | Laser on this trial |
| `distribution` | str array | Stimulus distribution label |
| `optional_fields`, `extra` | dict | Other mapped / unmapped CSV columns |
| `events` | dict | Event times (columns named `t_*`), float64 seconds on the session's clock, by name |

### Lag-1 view

`prev_stimulus`, `prev_choice`, `prev_correct`, `prev_category`, `prev_reaction_time`,
`prev_opto_on`, `prev_has_prev` are frozen on the raw session (the predecessor is the previous
non-abort trial; NaN where there is none) and sliced, never recomputed, through filtering and
pooling. `trials.prev_trial` exposes them as a `PrevTrial`.

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `.n_trials` | int | Total trial count |
| `.no_response` | bool array | True where `choice` is NaN |
| `.valid_mask` | bool array | `~abort & ~no_response` |

---

## SessionData

One behavioural session — metadata + trial data.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | str | `<animal>_<YYYYMMDD>`, plus `T<HHMM>` when the start time is known and `_1`, `_2` when sessions would otherwise share it |
| `source_id` | str or None | the reader's own name for the session (folder name, table value) |
| `experimenter` | str | `project.experimenter`, or the animal metadata's; part of `session_hash` |
| `session_idx` | int | Ordinal position within animal (0-based) |
| `date` | datetime.date | Session date |
| `metadata` | SessionMetadata | Task parameters |
| `trials` | TrialData | Trial-by-trial arrays |
| `session_type` | str | a type from `config.session_types`, else a mapped `session_type` column, else `'opto'` / `'regular'` from the trials |
| `session_type_source` | str | where `session_type` came from: `'config'`, `'column'` or `'derived'` |
| `task` | Task | the session's task (the config's `task:`, or a `tasks:` entry); it alone decides correctness |
| `task_name` | str | `'default'`, or the `tasks:` entry that `task_by` picked |
| `clock` | str | the recording clock the event times are on (`config.clock.name`; `''` if none) |
| `start` | datetime.time or None | time of day the session started, when the reader knows it |
| `filter_info` | dict or None | What filtering produced this session (`label`, counts, `selection`) |

### Properties and methods

| Name | Returns | Description |
|----------|---------|-------------|
| `.n_trials` | int | Total trials (reflects filtering) |
| `.stage` | str | Training stage (from metadata) |
| `.distribution` | str | Stimulus distribution (majority label of the trials, else metadata) |
| `.is_filtered` | bool | Whether filtering has been applied |
| `.summary()` | `SessionSummary` | counts and accuracy — `n_trials, n_valid, n_abort, n_no_response, accuracy, …` |
| `.get_arrays()` | dict | the trial arrays (a pure projection — **nothing is filtered, aborts included**); see `pool_arrays` for the keys |

### Filtering trials

Filtering uses module-level functions from `behav_utils.data.ops.filtering`; the data classes have
no filter methods.

```python
from behav_utils import filter_trials
from behav_utils.data.ops.filtering import build_mask, filter_session, opto_mask

clean = filter_trials(sessions, trial_type='non_opto')   # aborts and laser-on trials dropped
allv  = filter_trials(sessions, trial_type='all')        # aborts dropped, laser trials kept
opto  = filter_trials(sessions, trial_type='opto')       # laser-on trials only
post  = filter_trials(sessions, trial_type='post_opto')  # first laser-off trial after each run

one = filter_session(sessions[0])                                      # one session, standard mask
pre = filter_trials(sessions, mask_fn=lambda s: build_mask(s.trials) & opto_mask(s.trials, -1),
                    label='pre-opto')                                  # anything else: a mask_fn
```

`trial_type` always drops aborts and keeps no-response trials (NaN choice; every statistic ignores
them). `trial_type` and `mask_fn` are exclusive.

`opto_mask(trials, delta)`:
- `delta=0` — opto trials themselves
- `delta=k>0` — k-th non-abort, non-opto trial after each opto run
- `delta=-k` — k-th non-abort, non-opto trial before each opto run
- `delta='control'` — every non-abort, non-opto trial (adjacency is *not* considered)
- `delta=None` — every non-abort trial

---

## SessionMetadata

Task parameters constant within a session. Access by attribute or `.get()`:

```python
session.metadata.stage
session.metadata.animal_id
session.metadata.get('protocol')
session.metadata.fields          # raw dict
```

Only `animal_id` and `stage` are read by the library; everything else is your config's vocabulary.

---

## AnimalData

One animal — all sessions in chronological order.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `animal_id` | str | Animal identifier |
| `sessions` | list[SessionData] | Chronologically ordered sessions |
| `metadata` | dict | Animal-level metadata, merged from `animal_metadata.json` (group, sex, …) |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `.n_sessions` | int | Session count |
| `.session_ids` | list[str] | All session IDs |
| `.stages` | list[str] | Stages present, in order of first appearance |
| `.session_table` | DataFrame | one row per session: `session_idx, session_id, date, start, stage, distribution, session_type, session_type_source, task, n_trials, n_valid, n_abort, n_no_response, accuracy` — a view for picking sessions, not a statistics table |

### Working with an animal

```python
from behav_utils import select_sessions, filter_trials, TrialArrays
from behav_utils import compute_psychometric_curve, plot_psychometric_curve, PALETTE

# 1. Select sessions (presets come from your config's session_presets)
sessions = select_sessions(animal, preset='expert')

# 2. Filter trials
clean = filter_trials(sessions, trial_type='non_opto')

# 3. Compute a readout on the pooled arrays
curve = compute_psychometric_curve(TrialArrays.from_sessions(clean), n_bootstrap=200)

# 4. Draw
fig, ax = plt.subplots()
plot_psychometric_curve(curve, ax=ax, color=PALETTE[0])
```

---

## ExperimentData

All animals in one project.

| Field / method | Description |
|---|---|
| `animals` | `{animal_id: AnimalData}` |
| `config` | the `ProjectConfig` it was loaded with (dropped on `save`) |
| `metadata` | dict; `save()` stamps `behav_utils_version` and `config_name` |
| `.animal_ids`, `.n_animals` | sorted ids / count |
| `.get_animal(id)` | one animal (KeyError with the available ids otherwise) |
| `.summary()` | one row per animal: sessions, stages, first/last date |
| `.save(path)` / `ExperimentData.load(path)` | pickle; `load` warns if the library version differs |

```python
from behav_utils import load_experiment

experiment = load_experiment('config.yaml')           # strict=True: a broken session raises
animal = experiment.get_animal('A05')
experiment.save('snapshot.pkl')
```

---

## TrialArrays

The one input type every statistic and readout consumes: aligned 1-D float arrays
`choice, stimulus, category, prev_choice, prev_stimulus, prev_category, reaction_time`; the task's
score `correct` and `prev_correct` (1 correct, 0 error, NaN where the task gives none: no response,
abort, unscored) with the `task` that gave it; plus
`trial_key` (int64: the session's hash of experimenter, animal and session id in the upper 43 bits and the trial number in the low 20, the same on every load and machine and unique across animals) when built from sessions; `session.session_hash` is its upper part.

```python
from behav_utils import TrialArrays, compute_stats, PSYCHOMETRIC

arrays = TrialArrays.from_sessions(clean)      # the pipeline path
arrays.n_trials, arrays.n_responded            # counts
arrays.valid()                                 # responded trials only
arrays.lag1_pairs()                            # responded with a usable predecessor
arrays.take(arrays.stimulus > 0)               # row subset; prev_* and trial_key travel with the rows
arrays.trial_key                               # mask an external per-trial dataset to exactly these rows
```

`trial_key` survives every `take`, `valid` and resample (a repeated trial carries its key twice), so
a neural tensor indexed by the same key can be aligned to the trials a statistic actually used.
Every statistic reads correctness from `correct` / `prev_correct`, never from `choice == category`,
so a task's definition (an unscored boundary, go/no-go withholds) reaches every number. Sessions of
different tasks are not pooled: `pool_arrays` and `TrialArrays.concat` raise.

`TrialArrays.from_sequence(choice, stimulus, category=None, *, task=None)` builds one contiguous block (a
simulator output) with no key; the task (`TwoAFC()` unless given) decides the categories and the score, and
given categories are only checked against it.

---

## Pipeline pattern

Every analysis follows the same four steps:

```python
from behav_utils import (load_experiment, select_sessions, filter_trials, TrialArrays,
                         compute_stats, PSYCHOMETRIC,
                         compute_psychometric_curve, compute_update_matrix, compute_phase_stats,
                         plot_psychometric_curve, plot_update_matrix, plot_trajectory,
                         PALETTE, apply_style)

apply_style()

# 1. LOAD
experiment = load_experiment('config.yaml')
animal = experiment.get_animal('A05')

# 2. SELECT sessions, FILTER trials
sessions = select_sessions(animal, preset='expert')
clean = filter_trials(sessions, trial_type='non_opto')
arrays = TrialArrays.from_sessions(clean)

# 3. COMPUTE — scalars, readouts, per-session trajectory
scalars = compute_stats(arrays, ['accuracy', *PSYCHOMETRIC])          # pd.Series
curve   = compute_psychometric_curve(arrays, n_bootstrap=200)         # PsychometricCurve
um      = compute_update_matrix(arrays)                               # UpdateMatrix
traj    = compute_phase_stats(clean, ['accuracy', 'mu'], per_session=True)   # PhaseStats

# 4. PLOT — draw-only, one axes each
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_psychometric_curve(curve, ax=axes[0], color=PALETTE[0])
plot_update_matrix(um, ax=axes[1])
plot_trajectory(traj, 'accuracy', ax=axes[2])
```

### Comparing two conditions

```python
from behav_utils import filter_trials, compute_delta_stat, compute_interaction, PSYCHOMETRIC
from behav_utils.plotting import plot_comparison, plot_stat_comparison, plot_stat_comparison_grid

off = filter_trials(sessions, trial_type='non_opto')
on  = filter_trials(sessions, trial_type='opto')

# Within-session, per-trial randomised label: a permutation p is valid, and trials are a valid unit
d = compute_delta_stat({'off': off, 'on': on}, ['accuracy', *PSYCHOMETRIC], reference='off',
                       n_bootstrap=1000, n_permutations=1000, units=('trials', 'sessions'))
fig, axes = plt.subplots(1, 2)
plot_comparison(d, ax=axes[0])                    # psychometric curves of both conditions
plot_stat_comparison(d, 'mu', ax=axes[1])         # one stat: a point + CI per phase
plot_stat_comparison_grid(d, ['mu', 'sigma'])     # the layout helper: one panel per stat

# Different sessions (e.g. two distributions): no permutation, sessions are the unit
d2 = compute_delta_stat({'uniform': clean_uniform, 'hard': clean_hard}, ['mu'], reference='uniform',
                        units=('sessions',))
```

`compute_delta_stat` refuses `n_permutations > 0` when a phase shares no session with the
reference, and `n_permutations` defaults to 0. It returns a `DeltaStats`:

| field | what it holds |
|---|---|
| `phases` | `{label: PhaseSummary}` — observed `stats` (Series), `n_trials`, `n_sessions`, bootstrap `draws` per unit, optional `curve` / `update_matrix` |
| `contrasts` | `{key: Contrast}` — one per non-reference phase: `diff` (a − b), `difference_draws` per unit, `perm_p` (trial permutation, when requested), `um_diff` / `um_rmse` / `um_corr` |
| `reference` | the reference label |

`Contrast.boot(unit)` gives `ci_lo, ci_hi, p, median, n_draws` per stat; `Contrast.table(unit)` one
tidy row per stat (`diff, ci_lo, ci_hi, boot_p, perm_p, unit`). Units: `'trials'` (valid only when the
condition was randomised per trial) and `'sessions'` (always valid). `compute_interaction(d1, d2, key)`
gives the delta of deltas as an `Interaction`, with `plot_interaction(ix, stat, ax=)`.

### Group-level claims (across animals)

For across-animal comparisons (two groups, or paired on/off effects), use the group-level functions —
never pool trials across animals. `compute_phase_stats` turns each animal's sessions into one pooled
row per stat (stamped with the animal id from the data); `analysis.group` tests the per-animal
values. The *animal* is the unit.

```python
from behav_utils.analysis import collect_rows, compare_groups, paired_diff, bootstrap_units
from behav_utils import compute_phase_stats, PSYCHOMETRIC

# One pooled row per animal, stamped with its group
rows = []
for animal in experiment.animals.values():
    clean = filter_trials(select_sessions(animal, preset='expert'), trial_type='non_opto')
    r = compute_phase_stats(clean, [*PSYCHOMETRIC, 'accuracy'])
    rows += collect_rows(r.to_rows().to_dict('records'), animal=animal.animal_id,
                         group=animal.metadata['group'])

# Unpaired (between-group): one rank test per stat, control first
res = compare_groups(rows, groups=('control', 'treated'))       # GroupComparison
res.test('mu').p, res.test('mu').min_p                            # min_p = floor at these group sizes
res.table()                                                       # one tidy row per stat

# Paired (on vs off within the same animals): per-animal Δ, then an across-animal CI
points = pd.DataFrame(rows_on + rows_off)                 # rows stamped condition='on' / 'off'
delta = paired_diff(points, by='condition', a='on', b='off')
bootstrap_units(delta[delta.stat == 'mu'].delta.values)   # UnitBootstrap(point, lo, hi, n)
# or in one call, aligned by animal:
compare_groups(rows_on + rows_off, group_col='condition', groups=('on', 'off'), paired=True)
```

---

## Psychometric naming convention

Psychometric fit parameters use math names everywhere in code:

| Code key | Display label | Meaning |
|----------|---------------|---------|
| `mu` | PSE | Point of subjective equality (boundary) |
| `sigma` | slope | Noise width (smaller = steeper psychometric) |
| `lapse_low` | λ_low | Lower asymptote lapse |
| `lapse_high` | λ_high | Upper asymptote lapse |
| `accuracy` | Accuracy | Overall correctness |

Plot functions translate `mu`/`sigma` to `PSE`/`slope` for y-axis labels. You write `'mu'` in code;
the plot shows "PSE".

```python
# Per-session trajectory of PSE
r = compute_phase_stats(clean, ['accuracy', 'mu', 'sigma'], per_session=True)
plot_trajectory(r, 'mu')      # y-axis label: "PSE"
plot_trajectory(r, 'sigma')   # y-axis label: "slope"
```

---

## Synthetic Data

Synthetic sessions come from a model (`behav_utils.models`): anything with `simulate(design, rng)`. Each session
has every trial class (responses, no-responses where the task has them, aborts, and with `laser_fraction` laser
trials and the trials after them) and event times on a clock named `synthetic`, and is scored by its task:

```python
from behav_utils import PsychometricModel, generate_synthetic_animal, generate_synthetic_experiment

# One model for every session
animal = generate_synthetic_animal(PsychometricModel(sigma=0.3), n_sessions=20, trials_per_session=200,
                                   animal_id='SYN01', rng=0)

# A learning trajectory: one model per session, with laser trials
models = [PsychometricModel(sigma=0.8 - i * 0.03) for i in range(20)]
learner = generate_synthetic_animal(models, n_sessions=20, animal_id='LEARN01', laser_fraction=0.3, rng=1)

# An experiment: animals with groups, each on its own random stream
experiment = generate_synthetic_experiment(
    {'A01': {'model': PsychometricModel(sigma=0.3), 'metadata': {'group': 'control'}},
     'A02': {'model': PsychometricModel(sigma=0.5), 'metadata': {'group': 'treated'}}},
    rng=2,
)
```

`distribution` (a `behav_utils.tasks` distribution) and `label` set the stimuli and their label, per session if
given as lists; `task=GoNoGo()` gives go/no-go sessions, where a blank is a withhold; `timing=Timing(...)` sets the
inter-trial interval, stimulus delay and reaction times. A model of your own, a learner included, is shown in
`EXTENDING.md`; `simulate(model, make_design(...))` runs a single design, and `session_from_arrays` builds a
session from arrays you already have.
