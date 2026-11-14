# The pipeline

Every analysis follows the same order:

```text
load_experiment(config)          -> ExperimentData {animal_id: AnimalData [SessionData ...]}
select_sessions(animal, ...)     -> list of SessionData      which sessions
filter_trials(sessions, ...)     -> list of SessionData      which trials
TrialArrays.from_sessions(...)   -> TrialArrays              aligned arrays, previous trial carried along
compute_x(...)                   -> a typed result
plot_x(result, ax)               -> the figure and the axes
```

Three rules hold it together.

**Only selection and filtering decide which data are used.** `select_sessions` picks sessions, by stage,
distribution, session type, position or quality, or through a named preset, and `filter_trials` picks trials:
with or without the laser, aborts dropped, or any mask of your own. Every compute function then analyses all of
what it is given. A compute that filtered internally would hide a choice that belongs in plain sight, so none
does.

**Every compute returns a typed result**: a pandas Series or DataFrame, or a frozen dataclass whose `to_rows()`
gives a tidy table. Never a nested dict, and never a shape that depends on a mode argument.

**Every plotter only draws, on one panel.** `plot_x(result, ax=None)` takes the matching result, computes
nothing and returns the figure and the axes; the layout is yours. The two `*_grid` helpers are the exception:
they lay out one panel per statistic.

The loader is strict by default: a session that cannot be read as the config describes stops the load with an
error naming it, rather than being skipped quietly. The [configuration guide](../config_guide.md) describes the
config, and the [2-AFC tutorial](../tutorials/two_afc.ipynb) runs the whole pipeline on synthetic data.
