# Concepts

How the library thinks: what each step of an analysis may and may not do, what a task decides, the axis every
stimulus is put on, how sessions and trials keep their identity, and what each resampling unit can tell you.
The tutorials show the same ideas running.

```{toctree}
:maxdepth: 1

pipeline
tasks
stimulus-axis
identity-and-time
resampling
```

## What is deliberately not here

- Anything about a specific experiment: distribution names, experiment-specific normative observers,
  session-type semantics, genotypes, cohort logic, report layouts. Those belong to the project that uses the
  library.
- Model fitting (SBI, grid search): project code. The library simulates models and computes the statistics a
  fit would compare, but does not fit.
- Neural data. The library's contribution is the trial key (see [Identities and time](identity-and-time.md));
  sync tables, per-trial binning of spikes or frames, and encoding or decoding models belong in the project.
