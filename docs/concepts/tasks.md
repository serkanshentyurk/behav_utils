# Tasks, families and capabilities

A task is the part of a config that says what the trials were for. Given the stimuli, it decides three things:
each trial's category, the correct response, and how the response is scored, as an `Outcome` (correct, error,
no response, abort or unscored). The library knows no experiment; the meaning of a trial lives in the task, and
nothing else second-guesses it.

## Families

A family is defined by what a response is.

- **Binary**: the response is 0 or 1. In `TwoAFC` both responses are active choices; in `GoNoGo` 0 means
  withholding, so a blank response that is not an abort is scored as a withhold rather than lost.
- **Continuous**: the response is a value on the stimulus scale. `Reproduction` counts a response as correct
  within a tolerance of the stimulus.

Each family has its own statistics, readouts and plots and a descriptive model, and `IdealObserver` serves both
as the normative model: it gives the more probable category, or the posterior mean of the stimulus. Loading,
scoring, selection, pooling, contrasts, resampling and synthetic data work the same way for every family. A
K-choice family is planned (see the [roadmap](../ROADMAP.md)).

## Capabilities

What a task provides is a set of capability names, and statistics, readouts and models declare the
capabilities they need. Asking for one that the task cannot support raises a `RequirementError` that names what
is missing, what the task has and, where there is one, the statistic to use instead. Behaviour depends on a
task's capabilities and response space, never on its class, so a task of your own (see
[Extending](../EXTENDING.md)) gets every statistic its capabilities allow.

| Capability | Meaning | TwoAFC | GoNoGo | Reproduction |
|---|---|---|---|---|
| `binary` | responses are 0 or 1 | ✓ | ✓ | |
| `two_choices` | both responses are active choices | ✓ | | |
| `withhold` | response 0 is withholding | | ✓ | |
| `stimulus_axis` | a signed stimulus whose category is its side of the boundary at 0 | ✓ | ✓ | |
| `scored` | the task defines a correct response | ✓ | ✓ | ✓ |
| `continuous` | responses are values on the stimulus scale | | | ✓ |
| `k_choices` | responses are one of K > 2 categories, coded 0 to K−1 | | | |

The [statistics reference](../stats_reference.md) lists what every statistic needs, and which statistics each
built-in task refuses.
