# Roadmap

Plans after 1.0, not promises. Everything here is additive, so it can arrive in a 1.x minor release; anything
that would break the public API waits for 2.0 (*Stability* in [CONTRIBUTING.md](CONTRIBUTING.md)). Estimates are
upper bounds in focused hours, given only where one was made. When an item ships, it leaves this file and enters
the [CHANGELOG](CHANGELOG.md).

## 1.1: streams

Recordings alongside behaviour (pupil, imaging, spikes, licks), aligned to trials. Planned as six steps on a new
branch, about 115 h, built against the first real recording rather than in advance, so that no stream API is
frozen before it has met data.

What 1.0 already provides for it: event times in seconds on one recording clock per session (`t_*` columns,
checked at load), the durable `TrialArrays.trial_key`, and the reader's `files` slot for files that belong to a
session.

The agreed design:

- Own minimal types: `TimeSeries` (pupil size, ΔF/F per ROI, running speed), `Units` (sorted spikes) and `Events`
  (licks, laser on and off, rewards), on `SessionData.streams`, loaded per session on first access.
- Operations: `align(stream, session, event, window, bin)` gives an `Aligned` array (trials × time × channels,
  keyed by `trial_key`, NaN outside the trial or the recording); `reduce` gives one value per trial (and channel);
  `restrict` keeps time intervals; `add_trial_column` adds a per-trial value, after which trials are selected and
  filtered as usual (for example, psychometric curves split by baseline pupil).
- Storage: HDF5 through h5py, one file per session.
- numpy output with coordinates; `to_xarray()` and Pynapple converters (`to_pynapple()`, `from_pynapple()`) are
  optional, in a `streams` extra with h5py and scikit-learn.
- Readouts, each a frozen dataclass with one plotter: PSTH by condition and tuning to stimulus bins first, then
  per-time-bin decoding of choice or category with whole sessions held out, and choice probability.
- Synthetic streams with known tuning, for tests and for checking analyses.

The six steps: types, checks and synthetic streams; `align`, `reduce`, `restrict` and `add_trial_column`; HDF5
storage loaded on first access; PSTH and tuning; decoding and choice probability; Pynapple converters and the
first real recording.

Out of scope for streams: synchronisation, spike sorting, ROI extraction, motion correction, and matching ROIs
across days.

## 1.x

- **K-choice family** (about 30 h): a task with K responses, its model and its readouts (a confusion matrix,
  choice proportions per stimulus bin), on the template the continuous family set in 1.0. A three-choice task
  defined in `tests/test_generality.py` already runs through every interface.
- **Per-trial context:** tasks take a per-trial `context` (a rule, a distribution) in their methods, but nothing
  fills it yet; the loader and `make_design` would.
- **Stimuli with several features:** an added feature matrix beside the scalar stimulus, which stays on [-1, 1].
- **Reaction time from two events,** declared in the config (for example `t_response` minus `t_stim_on`).
- **Event times in `TrialArrays`,** so pooled analyses can use them without going back to the sessions.
- **Clocks that wrap at midnight;** today the jump back is reported as a clock problem.
- **Adding sessions to an existing store;** today a store is written whole.
- **Store migrations,** written when the store's layout first changes (schema version 2).
- **NWB export,** for sharing data; the store itself stays Parquet.
- **PyPI publishing;** today releases install from their GitHub tag.

## 2.0

- **Project vocabulary out of `plotting.COLOURS`.** Besides generic roles (`correct`, `error`, `no_response`,
  `mean_line`, `sem_fill`, `default`), it holds names from one study's models and phases (`BE`, `SC`, `naive`,
  `expert`, `post_shift`, `rule_flip` and others), against the rule that the library carries no project
  vocabulary. Removing them changes a public name, so it waits for 2.0; until then they stay and are not
  documented.

## Decided against

- **Matching lab-specific companion files** (which video or imaging file belongs to which session) inside the
  library. Layouts differ between labs; that logic belongs in a lab's own tooling, which can pass the files to a
  reader through `RawSession.files`.
- **General time-series or clinical-trial features.** The library stays about trial-based behaviour, with
  recordings aligned to trials.
- **Plugin entry points.** Registration stays an explicit import; worth revisiting once a third-party package
  exists.
- **Project-specific models in the library.** Models built for one study live in that study's code and plug in
  through the `Model` interface, so the library does not follow a thesis's revisions.
