# Identities and time

## Sessions

A session's id is built at load from the animal and the date: `<animal>_<YYYYMMDD>`, with `T<HHMM>` added when
the start time is known, and `_2`, `_3` for later sessions that would otherwise share it. The reader's own name
for the session, such as a folder name, is kept as `source_id`. `discover_sessions` shows the ids before any
trial is read, and `load_experiment` gives the same ones.

## Trials

Every row of a `TrialArrays` built from sessions carries a `trial_key`: an int64 whose upper 43 bits hash the
experimenter, the animal and the session id, and whose lower 20 bits are the trial number. It is the same on
every load and every machine, unique across animals, and it travels with the rows through `take`, `valid`,
`lag1_pairs`, `concat` and every resample, so a trial drawn twice carries its key twice.

That makes it the link to anything else recorded per trial, such as neural data: compute the keys of that
dataset's trials with `trial_key_of(session, trial_number)`, and mask it to exactly the rows a statistic used.
Arrays built with `TrialArrays.from_sequence`, as simulations are, have no keys.

Each trial also carries its previous trial (the `prev_*` arrays), computed once on the raw session, skipping
aborts, and kept through filtering and pooling. A filtered trial's previous trial is still the one the animal
saw, and pooled data never bridge two sessions.

## Time

Event times (columns named `t_*`) and reaction times are float64 seconds on the recording clock named in the
config's `clock:` block, and they are never re-zeroed: anything recorded on the same clock lines up with the
trials without an offset. Units are converted once, at load, and the loader checks each session for times that
go backwards, negative reaction times and, if configured, the order of events within a trial.
