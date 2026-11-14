# Pitfalls

Mistakes that are easy to make with this library, each with what to do instead. Most of them the library catches
and names in an error or a warning; the rest it cannot see. Problems specific to reading files are also in the
configuration guide's *Troubleshooting*.

## Loading

**A text column fails with "could not convert string to float".** A column whose raw values are text, such as
'L' and 'R', needs `dtype: float` with a `mapping` (`{L: 0, R: 1, N: .nan}`): the loader maps the values first
and converts them after. With `dtype: str` the mapped numbers become text again and the conversion fails.

**Every A choice turns into a no-response.** `no_response_value` must be stated whenever a choice column is
mapped, and it is never assumed: a value of 0 in a 0/1 column would make every A choice a no-response. Write
`no_response_value: null` when no raw value means no response; blank cells are always no-responses.

**A go/no-go session has most of its choices missing.** A withhold must arrive as choice 0. A NaN choice means
nothing was recorded, and it is left out of every statistic; the loader warns when more than 30% of a session's
choices are NaN, which usually means the withheld response was left blank and needs mapping to 0.

**Session ids change when a session is added.** Sessions that would share an id, two on one day without a start
time, are numbered `_1`, `_2` in name order, so adding or removing one renumbers the others; the loader warns.
Capture the start time in `date_regex` (groups `hour` and `minute`) for ids that do not depend on order.

**A setting is ignored, or the load stops at a key.** Config keys are checked, and a misspelt one raises a
`ConfigError` naming the closest valid key instead of being ignored. Settings that belong to your project and not
to the library go under `custom:`. Entries in `sessions_to_ignore` and `session_types` that contain an underscore
must be quoted in YAML.

## Selecting and filtering

**Laser trials end up in the analysis.** `filter_trials(trial_type='all')` keeps laser trials; `'non_opto'` leaves
them out. Every `trial_type` drops aborts and keeps no-response trials, whose choice is NaN and which every
statistic then ignores.

**The binned readouts put everything in the end bins.** The stimulus has to be on the library's axis, [−1, 1] with
the boundary at 0. The loader puts it there from the config's task block; data built by hand in raw units, such as
Hz or degrees, have to be mapped first (see [The stimulus axis](../concepts/stimulus-axis.md)).

## Statistics and contrasts

**A pooled value differs from the mean of the per-session values.** It should: the pooled value is the fit to all
the trials. A mean of per-session values weights a short session like a long one, and for a fitted parameter it is
a different quantity. Use `compute_phase_stats(...).pooled` for a phase, and the per-session values to show how
sessions vary.

**A permutation p is refused.** It is valid only when the condition was assigned per trial within the same
sessions, and the reference shares no trial with the phase it is compared with (`post_opto` lies inside
`non_opto`, for example). For conditions that come in different sessions, use `units=('sessions',)` (see
[Resampling units](../concepts/resampling.md)).

**A group test is never significant.** With few animals a rank test cannot give a small p: two-sided, the floor is
0.1 with three animals per group and about 0.03 with four. Each test reports this floor as `min_p`; compare against
it before reading a null result as no effect.

**Several animals count as one in a group test.** The animal id did not reach the rows. Stamp each animal's rows
with `collect_rows(..., animal=animal.animal_id)` (see [Compare groups of animals](group-comparison.md)).

**Trials from different sessions get mixed up.** `trial_number` restarts in every session; `TrialArrays.trial_key`
is the identity that is unique across sessions and animals and survives resampling (see
[Align another per-trial dataset](align-per-trial-data.md)).

**A loop over animals or sessions is slow.** `compute_psychometric_curve` bootstraps 1,000 times by default; pass
`n_bootstrap=0` inside loops, and bootstrap once, at the level that matters.
