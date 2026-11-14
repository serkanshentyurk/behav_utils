# Resampling units

A contrast between conditions comes with uncertainty from resampling, and what that uncertainty means depends
on what is resampled. `compute_delta_stat` takes the units to use and reports each. A phase is bootstrapped once
per unit and every contrast subtracts stored draws, so a reference shared by two contrasts cancels exactly.

**Trials.** Trials are resampled within each phase, stratified by stimulus bin so that every draw keeps the
phase's mix of stimuli. This is valid when the condition was randomised per trial, as with interleaved laser
trials. Statistics that depend on the order of trials, such as history effects, cannot be resampled this way:
asking for one with trial resampling raises an error, and the [statistics reference](../stats_reference.md) marks
which statistics allow it.

**Sessions.** Whole sessions are resampled. This is the honest unit when the condition varies by session, as
when comparing two distributions or two stages of training: the sessions, not the trials within them, are then
the independent observations. A phase with fewer than two sessions gets no session draws.

**Permutation.** With `n_permutations`, the condition labels are shuffled between trials, and the p value is
(1 + the number of shuffles at least as extreme) / (n + 1), so it is never zero and its floor is 1 / (n + 1).
It is off by default, and refused unless every compared phase shares at least one session with the reference
and no trial with it: shuffling labels is valid only when the label was assigned per trial, within the same
sessions.

**Animals.** Claims about animals rest on one value per animal. `collect_rows` gathers per-animal values,
`compare_groups` tests two groups (Mann–Whitney) or two conditions in the same animals (Wilcoxon signed-rank,
aligned by animal), and `bootstrap_units` gives an interval across animals. With few animals, a rank test cannot
give a small p: with three animals per group, a two-sided test cannot go below 0.1. Each test reports this floor
as `min_p`; check a p against it before reading a null result as no effect.

**Pooled, not averaged.** `compute_phase_stats` fits the pooled trials of a phase, and its pooled value is never
a mean of per-session values: such a mean would weight a short session like a long one and, for a fitted
parameter, differs from the fit to the pooled data. Per-session values come separately, with
`per_session=True`.

The [data-structures reference](../data_structures_reference.md) runs these contrasts on synthetic data.
