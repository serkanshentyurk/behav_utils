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

Sessions are drawn once per draw for all the phases of a contrast, within groups of sessions that belong to the
same phases. Two phases cut from the same sessions, such as laser on and off, therefore get the same sessions in
every draw: the contrast is paired, and what varies from session to session in both phases cancels in their
difference. Phases with different sessions are drawn separately, and each keeps its number of sessions. When
phases share only some of their sessions, the shared ones are drawn together and the rest on their own; a session
left alone in its group is the same in every draw, so the interval leaves out its variation, and
`compute_delta_stat` warns. Before 1.1 each phase drew its sessions independently, which made the interval of a
within-session contrast too wide.

A percentile interval from few sessions is too narrow, as for any bootstrap over few units: with about ten
sessions an interval meant to cover 95% covers roughly 90%, and with five roughly 80%. Read a session interval
from few sessions as rough.

**Permutation.** With `n_permutations`, the condition labels are shuffled between trials, and the p value is
(1 + the number of shuffles at least as extreme) / (n + 1), so it is never zero and its floor is 1 / (n + 1).
It is off by default, and refused unless every compared phase shares at least one session with the reference
and no trial with it: shuffling labels is valid only when the label was assigned per trial, within the same
sessions. The shuffle pools the trials of all those sessions, so it assumes behaviour does not change between
them.

**Animals.** Claims about animals rest on one value per animal. `collect_rows` gathers per-animal values,
`compare_groups` tests two groups (Mann–Whitney; Mann & Whitney, 1947) or two conditions in the same animals,
aligned by animal (Wilcoxon signed-rank; Wilcoxon, 1945), and `bootstrap_units` gives an interval across animals. With few animals, a rank test cannot
give a small p: with three animals per group, a two-sided test cannot go below 0.1. Each test reports this floor
as `min_p`; check a p against it before reading a null result as no effect.

**Pooled, not averaged.** `compute_phase_stats` fits the pooled trials of a phase, and its pooled value is never
a mean of per-session values: such a mean would weight a short session like a long one and, for a fitted
parameter, differs from the fit to the pooled data. Per-session values come separately, with
`per_session=True`.

The [data-structures reference](../data_structures_reference.md) runs these contrasts on synthetic data.

The intervals are percentile bootstrap intervals (Efron & Tibshirani, 1993), and the permutation p adds one to the
count and to the number of permutations (Phipson & Smyth, 2010); the [references](../references.md) give both in
full. Stratifying the trial bootstrap by stimulus bin, resampling whole sessions, and drawing shared sessions once
for all phases are the library's own choices for these designs.
