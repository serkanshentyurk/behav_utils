# behav_utils.stats

```{eval-rst}
.. automodule:: behav_utils.stats
```

## Constants

`PSYCHOMETRIC`, `LOGISTIC_HISTORY`, `SD_PROFILE`, `PSE_DYNAMICS` and `SDT` are tuples of the output names of the
fitters `psychometric`, `logistic_history`, `sd_profile`, `pse_dynamics` and `sdt`, to splat into a request, as in
`compute_stats(arrays, [*PSYCHOMETRIC, 'accuracy'])`. The [statistics reference](../stats_reference.md) lists
each fitter's outputs.
