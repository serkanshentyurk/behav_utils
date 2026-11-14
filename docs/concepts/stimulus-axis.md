# The stimulus axis

Everything after loading assumes one axis: stimuli on [−1, 1], with the category boundary at 0. Binning (one
fixed grid), the bounds of the psychometric fit and the split into hard and easy trials all rely on it, so a
statistic means the same in every project.

The loader puts each stimulus onto the axis using three settings in the config's `task:` block, described in
the [configuration guide](../config_guide.md):

- `boundary`, in raw units, goes to 0;
- `stimulus_range`, the raw minimum and maximum, go to −1 and +1;
- `stimulus_scale`, linear or log, is the scale on which the map is linear.

The map is piecewise linear, so an asymmetric range still puts the boundary at 0. For tones from 2 to 16 kHz with
the boundary at 8 kHz on a log scale, 2 kHz goes to −1, 8 kHz to 0 and 16 kHz to +1, and 4 kHz, halfway in
log units between 2 and 8, to −0.5. The defaults (range −1 to 1, boundary 0, linear) leave data that are already
on the axis untouched.

`TrialData.stimulus` holds the value on the axis and `TrialData.stimulus_raw` the value as read; a trial's
category comes from the value on the axis. A continuous task has no boundary: its stimulus and its response
share one linear map, with the range's midpoint at 0, so a response and its stimulus can be compared directly.
