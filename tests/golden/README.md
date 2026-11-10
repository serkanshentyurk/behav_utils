# Golden values

Every number the library produces on one small, frozen dataset, pinned from 0.10.0. `tests/test_golden.py`
recomputes them and fails when one moves, so a refactor that should change nothing is checked against
the numbers themselves, not only against the unit tests.

- `data/` — two animals × three sessions × 400 trials in the on-disk layout the loader reads, with laser,
  abort and no-response trials. Written once by `make_fixture.py` and committed; never regenerated (numpy
  does not promise identical random streams across versions).
- `config.yaml` — the config that loads it; `data_dir` is relative to the config file.
- `outputs.py` — the one computation: all statistics pooled per animal and trial type, per-session
  statistics, the five readouts, a laser-vs-off contrast (bootstrap and permutation, fixed seed) and rolling
  statistics.
- `expected/` — its output, written by `make_golden.py`, with the numpy, scipy and pandas versions used.

Tolerances: closed-form values must agree to 1e-12. Values that come out of an optimiser (psychometric and
history fits, PSE dynamics, the fitted readouts) agree to a relative 1e-6 when the numpy, scipy and pandas
versions match `expected/versions.json`, and to a relative 1e-3 (an absolute 1e-4 near zero) when they differ:
an optimiser can stop at a slightly different point on another build, most of all for a poorly determined
parameter such as `pse_tau`.

Regenerate `expected/` only for a change that is meant to move numbers, marked ▲ in the changelog:

```bash
python tests/golden/make_golden.py
```
