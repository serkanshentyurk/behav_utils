# Contributing to behav_utils

## Set up

```bash
git clone <repo> && cd sound_categorisation/behav_utils
pip install -e ".[dev]"          # library + pytest + ruff
pytest tests -q
ruff check .
```

No data or config is needed for the test suite: fixtures build synthetic animals and register
test-local presets in `tests/conftest.py`.

## Before opening a PR
- `ruff check .` is clean and `pytest tests -q` is green (CI runs both on 3.10 and 3.11).
- New public functions have a docstring saying what they return, and a test.
- A new statistic is registered via `@stat`/`@fit` with a docstring (it becomes the reference entry),
  tested, and `docs/stats_reference.md` is regenerated: `python docs/gen_stats_reference.py`.
- Anything that changes a number (a stat definition, the resampling engine, the psychometric fit)
  bumps the minor version and is called out in `CHANGELOG.md`; projects pin results to these versions.

## Rules that reviews enforce
See [ARCHITECTURE.md](ARCHITECTURE.md) and [LLM_CONTEXT.md](LLM_CONTEXT.md): typed results, draw-only
plotters, no project vocabulary, downward-only imports, `exchangeable=False` on order-dependent stats.

## Releasing
Bump `version` in `pyproject.toml` and `__version__` in `src/behav_utils/__init__.py` together; tag
`behav_utils-vX.Y.Z`. Install from a tag with
`pip install "behav_utils @ git+https://github.com/<org>/sound_categorisation.git@behav_utils-vX.Y.Z#subdirectory=behav_utils"`.
