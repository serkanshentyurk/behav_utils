# Contributing to behav_utils

## Set up

```bash
git clone https://github.com/serkanshentyurk/behav_utils.git && cd behav_utils
pip install -e ".[dev]"          # library + pytest + ruff
pytest tests -q
ruff check .
```

No data or config is needed for the test suite: fixtures build synthetic animals and register
test-local presets in `tests/conftest.py`.

## Building the docs

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

The site lands in `docs/_build/html`; open `index.html`. The build executes every notebook in `docs/tutorials/`,
and a failing cell, an internal link that does not resolve, or any other warning fails it. The docs workflow
(`.github/workflows/docs.yml`) runs the same build on every push and pull request and keeps the result for two
weeks as the run's `site` artifact; it also publishes main as `dev` and each release tag as its minor version
(see *Releasing*). Pages in `docs/` are MyST Markdown. The root
files EXTENDING, ARCHITECTURE, CONTRIBUTING, ROADMAP and CHANGELOG appear on the site through the pages of the same
name in `docs/`, which only include them, so edit the root files.

## Before opening a PR
- `ruff check src tests docs` is clean and `pytest tests -q` is green (CI runs both on 3.11 and 3.14, executes
  the tutorial notebooks, runs the code on every docs page, checks `docs/stats_reference.md` is regenerated, and
  builds the docs with warnings as errors).
- `python docs/run_doc_examples.py` passes. It runs each page of the site on its own, top to bottom, with nothing
  set up out of sight, and loads every YAML block on it as a config. So a page builds what its examples use in a
  visible block, and docstrings hold no code: they describe, and point to the page where an example runs.
- New public functions have a docstring saying what they return, and a test.
- A new statistic is registered via `@stat`/`@fit` with a docstring (it becomes the reference entry),
  tested, and `docs/stats_reference.md` is regenerated: `python docs/gen_stats_reference.py`.
- Anything that changes a number (a stat definition, the resampling engine, the psychometric fit)
  bumps the minor version and is called out in `CHANGELOG.md`; projects pin results to these versions.
- `tests/test_golden.py` stays green. A change that is meant to move numbers is marked ▲ in the CHANGELOG,
  and the golden values are regenerated on the reference machine: `python tests/golden/make_golden.py`.
- A new or changed public name or signature: put the name in one sub-package's `__all__` (and the
  top-level `__all__` if it is used often), then regenerate the list with `python docs/gen_public_api.py`;
  `tests/test_public_api.py` fails until you do. Helpers shared across modules start with `_`.

## Stability (from 1.0)
- Versions follow semantic versioning.
- Public means a name in `behav_utils.__all__` or in a sub-package's `__all__` (`docs/public_api.md`). A public name
  is renamed or removed only in a major release, and only after at least one minor release in which using it
  raises a `DeprecationWarning` naming its replacement. Adding a name, or an optional argument at the end, is a
  minor change.
- Names starting with `_`, and module paths below a sub-package, are private and can change in any release.
- A change that moves a number (marked ▲ in the CHANGELOG) arrives only in a minor or major release, never a patch,
  and the golden values are regenerated with it.
- A store's schema version rises when its layout changes; a newer behav_utils keeps reading older stores.

## Rules that reviews enforce
See [ARCHITECTURE.md](ARCHITECTURE.md) and `LLM_CONTEXT.md`: typed results, draw-only
plotters, no project vocabulary, downward-only imports, `exchangeable=False` on order-dependent stats.

## Docstrings

Google style, which the API pages render through napoleon, and `tests/test_docstrings.py` checks for the
sub-packages listed in its `DOCUMENTED`:

- The first line is a one-sentence summary ending in a full stop; `docs/public_api.md` quotes it.
- *Args* lists every parameter, one per line in signature order (`*args` and `**kwargs` included), with
  continuation lines indented. Types come from the annotations and are not repeated.
- *Returns* says what comes back, and *Raises* and *Warns* list what the function raises or warns on purpose.
  Avoid a colon in the first line of *Returns*: napoleon reads the text before it as the type.
- A dataclass lists every public field under *Attributes*; an enum lists its members there.
- No code: a docstring describes, and points to the docs page where an example runs. Cross-references use
  Sphinx roles (`:func:`, `:class:`).

## Releasing
When an item from `ROADMAP.md` ships, remove it there and record it in the CHANGELOG, so the two lists never
overlap.

Bump `__version__` in `src/behav_utils/__init__.py` (pyproject reads it), add the CHANGELOG entry, then tag
and push:

```bash
git tag -a vX.Y.Z -m "behav_utils X.Y.Z" && git push origin vX.Y.Z
```

`.github/workflows/release.yml` builds the sdist and wheel, checks them, installs the wheel in a clean
environment and runs the suite, and attaches both files to a GitHub release. The same tag publishes the docs
for its minor version (a later patch release replaces them), and the site then opens on that version; main is
always published as `dev`. GitHub Pages serves the `gh-pages` branch, which only the docs workflow writes. Install a release with
`pip install "behav_utils @ git+https://github.com/serkanshentyurk/behav_utils.git@vX.Y.Z"`.

## Maintenance

The canonical repository is `serkanshentyurk/behav_utils`; versions are tagged and released there.
Labs and projects that take the library on should **fork** it rather than copy it, keep the fork's
tags, and either open pull requests upstream (while the maintainer is active) or declare their fork
the live line in their project README and pin to its tags. Whichever repo a project pins is, for that
project, the version of record; never pin to a branch. A fork publishes its own docs once `Documentation` under
`[project.urls]` in pyproject.toml holds the fork's Pages address and, after its first push to main, the fork's
Pages settings serve the `gh-pages` branch.

## Known weak spots (what to look at next)

Honest list of the parts that have had the least validation against real data, in rough order of
how much a result would move if they are wrong. None is a known bug; each is a place to be sceptical.

- **`stats/dynamics.py` (`pse_dynamics`)** — a within-session exponential-vs-step fit of the PSE
  trajectory. Loosely identified below ~1000 trials at mouse-like sigma and sensitive to the starting
  window; `tau` beyond the block length means "did not converge", not a time constant. Report it with
  session-unit CIs across animals, and treat a single-animal `tau` as descriptive.
- **`stats/rt.py`** — the reaction-time statistics assume the CSV's RT column means the same thing
  across sessions and rigs (response onset vs. completion, timeout handling). The library cannot check
  that; the loader just maps the column. Verify the definition once per dataset before comparing RTs
  between conditions.
- **Trial-level bootstrap on history statistics** — `exchangeable=True` on the lag-1 stats is correct
  because `prev_*` is frozen, but stratification in the draw engine is by the current stimulus bin
  only. If a condition changes the *previous*-stimulus distribution, the trial bootstrap of `recency`
  can be narrower than it should be; the session unit is the safe comparison.
- **`SessionData.distribution`** — the majority label of the trials, which is right for the loader's
  per-trial distribution column but silently wrong for a session with a mid-session switch; use
  `find_switches` / `find_blocks` when blocks matter.
- **Stimulus normalisation at load** (new in 0.8.0) — piecewise-linear around the boundary. An
  asymmetric raw range therefore has different gain on the two sides of 0, which is what keeps the
  boundary at 0 and the ends at ±1, but means "0.3 from the boundary" is not the same raw distance on
  both sides. Fine for binning and the hard/easy split; think before interpreting `sigma` in raw units.
- **Neural alignment is deliberately absent.** `TrialArrays.trial_key` is the hook; per-trial binning
  of spikes or frames, sync tables and encoding models belong in the project. If a second project
  needs the same alignment code, that is the moment to add a sibling package, not before.
