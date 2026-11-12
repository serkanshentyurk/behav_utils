"""Golden values: every number pinned from 0.10.0 on a frozen dataset (see tests/golden/README.md)."""

from functools import lru_cache

import pytest

from .golden import outputs as golden
from .golden.continuous import outputs as golden_continuous


@lru_cache(maxsize=1)
def _actual() -> dict:
    names = golden.read_expected()['stats_pooled']['stat'].unique().tolist()
    return golden.compute_outputs(stat_names=names)


@pytest.mark.parametrize('table', golden.TABLES)
def test_golden_table(table):
    expected = golden.read_expected()[table]
    problems = golden.compare_table(table, expected, _actual()[table], golden.versions_match())
    assert not problems, '\n'.join(problems)


@lru_cache(maxsize=1)
def _actual_continuous() -> dict:
    return golden_continuous.compute_outputs()


@pytest.mark.parametrize('table', golden_continuous.TABLES)
def test_golden_continuous_table(table):
    """The continuous-response family, pinned from step 10 (closed-form, to 1e-12)."""
    expected = golden_continuous.read_expected()[table]
    problems = golden.compare_table(table, expected, _actual_continuous()[table], True)
    assert not problems, '\n'.join(problems)
