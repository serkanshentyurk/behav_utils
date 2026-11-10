"""Golden values: every number pinned from 0.10.0 on a frozen dataset (see tests/golden/README.md)."""

from functools import lru_cache

import pytest

from .golden import outputs as golden


@lru_cache(maxsize=1)
def _actual() -> dict:
    names = golden.read_expected()['stats_pooled']['stat'].unique().tolist()
    return golden.compute_outputs(stat_names=names)


@pytest.mark.parametrize('table', golden.TABLES)
def test_golden_table(table):
    expected = golden.read_expected()[table]
    problems = golden.compare_table(table, expected, _actual()[table], golden.versions_match())
    assert not problems, '\n'.join(problems)
