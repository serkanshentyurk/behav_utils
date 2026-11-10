"""The public API: every listed name exists and has one home, private helpers stay unlisted, and
docs/public_api.md is current."""

import importlib
import importlib.metadata
import importlib.util
from pathlib import Path

import pytest

import behav_utils

PACKAGES = ('config', 'tasks', 'data', 'stats', 'readouts', 'analysis', 'plotting')
ROOT = Path(__file__).resolve().parents[1]


def _package(name):
    return importlib.import_module(f'behav_utils.{name}')


def _homes() -> dict:
    homes = {}
    for package in PACKAGES:
        for name in _package(package).__all__:
            homes.setdefault(name, []).append(package)
    return homes


def test_top_level_names_resolve_and_are_unique():
    names = behav_utils.__all__
    assert len(names) == len(set(names))
    assert [n for n in names if not hasattr(behav_utils, n)] == []


@pytest.mark.parametrize('package', PACKAGES)
def test_package_names_resolve_and_are_unique(package):
    module = _package(package)
    assert len(module.__all__) == len(set(module.__all__))
    assert [n for n in module.__all__ if not hasattr(module, n)] == []


def test_every_name_has_one_home():
    assert {name: homes for name, homes in _homes().items() if len(homes) > 1} == {}


def test_top_level_names_are_their_homes_objects():
    homes = _homes()
    assert [n for n in behav_utils.__all__ if n not in homes] == []
    for name in behav_utils.__all__:
        assert getattr(behav_utils, name) is getattr(_package(homes[name][0]), name), name


def test_no_private_names_are_listed():
    listed = list(behav_utils.__all__) + [n for p in PACKAGES for n in _package(p).__all__]
    assert [n for n in listed if n.startswith('_')] == []


@pytest.mark.parametrize('module, name', [
    ('behav_utils.analysis.phase', 'infer_animal_id'),
    ('behav_utils.analysis.comparison', 'contrast_key'),
    ('behav_utils.analysis.resampling', 'calculate_min_n'),
    ('behav_utils.stats.registry', 'validate_names'),
    ('behav_utils.stats.registry', 'producer_of'),
    ('behav_utils.stats.registry', 'list_producers'),
])
def test_internal_helpers_are_private(module, name):
    mod = importlib.import_module(module)
    assert not hasattr(mod, name) and hasattr(mod, '_' + name)


def test_public_api_doc_is_current():
    spec = importlib.util.spec_from_file_location('gen_public_api', ROOT / 'docs' / 'gen_public_api.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert (ROOT / 'docs' / 'public_api.md').read_text() == generator.render(), \
        'docs/public_api.md is out of date: run python docs/gen_public_api.py'


def test_version_matches_the_installed_metadata():
    try:
        installed = importlib.metadata.version('behav_utils')
    except importlib.metadata.PackageNotFoundError:
        pytest.skip('behav_utils is not installed')
    assert installed == behav_utils.__version__, 'reinstall: pip install -e ".[dev]"'
