"""The store: experiments saved as Parquet and JSON, and reloaded exactly, all of it or in part.

Every test that touches files runs twice: through a stand-in for the store's two Parquet functions, so the
store's own logic is tested anywhere, and through real Parquet, which is skipped only where pyarrow is
not installed (it is in the dev extra, so CI runs it)."""

import dataclasses
import json
import math
import re
import sys
from datetime import date
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

import behav_utils.data.store as store
from behav_utils import (
    PsychometricModel,
    Reproduction,
    ReproductionModel,
    TrialArrays,
    TwoAFC,
    discover_sessions,
    generate_synthetic_animal,
    load_experiment,
    load_store,
    read_manifest,
    save_store,
)
from behav_utils.data import ExperimentData
from behav_utils.data.store import StoreError
from behav_utils.data.structures import SESSION_TABLE_COLUMNS, TrialData
from behav_utils.tasks import ResponseSpace, register_task
from tests.golden import outputs as golden
from tests.golden.continuous import outputs as golden_continuous

IO = ['stand-in', 'parquet']


def _use(io, monkeypatch):
    if io == 'parquet':
        pytest.importorskip('pyarrow')
    else:
        monkeypatch.setattr(store, '_write_table', lambda table, path: table.to_pickle(path))
        monkeypatch.setattr(store, '_read_table', lambda path: pd.read_pickle(path))


def _nan(v) -> bool:
    return v is None or (isinstance(v, (float, np.floating)) and math.isnan(v))


def _same_value(x, y) -> bool:
    return (_nan(x) and _nan(y)) or x == y


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in 'biuf' and b.dtype.kind in 'biuf':
        return np.array_equal(a.astype(float), b.astype(float), equal_nan=True)
    return all(_same_value(x, y) for x, y in zip(a.astype(object), b.astype(object)))


def _same_mapping(a, b) -> bool:
    return sorted(a) == sorted(b) and all(_same_value(a[k], b[k]) for k in a)


def assert_same_experiment(original, reloaded, *, as_text=()):
    """Every field of every session, the rebuilt lag-1 view included; ``as_text`` names extra columns
    expected back as text."""
    assert sorted(original.animals) == sorted(reloaded.animals)
    assert _same_mapping(original.metadata, reloaded.metadata)
    for animal_id, animal in original.animals.items():
        other = reloaded.animals[animal_id]
        assert _same_mapping(animal.metadata, other.metadata), animal_id
        assert [s.session_id for s in other.sessions] == [s.session_id for s in animal.sessions]
        for mine, theirs in zip(animal.sessions, other.sessions):
            for name in ('session_id', 'session_idx', 'date', 'start', 'session_type', 'session_type_source',
                         'source_id', 'experimenter', 'task', 'task_name', 'clock', 'csv_path'):
                assert getattr(theirs, name) == getattr(mine, name), (mine.session_id, name)
            assert _same_mapping(mine.metadata.fields, theirs.metadata.fields), mine.session_id
            for f in dataclasses.fields(TrialData):
                if f.metadata.get('per_trial'):
                    assert _same(getattr(mine.trials, f.name), getattr(theirs.trials, f.name)), (mine.session_id, f.name)
                elif f.metadata.get('per_trial_map'):
                    a, b = getattr(mine.trials, f.name), getattr(theirs.trials, f.name)
                    assert sorted(a) == sorted(b), (mine.session_id, f.name)
                    for key in a:
                        expected = a[key]
                        if key in as_text:
                            expected = np.array([np.nan if _nan(v) else str(v) for v in expected], dtype=object)
                        assert _same(expected, b[key]), (mine.session_id, f.name, key)


# ── the golden experiments ───────────────────────────────────────────────────

@pytest.mark.parametrize('io', IO)
def test_the_golden_2afc_experiment_round_trips_to_its_pinned_values(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    original = golden.load_golden_experiment()
    reloaded = load_store(save_store(original, tmp_path / 'store'))
    assert_same_experiment(original, reloaded)
    expected = golden.read_expected()
    actual = golden.compute_outputs(stat_names=expected['stats_pooled']['stat'].unique().tolist(), experiment=reloaded)
    for table in golden.TABLES:
        assert not golden.compare_table(table, expected[table], actual[table], golden.versions_match()), table


@pytest.mark.parametrize('io', IO)
def test_the_golden_continuous_experiment_round_trips_to_its_pinned_values(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    original = load_experiment(golden_continuous.CONFIG)
    reloaded = load_store(save_store(original, tmp_path / 'store'))
    assert_same_experiment(original, reloaded)
    expected, actual = golden_continuous.read_expected(), golden_continuous.compute_outputs(reloaded)
    for table in golden_continuous.TABLES:
        assert not golden.compare_table(table, expected[table], actual[table], True), table


# ── an awkward experiment, field by field ────────────────────────────────────

_RICH = """
project: {name: rich, experimenter: SS}
reader: {type: csv_folders, data_dir: data, behaviour_file: trials.csv, drop_last_row: false, min_trials_per_file: 5}
task:
  primary_stimulus: stimulus
  primary_choice: choice
  choice_mapping: {type: identity, no_response_value: -1}
tasks:
  pretraining: {type: go_nogo}
task_by: {field: stage, values: {GNG: pretraining}}
session_types:
  sham:
    R1: ['20260203']
scoring: {on_mismatch: ignore}
columns:
  trial_number: {csv_name: Trial, dtype: int}
  stimulus: {csv_name: Stim, dtype: float}
  choice: {csv_name: Choice, dtype: float}
  correct: {csv_name: Correct, dtype: bool}
  outcome: {csv_name: Outcome, dtype: str}
  abort: {csv_name: Abort, dtype: bool}
  opto_on: {csv_name: Laser, dtype: bool}
  reaction_time: {csv_name: RT, dtype: float}
  t_trial_start: {csv_name: Start, dtype: float}
  contrast: {csv_name: Contrast, dtype: float, optional: true}
session_metadata:
  stage: {csv_name: Stage, dtype: str}
  block: {csv_name: Block, dtype: int}
  gain: {csv_name: Gain, dtype: float}
  flag: {csv_name: Flag, dtype: bool}
"""


def _rich_experiment(root):
    rng = np.random.default_rng(7)
    sessions = [('R1', '2026_02_02', 'task', 1.5), ('R1', '2026_02_03', 'task', np.nan), ('R1', '2026_02_04', 'GNG', 2.0),
                ('R2', '2026_02_02', 'task', 0.5)]
    for k, (animal, day, stage, gain) in enumerate(sessions):
        n = 60
        table = pd.DataFrame({
            'Trial': range(1, n + 1), 'Stim': rng.uniform(-1, 1, n).round(4),
            'Choice': rng.choice([0, 1, 1, 0, -1], n), 'Correct': rng.random(n) < 0.7,
            'Outcome': rng.choice(['Correct', 'Incorrect', 'Abort'], n), 'Abort': rng.random(n) < 0.05,
            'Laser': rng.random(n) < 0.3, 'RT': rng.uniform(0.2, 0.9, n).round(3),
            'Start': (1000.0 + 4.0 * np.arange(n)).round(2), 'Stage': stage, 'Block': k + 1, 'Gain': gain,
            'Flag': k % 2 == 0, 'Note': rng.choice(['ok', '', 'late'], n),
        })
        if animal == 'R1':
            table['Contrast'] = rng.choice([0.25, 0.5, 1.0], n)
        folder = root / 'data' / animal / f'S_{day}'
        folder.mkdir(parents=True)
        table.to_csv(folder / 'trials.csv', index=False)
    (root / 'data' / 'animal_metadata.json').write_text(json.dumps({'R1': {'group': 'control', 'weight': 21.5},
                                                                     'R2': {'group': 'treated'}}))
    (root / 'config.yaml').write_text(_RICH)
    experiment = load_experiment(root / 'config.yaml')
    experiment.add_animal(generate_synthetic_animal(ReproductionModel(noise=0.05), n_sessions=2, trials_per_session=80,
                                                    task=Reproduction(tolerance=0.05), laser_fraction=0.3,
                                                    animal_id='S1', metadata={'group': 'synthetic'}, rng=1))
    experiment.add_animal(generate_synthetic_animal(PsychometricModel(), n_sessions=2, trials_per_session=80,
                                                    task=TwoAFC(b_side='negative', on_boundary='unscored'),
                                                    animal_id='S2', rng=2))
    experiment.metadata['cohort'] = 'C1'
    mixed = np.array([1, 'x', 2.5, None] * 15, dtype=object)
    experiment.animals['R1'].sessions[0].trials.extra['Mixed'] = mixed
    return experiment


@pytest.mark.parametrize('io', IO)
def test_an_awkward_experiment_round_trips_field_by_field(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    original = _rich_experiment(tmp_path)
    sessions = {s.session_id: s for a in original.animals.values() for s in a.sessions}
    assert sessions['R1_20260203'].session_type == 'sham' and sessions['R1_20260204'].task_name == 'pretraining'
    reloaded = load_store(save_store(original, tmp_path / 'store'))
    assert_same_experiment(original, reloaded, as_text=('Mixed',))
    assert _same(reloaded.animals['R1'].sessions[0].trials.extra['Mixed'][:4],
                 np.array(['1', 'x', '2.5', np.nan], dtype=object))


def test_loaded_text_columns_are_numpy_arrays_that_follow_filtering(tmp_path):
    from behav_utils.data import filter_trial_data

    trials = _rich_experiment(tmp_path).animals['R1'].sessions[0].trials
    for mapping in (trials.optional_fields, trials.extra, trials.events):
        assert all(isinstance(values, np.ndarray) for values in mapping.values())
    keep = np.arange(trials.n_trials) % 2 == 0
    kept = filter_trial_data(trials, keep)
    assert len(kept.extra['Note']) == keep.sum() and list(kept.extra['Note']) == list(trials.extra['Note'][keep])


@pytest.mark.parametrize('io', IO)
def test_keys_scores_and_tasks_survive(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    original = _rich_experiment(tmp_path)
    reloaded = load_store(save_store(original, tmp_path / 'store'))
    for animal_id in ('R1', 'S1', 'S2'):
        for mine, theirs in zip(original.animals[animal_id].sessions, reloaded.animals[animal_id].sessions):
            a, b = TrialArrays.from_sessions([mine]), TrialArrays.from_sessions([theirs])
            for name in ('trial_key', 'correct', 'prev_correct', 'truth', 'choice', 'prev_choice'):
                assert _same(getattr(a, name), getattr(b, name)), (animal_id, name)
            assert b.task == a.task and hash(b.task) == hash(a.task)


# ── the manifest ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('io', IO)
def test_the_manifest_is_the_session_table(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    original = _rich_experiment(tmp_path)
    save_store(original, tmp_path / 'store')
    manifest = read_manifest(tmp_path / 'store')
    table = original.session_table
    assert list(manifest.columns) == SESSION_TABLE_COLUMNS + ['task_name', 'task_spec', 'metadata_json', 'clock',
                                                              'trials_file']
    for column in SESSION_TABLE_COLUMNS:
        assert _same(manifest[column].to_numpy(), table[column].to_numpy()), column
    assert all(isinstance(d, date) for d in manifest['date'])
    assert json.loads(manifest['task_spec'].iloc[-1])['type'] == 'two_afc'
    assert all((tmp_path / 'store' / f).is_file() for f in manifest['trials_file'])
    info = json.loads((tmp_path / 'store' / 'store.json').read_text())
    assert info['schema_version'] == 1 and info['n_sessions'] == len(table) and info['n_animals'] == 4


@pytest.mark.parametrize('io', IO)
def test_the_manifest_and_discovery_share_the_identity_columns(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    save_store(golden.load_golden_experiment(), tmp_path / 'store')
    manifest, discovered = read_manifest(tmp_path / 'store'), discover_sessions(golden.CONFIG)
    for column in ('session_id', 'animal_id', 'experimenter', 'date', 'start', 'source_id', 'source'):
        assert _same(manifest[column].to_numpy(), discovered[column].to_numpy()), column


# ── loading a part ───────────────────────────────────────────────────────────

@pytest.mark.parametrize('io', IO)
def test_a_filtered_manifest_loads_exactly_those_sessions(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    original = golden.load_golden_experiment()
    save_store(original, tmp_path / 'store')
    manifest = read_manifest(tmp_path / 'store')
    chosen = manifest[(manifest.animal_id == 'G02') & (manifest.date > date(2026, 1, 12))]
    part = load_store(tmp_path / 'store', sessions=chosen)
    assert list(part.session_table['session_id']) == list(chosen['session_id'])
    assert [s.session_idx for s in part.animals['G02'].sessions] == list(range(len(chosen)))
    full = {s.session_id: s for s in original.animals['G02'].sessions}
    for session in part.animals['G02'].sessions:
        assert _same(TrialArrays.from_sessions([session]).trial_key,
                     TrialArrays.from_sessions([full[session.session_id]]).trial_key)


@pytest.mark.parametrize('io', IO)
def test_a_partial_load_opens_only_the_chosen_files(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    save_store(golden.load_golden_experiment(), tmp_path / 'store')
    manifest = read_manifest(tmp_path / 'store')
    keep = manifest['session_id'].iloc[0]
    for row in manifest.itertuples():
        if row.session_id != keep:
            (tmp_path / 'store' / row.trials_file).unlink()
    assert load_store(tmp_path / 'store', sessions=[keep]).session_table['session_id'].tolist() == [keep]
    with pytest.raises(StoreError, match='is missing; the store is incomplete'):
        load_store(tmp_path / 'store')


@pytest.mark.parametrize('io', IO)
def test_an_unknown_session_raises_when_strict_and_warns_otherwise(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    save_store(golden.load_golden_experiment(), tmp_path / 'store')
    with pytest.raises(StoreError, match='G09_20260101'):
        load_store(tmp_path / 'store', sessions=['G09_20260101'])
    with pytest.warns(UserWarning, match='G09_20260101'):
        part = load_store(tmp_path / 'store', sessions=['G09_20260101', 'G01_20260112'], strict=False)
    assert part.session_table['session_id'].tolist() == ['G01_20260112']


# ── writing safely ───────────────────────────────────────────────────────────

@pytest.mark.parametrize('io', IO)
def test_an_existing_store_is_replaced_only_when_asked(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    path = tmp_path / 'store'
    save_store(golden.load_golden_experiment(), path)
    smaller = load_store(path, sessions=['G01_20260112'])
    with pytest.raises(StoreError, match='overwrite=True'):
        save_store(smaller, path)
    save_store(smaller, path, overwrite=True)
    assert read_manifest(path)['session_id'].tolist() == ['G01_20260112']


@pytest.mark.parametrize('io', IO)
def test_a_directory_that_is_not_a_store_is_never_touched(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    path = tmp_path / 'notes'
    path.mkdir()
    (path / 'keep.txt').write_text('mine')
    for overwrite in (False, True):
        with pytest.raises(StoreError, match='not a behav_utils store'):
            save_store(golden.load_golden_experiment(), path, overwrite=overwrite)
    assert (path / 'keep.txt').read_text() == 'mine'


@pytest.mark.parametrize('io', IO)
def test_a_failed_write_leaves_nothing_and_an_old_store_intact(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    experiment = golden.load_golden_experiment()
    path = tmp_path / 'store'
    save_store(experiment, path)
    before = read_manifest(path)['session_id'].tolist()
    real_write, calls = store._write_table, {'n': 0}

    def failing(table, file):
        calls['n'] += 1
        if calls['n'] == 3:
            raise RuntimeError('disk full')
        real_write(table, file)

    monkeypatch.setattr(store, '_write_table', failing)
    with pytest.raises(RuntimeError, match='disk full'):
        save_store(experiment, tmp_path / 'fresh')
    calls['n'] = 0
    with pytest.raises(RuntimeError, match='disk full'):
        save_store(experiment, path, overwrite=True)
    assert not (tmp_path / 'fresh').exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith('.')]
    monkeypatch.setattr(store, '_write_table', real_write)
    assert read_manifest(path)['session_id'].tolist() == before


# ── versions, names, edge cases ──────────────────────────────────────────────

@pytest.mark.parametrize('io', IO)
def test_versions_and_formats_are_checked(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    path = save_store(golden.load_golden_experiment(), tmp_path / 'store')
    info = json.loads((path / 'store.json').read_text())
    (path / 'store.json').write_text(json.dumps({**info, 'schema_version': 99}))
    with pytest.raises(StoreError, match='upgrade behav_utils'):
        load_store(path)
    (path / 'store.json').write_text(json.dumps({**info, 'format': 'something else'}))
    with pytest.raises(StoreError, match='not a behav_utils store'):
        read_manifest(path)
    (path / 'store.json').unlink()
    with pytest.raises(StoreError, match='no store.json'):
        load_store(path)


@pytest.mark.parametrize('io', IO)
def test_awkward_ids_get_safe_unique_file_names(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    experiment = ExperimentData()
    for animal_id in ('A/1 x', 'A_1_x'):
        experiment.add_animal(generate_synthetic_animal(n_sessions=1, trials_per_session=30, animal_id=animal_id, rng=3))
    path = save_store(experiment, tmp_path / 'store')
    files = read_manifest(path)['trials_file'].tolist()
    assert len(set(files)) == 2 and all('/' not in f[len('trials/'):] and ' ' not in f for f in files)
    assert_same_experiment(experiment, load_store(path))


@pytest.mark.parametrize('io', IO)
def test_an_empty_experiment_round_trips(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    reloaded = load_store(save_store(ExperimentData(), tmp_path / 'store'))
    assert reloaded.animals == {} and len(read_manifest(tmp_path / 'store')) == 0


@register_task
@dataclasses.dataclass(frozen=True)
class Targets(TwoAFC):
    """A 2-AFC with a tuple setting, to check tasks come back hashable."""
    name: ClassVar[str] = 'targets_for_store_tests'
    targets: tuple = (0.25, 0.75)

    @property
    def response_space(self):
        return ResponseSpace('binary', ('A', 'B'))


@pytest.mark.parametrize('io', IO)
def test_a_custom_task_with_a_tuple_setting_comes_back_equal_and_hashable(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    experiment = ExperimentData()
    experiment.add_animal(generate_synthetic_animal(n_sessions=1, trials_per_session=30, task=Targets(), rng=4))
    task = load_store(save_store(experiment, tmp_path / 'store')).animals['SYN01'].sessions[0].task
    assert task == Targets() and hash(task) == hash(Targets())


def test_without_pyarrow_the_store_says_how_to_get_it(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'pyarrow', None)
    with pytest.raises(StoreError, match=r"pip install 'behav_utils\[store\]'"):
        save_store(golden.load_golden_experiment(), tmp_path / 'store')
    assert not (tmp_path / 'store').exists()


# ── the save and load methods ────────────────────────────────────────────────

@pytest.mark.parametrize('io', IO)
def test_experiment_save_and_load_are_the_store(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    original = _rich_experiment(tmp_path)
    path = original.save(tmp_path / 'store')
    assert (path / 'store.json').is_file()
    assert_same_experiment(original, ExperimentData.load(path), as_text=('Mixed',))
    with pytest.raises(StoreError, match='overwrite=True'):
        original.save(path)
    original.save(path, overwrite=True)
    part = ExperimentData.load(path, sessions=['R1_20260202', 'S1_20260105'])
    assert sorted(part.session_table['session_id']) == ['R1_20260202', 'S1_20260105']


@pytest.mark.parametrize('io', IO)
def test_animal_save_and_load_are_the_store(io, tmp_path, monkeypatch):
    _use(io, monkeypatch)
    from behav_utils.data import AnimalData

    experiment = _rich_experiment(tmp_path)
    single = experiment.animals['S1']
    assert_same_experiment(_wrap(single), _wrap(AnimalData.load(single.save(tmp_path / 'one'))))
    path = experiment.save(tmp_path / 'all')
    with pytest.raises(StoreError, match='pass animal_id'):
        AnimalData.load(path)
    with pytest.raises(StoreError, match="no animal 'R9'"):
        AnimalData.load(path, animal_id='R9')
    for row in read_manifest(path).itertuples():
        if row.animal_id != 'R2':
            (path / row.trials_file).unlink()          # loading R2 must not need anyone else's files
    assert_same_experiment(_wrap(experiment.animals['R2']), _wrap(AnimalData.load(path, animal_id='R2')))


def _wrap(animal):
    experiment = ExperimentData()
    experiment.add_animal(animal)
    return experiment


def test_a_pickle_snapshot_is_refused_with_directions(tmp_path):
    from behav_utils.data import AnimalData

    old = tmp_path / 'experiment.pkl'
    old.write_bytes(b'\x80\x04old pickle')
    for load in (ExperimentData.load, AnimalData.load, load_store, read_manifest):
        with pytest.raises(StoreError, match='pickle snapshots were removed in 1.0'):
            load(old)
    with pytest.raises(StoreError, match='a store is a directory'):
        ExperimentData().save(tmp_path / 'new.pkl')
    assert not (tmp_path / 'new.pkl').exists()


def test_pickle_is_imported_nowhere_in_the_library():
    import behav_utils

    root = Path(behav_utils.__file__).parent
    offenders = [str(f.relative_to(root)) for f in root.rglob('*.py')
                 if re.search(r'^\s*(import pickle|from pickle import)|pickle\.(dump|load)', f.read_text(), re.M)]
    assert offenders == []
