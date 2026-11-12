"""The golden computation and its comparison (see tests/golden/README.md).

``compute_outputs`` runs one fixed pipeline over the frozen dataset and returns one table per output;
``write_expected`` stores them in ``expected/``; ``compare_table`` checks a recomputed table against the
stored one. Only public names are used, so the computation survives refactors of the internals.
"""

from __future__ import annotations

import json
import platform
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / 'data'
CONFIG = HERE / 'config.yaml'
EXPECTED_DIR = HERE / 'expected'
VERSIONS_FILE = EXPECTED_DIR / 'versions.json'

TABLES = (
    'stats_pooled', 'stats_sessions',
    'readout_psychometric', 'readout_update_matrix', 'readout_conditional_psychometric',
    'readout_binned', 'readout_sd_profile',
    'delta', 'rolling',
)
TRIAL_TYPES = ('all', 'non_opto', 'opto')
DELTA_STATS = ('accuracy', 'side_bias', 'mu', 'sigma', 'win_stay', 'lose_shift')
ROLLING_STATS = ('accuracy', 'side_bias')
FITTED_READOUTS = frozenset({'readout_psychometric', 'readout_update_matrix', 'readout_conditional_psychometric'})
CLOSED_RTOL, CLOSED_ATOL = 1e-12, 1e-12
FITTED_RTOL_SAME = 1e-6
# Across numpy/scipy builds an optimiser can stop at a slightly different point, most visibly for
# poorly determined parameters (pse_tau moved by 7e-5 relative between a Mac and CI). A relative
# tolerance scales with the value; the absolute floor covers values near zero.
FITTED_RTOL_OTHER, FITTED_ATOL_OTHER = 1e-3, 1e-4


def fitted_stats() -> frozenset:
    """Statistics that come out of an optimiser."""
    from behav_utils.stats import LOGISTIC_HISTORY, PSE_DYNAMICS, PSYCHOMETRIC, SD_PROFILE
    return frozenset((*PSYCHOMETRIC, *LOGISTIC_HISTORY, *PSE_DYNAMICS, *SD_PROFILE,
                      'pse', 'psychometric_gof', 'history_interaction_r2'))


def load_golden_experiment():
    from behav_utils import load_experiment
    return load_experiment(CONFIG)


def _rows(df: pd.DataFrame, **labels) -> pd.DataFrame:
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df.copy()
    for i, (k, v) in enumerate((k, v) for k, v in labels.items() if k not in df.columns):
        df.insert(i, k, v)
    return df


def compute_outputs(stat_names=None, experiment=None) -> dict:
    """Every golden table, recomputed with the current code (on ``experiment`` if given, else on the
    golden dataset loaded with ``config.yaml``)."""
    from behav_utils import (
        TrialArrays,
        compute_binned_curve,
        compute_conditional_psychometric,
        compute_delta_stat,
        compute_phase_stats,
        compute_psychometric_curve,
        compute_rolling_stats,
        compute_sd_profile,
        compute_stats,
        compute_update_matrix,
        filter_trials,
        list_stats,
        select_sessions,
    )
    names = list(stat_names) if stat_names is not None else list(list_stats())
    experiment = experiment if experiment is not None else load_golden_experiment()
    out = {t: [] for t in TABLES}
    for animal in experiment.animal_ids:
        sessions = select_sessions(experiment.animals[animal])
        phases = {tt: filter_trials(sessions, trial_type=tt) for tt in TRIAL_TYPES}
        for tt, phase in phases.items():
            s = compute_stats(TrialArrays.from_sessions(phase), names, rng=np.random.default_rng(0), strict=False)
            out['stats_pooled'].append(pd.DataFrame({'animal': animal, 'trial_type': tt,
                                                     'stat': s.index, 'value': s.to_numpy()}))
        per_session = compute_phase_stats(phases['non_opto'], names, per_session=True, animal_id=animal,
                                          rng=np.random.default_rng(0))
        out['stats_sessions'].append(pd.concat([per_session.to_rows(), per_session.sessions], ignore_index=True))
        arrays = TrialArrays.from_sessions(phases['non_opto'])
        readouts = {
            'readout_psychometric': compute_psychometric_curve(arrays, n_bootstrap=0),
            'readout_update_matrix': compute_update_matrix(arrays),
            'readout_conditional_psychometric': compute_conditional_psychometric(arrays),
            'readout_binned': compute_binned_curve(arrays),
            'readout_sd_profile': compute_sd_profile(arrays),
        }
        for table, result in readouts.items():
            out[table].append(_rows(result.to_rows(), animal=animal))
        delta = compute_delta_stat({'off': phases['non_opto'], 'on': phases['opto']}, list(DELTA_STATS),
                                   reference='off', curve=False, n_bootstrap=100, units=('trials', 'sessions'),
                                   n_permutations=100, seed=7)
        contrast = delta.contrast('on')
        for unit in contrast.units:
            out['delta'].append(_rows(contrast.table(unit), animal=animal, unit=unit))
        rolling = compute_rolling_stats(phases['non_opto'], list(ROLLING_STATS))
        out['rolling'].append(_rows(rolling.curves, animal=animal))
    return {t: pd.concat(frames, ignore_index=True) for t, frames in out.items()}


def library_versions() -> dict:
    import scipy
    return {'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__}


def write_expected() -> None:
    import behav_utils
    tables = compute_outputs()
    EXPECTED_DIR.mkdir(exist_ok=True)
    for name, df in tables.items():
        df.to_csv(EXPECTED_DIR / f'{name}.csv', index=False, float_format='%.17g')
    meta = {**library_versions(), 'python': platform.python_version(),
            'behav_utils': behav_utils.__version__}
    VERSIONS_FILE.write_text(json.dumps(meta, indent=2) + '\n')
    print(f'wrote {len(tables)} tables to {EXPECTED_DIR} ({meta})')


@lru_cache(maxsize=1)
def read_expected() -> dict:
    if not EXPECTED_DIR.exists():
        raise FileNotFoundError('no golden values: run python tests/golden/make_golden.py (see tests/golden/README.md)')
    return {t: pd.read_csv(EXPECTED_DIR / f'{t}.csv') for t in TABLES}


def versions_match() -> bool:
    recorded = json.loads(VERSIONS_FILE.read_text())
    return all(recorded.get(k) == v for k, v in library_versions().items())


def _fitted_rows(table: str, df: pd.DataFrame) -> np.ndarray:
    if table in FITTED_READOUTS:
        return np.ones(len(df), dtype=bool)
    if 'stat' in df.columns:
        return df['stat'].astype(str).isin(fitted_stats()).to_numpy()
    return np.zeros(len(df), dtype=bool)


def compare_table(table: str, expected: pd.DataFrame, actual: pd.DataFrame, same_versions: bool) -> list:
    """Differences between a stored and a recomputed table, as readable lines (empty = identical)."""
    if list(expected.columns) != list(actual.columns):
        return [f'{table}: columns {list(actual.columns)} != expected {list(expected.columns)}']
    if len(expected) != len(actual):
        return [f'{table}: {len(actual)} rows, expected {len(expected)}']
    fitted = _fitted_rows(table, expected)
    problems = []
    for col in expected.columns:
        e, a = expected[col], actual[col].reset_index(drop=True)
        if pd.api.types.is_float_dtype(e) or pd.api.types.is_float_dtype(a):
            ev = pd.to_numeric(e, errors='coerce').to_numpy(dtype=float)
            av = pd.to_numeric(a, errors='coerce').to_numpy(dtype=float)
            closed = np.isclose(av, ev, rtol=CLOSED_RTOL, atol=CLOSED_ATOL, equal_nan=True)
            if same_versions:
                fit = np.isclose(av, ev, rtol=FITTED_RTOL_SAME, atol=CLOSED_ATOL, equal_nan=True)
            else:
                fit = np.isclose(av, ev, rtol=FITTED_RTOL_OTHER, atol=FITTED_ATOL_OTHER, equal_nan=True)
            bad = np.flatnonzero(~np.where(fitted, fit, closed))
        else:
            ev = e.fillna('').astype(str).to_numpy()
            av = a.fillna('').astype(str).to_numpy()
            bad = np.flatnonzero(ev != av)
        problems += [f'{table} row {i} {col}: {av[i]!r} != expected {ev[i]!r}' for i in bad[:5]]
    return problems
