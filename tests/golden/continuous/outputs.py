"""
Golden values for the continuous-response family: its statistics (per animal pooled and per session) and
its three readouts on the frozen reproduction fixture. Pinned from step 10's own output, not from 0.10.0,
and compared to 1e-12 (all closed-form). ``make_golden.py --family continuous`` rewrites only these.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from behav_utils import (
    TrialArrays,
    compute_error_distribution,
    compute_response_curve,
    compute_serial_bias,
    compute_stats,
    load_experiment,
)

HERE = Path(__file__).resolve().parent
CONFIG = HERE / 'config.yaml'
EXPECTED_DIR = HERE / 'expected'
STATS = ['accuracy', 'response_bias', 'error_sd', 'mean_abs_error', 'response_slope', 'serial_bias']
TABLES = ('stats', 'response_curve', 'error_distribution', 'serial_bias_curve')


def compute_outputs() -> dict:
    experiment = load_experiment(CONFIG)
    stats, curves, errors, serial = [], [], [], []
    for animal_id in sorted(experiment.animals):
        sessions = experiment.animals[animal_id].sessions
        pooled = TrialArrays.from_sessions(sessions)
        blocks = [('pooled', pooled)] + [(s.session_id, TrialArrays.from_sessions([s])) for s in sessions]
        for label, arrays in blocks:
            values = compute_stats(arrays, STATS, rng=np.random.default_rng(0))
            stats.append(pd.DataFrame({'animal': animal_id, 'block': label, 'stat': values.index,
                                       'value': values.to_numpy()}))
        curves.append(compute_response_curve(pooled).to_rows().assign(animal=animal_id))
        errors.append(compute_error_distribution(pooled).to_rows().assign(animal=animal_id))
        serial.append(compute_serial_bias(pooled).to_rows().assign(animal=animal_id))
    return {'stats': pd.concat(stats, ignore_index=True),
            'response_curve': pd.concat(curves, ignore_index=True),
            'error_distribution': pd.concat(errors, ignore_index=True),
            'serial_bias_curve': pd.concat(serial, ignore_index=True)}


def write_expected() -> None:
    EXPECTED_DIR.mkdir(exist_ok=True)
    for name, table in compute_outputs().items():
        table.to_csv(EXPECTED_DIR / f'{name}.csv', index=False, float_format='%.17g')


def read_expected() -> dict:
    return {name: pd.read_csv(EXPECTED_DIR / f'{name}.csv') for name in TABLES}
