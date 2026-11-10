"""Write the frozen golden dataset to tests/golden/data (run once; the CSVs are committed).

Two animals, three daily sessions each, 400 trials per session. Choices come from a lapsing cumulative
Gaussian with a pull toward the previous choice and, on laser trials, a shift of the criterion. Session 1
has no laser; sessions 2 and 3 have laser on about 30 % of trials; session 3 draws its stimuli near the
boundary. About 4 % of trials are aborts (blank choice) and 3 % no-responses (choice -1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent / 'data'
SEED = 20261110
N_TRIALS = 400
DATES = ('2026_01_12', '2026_01_13', '2026_01_14')
ANIMALS = {
    'G01': dict(mu=0.05, sigma=0.25, lapse=0.04, history=0.30, laser_shift=0.15),
    'G02': dict(mu=-0.08, sigma=0.35, lapse=0.06, history=0.10, laser_shift=0.0),
}


def _session(rng: np.random.Generator, animal: str, p: dict, s: int) -> pd.DataFrame:
    n = N_TRIALS
    if s == 2:
        stim = np.clip(rng.normal(0.0, 0.3, n), -1.0, 1.0)
        dist = 'narrow'
    else:
        stim = rng.uniform(-1.0, 1.0, n)
        dist = 'flat'
    laser = (rng.random(n) < 0.3) if s > 0 else np.zeros(n, dtype=bool)
    abort = rng.random(n) < 0.04
    no_resp = ~abort & (rng.random(n) < 0.03)
    choice = np.full(n, np.nan)
    rt = np.full(n, np.nan)
    prev = np.nan
    for t in range(n):
        if abort[t]:
            continue
        if no_resp[t]:
            choice[t] = -1.0
            prev = np.nan
            continue
        pull = 0.0 if np.isnan(prev) else p['history'] * (2.0 * prev - 1.0)
        shift = p['laser_shift'] if laser[t] else 0.0
        p_b = p['lapse'] + (1.0 - 2.0 * p['lapse']) * norm.cdf((stim[t] - p['mu'] + shift) / p['sigma'] + pull)
        choice[t] = float(rng.random() < p_b)
        rt[t] = 0.25 + rng.lognormal(-1.2, 0.4)
        prev = choice[t]
    stim = np.round(stim, 4)                  # what the CSV stores; label the trials from that value
    category = (stim > 0).astype(float)
    responded = ~abort & ~no_resp
    correct = responded & (choice == category)
    outcome = np.where(abort, 'Abort', np.where(no_resp, 'NoResponse', np.where(correct, 'Correct', 'Incorrect')))
    return pd.DataFrame({
        'TrialNum': np.arange(1, n + 1),
        'Stim': stim,
        'Choice': choice,
        'RT': np.round(rt, 4),
        'Abort': abort,
        'Laser': laser,
        'Dist': dist,
        'Outcome': outcome,
        'Correct': correct,
        'Mouse': animal,
        'Stage': 'task',
    })


def main() -> None:
    rng = np.random.default_rng(SEED)
    for animal, params in ANIMALS.items():
        for s, date in enumerate(DATES):
            folder = ROOT / animal / f'golden_{animal}_{date}'
            folder.mkdir(parents=True, exist_ok=True)
            _session(rng, animal, params, s).to_csv(folder / 'trials.csv', index=False)
    print(f'wrote {ROOT}')


if __name__ == '__main__':
    main()
