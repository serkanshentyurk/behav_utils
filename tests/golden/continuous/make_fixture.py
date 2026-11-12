"""
Write the frozen reproduction fixture under tests/golden/continuous/data. Run once; the CSVs are committed
and never regenerated. numpy only, so the fixture does not depend on the library's own generator.

Two animals x two sessions x 300 trials: a dot at Pos (cm on a 0-100 line), a poke at Poke. Responses pulled
towards the middle, biased, drawn towards the previous position (first animal only), with Gaussian noise,
about 3% aborts and 2% blank pokes. Everything stored is rounded first and nothing is derived from the
unrounded values.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    rng = np.random.default_rng(20260311)
    for animal, bias, slope, pull in (('C01', 3.0, 0.85, 0.12), ('C02', -2.0, 0.95, 0.0)):
        for day in (11, 12):
            n = 300
            pos = np.round(rng.uniform(0, 100, n), 3)
            previous = np.concatenate([[np.nan], pos[:-1]])
            drawn = np.where(np.isnan(previous), 0.0, pull * (previous - pos))
            poke = np.round(np.clip(50 + slope * (pos - 50) + bias + drawn + rng.normal(0, 4, n), 0, 100), 3)
            abort = rng.random(n) < 0.03
            blank = ~abort & (rng.random(n) < 0.02)
            poke[abort | blank] = np.nan
            folder = HERE / 'data' / animal / f'repro_{animal}_2026_03_{day:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'TrialNum': np.arange(1, n + 1), 'Pos': pos, 'Poke': poke, 'Abort': abort,
                          'Mouse': animal}).to_csv(folder / 'trials.csv', index=False)


if __name__ == '__main__':
    main()
