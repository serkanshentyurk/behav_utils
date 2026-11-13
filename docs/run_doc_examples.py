"""Execute every ```python block in docs/data_structures_reference.md on synthetic data.

    python docs/run_doc_examples.py

Fails loudly on the first block that no longer matches the API, so the reference cannot drift.
A block that needs real files on disk (``load_experiment('config.yaml')``) is run against a
temporary tree written by the prelude.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

DOC = Path(__file__).with_name('data_structures_reference.md')


def _prelude() -> dict:
    import numpy as np
    import pandas as pd
    import yaml

    from behav_utils import (
        PsychometricModel,
        SessionFilter,
        filter_trials,
        generate_synthetic_experiment,
        register_preset,
        select_sessions,
    )

    register_preset('expert', SessionFilter(stage='task', last_fraction=0.5))
    groups = [('A05', 'control'), ('A06', 'control'), ('A07', 'control'),
              ('A08', 'treated'), ('A09', 'treated'), ('A10', 'treated')]
    experiment = generate_synthetic_experiment(
        {aid: {'model': PsychometricModel(sigma=0.3 if grp == 'control' else 0.5), 'n_sessions': 6,
               'trials_per_session': 200, 'metadata': {'group': grp},
               'laser_fraction': 0.33 if aid == 'A05' else 0.0}          # opto trials for the first animal
         for aid, grp in groups}, rng=0)
    animal = experiment.get_animal('A05')
    sessions = select_sessions(animal, stage='task')
    clean = filter_trials(sessions, trial_type='non_opto')
    clean_uniform = clean[:3]
    clean_hard = clean[3:]

    from behav_utils import compute_phase_stats
    from behav_utils.analysis import collect_rows
    rows_on, rows_off = [], []
    for a in experiment.animals.values():
        on = filter_trials(select_sessions(a, stage='task'), trial_type='all')
        off = filter_trials(select_sessions(a, stage='task'), trial_type='non_opto')
        rows_on += collect_rows(compute_phase_stats(on, ['mu']).to_rows().to_dict('records'),
                                animal=a.animal_id, condition='on')
        rows_off += collect_rows(compute_phase_stats(off, ['mu']).to_rows().to_dict('records'),
                                 animal=a.animal_id, condition='off')

    # a minimal real config + data tree for the load_experiment block
    tmp = Path(tempfile.mkdtemp())
    for k, aid in enumerate(('A05', 'A06', 'A07', 'A08')):
        for d, seed in (('2024_03_15', 1 + 10 * k), ('2024_03_16', 2 + 10 * k)):
            sd = tmp / 'data' / aid / d
            sd.mkdir(parents=True)
            rng = np.random.default_rng(seed)
            stim = rng.uniform(-1, 1, 60)
            pd.DataFrame({'Trial_Number': range(1, 61), 'Stim': stim,
                          'Choice': (stim + rng.normal(0, 0.3, 60) > 0).astype(int),
                          'Opto': (np.arange(60) % 3 == 0)}).to_csv(
                sd / 'trial_summary.csv', index=False)
    cfg = {
        'project': {'name': 'doc-example'},
        'reader': {'type': 'csv_folders', 'data_dir': str(tmp / 'data'), 'behaviour_file': 'trial_summary*.csv',
                           'date_regex': r'(\d{4})_(\d{1,2})_(\d{1,2})', 'drop_last_row': False},
        'task': {'choice_mapping': {'type': 'identity', 'no_response_value': -1}},
        'columns': {'trial_number': {'csv_name': 'Trial_Number', 'dtype': 'int'},
                    'stimulus': {'csv_name': 'Stim', 'dtype': 'float'},
                    'choice': {'csv_name': 'Choice', 'dtype': 'float'},
                    'opto_on': {'csv_name': 'Opto', 'dtype': 'bool', 'optional': True, 'default': False}},
        'session_presets': {'expert': {'stage': ''}},
    }
    (tmp / 'config.yaml').write_text(yaml.safe_dump(cfg))
    import json
    (tmp / 'data' / 'animal_metadata.json').write_text(json.dumps(
        {'A05': {'group': 'control'}, 'A06': {'group': 'control'},
         'A07': {'group': 'treated'}, 'A08': {'group': 'treated'}}))
    import os
    os.chdir(tmp)

    import matplotlib.pyplot as plt
    return dict(np=np, pd=pd, plt=plt, experiment=experiment, animal=animal, sessions=sessions, session=sessions[0],
                clean=clean, clean_uniform=clean_uniform, clean_hard=clean_hard,
                rows_on=rows_on, rows_off=rows_off)


def main() -> int:
    ns = _prelude()
    blocks = re.findall(r'```python\n(.*?)```', DOC.read_text(), flags=re.S)
    skipped = []
    for i, block in enumerate(blocks, 1):
        needs = re.match(r'# requires: (\w+)', block)          # e.g. the store's example needs pyarrow
        if needs and importlib.util.find_spec(needs.group(1)) is None:
            skipped.append(f'block {i} (needs {needs.group(1)})')
            continue
        try:
            exec(compile(block, f'<block {i}>', 'exec'), ns)   # noqa: S102 — doc examples
        except Exception as e:  # noqa: BLE001
            print(f'block {i} failed: {type(e).__name__}: {e}\n---\n{block}', file=sys.stderr)
            return 1
        ns['plt'].close('all')
    ran = len(blocks) - len(skipped)
    print(f'{ran} blocks ran' + (f"; skipped {', '.join(skipped)}" if skipped else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
