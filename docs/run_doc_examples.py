"""Run the code on every page of the docs site, so no example can drift from the library.

    python docs/run_doc_examples.py                 # every page
    python docs/run_doc_examples.py EXTENDING.md    # some pages, as paths from the current folder

Each page runs in its own process and a fresh temporary folder, top to bottom, with nothing set up out of
sight: its ```python blocks run in one namespace, the way a reader would paste them, and its ```yaml blocks are
config examples, each of which must load with ``load_config`` once merged onto the packaged minimal config. A
fragment that sets its own reader ``type`` replaces the minimal config's reader instead of merging into it. A
Python block right after the line ``<!-- requires: <module> -->`` (invisible on GitHub and on the site) is
skipped where that module is not installed.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOT_PAGES = ('README.md', 'EXTENDING.md', 'ARCHITECTURE.md', 'CONTRIBUTING.md', 'ROADMAP.md', 'CHANGELOG.md')
BLOCK = re.compile(r'^```(python|yaml)\n(.*?)^```', re.M | re.S)


def site_pages() -> list[Path]:
    """The root files the site includes, then every page under docs/."""
    docs = sorted(p for p in (ROOT / 'docs').rglob('*.md') if '_build' not in p.parts)
    return [ROOT / name for name in ROOT_PAGES] + docs


def _merge(base: dict, fragment: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in fragment.items():
        if key == 'reader' and isinstance(value, dict) and 'type' in value:
            out[key] = value
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _check_config(block: str, folder: Path) -> None:
    from importlib.resources import files

    import yaml

    from behav_utils import load_config

    fragment = yaml.safe_load(block)
    if not isinstance(fragment, dict):
        raise ValueError('a yaml block on the site must be a config mapping')
    base = yaml.safe_load((files('behav_utils') / 'config_presets' / 'config_minimal.yaml').read_text())
    path = folder / 'example_config.yaml'
    path.write_text(yaml.safe_dump(_merge(base, fragment)))
    load_config(path)


def run_page(page: Path) -> int:
    """Run one page in this process; ``main`` gives each page a process and folder of its own."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    text = page.read_text()
    name = page.relative_to(ROOT)
    namespace: dict = {}
    counts = {'python': 0, 'yaml': 0}
    skipped = []
    for match in BLOCK.finditer(text):
        lang, code = match.groups()
        line = text.count('\n', 0, match.start()) + 1
        before = text[:match.start()].rstrip().rsplit('\n', 1)[-1].strip()
        needs = re.fullmatch(r'<!-- requires: (\w+) -->', before)
        if lang == 'python' and needs and importlib.util.find_spec(needs.group(1)) is None:
            skipped.append(f'line {line} (needs {needs.group(1)})')
            continue
        try:
            if lang == 'python':
                exec(compile(code, f'{name}:{line}', 'exec'), namespace)
                plt.close('all')
            else:
                _check_config(code, Path.cwd())
        except Exception as e:
            print(f'{name}: the {lang} block at line {line} failed: {type(e).__name__}: {e}\n---\n{code}',
                  file=sys.stderr)
            return 1
        counts[lang] += 1
    total = sum(counts.values())
    ran = ' and '.join(f'{n} {lang}' for lang, n in counts.items() if n) or 'no'
    print(f"{name}: {ran} block{'s' if total != 1 else ''} ran" + (f"; skipped {', '.join(skipped)}" if skipped else ''))
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ['--one']:
        return run_page(Path(argv[1]))
    pages = [(Path.cwd() / a).resolve() for a in argv] if argv else site_pages()
    failed = []
    for page in pages:
        if not BLOCK.search(page.read_text()):
            continue
        with tempfile.TemporaryDirectory() as folder:
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--one', str(page)], cwd=folder,
                                  env={**os.environ, 'MPLBACKEND': 'Agg'})
        if done.returncode:
            failed.append(str(page.relative_to(ROOT)))
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
