"""Rewrite the published site's version list and root redirect; the docs workflow runs it after each publish.

    python docs/write_versions.py <gh-pages checkout>

The folders ``dev`` and ``<major>.<minor>`` in the checkout are the published versions. ``versions.json`` lists
them for the version switcher, newest release first and marked preferred, then ``dev``; ``index.html`` at the
root redirects to the newest release, or to ``dev`` while there is none. The site's address is ``Documentation``
under ``[project.urls]`` in pyproject.toml.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def site_address() -> str:
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    return project['urls']['Documentation'].rstrip('/')


def write(pages: Path, site: str) -> list[dict]:
    """Write ``versions.json`` and ``index.html`` into ``pages`` and return the version list."""
    releases = sorted((p.name for p in pages.iterdir() if p.is_dir() and re.fullmatch(r'\d+\.\d+', p.name)),
                      key=lambda v: tuple(int(x) for x in v.split('.')), reverse=True)
    versions = [{'name': f'{v} (stable)' if i == 0 else v, 'version': v, 'url': f'{site}/{v}/'}
                for i, v in enumerate(releases)]
    if versions:
        versions[0]['preferred'] = True
    if (pages / 'dev').is_dir():
        versions.append({'name': 'dev', 'version': 'dev', 'url': f'{site}/dev/'})
    (pages / 'versions.json').write_text(json.dumps(versions, indent=2) + '\n')
    target = f'{releases[0]}/' if releases else 'dev/'
    (pages / 'index.html').write_text(
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>behav_utils</title>\n'
        f'<meta http-equiv="refresh" content="0; url={target}">\n<link rel="canonical" href="{site}/{target}">\n'
        f'</head>\n<body><a href="{target}">behav_utils documentation</a></body>\n</html>\n')
    return versions


if __name__ == '__main__':
    for version in write(Path(sys.argv[1]), site_address()):
        print(version['version'], version['url'], '(preferred)' if version.get('preferred') else '')
