"""
The docstring standard (CONTRIBUTING, *Docstrings*), checked for the public names that have been brought up to it.

For a top-level name whose home sub-package is in ``DOCUMENTED``: the docstring opens with a one-line summary
ending in a full stop; a function's *Args* names exactly its parameters, in order, and it has *Returns* when it
returns something; a dataclass's *Attributes* names exactly its public fields, and an enum's names its members.
``DOCUMENTED`` grows as each sub-package's docstrings are rewritten; step 10 extends the check to every public
name.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
import re

import behav_utils

DOCUMENTED = ('config', 'tasks', 'data')
SUB_PACKAGES = ('config', 'tasks', 'data', 'stats', 'readouts', 'models', 'analysis', 'plotting')
_HEADER = re.compile(r'^(Args|Returns|Raises|Warns|Attributes|Yields|Examples|Notes?|See Also):\s*$')
_ENTRY = re.compile(r'^\s*(\*{0,2}\w+)\s*(\([^)]*\))?\s*:')


def _home(name: str) -> str:
    for package in SUB_PACKAGES:
        if name in importlib.import_module(f'behav_utils.{package}').__all__:
            return package
    raise AssertionError(f'{name} has no home sub-package')


def _checked() -> list[str]:
    return [name for name in behav_utils.__all__ if _home(name) in DOCUMENTED]


def _sections(doc: str) -> dict[str, list[str]]:
    """The Google-style sections of a cleaned docstring: header -> the lines under it."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in doc.splitlines():
        header = _HEADER.match(line)
        if header:
            current = header.group(1)
            sections[current] = []
        elif current and (not line.strip() or line.startswith(' ')):
            sections[current].append(line)
        elif line.strip():
            current = None
    return sections


def _entries(lines: list[str]) -> list[str]:
    """The names an Args or Attributes section documents: its least indented entry lines."""
    body = [line for line in lines if line.strip()]
    if not body:
        return []
    indent = min(len(line) - len(line.lstrip()) for line in body)
    names = []
    for line in body:
        match = _ENTRY.match(line)
        if len(line) - len(line.lstrip()) == indent and match:
            names.append(match.group(1))
    return names


def _parameters(func) -> list[str]:
    names = []
    for p in inspect.signature(func).parameters.values():
        if p.name in ('self', 'cls'):
            continue
        prefix = {p.VAR_POSITIONAL: '*', p.VAR_KEYWORD: '**'}.get(p.kind, '')
        names.append(prefix + p.name)
    return names


def _problems(name: str) -> list[str]:
    obj = getattr(behav_utils, name)
    doc = inspect.getdoc(obj) or ''
    found = []
    first = doc.splitlines()[0] if doc else ''
    if not first.endswith('.'):
        found.append('the first line is not a one-line summary ending in a full stop')
    sections = _sections(doc)
    if inspect.isfunction(obj):
        params = _parameters(obj)
        documented = _entries(sections.get('Args', []))
        if documented != params:
            found.append(f'Args names {documented}, the parameters are {params}')
        returned = inspect.signature(obj).return_annotation
        if returned not in (inspect.Signature.empty, None, 'None') and 'Returns' not in sections:
            found.append('no Returns section')
    elif dataclasses.is_dataclass(obj):
        fields = sorted(f.name for f in dataclasses.fields(obj) if not f.name.startswith('_'))
        documented = sorted(_entries(sections.get('Attributes', [])))
        if documented != fields:
            found.append(f'Attributes names {documented}, the public fields are {fields}')
    elif inspect.isclass(obj) and issubclass(obj, enum.Enum):
        members = sorted(obj.__members__)
        documented = sorted(_entries(sections.get('Attributes', [])))
        if documented != members:
            found.append(f'Attributes names {documented}, the members are {members}')
    return found


def test_documented_names_follow_the_standard():
    failures = {name: problems for name in _checked() if (problems := _problems(name))}
    assert not failures, '\n'.join(f'{name}: {"; ".join(p)}' for name, p in failures.items())


def test_the_check_reads_sections_and_entries():
    doc = 'Summary.\n\nArgs:\n    a: First.\n        Continued.\n    **kw: Rest.\n\nReturns:\n    Something.'
    sections = _sections(doc)
    assert _entries(sections['Args']) == ['a', '**kw']
    assert 'Returns' in sections
