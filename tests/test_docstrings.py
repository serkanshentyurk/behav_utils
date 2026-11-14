"""
The docstring standard (CONTRIBUTING, *Docstrings*), checked for the public API.

Checked: every top-level name; and, for the sub-packages listed in ``SUB_PACKAGE_NAMES``, the names public only
in the sub-package and the public methods its classes define. A docstring opens with a one-line summary ending in a full stop; a
function's or method's *Args* names exactly its parameters, in order, and it has *Returns* when it returns
something; a dataclass's *Attributes* names exactly its public fields, and an enum's names its members.
``SUB_PACKAGE_NAMES`` grows as each sub-package's own names are brought up to the standard.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
import re

import behav_utils

SUB_PACKAGES = ('config', 'tasks', 'data', 'stats', 'readouts', 'models', 'analysis', 'plotting')
SUB_PACKAGE_NAMES = ('config', 'tasks')
_HEADER = re.compile(r'^(Args|Returns|Raises|Warns|Attributes|Yields|Examples|Notes?|See Also):\s*$')
_ENTRY = re.compile(r'^\s*(\*{0,2}\w+)\s*(\([^)]*\))?\s*:')


def _targets() -> list[tuple[str, object]]:
    """(label, object) for every name and method the standard is checked on."""
    found = [(name, getattr(behav_utils, name)) for name in behav_utils.__all__]
    with_methods = []
    for package in SUB_PACKAGE_NAMES:
        module = importlib.import_module(f'behav_utils.{package}')
        own = [(name if name in behav_utils.__all__ else f'{package}.{name}', getattr(module, name))
               for name in module.__all__]
        found += [(label, obj) for label, obj in own if '.' in label]
        with_methods += own
    methods = []
    for label, obj in with_methods:
        if inspect.isclass(obj) and not issubclass(obj, (enum.Enum, BaseException)):
            for key, value in vars(obj).items():
                func = getattr(value, '__func__', value)
                if not key.startswith('_') and inspect.isfunction(func):
                    methods.append((f'{label}.{key}', func))
    return found + methods


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


def _problems(obj) -> list[str]:
    if not (inspect.isfunction(obj) or inspect.isclass(obj)):
        return []
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
    elif issubclass(obj, enum.Enum):
        members = sorted(obj.__members__)
        documented = sorted(_entries(sections.get('Attributes', [])))
        if documented != members:
            found.append(f'Attributes names {documented}, the members are {members}')
    return found


def test_documented_names_follow_the_standard():
    failures = {label: problems for label, obj in _targets() if (problems := _problems(obj))}
    assert not failures, '\n'.join(f'{label}: {"; ".join(p)}' for label, p in failures.items())


def test_the_check_reads_sections_and_entries():
    doc = 'Summary.\n\nArgs:\n    a: First.\n        Continued.\n    **kw: Rest.\n\nReturns:\n    Something.'
    sections = _sections(doc)
    assert _entries(sections['Args']) == ['a', '**kw']
    assert 'Returns' in sections
