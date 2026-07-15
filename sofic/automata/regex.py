"""Automata to regular-expression conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sofic.automata.base import LabeledAutomaton
from sofic.graph import ATTR_SYMBOL, EPSILON


@dataclass(frozen=True, slots=True)
class _Regex:
    kind: str
    parts: tuple[Any, ...] = ()

    def pattern(self) -> str:
        if self.kind == "empty":
            return "(?!)"
        if self.kind == "epsilon":
            return "(?:)"
        if self.kind == "literal":
            return re.escape(str(self.parts[0]))
        if self.kind == "union":
            return "(?:" + "|".join(part.pattern() for part in self.parts) + ")"
        if self.kind == "concat":
            return "".join(_atom(part) for part in self.parts)
        if self.kind == "star":
            return _atom(self.parts[0]) + "*"
        raise ValueError(f"unknown regex node {self.kind!r}")


_EMPTY = _Regex("empty")
_EPSILON = _Regex("epsilon")


def automaton_to_regex(automaton: LabeledAutomaton) -> str:
    """Return a Python-regex-compatible expression for ``automaton``.

    The result uses ``(?!)`` for the empty language and ``(?:)`` for epsilon.
    Symbols are rendered with ``str(symbol)`` and escaped for Python's
    :mod:`re` engine.
    """
    start = object()
    final = object()
    states = sorted(automaton.states(), key=repr)
    remaining: list[Any] = [start, *states, final]
    labels: dict[tuple[Any, Any], _Regex] = {}

    for state in automaton.initial_states:
        labels[(start, state)] = _union(labels.get((start, state), _EMPTY), _EPSILON)
    for state in automaton.accepting_states:
        labels[(state, final)] = _union(labels.get((state, final), _EMPTY), _EPSILON)
    for transition in automaton.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is None:
            continue
        label = _EPSILON if symbol is EPSILON else _Regex("literal", (symbol,))
        key = (transition.source, transition.target)
        labels[key] = _union(labels.get(key, _EMPTY), label)

    for state in states:
        others = [candidate for candidate in remaining if candidate != state]
        loop = _star(labels.get((state, state), _EMPTY))
        for source in others:
            left = labels.get((source, state), _EMPTY)
            if left == _EMPTY:
                continue
            for target in others:
                right = labels.get((state, target), _EMPTY)
                if right == _EMPTY:
                    continue
                key = (source, target)
                labels[key] = _union(labels.get(key, _EMPTY), _concat(left, loop, right))
        labels = {key: value for key, value in labels.items() if state not in key}
        remaining = others

    return labels.get((start, final), _EMPTY).pattern()


def _union(*terms: _Regex) -> _Regex:
    parts: list[_Regex] = []
    for term in terms:
        if term == _EMPTY:
            continue
        if term.kind == "union":
            parts.extend(term.parts)
        else:
            parts.append(term)
    unique = sorted(set(parts), key=lambda part: part.pattern())
    if not unique:
        return _EMPTY
    if len(unique) == 1:
        return unique[0]
    return _Regex("union", tuple(unique))


def _concat(*terms: _Regex) -> _Regex:
    parts: list[_Regex] = []
    for term in terms:
        if term == _EMPTY:
            return _EMPTY
        if term == _EPSILON:
            continue
        if term.kind == "concat":
            parts.extend(term.parts)
        else:
            parts.append(term)
    if not parts:
        return _EPSILON
    if len(parts) == 1:
        return parts[0]
    return _Regex("concat", tuple(parts))


def _star(term: _Regex) -> _Regex:
    if term in {_EMPTY, _EPSILON}:
        return _EPSILON
    if term.kind == "star":
        return term
    return _Regex("star", (term,))


def _atom(term: _Regex) -> str:
    if term.kind in {"literal", "epsilon", "empty", "star"}:
        return term.pattern()
    return "(?:" + term.pattern() + ")"
