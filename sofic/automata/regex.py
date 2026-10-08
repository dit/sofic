"""Conversions between automata and regular expressions.

Regular expressions use a small dialect that coincides with Python's :mod:`re`
syntax whenever every symbol is a single-character string:

* a single character is a literal symbol; ``\\c`` escapes any character ``c``;
* ``'...'`` quotes a multi-character symbol, with ``\\'`` and ``\\\\`` escapes
  inside the quotes (``'ab'`` is one symbol, ``ab`` is two);
* ``ε`` or ``(?:)`` denotes the empty word and ``∅`` or ``(?!)`` the empty
  language;
* ``|`` is alternation, juxtaposition is concatenation, and the postfix
  operators ``*``, ``+``, ``?`` bind tightest;
* ``(...)`` and ``(?:...)`` group.

The characters ``. [ ] { } ^ $`` are reserved (their :mod:`re` meaning is not
supported) and must be escaped to be used as symbols.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from sofic.automata.base import LabeledAutomaton
from sofic.automata.nfa import NFA
from sofic.exceptions import RegexSyntaxError
from sofic.graph import ATTR_SYMBOL, EPSILON

__all__ = ["Regex", "RegexSyntaxError", "automaton_to_regex", "parse_regex", "regex_to_nfa"]


@dataclass(frozen=True, slots=True)
class Regex:
    """Regular-expression syntax tree.

    ``kind`` is one of ``"empty"``, ``"epsilon"``, ``"literal"`` (``parts`` is
    the symbol), ``"union"``, ``"concat"``, ``"star"``, ``"plus"``, or
    ``"optional"`` (``parts`` holds the operand nodes).
    """

    kind: str
    parts: tuple[Any, ...] = ()

    def pattern(self) -> str:
        """Render this tree in the module's regular-expression dialect."""
        if self.kind == "empty":
            return "(?!)"
        if self.kind == "epsilon":
            return "(?:)"
        if self.kind == "literal":
            return _render_symbol(self.parts[0])
        if self.kind == "union":
            return "(?:" + "|".join(part.pattern() for part in self.parts) + ")"
        if self.kind == "concat":
            return "".join(_atom(part) for part in self.parts)
        if self.kind in _POSTFIX_KINDS:
            return _atom(self.parts[0]) + _POSTFIX_KINDS[self.kind]
        raise ValueError(f"unknown regex node {self.kind!r}")


_EMPTY = Regex("empty")
_EPSILON = Regex("epsilon")
_POSTFIX_KINDS = {"star": "*", "plus": "+", "optional": "?"}
_POSTFIX_OPERATORS = {operator: kind for kind, operator in _POSTFIX_KINDS.items()}
_QUOTE = "'"
_RESERVED_SYMBOLS = frozenset({"ε", "∅", _QUOTE})
_UNSUPPORTED_METACHARACTERS = frozenset(".[]{}^$")


def automaton_to_regex(automaton: LabeledAutomaton) -> str:
    """Return a regular expression for ``automaton`` by state elimination.

    The construction follows Kleene's theorem :cite:`Kleene1956` in the
    state-elimination form of :cite:`HopcroftUllman1979`. The result uses
    ``(?!)`` for the empty language and ``(?:)`` for epsilon. Symbols are
    rendered with ``str(symbol)``: single characters are escaped as for
    Python's :mod:`re` engine, so the output is a valid :mod:`re` pattern when
    every symbol is a single-character string, and longer symbols are quoted
    (``'ab'``) so the output stays unambiguous. :func:`parse_regex` and
    :func:`regex_to_nfa` read the output back.
    """
    start = object()
    final = object()
    states = sorted(automaton.states(), key=repr)
    remaining: list[Any] = [start, *states, final]
    labels: dict[tuple[Any, Any], Regex] = {}

    for state in automaton.initial_states:
        labels[(start, state)] = _union(labels.get((start, state), _EMPTY), _EPSILON)
    for state in automaton.accepting_states:
        labels[(state, final)] = _union(labels.get((state, final), _EMPTY), _EPSILON)
    for transition in automaton.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is None:
            continue
        label = _EPSILON if symbol is EPSILON else Regex("literal", (symbol,))
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


def parse_regex(text: str) -> Regex:
    """Parse ``text`` in the module's regular-expression dialect.

    The tree is returned unsimplified. Raises :class:`RegexSyntaxError` on
    unbalanced parentheses, operators without operands, empty alternatives,
    unterminated quotes or escapes, and reserved metacharacters.
    """
    return _Parser(text).parse()


def regex_to_nfa(text: str, *, alphabet: Iterable[Any] | None = None) -> NFA:
    """Return an ε-NFA for ``text`` via Thompson's construction.

    Implements :cite:`Thompson1968`: every literal and ε becomes a two-state
    fragment, and union, concatenation, and the postfix operators wire
    fragments together with ε-transitions, so the NFA has one initial and one
    accepting state and at most two states per syntax-tree node.

    Without ``alphabet``, symbols are the parsed strings and the input alphabet
    is the set of symbols occurring in ``text``. With ``alphabet``, each parsed
    string is resolved to the alphabet element equal to it, or else to the
    unique element whose ``str`` matches it, so non-string symbols round-trip
    through :func:`automaton_to_regex`; the NFA's input alphabet is
    ``alphabet``.
    """
    tree = parse_regex(text)
    symbols = None if alphabet is None else frozenset(alphabet)
    resolve = _symbol_resolver(symbols)
    transitions: list[tuple[int, int, Any]] = []
    count = 0

    def new_state() -> int:
        nonlocal count
        count += 1
        return count - 1

    def build(node: Regex) -> tuple[int, int]:
        if node.kind == "concat":
            fragments = [build(part) for part in node.parts]
            for (_, accept), (start, _) in zip(fragments, fragments[1:], strict=False):
                transitions.append((accept, start, EPSILON))
            return fragments[0][0], fragments[-1][1]
        start, accept = new_state(), new_state()
        if node.kind == "epsilon":
            transitions.append((start, accept, EPSILON))
        elif node.kind == "literal":
            transitions.append((start, accept, resolve(node.parts[0])))
        elif node.kind == "union":
            for part in node.parts:
                inner_start, inner_accept = build(part)
                transitions.append((start, inner_start, EPSILON))
                transitions.append((inner_accept, accept, EPSILON))
        elif node.kind in _POSTFIX_KINDS:
            inner_start, inner_accept = build(node.parts[0])
            transitions.append((start, inner_start, EPSILON))
            transitions.append((inner_accept, accept, EPSILON))
            if node.kind != "optional":
                transitions.append((inner_accept, inner_start, EPSILON))
            if node.kind != "plus":
                transitions.append((start, accept, EPSILON))
        return start, accept

    start, accept = build(tree)
    if symbols is None:
        symbols = frozenset(symbol for _, _, symbol in transitions if symbol is not EPSILON)
    nfa = NFA(input_alphabet=symbols, initial_states=frozenset({start}), accepting_states=frozenset({accept}))
    for state in range(count):
        nfa.graph.add_state(state)
    for source, target, symbol in transitions:
        nfa.add_transition(source, target, symbol)
    return nfa


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    def parse(self) -> Regex:
        tree = self._union()
        if self.position < len(self.text):
            self._fail("unbalanced parenthesis ')'")
        return tree

    def _fail(self, message: str, position: int | None = None) -> None:
        raise RegexSyntaxError(message, self.text, self.position if position is None else position)

    def _peek(self) -> str | None:
        return self.text[self.position] if self.position < len(self.text) else None

    def _union(self) -> Regex:
        alternatives = [self._concat()]
        while self._peek() == "|":
            self.position += 1
            alternatives.append(self._concat())
        return alternatives[0] if len(alternatives) == 1 else Regex("union", tuple(alternatives))

    def _concat(self) -> Regex:
        factors: list[Regex] = []
        while self._peek() not in {None, "|", ")"}:
            factors.append(self._repeat())
        if not factors:
            self._fail("expected an expression")
        return factors[0] if len(factors) == 1 else Regex("concat", tuple(factors))

    def _repeat(self) -> Regex:
        if self._peek() in _POSTFIX_OPERATORS:
            self._fail(f"nothing to repeat before {self._peek()!r}")
        node = self._atom()
        if (operator := self._peek()) in _POSTFIX_OPERATORS:
            self.position += 1
            node = Regex(_POSTFIX_OPERATORS[operator], (node,))
            if self._peek() in _POSTFIX_OPERATORS:
                self._fail("multiple repeat")
        return node

    def _atom(self) -> Regex:
        text = self.text
        position = self.position
        char = text[position]
        if text.startswith("(?:)", position):
            self.position += 4
            return _EPSILON
        if text.startswith("(?!)", position):
            self.position += 4
            return _EMPTY
        if char == "(":
            if text.startswith("(?:", position):
                self.position += 3
            elif text.startswith("(?", position):
                self._fail("unsupported group extension '(?'")
            else:
                self.position += 1
            node = self._union()
            if self._peek() != ")":
                self._fail("missing ')' for parenthesis", position)
            self.position += 1
            return node
        self.position += 1
        if char == "ε":
            return _EPSILON
        if char == "∅":
            return _EMPTY
        if char == "\\":
            if self.position >= len(text):
                self._fail("dangling backslash", position)
            self.position += 1
            return Regex("literal", (text[self.position - 1],))
        if char == _QUOTE:
            return Regex("literal", (self._quoted(position),))
        if char in _UNSUPPORTED_METACHARACTERS:
            self._fail(f"unsupported metacharacter {char!r} (escape it as '\\{char}')", position)
        return Regex("literal", (char,))

    def _quoted(self, opening: int) -> str:
        chars: list[str] = []
        text = self.text
        while self.position < len(text):
            char = text[self.position]
            self.position += 1
            if char == _QUOTE:
                return "".join(chars)
            if char == "\\":
                if self.position >= len(text):
                    break
                char = text[self.position]
                self.position += 1
            chars.append(char)
        self._fail("unterminated quoted symbol", opening)
        raise AssertionError  # pragma: no cover


def _symbol_resolver(symbols: frozenset[Any] | None) -> Callable[[str], Any]:
    if symbols is None:
        return lambda token: token
    by_text: dict[str, list[Any]] = {}
    for symbol in symbols:
        by_text.setdefault(str(symbol), []).append(symbol)

    def resolve(token: str) -> Any:
        if token in symbols:
            return token
        matches = by_text.get(token, [])
        if len(matches) != 1:
            reason = "not in" if not matches else "ambiguous in"
            raise ValueError(f"regex symbol {token!r} is {reason} the alphabet")
        return matches[0]

    return resolve


def _render_symbol(symbol: Any) -> str:
    text = str(symbol)
    if len(text) == 1:
        return "\\" + text if text in _RESERVED_SYMBOLS else re.escape(text)
    return _QUOTE + text.replace("\\", "\\\\").replace(_QUOTE, "\\" + _QUOTE) + _QUOTE


def _union(*terms: Regex) -> Regex:
    parts: list[Regex] = []
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
    return Regex("union", tuple(unique))


def _concat(*terms: Regex) -> Regex:
    parts: list[Regex] = []
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
    return Regex("concat", tuple(parts))


def _star(term: Regex) -> Regex:
    if term in {_EMPTY, _EPSILON}:
        return _EPSILON
    if term.kind == "star":
        return term
    return Regex("star", (term,))


def _atom(term: Regex) -> str:
    if term.kind in {"literal", "epsilon", "empty", *_POSTFIX_KINDS}:
        return term.pattern()
    return "(?:" + term.pattern() + ")"
