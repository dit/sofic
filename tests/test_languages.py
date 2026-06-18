"""Tests for regular-language wrappers and quotients."""

import pytest

from pensive.automata.dfa import DFA
from pensive.automata.languages.base import AutomatonLanguage, ExplicitLanguage, as_language
from pensive.automata.languages.quotients import left_quotient, right_quotient
from pensive.automata.languages.residuals import is_composed_residual
from pensive.graph import ATTR_SYMBOL


def _simple_dfa() -> DFA:
    dfa = DFA(input_alphabet=frozenset({"a", "b"}), initial_states=frozenset({"q0"}), accepting_states=frozenset({"q1"}))
    dfa.graph.add_state("q0")
    dfa.graph.add_state("q1")
    dfa.add_transition("q0", "q1", "a")
    dfa.add_transition("q0", "q0", "b")
    dfa.add_transition("q1", "q1", "a")
    dfa.add_transition("q1", "q0", "b")
    return dfa


def test_explicit_language_membership():
    lang = ExplicitLanguage(
        positive={("a",), ("ab",)},
        negative={("b",)},
        alphabet=frozenset({"a", "b"}),
    )
    assert ("a",) in lang
    assert ("ab",) in lang
    assert ("b",) not in lang
    assert ("aa",) not in lang


def test_automaton_language():
    dfa = _simple_dfa()
    lang = AutomatonLanguage(dfa)
    assert ("a",) in lang
    assert ("b",) not in lang
    assert as_language(dfa) is not dfa
    assert ("a",) in as_language(dfa)


def test_left_quotient():
    lang = ExplicitLanguage(positive={("a", "b"), ("a", "c")}, alphabet=frozenset({"a", "b", "c"}))
    quot = left_quotient(("a",), lang)
    assert isinstance(quot, ExplicitLanguage)
    assert ("b",) in quot
    assert ("c",) in quot
    assert ("a",) not in quot


def test_right_quotient():
    lang = ExplicitLanguage(positive={("b", "a"), ("c", "a")}, alphabet=frozenset({"a", "b", "c"}))
    quot = right_quotient(lang, ("a",))
    assert ("b",) in quot
    assert ("c",) in quot


def test_is_composed_residual():
    r1 = ExplicitLanguage(positive={("a",)})
    r2 = ExplicitLanguage(positive={("b",)})
    union = ExplicitLanguage(positive={("a",), ("b",)})
    all_residuals = frozenset({r1, r2, union})
    assert is_composed_residual(union, all_residuals)
    assert not is_composed_residual(r1, all_residuals)


def test_atoms_from_dfa():
    from pensive.automata.languages.atoms import atoms

    dfa = _simple_dfa()
    result = atoms(AutomatonLanguage(dfa))
    assert len(result) >= 1

