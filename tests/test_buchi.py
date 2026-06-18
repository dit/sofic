"""Tests for Büchi automata."""

import pytest

from pensive.automata.buchi import BuchiAutomaton
from pensive.graph import ATTR_SYMBOL


def _accepting_loop_ba() -> BuchiAutomaton:
    ba = BuchiAutomaton(
        input_alphabet=frozenset({"a", "b"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    ba.graph.add_state("q0")
    ba.graph.add_state("q1")
    ba.graph.add_transition("q0", "q1", **{ATTR_SYMBOL: "a"})
    ba.graph.add_transition("q1", "q1", **{ATTR_SYMBOL: "a"})
    ba.graph.add_transition("q1", "q0", **{ATTR_SYMBOL: "b"})
    return ba


def _rejecting_loop_ba() -> BuchiAutomaton:
    ba = BuchiAutomaton(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    ba.graph.add_state("q0")
    ba.graph.add_state("q1")
    ba.graph.add_transition("q0", "q1", **{ATTR_SYMBOL: "a"})
    return ba


def test_accepts_lasso_visits_accepting_infinitely():
    ba = _accepting_loop_ba()
    assert ba.accepts_lasso((), ("a",))
    assert ba.accepts_lasso(("a",), ("a",))


def test_rejects_lasso_finite_accept_visits():
    ba = _rejecting_loop_ba()
    assert not ba.accepts_lasso((), ("a",))


def test_accepts_omega_periodic_pair():
    ba = _accepting_loop_ba()
    assert ba.accepts_omega(((), ("a",)))
    assert ba.accepts_omega((("a",), ("a",)))


def test_accepts_omega_non_periodic_raises():
    ba = _accepting_loop_ba()
    with pytest.raises(NotImplementedError):
        ba.accepts_omega(("a", "b", "a"))
