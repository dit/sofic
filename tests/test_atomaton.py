"""Tests for átomaton skeletons."""

from pensive.automata.atomaton import Atomaton, MaximizedPrimeAtomaton
from pensive.automata.dfa import DFA
from pensive.automata.rfsa import CanonicalRFSA


def test_atomaton_validate():
    auto = Atomaton(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q0"}),
    )
    auto.graph.add_state("q0")
    auto.validate()


def _lang_dfa() -> DFA:
    dfa = DFA(input_alphabet=frozenset({"a"}), initial_states=frozenset({"q0"}), accepting_states=frozenset({"q0"}))
    dfa.graph.add_state("q0")
    dfa.add_transition("q0", "q0", "a")
    return dfa


def test_atomaton_from_language():
    auto = Atomaton.from_language(_lang_dfa())
    auto.validate()


def test_mpa_from_canonical_rfsa():
    rfsa = CanonicalRFSA.from_language(_lang_dfa())
    mpa = MaximizedPrimeAtomaton.from_canonical_rfsa(rfsa)
    mpa.validate()
