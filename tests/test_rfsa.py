"""Tests for residual and canonical RFSA skeletons."""

from pensive.automata.dfa import DFA
from pensive.automata.observation import ObservationTable
from pensive.automata.rfsa import CanonicalRFSA, ResidualFiniteStateAutomaton


def test_rfsa_validate():
    rfsa = ResidualFiniteStateAutomaton(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q0"}),
    )
    rfsa.graph.add_state("q0")
    rfsa.validate()


def _lang_dfa() -> DFA:
    dfa = DFA(input_alphabet=frozenset({"a"}), initial_states=frozenset({"q0"}), accepting_states=frozenset({"q0"}))
    dfa.graph.add_state("q0")
    dfa.add_transition("q0", "q0", "a")
    return dfa


def test_canonical_rfsa_from_language():
    rfsa = CanonicalRFSA.from_language(_lang_dfa())
    rfsa.validate()


def test_canonical_rfsa_from_table():
    CanonicalRFSA.from_observation_table(ObservationTable())
