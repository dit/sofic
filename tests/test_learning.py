"""Tests for NL* learning."""

from pensive.automata.dfa import DFA
from pensive.automata.languages.base import AutomatonLanguage
from pensive.automata.learning import learn_maximized_prime_atomaton


def _teacher_dfa() -> AutomatonLanguage:
    dfa = DFA(
        input_alphabet=frozenset({"a", "b"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    dfa.graph.add_state("q0")
    dfa.graph.add_state("q1")
    dfa.add_transition("q0", "q1", "a")
    dfa.add_transition("q0", "q0", "b")
    dfa.add_transition("q1", "q1", "a")
    dfa.add_transition("q1", "q0", "b")
    return AutomatonLanguage(dfa)


def test_learn_mpa():
    teacher = _teacher_dfa()
    learned = learn_maximized_prime_atomaton(teacher, frozenset({"a", "b"}))
    learned.validate()
    assert learned.recognizes(("a",)) == (("a",) in teacher)
