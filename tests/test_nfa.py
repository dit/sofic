"""Tests for NFA and epsilon closure."""

from pensive.automata.nfa import NFA


def _nfa_with_epsilon() -> NFA:
    nfa = NFA(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    for state in ("q0", "q1", "q2"):
        nfa.graph.add_state(state)
    nfa.add_transition("q0", "q1")  # epsilon
    nfa.add_transition("q0", "q2", "a")
    return nfa


def test_epsilon_closure():
    nfa = _nfa_with_epsilon()
    closure = nfa.epsilon_closure({"q0"})
    assert closure == {"q0", "q1"}


def test_recognizes_via_epsilon():
    nfa = _nfa_with_epsilon()
    assert nfa.recognizes(())
    assert not nfa.recognizes(("a",))


def test_validate_and_round_trip():
    nfa = _nfa_with_epsilon()
    nfa.validate()
    restored = NFA.from_networkx(
        nfa.to_networkx(),
        input_alphabet=nfa.input_alphabet,
        initial_states=nfa.initial_states,
        accepting_states=nfa.accepting_states,
    )
    assert restored.recognizes(())


def test_words_of_length_and_iter_language_with_epsilon():
    nfa = _nfa_with_epsilon()
    assert list(nfa.words_of_length(0)) == [()]
    assert list(nfa.words_of_length(1)) == []
    assert list(nfa.iter_language(max_length=1)) == [()]


def test_operations_respect_multiple_initial_states():
    nfa = NFA(
        input_alphabet=frozenset({"a", "b"}),
        initial_states=frozenset({"qa", "qb"}),
        accepting_states=frozenset({"fa", "fb"}),
    )
    for state in ("qa", "qb", "fa", "fb"):
        nfa.graph.add_state(state)
    nfa.add_transition("qa", "fa", "a")
    nfa.add_transition("qb", "fb", "b")

    union = nfa.union(_nfa_with_epsilon())
    star = nfa.kleene_star()

    assert union.recognizes(("a",))
    assert union.recognizes(("b",))
    assert star.recognizes(())
    assert star.recognizes(("a", "b"))
    assert star.recognizes(("b", "a"))
