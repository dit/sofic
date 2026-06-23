"""Tests for generator conversions."""

from pensive.automata.dfa import DFA
from pensive.automata.nfa import NFA
from pensive.generators.base import HiddenMarkovModel
from pensive.generators.mealy import MealyHMM
from pensive.generators.moore import MooreHMM
from pensive.generators.pfa import ProbabilisticFiniteAutomaton
from pensive.graph import ATTR_EMISSION, ATTR_EMISSION_DIST, ATTR_PROB, ATTR_SYMBOL
from pensive.shifts.sofic import SoficShift


def _golden_mean_support_hmm() -> MealyHMM:
    hmm = MealyHMM(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset({0, 1}),
    )
    for state in ("A", "B"):
        hmm.graph.add_state(state)
    hmm.graph.add_transition("A", "A", **{ATTR_PROB: 0.5, ATTR_EMISSION: 0})
    hmm.graph.add_transition("A", "B", **{ATTR_PROB: 0.5, ATTR_EMISSION: 1})
    hmm.graph.add_transition("B", "A", **{ATTR_PROB: 1.0, ATTR_EMISSION: 0})
    hmm.graph.add_transition("B", "B", **{ATTR_PROB: 0.0, ATTR_EMISSION: 1})
    return hmm


def test_moore_to_mealy():
    moore = MooreHMM(
        initial_distribution={"q0": 1.0},
        observation_alphabet=frozenset({"0"}),
    )
    moore.graph.add_state("q0")
    moore.graph.add_state("q0", **{ATTR_EMISSION_DIST: {"0": 1.0}})
    moore.graph.add_transition("q0", "q0", **{ATTR_PROB: 1.0})
    mealy = moore.to_mealy()
    mealy.validate()
    assert isinstance(mealy, MealyHMM)


def test_pfa_to_mealy():
    pfa = ProbabilisticFiniteAutomaton(
        initial_distribution={"q0": 1.0},
        output_alphabet=frozenset({"a"}),
    )
    pfa.graph.add_state("q0")
    pfa.graph.add_transition("q0", "q0", **{ATTR_PROB: 1.0, ATTR_EMISSION: "a"})
    hmm = pfa.to_mealy()
    hmm.validate()


def test_hmm_to_sofic_shift_strips_probabilities():
    hmm = _golden_mean_support_hmm()
    shift = hmm.to_sofic_shift()

    assert isinstance(shift, SoficShift)
    assert set(shift.states()) == {"A", "B"}
    assert shift.symbol_alphabet == frozenset({0, 1})
    assert {
        (transition.source, transition.target, transition.data[ATTR_SYMBOL]) for transition in shift.transitions()
    } == {
        ("A", "A", 0),
        ("A", "B", 1),
        ("B", "A", 0),
    }
    assert all(ATTR_PROB not in transition.data for transition in shift.transitions())
    shift.validate()


def test_hmm_to_support_nfa_uses_fresh_epsilon_start():
    hmm = _golden_mean_support_hmm()
    nfa = hmm.to_support_nfa()

    assert isinstance(nfa, NFA)
    assert nfa.input_alphabet == frozenset({0, 1})
    assert nfa.accepting_states == frozenset({"A", "B"})

    [start] = list(nfa.initial_states)
    assert start not in {"A", "B"}
    assert nfa.epsilon_closure({start}) == {start, "A", "B"}
    assert nfa.recognizes(())
    assert nfa.recognizes((1, 0, 1))
    assert not nfa.recognizes((1, 1))
    nfa.validate()


def test_hmm_to_support_dfa_starts_from_all_states_and_accepts_recurrent_subsets():
    hmm = _golden_mean_support_hmm()
    dfa = hmm.to_support_dfa()

    assert isinstance(dfa, DFA)
    assert dfa.initial_states == frozenset({frozenset({"A", "B"})})
    assert frozenset() not in set(dfa.states())
    assert dfa.accepting_states == frozenset({frozenset({"A"}), frozenset({"B"})})
    assert not dfa.recognizes(())
    assert dfa.recognizes((0,))
    assert dfa.recognizes((1,))
    assert dfa.recognizes((1, 0, 1))
    assert not dfa.recognizes((1, 1))
    dfa.validate()


def test_moore_hmm_support_conversions_delegate_to_mealy_support():
    moore = MooreHMM(
        initial_distribution={"A": 1.0},
        observation_alphabet=frozenset({"0", "1"}),
    )
    moore.graph.add_state("A", **{ATTR_EMISSION_DIST: {"0": 1.0}})
    moore.graph.add_state("B", **{ATTR_EMISSION_DIST: {"1": 1.0}})
    moore.graph.add_transition("A", "B", **{ATTR_PROB: 1.0})
    moore.graph.add_transition("B", "A", **{ATTR_PROB: 1.0})

    assert moore.to_support_nfa().recognizes(("0", "1", "0"))
    assert not moore.to_support_nfa().recognizes(("1", "1"))
    assert {
        (transition.source, transition.target, transition.data[ATTR_SYMBOL])
        for transition in moore.to_sofic_shift().transitions()
    } == {
        ("A", "B", "0"),
        ("B", "A", "1"),
    }


def test_hmm_support_conversions_use_to_mealy_hook():
    class WrappedHMM(HiddenMarkovModel):
        def __init__(self, support: MealyHMM) -> None:
            super().__init__(
                initial_distribution=support.initial_distribution,
                observation_alphabet=support.observation_alphabet,
            )
            self._support = support

        def to_mealy(self) -> MealyHMM:
            return self._support

    wrapped = WrappedHMM(_golden_mean_support_hmm())
    assert wrapped.to_support_nfa().recognizes((1, 0, 1))
    assert not wrapped.to_support_nfa().recognizes((1, 1))
