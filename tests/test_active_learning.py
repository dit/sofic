"""Tests for active automata learning (L*, TTT, Mealy L*)."""

from __future__ import annotations

import itertools

import pytest

from sofic.automata.active import (
    ExhaustiveEquivalenceOracle,
    FunctionMembershipOracle,
    LanguageMembershipOracle,
    RandomWalkEquivalenceOracle,
    learn_dfa_from_language,
    learn_dfa_lstar,
    learn_dfa_ttt,
    learn_mealy_from_transducer,
)
from sofic.automata.dfa import DFA
from sofic.automata.transducers import MealyMachine

ALPHABET = ("a", "b")


def _parity_dfa() -> DFA:
    dfa = DFA(input_alphabet=frozenset(ALPHABET))
    dfa.graph.add_state("even")
    dfa.graph.add_state("odd")
    dfa.add_transition("even", "odd", "a")
    dfa.add_transition("even", "even", "b")
    dfa.add_transition("odd", "even", "a")
    dfa.add_transition("odd", "odd", "b")
    dfa.initial_states = frozenset({"even"})
    dfa.accepting_states = frozenset({"even"})
    dfa.validate()
    return dfa


def _a_before_b_dfa() -> DFA:
    dfa = DFA(input_alphabet=frozenset(ALPHABET))
    for state in ("q0", "q1", "dead"):
        dfa.graph.add_state(state)
    dfa.add_transition("q0", "q0", "a")
    dfa.add_transition("q0", "q1", "b")
    dfa.add_transition("q1", "dead", "a")
    dfa.add_transition("q1", "q1", "b")
    dfa.add_transition("dead", "dead", "a")
    dfa.add_transition("dead", "dead", "b")
    dfa.initial_states = frozenset({"q0"})
    dfa.accepting_states = frozenset({"q0", "q1"})
    dfa.validate()
    return dfa


def _toggle_mealy() -> MealyMachine:
    machine = MealyMachine(
        input_alphabet=frozenset({"x", "y"}),
        output_alphabet=frozenset({0, 1}),
        initial_states=frozenset({"s0"}),
    )
    machine.graph.add_state("s0")
    machine.graph.add_state("s1")
    machine.add_transition("s0", "s1", "x", output=1)
    machine.add_transition("s1", "s0", "x", output=0)
    machine.add_transition("s0", "s0", "y", output=0)
    machine.add_transition("s1", "s1", "y", output=1)
    machine.validate()
    return machine


def _dfa_equivalent(left: DFA, right: DFA, alphabet, max_len=9) -> bool:
    for length in range(max_len + 1):
        for word in itertools.product(sorted(alphabet), repeat=length):
            if left.recognizes(word) != right.recognizes(word):
                return False
    return True


def _mealy_equivalent(left: MealyMachine, right: MealyMachine, alphabet, max_len=8) -> bool:
    for length in range(1, max_len + 1):
        for word in itertools.product(sorted(alphabet), repeat=length):
            if next(iter(left.transduce(word))) != next(iter(right.transduce(word))):
                return False
    return True


# ------------------------------------------------------------------------- DFA L*/TTT


@pytest.mark.parametrize("algorithm", ["lstar", "ttt"])
@pytest.mark.parametrize("target_factory", [_parity_dfa, _a_before_b_dfa])
def test_learns_target_dfa(algorithm, target_factory):
    target = target_factory()
    learned = learn_dfa_from_language(target, ALPHABET, algorithm=algorithm, max_length=10)
    learned.validate()
    assert _dfa_equivalent(learned, target, ALPHABET)


@pytest.mark.parametrize("algorithm", ["lstar", "ttt"])
def test_learns_minimal_state_count(algorithm):
    target = _parity_dfa()
    learned = learn_dfa_from_language(target, ALPHABET, algorithm=algorithm, max_length=10)
    assert len(list(learned.states())) == 2


def test_lstar_and_ttt_agree():
    target = _a_before_b_dfa()
    lstar = learn_dfa_from_language(target, ALPHABET, algorithm="lstar", max_length=10)
    ttt = learn_dfa_from_language(target, ALPHABET, algorithm="ttt", max_length=10)
    assert _dfa_equivalent(lstar, ttt, ALPHABET)


def test_learn_dfa_from_predicate_with_random_oracle():
    def even_a(word):
        return word.count("a") % 2 == 0

    membership = FunctionMembershipOracle(even_a)
    equivalence = RandomWalkEquivalenceOracle(membership, ALPHABET, num_walks=3000, max_steps=20, rng=0)
    learned = learn_dfa_lstar(ALPHABET, membership, equivalence, max_rounds=50)
    learned.validate()
    for length in range(8):
        for word in itertools.product(ALPHABET, repeat=length):
            assert learned.recognizes(word) == even_a(word)


def test_single_state_language():
    # accept everything
    membership = FunctionMembershipOracle(lambda word: True)
    equivalence = ExhaustiveEquivalenceOracle(membership, ALPHABET, max_length=6)
    for learner in (learn_dfa_lstar, learn_dfa_ttt):
        learned = learner(ALPHABET, membership, equivalence, max_rounds=20)
        assert len(list(learned.states())) == 1
        assert learned.recognizes(("a", "b", "a"))


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError):
        learn_dfa_from_language(_parity_dfa(), ALPHABET, algorithm="nope")


def test_language_membership_oracle_rejects_bad_model():
    with pytest.raises(TypeError):
        LanguageMembershipOracle(object())


# ----------------------------------------------------------------------------- Mealy


def test_learns_target_mealy():
    target = _toggle_mealy()
    learned = learn_mealy_from_transducer(target, {"x", "y"}, max_length=10)
    learned.validate()
    assert len(list(learned.states())) == 2
    assert _mealy_equivalent(learned, target, {"x", "y"})


def test_mealy_learns_last_symbol_echo():
    # Mealy machine that outputs whichever input symbol it just read (1-state).
    def echo(word):
        return tuple(word)

    from sofic.automata.active import FunctionMealyOracle, MealyExhaustiveEquivalenceOracle, learn_mealy_lstar

    oracle = FunctionMealyOracle(echo)
    alphabet = ("0", "1")
    equivalence = MealyExhaustiveEquivalenceOracle(oracle, alphabet, max_length=6)
    learned = learn_mealy_lstar(alphabet, oracle, equivalence, max_rounds=20)
    learned.validate()
    assert len(list(learned.states())) == 1
    assert next(iter(learned.transduce(("0", "1", "1")))) == ("0", "1", "1")
