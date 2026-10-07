"""Sanity checks of the brute-force oracles on hand-built models with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sofic.automata.buchi import BuchiAutomaton
from sofic.automata.nfa import NFA
from sofic.automata.nwa import NestedWord, NestedWordAutomaton
from sofic.automata.transducers import MealyMachine
from sofic.automata.vpa import VisiblyPushdownAutomaton
from sofic.generators.mealy import MealyHMM
from sofic.graph import EPSILON, KIND_CALL, KIND_RETURN
from tests import oracles


def test_words_enumerates_shortlex():
    assert oracles.words("ab", 2) == [(), ("a",), ("b",), ("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]
    assert oracles.words("ab", 2, min_length=2) == [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]


def _ends_in_one_nfa() -> NFA:
    nfa = NFA(input_alphabet=frozenset("01"), initial_states=frozenset({0}), accepting_states=frozenset({2}))
    nfa.add_transition(0, 0, "0")
    nfa.add_transition(0, 0, "1")
    nfa.add_transition(0, 1, "1")
    nfa.add_transition(1, 2, EPSILON)
    return nfa


def test_nfa_membership_with_epsilon():
    nfa = _ends_in_one_nfa()
    expected = frozenset(w for w in oracles.words("01", 4) if w and w[-1] == "1")
    assert oracles.language(lambda w: oracles.nfa_accepts(nfa, w), "01", 4) == expected


def test_nerode_class_count():
    # "ends in 1" has a 2-state minimal DFA; "even number of 1s and length divisible by 2" has 4.
    assert oracles.nerode_class_count(lambda w: bool(w) and w[-1] == "1", "01", 2) == 2
    assert oracles.nerode_class_count(lambda w: w.count("1") % 2 == 0 and len(w) % 2 == 0, "01", 3) == 4
    assert oracles.nerode_class_count(lambda w: True, "01", 3) == 1


def _noisy_flip_hmm() -> MealyHMM:
    """Two states; from each, stay w.p. 0.7 emitting the state's bit, else switch emitting the other bit."""
    hmm = MealyHMM(initial_distribution={"A": 0.5, "B": 0.5})
    hmm.add_transition("A", "A", "0", 0.7)
    hmm.add_transition("A", "B", "1", 0.3)
    hmm.add_transition("B", "B", "1", 0.6)
    hmm.add_transition("B", "A", "0", 0.4)
    hmm.add_transition("B", "A", "1", 0.0)
    return hmm


def test_hmm_word_probabilities_and_block_entropy():
    hmm = _noisy_flip_hmm()
    assert oracles.word_probability(hmm, ()) == pytest.approx(1.0)
    # The emitted symbol determines the next state; P(0) = 0.5*0.7 + 0.5*0.4.
    assert oracles.word_probability(hmm, ("0",)) == pytest.approx(0.55)
    assert oracles.word_probability(hmm, ("0", "1")) == pytest.approx(0.55 * 0.3)
    for length in range(4):
        assert sum(oracles.word_distribution(hmm, length).values()) == pytest.approx(1.0)
    dist = oracles.word_distribution(hmm, 1)
    assert oracles.block_entropy(dist) == pytest.approx(-(0.55 * math.log2(0.55) + 0.45 * math.log2(0.45)))
    assert oracles.block_entropy({"x": 0.5, "y": 0.5}) == pytest.approx(1.0)
    assert oracles.block_entropy({"x": 1.0}) == 0.0


def test_forward_backward_gamma_xi_consistency():
    hmm = _noisy_flip_hmm()
    obs = ("0", "1", "1")
    states = ["A", "B"]
    alpha = oracles.forward(hmm, obs, states=states)
    beta = oracles.backward(hmm, obs, states=states)
    total = oracles.word_probability(hmm, obs)
    np.testing.assert_allclose(alpha[0], [0.5, 0.5])
    np.testing.assert_allclose(beta[-1], [1.0, 1.0])
    np.testing.assert_allclose((alpha * beta).sum(axis=1), total)
    gamma = oracles.gamma(hmm, obs, states=states)
    np.testing.assert_allclose(gamma, alpha * beta / total)
    xi = oracles.xi(hmm, obs, states=states)
    np.testing.assert_allclose(xi.sum(axis=2), gamma[:-1])
    np.testing.assert_allclose(xi.sum(axis=1), gamma[1:])
    assert not oracles.gamma(hmm, ("2",), states=states).any()


def test_viterbi_paths():
    hmm = _noisy_flip_hmm()
    best, paths = oracles.viterbi_paths(hmm, ("0", "0"))
    # A->A->A emits 00 with 0.5*0.7*0.7; B->A->A gives 0.5*0.4*0.7.
    assert best == pytest.approx(0.5 * 0.7 * 0.7)
    assert paths == [("A", "A", "A")]
    assert oracles.path_probability(hmm, ("B", "A", "A"), ("0", "0")) == pytest.approx(0.5 * 0.4 * 0.7)
    assert oracles.viterbi_paths(hmm, ("2",)) == (0.0, [])
    symmetric = MealyHMM(initial_distribution={0: 0.5, 1: 0.5})
    symmetric.add_transition(0, 0, "x", 1.0)
    symmetric.add_transition(1, 1, "x", 1.0)
    assert sorted(oracles.viterbi_paths(symmetric, ("x",))[1]) == [(0, 0), (1, 1)]


def _dyck_vpa(*, bottom: bool, wildcard: bool = False) -> VisiblyPushdownAutomaton:
    vpa = VisiblyPushdownAutomaton(
        call_alphabet=frozenset({"c"}),
        return_alphabet=frozenset({"r"}),
        internal_alphabet=frozenset({"i"}),
        stack_alphabet=frozenset({"A"} | ({"Z"} if bottom else set())),
        bottom_stack_symbol="Z" if bottom else None,
        initial_state=0,
        accepting_states=frozenset({0}),
    )
    vpa.graph.add_state(0)
    vpa.add_call_transition(0, 0, "c", "A")
    vpa.add_return_transition(0, 0, "r", None if wildcard else "A")
    vpa.add_internal_transition(0, 0, "i")
    return vpa


def test_vpa_membership_semantics():
    plain = _dyck_vpa(bottom=False)
    assert oracles.vpa_accepts(plain, tuple("cicr"))
    assert oracles.vpa_accepts(plain, tuple("cc"))  # pending calls allowed
    assert not oracles.vpa_accepts(plain, tuple("r"))  # pending return needs a bottom symbol
    wildcard_no_bottom = _dyck_vpa(bottom=False, wildcard=True)
    assert not oracles.vpa_accepts(wildcard_no_bottom, tuple("r"))
    wildcard_bottom = _dyck_vpa(bottom=True, wildcard=True)
    assert oracles.vpa_accepts(wildcard_bottom, tuple("rcr"))
    bottom_guard = _dyck_vpa(bottom=True)
    bottom_guard.add_return_transition(0, 0, "r", "Z")
    assert oracles.vpa_accepts(bottom_guard, tuple("rrcr"))
    assert oracles.vpa_accepts(bottom_guard, tuple("cr"))
    only_bottom = _dyck_vpa(bottom=True)
    only_bottom.graph = type(only_bottom.graph)()
    only_bottom.graph.add_state(0)
    only_bottom.add_call_transition(0, 0, "c", "A")
    only_bottom.add_return_transition(0, 0, "r", "Z")
    assert oracles.vpa_accepts(only_bottom, tuple("r"))
    assert not oracles.vpa_accepts(only_bottom, tuple("cr"))  # bottom guard never pops a pushed symbol


def test_nwa_membership_semantics():
    nwa = NestedWordAutomaton(
        call_alphabet=frozenset({"c"}),
        return_alphabet=frozenset({"r"}),
        internal_alphabet=frozenset({"i"}),
        hier_alphabet=frozenset({"A", "B", "Z"}),
        bottom_hier_state="Z",
        initial_state=0,
        accepting_states=frozenset({0}),
    )
    nwa.graph.add_state(0)
    nwa.add_call_transition(0, 0, "c", "A")
    nwa.add_return_transition(0, 0, "r", "A")
    assert oracles.nwa_accepts(nwa, tuple("ccr"))
    assert not oracles.nwa_accepts(nwa, tuple("r"))
    nwa.add_return_transition(0, 0, "r", "Z")
    assert oracles.nwa_accepts(nwa, tuple("rcr"))
    # Explicit nested words: the same return may be matched or pending.
    matched = NestedWord(symbols=("c", "r"), kinds=(KIND_CALL, KIND_RETURN), matching=(1, 0))
    unmatched = NestedWord(symbols=("c", "r"), kinds=(KIND_CALL, KIND_RETURN), matching=(None, None))
    assert oracles.nwa_accepts(nwa, matched)
    assert oracles.nwa_accepts(nwa, unmatched)
    nwa.graph = type(nwa.graph)()
    nwa.graph.add_state(0)
    nwa.add_call_transition(0, 0, "c", "B")
    nwa.add_return_transition(0, 0, "r", "A")
    assert not oracles.nwa_accepts(nwa, matched)


def test_buchi_lasso_acceptance():
    # Infinitely many 1s.
    ba = BuchiAutomaton(input_alphabet=frozenset("01"), initial_states=frozenset({0}), accepting_states=frozenset({1}))
    for source in (0, 1):
        ba.add_transition(source, 0, "0")
        ba.add_transition(source, 1, "1")
    assert oracles.buchi_accepts_lasso(ba, (), ("0", "1"))
    assert oracles.buchi_accepts_lasso(ba, ("1",), ("1",))
    assert not oracles.buchi_accepts_lasso(ba, ("1", "1"), ("0",))
    # Finitely many 1s (needs nondeterminism): guess the last 1, then loop on 0 in accepting state.
    fin = BuchiAutomaton(input_alphabet=frozenset("01"), initial_states=frozenset({0}), accepting_states=frozenset({1}))
    fin.add_transition(0, 0, "0")
    fin.add_transition(0, 0, "1")
    fin.add_transition(0, 1, EPSILON)
    fin.add_transition(1, 1, "0")
    assert oracles.buchi_accepts_lasso(fin, ("1", "1"), ("0",))
    assert not oracles.buchi_accepts_lasso(fin, (), ("0", "1"))
    # An epsilon self-loop on an accepting state does not count as an infinite run.
    eps = BuchiAutomaton(input_alphabet=frozenset("0"), initial_states=frozenset({0}), accepting_states=frozenset({1}))
    eps.add_transition(0, 1, EPSILON)
    eps.add_transition(1, 1, EPSILON)
    eps.add_transition(1, 0, "0")
    eps.add_transition(0, 0, "0")
    assert oracles.buchi_accepts_lasso(eps, (), ("0",))
    eps.graph = type(eps.graph)()
    eps.add_transition(0, 0, "0")
    eps.add_transition(0, 1, EPSILON)
    eps.add_transition(1, 1, EPSILON)
    assert not oracles.buchi_accepts_lasso(eps, (), ("0",))
    with pytest.raises(ValueError):
        oracles.buchi_accepts_lasso(eps, (), ())


def test_transducer_relation_and_composition():
    # Nondeterministically copy each bit or erase it.
    copy_or_erase = MealyMachine(
        input_alphabet=frozenset("01"), output_alphabet=frozenset("01"), initial_states=frozenset({0})
    )
    for bit in "01":
        copy_or_erase.add_transition(0, 0, bit, bit)
        copy_or_erase.add_transition(0, 0, bit, EPSILON)
    assert oracles.transducer_outputs(copy_or_erase, ("0", "1")) == {(), ("0",), ("1",), ("0", "1")}
    flip = MealyMachine(input_alphabet=frozenset("01"), output_alphabet=frozenset("01"), initial_states=frozenset({0}))
    flip.add_transition(0, 0, "0", "1")
    flip.add_transition(0, 0, "1", "0")
    relation = oracles.transducer_relation(copy_or_erase, [("0", "1")])
    composed = oracles.compose_relations(relation, lambda y: oracles.transducer_outputs(flip, y))
    assert composed == {("0", "1"): {(), ("1",), ("0",), ("1", "0")}}
    assert oracles.compose_relations({("a",): {("b",)}}, {("b",): {("c",)}}) == {("a",): {("c",)}}
    pump = MealyMachine(input_alphabet=frozenset("0"), output_alphabet=frozenset("x"), initial_states=frozenset({0}))
    pump.add_transition(0, 0, EPSILON, "x")
    with pytest.raises(ValueError, match="productive"):
        oracles.transducer_outputs(pump, ())
