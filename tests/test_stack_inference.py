"""Tests for stack HMM inference (PAPNI, stack-CSSR, Bayes, enumeration)."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.automata.learning.papni import DyckAlphabet, is_well_matched, learn_sofic_dyck_shift_papni, papni_encode
from sofic.automata.learning.rpni import learn_dfa_rpni
from sofic.examples.shifts import dyck_shift_order, motzkin_shift
from sofic.generators.stack_hmm import HiddenMarkovStackModel
from sofic.inference.bayesian.stack_hmm import (
    ModelComparisonStackHMM,
    StackHMMPosterior,
)
from sofic.inference.cssr import cssr
from sofic.inference.cssr.stack import (
    fit_stack_hmm_mle,
    learn_stack_hmm_papni,
    stack_cssr,
    stack_subtree_merge,
)
from sofic.shifts.dyck_enumeration import (
    count_dyck_graph_strings,
    dyck_graph_string_to_shift,
    iter_sofic_dyck_topologies,
    shift_to_dyck_graph_string,
)


def _balanced_dyck_alphabet() -> DyckAlphabet:
    return DyckAlphabet(
        call_alphabet=frozenset({"("}),
        return_alphabet=frozenset({")"}),
        internal_alphabet=frozenset(),
    )


def _uniform_probabilities(shift):
    from sofic.shifts.sofic_dyck import transition_ref

    refs = [transition_ref(transition) for transition in shift.transitions()]
    return {ref: 1.0 / len(refs) for ref in refs}


def test_is_well_matched_and_papni_encode():
    alphabet = _balanced_dyck_alphabet()
    assert is_well_matched(("(", ")"), alphabet)
    assert not is_well_matched((")", "("), alphabet)
    assert papni_encode(("(", ")"), alphabet) == ("(", (")", "("))


def test_rpni_learns_balanced_parentheses_language():
    positive = [("(", ")"), ("(", "(", ")", ")")]
    negative = [(")", "(")]
    dfa = learn_dfa_rpni(positive, negative)
    assert dfa.recognizes(("(", ")"))


def test_papni_recovers_dyck_shift_topology():
    alphabet = _balanced_dyck_alphabet()
    positive = [
        ("(", ")"),
        ("(", "(", ")", ")"),
        ("(", "(", "(", ")", ")", ")"),
    ]
    learned = learn_sofic_dyck_shift_papni(positive, alphabet=alphabet)
    assert learned.is_admissible_word(("(", ")"))
    assert learned.is_admissible_word(("(", "(", ")", ")"))


def test_fit_stack_hmm_mle_assigns_positive_mass():
    shift = dyck_shift_order(1, call_symbols=("(",), return_symbols=(")",))
    sequence = ("(", ")", "(", "(", ")", ")")
    model = fit_stack_hmm_mle(shift, sequence)
    model.validate()
    assert model.word_probability(sequence) > 0.0


def test_learn_stack_hmm_papni_end_to_end():
    alphabet = DyckAlphabet(
        call_alphabet=frozenset({"a"}),
        return_alphabet=frozenset({"A"}),
        internal_alphabet=frozenset(),
    )
    positive = [("a", "A"), ("a", "a", "A", "A")]
    sequence = ("a", "a", "A", "A", "a", "A")
    model = learn_stack_hmm_papni(positive, alphabet=alphabet, sequence=sequence)
    model.validate()
    assert model.word_probability(("a", "A")) > 0.0


def test_stack_cssr_recovers_motzkin_structure():
    shift = motzkin_shift()
    probs = _uniform_probabilities(shift)
    oracle = HiddenMarkovStackModel.from_sofic_dyck_shift(shift, probs)
    rng = np.random.default_rng(0)
    observations, _ = oracle.sample(5000, rng=rng)
    alphabet = DyckAlphabet(
        call_alphabet=shift.call_alphabet,
        return_alphabet=shift.return_alphabet,
        internal_alphabet=shift.internal_alphabet,
    )
    inferred = stack_cssr(observations, alphabet=alphabet, Lmax=3, max_stack_depth=4, alpha=0.001)
    inferred.validate()
    assert inferred.matched_edges
    prefix = tuple(observations[:12])
    assert oracle.word_probability(prefix) > 0.0


def test_stack_subtree_merge_runs_on_sample():
    shift = dyck_shift_order(1)
    probs = _uniform_probabilities(shift)
    oracle = HiddenMarkovStackModel.from_sofic_dyck_shift(shift, probs)
    rng = np.random.default_rng(1)
    observations, _ = oracle.sample(2000, rng=rng)
    alphabet = DyckAlphabet(
        call_alphabet=shift.call_alphabet,
        return_alphabet=shift.return_alphabet,
        internal_alphabet=shift.internal_alphabet,
    )
    inferred = stack_subtree_merge(observations, alphabet=alphabet, L=2, max_stack_depth=3)
    inferred.validate()


def test_dirichlet_stack_hmm_log_evidence():
    shift = dyck_shift_order(1)
    probs = _uniform_probabilities(shift)
    model = HiddenMarkovStackModel.from_sofic_dyck_shift(shift, probs)
    sequence = ("(", ")", "(", ")")
    posterior = StackHMMPosterior(model, sequence, max_stack_depth=4)
    assert np.isfinite(posterior.log_evidence())


def test_model_comparison_stack_hmm_prefers_true_topology():
    shift = dyck_shift_order(1, call_symbols=("(",), return_symbols=(")",))
    probs = _uniform_probabilities(shift)
    true_model = HiddenMarkovStackModel.from_sofic_dyck_shift(shift, probs)
    rng = np.random.default_rng(2)
    observations, _ = true_model.sample(300, rng=rng)

    candidates = [true_model]
    for topology in iter_sofic_dyck_topologies(call_symbols=("(",), return_symbols=(")",)):
        candidates.append(HiddenMarkovStackModel.from_sofic_dyck_shift(topology, _uniform_probabilities(topology)))
    comparison = ModelComparisonStackHMM(candidates, observations, max_stack_depth=4)
    assert np.isfinite(comparison.log_evidences()[0])
    best = comparison.most_probable_model()
    best.validate()


def test_dyck_graph_round_trip():
    shift = dyck_shift_order(1, call_symbols=("(",), return_symbols=(")",))
    spec = shift_to_dyck_graph_string(shift)
    rebuilt = dyck_graph_string_to_shift(spec)
    assert rebuilt.is_admissible_word(("(", ")"))
    assert count_dyck_graph_strings(call_symbols=("(",), return_symbols=(")",)) > 0


@pytest.mark.parametrize("method", ["papni", "stack_cssr", "flat_cssr"])
def test_benchmark_passive_paths(method: str):
    shift = dyck_shift_order(1, call_symbols=("(",), return_symbols=(")",))
    probs = _uniform_probabilities(shift)
    oracle = HiddenMarkovStackModel.from_sofic_dyck_shift(shift, probs)
    rng = np.random.default_rng(3)
    observations, _ = oracle.sample(4000, rng=rng)
    alphabet = DyckAlphabet(
        call_alphabet=shift.call_alphabet,
        return_alphabet=shift.return_alphabet,
        internal_alphabet=shift.internal_alphabet,
    )

    if method == "papni":
        alphabet_bm = DyckAlphabet(
            call_alphabet=shift.call_alphabet,
            return_alphabet=shift.return_alphabet,
            internal_alphabet=shift.internal_alphabet,
        )
        positive = []
        for end in range(2, len(observations) + 1):
            prefix = tuple(observations[:end])
            if is_well_matched(prefix, alphabet_bm):
                positive.append(prefix)
        if not positive:
            positive = [("(", ")")]
        inferred = learn_stack_hmm_papni(positive, alphabet=alphabet_bm, sequence=observations)
    elif method == "stack_cssr":
        inferred = stack_cssr(observations, alphabet=alphabet, Lmax=3, max_stack_depth=4, alpha=0.001)
    else:
        flat = cssr(observations, Lmax=3, alpha=0.001)
        inferred = HiddenMarkovStackModel(
            call_alphabet=alphabet.call_alphabet,
            return_alphabet=alphabet.return_alphabet,
            internal_alphabet=alphabet.internal_alphabet,
            initial_distribution={"s0": 1.0},
        )
        inferred.graph.add_state("s0")
        for transition in flat.transitions():
            symbol = transition.data.get("emission")
            prob = float(transition.data.get("prob", 0.0))
            if symbol in alphabet.call_alphabet:
                inferred.add_call_transition("s0", "s0", symbol, prob)
            elif symbol in alphabet.return_alphabet:
                inferred.add_return_transition("s0", "s0", symbol, prob)
            else:
                inferred.add_internal_transition("s0", "s0", symbol, prob)
        from sofic.graph import KIND_CALL, KIND_RETURN

        for call in inferred.transitions():
            if call.data.get("kind") != KIND_CALL:
                continue
            call_ref = (call.source, call.target, call.key)
            for ret in inferred.transitions():
                if ret.data.get("kind") != KIND_RETURN:
                    continue
                ret_ref = (ret.source, ret.target, ret.key)
                inferred.add_matched_pair(call_ref, ret_ref)

    inferred.validate()
    assert list(inferred.transitions())
    oracle_prob = oracle.word_probability(tuple(observations[:12]))
    assert oracle_prob > 0.0


def _held_out_bits_per_symbol(model, word) -> float:
    return -np.log2(model.word_probability(tuple(word))) / len(word)


@pytest.mark.parametrize("Lmax", [2, 3])
def test_stack_cssr_matches_motzkin_likelihood(Lmax: int):
    """Regression: homogenization only reached stacks of depth <= Lmax and return
    edges were paired with unobserved calls, so held-out words got probability 0."""
    shift = motzkin_shift()
    oracle = HiddenMarkovStackModel.from_sofic_dyck_shift(shift, _uniform_probabilities(shift))
    alphabet = DyckAlphabet(
        call_alphabet=shift.call_alphabet,
        return_alphabet=shift.return_alphabet,
        internal_alphabet=shift.internal_alphabet,
    )
    observations, _ = oracle.sample(5000, rng=np.random.default_rng(0))
    held_out, _ = oracle.sample(60, rng=np.random.default_rng(99))
    inferred = stack_cssr(observations, alphabet=alphabet, Lmax=Lmax, max_stack_depth=4, alpha=0.001)
    inferred.validate()
    assert len(list(inferred.states())) == 1
    assert _held_out_bits_per_symbol(inferred, held_out) == pytest.approx(
        _held_out_bits_per_symbol(oracle, held_out), abs=0.05
    )


def test_stack_cssr_exact_and_bonferroni_run():
    shift = motzkin_shift()
    oracle = HiddenMarkovStackModel.from_sofic_dyck_shift(shift, _uniform_probabilities(shift))
    observations, _ = oracle.sample(3000, rng=np.random.default_rng(1))
    alphabet = DyckAlphabet(
        call_alphabet=shift.call_alphabet,
        return_alphabet=shift.return_alphabet,
        internal_alphabet=shift.internal_alphabet,
    )
    for kwargs in ({"test": "exact"}, {"correction": "bonferroni"}):
        inferred = stack_cssr(observations, alphabet=alphabet, Lmax=3, max_stack_depth=4, alpha=0.001, **kwargs)
        inferred.validate()
        assert oracle.word_probability(tuple(observations[:12])) > 0.0
    with pytest.raises(ValueError, match="unknown correction"):
        stack_cssr(observations, alphabet=alphabet, Lmax=2, correction="holm")


@pytest.mark.parametrize("seed", [1, 5])
def test_stack_cssr_keeps_histories_when_splitting(seed):
    """Regression: determinization dropped histories that never emitted the splitting
    symbol, leaving states whose successors were all unplaced (zero outgoing mass)."""
    from sofic.shifts.sofic_dyck import transition_ref

    shift = motzkin_shift()
    rng = np.random.default_rng(seed)
    refs = [transition_ref(transition) for transition in shift.transitions()]
    weights = rng.dirichlet(np.ones(len(refs)))
    oracle = HiddenMarkovStackModel.from_sofic_dyck_shift(shift, dict(zip(refs, map(float, weights), strict=True)))
    observations, _ = oracle.sample(4000, rng=rng)
    alphabet = DyckAlphabet(
        call_alphabet=shift.call_alphabet,
        return_alphabet=shift.return_alphabet,
        internal_alphabet=shift.internal_alphabet,
    )
    inferred = stack_cssr(observations, alphabet=alphabet, Lmax=3, max_stack_depth=4, alpha=0.01)
    inferred.validate()
    held_out, _ = oracle.sample(400, rng=np.random.default_rng(100 + seed))
    windows = [tuple(held_out[i : i + 6]) for i in range(0, 390, 6)]
    supported = sum(inferred.word_probability(window) > 0.0 for window in windows)
    assert supported >= 0.9 * len(windows)
