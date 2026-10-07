"""Conditioning a stationary process on a regular constraint (Doob h-transform)."""

from __future__ import annotations

import math

import pytest

from sofic.automata.dfa import DFA
from sofic.automata.nfa import NFA
from sofic.examples import bernoulli, binary_markov_chain, golden_mean
from sofic.exceptions import NonDeterministicError
from sofic.generators.conditioning import condition_on_language
from sofic.generators.mealy import MealyHMM
from sofic.shifts.sft import ShiftOfFiniteType
from sofic.shifts.topological_anatomy import parry_measure_sofic

PHI = (1 + math.sqrt(5)) / 2


def _uniform(symbols: str) -> MealyHMM:
    hmm = MealyHMM(observation_alphabet=frozenset(symbols), initial_distribution={"A": 1.0})
    for symbol in symbols:
        hmm.add_transition("A", "A", symbol, 1 / len(symbols))
    return hmm


def _sft(forbidden: set[str], symbols: str) -> ShiftOfFiniteType:
    return ShiftOfFiniteType.from_forbidden_words({tuple(word) for word in forbidden}, frozenset(symbols))


def _language_dfa(shift: ShiftOfFiniteType) -> DFA:
    states = frozenset(shift.states())
    nfa = NFA(
        graph=shift.graph.copy(),
        input_alphabet=shift.symbol_alphabet,
        initial_states=states,
        accepting_states=states,
    )
    return nfa.determinize()


def _dfa(edges: list[tuple[str, str, str]], accepting: str, initial: str = "q") -> DFA:
    dfa = DFA(
        input_alphabet=frozenset("01"), initial_states=frozenset({initial}), accepting_states=frozenset(accepting)
    )
    dfa.graph.add_state(initial)
    for source, target, _symbol in edges:
        dfa.graph.add_state(source)
        dfa.graph.add_state(target)
    for source, target, symbol in edges:
        dfa.add_transition(source, target, symbol)
    return dfa


def _stationary_words(model, length: int) -> dict[tuple[str, ...], float]:
    return model.word_probabilities(length, start=model.to_mealy().stationary_distribution())


def _window_law(hmm, dfa: DFA, n: int, offset: int, width: int) -> dict[tuple[str, ...], float]:
    """Brute-force law of ``X_{offset:offset+width}`` given ``X_{0:n} ∈ L``."""
    law: dict[tuple[str, ...], float] = {}
    for word, prob in _stationary_words(hmm, n).items():
        if dfa.recognizes(word):
            window = word[offset : offset + width]
            law[window] = law.get(window, 0.0) + prob
    total = sum(law.values())
    return {window: prob / total for window, prob in law.items()}


def _distance(a: dict, b: dict) -> float:
    return max(abs(a.get(word, 0.0) - b.get(word, 0.0)) for word in set(a) | set(b))


def test_fair_coin_conditioned_on_golden_mean_is_the_parry_measure():
    shift = _sft({"11"}, "01")
    conditioned = condition_on_language(bernoulli(), _language_dfa(shift))
    assert conditioned.is_equal_process(golden_mean(1 / PHI), rtol=1e-7, atol=1e-10)
    assert conditioned.entropy_rate() == pytest.approx(shift.topological_entropy(), abs=1e-10)
    assert conditioned.entropy_rate() == pytest.approx(math.log2(PHI), abs=1e-10)


@pytest.mark.parametrize(
    ("forbidden", "symbols"),
    [({"11"}, "01"), ({"111"}, "01"), ({"11", "20"}, "012"), ({"00", "12", "21"}, "012")],
)
def test_uniform_iid_conditioned_on_sft_is_its_parry_measure(forbidden, symbols):
    shift = _sft(forbidden, symbols)
    conditioned = condition_on_language(_uniform(symbols), _language_dfa(shift))
    parry = parry_measure_sofic(shift)
    for length in range(1, 6):
        assert _distance(_stationary_words(conditioned, length), _stationary_words(parry, length)) < 1e-10
    assert conditioned.entropy_rate() == pytest.approx(shift.topological_entropy(), abs=1e-10)


@pytest.mark.parametrize("model", [binary_markov_chain(0.4, 0.3), binary_markov_chain(0.7, 0.2), golden_mean(0.3)])
def test_conditioned_markov_chain_is_the_limit_of_conditioned_windows(model):
    dfa = _language_dfa(_sft({"00"}, "01"))
    conditioned = condition_on_language(model, dfa)
    width = 2
    limit = _stationary_words(conditioned, width)
    errors = [_distance(_window_law(model, dfa, n, (n - width) // 2, width), limit) for n in range(4, 13, 2)]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(errors, errors[1:], strict=False))
    assert errors[-1] < 1e-4


def test_conditioning_on_the_full_language_changes_nothing():
    universal = _dfa([("q", "q", "0"), ("q", "q", "1")], "q")
    model = binary_markov_chain(0.4, 0.3)
    conditioned = condition_on_language(model, universal)
    for length in range(1, 5):
        assert _distance(_stationary_words(conditioned, length), _stationary_words(model, length)) < 1e-10


def test_incomplete_dfa_rejects_missing_transitions():
    no_ones = _dfa([("q", "q", "0")], "q")
    conditioned = condition_on_language(bernoulli(0.3), no_ones)
    assert _stationary_words(conditioned, 3) == pytest.approx({("0", "0", "0"): 1.0})
    assert conditioned.entropy_rate() == pytest.approx(0.0, abs=1e-12)


_SPLIT = [("s", "a", "0"), ("s", "b", "1"), ("a", "a", "0"), ("b", "b", "1")]


def test_two_maximal_components_are_rejected():
    split = _dfa(_SPLIT, "sab", initial="s")
    with pytest.raises(ValueError, match="maximal growth"):
        condition_on_language(bernoulli(), split)


def test_dominant_component_wins():
    split = _dfa(_SPLIT, "sab", initial="s")
    conditioned = condition_on_language(bernoulli(0.8), split)
    assert _stationary_words(conditioned, 2) == pytest.approx({("1", "1"): 1.0})


def test_empty_constraints_are_rejected():
    rejecting = _dfa([("q", "q", "0")], "")
    with pytest.raises(ValueError, match="empty word"):
        condition_on_language(bernoulli(), rejecting)
    finite = _dfa([("q", "r", "0")], "qr")
    with pytest.raises(ValueError, match="no infinite word"):
        condition_on_language(bernoulli(), finite)


def test_nondeterministic_constraint_is_rejected():
    nfa = NFA(input_alphabet=frozenset("01"), initial_states=frozenset({"q", "r"}), accepting_states=frozenset({"q"}))
    nfa.add_transition("q", "q", "0")
    with pytest.raises(NonDeterministicError):
        condition_on_language(bernoulli(), nfa)


def test_a_process_already_in_the_constraint_is_unchanged():
    model = golden_mean(0.3)
    conditioned = condition_on_language(model, _language_dfa(_sft({"11"}, "01")))
    assert conditioned.is_equal_process(model, rtol=1e-7, atol=1e-10)
    assert conditioned.observation_alphabet == frozenset("01")
    assert {state[0] for state in conditioned.states()} <= set(model.states())
