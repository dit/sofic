"""Tests for reusable Hypothesis strategies."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.automata.enumeration.icdfa import dfa_to_icdfa_string
from sofic.automata.wheeler import is_wheeler
from sofic.graph import ATTR_SYMBOL, EPSILON
from sofic.testing.strategies import (
    buchi_automata,
    dfas,
    epsilon_machines,
    lassos,
    markov_chains,
    mealy_hmms,
    mealy_transducers,
    nfas,
    nwas,
    sfts,
    sofic_shifts,
    vpas,
    wheeler_nfas,
)


@given(dfa=dfas(max_states=3))
@settings(max_examples=25)
def test_dfas_strategy_generates_valid_complete_icdfas(dfa):
    dfa.validate()

    assert dfa.is_deterministic()
    assert len(dfa.initial_states) == 1
    for state in dfa.states():
        for symbol in dfa.input_alphabet:
            assert len(dfa.delta(state, symbol)) == 1

    encoded = dfa_to_icdfa_string(dfa, symbol_order=tuple(sorted(dfa.input_alphabet)))
    assert encoded.n == len(list(dfa.states()))
    assert encoded.k == len(dfa.input_alphabet)


@given(machine=epsilon_machines(max_states=3))
@settings(max_examples=25)
def test_epsilon_machines_strategy_generates_valid_models(machine):
    machine.validate()

    assert machine.is_unifilar()
    assert machine.is_irreducible()
    assert machine.is_stationary()
    assert machine.observation_alphabet == frozenset({"0", "1"})
    assert sum(machine.initial_distribution.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "strategy",
    [
        nfas(),
        nfas(allow_epsilon=False, max_states=4),
        buchi_automata(),
        wheeler_nfas(),
        markov_chains(),
        mealy_hmms(),
        sofic_shifts(),
        sfts(),
        vpas(),
        nwas(),
        mealy_transducers(),
    ],
    ids=[
        "nfas",
        "nfas-no-epsilon",
        "buchi_automata",
        "wheeler_nfas",
        "markov_chains",
        "mealy_hmms",
        "sofic_shifts",
        "sfts",
        "vpas",
        "nwas",
        "mealy_transducers",
    ],
)
@settings(max_examples=25)
@given(data=st.data())
def test_new_strategies_generate_valid_models(strategy, data):
    model = data.draw(strategy)
    model.validate()


@given(nfa=nfas(allow_epsilon=False))
@settings(max_examples=25)
def test_nfas_without_epsilon_have_no_epsilon_edges(nfa):
    assert nfa.initial_states
    assert all(t.data[ATTR_SYMBOL] is not EPSILON for t in nfa.transitions())


@given(nfa=wheeler_nfas(max_states=5))
@settings(max_examples=50)
def test_wheeler_nfas_are_wheeler(nfa):
    nfa.validate()
    assert is_wheeler(nfa)


@given(hmm=mealy_hmms())
@settings(max_examples=25)
def test_mealy_hmms_rows_are_stochastic(hmm):
    assert sum(hmm.initial_distribution.values()) == pytest.approx(1.0)
    assert all(prob > 0 for prob in hmm.initial_distribution.values())
    for state in hmm.states():
        probs = [t.data["prob"] for t in hmm.graph.out_transitions(state)]
        assert probs
        assert all(prob > 0 for prob in probs)
        assert sum(probs) == pytest.approx(1.0)


@given(lasso=lassos(max_prefix_length=2, max_loop_length=2))
@settings(max_examples=25)
def test_lassos_have_nonempty_loops(lasso):
    prefix, loop = lasso
    assert len(prefix) <= 2
    assert 1 <= len(loop) <= 2


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (vpas, {"call_alphabet": ("x",), "return_alphabet": ("x",)}, "must be disjoint"),
        (nwas, {"hier_alphabet": ("A", "Z")}, "bottom symbol"),
        (sfts, {"max_word_length": 0}, "max_word_length must be positive"),
        (nfas, {"max_transitions": -1}, "max_transitions must be nonnegative"),
        (lassos, {"max_loop_length": 0}, "max_loop_length must be positive"),
    ],
)
def test_new_strategies_validate_arguments(factory, kwargs, message):
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alphabet": ()}, "alphabet must be non-empty"),
        ({"alphabet": ("0", "0")}, "alphabet symbols must be unique"),
        ({"min_states": 0}, "min_states must be positive"),
        ({"min_states": 3, "max_states": 2}, "min_states must be less than or equal to max_states"),
    ],
)
def test_dfas_strategy_validates_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        dfas(**kwargs)


def test_epsilon_machines_strategy_rejects_empty_enumeration():
    with pytest.raises(ValueError, match="no topological epsilon-machine strings"):
        epsilon_machines(alphabet=("0",), min_states=2, max_states=2)
