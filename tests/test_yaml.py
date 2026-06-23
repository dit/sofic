"""Tests for YAML model serialization."""

from __future__ import annotations

import numpy as np
import pytest

from pensive.automata.atomaton import Atomaton
from pensive.automata.dfa import DFA
from pensive.automata.nfa import NFA
from pensive.automata.nwa import NestedWordAutomaton
from pensive.automata.transducers import MealyMachine, MooreMachine
from pensive.automata.vpa import (
    CallDrivenAutomaton,
    CanonicalVisiblyPushdownAutomaton,
    CompositeVisiblyPushdownAutomaton,
    DeterministicVisiblyPushdownAutomaton,
    MultipleEntryVisiblyPushdownAutomaton,
    SingleEntryVisiblyPushdownAutomaton,
    VisiblyPushdownAutomaton,
)
from pensive.examples.epsilon_machines import bernoulli, golden_mean
from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from pensive.generators.markov import MarkovChain
from pensive.generators.mealy import MealyHMM
from pensive.generators.moore import MooreHMM
from pensive.generators.nmachine import NMachine
from pensive.generators.pfa import ProbabilisticFiniteAutomaton
from pensive.generators.quasi_realization import QuasiRealization
from pensive.generators.stack_hmm import HiddenMarkovStackModel
from pensive.graph import (
    ATTR_EMISSION,
    ATTR_KIND,
    ATTR_OUTPUT,
    ATTR_QUASIPROB,
    ATTR_STACK_SYMBOL,
    ATTR_SYMBOL,
    EPSILON,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
)
from pensive.serialization import model_from_yaml, model_to_dict
from pensive.shifts.markov_dyck import MarkovDyckShift
from pensive.shifts.sft import ShiftOfFiniteType
from pensive.shifts.sofic import SoficShift
from pensive.shifts.sofic_dyck import SoficDyckShift
from pensive.shifts.tmc import TopologicalMarkovChain


def _round_trip(model):
    restored = type(model).from_yaml(model.to_yaml())
    assert type(restored) is type(model)
    assert model_to_dict(restored) == model_to_dict(model)
    restored.validate()
    return restored


def _metadata_keys(model) -> set[str]:
    return {item["key"] for item in model_to_dict(model)["metadata"]["items"]}


def test_nfa_round_trip_preserves_epsilon_tuple_state_and_edge_key():
    nfa = NFA(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({("q", 0)}),
        accepting_states=frozenset({("q", 1)}),
    )
    nfa.graph.add_state(("q", 0), label=("start", 0))
    nfa.graph.add_state(("q", 1))
    key = nfa.add_transition(("q", 0), ("q", 1), EPSILON)
    nfa.add_transition(("q", 1), ("q", 1), "a")

    restored = _round_trip(nfa)
    transition = next(t for t in restored.transitions() if t.key == key)
    assert transition.data[ATTR_SYMBOL] is EPSILON


def test_classmethod_from_yaml_rejects_unexpected_model_type():
    nfa = NFA(input_alphabet=frozenset(), initial_states=frozenset(), accepting_states=frozenset())
    with pytest.raises(TypeError, match="not a DFA"):
        DFA.from_yaml(nfa.to_yaml())


def test_labeled_automata_round_trip():
    dfa = DFA(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    dfa.graph.add_state("q0")
    dfa.graph.add_state("q1")
    dfa.add_transition("q0", "q1", "a")
    dfa.add_transition("q1", "q1", "a")
    _round_trip(dfa)

    atomaton = Atomaton(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q0"}),
    )
    atomaton.graph.add_state("q0")
    atomaton.add_transition("q0", "q0", "a")
    _round_trip(atomaton)


def test_transducers_round_trip():
    mealy = MealyMachine(
        input_alphabet=frozenset({"a"}),
        output_alphabet=frozenset({"0", "1"}),
        initial_states=frozenset({"q0"}),
    )
    mealy.graph.add_state("q0")
    mealy.graph.add_transition("q0", "q0", **{ATTR_SYMBOL: "a", ATTR_OUTPUT: "1"})
    _round_trip(mealy)

    moore = MooreMachine(
        input_alphabet=frozenset({"a"}),
        output_alphabet=frozenset({"0"}),
        initial_states=frozenset({"q0"}),
    )
    moore.graph.add_state("q0", **{ATTR_OUTPUT: "0"})
    moore.graph.add_transition("q0", "q0", **{ATTR_SYMBOL: "a"})
    _round_trip(moore)


def test_nested_word_automaton_round_trip():
    nwa = NestedWordAutomaton(
        call_alphabet=frozenset({"("}),
        return_alphabet=frozenset({")"}),
        internal_alphabet=frozenset({"i"}),
        hier_alphabet=frozenset({"S"}),
        initial_state="q",
        accepting_states=frozenset({"q"}),
    )
    nwa.graph.add_state("q")
    nwa.add_call_transition("q", "q", "(", "S")
    nwa.add_return_transition("q", "q", ")", "S")
    nwa.add_internal_transition("q", "q", "i")
    _round_trip(nwa)


def _base_vpa(cls=VisiblyPushdownAutomaton):
    vpa = cls(
        call_alphabet=frozenset({"c"}),
        return_alphabet=frozenset({"r"}),
        internal_alphabet=frozenset({"i"}),
        stack_alphabet=frozenset({"S"}),
        initial_state="q",
        accepting_states=frozenset({"q"}),
    )
    vpa.graph.add_state("q")
    vpa.graph.add_transition("q", "q", **{ATTR_KIND: KIND_CALL, ATTR_SYMBOL: "c", ATTR_STACK_SYMBOL: "S"})
    vpa.graph.add_transition("q", "q", **{ATTR_KIND: KIND_RETURN, ATTR_SYMBOL: "r", ATTR_STACK_SYMBOL: "S"})
    vpa.graph.add_transition("q", "q", **{ATTR_KIND: KIND_INTERNAL, ATTR_SYMBOL: "i"})
    return vpa


def test_vpa_variants_round_trip():
    _round_trip(_base_vpa())
    _round_trip(_base_vpa(DeterministicVisiblyPushdownAutomaton))

    cda = CallDrivenAutomaton(
        call_alphabet=frozenset({"c"}),
        return_alphabet=frozenset({"r"}),
        stack_alphabet=frozenset({"m"}),
        initial_state="m",
        accepting_states=frozenset({"m"}),
        modules={"main": {"m"}, "proc": {"e"}},
        base_module="main",
        call_partition={"c": "proc"},
        call_entries={"c": "e"},
    )
    for state in ("m", "e"):
        cda.graph.add_state(state)
    cda.add_call_transition("m", "e", "c", "m")
    cda.add_return_transition("e", "m", "r", "m")
    _round_trip(cda)

    mevpa = MultipleEntryVisiblyPushdownAutomaton(
        call_alphabet=cda.call_alphabet,
        return_alphabet=cda.return_alphabet,
        stack_alphabet=cda.stack_alphabet,
        initial_state=cda.initial_state,
        accepting_states=cda.accepting_states,
        modules=cda.modules,
        base_module=cda.base_module,
        call_partition=cda.call_partition,
        call_entries=cda.call_entries,
        entry_states={"main": {"m"}, "proc": {"e"}},
        graph=cda.graph.copy(),
    )
    _round_trip(mevpa)

    sevpa = SingleEntryVisiblyPushdownAutomaton(
        call_alphabet=frozenset({"c"}),
        return_alphabet=frozenset({"r"}),
        stack_alphabet=frozenset({("m", "c")}),
        initial_state="m",
        accepting_states=frozenset({"m"}),
        modules={"main": {"m"}, "proc": {"e"}},
        base_module="main",
        call_partition={"c": "proc"},
        call_entries={"c": "e"},
        entry_states={"proc": "e"},
    )
    for state in ("m", "e"):
        sevpa.graph.add_state(state)
    sevpa.add_call_transition("m", "e", "c", ("m", "c"))
    sevpa.add_return_transition("e", "m", "r", ("m", "c"))
    _round_trip(sevpa)

    canonical = CanonicalVisiblyPushdownAutomaton(
        internal_alphabet=frozenset({"i"}),
        initial_state="q",
        accepting_states=frozenset({"q"}),
        summary_representatives={"q": (0, None)},
    )
    canonical.graph.add_state("q")
    canonical.add_internal_transition("q", "q", "i")
    _round_trip(canonical)


def test_composite_vpa_round_trip():
    union = CompositeVisiblyPushdownAutomaton(operation="union", operands=(_base_vpa(), _base_vpa()))
    restored = _round_trip(union)
    assert restored.operation == "union"
    assert len(restored.operands) == 2


def test_stochastic_generators_round_trip():
    chain = MarkovChain(initial_distribution={"A": 1.0})
    chain.graph.add_state("A")
    chain.add_transition("A", "A", 1.0)
    _round_trip(chain)

    mealy = MealyHMM(initial_distribution={"q": 1.0}, observation_alphabet=frozenset({"0"}))
    mealy.graph.add_state("q")
    mealy.add_transition("q", "q", "0", 1.0)
    restored_mealy = _round_trip(mealy)
    assert _metadata_keys(mealy) == {"initial_distribution"}
    assert restored_mealy.observation_alphabet == mealy.observation_alphabet

    moore = MooreHMM(initial_distribution={"q": 1.0}, observation_alphabet=frozenset({"0"}))
    moore.graph.add_state("q")
    moore.set_emission_distribution("q", {"0": 1.0})
    moore.add_transition("q", "q", 1.0)
    restored_moore = _round_trip(moore)
    assert _metadata_keys(moore) == {"initial_distribution"}
    assert restored_moore.observation_alphabet == moore.observation_alphabet

    pfa = ProbabilisticFiniteAutomaton(initial_distribution={"q": 1.0}, output_alphabet=frozenset({"0"}))
    pfa.graph.add_state("q")
    pfa.add_transition("q", "q", "0", 1.0)
    restored_pfa = _round_trip(pfa)
    assert _metadata_keys(pfa) == {"initial_distribution"}
    assert restored_pfa.output_alphabet == pfa.output_alphabet


def test_epsilon_bidirectional_and_mixed_state_round_trip():
    eps = bernoulli(0.5)
    _round_trip(eps)

    bidir = BidirectionalEpsilonMachine.from_pair(eps, eps)
    _round_trip(bidir)

    msp = golden_mean(0.5).mixed_state_presentation()
    restored = _round_trip(msp)
    assert restored.initial_mixed_state == msp.initial_mixed_state
    assert restored.pure_states == msp.pure_states


def test_quasi_models_round_trip():
    nm = NMachine(initial_quasidistribution={"q": 1.0}, observation_alphabet=frozenset({"0"}))
    nm.graph.add_state("q")
    nm.graph.add_transition("q", "q", **{ATTR_QUASIPROB: 1.0, ATTR_EMISSION: "0"})
    restored_nm = _round_trip(nm)
    assert _metadata_keys(nm) == {"initial_quasidistribution"}
    assert restored_nm.observation_alphabet == nm.observation_alphabet

    qr = QuasiRealization(
        pi=np.array([1.0, 0.0]),
        tau=np.ones(2),
        symbol_maps={"0": np.array([[0.5, 0.5], [0.2, 0.8]])},
    )
    restored = _round_trip(qr)
    assert restored.word_probability(("0",)) == pytest.approx(qr.word_probability(("0",)))


def test_stack_hmm_and_sofic_dyck_edge_refs_round_trip():
    model = HiddenMarkovStackModel(
        initial_distribution={"q": 1.0},
        call_alphabet=frozenset({"a"}),
        return_alphabet=frozenset({"A"}),
        allow_empty_stack_returns=False,
    )
    model.graph.add_state("q")
    call = model.add_call_transition("q", "q", "a", 0.6)
    ret = model.add_return_transition("q", "q", "A", 0.4)
    model.add_matched_pair(call, ret)
    restored = _round_trip(model)
    assert _metadata_keys(model) == {"initial_distribution", "matched_edges", "allow_empty_stack_returns"}
    assert restored.call_alphabet == model.call_alphabet
    assert restored.return_alphabet == model.return_alphabet
    assert restored.matched_edges == model.matched_edges
    assert restored.word_probability(("a", "A")) > 0.0

    shift = SoficDyckShift(call_alphabet=frozenset({"a"}), return_alphabet=frozenset({"A"}))
    shift.graph.add_state("q")
    call = shift.add_call_transition("q", "q", "a")
    ret = shift.add_return_transition("q", "q", "A")
    shift.add_matched_pair(call, ret)
    restored_shift = _round_trip(shift)
    assert restored_shift.matched_edges == shift.matched_edges


def test_shift_models_round_trip():
    sofic = SoficShift(symbol_alphabet=frozenset({"0"}))
    sofic.graph.add_state("q")
    sofic.add_transition("q", "q", "0")
    _round_trip(sofic)

    tmc = TopologicalMarkovChain(symbol_alphabet=frozenset({"0"}))
    tmc.graph.add_state(0)
    tmc.graph.add_transition(0, 0, **{ATTR_SYMBOL: "0"})
    _round_trip(tmc)

    sft = ShiftOfFiniteType.from_forbidden_words({("1", "1")}, frozenset({"0", "1"}))
    restored_sft = _round_trip(sft)
    assert restored_sft.forbidden_words() == sft.forbidden_words()

    markov_dyck = MarkovDyckShift.from_adjacency(np.array([[1]], dtype=int), labels=("x",))
    _round_trip(markov_dyck)


def test_read_write_yaml_file(tmp_path):
    eps = bernoulli(0.5)
    path = tmp_path / "model.yaml"

    eps.write_yaml(path)
    restored = model_from_yaml(path.read_text())

    assert type(restored) is type(eps)
    assert model_to_dict(restored) == model_to_dict(eps)
