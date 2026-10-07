"""Property-based round-trip tests for YAML serialization and ``copy()``."""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Hashable, Mapping
from typing import Any

import networkx as nx
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import sofic.examples as examples
from sofic.base import StateMachine
from sofic.exceptions import SoficValidationError
from sofic.graph import TransitionGraph
from sofic.serialization import _registry_by_type, model_from_dict, model_from_yaml, model_to_dict, model_to_yaml
from sofic.shifts import (
    LeftFischerCover,
    LeftKriegerCover,
    RightFischerCover,
    RightKriegerCover,
    TopologicalMarkovChain,
)
from sofic.testing import (
    buchi_automata,
    dfas,
    epsilon_machines,
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

STATE_FIELDS = frozenset({"initial_states", "accepting_states", "initial_distribution", "initial_state"})


def _tagged(value: Any) -> Any:
    """Type-tagged structural key so ``1``, ``1.0``, ``True`` and ``"1"`` stay distinct."""
    if isinstance(value, tuple):
        return ("tuple", tuple(_tagged(item) for item in value))
    if isinstance(value, frozenset | set):
        return (type(value).__name__, frozenset(_tagged(item) for item in value))
    if isinstance(value, Mapping):
        return ("dict", frozenset((_tagged(k), _tagged(v)) for k, v in value.items()))
    if isinstance(value, list):
        return ("list", tuple(_tagged(item) for item in value))
    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), value.shape, _tagged(value.tolist()))
    if isinstance(value, StateMachine):
        return ("model", repr(model_to_dict(value)))
    return (type(value).__name__, value if isinstance(value, Hashable) else repr(value))


def assert_same_model(restored: StateMachine, original: StateMachine) -> None:
    assert type(restored) is type(original)
    nodes = lambda m: {_tagged(s): _tagged(dict(a)) for s, a in m.graph.nx.nodes(data=True)}  # noqa: E731
    edges = lambda m: Counter(  # noqa: E731
        (_tagged(s), _tagged(t), k, _tagged(dict(d))) for s, t, k, d in m.graph.nx.edges(keys=True, data=True)
    )
    assert nodes(restored) == nodes(original)
    assert edges(restored) == edges(original)
    for field in _registry_by_type()[type(original)].fields:
        assert _tagged(getattr(restored, field)) == _tagged(getattr(original, field)), field


LARGE_MODEL_STATES = 500


def round_trip(model: StateMachine) -> StateMachine:
    """YAML round trip; very large models skip the (slow, pure-Python) YAML text layer."""
    if len(model.graph.nx) > LARGE_MODEL_STATES:
        restored = model_from_dict(model_to_dict(model))
    else:
        restored = model_from_yaml(model_to_yaml(model))
    assert_same_model(restored, model)
    assert model_to_dict(restored) == model_to_dict(model)
    return restored


def relabel_states(model: StateMachine, mapping: dict[Hashable, Hashable]) -> StateMachine:
    """Copy of ``model`` with every state renamed, including state-valued metadata."""
    result = model.copy()
    result.graph = TransitionGraph(nx.relabel_nodes(model.graph.nx, mapping, copy=True))
    for field in STATE_FIELDS & set(vars(result)):
        value = getattr(result, field)
        if isinstance(value, frozenset):
            setattr(result, field, frozenset(mapping[s] for s in value))
        elif isinstance(value, dict):
            setattr(result, field, {mapping[s]: p for s, p in value.items()})
        elif value is not None:
            setattr(result, field, mapping[value])
    return result


# --------------------------------------------------------------------------- strategy-generated models


ADVERSARIAL_ALPHABETS = [("0", "1"), (1, "1"), ("yes", "null"), ("~", "1.0"), ("a b", 'q"')]

MODEL_STRATEGIES = {
    "dfa": st.sampled_from(ADVERSARIAL_ALPHABETS).flatmap(lambda a: dfas(alphabet=a)),
    "nfa": st.sampled_from(ADVERSARIAL_ALPHABETS).flatmap(lambda a: nfas(alphabet=a)),
    "buchi": buchi_automata(allow_epsilon=True),
    "wheeler": wheeler_nfas(),
    "markov": markov_chains(),
    "mealy_hmm": st.sampled_from(ADVERSARIAL_ALPHABETS).flatmap(lambda a: mealy_hmms(alphabet=a)),
    "epsilon_machine": epsilon_machines(),
    "sofic": st.sampled_from(ADVERSARIAL_ALPHABETS).flatmap(lambda a: sofic_shifts(alphabet=a)),
    "sft": sfts(),
    "vpa": vpas(),
    "nwa": nwas(),
    "transducer": mealy_transducers(),
}

any_model = st.one_of(*MODEL_STRATEGIES.values())

state_labels = st.one_of(
    st.integers(-(2**70), 2**70),
    st.text(max_size=6),
    st.sampled_from(["yes", "no", "null", "~", "1", "1.0", "0x1F", "1_000", "true", "", " ", "- a", "&x", "*y"]),
    st.tuples(st.integers(0, 3), st.text(max_size=2)),
    st.frozensets(st.integers(0, 3), max_size=3),
)


@pytest.mark.parametrize("name", sorted(MODEL_STRATEGIES))
@given(data=st.data())
def test_strategy_models_round_trip(name, data):
    round_trip(data.draw(MODEL_STRATEGIES[name]))


@given(any_model, st.data())
def test_round_trip_survives_adversarial_state_labels(model, data):
    states = list(model.states())
    labels = data.draw(st.lists(state_labels, min_size=len(states), max_size=len(states), unique_by=_tagged))
    assume(len(set(labels)) == len(labels))
    relabeled = relabel_states(model, dict(zip(states, labels, strict=True)))
    round_trip(relabeled)


@given(sofic_shifts(max_states=3))
def test_cover_models_round_trip(shift):
    round_trip(RightKriegerCover.from_presentation(shift))
    round_trip(LeftKriegerCover.from_presentation(shift))
    try:
        round_trip(RightFischerCover.from_presentation(shift))
        round_trip(LeftFischerCover.from_presentation(shift))
    except SoficValidationError:
        pass


@given(st.integers(1, 3).flatmap(lambda n: st.lists(st.integers(0, 2), min_size=n * n, max_size=n * n)))
def test_topological_markov_chain_round_trips_with_multiplicity(entries):
    n = int(len(entries) ** 0.5)
    round_trip(TopologicalMarkovChain.from_adjacency(np.array(entries).reshape(n, n)))


# --------------------------------------------------------------------------- copy()


def _mutate(model: StateMachine) -> None:
    model.graph.add_state("__fresh__", note=["mutable"])
    for _source, _target, _key, data in model.graph.nx.edges(keys=True, data=True):
        data["mutated"] = True
        break
    for name, value in vars(model).items():
        if isinstance(value, dict):
            value["__fresh__"] = 0.0
        elif isinstance(value, set | list):
            getattr(model, name).clear()


@given(any_model)
def test_copy_equals_original_and_is_independent(model):
    snapshot = model_to_yaml(model)
    clone = model.copy()
    assert_same_model(clone, model)
    _mutate(clone)
    assert model_to_yaml(model) == snapshot
    assert "__fresh__" not in set(model.states())


# --------------------------------------------------------------------------- example factories


def _required(factory: Any) -> list[str]:
    return [
        p.name
        for p in inspect.signature(factory).parameters.values()
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


PARAMETRIZED_FACTORIES = {
    "afc": lambda: examples.afc(3),
    "afc2": lambda: examples.afc2(3),
    "bmc_em": lambda: examples.bmc_em(0.3, 0.6),
    "bmc_gen": lambda: examples.bmc_gen(0.3, 0.6),
    "bmc_lohr": lambda: examples.bmc_lohr(0.3),
    "cyclic_branching": lambda: examples.cyclic_branching(3, 2),
    "iid": lambda: examples.iid(3),
    "lollipop": lambda: examples.lollipop(2, 3),
    "markov_skeleton": lambda: examples.markov_skeleton(2, 2),
    "multiple_n": lambda: examples.multiple_n(3),
    "period": lambda: examples.period(3),
    "periodic": lambda: examples.periodic("0011"),
    "rk_gm": lambda: examples.rk_gm(2, 2),
    "rn_gm": lambda: examples.rn_gm(2, 2),
    "stretched_gm": lambda: examples.stretched_gm(2),
    "uniform_mealymc": lambda: examples.uniform_mealymc(1, 2),
}


def example_factories() -> dict[str, Any]:
    """Every model factory in :mod:`sofic.examples`: defaults where possible, curated arguments otherwise."""
    found: dict[str, Any] = {}
    candidates = [(name, getattr(examples, name)) for name in sorted(set(examples.__all__))]
    for list_name in ("process_list", "transducer_list", "epsilon_transducer_list"):
        candidates += [(factory.__name__, factory) for factory in getattr(examples, list_name)]
    for name, factory in candidates:
        if not inspect.isfunction(factory) or inspect.signature(factory).return_annotation in ("float", "dict"):
            continue
        if not _required(factory):
            found[name] = factory
    found.update(PARAMETRIZED_FACTORIES)
    return found


EXAMPLE_FACTORIES = example_factories()


@pytest.mark.parametrize("name", sorted(EXAMPLE_FACTORIES))
def test_example_factory_round_trips_and_copies(name):
    model = EXAMPLE_FACTORIES[name]()
    if not isinstance(model, StateMachine):
        pytest.skip(f"{name} returns {type(model).__name__}, not a model")
    assert type(model) in _registry_by_type(), f"{name}: {type(model).__qualname__} is not registered"
    round_trip(model)
    snapshot = model_to_dict(model)
    clone = model.copy()
    assert_same_model(clone, model)
    _mutate(clone)
    assert model_to_dict(model) == snapshot
