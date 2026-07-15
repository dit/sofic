"""YAML serialization for sofic state-machine models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import yaml

from sofic.base import StateMachine
from sofic.graph import (
    ATTR_EMISSION,
    ATTR_EMISSION_DIST,
    ATTR_KIND,
    ATTR_SYMBOL,
    EPSILON,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
    TransitionGraph,
)

SCHEMA = "sofic.model"
VERSION = 1
_TYPE_KEY = "__sofic_type__"


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    cls: type[StateMachine]
    fields: tuple[str, ...]
    builder: str = "default"


def model_to_yaml(model: StateMachine) -> str:
    """Return a YAML representation of ``model``."""
    return yaml.safe_dump(model_to_dict(model), sort_keys=False)


def model_from_yaml(text: str, *, validate: bool = True) -> StateMachine:
    """Reconstruct a sofic model from YAML text."""
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise TypeError("sofic model YAML must load to a mapping")
    return model_from_dict(loaded, validate=validate)


def from_yaml(text: str, *, validate: bool = True) -> StateMachine:
    """Alias for :func:`model_from_yaml`."""
    return model_from_yaml(text, validate=validate)


def read_yaml(path: str | Path, *, validate: bool = True) -> StateMachine:
    """Read a sofic model from a YAML file."""
    return model_from_yaml(Path(path).read_text(encoding="utf-8"), validate=validate)


def model_to_dict(model: StateMachine) -> dict[str, Any]:
    """Return a safe-YAML-compatible mapping for ``model``."""
    if not isinstance(model, StateMachine):
        raise TypeError(f"expected a StateMachine, got {type(model).__name__}")
    registry = _registry_by_type()
    spec = registry.get(type(model))
    if spec is None:
        raise TypeError(f"YAML serialization is not registered for {type(model).__qualname__}")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "class": _class_path(type(model)),
        "graph": _graph_to_data(model.graph),
        "metadata": _encode(_metadata_for(model, spec)),
    }


def model_from_dict(data: Mapping[str, Any], *, validate: bool = True) -> StateMachine:
    """Reconstruct a sofic model from a decoded YAML mapping."""
    if data.get("schema") != SCHEMA:
        raise ValueError(f"unsupported sofic YAML schema {data.get('schema')!r}")
    if data.get("version") != VERSION:
        raise ValueError(f"unsupported sofic YAML version {data.get('version')!r}")
    class_path = data.get("class")
    if not isinstance(class_path, str):
        raise TypeError("sofic model YAML requires a string class path")
    spec = _registry_by_path().get(class_path)
    if spec is None:
        raise ValueError(f"unregistered sofic model class {class_path!r}")

    graph = _graph_from_data(data.get("graph", {}), validate=validate)
    metadata = _decode(data.get("metadata", {}), validate=validate)
    if not isinstance(metadata, dict):
        raise TypeError("sofic model metadata must decode to a mapping")

    model = _build_model(spec, graph, metadata)
    if validate:
        model.validate()
    return model


def _metadata_for(model: StateMachine, spec: _ModelSpec) -> dict[str, Any]:
    return {field: getattr(model, field) for field in spec.fields}


def _build_model(spec: _ModelSpec, graph: TransitionGraph, metadata: dict[str, Any]) -> StateMachine:
    if spec.builder == "composite_vpa":
        return spec.cls(operation=metadata["operation"], operands=metadata["operands"])
    if spec.builder == "hidden_hmm":
        return spec.cls(graph=graph, observation_alphabet=_observation_alphabet(graph), **metadata)
    if spec.builder == "pfa":
        return spec.cls(graph=graph, output_alphabet=_edge_emission_alphabet(graph), **metadata)
    if spec.builder == "stack_hmm":
        return spec.cls(graph=graph, **_stack_alphabets(graph), **metadata)
    if spec.builder == "sft":
        has_spec = bool(metadata.pop("_has_forbidden_word_spec"))
        forbidden = metadata.pop("_forbidden_words")
        return spec.cls(
            forbidden_words=forbidden if has_spec else None,
            graph=graph,
            **metadata,
        )
    return spec.cls(graph=graph, **metadata)


def _edge_emission_alphabet(graph: TransitionGraph) -> frozenset[Any]:
    return frozenset(
        transition.data[ATTR_EMISSION] for transition in graph.transitions() if ATTR_EMISSION in transition.data
    )


def _state_emission_alphabet(graph: TransitionGraph) -> frozenset[Any]:
    symbols: set[Any] = set()
    for state in graph.states():
        distribution = graph.state_attrs(state).get(ATTR_EMISSION_DIST)
        if isinstance(distribution, Mapping):
            symbols.update(distribution)
    return frozenset(symbols)


def _observation_alphabet(graph: TransitionGraph) -> frozenset[Any]:
    return _edge_emission_alphabet(graph) | _state_emission_alphabet(graph)


def _stack_alphabets(graph: TransitionGraph) -> dict[str, frozenset[Any]]:
    calls: set[Any] = set()
    returns: set[Any] = set()
    internals: set[Any] = set()
    for transition in graph.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is None:
            continue
        kind = transition.data.get(ATTR_KIND)
        if kind == KIND_CALL:
            calls.add(symbol)
        elif kind == KIND_RETURN:
            returns.add(symbol)
        elif kind == KIND_INTERNAL:
            internals.add(symbol)
    symbol_alphabet = frozenset(calls | returns | internals)
    return {
        "call_alphabet": frozenset(calls),
        "return_alphabet": frozenset(returns),
        "internal_alphabet": frozenset(internals),
        "symbol_alphabet": symbol_alphabet,
    }


def _graph_to_data(graph: TransitionGraph) -> dict[str, Any]:
    nodes = [{"id": _encode(state), "attrs": _encode(dict(attrs))} for state, attrs in graph.nx.nodes(data=True)]
    edges = [
        {
            "source": _encode(source),
            "target": _encode(target),
            "key": _encode(key),
            "attrs": _encode(dict(attrs)),
        }
        for source, target, key, attrs in graph.nx.edges(keys=True, data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def _graph_from_data(data: Any, *, validate: bool = True) -> TransitionGraph:
    if not isinstance(data, Mapping):
        raise TypeError("sofic model graph must be a mapping")
    graph = nx.MultiDiGraph()
    for node in data.get("nodes", []):
        if not isinstance(node, Mapping):
            raise TypeError("graph node records must be mappings")
        state = _decode(node["id"], validate=validate)
        attrs = _decode(node.get("attrs", {_TYPE_KEY: "dict", "items": []}), validate=validate)
        if not isinstance(attrs, dict):
            raise TypeError("graph node attrs must decode to a dict")
        graph.add_node(state, **attrs)
    for edge in data.get("edges", []):
        if not isinstance(edge, Mapping):
            raise TypeError("graph edge records must be mappings")
        source = _decode(edge["source"], validate=validate)
        target = _decode(edge["target"], validate=validate)
        key = _decode(edge["key"], validate=validate)
        attrs = _decode(edge.get("attrs", {_TYPE_KEY: "dict", "items": []}), validate=validate)
        if not isinstance(attrs, dict):
            raise TypeError("graph edge attrs must decode to a dict")
        graph.add_edge(source, target, key=key, **attrs)
    return TransitionGraph(graph)


def _encode(value: Any) -> Any:
    from sofic.generators.mixed_state import MixedState

    if value is EPSILON:
        return {_TYPE_KEY: "epsilon"}
    if isinstance(value, MixedState):
        return {_TYPE_KEY: "mixed_state", "belief": [_encode(item) for item in value.belief]}
    if isinstance(value, StateMachine):
        return {_TYPE_KEY: "model", "value": model_to_dict(value)}
    if isinstance(value, np.ndarray):
        return {
            _TYPE_KEY: "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _encode(value.tolist()),
        }
    if isinstance(value, np.generic):
        return _encode(value.item())
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, tuple):
        return {_TYPE_KEY: "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, frozenset):
        return {_TYPE_KEY: "frozenset", "items": [_encode(item) for item in _stable_iterable(value)]}
    if isinstance(value, set):
        return {_TYPE_KEY: "set", "items": [_encode(item) for item in _stable_iterable(value)]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {
            _TYPE_KEY: "dict",
            "items": [{"key": _encode(key), "value": _encode(item_value)} for key, item_value in value.items()],
        }
    raise TypeError(f"cannot YAML-serialize value of type {type(value).__qualname__}: {value!r}")


def _decode(value: Any, *, validate: bool = True) -> Any:
    from sofic.generators.mixed_state import MixedState

    if isinstance(value, list):
        return [_decode(item, validate=validate) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get(_TYPE_KEY)
    if tag is None:
        return {key: _decode(item_value, validate=validate) for key, item_value in value.items()}
    if tag == "epsilon":
        return EPSILON
    if tag == "mixed_state":
        return MixedState(tuple(float(_decode(item, validate=validate)) for item in value["belief"]))
    if tag == "model":
        return model_from_dict(value["value"], validate=validate)
    if tag == "ndarray":
        array = np.asarray(_decode(value["data"], validate=validate), dtype=value["dtype"])
        return array.reshape(tuple(value["shape"]))
    if tag == "tuple":
        return tuple(_decode(item, validate=validate) for item in value["items"])
    if tag == "frozenset":
        return frozenset(_decode(item, validate=validate) for item in value["items"])
    if tag == "set":
        return {_decode(item, validate=validate) for item in value["items"]}
    if tag == "dict":
        return {
            _decode(item["key"], validate=validate): _decode(item["value"], validate=validate)
            for item in value["items"]
        }
    raise ValueError(f"unknown sofic YAML value tag {tag!r}")


def _stable_iterable(values: set[Any] | frozenset[Any]) -> list[Any]:
    return sorted(values, key=repr)


def _class_path(cls: type[Any]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


@cache
def _registry_by_path() -> dict[str, _ModelSpec]:
    return {_class_path(spec.cls): spec for spec in _specs()}


@cache
def _registry_by_type() -> dict[type[StateMachine], _ModelSpec]:
    return {spec.cls: spec for spec in _specs()}


def _spec(cls: type[StateMachine], fields: tuple[str, ...], builder: str = "default") -> _ModelSpec:
    return _ModelSpec(cls=cls, fields=fields, builder=builder)


@cache
def _specs() -> tuple[_ModelSpec, ...]:
    from sofic.automata.atomaton import Atomaton, AtomicAutomaton, MaximizedPrimeAtomaton
    from sofic.automata.buchi import BuchiAutomaton
    from sofic.automata.dfa import DFA
    from sofic.automata.nfa import NFA
    from sofic.automata.nwa import NestedWordAutomaton
    from sofic.automata.rfsa import CanonicalRFSA, ResidualFiniteStateAutomaton
    from sofic.automata.transducers import MealyMachine, MooreMachine
    from sofic.automata.unifilar import UnifilarAutomaton
    from sofic.automata.vpa import (
        CallDrivenAutomaton,
        CanonicalVisiblyPushdownAutomaton,
        CompositeVisiblyPushdownAutomaton,
        DeterministicVisiblyPushdownAutomaton,
        MultipleEntryVisiblyPushdownAutomaton,
        SingleEntryVisiblyPushdownAutomaton,
        VisiblyPushdownAutomaton,
    )
    from sofic.generators.base import HiddenMarkovModel, QuasiStochasticModel, StochasticModel
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.epsilon_machine import EpsilonMachine
    from sofic.generators.markov import MarkovChain
    from sofic.generators.mealy import MealyHMM
    from sofic.generators.mixed_state import MixedStatePresentation
    from sofic.generators.moore import MooreHMM
    from sofic.generators.nmachine import NMachine
    from sofic.generators.pfa import ProbabilisticFiniteAutomaton
    from sofic.generators.quasi_realization import QuasiRealization
    from sofic.generators.stack_hmm import HiddenMarkovStackModel
    from sofic.shifts.base import SymbolicModel
    from sofic.shifts.covers import LeftFischerCover, LeftKriegerCover, RightFischerCover, RightKriegerCover
    from sofic.shifts.markov_dyck import MarkovDyckShift
    from sofic.shifts.sft import ShiftOfFiniteType
    from sofic.shifts.sofic import SoficShift
    from sofic.shifts.sofic_dyck import SoficDyckShift
    from sofic.shifts.tmc import TopologicalMarkovChain

    labeled = ("input_alphabet", "initial_states", "accepting_states")
    transducer = ("input_alphabet", "output_alphabet", "initial_states")
    symbolic = ("symbol_alphabet",)
    stochastic = ("initial_distribution",)
    hidden = ("initial_distribution",)
    quasi = ("initial_quasidistribution",)
    vpa = (
        "input_alphabet",
        "call_alphabet",
        "return_alphabet",
        "internal_alphabet",
        "stack_alphabet",
        "bottom_stack_symbol",
        "initial_state",
        "accepting_states",
    )
    cda = (
        *vpa,
        "modules",
        "base_module",
        "call_partition",
        "call_entries",
    )
    nwa = (
        "input_alphabet",
        "call_alphabet",
        "return_alphabet",
        "internal_alphabet",
        "hier_alphabet",
        "bottom_hier_state",
        "initial_state",
        "accepting_states",
    )
    dyck = (
        "symbol_alphabet",
        "call_alphabet",
        "return_alphabet",
        "internal_alphabet",
        "matched_edges",
    )

    return (
        _spec(NFA, labeled),
        _spec(DFA, labeled),
        _spec(BuchiAutomaton, labeled),
        _spec(UnifilarAutomaton, labeled),
        _spec(AtomicAutomaton, labeled),
        _spec(Atomaton, labeled),
        _spec(MaximizedPrimeAtomaton, labeled),
        _spec(ResidualFiniteStateAutomaton, labeled),
        _spec(CanonicalRFSA, labeled),
        _spec(MealyMachine, transducer),
        _spec(MooreMachine, transducer),
        _spec(NestedWordAutomaton, nwa),
        _spec(VisiblyPushdownAutomaton, vpa),
        _spec(DeterministicVisiblyPushdownAutomaton, vpa),
        _spec(CallDrivenAutomaton, cda),
        _spec(MultipleEntryVisiblyPushdownAutomaton, (*cda, "entry_states")),
        _spec(SingleEntryVisiblyPushdownAutomaton, (*cda, "entry_states")),
        _spec(CanonicalVisiblyPushdownAutomaton, (*vpa, "summary_representatives")),
        _spec(CompositeVisiblyPushdownAutomaton, ("operation", "operands"), builder="composite_vpa"),
        _spec(StochasticModel, stochastic),
        _spec(HiddenMarkovModel, hidden, builder="hidden_hmm"),
        _spec(MarkovChain, stochastic),
        _spec(MealyHMM, hidden, builder="hidden_hmm"),
        _spec(MooreHMM, hidden, builder="hidden_hmm"),
        _spec(EpsilonMachine, hidden, builder="hidden_hmm"),
        _spec(BidirectionalEpsilonMachine, (*hidden, "forward_machine", "reverse_machine"), builder="hidden_hmm"),
        _spec(
            MixedStatePresentation,
            (
                *hidden,
                "basis_states",
                "initial_mixed_state",
                "pure_states",
                "recurrent_states",
                "transient_states",
            ),
            builder="hidden_hmm",
        ),
        _spec(ProbabilisticFiniteAutomaton, ("initial_distribution",), builder="pfa"),
        _spec(
            HiddenMarkovStackModel,
            (
                "initial_distribution",
                "matched_edges",
                "allow_empty_stack_returns",
            ),
            builder="stack_hmm",
        ),
        _spec(QuasiStochasticModel, quasi),
        _spec(NMachine, quasi, builder="hidden_hmm"),
        _spec(QuasiRealization, (*quasi, "pi", "tau", "symbol_maps")),
        _spec(SymbolicModel, symbolic),
        _spec(SoficShift, symbolic),
        _spec(TopologicalMarkovChain, symbolic),
        _spec(ShiftOfFiniteType, (*symbolic, "_forbidden_words", "_has_forbidden_word_spec"), builder="sft"),
        _spec(SoficDyckShift, dyck),
        _spec(MarkovDyckShift, dyck),
        _spec(LeftFischerCover, symbolic),
        _spec(RightFischerCover, symbolic),
        _spec(LeftKriegerCover, symbolic),
        _spec(RightKriegerCover, symbolic),
    )
