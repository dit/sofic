"""Composition and probability-aware helpers for finite-state transducers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from itertools import product
from typing import Any

import numpy as np

from pensive.automata.transducers import ERROR_STATE, ERROR_SYMBOL, MealyMachine
from pensive.generators.base import HiddenMarkovModel
from pensive.generators.mealy import MealyHMM
from pensive.graph import ATTR_EMISSION, ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL, EPSILON


def cartesian_product_gg(
    generators: Sequence[HiddenMarkovModel],
    *,
    create_using: type[MealyHMM] | None = None,
) -> MealyHMM:
    """Return the Cartesian product generator with tuple-valued emissions."""
    if not generators:
        raise ValueError("at least one generator is required")
    mealy_generators = [generator.to_mealy() for generator in generators]
    cls = create_using or MealyHMM

    states_by_model = [tuple(generator.states()) for generator in mealy_generators]
    initial: dict[tuple[Hashable, ...], float] = {}
    for state_tuple in product(*states_by_model):
        mass = 1.0
        for generator, state in zip(mealy_generators, state_tuple, strict=True):
            mass *= float(generator.initial_distribution.get(state, 0.0))
        if mass:
            initial[state_tuple] = mass

    alphabets = [tuple(generator.observation_alphabet) for generator in mealy_generators]
    observation_alphabet = frozenset(product(*alphabets)) if all(alphabets) else frozenset()
    result = cls(initial_distribution=initial, observation_alphabet=observation_alphabet)
    for state_tuple in product(*states_by_model):
        result.graph.add_state(state_tuple)

    edge_map: dict[tuple[tuple[Hashable, ...], tuple[Hashable, ...], tuple[Any, ...]], float] = defaultdict(float)
    for source_tuple in product(*states_by_model):
        outgoing_groups = [list(generator.graph.out_transitions(state)) for generator, state in zip(mealy_generators, source_tuple, strict=True)]
        for edge_tuple in product(*outgoing_groups):
            target = tuple(edge.target for edge in edge_tuple)
            emission = tuple(edge.data.get(ATTR_EMISSION) for edge in edge_tuple)
            prob = float(np.prod([_prob(edge.data) for edge in edge_tuple]))
            edge_map[(source_tuple, target, emission)] += prob

    for (source, target, emission), prob in edge_map.items():
        if prob:
            result.graph.add_transition(source, target, **{ATTR_EMISSION: emission, ATTR_PROB: prob})
    return result


def cartesian_product_tt(
    transducers: Sequence[MealyMachine],
    *,
    create_using: type[MealyMachine] | None = None,
    normalize: bool = True,
) -> MealyMachine:
    """Return the Cartesian product transducer with tuple-valued input/output symbols."""
    if not transducers:
        raise ValueError("at least one transducer is required")
    cls = create_using or MealyMachine
    states_by_model = [tuple(transducer.states()) for transducer in transducers]
    initial_states = frozenset(product(*[tuple(t.initial_states) for t in transducers]))
    input_alphabet = frozenset(product(*[tuple(t.alphabets()[0]) for t in transducers]))
    output_alphabet = frozenset(product(*[tuple(t.alphabets()[1]) for t in transducers]))

    edges: list[tuple[Hashable, Hashable, Any, Any, float]] = []
    for source_tuple in product(*states_by_model):
        outgoing_groups = [list(transducer.graph.out_transitions(state)) for transducer, state in zip(transducers, source_tuple, strict=True)]
        for edge_tuple in product(*outgoing_groups):
            target = tuple(edge.target for edge in edge_tuple)
            input_symbol = tuple(_input(edge.data) for edge in edge_tuple)
            output_symbol = tuple(_output(edge.data) for edge in edge_tuple)
            prob = float(np.prod([_prob(edge.data) for edge in edge_tuple]))
            edges.append((source_tuple, target, input_symbol, output_symbol, prob))

    result = _build_transducer(
        edges,
        cls=cls,
        states=product(*states_by_model),
        initial_states=initial_states,
        input_alphabet=input_alphabet,
        output_alphabet=output_alphabet,
    )
    if normalize:
        _normalize_transducer_rows(result)
    return result


def compose_tt(
    transducers: Sequence[MealyMachine],
    *,
    complete: bool = True,
    create_using: type[MealyMachine] | None = None,
    normalize: bool = True,
) -> MealyMachine:
    """Serially compose transducers.

    ``compose_tt((t0, t1))`` returns the transducer that feeds ``t0``'s output
    into ``t1``. State labels are tuples ordered like the input transducers.
    """
    if not transducers:
        raise ValueError("at least one transducer is required")
    result = transducers[0].copy()
    for transducer in transducers[1:]:
        result = _compose_pair_tt(result, transducer, complete=complete, create_using=create_using, normalize=normalize)
    return result


def compose_tg(
    transducer: MealyMachine,
    generator: HiddenMarkovModel,
    *,
    complete: bool = True,
    joint: bool = True,
    normalize: bool = True,
    create_using: type[MealyHMM] | None = None,
) -> MealyHMM:
    """Compose a transducer with a generator.

    The result is a generator on joint ``(input, output)`` emissions by default.
    Set ``joint=False`` to marginalize to output symbols only.
    """
    gen = generator.to_mealy()
    generator_alphabet = frozenset(gen.observation_alphabet)
    transducer_inputs = transducer.alphabets()[0]
    if generator_alphabet and transducer_inputs and not (generator_alphabet & transducer_inputs):
        raise ValueError("generator outputs do not intersect transducer inputs")

    work = transducer.complete(generator_alphabet, copy=True) if complete else transducer.copy()
    cls = create_using or MealyHMM
    t_initial = _transducer_initial_distribution(work)
    initial: dict[tuple[Hashable, Hashable], float] = {}
    for g_state, g_mass in gen.initial_distribution.items():
        for t_state, t_mass in t_initial.items():
            mass = float(g_mass) * float(t_mass)
            if mass:
                initial[(g_state, t_state)] = mass

    states = [(g_state, t_state) for g_state in gen.states() for t_state in work.states()]
    edge_map: dict[tuple[tuple[Hashable, Hashable], tuple[Hashable, Hashable], Any], float] = defaultdict(float)
    for g_state, t_state in states:
        source = (g_state, t_state)
        for t_edge in work.graph.out_transitions(t_state):
            if _input(t_edge.data) is EPSILON:
                emission = _output(t_edge.data)
                edge_map[(source, (g_state, t_edge.target), emission)] += _prob(t_edge.data)
        for g_edge in gen.graph.out_transitions(g_state):
            input_symbol = g_edge.data.get(ATTR_EMISSION)
            if input_symbol is None:
                continue
            for t_edge in work.graph.out_transitions(t_state):
                if _input(t_edge.data) != input_symbol:
                    continue
                output_symbol = _output(t_edge.data)
                emission = (input_symbol, output_symbol) if joint else output_symbol
                prob = _prob(g_edge.data) * _prob(t_edge.data)
                edge_map[(source, (g_edge.target, t_edge.target), emission)] += prob

    observation_alphabet = frozenset(emission for _source, _target, emission in edge_map)
    result = cls(initial_distribution=initial, observation_alphabet=observation_alphabet)
    for state in states:
        result.graph.add_state(state)
    for (source, target, emission), prob in edge_map.items():
        if prob:
            result.graph.add_transition(source, target, **{ATTR_EMISSION: emission, ATTR_PROB: prob})
    if normalize:
        _normalize_hmm_rows(result)
    return result


def transduce_generator(
    transducer: MealyMachine,
    generator: HiddenMarkovModel,
    *,
    complete: bool = True,
    normalize: bool = True,
    create_using: type[MealyHMM] | None = None,
) -> MealyHMM:
    """Return the output-only generator induced by driving ``transducer`` with ``generator``."""
    return compose_tg(
        transducer,
        generator,
        complete=complete,
        joint=False,
        normalize=normalize,
        create_using=create_using,
    )


def _compose_pair_tt(
    left: MealyMachine,
    right: MealyMachine,
    *,
    complete: bool,
    create_using: type[MealyMachine] | None,
    normalize: bool,
) -> MealyMachine:
    left_work = left.complete(copy=True) if complete else left.copy()
    right_alphabet = right.alphabets()[0] | left_work.alphabets()[1]
    right_work = right.complete(right_alphabet, copy=True) if complete else right.copy()
    cls = create_using or MealyMachine

    left_states = tuple(left_work.states())
    right_states = tuple(right_work.states())
    states = [(left_state, right_state) for left_state in left_states for right_state in right_states]
    initial_states = frozenset(product(left_work.initial_states, right_work.initial_states))
    input_alphabet = left_work.alphabets()[0]
    output_alphabet = right_work.alphabets()[1]

    edges: list[tuple[Hashable, Hashable, Any, Any, float]] = []
    for left_state, right_state in states:
        source = (left_state, right_state)
        for right_edge in right_work.graph.out_transitions(right_state):
            if _input(right_edge.data) is EPSILON:
                edges.append(
                    (
                        source,
                        (left_state, right_edge.target),
                        EPSILON,
                        _output(right_edge.data),
                        _prob(right_edge.data),
                    )
                )
        for left_edge in left_work.graph.out_transitions(left_state):
            left_input = _input(left_edge.data)
            middle = _output(left_edge.data)
            if middle is EPSILON:
                edges.append((source, (left_edge.target, right_state), left_input, EPSILON, _prob(left_edge.data)))
                continue
            for right_edge in right_work.graph.out_transitions(right_state):
                if _input(right_edge.data) != middle:
                    continue
                edges.append(
                    (
                        source,
                        (left_edge.target, right_edge.target),
                        left_input,
                        _output(right_edge.data),
                        _prob(left_edge.data) * _prob(right_edge.data),
                    )
                )

    result = _build_transducer(
        edges,
        cls=cls,
        states=states,
        initial_states=initial_states,
        input_alphabet=input_alphabet,
        output_alphabet=output_alphabet,
    )
    if normalize:
        _normalize_transducer_rows(result)
    return result


def _build_transducer(
    edges: Iterable[tuple[Hashable, Hashable, Any, Any, float]],
    *,
    cls: type[MealyMachine],
    states: Iterable[Hashable],
    initial_states: frozenset[Hashable],
    input_alphabet: frozenset[Any],
    output_alphabet: frozenset[Any],
) -> MealyMachine:
    merged: dict[tuple[Hashable, Hashable, Any, Any], float] = defaultdict(float)
    all_states = list(dict.fromkeys(states))
    for source, target, input_symbol, output_symbol, prob in edges:
        all_states.extend([source, target])
        merged[(source, target, input_symbol, output_symbol)] += float(prob)

    result = cls(
        input_alphabet=frozenset(symbol for symbol in input_alphabet if symbol is not EPSILON),
        output_alphabet=frozenset(symbol for symbol in output_alphabet if symbol is not EPSILON),
        initial_states=initial_states,
    )
    for state in dict.fromkeys(all_states):
        result.graph.add_state(state)
    for (source, target, input_symbol, output_symbol), prob in merged.items():
        if prob:
            result.add_transition(source, target, input_symbol, output_symbol, prob=prob)
            if input_symbol is not EPSILON:
                result.input_alphabet = result.input_alphabet | frozenset({input_symbol})
            if output_symbol is not EPSILON:
                result.output_alphabet = result.output_alphabet | frozenset({output_symbol})
    return result


def _normalize_transducer_rows(transducer: MealyMachine) -> None:
    totals: dict[tuple[Hashable, Any], float] = defaultdict(float)
    for transition in transducer.transitions():
        totals[(transition.source, _input(transition.data))] += _prob(transition.data)
    for source, target, key, data in transducer.graph.nx.edges(keys=True, data=True):
        total = totals[(source, data.get(ATTR_SYMBOL, EPSILON))]
        if total > 0.0:
            transducer.graph.nx[source][target][key][ATTR_PROB] = float(data.get(ATTR_PROB, 1.0)) / total


def _normalize_hmm_rows(hmm: MealyHMM) -> None:
    totals: dict[Hashable, float] = defaultdict(float)
    for transition in hmm.transitions():
        totals[transition.source] += _prob(transition.data)
    for source, target, key, data in hmm.graph.nx.edges(keys=True, data=True):
        total = totals[source]
        if total > 0.0:
            hmm.graph.nx[source][target][key][ATTR_PROB] = float(data.get(ATTR_PROB, 1.0)) / total


def _transducer_initial_distribution(transducer: MealyMachine) -> dict[Hashable, float]:
    states = tuple(transducer.initial_states) or tuple(transducer.states())
    if not states:
        return {}
    mass = 1.0 / len(states)
    return dict.fromkeys(states, mass)


def _input(data: dict[str, Any]) -> Any:
    return data.get(ATTR_SYMBOL, EPSILON)


def _output(data: dict[str, Any]) -> Any:
    return data.get(ATTR_OUTPUT, EPSILON)


def _prob(data: dict[str, Any]) -> float:
    return float(data.get(ATTR_PROB, 1.0))


__all__ = [
    "ERROR_STATE",
    "ERROR_SYMBOL",
    "cartesian_product_gg",
    "cartesian_product_tt",
    "compose_tg",
    "compose_tt",
    "transduce_generator",
]
