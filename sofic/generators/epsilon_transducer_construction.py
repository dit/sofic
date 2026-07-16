"""Epsilon-transducer construction via causal-state merging.

For a joint-unifilar stochastic transducer, merge channel-equivalent states by
Hopcroft-style partition refinement on ``(input, output, probability,
successor_block)`` transition signatures -- the input-output analog of the
ε-machine construction in :mod:`sofic.generators.epsilon_construction` (Barnett &
Crutchfield, J. Stat. Phys. 161:2 (2015)).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sofic.automata.transducers import MealyMachine
from sofic.exceptions import StochasticValidationError
from sofic.generators.epsilon_transducer import EpsilonTransducer
from sofic.graph import ATTR_EMISSION, ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL, EPSILON, TransitionGraph
from sofic.states import sequential_labels

TransducerSignature = tuple[tuple[Any, Any, Any, int], ...]

_PROB_DIGITS = 12


def build_epsilon_transducer(channel: MealyMachine, *, name: str | None = None) -> EpsilonTransducer:
    """Minimize a joint-unifilar stochastic transducer to its causal states."""
    from sofic.properties import is_unifilar_transducer

    work = channel.copy()
    if not is_unifilar_transducer(work):
        raise StochasticValidationError(
            "channel must be joint-unifilar (each (state, input, output) has one successor) "
            "to build an epsilon-transducer; reconstruct from paired data with "
            "EpsilonTransducer.from_paired_sequences instead"
        )
    _trim_unreachable(work)
    partitions = _refine_partitions(work)
    result = _quotient_transducer(work, partitions)
    if name is not None:
        result.name = name
    return result


def _trim_unreachable(tr: MealyMachine) -> None:
    """Drop states not forward-reachable from the initial states (in place)."""
    if not tr.initial_states:
        return
    reachable = tr.graph.forward_reachable(set(tr.initial_states))
    for state in list(tr.states()):
        if state not in reachable:
            tr.graph.nx.remove_node(state)


def _refine_partitions(tr: MealyMachine) -> list[set[Any]]:
    partitions: list[set[Any]] = [set(tr.states())]
    changed = True
    while changed:
        changed = False
        state_to_block = _state_to_block_index(partitions)
        new_partitions: list[set[Any]] = []
        for block in partitions:
            subblocks = _split_block(tr, block, state_to_block)
            if len(subblocks) > 1:
                changed = True
            new_partitions.extend(subblocks)
        partitions = new_partitions
    return partitions


def _state_to_block_index(partitions: list[set[Any]]) -> dict[Any, int]:
    mapping: dict[Any, int] = {}
    for index, block in enumerate(partitions):
        for state in block:
            mapping[state] = index
    return mapping


def _split_block(tr: MealyMachine, block: set[Any], state_to_block: dict[Any, int]) -> list[set[Any]]:
    signatures: dict[TransducerSignature, set[Any]] = defaultdict(set)
    for state in block:
        signatures[_transition_signature(tr, state, state_to_block)].add(state)
    return list(signatures.values())


def _transition_signature(tr: MealyMachine, state: Any, state_to_block: dict[Any, int]) -> TransducerSignature:
    triples: list[tuple[Any, Any, Any, int]] = []
    for transition in tr.graph.out_transitions(state):
        symbol = transition.data.get(ATTR_SYMBOL, EPSILON)
        output = transition.data.get(ATTR_OUTPUT, EPSILON)
        prob = round(float(transition.data.get(ATTR_PROB, 1.0)), _PROB_DIGITS)
        triples.append((symbol, output, prob, state_to_block[transition.target]))
    return tuple(sorted(triples, key=repr))


def _quotient_transducer(tr: MealyMachine, partitions: list[set[Any]]) -> EpsilonTransducer:
    state_map: dict[Any, int] = _state_to_block_index(partitions)
    labels = sequential_labels(len(partitions))

    graph = TransitionGraph()
    for label in labels:
        graph.add_state(label)

    inputs: set[Any] = set()
    outputs: set[Any] = set()
    for block_index, block in enumerate(partitions):
        representative = next(iter(block))
        source = labels[block_index]
        for transition in tr.graph.out_transitions(representative):
            symbol = transition.data.get(ATTR_SYMBOL, EPSILON)
            output = transition.data.get(ATTR_OUTPUT, EPSILON)
            prob = float(transition.data.get(ATTR_PROB, 1.0))
            target = labels[state_map[transition.target]]
            attrs: dict[str, Any] = {ATTR_SYMBOL: symbol, ATTR_PROB: prob}
            if output is not EPSILON:
                attrs[ATTR_OUTPUT] = output
            graph.add_transition(source, target, **attrs)
            if symbol is not EPSILON and symbol is not None:
                inputs.add(symbol)
            if output is not EPSILON and output is not None:
                outputs.add(output)

    initial_states = frozenset(labels[state_map[state]] for state in tr.initial_states if state in state_map)
    support = initial_states or frozenset(labels)
    mass = 1.0 / len(support)
    initial_distribution = dict.fromkeys(support, mass)

    result = EpsilonTransducer(
        input_alphabet=frozenset(inputs),
        output_alphabet=frozenset(outputs),
        initial_states=initial_states,
        initial_distribution=initial_distribution,
        graph=graph,
    )
    result.validate()
    return result


def from_joint_generator(generator: Any) -> EpsilonTransducer:
    """Build the ε-transducer from a generator emitting ``(input, output)`` pairs.

    The joint process is first reduced to its (joint-unifilar) ε-machine, then
    conditionalized to the channel law ``T(y, s' | s, x) = P(x, y, s' | s) /
    P(x | s)`` before causal-state minimization.
    """
    from sofic.generators.epsilon_machine import EpsilonMachine

    mealy = generator.to_mealy()
    joint = EpsilonMachine.from_hmm(mealy)
    channel = _channel_from_joint(joint)
    return build_epsilon_transducer(channel)


def _channel_from_joint(joint: Any) -> MealyMachine:
    marginals: dict[tuple[Any, Any], float] = defaultdict(float)
    inputs: set[Any] = set()
    outputs: set[Any] = set()
    for transition in joint.transitions():
        emission = transition.data.get(ATTR_EMISSION)
        pair = _as_pair(emission)
        marginals[(transition.source, pair[0])] += float(transition.data.get(ATTR_PROB, 0.0))

    channel = MealyMachine(initial_states=frozenset(joint.initial_distribution))
    for state in joint.states():
        channel.graph.add_state(state)
    for transition in joint.transitions():
        emission = transition.data.get(ATTR_EMISSION)
        x, y = _as_pair(emission)
        prob = float(transition.data.get(ATTR_PROB, 0.0))
        denom = marginals[(transition.source, x)]
        if denom <= 0.0:
            continue
        channel.add_transition(transition.source, transition.target, x, y, prob=prob / denom)
        inputs.add(x)
        outputs.add(y)
    channel.input_alphabet = frozenset(inputs)
    channel.output_alphabet = frozenset(outputs)
    channel.validate()
    return channel


def _as_pair(emission: Any) -> tuple[Any, Any]:
    if not isinstance(emission, tuple) or len(emission) != 2:
        raise TypeError("joint generator must emit length-2 (input, output) tuples")
    return emission
