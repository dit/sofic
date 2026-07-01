"""Bidirectional ε-machine construction (Ellison et al., arXiv:1107.2168, Sec. VII)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np

from pensive.exceptions import StochasticValidationError
from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from pensive.generators.epsilon_construction import (
    _refine_probabilistic_partitions,
    _unifilar_presentation,
)
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.generators.mixed_state import MixedState
from pensive.generators.reversal import time_reverse_stochastic
from pensive.generators.stationary import stationary_distribution_hmm
from pensive.graph import ATTR_EMISSION, ATTR_FUTURE_SYMBOL, ATTR_PROB, TransitionGraph
from pensive.states import next_sequential_label_index, sequential_labels


def build_bidirectional_epsilon_machine(
    forward: EpsilonMachine,
    reverse: EpsilonMachine,
) -> BidirectionalEpsilonMachine:
    """Build M± from forward and reverse ε-machines via Eq. (15)."""
    if forward.observation_alphabet != reverse.observation_alphabet:
        raise StochasticValidationError("forward and reverse observation alphabets must match")

    reverse = _relabel_collision_free(reverse, forward)
    rev_time = time_reverse_stochastic(reverse)
    compatible = _compatible_pairs(forward, reverse)
    graph = _build_eq15_graph(forward, rev_time, sources=compatible)
    graph, initial = _prune_to_stationary_support(graph, forward, reverse)
    if not initial:
        raise StochasticValidationError("bidirectional machine has empty joint support")

    bidir = BidirectionalEpsilonMachine(
        graph=graph,
        initial_distribution=initial,
        observation_alphabet=forward.observation_alphabet,
        forward_machine=forward,
        reverse_machine=reverse,
    )
    bidir._joint_pi = dict(initial)
    bidir.validate()
    _validate_bidirectional_anatomy(bidir)
    return bidir


def _build_eq15_graph(
    forward: EpsilonMachine,
    rev_time: EpsilonMachine,
    *,
    sources: set[tuple[Hashable, Hashable]] | None = None,
) -> TransitionGraph:
    """Build the provisional Eq. (15) graph before pruning transient joint states."""
    reverse_states = sorted(rev_time.states(), key=repr)
    raw: dict[tuple[Hashable, Hashable], list[tuple[tuple[Hashable, Hashable], Any, float]]] = defaultdict(list)

    for alpha in sorted(forward.states(), key=repr):
        for gamma in reverse_states:
            source = (alpha, gamma)
            if sources is not None and source not in sources:
                continue
            for transition in forward.graph.out_transitions(alpha):
                symbol = transition.data.get(ATTR_EMISSION)
                prob_forward = float(transition.data.get(ATTR_PROB, 0.0))
                if symbol is None or prob_forward <= 0.0:
                    continue
                beta = transition.target
                for delta in reverse_states:
                    tex = _reverse_tex_probability(rev_time, gamma, delta, symbol)
                    if tex <= 0.0:
                        continue
                    raw[source].append(((beta, delta), symbol, tex))

    return _graph_from_raw(raw)


def _graph_from_raw(
    raw: dict[tuple[Hashable, Hashable], list[tuple[tuple[Hashable, Hashable], Any, float]]],
) -> TransitionGraph:
    graph = TransitionGraph()
    registry: dict[tuple[Hashable, Hashable], tuple[Hashable, Hashable]] = {}

    def _intern(pair: tuple[Hashable, Hashable]) -> tuple[Hashable, Hashable]:
        existing = registry.get(pair)
        if existing is not None:
            return existing
        registry[pair] = pair
        return pair

    for source, entries in raw.items():
        if not entries:
            continue
        source = _intern(source)
        graph.add_state(source)
        aggregated: dict[tuple[tuple[Hashable, Hashable], Any], float] = defaultdict(float)
        for target, symbol, weight in entries:
            aggregated[(_intern(target), symbol)] += weight
        total = sum(aggregated.values())
        if total <= 0.0:
            continue
        for (target, symbol), weight in aggregated.items():
            prob = _clean_probability(weight / total)
            graph.add_transition(
                source,
                target,
                **{ATTR_PROB: prob, ATTR_EMISSION: symbol},
            )
    return graph


def _clean_probability(probability: float) -> float:
    rounded = round(float(probability), 15)
    if np.isclose(probability, rounded, rtol=0.0, atol=1e-15):
        return rounded
    return float(probability)


def _reverse_tex_probability(
    rev_time: EpsilonMachine,
    gamma: Hashable,
    delta: Hashable,
    symbol: Any,
) -> float:
    """T̃_x(γ, δ) from the time-reversed reverse ε-machine (Eq. 18)."""
    total = 0.0
    for transition in rev_time.graph.out_transitions(gamma):
        if transition.data.get(ATTR_EMISSION) != symbol:
            continue
        if transition.target != delta:
            continue
        total += float(transition.data.get(ATTR_PROB, 0.0))
    return total


def _prune_to_stationary_support(
    graph: TransitionGraph,
    forward: EpsilonMachine,
    reverse: EpsilonMachine,
    *,
    tol: float = 1e-12,
) -> tuple[TransitionGraph, dict[tuple[Hashable, Hashable], float]]:
    """Drop transient joint states and return a margin-matching stationary joint π."""
    states = list(graph.states())
    if not states:
        return graph, {}

    keep = _recurrent_support(graph) or set(states)
    trimmed = _restrict_graph(graph, keep)
    initial = _joint_pi_minimum_support(trimmed, forward, reverse, tol=tol)
    if not initial:
        return trimmed, {}

    trimmed = _restrict_graph(trimmed, set(initial.keys()))
    keep = _recurrent_support(trimmed) or set(initial.keys())
    trimmed = _restrict_graph(trimmed, keep)
    initial = _joint_pi_minimum_support(trimmed, forward, reverse, tol=tol)
    if initial:
        trimmed = _restrict_graph(trimmed, set(initial.keys()))
    return trimmed, initial


def _compatible_pairs(
    forward: EpsilonMachine,
    reverse: EpsilonMachine,
) -> set[tuple[Hashable, Hashable]]:
    """Joint states (α, γ) with positive measure in the bidirectional presentation.

    Eq. (15) is evaluated on every forward/reverse pair that passes the reverse
    future-symbol filter.  When multiple undirected components admit a
    margin-matching joint π, :func:`_joint_pi_minimum_support` breaks ties by
    the information-anatomy identity ``h_μ = b_μ + r_μ``.
    """
    future_symbol = _infer_future_symbols(reverse)
    pairs: set[tuple[Hashable, Hashable]] = set()
    for alpha in forward.states():
        for gamma in reverse.states():
            if gamma in future_symbol:
                required = future_symbol[gamma]
                if required not in forward.observation_alphabet:
                    continue
                if not _forward_emits(forward, alpha, required):
                    continue
            pairs.add((alpha, gamma))
    return pairs


def _forward_emits(forward: EpsilonMachine, state: Hashable, symbol: Any) -> bool:
    return any(
        transition.data.get(ATTR_EMISSION) == symbol and float(transition.data.get(ATTR_PROB, 0.0)) > 0.0
        for transition in forward.graph.out_transitions(state)
    )


def _infer_future_symbols(reverse: EpsilonMachine) -> dict[Hashable, Any]:
    """Map reverse causal states to the future symbol X₀ they carry when annotated."""
    mapping: dict[Hashable, Any] = {}
    alphabet = reverse.observation_alphabet
    for state in reverse.states():
        attrs = reverse.graph.state_attrs(state)
        if ATTR_FUTURE_SYMBOL in attrs:
            mapping[state] = attrs[ATTR_FUTURE_SYMBOL]
        elif state in alphabet:
            mapping[state] = state
    return mapping


def _joint_pi_from_marginals(
    graph: TransitionGraph,
    forward: EpsilonMachine,
    reverse: EpsilonMachine,
    *,
    tol: float = 1e-12,
    max_iter: int = 256,
) -> dict[tuple[Hashable, Hashable], float]:
    """Match forward/reverse stationary marginals on the retained joint support."""
    pairs = [state for state in graph.states() if isinstance(state, tuple) and len(state) == 2]
    if not pairs:
        return {}

    pi_plus = _state_distribution(forward)
    pi_minus = _state_distribution(reverse)
    return _ipf_joint_on_pairs(pairs, pi_plus, pi_minus, tol=tol, max_iter=max_iter)


def _ipf_joint_on_pairs(
    pairs: list[tuple[Hashable, Hashable]],
    pi_plus: dict[Hashable, float],
    pi_minus: dict[Hashable, float],
    *,
    tol: float = 1e-12,
    max_iter: int = 256,
) -> dict[tuple[Hashable, Hashable], float]:
    if not pairs:
        return {}
    joint = {pair: 1.0 / len(pairs) for pair in pairs}

    for _ in range(max_iter):
        for alpha in {pair[0] for pair in pairs}:
            target = pi_plus.get(alpha, 0.0)
            current = sum(joint[pair] for pair in pairs if pair[0] == alpha)
            if current <= tol or target <= tol:
                continue
            scale = target / current
            for pair in pairs:
                if pair[0] == alpha:
                    joint[pair] *= scale
        for gamma in {pair[1] for pair in pairs}:
            target = pi_minus.get(gamma, 0.0)
            current = sum(joint[pair] for pair in pairs if pair[1] == gamma)
            if current <= tol or target <= tol:
                continue
            scale = target / current
            for pair in pairs:
                if pair[1] == gamma:
                    joint[pair] *= scale

    total = sum(joint.values())
    if total <= tol:
        return {}
    joint = {pair: mass / total for pair, mass in joint.items() if mass > tol}
    return joint


def _joint_pi_on_pair_subset(
    pairs: list[tuple[Hashable, Hashable]],
    pi_plus: dict[Hashable, float],
    pi_minus: dict[Hashable, float],
    *,
    tol: float = 1e-12,
) -> dict[tuple[Hashable, Hashable], float] | None:
    """Return a joint matching marginals on ``pairs`` only, or ``None`` if infeasible."""
    if not pairs:
        return None

    for alpha, target in pi_plus.items():
        subset = [pair for pair in pairs if pair[0] == alpha]
        if not subset:
            if target > tol:
                return None
            continue
        if len(subset) == 1:
            if target < -tol:
                return None
            continue
    for gamma, target in pi_minus.items():
        subset = [pair for pair in pairs if pair[1] == gamma]
        if not subset:
            if target > tol:
                return None
            continue

    joint = _ipf_joint_on_pairs(pairs, pi_plus, pi_minus, tol=tol)
    if not joint:
        return None

    for alpha, target in pi_plus.items():
        got = sum(joint.get(pair, 0.0) for pair in pairs if pair[0] == alpha)
        if abs(got - target) > 1e-8:
            return None
    for gamma, target in pi_minus.items():
        got = sum(joint.get(pair, 0.0) for pair in pairs if pair[1] == gamma)
        if abs(got - target) > 1e-8:
            return None
    return joint


def _undirected_components(
    graph: TransitionGraph,
) -> list[set[tuple[Hashable, Hashable]]]:
    import networkx as nx

    undirected = graph.nx.to_undirected()
    return [set(component) for component in nx.connected_components(undirected)]


def _anatomy_gap_for_joint(
    graph: TransitionGraph,
    joint: dict[tuple[Hashable, Hashable], float],
    forward: EpsilonMachine,
    reverse: EpsilonMachine,
) -> float:
    """Return |b_μ + r_μ − h_μ| for a candidate joint support, or inf if unavailable."""
    try:
        provisional = BidirectionalEpsilonMachine(
            graph=_restrict_graph(graph, set(joint.keys())),
            initial_distribution=joint,
            observation_alphabet=forward.observation_alphabet,
            forward_machine=forward,
            reverse_machine=reverse,
        )
        provisional._joint_pi = dict(joint)
        h_mu = provisional.entropy_rate()
        b_mu = provisional.bound_information()
        r_mu = provisional.ephemeral_information()
    except (ImportError, StochasticValidationError, ValueError):
        return float("inf")
    return abs(b_mu + r_mu - h_mu)


def _joint_pi_minimum_support(
    graph: TransitionGraph,
    forward: EpsilonMachine,
    reverse: EpsilonMachine,
    *,
    tol: float = 1e-12,
) -> dict[tuple[Hashable, Hashable], float]:
    """Pick the smallest feasible undirected component with a marginal-matching joint π."""
    pairs = [state for state in graph.states() if isinstance(state, tuple) and len(state) == 2]
    if not pairs:
        return {}

    pi_plus = _state_distribution(forward)
    pi_minus = _state_distribution(reverse)
    components = sorted(
        _undirected_components(graph),
        key=lambda component: (len(component), sorted(component, key=repr)),
    )

    best: dict[tuple[Hashable, Hashable], float] | None = None
    best_support = len(pairs) + 1
    best_anatomy_gap = float("inf")
    for component in components:
        component_pairs = [pair for pair in pairs if pair in component]
        joint = _joint_pi_on_pair_subset(component_pairs, pi_plus, pi_minus, tol=tol)
        if joint is None:
            continue
        support = len(joint)
        anatomy_gap = _anatomy_gap_for_joint(graph, joint, forward, reverse)
        if support < best_support or (support == best_support and anatomy_gap < best_anatomy_gap):
            best = joint
            best_support = support
            best_anatomy_gap = anatomy_gap
            if support == 1 and anatomy_gap <= 1e-9:
                break

    if best is not None:
        return best
    return _joint_pi_from_marginals(graph, forward, reverse, tol=tol)


def _state_distribution(model: EpsilonMachine) -> dict[Hashable, float]:
    idx = model.reindex()
    pi = model.stationary_distribution()
    return {idx.state(i): float(pi[i]) for i in range(len(idx))}


def _canonical_joint_state(
    state: tuple[Hashable, Hashable],
    keep: set[tuple[Hashable, Hashable]],
) -> tuple[Hashable, Hashable] | None:
    if state in keep:
        return state
    for key in keep:
        if key == state:
            return key
    return None


def _restrict_graph(graph: TransitionGraph, keep: set[tuple[Hashable, Hashable]]) -> TransitionGraph:
    trimmed = TransitionGraph()
    for state in keep:
        trimmed.add_state(state)
    for transition in graph.transitions():
        source = _canonical_joint_state(transition.source, keep)
        target = _canonical_joint_state(transition.target, keep)
        if source is None or target is None:
            continue
        trimmed.add_transition(source, target, **transition.data)
    return trimmed


def _relabel_epsilon_machine(
    machine: EpsilonMachine,
    mapping: Mapping[Hashable, Hashable],
) -> EpsilonMachine:
    from pensive.generators.stochastic import normalize_row_weights

    graph = TransitionGraph()
    for state in machine.states():
        attrs = machine.graph.state_attrs(state)
        graph.add_state(mapping[state], **attrs)
    for state in machine.states():
        outgoing = list(machine.graph.out_transitions(state))
        merged: dict[tuple[Hashable, Any], float] = {}
        for transition in outgoing:
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            emission = transition.data.get(ATTR_EMISSION)
            key = (mapping[transition.target], emission)
            merged[key] = merged.get(key, 0.0) + prob
        merged = normalize_row_weights(merged)
        source = mapping[state]
        for (target, emission), prob in merged.items():
            attrs = {ATTR_PROB: prob}
            if emission is not None:
                attrs[ATTR_EMISSION] = emission
            graph.add_transition(source, target, **attrs)
    initial = {mapping[state]: prob for state, prob in machine.initial_distribution.items()}
    eps = EpsilonMachine(
        graph=graph,
        initial_distribution=initial,
        observation_alphabet=machine.observation_alphabet,
    )
    eps.validate()
    return eps


def _relabel_collision_free(
    reverse: EpsilonMachine,
    forward: EpsilonMachine,
) -> EpsilonMachine:
    """Relabel ``reverse`` to sequential capital letters when names collide with ``forward``."""
    if not set(reverse.states()) & set(forward.states()):
        return reverse
    states = sorted(reverse.states(), key=repr)
    start = next_sequential_label_index(forward.states())
    labels = sequential_labels(len(states), start=start)
    mapping = dict(zip(states, labels, strict=True))
    return _relabel_epsilon_machine(reverse, mapping)


def _collision_relabel_mapping(
    machine: EpsilonMachine,
    forward: EpsilonMachine,
) -> dict[Hashable, Hashable]:
    """Return the state relabeling applied by :func:`_relabel_collision_free`."""
    if not set(machine.states()) & set(forward.states()):
        return {state: state for state in machine.states()}
    states = sorted(machine.states(), key=repr)
    start = next_sequential_label_index(forward.states())
    labels = sequential_labels(len(states), start=start)
    return dict(zip(states, labels, strict=True))


def _state_forward_belief(
    state: Any,
    basis: tuple[Hashable, ...],
) -> np.ndarray:
    """Belief vector over ``basis`` for a presentation or mixed state."""
    if isinstance(state, MixedState):
        return np.asarray(state.belief, dtype=float)
    if state in basis:
        vector = np.zeros(len(basis), dtype=float)
        vector[basis.index(state)] = 1.0
        return vector
    raise ValueError(f"cannot extract forward belief for state {state!r}")


def _belief_covering(
    belief: np.ndarray,
    basis: tuple[Hashable, ...],
    *,
    tol: float = 1e-12,
) -> frozenset[Hashable]:
    return frozenset(basis[index] for index, weight in enumerate(belief) if weight > tol)


def _reverse_forward_coverings(
    forward: EpsilonMachine,
    reverse: EpsilonMachine,
    *,
    tol: float = 1e-12,
) -> dict[Hashable, frozenset[Hashable]]:
    """Map each reverse causal state to forward basis states in its MSP covering."""
    from pensive.generators.mixed_state_construction import build_mixed_state_presentation

    rev_hmm = time_reverse_stochastic(forward)
    reverse_pre = EpsilonMachine.from_hmm(rev_hmm)
    msp = build_mixed_state_presentation(rev_hmm)
    basis = tuple(msp.basis_states)
    presentation = _unifilar_presentation(msp)
    partitions = _refine_probabilistic_partitions(presentation)
    stationary = presentation.stationary_distribution()
    idx = presentation.reindex()
    labels = sequential_labels(len(partitions))

    pre_coverings: dict[Hashable, frozenset[Hashable]] = {}
    for block_index, block in enumerate(partitions):
        averaged = np.zeros(len(basis), dtype=float)
        total_mass = 0.0
        for state in block:
            mass = float(stationary[idx.index(state)])
            if mass <= tol:
                continue
            averaged += mass * _state_forward_belief(state, basis)
            total_mass += mass
        if total_mass <= tol:
            continue
        averaged /= total_mass
        pre_coverings[labels[block_index]] = _belief_covering(averaged, basis, tol=tol)

    relabel = _collision_relabel_mapping(reverse_pre, forward)
    inv_relabel = {final: pre for pre, final in relabel.items()}
    coverings: dict[Hashable, frozenset[Hashable]] = {}
    for gamma in reverse.states():
        pre_state = inv_relabel.get(gamma)
        if pre_state is None:
            continue
        cover = pre_coverings.get(pre_state)
        if cover is not None:
            coverings[gamma] = cover
    return coverings


def _validate_bidirectional_anatomy(bidir: BidirectionalEpsilonMachine, *, tol: float = 1e-9) -> None:
    """Ensure positive-support joint states form one weakly connected component."""
    import networkx as nx

    graph = bidir.to_networkx()
    positive = set(bidir._joint_pi or bidir.joint_distribution())
    if positive:
        subgraph = graph.subgraph(positive).copy()
        if subgraph.number_of_nodes() > 0:
            components = list(nx.weakly_connected_components(subgraph))
            if len(components) != 1:
                raise StochasticValidationError(
                    f"bidirectional machine has {len(components)} weak components on positive support"
                )


def infer_reverse_epsilon_machine(forward: EpsilonMachine) -> EpsilonMachine:
    """Infer a reverse ε-machine presentation for bidirectional construction.

    Time-reverses ``forward``, then builds the generator ε-machine via MSP and
    probabilistic state merging (:meth:`EpsilonMachine.from_hmm`).
    """
    rev_hmm = time_reverse_stochastic(forward)
    reverse = EpsilonMachine.from_hmm(rev_hmm)
    return _relabel_collision_free(reverse, forward)


def _recurrent_support(graph: TransitionGraph, *, tol: float = 1e-12) -> set[tuple[Hashable, Hashable]]:
    """Keep joint states in the recurrent class with stochastic outgoing edges."""
    keep = set(graph.states())
    changed = True
    while changed:
        changed = False
        next_keep: set[tuple[Hashable, Hashable]] = set()
        for state in keep:
            outgoing = [transition for transition in graph.out_transitions(state) if transition.target in keep]
            if not outgoing:
                changed = True
                continue
            total = sum(float(t.data.get(ATTR_PROB, 0.0)) for t in outgoing)
            if not np.isclose(total, 1.0, atol=1e-9):
                changed = True
                continue
            next_keep.add(state)
        keep = next_keep
    return keep


def joint_distribution(bidir: BidirectionalEpsilonMachine) -> dict[tuple[Hashable, Hashable], float]:
    """Return π(α, γ) = P(S⁺ = α, S⁻ = γ) under the bidirectional stationary distribution."""
    if bidir._joint_pi is not None:
        return dict(bidir._joint_pi)

    idx = bidir.reindex()
    pi = stationary_distribution_hmm(bidir)
    joint: dict[tuple[Hashable, Hashable], float] = {}
    for index, state in enumerate(idx.states):
        alpha, gamma = state
        mass = float(pi[index])
        if mass > 0.0:
            joint[(alpha, gamma)] = mass
    return joint


def bidirectional_step_distribution(bidir: BidirectionalEpsilonMachine) -> Any:
    """Stationary joint ``Pr(S⁺₀, S⁻₀, X₀, S⁺₁, S⁻₁)`` as a 5-RV dit Distribution.

    Random-variable indices (dit ``X0`` … ``X4``):

    * ``0`` — forward causal state ``S⁺₀``
    * ``1`` — reverse causal state ``S⁻₀``
    * ``2`` — present emission ``X₀``
    * ``3`` — forward causal state ``S⁺₁``
    * ``4`` — reverse causal state ``S⁻₁``
    """
    dit = _require_dit_for_step()

    joint = joint_distribution(bidir)
    outcomes: list[tuple[Any, ...]] = []
    probs: list[float] = []
    for (alpha, gamma), mass in joint.items():
        if mass <= 0.0:
            continue
        for transition in bidir.graph.out_transitions((alpha, gamma)):
            symbol = transition.data.get(ATTR_EMISSION)
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            if symbol is None or prob <= 0.0:
                continue
            beta, delta = transition.target
            weight = mass * prob
            if weight <= 0.0:
                continue
            outcomes.append((alpha, gamma, symbol, beta, delta))
            probs.append(weight)

    if not probs:
        raise StochasticValidationError("bidirectional step distribution is empty")

    total = sum(probs)
    return dit.Distribution(outcomes, [p / total for p in probs])


def _require_dit_for_step():
    try:
        import dit
    except ImportError as exc:
        raise ImportError("dit is required for bidirectional step distributions") from exc
    return dit


def forward_epsilon_machine(bidir: BidirectionalEpsilonMachine) -> EpsilonMachine:
    """Marginalize M± to recover M⁺ (paper Eq. after 2454)."""
    return _marginalize_to_epsilon(bidir, project_forward=True)


def reverse_epsilon_machine(bidir: BidirectionalEpsilonMachine) -> EpsilonMachine:
    """Marginalize M± to recover M⁻ (paper Eq. after 2464)."""
    return _marginalize_to_epsilon(bidir, project_forward=False)


def _marginalize_to_epsilon(
    bidir: BidirectionalEpsilonMachine,
    *,
    project_forward: bool,
) -> EpsilonMachine:
    joint = joint_distribution(bidir)
    side = bidir.forward_machine if project_forward else bidir.reverse_machine
    coord = 0 if project_forward else 1

    pi_marginal: dict[Hashable, float] = {}
    for pair, mass in joint.items():
        state = pair[coord]
        pi_marginal[state] = pi_marginal.get(state, 0.0) + mass

    graph = TransitionGraph()
    for state in side.states():
        if project_forward:
            graph.add_state(state)
        else:
            attrs = side.graph.state_attrs(state)
            if ATTR_FUTURE_SYMBOL in attrs:
                graph.add_state(state, **{ATTR_FUTURE_SYMBOL: attrs[ATTR_FUTURE_SYMBOL]})
            else:
                graph.add_state(state)

    weights: dict[tuple[Hashable, Hashable, Any], float] = {}
    for pair, mass in joint.items():
        source = pair[coord]
        pi_source = pi_marginal.get(source, 0.0)
        if mass <= 0.0 or pi_source <= 0.0:
            continue
        for transition in side.graph.out_transitions(source):
            symbol = transition.data.get(ATTR_EMISSION)
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            if symbol is None or prob <= 0.0:
                continue
            key = (source, transition.target, symbol)
            weights[key] = weights.get(key, 0.0) + mass * prob / pi_source

    for (source, target, symbol), prob in weights.items():
        if prob <= 0.0:
            continue
        existing = [
            t for t in graph.out_transitions(source) if t.data.get(ATTR_EMISSION) == symbol and t.target == target
        ]
        if existing:
            continue
        graph.add_transition(source, target, **{ATTR_PROB: prob, ATTR_EMISSION: symbol})

    eps = EpsilonMachine(
        graph=graph,
        initial_distribution=pi_marginal,
        observation_alphabet=bidir.observation_alphabet,
    )
    eps.validate()
    return eps
