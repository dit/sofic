"""Bidirectional ε-machine construction (Ellison et al., arXiv:1107.2168, Sec. VII)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np

from sofic.exceptions import StochasticValidationError
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.prob import (
    as_prob,
    has_symbolic,
    is_positive_mass,
    is_symbolic,
    is_zero,
    probs_equal,
    row_sums_to_one,
    simplify_prob,
    sum_probs,
    zeros,
)
from sofic.generators.reversal import time_reverse_stochastic
from sofic.generators.stationary import (
    stationary_distribution_from_transition,
    stationary_distribution_hmm,
)
from sofic.graph import ATTR_EMISSION, ATTR_FUTURE_SYMBOL, ATTR_PROB, TransitionGraph
from sofic.states import next_sequential_label_index, sequential_labels


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
    raw: dict[tuple[Hashable, Hashable], list[tuple[tuple[Hashable, Hashable], Any, Any]]] = defaultdict(list)

    for alpha in sorted(forward.states(), key=repr):
        for gamma in reverse_states:
            source = (alpha, gamma)
            if sources is not None and source not in sources:
                continue
            for transition in forward.graph.out_transitions(alpha):
                symbol = transition.data.get(ATTR_EMISSION)
                prob_forward = as_prob(transition.data.get(ATTR_PROB, 0.0))
                if symbol is None or not is_positive_mass(prob_forward):
                    continue
                beta = transition.target
                for delta in reverse_states:
                    tex = _reverse_tex_probability(rev_time, gamma, delta, symbol)
                    if not is_positive_mass(tex):
                        continue
                    raw[source].append(((beta, delta), symbol, tex))

    return _graph_from_raw(raw)


def _graph_from_raw(
    raw: dict[tuple[Hashable, Hashable], list[tuple[tuple[Hashable, Hashable], Any, Any]]],
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
        aggregated: dict[tuple[tuple[Hashable, Hashable], Any], Any] = {}
        for target, symbol, weight in entries:
            key = (_intern(target), symbol)
            if key in aggregated:
                aggregated[key] = sum_probs([aggregated[key], weight])
            else:
                aggregated[key] = as_prob(weight)
        total = sum_probs(aggregated.values())
        if not is_positive_mass(total):
            continue
        symbolic = is_symbolic(total) or has_symbolic(aggregated.values())
        for (target, symbol), weight in aggregated.items():
            if symbolic:
                prob = _clean_probability(simplify_prob(as_prob(weight) / as_prob(total)))
            else:
                prob = _clean_probability(float(weight) / float(total))
            graph.add_transition(
                source,
                target,
                **{ATTR_PROB: prob, ATTR_EMISSION: symbol},
            )
    return graph


def _clean_probability(probability: Any) -> Any:
    if is_symbolic(probability):
        return simplify_prob(probability)
    rounded = round(float(probability), 15)
    if np.isclose(probability, rounded, rtol=0.0, atol=1e-15):
        return rounded
    return float(probability)


def _reverse_tex_probability(
    rev_time: EpsilonMachine,
    gamma: Hashable,
    delta: Hashable,
    symbol: Any,
) -> Any:
    """T̃_x(γ, δ) from the time-reversed reverse ε-machine (Eq. 18)."""
    masses: list[Any] = []
    for transition in rev_time.graph.out_transitions(gamma):
        if transition.data.get(ATTR_EMISSION) != symbol:
            continue
        if transition.target != delta:
            continue
        masses.append(as_prob(transition.data.get(ATTR_PROB, 0.0)))
    if not masses:
        return 0.0
    return sum_probs(masses)


def _prune_to_stationary_support(
    graph: TransitionGraph,
    forward: EpsilonMachine,
    reverse: EpsilonMachine,
    *,
    tol: float = 1e-12,
) -> tuple[TransitionGraph, dict[tuple[Hashable, Hashable], Any]]:
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
    future-symbol filter.  When multiple undirected components admit a stationary
    joint π, :func:`_joint_pi_minimum_support` selects the one whose marginals
    match the forward/reverse causal-state stationary distributions.
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
        transition.data.get(ATTR_EMISSION) == symbol
        and is_positive_mass(as_prob(transition.data.get(ATTR_PROB, 0.0)))
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


def _joint_pi_on_pair_subset(
    graph: TransitionGraph,
    pairs: list[tuple[Hashable, Hashable]],
    *,
    tol: float = 1e-12,
) -> dict[tuple[Hashable, Hashable], Any] | None:
    """Stationary joint π over a closed joint-state component, or ``None`` if unavailable.

    The joint stationary distribution is the normalized left eigenvector (eigenvalue
    one) of the component's row-stochastic transition matrix — the general HMM
    stationary distribution.  A component that is not row-stochastic (i.e. leaks
    probability outside ``pairs``) is not a closed recurrent class and is rejected.
    """
    if not pairs:
        return None

    states = sorted(pairs, key=repr)
    index = {state: i for i, state in enumerate(states)}
    n = len(states)
    edge_probs = [
        as_prob(transition.data.get(ATTR_PROB, 0.0))
        for state in states
        for transition in graph.out_transitions(state)
    ]
    symbolic = has_symbolic(edge_probs)
    matrix = zeros((n, n), symbolic=symbolic)
    for state in states:
        row = index[state]
        for transition in graph.out_transitions(state):
            column = index.get(transition.target)
            if column is None:
                continue
            matrix[row, column] = as_prob(matrix[row, column]) + as_prob(transition.data.get(ATTR_PROB, 0.0))

    if symbolic:
        for row in range(n):
            if not row_sums_to_one([as_prob(matrix[row, j]) for j in range(n)]):
                return None
    elif not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
        return None

    try:
        pi = stationary_distribution_from_transition(matrix)
    except (StochasticValidationError, ValueError):
        return None

    if symbolic:
        joint = {
            states[i]: simplify_prob(as_prob(pi[i]))
            for i in range(n)
            if is_positive_mass(pi[i])
        }
    else:
        joint = {states[i]: float(pi[i]) for i in range(n) if float(pi[i]) > tol}
    return joint or None


def _undirected_components(
    graph: TransitionGraph,
) -> list[set[tuple[Hashable, Hashable]]]:
    import networkx as nx

    undirected = graph.nx.to_undirected()
    return [set(component) for component in nx.connected_components(undirected)]


def _anatomy_gap_for_joint(
    graph: TransitionGraph,
    joint: dict[tuple[Hashable, Hashable], Any],
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

    if is_symbolic(h_mu) or is_symbolic(b_mu) or is_symbolic(r_mu):
        import sympy as sp

        gap = sp.simplify(sp.sympify(b_mu) + sp.sympify(r_mu) - sp.sympify(h_mu))
        return 0.0 if gap == 0 else float("inf")
    return abs(float(b_mu) + float(r_mu) - float(h_mu))


def _stationary_state_probabilities(machine: EpsilonMachine) -> dict[Hashable, Any]:
    """Stationary causal-state distribution of ``machine`` keyed by state label."""
    index = machine.reindex()
    pi = machine.stationary_distribution()
    symbolic = pi.dtype == object or has_symbolic(pi.ravel())
    if symbolic:
        return {state: simplify_prob(as_prob(pi[i])) for i, state in enumerate(index.states)}
    return {state: float(pi[i]) for i, state in enumerate(index.states)}


def _abs_marginal_error(left: Any, right: Any) -> float:
    """Absolute marginal deviation (exact sympy simplify, else numeric)."""
    if probs_equal(left, right):
        return 0.0
    if is_symbolic(left) or is_symbolic(right):
        import sympy as sp

        simplified = sp.simplify(sp.Abs(sp.sympify(left) - sp.sympify(right)))
        if getattr(simplified, "free_symbols", None):
            return float("inf")
        try:
            return abs(float(simplified))
        except (TypeError, ValueError):
            return float("inf")
    return abs(float(left) - float(right))


def _joint_marginal_mismatch(
    joint: dict[tuple[Hashable, Hashable], Any],
    pi_plus: dict[Hashable, Any],
    pi_minus: dict[Hashable, Any],
) -> float:
    """Max abs deviation of ``joint``'s marginals from the target stationary marginals.

    A valid bidirectional presentation is a closed recurrent joint class whose
    forward/reverse marginals equal the forward/reverse causal-state stationary
    distributions.  Spurious closed sub-cycles (e.g. the all-``0`` period-3 cycle
    of the Nemo process) violate this and are rejected by the selector below.
    """
    forward_marginal: dict[Hashable, Any] = {}
    reverse_marginal: dict[Hashable, Any] = {}
    for (alpha, gamma), mass in joint.items():
        forward_marginal[alpha] = sum_probs([forward_marginal.get(alpha, 0), mass])
        reverse_marginal[gamma] = sum_probs([reverse_marginal.get(gamma, 0), mass])

    error = 0.0
    for state in set(pi_plus) | set(forward_marginal):
        error = max(
            error,
            _abs_marginal_error(forward_marginal.get(state, 0), pi_plus.get(state, 0)),
        )
    for state in set(pi_minus) | set(reverse_marginal):
        error = max(
            error,
            _abs_marginal_error(reverse_marginal.get(state, 0), pi_minus.get(state, 0)),
        )
    return error


def _joint_pi_minimum_support(
    graph: TransitionGraph,
    forward: EpsilonMachine,
    reverse: EpsilonMachine,
    *,
    tol: float = 1e-12,
    marginal_tol: float = 1e-6,
) -> dict[tuple[Hashable, Hashable], Any]:
    """Pick the closed undirected component that is the true bidirectional class.

    The joint π on each component is the general HMM stationary distribution (left
    eigenvector for eigenvalue one).  When the Eq. (15) graph has more than one
    closed recurrent component (e.g. the Nemo process, which admits a spurious
    all-``0`` period-3 cycle alongside the genuine 6-state machine), the correct
    class is the one whose forward/reverse marginals equal the forward/reverse
    causal-state stationary distributions.  Among marginal-matching components the
    information-anatomy identity ``h_μ = b_μ + r_μ`` and then the support size break
    remaining ties; if none match we fall back to the smallest-mismatch component.
    """
    pairs = [state for state in graph.states() if isinstance(state, tuple) and len(state) == 2]
    if not pairs:
        return {}

    pi_plus = _stationary_state_probabilities(forward)
    pi_minus = _stationary_state_probabilities(reverse)

    components = sorted(
        _undirected_components(graph),
        key=lambda component: (len(component), sorted(component, key=repr)),
    )

    candidates: list[tuple[float, float, int, dict[tuple[Hashable, Hashable], Any]]] = []
    for component in components:
        component_pairs = [pair for pair in pairs if pair in component]
        joint = _joint_pi_on_pair_subset(graph, component_pairs, tol=tol)
        if joint is None:
            continue
        marginal_error = _joint_marginal_mismatch(joint, pi_plus, pi_minus)
        anatomy_gap = _anatomy_gap_for_joint(graph, joint, forward, reverse)
        candidates.append((marginal_error, anatomy_gap, len(joint), joint))

    if not candidates:
        return {}

    all_symbolic = all(has_symbolic(candidate[3].values()) for candidate in candidates)
    if all_symbolic:
        matching = [candidate for candidate in candidates if candidate[0] == 0.0]
    else:
        matching = [candidate for candidate in candidates if candidate[0] <= marginal_tol]
    if matching:
        best = min(matching, key=lambda candidate: (candidate[1], candidate[2]))
    else:
        best = min(candidates, key=lambda candidate: (candidate[0], candidate[1], candidate[2]))
    return best[3]


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
    from sofic.generators.stochastic import normalize_row_weights

    graph = TransitionGraph()
    for state in machine.states():
        attrs = machine.graph.state_attrs(state)
        graph.add_state(mapping[state], **attrs)
    for state in machine.states():
        outgoing = list(machine.graph.out_transitions(state))
        merged: dict[tuple[Hashable, Any], Any] = {}
        for transition in outgoing:
            prob = as_prob(transition.data.get(ATTR_PROB, 0.0))
            emission = transition.data.get(ATTR_EMISSION)
            key = (mapping[transition.target], emission)
            if key in merged:
                merged[key] = sum_probs([merged[key], prob])
            else:
                merged[key] = prob
        merged = normalize_row_weights(merged)
        source = mapping[state]
        for (target, emission), prob in merged.items():
            attrs = {ATTR_PROB: as_prob(prob)}
            if emission is not None:
                attrs[ATTR_EMISSION] = emission
            graph.add_transition(source, target, **attrs)
    initial = {mapping[state]: as_prob(prob) for state, prob in machine.initial_distribution.items()}
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
            probs = [as_prob(t.data.get(ATTR_PROB, 0.0)) for t in outgoing]
            if not row_sums_to_one(probs):
                changed = True
                continue
            next_keep.add(state)
        keep = next_keep
    return keep


def joint_distribution(bidir: BidirectionalEpsilonMachine) -> dict[tuple[Hashable, Hashable], Any]:
    """Return π(α, γ) = P(S⁺ = α, S⁻ = γ) under the bidirectional stationary distribution."""
    if bidir._joint_pi is not None:
        return dict(bidir._joint_pi)

    idx = bidir.reindex()
    pi = stationary_distribution_hmm(bidir)
    symbolic = pi.dtype == object or has_symbolic(pi.ravel())
    joint: dict[tuple[Hashable, Hashable], Any] = {}
    for index, state in enumerate(idx.states):
        alpha, gamma = state
        mass = as_prob(pi[index])
        if symbolic:
            if is_positive_mass(mass):
                joint[(alpha, gamma)] = simplify_prob(mass)
        elif float(mass) > 0.0:
            joint[(alpha, gamma)] = float(mass)
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
    probs: list[Any] = []
    for (alpha, gamma), mass in joint.items():
        if not is_positive_mass(mass):
            continue
        for transition in bidir.graph.out_transitions((alpha, gamma)):
            symbol = transition.data.get(ATTR_EMISSION)
            prob = as_prob(transition.data.get(ATTR_PROB, 0.0))
            if symbol is None or not is_positive_mass(prob):
                continue
            beta, delta = transition.target
            if has_symbolic([mass, prob]):
                weight = simplify_prob(as_prob(mass) * as_prob(prob))
            else:
                weight = float(mass) * float(prob)
            if not is_positive_mass(weight):
                continue
            outcomes.append((alpha, gamma, symbol, beta, delta))
            probs.append(weight)

    if not probs:
        raise StochasticValidationError("bidirectional step distribution is empty")

    total = sum_probs(probs)
    if has_symbolic(probs) or is_symbolic(total):
        from dit.symbolic import symbolic_distribution

        return symbolic_distribution(
            outcomes,
            [simplify_prob(as_prob(p) / as_prob(total)) for p in probs],
        )
    return dit.Distribution(outcomes, [float(p) / float(total) for p in probs])


def _require_dit_for_step():
    from sofic.generators.measures import require_dit

    return require_dit("bidirectional step distributions")


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
    symbolic = has_symbolic(joint.values())

    pi_marginal: dict[Hashable, Any] = {}
    for pair, mass in joint.items():
        state = pair[coord]
        if state in pi_marginal:
            pi_marginal[state] = sum_probs([pi_marginal[state], mass])
        else:
            pi_marginal[state] = as_prob(mass)

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

    weights: dict[tuple[Hashable, Hashable, Any], Any] = {}
    for pair, mass in joint.items():
        source = pair[coord]
        pi_source = pi_marginal.get(source, 0)
        if not is_positive_mass(mass) or is_zero(pi_source):
            continue
        for transition in side.graph.out_transitions(source):
            symbol = transition.data.get(ATTR_EMISSION)
            prob = as_prob(transition.data.get(ATTR_PROB, 0.0))
            if symbol is None or not is_positive_mass(prob):
                continue
            key = (source, transition.target, symbol)
            contribution = (
                simplify_prob(as_prob(mass) * as_prob(prob) / as_prob(pi_source))
                if symbolic or has_symbolic([mass, prob, pi_source])
                else float(mass) * float(prob) / float(pi_source)
            )
            if key in weights:
                weights[key] = sum_probs([weights[key], contribution])
            else:
                weights[key] = contribution

    for (source, target, symbol), prob in weights.items():
        if not is_positive_mass(prob):
            continue
        existing = [
            t for t in graph.out_transitions(source) if t.data.get(ATTR_EMISSION) == symbol and t.target == target
        ]
        if existing:
            continue
        graph.add_transition(source, target, **{ATTR_PROB: as_prob(prob), ATTR_EMISSION: symbol})

    eps = EpsilonMachine(
        graph=graph,
        initial_distribution=pi_marginal,
        observation_alphabet=bidir.observation_alphabet,
    )
    eps.validate()
    return eps
