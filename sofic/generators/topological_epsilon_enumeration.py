"""Enumeration of canonical topological ε-machines.

Implements Algorithm 2 of Johnson, Crutchfield, Ellison & McTague (2010),
*Enumerating Finitary Processes* (arXiv:1011.0036), filtering incomplete
accessible DFA strings to strongly connected, minimal, canonical representatives.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence

import numpy as np

from sofic.automata.idfa import (
    MISSING_TRANSITION,
    IDFAEnumerationError,
    _delta_table,
    idfa_string_to_topological_graph,
    iter_idfa_strings,
    rank_idfa_string,
    reroot_idfa_string,
    transition_count,
    validate_idfa_string,
)
from sofic.exceptions import SoficValidationError
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.graph import ATTR_EMISSION, ATTR_PROB

__all__ = [
    "TopologicalEpsilonEnumerationError",
    "count_topological_epsilon_machines",
    "epsilon_machine_to_idfa_string",
    "idfa_string_to_epsilon_machine",
    "idfa_string_to_topological_graph",
    "is_canonical_topological_epsilon",
    "is_minimal_idfa",
    "is_topological_epsilon_string",
    "iter_topological_epsilon_machines",
    "iter_topological_epsilon_strings",
]


class TopologicalEpsilonEnumerationError(SoficValidationError):
    """Raised when topological ε-machine enumeration fails."""


def _uniform_stationary_distribution(
    transitions: Sequence[int],
    *,
    n: int,
    k: int,
) -> dict[int, float]:
    table = _delta_table(transitions, n=n, k=k)
    outdegree = [sum(1 for symbol in range(k) if table[state][symbol] is not None) for state in range(n)]
    matrix = np.zeros((n, n), dtype=float)
    for state in range(n):
        if outdegree[state] == 0:
            continue
        weight = 1.0 / outdegree[state]
        for symbol in range(k):
            target = table[state][symbol]
            if target is not None:
                matrix[state, target] += weight
    eigenvalues, vectors = np.linalg.eig(matrix.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    distribution = np.real(vectors[:, index])
    if distribution.sum() < 0.0:
        distribution = -distribution
    distribution = np.maximum(distribution, 0.0)
    total = float(distribution.sum())
    if total <= 0.0:
        raise TopologicalEpsilonEnumerationError("failed to compute stationary distribution")
    distribution /= total
    return {state: float(distribution[state]) for state in range(n)}


def idfa_string_to_epsilon_machine(
    transitions: Sequence[int],
    *,
    n: int,
    k: int,
    alphabet: Sequence[object] | None = None,
) -> EpsilonMachine:
    """Decode an IDFA string into an :class:`~sofic.generators.epsilon_machine.EpsilonMachine`.

    Outgoing edges from each state receive uniform probability ``1 / outdegree``,
    matching the topological ε-machine convention of Johnson et al. (2010).
    """
    validate_idfa_string(transitions, n=n, k=k)
    if alphabet is None:
        symbols = tuple(range(k))
    else:
        if len(alphabet) != k:
            raise TopologicalEpsilonEnumerationError("alphabet length must equal k")
        symbols = tuple(alphabet)

    table = _delta_table(transitions, n=n, k=k)
    outdegree = [sum(1 for symbol in range(k) if table[state][symbol] is not None) for state in range(n)]
    eps = EpsilonMachine(
        initial_distribution=_uniform_stationary_distribution(transitions, n=n, k=k),
        observation_alphabet=frozenset(symbols),
    )
    for state in range(n):
        eps.graph.add_state(state)
    for state in range(n):
        if outdegree[state] == 0:
            continue
        probability = 1.0 / outdegree[state]
        for symbol_index, symbol in enumerate(symbols):
            target = table[state][symbol_index]
            if target is None:
                continue
            eps.graph.add_transition(
                state,
                target,
                **{ATTR_PROB: probability, ATTR_EMISSION: symbol},
            )
    eps.validate()
    return eps


def epsilon_machine_to_idfa_string(
    eps: EpsilonMachine,
    *,
    symbol_order: Sequence[object] | None = None,
    canonical: bool = True,
) -> tuple[int, ...]:
    """Encode an ε-machine as an incomplete accessible DFA transition string.

    Probabilities are ignored.  Missing symbol transitions are encoded with
    :data:`sofic.automata.idfa.MISSING_TRANSITION`.  If ``canonical`` is true,
    all states are tried as roots and the rank-minimal IDFA string is returned.
    Otherwise, the first state in deterministic label order is used as the root.
    """
    from sofic.generators.synchronization import graph_from_epsilon_machine

    graph = graph_from_epsilon_machine(eps)
    states = tuple(graph.states)
    if not states:
        raise TopologicalEpsilonEnumerationError("epsilon machine must have at least one state")

    symbols = _validated_symbol_order(graph.alphabet, symbol_order)
    if not symbols:
        raise TopologicalEpsilonEnumerationError("alphabet must be non-empty")

    if not canonical:
        root = _sorted_by_repr(states)[0]
        return _encode_topological_graph_from_root(graph.states, graph.transitions, symbols, root)

    candidates: list[tuple[int, tuple[int, ...]]] = []
    for root in states:
        transitions = _encode_topological_graph_from_root(graph.states, graph.transitions, symbols, root)
        try:
            rank = rank_idfa_string(transitions, n=len(states), k=len(symbols))
        except IDFAEnumerationError:
            continue
        candidates.append((rank, transitions))
    if not candidates:
        raise TopologicalEpsilonEnumerationError("encoded graph is not an accessible IDFA string")
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[1]


def _validated_symbol_order(
    alphabet: frozenset[object],
    symbol_order: Sequence[object] | None,
) -> tuple[object, ...]:
    if symbol_order is None:
        return tuple(_sorted_by_repr(alphabet))

    symbols = tuple(symbol_order)
    if len(frozenset(symbols)) != len(symbols):
        raise TopologicalEpsilonEnumerationError("symbol_order must contain unique symbols")
    if frozenset(symbols) != alphabet:
        raise TopologicalEpsilonEnumerationError("symbol_order must match the epsilon machine alphabet")
    return symbols


def _sorted_by_repr(values: Sequence[object] | frozenset[object]) -> tuple[object, ...]:
    return tuple(sorted(values, key=lambda value: (type(value).__module__, type(value).__qualname__, repr(value))))


def _encode_topological_graph_from_root(
    states: frozenset[Hashable],
    edges: Mapping[tuple[Hashable, object], Hashable],
    symbols: Sequence[object],
    root: Hashable,
) -> tuple[int, ...]:
    edge_map = dict(edges)
    if root not in states:
        raise TopologicalEpsilonEnumerationError(f"unknown root state {root!r}")

    state_to_index: dict[Hashable, int] = {root: 0}
    index_to_state: list[Hashable] = [root]
    transitions: list[int] = []
    state_index = 0

    while state_index < len(index_to_state):
        state = index_to_state[state_index]
        for symbol in symbols:
            target = edge_map.get((state, symbol))
            if target is None:
                transitions.append(MISSING_TRANSITION)
                continue
            if target not in states:
                raise TopologicalEpsilonEnumerationError(f"transition target {target!r} is not a graph state")
            if target not in state_to_index:
                state_to_index[target] = len(index_to_state)
                index_to_state.append(target)
            transitions.append(state_to_index[target])
        state_index += 1

    if len(index_to_state) != len(states):
        raise TopologicalEpsilonEnumerationError(
            "epsilon machine graph must be initially connected from the selected root"
        )

    validate_idfa_string(transitions, n=len(states), k=len(symbols))
    return tuple(transitions)


def _reachable_states(transitions: Sequence[int], *, n: int, k: int, start: int) -> set[int]:
    table = _delta_table(transitions, n=n, k=k)
    seen = {start}
    stack = [start]
    while stack:
        state = stack.pop()
        for symbol in range(k):
            target = table[state][symbol]
            if target is not None and target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def is_strongly_connected_idfa(transitions: Sequence[int], *, n: int, k: int) -> bool:
    """Return whether every state can reach every other state."""
    return all(len(_reachable_states(transitions, n=n, k=k, start=source)) == n for source in range(n))


def is_minimal_idfa(transitions: Sequence[int], *, n: int, k: int) -> bool:
    """Return whether the incomplete accessible DFA is minimal (Algorithm 2 step 4)."""
    validate_idfa_string(transitions, n=n, k=k)
    if n == 1:
        return True

    table = _delta_table(transitions, n=n, k=k)
    partition: list[set[int]] = [set(range(n))]

    changed = True
    while changed:
        changed = False
        block_index = {state: index for index, block in enumerate(partition) for state in block}
        new_partition: list[set[int]] = []
        for block in partition:
            groups: dict[tuple[object, ...], set[int]] = {}
            for state in block:
                signature = tuple(
                    None if table[state][symbol] is None else block_index[table[state][symbol]] for symbol in range(k)
                )
                groups.setdefault(signature, set()).add(state)
            if len(groups) > 1:
                changed = True
            new_partition.extend(groups.values())
        partition = new_partition
    return all(len(block) == 1 for block in partition)


def is_topological_epsilon_string(
    transitions: Sequence[int],
    *,
    n: int,
    k: int,
    check_minimal: bool = True,
) -> bool:
    """Return whether ``transitions`` passes the structural ε-machine tests."""
    validate_idfa_string(transitions, n=n, k=k)
    defined = transition_count(transitions)
    if defined < n:
        return False
    if n > 1 and defined >= n * k:
        return False
    if not is_strongly_connected_idfa(transitions, n=n, k=k):
        return False
    return not (check_minimal and not is_minimal_idfa(transitions, n=n, k=k))


def is_canonical_topological_epsilon(
    transitions: Sequence[int],
    *,
    n: int,
    k: int,
    check_minimal: bool = True,
) -> bool:
    """Return whether ``transitions`` is a canonical topological ε-machine (Algorithm 2)."""
    if not is_topological_epsilon_string(transitions, n=n, k=k, check_minimal=check_minimal):
        return False
    if n == 1:
        return True

    try:
        rank = rank_idfa_string(transitions, n=n, k=k)
    except IDFAEnumerationError:
        return False
    for root in range(1, n):
        rotated = reroot_idfa_string(transitions, new_root=root, n=n, k=k)
        if rotated is None:
            return False
        try:
            rotated_rank = rank_idfa_string(rotated, n=n, k=k)
        except IDFAEnumerationError:
            continue
        if rotated_rank < rank:
            return False
        if rotated_rank == rank and rotated < transitions:
            return False
    return True


def iter_topological_epsilon_strings(
    k: int,
    n: int,
    *,
    check_minimal: bool = True,
) -> Iterator[tuple[int, ...]]:
    """Yield canonical topological ε-machine transition strings.

    By default, strings whose states are not minimal causal states are filtered
    out.  Pass ``check_minimal=False`` to enumerate the broader structural class.
    """
    for transitions in iter_idfa_strings(k, n):
        if is_canonical_topological_epsilon(transitions, n=n, k=k, check_minimal=check_minimal):
            yield transitions


def iter_topological_epsilon_machines(
    k: int,
    n: int,
    *,
    alphabet: Sequence[object] | None = None,
    check_minimal: bool = True,
) -> Iterator[EpsilonMachine]:
    """Yield uniform-probability :class:`~sofic.generators.epsilon_machine.EpsilonMachine` objects."""
    for transitions in iter_topological_epsilon_strings(k, n, check_minimal=check_minimal):
        yield idfa_string_to_epsilon_machine(transitions, n=n, k=k, alphabet=alphabet)


def count_topological_epsilon_machines(k: int, n: int) -> int:
    """Return ``E_{n,k}``, the number of canonical topological ε-machines."""
    return sum(1 for _ in iter_topological_epsilon_strings(k, n))
