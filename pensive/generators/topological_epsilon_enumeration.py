"""Enumeration of canonical topological ε-machines.

Implements Algorithm 2 of Johnson, Crutchfield, Ellison & McTague (2010),
*Enumerating Finitary Processes* (arXiv:1011.0036), filtering incomplete
accessible DFA strings to strongly connected, minimal, canonical representatives.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from pensive.automata.idfa import (
    _delta_table,
    idfa_string_to_topological_graph,
    iter_idfa_strings,
    reroot_idfa_string,
    transition_count,
    validate_idfa_string,
)
from pensive.exceptions import PensiveValidationError
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.graph import ATTR_EMISSION, ATTR_PROB

__all__ = [
    "TopologicalEpsilonEnumerationError",
    "count_topological_epsilon_machines",
    "idfa_string_to_epsilon_machine",
    "idfa_string_to_topological_graph",
    "is_canonical_topological_epsilon",
    "is_minimal_idfa",
    "is_topological_epsilon_string",
    "iter_topological_epsilon_machines",
    "iter_topological_epsilon_strings",
]


class TopologicalEpsilonEnumerationError(PensiveValidationError):
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
    """Decode an IDFA string into an :class:`~pensive.generators.epsilon_machine.EpsilonMachine`.

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
    partition: list[set[int]] = [{state} for state in range(n)]

    def block_index(state: int, blocks: list[set[int]]) -> int:
        for index, block in enumerate(blocks):
            if state in block:
                return index
        raise TopologicalEpsilonEnumerationError(f"state {state} missing from partition")

    changed = True
    while changed:
        changed = False
        new_partition: list[set[int]] = []
        for block in partition:
            groups: dict[tuple[object, ...], set[int]] = {}
            for state in block:
                signature = tuple(
                    None if table[state][symbol] is None else block_index(table[state][symbol], partition)
                    for symbol in range(k)
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


def is_canonical_topological_epsilon(transitions: Sequence[int], *, n: int, k: int) -> bool:
    """Return whether ``transitions`` is a canonical topological ε-machine (Algorithm 2)."""
    if not is_topological_epsilon_string(transitions, n=n, k=k):
        return False
    if n == 1:
        return True
    from pensive.automata.idfa import IDFAEnumerationError, rank_idfa_string

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


def iter_topological_epsilon_strings(k: int, n: int) -> Iterator[tuple[int, ...]]:
    """Yield canonical topological ε-machine transition strings."""
    for transitions in iter_idfa_strings(k, n):
        if is_canonical_topological_epsilon(transitions, n=n, k=k):
            yield transitions


def iter_topological_epsilon_machines(
    k: int,
    n: int,
    *,
    alphabet: Sequence[object] | None = None,
) -> Iterator[EpsilonMachine]:
    """Yield uniform-probability :class:`~pensive.generators.epsilon_machine.EpsilonMachine` objects."""
    for transitions in iter_topological_epsilon_strings(k, n):
        yield idfa_string_to_epsilon_machine(transitions, n=n, k=k, alphabet=alphabet)


def count_topological_epsilon_machines(k: int, n: int) -> int:
    """Return ``E_{n,k}``, the number of canonical topological ε-machines."""
    return sum(1 for _ in iter_topological_epsilon_strings(k, n))
