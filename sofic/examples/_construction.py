"""Private helpers shared by the example constructor modules."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_PROB


def _uniform_initial(states: Sequence[Hashable]) -> dict[Hashable, float]:
    if not states:
        return {}
    mass = 1.0 / len(states)
    return dict.fromkeys(states, mass)


def _stationary_initial(states: Sequence[Hashable], transition: Any) -> dict[Hashable, Any]:
    """Stationary distribution of ``transition`` keyed by ``states``.

    Float matrices are row-normalized first and fall back to the uniform
    distribution when a row is empty or the solve fails; exact (sympy) matrices
    are solved as given and keep exact entries.
    """
    from sofic.generators.prob import as_prob, has_symbolic
    from sofic.generators.stationary import stationary_distribution_from_transition

    if not states:
        return {}
    matrix = np.asarray(transition)
    if not (matrix.dtype == object and has_symbolic(matrix.ravel())):
        matrix = np.asarray(matrix, dtype=float)
        row_sums = matrix.sum(axis=1)
        if np.any(row_sums <= 0.0):
            return _uniform_initial(states)
        matrix = matrix / row_sums[:, None]
    try:
        pi = stationary_distribution_from_transition(matrix)
    except (ValueError, np.linalg.LinAlgError):
        return _uniform_initial(states)
    if pi.dtype == object or has_symbolic(pi.ravel()):
        return {state: as_prob(pi[i]) for i, state in enumerate(states)}
    return {state: float(pi[i]) for i, state in enumerate(states)}


def _normalize_edges(
    edges: Sequence[tuple[Hashable, Hashable, Any, float]],
) -> list[tuple[Hashable, Hashable, Any, float]]:
    row_totals: dict[Hashable, float] = {}
    for source, _target, _symbol, prob in edges:
        row_totals[source] = row_totals.get(source, 0.0) + float(prob)
    normalized = []
    for source, target, symbol, prob in edges:
        total = row_totals[source]
        normalized.append((source, target, symbol, float(prob) / total if total else 0.0))
    return normalized


def _edge_machine(
    edges: Iterable[tuple[Hashable, Hashable, Any, float]],
    *,
    machine_type: type[MealyHMM] = EpsilonMachine,
    name: str | None = None,
    initial_distribution: Mapping[Hashable, float] | None = None,
    normalize: bool = True,
    validate: bool = True,
) -> MealyHMM:
    edge_list = list(edges)
    if normalize:
        edge_list = _normalize_edges(edge_list)

    states = list(dict.fromkeys([source for source, *_ in edge_list] + [target for _source, target, *_ in edge_list]))
    symbols = frozenset(symbol for _source, _target, symbol, _prob in edge_list)
    if initial_distribution is not None:
        initial = dict(initial_distribution)
    else:
        index = {state: i for i, state in enumerate(states)}
        transition = np.zeros((len(states), len(states)), dtype=float)
        for source, target, _symbol, prob in edge_list:
            transition[index[source], index[target]] += float(prob)
        initial = _stationary_initial(states, transition)

    machine = machine_type(initial_distribution=initial, observation_alphabet=symbols)
    if name is not None:
        machine.name = name
    for state in states:
        machine.graph.add_state(state)
    for source, target, symbol, prob in edge_list:
        if prob > 0.0:
            machine.graph.add_transition(source, target, **{ATTR_EMISSION: symbol, ATTR_PROB: float(prob)})
    if validate:
        machine.validate()
    return machine


def _relabel(
    machine: MealyHMM,
    *,
    symbols: Mapping[Any, Any] | None = None,
    states: Mapping[Hashable, Hashable] | None = None,
    machine_type: type[MealyHMM] | None = None,
    name: str | None = None,
) -> MealyHMM:
    """Copy ``machine`` with emissions and states renamed; probabilities are kept as-is.

    A ``states`` mapping that covers every state also fixes the output state
    order (the order of its values).
    """
    symbol_map = dict(symbols or {})
    state_map = dict(states or {})
    old_states = list(machine.graph.states())
    if set(state_map) >= set(old_states):
        old_states = [old for old in state_map if old in set(old_states)]
    out = (machine_type or type(machine))(
        initial_distribution={state_map.get(s, s): p for s, p in machine.initial_distribution.items()},
        observation_alphabet=frozenset(symbol_map.get(x, x) for x in machine.observation_alphabet),
    )
    if name is not None:
        out.name = name
    for state in old_states:
        out.graph.add_state(state_map.get(state, state), **dict(machine.graph.state_attrs(state)))
    for t in machine.transitions():
        data = dict(t.data)
        data[ATTR_EMISSION] = symbol_map.get(data[ATTR_EMISSION], data[ATTR_EMISSION])
        out.graph.add_transition(state_map.get(t.source, t.source), state_map.get(t.target, t.target), **data)
    out.validate()
    return out
