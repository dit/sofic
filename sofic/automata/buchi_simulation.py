"""Büchi acceptance for lasso and ultimately periodic ω-words."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import networkx as nx

from sofic.automata.buchi import BuchiAutomaton
from sofic.graph import ATTR_SYMBOL, EPSILON


def accepts_lasso_buchi(ba: BuchiAutomaton, prefix: Sequence[Any], loop: Sequence[Any]) -> bool:
    """Accept if repeating ``loop`` after ``prefix`` visits an accepting state infinitely often.

    ``loop`` must be non-empty: ``prefix loop^omega`` is an infinite word only then.
    """
    if not loop:
        raise ValueError("an ultimately periodic omega-word needs a non-empty loop")
    post = ba._run_nfa(prefix)
    if not post:
        return False
    return _accepting_cycle_reachable(ba, post, loop)


def accepts_omega_buchi(ba: BuchiAutomaton, word: Sequence[Any]) -> bool:
    """Accept ultimately periodic ω-words represented as ``(prefix, loop)``."""
    if (
        len(word) == 2
        and not isinstance(word, (str, bytes))
        and isinstance(word[0], Sequence)
        and not isinstance(word[0], (str, bytes))
        and isinstance(word[1], Sequence)
        and not isinstance(word[1], (str, bytes))
    ):
        return accepts_lasso_buchi(ba, word[0], word[1])
    raise NotImplementedError("BuchiAutomaton.accepts_omega supports ultimately periodic inputs as (prefix, loop) only")


def accepted_lasso_buchi(ba: BuchiAutomaton) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    """Return ``(prefix, loop)`` with ``prefix loop^omega`` accepted, or ``None`` when the ω-language is empty.

    The ω-language is non-empty iff some accepting state reachable from an
    initial state lies in a strongly connected component with a
    symbol-consuming edge :cite:`BaierKatoen2008`.
    """
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(ba.states())
    for transition in ba.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is not None:
            graph.add_edge(transition.source, transition.target, symbol=symbol)

    def word(path: list[Hashable]) -> list[Any]:
        symbols: list[Any] = []
        for source, target in zip(path, path[1:], strict=False):
            labels = [data["symbol"] for data in graph.get_edge_data(source, target).values()]
            consuming = sorted((label for label in labels if label is not EPSILON), key=repr)
            if consuming and EPSILON not in labels:
                symbols.append(consuming[0])
        return symbols

    reachable: set[Hashable] = set()
    for initial in ba.initial_states:
        reachable |= {initial} | nx.descendants(graph, initial)
    for component in sorted(nx.strongly_connected_components(graph.subgraph(reachable)), key=repr):
        accepting = sorted((state for state in component if state in ba.accepting_states), key=repr)
        if not accepting:
            continue
        sub = graph.subgraph(component)
        consuming = [(u, v, d["symbol"]) for u, v, d in sub.edges(data=True) if d["symbol"] is not EPSILON]
        if not consuming:
            continue
        target = accepting[0]
        source = min(
            ba.initial_states,
            key=lambda s: (not nx.has_path(graph, s, target), repr(s)),
        )
        prefix = word(nx.shortest_path(graph, source, target))
        u, v, symbol = sorted(consuming, key=repr)[0]
        loop = [*word(nx.shortest_path(sub, target, u)), symbol, *word(nx.shortest_path(sub, v, target))]
        return tuple(prefix), tuple(loop)
    return None


def _accepting_cycle_reachable(ba: BuchiAutomaton, start: set[Hashable], loop: Sequence[Any]) -> bool:
    """Search the product of ``ba`` with loop positions for a reachable accepting cycle.

    Product node ``(state, i)`` means ``state`` is about to read ``loop[i]``. A run on
    ``loop^omega`` is accepting iff it reaches a strongly connected component that
    contains an accepting state and at least one symbol-consuming edge (pure epsilon
    cycles consume no input).
    """
    period = len(loop)
    product = nx.DiGraph()
    seeds = [(state, 0) for state in start]
    product.add_nodes_from(seeds)
    stack = list(seeds)
    while stack:
        node = stack.pop()
        state, pos = node
        moves = [((target, pos), False) for target in ba.delta(state, EPSILON)]
        moves += [((target, (pos + 1) % period), True) for target in ba.delta(state, loop[pos])]
        for succ, consumes in moves:
            if succ not in product:
                product.add_node(succ)
                stack.append(succ)
            if product.has_edge(node, succ):
                product.edges[node, succ]["consumes"] |= consumes
            else:
                product.add_edge(node, succ, consumes=consumes)
    accepting = ba.accepting_states
    for component in nx.strongly_connected_components(product):
        if not any(state in accepting for state, _ in component):
            continue
        if any(data["consumes"] for _, _, data in product.subgraph(component).edges(data=True)):
            return True
    return False
