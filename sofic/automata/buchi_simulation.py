"""Büchi acceptance for lasso and ultimately periodic ω-words."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import networkx as nx

from sofic.automata.buchi import BuchiAutomaton
from sofic.graph import EPSILON


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
