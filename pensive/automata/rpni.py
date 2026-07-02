"""Passive DFA learning via the RPNI state-merging algorithm."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pensive.automata.dfa import DFA

__all__ = ["learn_dfa_rpni"]


@dataclass
class _PTANode:
    """Prefix-tree acceptor node."""

    transitions: dict[Any, int] = field(default_factory=dict)
    accepting: bool = False


def _build_pta(
    positive: Sequence[Sequence[Any]],
    negative: Sequence[Sequence[Any]],
) -> tuple[list[_PTANode], dict[tuple[Any, ...], int]]:
    """Build a prefix-tree acceptor from labeled samples."""
    nodes: list[_PTANode] = [_PTANode()]
    index: dict[tuple[Any, ...], int] = {(): 0}

    def ensure(prefix: tuple[Any, ...]) -> int:
        if prefix not in index:
            index[prefix] = len(nodes)
            nodes.append(_PTANode())
        return index[prefix]

    for word in positive:
        prefix: tuple[Any, ...] = ()
        for symbol in word:
            node_id = index[prefix]
            next_prefix = prefix + (symbol,)
            if symbol not in nodes[node_id].transitions:
                nodes[node_id].transitions[symbol] = ensure(next_prefix)
            prefix = next_prefix
        nodes[index[prefix]].accepting = True

    for word in negative:
        prefix = ()
        for symbol in word:
            node_id = index[prefix]
            next_prefix = prefix + (symbol,)
            if symbol not in nodes[node_id].transitions:
                nodes[node_id].transitions[symbol] = ensure(next_prefix)
            prefix = next_prefix
        nodes[index[prefix]].accepting = False

    return nodes, index


def _collect_alphabet(
    positive: Sequence[Sequence[Any]],
    negative: Sequence[Sequence[Any]],
) -> tuple[Any, ...]:
    symbols: set[Any] = set()
    for word in positive:
        symbols.update(word)
    for word in negative:
        symbols.update(word)
    return tuple(sorted(symbols, key=repr))


def _compatible_merge(
    nodes: list[_PTANode],
    left: int,
    right: int,
    merge_map: dict[int, int],
) -> bool:
    """Return whether merging ``right`` into ``left`` is RPNI-compatible."""
    if nodes[left].accepting != nodes[right].accepting:
        return False
    symbols = set(nodes[left].transitions) | set(nodes[right].transitions)
    for symbol in symbols:
        left_target = nodes[left].transitions.get(symbol)
        right_target = nodes[right].transitions.get(symbol)
        if left_target is None or right_target is None:
            if left_target is not None or right_target is not None:
                return False
            continue
        if merge_map.get(left_target, left_target) != merge_map.get(right_target, right_target):
            return False
    return True


def _apply_merge(nodes: list[_PTANode], left: int, right: int, merge_map: dict[int, int]) -> None:
    """Merge PTA node ``right`` into ``left``."""
    merge_map[right] = left
    for symbol, target in nodes[right].transitions.items():
        if symbol not in nodes[left].transitions:
            nodes[left].transitions[symbol] = target
    nodes[right].transitions.clear()
    nodes[right].accepting = nodes[left].accepting


def _rpni_merge(nodes: list[_PTANode]) -> dict[int, int]:
    """Greedy RPNI state merging on a PTA."""
    merge_map: dict[int, int] = {index: index for index in range(len(nodes))}

    def representative(node_id: int) -> int:
        while merge_map[node_id] != node_id:
            merge_map[node_id] = merge_map[merge_map[node_id]]
            node_id = merge_map[node_id]
        return node_id

    changed = True
    while changed:
        changed = False
        reps = sorted({representative(index) for index in range(len(nodes))})
        for left in reps:
            for right in reps:
                if left >= right:
                    continue
                if not _compatible_merge(nodes, left, right, merge_map):
                    continue
                _apply_merge(nodes, left, right, merge_map)
                changed = True
                break
            if changed:
                break
    return merge_map


def _pta_to_dfa(
    nodes: list[_PTANode],
    merge_map: dict[int, int],
    alphabet: Sequence[Any],
) -> DFA:
    def representative(node_id: int) -> int:
        while merge_map[node_id] != node_id:
            merge_map[node_id] = merge_map[merge_map[node_id]]
            node_id = merge_map[node_id]
        return node_id

    blocks: dict[int, list[int]] = {}
    for node_id in range(len(nodes)):
        rep = representative(node_id)
        blocks.setdefault(rep, []).append(node_id)

    rep_to_label: dict[int, Hashable] = {}
    for block_id, (rep, _members) in enumerate(sorted(blocks.items(), key=lambda item: min(item[1]))):
        rep_to_label[rep] = f"q{block_id}"

    dfa = DFA(input_alphabet=frozenset(alphabet))
    for label in rep_to_label.values():
        dfa.graph.add_state(label)

    initial = rep_to_label[representative(0)]
    dfa.initial_states = frozenset({initial})

    accepting: set[Hashable] = set()
    for rep, members in blocks.items():
        label = rep_to_label[rep]
        if any(nodes[member].accepting for member in members):
            accepting.add(label)
        rep_node = nodes[rep]
        for symbol in alphabet:
            target = rep_node.transitions.get(symbol)
            if target is None:
                continue
            target_label = rep_to_label[representative(target)]
            dfa.add_transition(label, target_label, symbol)

    dfa.accepting_states = frozenset(accepting)
    dfa.validate()
    return dfa


def learn_dfa_rpni(
    positive: Sequence[Sequence[Any]],
    negative: Sequence[Sequence[Any]] | None = None,
) -> DFA:
    """Learn a minimal compatible DFA from positive and negative samples.

    Implements the classic RPNI greedy merge strategy on a prefix-tree
    acceptor built from the sample sets.
    """
    pos = [tuple(word) for word in positive]
    neg = [tuple(word) for word in (negative or ())]
    if not pos and not neg:
        raise ValueError("at least one positive or negative sample is required")

    alphabet = _collect_alphabet(pos, neg)
    nodes, _index = _build_pta(pos, neg)
    merge_map = _rpni_merge(nodes)
    return _pta_to_dfa(nodes, merge_map, alphabet)
