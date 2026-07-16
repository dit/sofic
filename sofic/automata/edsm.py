"""Passive DFA learning via Evidence-Driven State Merging (blue-fringe).

EDSM upgrades the greedy RPNI merge order (:func:`sofic.automata.rpni.learn_dfa_rpni`)
with the evidence-driven, red/blue "blue-fringe" strategy that won the Abbadingo
One competition :cite:`Lang1998`. Starting from the augmented prefix-tree
acceptor of the labeled sample, it maintains a set of confirmed *red* states and
their *blue* fringe; at each step it either promotes a blue state that cannot be
merged with any red state, or commits the single highest-*evidence* merge, where
the evidence of a merge is the number of identically-labeled state pairs it
folds together. The result is the practical state-of-the-art passive DFA
heuristic and, like RPNI, is consistent with the sample.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

from sofic.automata.dfa import DFA

__all__ = ["learn_dfa_edsm"]

_ACCEPT = 1
_REJECT = -1
_UNKNOWN = 0


@dataclass
class _MergeState:
    """Union-find over augmented-PTA nodes with per-block labels and transitions."""

    parent: list[int]
    label: list[int]
    trans: list[dict[Any, int]]

    def clone(self) -> _MergeState:
        return _MergeState(
            parent=self.parent.copy(),
            label=self.label.copy(),
            trans=[dict(row) for row in self.trans],
        )

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root


def _build_apta(
    positive: Sequence[Sequence[Any]],
    negative: Sequence[Sequence[Any]],
) -> tuple[list[int], list[dict[Any, int]]]:
    """Build an augmented PTA: labels are accept (+1), reject (-1), or unknown (0)."""
    label: list[int] = [_UNKNOWN]
    trans: list[dict[Any, int]] = [{}]

    def ensure(node: int, symbol: Any) -> int:
        if symbol not in trans[node]:
            trans[node][symbol] = len(label)
            label.append(_UNKNOWN)
            trans.append({})
        return trans[node][symbol]

    for word, word_label in [(w, _ACCEPT) for w in positive] + [(w, _REJECT) for w in negative]:
        node = 0
        for symbol in word:
            node = ensure(node, symbol)
        if label[node] != _UNKNOWN and label[node] != word_label:
            raise ValueError(f"contradictory labels for word {tuple(word)!r}")
        label[node] = word_label

    return label, trans


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


def _fold(state: _MergeState, red: int, blue: int) -> int | None:
    """Merge ``blue`` into ``red`` with deterministic folding.

    Returns the evidence score (number of identically-labeled pairs folded) or
    ``None`` when a label conflict makes the merge inconsistent. Mutates
    ``state`` (call on a clone to evaluate a candidate).
    """
    stack: list[tuple[int, int]] = [(red, blue)]
    score = 0
    while stack:
        left, right = stack.pop()
        x = state.find(left)
        y = state.find(right)
        if x == y:
            continue
        label_x, label_y = state.label[x], state.label[y]
        if label_x != _UNKNOWN and label_y != _UNKNOWN:
            if label_x != label_y:
                return None
            score += 1
        # Determinism forces children on shared symbols to merge too.
        for symbol, child in state.trans[y].items():
            if symbol in state.trans[x]:
                stack.append((state.trans[x][symbol], child))
        state.parent[y] = x
        if state.label[x] == _UNKNOWN:
            state.label[x] = label_y
        for symbol, child in state.trans[y].items():
            state.trans[x].setdefault(symbol, child)
        state.trans[y] = {}
    return score


def _blue_fringe(state: _MergeState, red: list[int]) -> list[int]:
    """Return fringe (blue) representatives in canonical (red, symbol) order."""
    red_reps = {state.find(r) for r in red}
    fringe: list[int] = []
    seen: set[int] = set()
    for r in red:
        rep = state.find(r)
        for symbol in sorted(state.trans[rep], key=repr):
            child = state.find(state.trans[rep][symbol])
            if child not in red_reps and child not in seen:
                seen.add(child)
                fringe.append(child)
    return fringe


def _to_dfa(state: _MergeState, red: list[int], alphabet: Sequence[Any]) -> DFA:
    reps = sorted({state.find(r) for r in red})
    rep_to_label: dict[int, Hashable] = {rep: f"q{index}" for index, rep in enumerate(reps)}

    dfa = DFA(input_alphabet=frozenset(alphabet))
    for name in rep_to_label.values():
        dfa.graph.add_state(name)
    dfa.initial_states = frozenset({rep_to_label[state.find(0)]})

    accepting: set[Hashable] = set()
    for rep in reps:
        name = rep_to_label[rep]
        if state.label[rep] == _ACCEPT:
            accepting.add(name)
        for symbol in sorted(state.trans[rep], key=repr):
            target = state.find(state.trans[rep][symbol])
            dfa.add_transition(name, rep_to_label[target], symbol)
    dfa.accepting_states = frozenset(accepting)
    dfa.validate()
    return dfa


def learn_dfa_edsm(
    positive: Sequence[Sequence[Any]],
    negative: Sequence[Sequence[Any]] | None = None,
) -> DFA:
    """Learn a DFA from labeled samples by evidence-driven state merging.

    Parameters
    ----------
    positive
        Words that must be accepted.
    negative
        Words that must be rejected.

    Returns
    -------
    DFA
        A deterministic automaton consistent with the sample, built with the
        blue-fringe EDSM heuristic :cite:`Lang1998`.
    """
    pos = [tuple(word) for word in positive]
    neg = [tuple(word) for word in (negative or ())]
    if not pos and not neg:
        raise ValueError("at least one positive or negative sample is required")

    alphabet = _collect_alphabet(pos, neg)
    label, trans = _build_apta(pos, neg)
    state = _MergeState(parent=list(range(len(label))), label=label, trans=trans)

    red: list[int] = [0]
    while True:
        fringe = _blue_fringe(state, red)
        if not fringe:
            break

        scored: list[tuple[int, _MergeState]] = []
        promote: int | None = None
        for blue in fringe:
            candidates: list[tuple[int, _MergeState]] = []
            for r in red:
                rep = state.find(r)
                if rep == blue:
                    continue
                trial = state.clone()
                evidence = _fold(trial, rep, blue)
                if evidence is not None:
                    candidates.append((evidence, trial))
            if not candidates:
                promote = blue
                break
            scored.extend(candidates)

        if promote is not None:
            red.append(promote)
            continue

        best = max(scored, key=lambda item: item[0])
        state = best[1]

    return _to_dfa(state, red, alphabet)
