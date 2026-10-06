"""ε-machine reconstruction by merging depth-``L`` subtrees.

Follows Crutchfield & Young (PRL 1989; PRE 1994).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.inference.cssr.counts import History, SuffixCounts
from sofic.inference.cssr.process import _suffix_reconstruct, suggest_lmax
from sofic.inference.cssr.significance import MorphTest, morphs_differ

#: Significance level used by subtree merging when ``delta = 0`` and to resolve truncated successors.
_SUBTREE_ALPHA = 0.01


def _morph_distance(
    counts: SuffixCounts,
    left: History,
    right: History,
    *,
    delta: float,
) -> float:
    left_morph = counts.morph(left)
    right_morph = counts.morph(right)
    return 0.5 * sum(abs(left_morph[s] - right_morph[s]) for s in counts.alphabet)


def _morphs_equivalent(
    counts: SuffixCounts,
    left: History,
    right: History,
    *,
    delta: float,
    alpha: float = _SUBTREE_ALPHA,
    test: MorphTest = "g",
) -> bool:
    if delta > 0.0:
        return _morph_distance(counts, left, right, delta=delta) <= delta
    return not morphs_differ(counts, {left}, {right}, alpha=alpha, test=test)


def _cluster_histories_by_morph(
    counts: SuffixCounts,
    histories: set[History],
    *,
    delta: float,
    alpha: float = _SUBTREE_ALPHA,
    test: MorphTest = "g",
) -> dict[int, set[History]]:
    parent: dict[History, History] = {history: history for history in histories}

    def find(history: History) -> History:
        root = history
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    def union(left: History, right: History) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    history_list = sorted(histories)
    for index, left in enumerate(history_list):
        for right in history_list[index + 1 :]:
            if _morphs_equivalent(counts, left, right, delta=delta, alpha=alpha, test=test):
                union(left, right)

    clusters: dict[History, set[History]] = defaultdict(set)
    for history in histories:
        clusters[find(history)].add(history)

    states: dict[int, set[History]] = {}
    for state_id, (_root, members) in enumerate(clusters.items()):
        states[state_id] = set(members)
    return states


def subtree_merge(
    sequence: Sequence[Any],
    *,
    L: int | Literal["auto"],
    delta: float = 0.0,
    alphabet: Sequence[Any] | None = None,
    alpha: float = _SUBTREE_ALPHA,
    test: MorphTest = "g",
    correction: Literal["bonferroni"] | None = None,
) -> EpsilonMachine:
    """Reconstruct an ε-machine by merging depth-``L`` subtrees (Crutchfield--Young).

    Histories up to length ``L`` are clustered by next-symbol distribution: within
    total-variation distance ``delta``, or, when ``delta = 0``, unless a morph test
    (``test``, at level ``alpha``) tells them apart. The clusters are then
    determinized as in :func:`cssr`.

    ``L="auto"`` uses :func:`suggest_lmax`. ``correction="bonferroni"`` divides
    ``alpha`` by the number of history pairs compared, so that no pair is split
    apart by chance; since a rejected test *separates* histories, this makes the
    reconstruction more conservative (fewer states).
    """
    if L == "auto":
        L = suggest_lmax(sequence, alpha=alpha)
    if L < 0:
        raise ValueError("L must be non-negative")
    seq = tuple(sequence)
    if len(seq) < 2:
        raise ValueError("sequence must contain at least two symbols")
    counts = SuffixCounts.from_sequence(seq, alphabet=alphabet, max_length=L + 1)

    histories = {history for history in counts.history_counts if len(history) <= L}
    histories.add(())
    if correction == "bonferroni":
        alpha /= max(1, len(histories) * (len(histories) - 1) // 2)
    elif correction is not None:
        raise ValueError(f"unknown correction {correction!r}")
    states = list(_cluster_histories_by_morph(counts, histories, delta=delta, alpha=alpha, test=test).values())

    return _suffix_reconstruct(states, counts, seq, Lmax=L, alpha=alpha, test=test)
