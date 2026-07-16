"""Probabilistic-automaton learning via the ALERGIA state-merging algorithm.

ALERGIA :cite:`Carrasco1994` learns a probabilistic deterministic automaton from
unlabeled positive strings by merging states of a frequency prefix-tree acceptor
whenever a Hoeffding-bound test cannot distinguish their outgoing (and recursive)
transition statistics. It is the stochastic, unlabeled counterpart of RPNI/EDSM
and a state-merging alternative to Causal-State Splitting Reconstruction
(:func:`sofic.generators.epsilon_inference.cssr`).

The learned automaton is returned as a
:class:`~sofic.generators.pfa.ProbabilisticFiniteAutomaton` describing the
symbol-generation process: per-state transition probabilities are renormalized
over the alphabet (the string-termination mass of the underlying PDFA is
dropped), so each state's outgoing masses sum to one, matching sofic's
row-stochastic generator convention.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

from sofic.generators.pfa import ProbabilisticFiniteAutomaton

__all__ = ["learn_pfa_alergia"]


@dataclass
class _FPTA:
    """Frequency prefix-tree acceptor with union-find over its nodes."""

    parent: list[int]
    count: list[int]  # arrivals at the block
    final: list[int]  # strings terminating in the block
    tfreq: list[dict[Any, int]]  # symbol -> transition frequency out of the block
    tchild: list[dict[Any, int]]  # symbol -> child node id

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root


def _build_fpta(samples: Sequence[Sequence[Any]]) -> tuple[_FPTA, tuple[Any, ...]]:
    count = [0]
    final = [0]
    tfreq: list[dict[Any, int]] = [{}]
    tchild: list[dict[Any, int]] = [{}]
    alphabet: set[Any] = set()

    def ensure(node: int, symbol: Any) -> int:
        if symbol not in tchild[node]:
            tchild[node][symbol] = len(count)
            tfreq[node][symbol] = 0
            count.append(0)
            final.append(0)
            tfreq.append({})
            tchild.append({})
        return tchild[node][symbol]

    for word in samples:
        node = 0
        count[0] += 1
        for symbol in word:
            alphabet.add(symbol)
            child = ensure(node, symbol)
            tfreq[node][symbol] += 1
            count[child] += 1
            node = child
        final[node] += 1

    fpta = _FPTA(parent=list(range(len(count))), count=count, final=final, tfreq=tfreq, tchild=tchild)
    return fpta, tuple(sorted(alphabet, key=repr))


def _hoeffding_compatible(f1: int, n1: int, f2: int, n2: int, alpha: float) -> bool:
    """Hoeffding-bound two-proportion test (compatible when within the bound)."""
    if n1 == 0 or n2 == 0:
        return True
    bound = math.sqrt(0.5 * math.log(2.0 / alpha)) * (1.0 / math.sqrt(n1) + 1.0 / math.sqrt(n2))
    return abs(f1 / n1 - f2 / n2) <= bound


def _compatible(fpta: _FPTA, a: int, b: int, alpha: float, seen: set[tuple[int, int]]) -> bool:
    """Recursive ALERGIA compatibility of blocks ``a`` and ``b``."""
    a, b = fpta.find(a), fpta.find(b)
    if a == b:
        return True
    key = (a, b) if a < b else (b, a)
    if key in seen:
        return True
    seen.add(key)

    na, nb = fpta.count[a], fpta.count[b]
    if not _hoeffding_compatible(fpta.final[a], na, fpta.final[b], nb, alpha):
        return False
    symbols = set(fpta.tfreq[a]) | set(fpta.tfreq[b])
    for symbol in symbols:
        fa = fpta.tfreq[a].get(symbol, 0)
        fb = fpta.tfreq[b].get(symbol, 0)
        if not _hoeffding_compatible(fa, na, fb, nb, alpha):
            return False
    for symbol in set(fpta.tchild[a]) & set(fpta.tchild[b]):
        if not _compatible(fpta, fpta.tchild[a][symbol], fpta.tchild[b][symbol], alpha, seen):
            return False
    return True


def _merge(fpta: _FPTA, red: int, blue: int) -> None:
    """Fold ``blue`` into ``red``, accumulating counts and recursing on shared symbols."""
    stack: list[tuple[int, int]] = [(red, blue)]
    while stack:
        left, right = stack.pop()
        x, y = fpta.find(left), fpta.find(right)
        if x == y:
            continue
        fpta.parent[y] = x
        fpta.count[x] += fpta.count[y]
        fpta.final[x] += fpta.final[y]
        for symbol, freq in fpta.tfreq[y].items():
            if symbol in fpta.tchild[x]:
                fpta.tfreq[x][symbol] += freq
                stack.append((fpta.tchild[x][symbol], fpta.tchild[y][symbol]))
            else:
                fpta.tchild[x][symbol] = fpta.tchild[y][symbol]
                fpta.tfreq[x][symbol] = freq
        fpta.tfreq[y] = {}
        fpta.tchild[y] = {}


def _to_pfa(fpta: _FPTA, red: list[int], alphabet: Sequence[Any]) -> ProbabilisticFiniteAutomaton:
    reps = sorted({fpta.find(r) for r in red})
    rep_to_label: dict[int, Hashable] = {rep: f"q{index}" for index, rep in enumerate(reps)}

    pfa = ProbabilisticFiniteAutomaton(
        initial_distribution={rep_to_label[fpta.find(0)]: 1.0},
        output_alphabet=frozenset(alphabet),
    )
    for name in rep_to_label.values():
        pfa.graph.add_state(name)

    for rep in reps:
        name = rep_to_label[rep]
        total = sum(fpta.tfreq[rep].get(symbol, 0) for symbol in fpta.tchild[rep])
        if total <= 0:
            continue  # pure terminal block: no outgoing edges (allowed)
        for symbol in sorted(fpta.tchild[rep], key=repr):
            freq = fpta.tfreq[rep].get(symbol, 0)
            if freq <= 0:
                continue
            target = rep_to_label[fpta.find(fpta.tchild[rep][symbol])]
            pfa.add_transition(name, target, symbol, freq / total)
    pfa.validate()
    return pfa


def learn_pfa_alergia(
    samples: Sequence[Sequence[Any]],
    *,
    alpha: float = 0.05,
) -> ProbabilisticFiniteAutomaton:
    """Learn a probabilistic finite automaton from positive strings by ALERGIA.

    Parameters
    ----------
    samples
        Observed strings drawn from the target process (e.g. realizations, or a
        long sequence split into windows).
    alpha
        Significance level of the Hoeffding compatibility test. Smaller ``alpha``
        merges more aggressively (fewer states); larger ``alpha`` is more
        conservative.

    Returns
    -------
    ProbabilisticFiniteAutomaton
        A row-stochastic generator for the symbol process, with per-state
        transition probabilities estimated from the merged frequencies
        :cite:`Carrasco1994`.
    """
    strings = [tuple(word) for word in samples]
    if not strings:
        raise ValueError("at least one sample string is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    fpta, alphabet = _build_fpta(strings)

    red: list[int] = [0]
    while True:
        red_reps = {fpta.find(r) for r in red}
        blue: int | None = None
        for r in red:
            rep = fpta.find(r)
            for symbol in sorted(fpta.tchild[rep], key=repr):
                child = fpta.find(fpta.tchild[rep][symbol])
                if child not in red_reps:
                    blue = child
                    break
            if blue is not None:
                break
        if blue is None:
            break

        merged = False
        for r in red:
            rep = fpta.find(r)
            if rep == blue:
                continue
            if _compatible(fpta, rep, blue, alpha, set()):
                _merge(fpta, rep, blue)
                merged = True
                break
        if not merged:
            red.append(blue)

    return _to_pfa(fpta, red, alphabet)
