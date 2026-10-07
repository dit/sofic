"""Property-based tests for Büchi automata and Wheeler automata / indexes."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from itertools import permutations
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from sofic.automata import (
    NFA,
    BuchiAutomaton,
    determinize_wheeler,
    is_wheeler,
    minimize,
    minimize_wheeler,
    wheeler_index,
    wheeler_order,
)
from sofic.graph import ATTR_SYMBOL
from sofic.testing import buchi_automata, lassos, nfas, wheeler_nfas
from tests import oracles

AB = ("0", "1")
N = 5


def lang(aut: Any, n: int = N) -> frozenset[tuple[Any, ...]]:
    return frozenset(w for w in oracles.words(AB, n) if aut.recognizes(w))


def truth(aut: Any, n: int = N) -> frozenset[tuple[Any, ...]]:
    return oracles.language(lambda w: oracles.nfa_accepts(aut, w), AB, n)


def edges(aut: Any) -> list[tuple[Hashable, Any, Hashable]]:
    return [(t.source, t.data.get(ATTR_SYMBOL), t.target) for t in aut.transitions()]


def satisfies_wheeler_axioms(aut: Any, order: Sequence[Hashable]) -> bool:
    """Gagie-Manzini-Sirén axioms, with labels ordered by ``repr``."""
    rank = {state: index for index, state in enumerate(order)}
    arcs = edges(aut)
    entered = {target for _source, _symbol, target in arcs}
    sources = [rank[s] for s in rank if s not in entered]
    if sources and any(rank[t] < max(sources) for t in entered):
        return False
    for u, a, u2 in arcs:
        for v, b, v2 in arcs:
            if repr(a) < repr(b) and not rank[u2] < rank[v2]:
                return False
            if a == b and rank[u] < rank[v] and not rank[u2] <= rank[v2]:
                return False
    return True


def admits_wheeler_order(aut: Any) -> bool:
    return any(satisfies_wheeler_axioms(aut, order) for order in permutations(aut.states()))


# --------------------------------------------------------------------------- Büchi


@given(buchi_automata() | buchi_automata(allow_epsilon=True), lassos())
def test_accepts_lasso_matches_oracle(ba: BuchiAutomaton, lasso: tuple[tuple[Any, ...], tuple[Any, ...]]) -> None:
    prefix, loop = lasso
    assert ba.accepts_lasso(prefix, loop) == oracles.buchi_accepts_lasso(ba, prefix, loop)


@given(buchi_automata(), lassos(), st.integers(0, 3))
def test_accepts_lasso_invariant_under_unrolling_and_rotation(
    ba: BuchiAutomaton, lasso: tuple[tuple[Any, ...], tuple[Any, ...]], shift: int
) -> None:
    prefix, loop = lasso
    expected = ba.accepts_lasso(prefix, loop)
    k = shift % len(loop)
    assert ba.accepts_lasso(prefix + loop, loop) == expected
    assert ba.accepts_lasso(prefix, loop + loop) == expected
    assert ba.accepts_lasso(prefix + loop[:k], loop[k:] + loop[:k]) == expected


# --------------------------------------------------------------------------- Wheeler order


@given(nfas(allow_epsilon=False, max_states=4) | wheeler_nfas())
def test_wheeler_order_is_sound_and_complete(a: NFA) -> None:
    order = wheeler_order(a)
    exists = admits_wheeler_order(a)
    assert (order is not None) == exists
    assert is_wheeler(a) == exists
    if order is not None:
        assert sorted(order.states, key=repr) == sorted(a.states(), key=repr)
        assert satisfies_wheeler_axioms(a, order.states)


@given(wheeler_nfas())
def test_wheeler_strategy_generates_wheeler_nfas(a: NFA) -> None:
    assert wheeler_order(a) is not None


# --------------------------------------------------------------------------- Wheeler determinization / minimization


@given(wheeler_nfas())
def test_determinize_wheeler_preserves_language_and_wheelerness(a: NFA) -> None:
    dfa = determinize_wheeler(a)
    assert dfa.is_deterministic()
    assert lang(dfa) == truth(a)
    assert is_wheeler(dfa)


@given(wheeler_nfas())
def test_minimize_wheeler_preserves_language_and_is_idempotent(a: NFA) -> None:
    dfa = determinize_wheeler(a)
    smallest = minimize_wheeler(dfa)
    assert lang(smallest) == truth(a)
    assert is_wheeler(smallest)
    assert len(list(minimize(a).states())) <= len(list(smallest.states())) <= len(list(dfa.states()))
    assert len(list(minimize_wheeler(smallest).states())) == len(list(smallest.states()))


# --------------------------------------------------------------------------- Wheeler index


@given(wheeler_nfas())
def test_wheeler_index_membership_and_reachability(a: NFA) -> None:
    index = wheeler_index(a)
    for w in oracles.words(AB, N):
        assert index.contains(w) == oracles.nfa_accepts(a, w)
        assert frozenset(index.states_reached(w)) == oracles.nfa_reachable(a, w)
        assert index.count_states(w) == len(oracles.nfa_reachable(a, w))


@given(wheeler_nfas(), st.integers(0, N))
def test_wheeler_index_counts_ranks_and_enumerates(a: NFA, n: int) -> None:
    index = wheeler_index(a)
    expected = frozenset(w for w in truth(a, n) if len(w) == n)
    assert index.count_words(n) == len(expected)
    listed = list(index.words_of_length(n))
    assert frozenset(listed) == expected
    assert len(listed) == len(expected)
    for i, w in enumerate(listed):
        assert index.rank_word(w) == i
        assert index.unrank_word(i, n) == w
