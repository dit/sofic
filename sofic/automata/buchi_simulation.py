"""Büchi acceptance for lasso and ultimately periodic ω-words."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

from sofic.automata.buchi import BuchiAutomaton


def accepts_lasso_buchi(ba: BuchiAutomaton, prefix: Sequence[Any], loop: Sequence[Any]) -> bool:
    """Accept if repeating ``loop`` after ``prefix`` visits an accepting state infinitely often."""
    post = ba._run_nfa(prefix)
    if not post:
        return False

    reachable = _loop_closure(ba, post, loop)
    candidates = reachable & ba.accepting_states
    if not candidates:
        return False

    return any(_can_revisit_on_loop(ba, state, loop) for state in candidates)


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


def _loop_closure(ba: BuchiAutomaton, start: set[Hashable], loop: Sequence[Any]) -> set[Hashable]:
    """States reachable from ``start`` by reading ``loop`` zero or more times."""
    if not loop:
        return set(start)

    reachable = set(start)
    frontier = set(start)
    while frontier:
        after = ba._run_nfa(loop, start=frontier)
        new = after - reachable
        if not new:
            break
        reachable |= new
        frontier = new
    return reachable


def _can_revisit_on_loop(ba: BuchiAutomaton, state: Hashable, loop: Sequence[Any]) -> bool:
    """Return whether ``state`` can be revisited by reading ``loop`` one or more times."""
    if not loop:
        return False

    current = {state}
    limit = max(len(list(ba.states())), 1) + 1
    for _ in range(limit):
        current = ba._run_nfa(loop, start=current)
        if state in current:
            return True
    return False
