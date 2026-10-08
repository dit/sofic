"""Passive DFA learning of k-testable languages in the strict sense."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any

from sofic.automata.dfa import DFA

__all__ = ["learn_dfa_k_testable"]


def learn_dfa_k_testable(
    samples: Sequence[Sequence[Any]],
    k: int,
    *,
    alphabet: Iterable[Any] | None = None,
) -> DFA:
    """Learn the smallest strictly k-testable language containing ``samples``.

    Implements the k-TSSI algorithm of García & Vidal :cite:`GarciaVidal1990`.
    From the positive samples it collects the length-``k - 1`` prefixes ``I``
    and suffixes ``F``, the length-``k`` factors ``T``, and the sample words
    shorter than ``k - 1``. A word ``w`` is accepted iff it is one of those
    short sample words, or ``|w| >= k - 1``, its prefix is in ``I``, its suffix
    is in ``F``, and every length-``k`` factor is in ``T``. The language grows
    monotonically with the sample and identifies every strictly k-testable
    language in the limit from positive data; it shrinks (weakly) as ``k`` grows.

    States are the sample prefixes shorter than ``k - 1`` together with the
    reachable length-``k - 1`` windows, labelled by the tuples themselves; the
    DFA is partial and trim on the accessible side. ``alphabet`` defaults to
    the symbols occurring in ``samples`` and must contain all of them.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    words = [tuple(word) for word in samples]
    window = k - 1
    occurring = {symbol for word in words for symbol in word}
    if alphabet is None:
        symbols = tuple(sorted(occurring, key=repr))
    else:
        symbols = tuple(sorted(set(alphabet), key=repr))
        if missing := occurring - set(symbols):
            raise ValueError(f"sample symbols {sorted(missing, key=repr)!r} are not in the alphabet")

    heads = {word[:length] for word in words for length in range(min(len(word), window) + 1)}
    suffixes = {word[len(word) - window :] for word in words if len(word) >= window}
    factors = {word[index : index + k] for word in words for index in range(len(word) - k + 1)}
    short = {word for word in words if len(word) < window}

    dfa = DFA(input_alphabet=frozenset(symbols), initial_states=frozenset({()}))
    dfa.graph.add_state(())
    accepting: set[tuple[Any, ...]] = set()
    queue: deque[tuple[Any, ...]] = deque([()])
    seen = {()}
    while queue:
        state = queue.popleft()
        if state in short or (len(state) == window and state in suffixes):
            accepting.add(state)
        for symbol in symbols:
            extended = (*state, symbol)
            if len(state) < window:
                if extended not in heads:
                    continue
                target = extended
            else:
                if extended not in factors:
                    continue
                target = extended[1:]
            if target not in seen:
                seen.add(target)
                dfa.graph.add_state(target)
                queue.append(target)
            dfa.add_transition(state, target, symbol)

    dfa.accepting_states = frozenset(accepting)
    dfa.validate()
    return dfa
